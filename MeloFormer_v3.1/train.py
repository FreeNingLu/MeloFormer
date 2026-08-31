#!/usr/bin/env python3
"""MeloFormer v3.1 训练脚本 - 支持单卡/DDP 多卡训练 (v3.0架构 + v2.3工程优化)"""

import os
import sys
import math
import time
import json
import hashlib
import random
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime

import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from tqdm import tqdm

from model import MeloFormer, create_model, set_flex_backend
from data import HIDTokenizerV2

# --- Training constants ---
DYNAMO_CACHE_SIZE = 16384
DYNAMO_ACCUM_CACHE_SIZE = 32768
WARMUP_PHASE_STEPS = 10       # steps before switching to multi-worker
DEFAULT_NUM_WORKERS = 16
BATCH_SIZE = 1                # v2.1 constraint: captured tensor mask_mod requires batch_size=1
BUCKET_SIZE = 512             # sequence length bucketing for FlexAttention kernel reuse
SHARD_CACHE_SIZE = 30         # LRU shard cache slots (~30-35GB at 120GB memory)
SHARD_PROGRESS_INTERVAL = 500 # print progress every N shards during Arrow merge

# Collate 字段: 在 dataset __getitem__, collate_fn, train/val loop 中共用
TENSOR_FIELDS = ['token_ids', 'chord_ids', 'instrument_ids', 'token_type_ids', 'note_ids', 'position_ids']

# --- HuggingFace datasets (可选，用于 Arrow 格式) ---
try:
    from datasets import load_from_disk
    HAS_HF_DATASETS = True
except ImportError:
    HAS_HF_DATASETS = False


# ---------------------------------------------------------------------------
# Torch configuration
# ---------------------------------------------------------------------------

def configure_torch_for_training():
    """Configure torch._dynamo cache and TF32 settings at startup."""
    # FlexAttention 的 create_block_mask 会为每个不同的序列长度重编译
    # 即使使用 512 的分桶，8192 长度也有 16 种变体
    # 加上 batch size、num_chords 等变化，需要非常大的缓存
    torch._dynamo.config.cache_size_limit = DYNAMO_CACHE_SIZE
    torch._dynamo.config.accumulated_cache_size_limit = DYNAMO_ACCUM_CACHE_SIZE
    torch._dynamo.config.suppress_errors = True  # 超限时回退到 eager 模式

    # H800 TF32 优化: matmul 和 convolution 操作自动使用 TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class ArrowDataset(Dataset):
    """
    HuggingFace Arrow 格式数据集 - 零拷贝内存映射

    优势:
    1. 零拷贝 - 不需要反序列化，直接内存映射
    2. O(1) 随机访问 - 任意样本直接定位
    3. 多进程安全 - num_workers 可以开很大
    4. 已按长度排序 - 减少 padding

    支持三种目录结构:
    - 单个 HF Dataset 目录 (load_from_disk 直接加载)
    - 分片子目录 shard_XXXX/ (自动合并 + MD5 缓存)
    """

    def __init__(
        self,
        data_path: str,
        max_seq_len: int = 24576,
        max_chords: int = 2048,
        max_samples: int = None,
    ):
        if not HAS_HF_DATASETS:
            raise ImportError("需要安装 datasets: pip install datasets")

        self.max_seq_len = max_seq_len
        self.max_chords = max_chords

        print(f"加载 Arrow 数据集: {data_path}")
        self.dataset = self._load_arrow(Path(data_path))

        if max_samples and max_samples < len(self.dataset):
            self.dataset = self.dataset.select(range(max_samples))

        print(f"Arrow 数据集: {len(self.dataset):,} 个样本")

    @staticmethod
    def _load_arrow(data_path_obj: Path):
        """加载 Arrow 数据集，支持分片子目录 + MD5 缓存合并."""
        shard_dirs = sorted(data_path_obj.glob('shard_*'))
        if not (shard_dirs and shard_dirs[0].is_dir()):
            return load_from_disk(str(data_path_obj))

        from datasets import concatenate_datasets

        # 缓存路径: 与数据集同级的 _merged_cache 目录
        cache_dir = data_path_obj.parent / (data_path_obj.name + '_merged_cache')
        # 缓存 key: meta.json 内容的 hash + 分片数量
        meta_path = data_path_obj / 'meta.json'
        cache_key_src = meta_path.read_text() if meta_path.exists() else ''
        cache_key_src += f'|shards={len(shard_dirs)}'
        cache_key = hashlib.md5(cache_key_src.encode()).hexdigest()[:16]
        cache_flag = cache_dir / f'.cache_key_{cache_key}'

        if cache_dir.exists() and cache_flag.exists():
            print(f"从缓存加载合并数据集: {cache_dir}")
            dataset = load_from_disk(str(cache_dir))
            print(f"缓存命中: {len(dataset):,} 个样本")
            return dataset

        print(f"检测到分片格式: {len(shard_dirs)} 个分片，逐个加载并合并...")
        datasets_list = []
        for i, shard_dir in enumerate(shard_dirs):
            ds = load_from_disk(str(shard_dir))
            datasets_list.append(ds)
            if (i + 1) % SHARD_PROGRESS_INTERVAL == 0:
                print(f"  已加载 {i + 1}/{len(shard_dirs)} 个分片...")
        dataset = concatenate_datasets(datasets_list)
        print(f"合并完成: {len(dataset):,} 个样本，保存缓存到 {cache_dir} ...")
        cache_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(cache_dir))
        cache_flag.touch()
        print(f"缓存已保存")
        return dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        item = self.dataset[idx]

        sample = {
            'token_ids': torch.tensor(item['token_ids'], dtype=torch.long),
            'chord_ids': torch.tensor(item['chord_ids'], dtype=torch.long),
            'instrument_ids': torch.tensor(item['instrument_ids'], dtype=torch.long),
            'token_type_ids': torch.tensor(item['token_type_ids'], dtype=torch.long),
            'note_ids': torch.tensor(item['note_ids'], dtype=torch.long),
            'length': item['length'],
        }

        if 'position_ids' in item:
            sample['position_ids'] = torch.tensor(item['position_ids'], dtype=torch.long)
        else:
            sample['position_ids'] = torch.arange(sample['length'], dtype=torch.long)

        num_chords = sample['chord_ids'].max().item() + 1
        if num_chords > self.max_chords:
            return None

        if sample['length'] > self.max_seq_len:
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    sample[k] = v[:self.max_seq_len]
            sample['length'] = self.max_seq_len

        return sample


class PreprocessedDataset(Dataset):
    """
    加载预处理数据 - 支持两种格式:
    1. 分片格式 (推荐): shard_0000.pt, shard_0001.pt, ... 每个分片包含多个样本
    2. 单文件格式 (旧版): 每个 .pt 文件一个样本

    分片格式使用懒加载，避免内存爆炸
    """

    def __init__(
        self,
        data_path: str,
        max_seq_len: int = 24576,
        max_chords: int = 2048,
        shuffle_files: bool = True,
        max_samples: int = None,
    ):
        self.max_seq_len = max_seq_len
        self.max_chords = max_chords
        self.samples = []
        self.lazy_mode = False
        self._skipped_count = 0

        data_path = Path(data_path)
        shard_files = sorted(data_path.glob('shard_*.pt'))

        if shard_files:
            self._init_shard_mode(data_path, shard_files, max_samples, shuffle_files)
        else:
            self._init_single_file_mode(data_path, max_samples, shuffle_files)

    def _init_shard_mode(self, data_path, shard_files, max_samples, shuffle_files):
        """分片格式 - 使用懒加载 (只记录索引，不加载数据)"""
        print(f"检测到分片格式: {len(shard_files)} 个分片")

        meta_path = data_path / 'meta.json'
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            total_samples = meta.get('total_samples', 0)
            shard_size = meta.get('shard_size', 10000)
            print(f"从 meta.json 读取: {total_samples:,} 个样本")
        else:
            first_shard = torch.load(shard_files[0], weights_only=False)
            shard_size = len(first_shard)
            total_samples = shard_size * len(shard_files)
            del first_shard
            print(f"估算样本数: {total_samples:,}")

        self.lazy_mode = True
        self.shard_files = shard_files
        self.shard_size = shard_size
        self.total_samples = min(total_samples, max_samples) if max_samples else total_samples

        # LRU 分片缓存
        self._shard_cache = {}
        self._cache_order = []
        self._max_cache_size = SHARD_CACHE_SIZE

        # 构建索引映射: global_idx -> (shard_idx, local_idx)
        self.index_map = []
        sample_count = 0
        for shard_idx, shard_file in enumerate(shard_files):
            if shard_idx == len(shard_files) - 1:
                shard_samples = self.total_samples - sample_count
            else:
                shard_samples = shard_size

            for local_idx in range(shard_samples):
                if max_samples and sample_count >= max_samples:
                    break
                self.index_map.append((shard_idx, local_idx))
                sample_count += 1

            if max_samples and sample_count >= max_samples:
                break

        if shuffle_files:
            random.shuffle(self.index_map)

        print(f"懒加载模式: {len(self.index_map):,} 个样本 (缓存 {self._max_cache_size} 个分片)")

    def _init_single_file_mode(self, data_path, max_samples, shuffle_files):
        """单文件格式 (旧版兼容) - 加载到内存"""
        if data_path.is_file():
            with open(data_path, 'r') as f:
                files = [line.strip() for line in f if line.strip()]
        else:
            files = [str(f) for f in data_path.glob('*.pt')]

        if shuffle_files:
            random.shuffle(files)
        if max_samples:
            files = files[:max_samples]

        print(f"检测到单文件格式: {len(files)} 个文件")
        for f in tqdm(files, desc="加载文件"):
            try:
                data = torch.load(f, weights_only=True)
                if data['length'] > self.max_seq_len:
                    for key in TENSOR_FIELDS:
                        if key in data:
                            data[key] = data[key][:self.max_seq_len]
                    data['length'] = self.max_seq_len
                self.samples.append(data)
            except Exception:
                pass

        print(f"已加载 {len(self.samples):,} 个样本到内存")
        if shuffle_files:
            random.shuffle(self.samples)

    def _load_shard(self, shard_idx: int):
        """加载分片 (带 LRU 缓存)"""
        if shard_idx in self._shard_cache:
            self._cache_order.remove(shard_idx)
            self._cache_order.append(shard_idx)
            return self._shard_cache[shard_idx]

        shard_data = torch.load(self.shard_files[shard_idx], weights_only=False)

        if len(self._cache_order) >= self._max_cache_size:
            oldest = self._cache_order.pop(0)
            del self._shard_cache[oldest]

        self._shard_cache[shard_idx] = shard_data
        self._cache_order.append(shard_idx)
        return shard_data

    def __len__(self) -> int:
        if self.lazy_mode:
            return len(self.index_map)
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        if self.lazy_mode:
            shard_idx, local_idx = self.index_map[idx]
            shard_data = self._load_shard(shard_idx)
            sample = shard_data[local_idx]
        else:
            sample = self.samples[idx]

        if 'chord_ids' in sample:
            num_chords = int(sample['chord_ids'].max().item()) + 1
            if num_chords > self.max_chords:
                self._skipped_count += 1
                return None
            sample['num_chords'] = num_chords  # 预计算，避免 forward 中 .max().item() CPU 同步

        if sample['length'] > self.max_seq_len:
            sample = {k: v[:self.max_seq_len] if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
            sample['length'] = self.max_seq_len

        return sample


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def _bucket_length(length: int, bucket_size: int = BUCKET_SIZE) -> int:
    """将长度向上取整到最近的桶边界"""
    return ((length + bucket_size - 1) // bucket_size) * bucket_size


def efficient_collate_fn(batch: List[Optional[Dict]]) -> Optional[Dict[str, torch.Tensor]]:
    """
    高效的批次整理函数
    - 序列长度分桶: 减少 FlexAttention 重编译次数
    - 动态 padding 到 batch 内最大长度 (减少无效计算)
    - 过滤掉 None 样本 (超过 max_chords 的样本)
    """
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None

    batch = sorted(batch, key=lambda x: x['length'], reverse=True)
    max_len = _bucket_length(batch[0]['length'])
    batch_size = len(batch)

    # 预分配张量
    token_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    chord_ids = torch.full((batch_size, max_len), -1, dtype=torch.long)
    position_ids = torch.full((batch_size, max_len), -1, dtype=torch.long)
    instrument_ids = torch.full((batch_size, max_len), 129, dtype=torch.long)
    token_type_ids = torch.full((batch_size, max_len), -1, dtype=torch.long)
    note_ids = torch.full((batch_size, max_len), -1, dtype=torch.long)
    lengths = torch.zeros(batch_size, dtype=torch.long)

    max_num_chords = 0
    for i, item in enumerate(batch):
        length = item['length']
        token_ids[i, :length] = item['token_ids']
        labels[i, :length] = item['token_ids']
        chord_ids[i, :length] = item['chord_ids']
        position_ids[i, :length] = item['position_ids']
        instrument_ids[i, :length] = item['instrument_ids']
        if 'token_type_ids' in item:
            token_type_ids[i, :length] = item['token_type_ids']
        if 'note_ids' in item:
            note_ids[i, :length] = item['note_ids']
        lengths[i] = length
        if 'num_chords' in item:
            max_num_chords = max(max_num_chords, item['num_chords'])

    result = {
        'token_ids': token_ids,
        'labels': labels,
        'chord_ids': chord_ids,
        'position_ids': position_ids,
        'instrument_ids': instrument_ids,
        'token_type_ids': token_type_ids,
        'note_ids': note_ids,
        'lengths': lengths,
    }
    if max_num_chords > 0:
        result['num_chords'] = max_num_chords
    return result


class PackingCollator:
    """
    序列打包 Collator - 将多首短曲拼接成目标长度

    优势:
    - 消除 Padding 浪费，有效 token 利用率从 50-60% 提升到 ~100%
    - 训练速度提升 ~10x（短序列占比高时）

    原理:
    - 贪心算法将多首曲子拼接到 target_length
    - chord_ids 累积偏移：保证不同曲子的 chord 编号不碰撞
    - doc_ids 标记每个 token 的文档归属，在 mask_mod 中隔离跨文档注意力
    - sum_doc_ids 标记每个 summary token 的文档归属（从 chord_ids + doc_ids 派生）

    约束:
    - 打包后累积 num_chords 不超过 max_chords（否则 Summary Token 序列溢出）
    """

    def __init__(self, target_length: int = 16384, max_chords: int = 256, bucket_size: int = BUCKET_SIZE):
        self.target_length = target_length
        self.max_chords = max_chords
        self.bucket_size = bucket_size

    def __call__(self, batch: List[Optional[Dict]]) -> Optional[Dict[str, torch.Tensor]]:
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None

        batch = sorted(batch, key=lambda x: x['length'], reverse=True)

        # 贪心打包，同时约束 token 长度和 chord 数量
        packed_sequences = []
        current_pack = {'items': [], 'total_len': 0, 'total_chords': 0}

        for item in batch:
            length = item['length']
            item_num_chords = item.get('num_chords', int(item['chord_ids'].max().item()) + 1)

            if length > self.target_length:
                # 超长序列：单独打包，截断
                if current_pack['items']:
                    packed_sequences.append(current_pack)
                truncated_item = {
                    k: v[:self.target_length] if isinstance(v, torch.Tensor) else v
                    for k, v in item.items()
                }
                truncated_item['length'] = self.target_length
                # 截断后重新计算 num_chords
                trunc_nc = int(truncated_item['chord_ids'][:self.target_length].max().item()) + 1
                truncated_item['num_chords'] = trunc_nc
                packed_sequences.append({'items': [truncated_item], 'total_len': self.target_length, 'total_chords': trunc_nc})
                current_pack = {'items': [], 'total_len': 0, 'total_chords': 0}
            elif (current_pack['total_len'] + length <= self.target_length
                  and current_pack['total_chords'] + item_num_chords <= self.max_chords):
                # 能放进当前 pack
                item['num_chords'] = item_num_chords
                current_pack['items'].append(item)
                current_pack['total_len'] += length
                current_pack['total_chords'] += item_num_chords
            else:
                # 放不下，开新 pack
                if current_pack['items']:
                    packed_sequences.append(current_pack)
                item['num_chords'] = item_num_chords
                current_pack = {'items': [item], 'total_len': length, 'total_chords': item_num_chords}

        if current_pack['items']:
            packed_sequences.append(current_pack)

        batch_size = len(packed_sequences)
        max_len = self._bucket_length(max(p['total_len'] for p in packed_sequences))

        # 预分配张量
        token_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        doc_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        chord_ids = torch.full((batch_size, max_len), -1, dtype=torch.long)
        position_ids = torch.full((batch_size, max_len), -1, dtype=torch.long)
        instrument_ids = torch.full((batch_size, max_len), 129, dtype=torch.long)
        token_type_ids = torch.full((batch_size, max_len), -1, dtype=torch.long)
        note_ids = torch.full((batch_size, max_len), -1, dtype=torch.long)
        lengths = torch.zeros(batch_size, dtype=torch.long)
        max_num_chords = 0

        for b_idx, pack in enumerate(packed_sequences):
            offset = 0
            chord_offset = 0  # 累积 chord 偏移，保证不同曲子 chord 编号不碰撞

            for doc_idx, item in enumerate(pack['items']):
                length = item['length']
                end_pos = offset + length

                token_ids[b_idx, offset:end_pos] = item['token_ids'][:length]
                labels[b_idx, offset:end_pos] = item['token_ids'][:length]

                # chord_ids 加偏移：曲 A 的 chord [0,1,2]，曲 B 的 chord 变为 [3,4]
                chord_ids[b_idx, offset:end_pos] = item['chord_ids'][:length] + chord_offset

                instrument_ids[b_idx, offset:end_pos] = item['instrument_ids'][:length]

                if 'position_ids' in item:
                    position_ids[b_idx, offset:end_pos] = item['position_ids'][:length]
                if 'token_type_ids' in item:
                    token_type_ids[b_idx, offset:end_pos] = item['token_type_ids'][:length]
                if 'note_ids' in item:
                    note_ids[b_idx, offset:end_pos] = item['note_ids'][:length]

                offset = end_pos
                chord_offset += item['num_chords']

            lengths[b_idx] = pack['total_len']
            max_num_chords = max(max_num_chords, chord_offset)

            # EOS-based doc_ids: 扫描 token_ids 中的 EOS (id=2) 位置自动派生
            # 每个 EOS 后 doc_id 递增，padding 区域填 -1
            doc_id = 0
            for pos in range(pack['total_len']):
                doc_ids[b_idx, pos] = doc_id
                if token_ids[b_idx, pos].item() == 2:  # EOS_TOKEN = 2
                    doc_id += 1

        result = {
            'token_ids': token_ids,
            'labels': labels,
            'doc_ids': doc_ids,
            'chord_ids': chord_ids,
            'position_ids': position_ids,
            'instrument_ids': instrument_ids,
            'token_type_ids': token_type_ids,
            'note_ids': note_ids,
            'lengths': lengths,
        }
        if max_num_chords > 0:
            result['num_chords'] = max_num_chords
        return result

    def _bucket_length(self, length: int) -> int:
        """将长度向上取整到最近的桶边界"""
        bucketed = ((length + self.bucket_size - 1) // self.bucket_size) * self.bucket_size
        return min(bucketed, self.target_length)


# ---------------------------------------------------------------------------
# Distributed + LR schedule utilities
# ---------------------------------------------------------------------------

def setup_distributed():
    """设置分布式训练"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    """清理分布式训练"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
):
    """余弦退火 + 线性预热"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class H800Trainer:
    """H800 优化训练器"""

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        args: argparse.Namespace = None,
    ):
        self.args = args
        self.rank, self.world_size, self.local_rank = setup_distributed()
        self.is_main = self.rank == 0

        self.device = torch.device(f'cuda:{self.local_rank}')
        self.model = model.to(self.device)

        # Gradient Checkpointing
        if args.gradient_checkpointing:
            gc_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
            self.model.gradient_checkpointing_enable(autocast_dtype=gc_dtype)
            if self.is_main:
                print("[+] Gradient Checkpointing enabled")

        # torch.compile 说明:
        # MeloFormer v2.1 不在模型层面使用 torch.compile(model)，原因:
        #   1. FlexAttention 内部已独立 torch.compile(flex_attention, dynamic=True)
        #      双层 compile 导致 Dynamo 反复 graph break + 重编译，ms/step 持续增长
        #   2. chord_ids 变长 → 每个 batch 形状不同，模型层面的 shape guard 频繁失效
        #   3. _compiled_create_block_mask 的 CPU tensor 元数据与 Dynamo trace 冲突
        # FlexAttention Triton kernel 级别编译已提供主要加速（~8-15x sparse speedup）
        if self.is_main:
            print("[+] FlexAttention kernel-level compile active (torch.compile(flex_attention))")

        # DDP
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=False,
            )

        # Two-phase DataLoader (v2.1: batch_size fixed at 1)
        self._phase = 1
        self._target_num_workers = args.num_workers if args.num_workers > 0 else DEFAULT_NUM_WORKERS

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # 序列打包配置
        self._use_packing = getattr(args, 'use_packing', False)
        if self._use_packing:
            self._collate_fn = PackingCollator(
                target_length=args.max_seq_len,
                max_chords=args.max_chords,
                bucket_size=BUCKET_SIZE,
            )
            if self.is_main:
                print(f"[+] 序列打包已启用 (target_length={args.max_seq_len})")
        else:
            self._collate_fn = efficient_collate_fn

        # Phase 1 DataLoaders: batch_size=1, num_workers=0
        train_sampler = DistributedSampler(train_dataset, shuffle=True) if self.world_size > 1 else None
        self.train_sampler = train_sampler
        self.train_loader = self._make_dataloader(
            train_dataset,
            num_workers=0,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
        )

        if val_dataset is not None:
            val_sampler = DistributedSampler(val_dataset, shuffle=False) if self.world_size > 1 else None
            self.val_sampler = val_sampler
            self.val_loader = self._make_dataloader(
                val_dataset,
                num_workers=0,
                sampler=val_sampler,
                shuffle=False,
            )
        else:
            self.val_sampler = None
            self.val_loader = None

        # 优化器 (fused Adam for H800)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
            fused=True,
        )

        # 学习率调度器
        total_steps = len(self.train_loader) * args.epochs // args.gradient_accumulation_steps
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps,
        )

        # 混合精度配置
        if args.fp16:
            self.scaler = torch.amp.GradScaler('cuda')
            self.autocast_dtype = torch.float16
            if self.is_main:
                print("[+] FP16 混合精度 (GradScaler enabled)")
                print("[!] 警告: H800 推荐使用 BF16 而非 FP16")
        elif args.bf16:
            self.scaler = None
            self.autocast_dtype = torch.bfloat16
            if self.is_main:
                print("[+] BF16 混合精度 (H800 推荐)")
        else:
            self.scaler = None
            self.autocast_dtype = torch.float32
            if self.is_main:
                print("[+] FP32 精度")

        if self.is_main:
            tf32_matmul = torch.backends.cuda.matmul.allow_tf32
            tf32_cudnn = torch.backends.cudnn.allow_tf32
            print(f"[+] TF32: matmul={tf32_matmul}, cudnn={tf32_cudnn}")

        # FlexAttention + Gradient Checkpointing + 混合精度兼容性
        if args.gradient_checkpointing:
            base_model = self.model.module if hasattr(self.model, 'module') else self.model
            base_model._autocast_dtype = self.autocast_dtype

        # 状态
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # 日志 & 检查点目录
        self.log_dir = Path(args.output_dir) / 'logs'
        self.checkpoint_dir = Path(args.output_dir) / 'checkpoints'

        # 训练统计（用于最终汇总）
        self._step_log_rows: List[Dict] = []
        self._epoch_log_rows: List[Dict] = []

        if self.is_main:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # 文件日志 handler
            run_ts = datetime.now().strftime('%Y%m%d-%H%M%S')
            log_file = self.log_dir / f'train_{run_ts}.log'
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setFormatter(logging.Formatter('%(asctime)s %(message)s', datefmt='%H:%M:%S'))
            self._file_logger = logging.getLogger('meloformer_train')
            self._file_logger.setLevel(logging.INFO)
            self._file_logger.addHandler(fh)
            self._log_file = log_file
            print(f'[+] 训练日志: {log_file}')

            if args.use_wandb:
                import wandb
                wandb.init(
                    project=args.wandb_project,
                    name=args.run_name or f"meloformer-h800-{run_ts}",
                    config=vars(args),
                )

    def _make_dataloader(self, dataset, num_workers, sampler, shuffle=False):
        """Construct a DataLoader with standard settings. DRY helper for phase switches."""
        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            prefetch_factor=4 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )

    def _move_batch_to_device(self, batch):
        """Move model-input fields from batch dict to self.device."""
        token_ids = batch['token_ids'].to(self.device, non_blocking=True)
        labels = batch['labels'].to(self.device, non_blocking=True)
        chord_ids = batch['chord_ids'].to(self.device, non_blocking=True)
        instrument_ids = batch['instrument_ids'].to(self.device, non_blocking=True)
        token_type_ids = batch['token_type_ids'].to(self.device, non_blocking=True)
        note_ids = batch['note_ids'].to(self.device, non_blocking=True)

        doc_ids = None
        if 'doc_ids' in batch:
            doc_ids = batch['doc_ids'].to(self.device, non_blocking=True)

        num_chords = batch.get('num_chords', None)  # int，collator 预计算，避免 forward 内 .max().item()

        return token_ids, labels, chord_ids, instrument_ids, token_type_ids, note_ids, doc_ids, num_chords

    def train_epoch(self) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        step_start = time.time()
        epoch_start = time.time()

        if self.world_size > 1:
            self.train_loader.sampler.set_epoch(self.epoch)

        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {self.epoch + 1}',
            disable=not self.is_main,
            dynamic_ncols=True,
            leave=True,
        )

        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue

            token_ids, labels, chord_ids, instrument_ids, token_type_ids, note_ids, doc_ids, num_chords = \
                self._move_batch_to_device(batch)

            # FlexAttention 要求 forward 和 backward 都在相同的 autocast 上下文中
            with torch.autocast(device_type='cuda', dtype=self.autocast_dtype):
                logits = self.model(
                    token_ids,
                    chord_ids=chord_ids,
                    instrument_ids=instrument_ids,
                    token_type_ids=token_type_ids,
                    note_ids=note_ids,
                    num_chords=num_chords,
                    doc_ids=doc_ids,
                )

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                loss = loss / self.args.gradient_accumulation_steps

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

            batch_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * self.args.gradient_accumulation_steps
            total_tokens += batch_tokens
            num_batches += 1

            # 梯度累积
            if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                # Two-phase DataLoader switch
                if self.global_step == WARMUP_PHASE_STEPS and self._phase == 1:
                    if self.is_main:
                        tqdm.write(f"\n{'='*60}")
                        tqdm.write(f"Phase 2: num_workers: 0 -> {self._target_num_workers}")
                        tqdm.write(f"{'='*60}")
                    self._phase = 2
                    self.train_loader = self._make_dataloader(
                        self.train_dataset,
                        num_workers=self._target_num_workers,
                        sampler=self.train_sampler,
                        shuffle=(self.train_sampler is None),
                    )

                # 日志
                if self.is_main and self.global_step % self.args.log_interval == 0:
                    now = time.time()
                    step_time_ms = (now - step_start) * 1000 / self.args.log_interval
                    elapsed = now - epoch_start
                    avg_loss = total_loss / num_batches
                    tokens_per_sec = total_tokens / max(elapsed, 1e-6)
                    lr = self.scheduler.get_last_lr()[0]
                    mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                    torch.cuda.reset_peak_memory_stats()

                    row = {
                        'step': self.global_step,
                        'epoch': self.epoch + 1,
                        'loss': avg_loss,
                        'lr': lr,
                        'tok/s': tokens_per_sec,
                        'ms/step': step_time_ms,
                        'mem_gb': mem_gb,
                    }
                    self._step_log_rows.append(row)

                    pbar.set_postfix(
                        loss=f'{avg_loss:.4f}',
                        lr=f'{lr:.2e}',
                        tok_s=f'{tokens_per_sec:.0f}',
                        ms=f'{step_time_ms:.0f}',
                        mem=f'{mem_gb:.1f}G',
                    )
                    tqdm.write(
                        f'  Step {self.global_step:>6} | '
                        f'Loss {avg_loss:.4f} | '
                        f'LR {lr:.2e} | '
                        f'Tok/s {tokens_per_sec:>7.0f} | '
                        f'{step_time_ms:>5.0f}ms/step | '
                        f'Mem {mem_gb:.1f}G'
                    )
                    self._file_logger.info(
                        'step=%d epoch=%d loss=%.4f lr=%.2e tok/s=%.0f ms/step=%.0f mem_gb=%.2f',
                        self.global_step, self.epoch + 1, avg_loss, lr,
                        tokens_per_sec, step_time_ms, mem_gb,
                    )
                    step_start = now

                    if self.args.use_wandb:
                        import wandb
                        wandb.log({
                            'train/loss': avg_loss,
                            'train/learning_rate': lr,
                            'train/tokens_per_sec': tokens_per_sec,
                            'train/ms_per_step': step_time_ms,
                            'train/mem_gb': mem_gb,
                            'train/step': self.global_step,
                        })

                # 保存检查点
                if self.is_main and self.global_step % self.args.save_interval == 0:
                    self.save_checkpoint(f'step_{self.global_step}')

        pbar.close()
        return {'loss': total_loss / max(num_batches, 1)}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            if batch is None:
                continue

            token_ids, labels, chord_ids, instrument_ids, token_type_ids, note_ids, doc_ids, num_chords = \
                self._move_batch_to_device(batch)

            with torch.autocast(device_type='cuda', dtype=self.autocast_dtype):
                logits = self.model(
                    token_ids,
                    chord_ids=chord_ids,
                    instrument_ids=instrument_ids,
                    token_type_ids=token_type_ids,
                    note_ids=note_ids,
                    num_chords=num_chords,
                    doc_ids=doc_ids,
                )

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        if self.world_size > 1:
            loss_tensor = torch.tensor([avg_loss], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / self.world_size

        return {'val_loss': avg_loss}

    def train(self):
        """完整训练循环"""
        if self.is_main:
            print("\n" + "=" * 60)
            print("MeloFormer H800 训练")
            print("=" * 60)
            print(f"设备: {self.device}")
            print(f"世界大小: {self.world_size}")
            print(f"批大小 (per GPU): {BATCH_SIZE}")
            print(f"梯度累积: {self.args.gradient_accumulation_steps}")
            print(f"有效批大小: {BATCH_SIZE * self.world_size * self.args.gradient_accumulation_steps}")
            print(f"Epochs: {self.args.epochs}")
            print(f"学习率: {self.args.learning_rate}")
            print(f"混合精度: {'BF16' if self.args.bf16 else ('FP16' if self.args.fp16 else 'FP32')}")
            print(f"TF32: {torch.backends.cuda.matmul.allow_tf32}")
            print(f"Attention: FlexAttention (v2.1 captured tensor mask_mod)")
            print("=" * 60 + "\n")

        for epoch in range(self.args.epochs):
            self.epoch = epoch

            if self.is_main:
                print(f"\n{'='*50}")
                print(f"Epoch {epoch + 1}/{self.args.epochs}")
                print(f"{'='*50}")

            epoch_start = time.time()
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            epoch_time = time.time() - epoch_start

            if self.is_main:
                val_str = ''
                best_str = ''
                if val_metrics:
                    val_str = f" | Val Loss: {val_metrics['val_loss']:.4f}"
                    if val_metrics['val_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val_loss']
                        self.save_checkpoint('best')
                        best_str = ' *best*'

                epoch_row = {
                    'epoch': epoch + 1,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics.get('val_loss', float('nan')),
                    'time_s': epoch_time,
                }
                self._epoch_log_rows.append(epoch_row)

                msg = (
                    f"Epoch {epoch + 1} | "
                    f"Train Loss: {train_metrics['loss']:.4f}{val_str} | "
                    f"Time: {epoch_time:.1f}s{best_str}"
                )
                tqdm.write(msg)
                self._file_logger.info(msg)

                if self.args.use_wandb:
                    import wandb
                    wandb.log({
                        'epoch': epoch + 1,
                        'epoch_time': epoch_time,
                        **{f'train/{k}': v for k, v in train_metrics.items()},
                        **{f'val/{k}': v for k, v in val_metrics.items()},
                    })

            if self.is_main:
                self.save_checkpoint(f'epoch_{epoch + 1}')

        if self.is_main:
            self.save_checkpoint('final')
            self._print_summary()


    def _print_summary(self):
        """训练结束后打印汇总表格并写入日志文件"""
        lines = []
        lines.append('')
        lines.append('=' * 78)
        lines.append('  训练汇总 - Per-Step 统计')
        lines.append('=' * 78)
        hdr = f"{'Step':>7}  {'Epoch':>5}  {'Loss':>8}  {'LR':>9}  {'Tok/s':>8}  {'ms/step':>8}  {'Mem(G)':>7}"
        lines.append(hdr)
        lines.append('-' * 78)
        for r in self._step_log_rows:
            lines.append(
                f"{r['step']:>7}  {r['epoch']:>5}  {r['loss']:>8.4f}  "
                f"{r['lr']:>9.2e}  {r['tok/s']:>8.0f}  "
                f"{r['ms/step']:>8.0f}  {r['mem_gb']:>7.2f}"
            )
        if self._step_log_rows:
            avg_loss   = sum(r['loss']    for r in self._step_log_rows) / len(self._step_log_rows)
            avg_toks   = sum(r['tok/s']   for r in self._step_log_rows) / len(self._step_log_rows)
            avg_ms     = sum(r['ms/step'] for r in self._step_log_rows) / len(self._step_log_rows)
            avg_mem    = sum(r['mem_gb']  for r in self._step_log_rows) / len(self._step_log_rows)
            lines.append('-' * 78)
            lines.append(
                f"{'avg':>7}  {'':>5}  {avg_loss:>8.4f}  "
                f"{'':>9}  {avg_toks:>8.0f}  "
                f"{avg_ms:>8.0f}  {avg_mem:>7.2f}"
            )
        lines.append('')
        lines.append('=' * 78)
        lines.append('  训练汇总 - Per-Epoch 统计')
        lines.append('=' * 78)
        lines.append(f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>9}  {'Time(s)':>8}")
        lines.append('-' * 40)
        for r in self._epoch_log_rows:
            val_str = f"{r['val_loss']:>9.4f}" if not (r['val_loss'] != r['val_loss']) else '        -'
            lines.append(f"{r['epoch']:>6}  {r['train_loss']:>10.4f}  {val_str}  {r['time_s']:>8.1f}")
        lines.append('=' * 78)
        lines.append('')

        summary = '\n'.join(lines)
        tqdm.write(summary)
        self._file_logger.info(summary)

        # 同时写一份独立 summary 文件
        summary_path = self.log_dir / 'summary.txt'
        summary_path.write_text(summary, encoding='utf-8')
        tqdm.write(f'[+] 汇总写入: {summary_path}')

    def save_checkpoint(self, name: str):
        """保存检查点"""
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'args': vars(self.args),
        }

        path = self.checkpoint_dir / f'{name}.pt'
        torch.save(checkpoint, path)
        print(f"检查点已保存: {path}")

    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)

        model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"检查点已加载: {path}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _validate_batch_size(value):
    """Fail fast: batch_size must be 1 in v2.1."""
    ivalue = int(value)
    if ivalue != BATCH_SIZE:
        raise argparse.ArgumentTypeError(
            f"v2.1 requires batch_size={BATCH_SIZE} (captured tensor mask_mod constraint). "
            f"Use --gradient_accumulation_steps to control effective batch size."
        )
    return ivalue


def parse_args():
    parser = argparse.ArgumentParser(description='MeloFormer H800 训练')

    # 数据
    parser.add_argument('--data_dir', type=str, required=True, help='预处理数据目录')
    parser.add_argument('--val_split', type=float, default=0.05, help='验证集比例')
    parser.add_argument('--max_seq_len', type=int, default=24576, help='最大序列长度 (H800: 24K+)')
    parser.add_argument('--max_samples', type=int, default=None, help='限制加载样本数（用于快速测试）')
    parser.add_argument('--use_arrow', action='store_true',
                        help='使用 HuggingFace Arrow 格式 (需要先用 convert_to_arrow.py 转换)')

    # 模型
    parser.add_argument('--model_size', type=str, default='base',
                        choices=['8m', '62m', '177m', '366m', '600m', '800m', '1.5b',
                                 'small', 'base', 'large', 'xlarge'], help='模型大小')
    parser.add_argument('--max_chords', type=int, default=2048,
                        help='最大 chord 数量 (默认 2048，覆盖 99.8%% 样本)')

    # v1.4 高级优化
    parser.add_argument('--num_kv_heads', type=int, default=None,
                        help='GQA KV 头数 (默认: 根据 model_size 自动配置)')
    parser.add_argument('--use_rms_norm', action='store_true', default=True,
                        help='使用 RMSNorm (v1.4 默认开启)')
    parser.add_argument('--no_rms_norm', action='store_true',
                        help='禁用 RMSNorm，使用 LayerNorm')

    # v3.0 架构优化开关
    parser.add_argument('--sub_attn_bias', action='store_true', default=False,
                        help='启用子注意力 additive bias (v3.0)')
    parser.add_argument('--k2v2_gate', action='store_true', default=False,
                        help='启用 K2/V2 需求门控 (v3.0)')
    parser.add_argument('--depth_selector', action='store_true', default=False,
                        help='启用 DepthSelector 跨层聚合 (v3.0)')

    # 训练
    parser.add_argument('--epochs', type=int, default=50, help='训练 epochs')
    parser.add_argument('--batch_size', type=_validate_batch_size, default=BATCH_SIZE,
                        help=f'批大小 (per GPU)，v2.1 固定为 {BATCH_SIZE}，使用 --gradient_accumulation_steps 控制有效批大小')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='梯度累积步数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='权重衰减')
    parser.add_argument('--warmup_steps', type=int, default=2000, help='预热步数')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪')

    # 序列打包优化
    parser.add_argument('--use_packing', action='store_true',
                        help='启用序列打包 (Sequence Packing)，将多首短曲拼接成目标长度，提升训练效率 1.5-2x')

    # H800 优化
    parser.add_argument('--bf16', action='store_true', default=True, help='使用 BF16 (推荐)')
    parser.add_argument('--fp16', action='store_true', help='使用 FP16')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True, help='使用梯度检查点')
    # --compile 已废弃: MeloFormer v2.1 不在模型层面使用 torch.compile(model)
    # FlexAttention 内部已独立编译，双层 compile 导致持续 graph break
    # 保留参数仅为兼容旧命令行，无实际效果
    parser.add_argument('--compile', action='store_true', default=False, help='(已废弃，无效果)')
    parser.add_argument('--no_compile', action='store_true', help='(已废弃，无效果)')
    parser.add_argument('--flex_backend', type=str, default='TRITON', choices=['TRITON', 'FA4'],
                        help='FlexAttention 后端: TRITON (默认，dynamic=True) 或 FA4 (需要 PyTorch nightly + FlashAttention-4)')
    parser.add_argument('--warmup_compile', action='store_true', default=False,
                        help='训练前预编译实际会遇到的 FlexAttention kernel 桶组合')

    # 输出
    parser.add_argument('--output_dir', type=str, default='./runs', help='输出目录 (checkpoints, logs)')
    parser.add_argument('--log_interval', type=int, default=50, help='日志间隔')
    parser.add_argument('--save_interval', type=int, default=2000, help='保存间隔')

    # 其他
    parser.add_argument('--num_workers', type=int, default=DEFAULT_NUM_WORKERS,
                        help=f'数据加载工作进程数 (H800 推荐 {DEFAULT_NUM_WORKERS}+)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')

    # WandB
    parser.add_argument('--use_wandb', action='store_true', help='使用 WandB 记录')
    parser.add_argument('--wandb_project', type=str, default='meloformer-h800', help='WandB 项目名')
    parser.add_argument('--run_name', type=str, default=None, help='运行名称')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# FlexAttention 预编译 Warm-up
# ---------------------------------------------------------------------------

def _scan_dataset_bucket_distribution(
    dataset,
    max_seq_len: int,
    max_chords: int,
    max_samples: int = 5000,
) -> List[Tuple[int, int]]:
    """
    扫描数据集前 max_samples 个样本，统计 (bucketed_seq_len, bucketed_num_chords) 分布。
    返回按频率降序排列的桶组合列表。
    """
    from model.attention_flex_summary import SEQ_BUCKET_SIZE, CHORD_BUCKET_SIZE, _bucket
    from collections import Counter

    counter = Counter()
    for i in range(min(len(dataset), max_samples)):
        try:
            sample = dataset[i]
            if sample is None:
                continue
            seq_len = min(sample['length'], max_seq_len)
            if 'num_chords' in sample:
                nc = sample['num_chords']
            elif 'chord_ids' in sample:
                nc = int(sample['chord_ids'][:seq_len].max().item()) + 1
            else:
                continue
            if nc > max_chords:
                continue
            bsl = _bucket(seq_len, SEQ_BUCKET_SIZE)
            bnc = _bucket(nc, CHORD_BUCKET_SIZE)
            counter[(bsl, bnc)] += 1
        except Exception:
            continue

    return [combo for combo, _ in counter.most_common()]


def warmup_flex_compile(
    model: nn.Module,
    device: torch.device,
    max_seq_len: int,
    max_chords: int,
    is_main: bool = True,
):
    """
    FlexAttention TRITON 预编译。

    dynamic=True 一次 fwd+bwd 编译即覆盖所有 shape，
    后续不同 seq_len/num_chords 自动复用同一 kernel。
    """
    import time

    if is_main:
        print("\n[Warmup] TRITON dynamic=True: 单次 fwd+bwd 编译...")

    base_model = model.module if hasattr(model, 'module') else model

    # 构造 dummy 输入: 使用较小的序列避免 OOM
    warmup_seq = min(max_seq_len, 2048)
    warmup_chords = min(max_chords, 64)
    dummy_t = torch.randint(0, 100, (1, warmup_seq), dtype=torch.long, device=device)
    dummy_c = torch.zeros(1, warmup_seq, dtype=torch.long, device=device)
    tokens_per_chord = max(1, warmup_seq // warmup_chords)
    for t in range(warmup_seq):
        dummy_c[0, t] = min(t // tokens_per_chord, warmup_chords - 1)

    t0 = time.time()
    try:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = base_model(dummy_t, chord_ids=dummy_c, num_chords=warmup_chords)
            loss = out.sum()
            loss.backward()
        base_model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        elapsed = time.time() - t0
        if is_main:
            print(f"[Warmup] 编译完成 ({elapsed:.1f}s)\n")
    except Exception as e:
        if is_main:
            print(f"[Warmup] 编译失败: {e}\n")


# ---------------------------------------------------------------------------

def main():
    # Configure torch before anything else
    configure_torch_for_training()

    args = parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 检测混合精度可用性
    if args.bf16:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            pass
        else:
            print("[Warning] BF16 不支持 (需要 Ampere+ GPU)，回退到 FP32")
            args.bf16 = False
    if args.fp16 and args.bf16:
        print("[Warning] 同时指定了 --fp16 和 --bf16，优先使用 FP16")
        args.bf16 = False

    # 创建 tokenizer
    tokenizer = HIDTokenizerV2()

    # 加载预处理数据集
    print("加载预处理数据集...")
    if args.use_arrow:
        full_dataset = ArrowDataset(
            args.data_dir,
            max_seq_len=args.max_seq_len,
            max_chords=args.max_chords,
            max_samples=args.max_samples,
        )
    else:
        full_dataset = PreprocessedDataset(
            args.data_dir,
            max_seq_len=args.max_seq_len,
            max_chords=args.max_chords,
            max_samples=args.max_samples,
        )
    print(f"max_chords={args.max_chords} (超过此值的样本将被跳过)")

    # 划分训练/验证集
    total_size = len(full_dataset)
    val_size = int(total_size * args.val_split)
    train_size = total_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")

    # v2.1 参数处理 (2D RoPE 已移除，统一使用 1D RoPE)
    use_rms_norm = args.use_rms_norm and not args.no_rms_norm

    # 创建模型 (FlexAttention + GQA + RMSNorm + 1D RoPE)
    set_flex_backend(args.flex_backend)

    print(f"\n创建模型 (size={args.model_size}, max_chords={args.max_chords}, attention=FlexAttention/{args.flex_backend})...")
    print(f"  v2.1 特性: GQA={'auto' if args.num_kv_heads is None else args.num_kv_heads}, "
          f"RMSNorm={use_rms_norm}, RoPE=1D")
    model = create_model(
        vocab_size=tokenizer.vocab_size,
        model_size=args.model_size,
        max_seq_len=args.max_seq_len,
        max_chords=args.max_chords,
        chord_start_id=tokenizer.chord_start_id,
        chord_end_id=tokenizer.chord_end_id,
        num_kv_heads=args.num_kv_heads,
        use_rms_norm=use_rms_norm,
        sub_attn_bias=args.sub_attn_bias,
        k2v2_gate=args.k2v2_gate,
        depth_selector=args.depth_selector,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params / 1e6:.2f}M")

    # 创建训练器
    trainer = H800Trainer(model, train_dataset, val_dataset, args)

    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # FlexAttention 预编译 warm-up (可选, 首次运行时显著减少编译停顿)
    if getattr(args, 'warmup_compile', False):
        warmup_flex_compile(
            model, trainer.device, args.max_seq_len, args.max_chords,
            is_main=trainer.is_main,
        )

    # 开始训练
    try:
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()
