#!/usr/bin/env python3
"""MeloFormer 训练脚本 - 支持单卡/DDP 多卡训练"""

import os
import sys
import math
import time
import json
import random
import argparse
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

import torch

# 增加 dynamo cache size 避免 FlexAttention 重编译问题
# FlexAttention 的 create_block_mask 会为每个不同的序列长度重编译
# 即使使用 512 的分桶，8192 长度也有 16 种变体
# 加上 batch size、num_bars 等变化，需要非常大的缓存
import torch._dynamo
torch._dynamo.config.cache_size_limit = 16384  # v1.2.1: 从 2048 增加到 16384
torch._dynamo.config.accumulated_cache_size_limit = 32768  # 对应增加累积缓存限制
torch._dynamo.config.suppress_errors = True  # 如果仍然超限，回退到 eager 模式而不是崩溃

# H800 TF32 优化 (v1.3)
# TF32 在 H800/A100 上提供 FP32 精度的 8x 加速
# 对于 matmul 和 convolution 操作自动使用 TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from tqdm import tqdm

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    print("[Warning] Flash Attention not available, using PyTorch SDPA")

from model import MeloFormer, create_model
from model.attention_flex_summary import set_static_compile_mode, is_static_compile_mode, MAX_SEQ_LEN, MAX_SUM_LEN
from data import HIDTokenizerV2

# HuggingFace datasets (可选，用于 Arrow 格式)
try:
    from datasets import load_from_disk
    HAS_HF_DATASETS = True
except ImportError:
    HAS_HF_DATASETS = False


class ArrowDataset(Dataset):
    """
    HuggingFace Arrow 格式数据集 - 零拷贝内存映射

    优势:
    1. 零拷贝 - 不需要反序列化，直接内存映射
    2. O(1) 随机访问 - 任意样本直接定位
    3. 多进程安全 - num_workers 可以开很大
    4. 已按长度排序 - 减少 padding
    """

    def __init__(
        self,
        data_path: str,
        max_seq_len: int = 24576,
        max_bars: int = 2048,
        max_samples: int = None,
    ):
        if not HAS_HF_DATASETS:
            raise ImportError("需要安装 datasets: pip install datasets")

        self.max_seq_len = max_seq_len
        self.max_bars = max_bars

        print(f"加载 Arrow 数据集: {data_path}")
        self.dataset = load_from_disk(data_path)

        if max_samples and max_samples < len(self.dataset):
            self.dataset = self.dataset.select(range(max_samples))

        print(f"Arrow 数据集: {len(self.dataset):,} 个样本")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        item = self.dataset[idx]

        # 转换为 tensor
        sample = {
            'token_ids': torch.tensor(item['token_ids'], dtype=torch.long),
            'chord_ids': torch.tensor(item['chord_ids'], dtype=torch.long),
            'instrument_ids': torch.tensor(item['instrument_ids'], dtype=torch.long),
            'token_type_ids': torch.tensor(item['token_type_ids'], dtype=torch.long),
            'note_ids': torch.tensor(item['note_ids'], dtype=torch.long),
            'length': item['length'],
        }

        # position_ids 如果没有，生成默认的
        if 'position_ids' in item:
            sample['position_ids'] = torch.tensor(item['position_ids'], dtype=torch.long)
        else:
            sample['position_ids'] = torch.arange(sample['length'], dtype=torch.long)

        # 检查 bar 数量
        num_bars = sample['chord_ids'].max().item() + 1
        if num_bars > self.max_bars:
            return None

        # 截断
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
        max_bars: int = 2048,
        shuffle_files: bool = True,
        max_samples: int = None,
    ):
        self.max_seq_len = max_seq_len
        self.max_bars = max_bars
        self.samples = []
        self.lazy_mode = False  # 是否使用懒加载
        self._skipped_count = 0  # 跳过的样本计数

        data_path = Path(data_path)

        # 检测数据格式
        shard_files = sorted(data_path.glob('shard_*.pt'))

        if shard_files:
            # 分片格式 - 使用懒加载 (只记录索引，不加载数据)
            print(f"检测到分片格式: {len(shard_files)} 个分片")

            # 读取 meta.json 获取样本数量
            meta_path = data_path / 'meta.json'
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                total_samples = meta.get('total_samples', 0)
                shard_size = meta.get('shard_size', 10000)
                print(f"从 meta.json 读取: {total_samples:,} 个样本")
            else:
                # 没有 meta.json，扫描第一个分片估算
                first_shard = torch.load(shard_files[0], weights_only=False)
                shard_size = len(first_shard)
                total_samples = shard_size * len(shard_files)
                del first_shard
                print(f"估算样本数: {total_samples:,}")

            # 设置懒加载模式
            self.lazy_mode = True
            self.shard_files = shard_files
            self.shard_size = shard_size
            self.total_samples = min(total_samples, max_samples) if max_samples else total_samples

            # 分片缓存 (LRU 缓存最近使用的分片)
            self._shard_cache = {}
            self._cache_order = []
            # 120GB 内存优化：缓存 30 个分片（约占用 30-35GB）
            # 渐进式切换策略：Step 50 才切换到 num_workers=8
            # 给编译和初始训练预留足够内存缓冲
            self._max_cache_size = 30  # 平衡速度和内存安全

            # 构建索引映射: global_idx -> (shard_idx, local_idx)
            self.index_map = []
            sample_count = 0
            for shard_idx, shard_file in enumerate(shard_files):
                # 最后一个分片可能不满
                if shard_idx == len(shard_files) - 1:
                    remaining = self.total_samples - sample_count
                    shard_samples = remaining
                else:
                    shard_samples = shard_size

                for local_idx in range(shard_samples):
                    if max_samples and sample_count >= max_samples:
                        break
                    self.index_map.append((shard_idx, local_idx))
                    sample_count += 1

                if max_samples and sample_count >= max_samples:
                    break

            # Shuffle 索引
            if shuffle_files:
                random.shuffle(self.index_map)

            print(f"懒加载模式: {len(self.index_map):,} 个样本 (缓存 {self._max_cache_size} 个分片)")

        else:
            # 单文件格式 (旧版兼容) - 仍然加载到内存
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
                    if data['length'] > max_seq_len:
                        for key in ['token_ids', 'chord_ids', 'position_ids', 'instrument_ids', 'token_type_ids', 'note_ids']:
                            if key in data:
                                data[key] = data[key][:max_seq_len]
                        data['length'] = max_seq_len
                    self.samples.append(data)
                except:
                    pass

            print(f"已加载 {len(self.samples):,} 个样本到内存")

            if shuffle_files:
                random.shuffle(self.samples)

    def _load_shard(self, shard_idx: int):
        """加载分片 (带 LRU 缓存)"""
        if shard_idx in self._shard_cache:
            # 移到最近使用
            self._cache_order.remove(shard_idx)
            self._cache_order.append(shard_idx)
            return self._shard_cache[shard_idx]

        # 加载新分片
        shard_data = torch.load(self.shard_files[shard_idx], weights_only=False)

        # 缓存管理
        if len(self._cache_order) >= self._max_cache_size:
            # 删除最旧的
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

        # 检查 bar 数量是否超过 max_bars
        if 'chord_ids' in sample:
            num_bars = sample['chord_ids'].max().item() + 1
            if num_bars > self.max_bars:
                self._skipped_count += 1
                return None  # 跳过超长样本

        # 截断序列长度
        if sample['length'] > self.max_seq_len:
            sample = {k: v[:self.max_seq_len] if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
            sample['length'] = self.max_seq_len

        return sample


def efficient_collate_fn(batch: List[Optional[Dict]]) -> Optional[Dict[str, torch.Tensor]]:
    """
    高效的批次整理函数
    - 序列长度分桶: 减少 FlexAttention 重编译次数
    - 动态 padding 到 batch 内最大长度 (减少无效计算)
    - 过滤掉 None 样本 (超过 max_bars 的样本)

    v1.3 更新: 静态编译模式下，所有序列 Padding 到固定 MAX_SEQ_LEN
    """
    # 过滤掉 None 样本
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None  # 整个 batch 都被跳过

    # 按长度排序 (减少 padding)
    batch = sorted(batch, key=lambda x: x['length'], reverse=True)

    # 静态编译模式: 使用固定长度
    if is_static_compile_mode():
        max_len = MAX_SEQ_LEN
    else:
        max_len = batch[0]['length']
        # 序列长度分桶 (关键优化！)
        # 将 max_len 向上取整到最近的桶边界
        # 桶大小: 512 tokens
        # 这样 FlexAttention 只需要编译 16 个 kernel (8192/512=16)
        BUCKET_SIZE = 512
        max_len = ((max_len + BUCKET_SIZE - 1) // BUCKET_SIZE) * BUCKET_SIZE
        max_len = min(max_len, 16384)  # 不超过 max_seq_len

    batch_size = len(batch)

    # 预分配张量
    token_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    chord_ids = torch.full((batch_size, max_len), -1, dtype=torch.long)
    position_ids = torch.full((batch_size, max_len), -1, dtype=torch.long)
    instrument_ids = torch.full((batch_size, max_len), 129, dtype=torch.long)
    token_type_ids = torch.full((batch_size, max_len), -1, dtype=torch.long)  # Token 类型
    note_ids = torch.full((batch_size, max_len), -1, dtype=torch.long)  # 音符 ID
    lengths = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        length = item['length']
        token_ids[i, :length] = item['token_ids']
        labels[i, :length] = item['token_ids']
        chord_ids[i, :length] = item['chord_ids']
        position_ids[i, :length] = item['position_ids']
        instrument_ids[i, :length] = item['instrument_ids']
        # Token Type Sparsity 需要的字段
        if 'token_type_ids' in item:
            token_type_ids[i, :length] = item['token_type_ids']
        if 'note_ids' in item:
            note_ids[i, :length] = item['note_ids']
        lengths[i] = length

    return {
        'token_ids': token_ids,
        'labels': labels,
        'chord_ids': chord_ids,
        'position_ids': position_ids,
        'instrument_ids': instrument_ids,
        'token_type_ids': token_type_ids,
        'note_ids': note_ids,
        'lengths': lengths,
    }


class PackingCollator:
    """
    序列打包 Collator - 将多首短曲拼接成目标长度

    优势:
    - 消除 Padding 浪费，有效 token 利用率从 50-60% 提升到 ~100%
    - 训练速度提升 1.5-2x

    原理:
    - 贪心算法将多首曲子拼接到 target_length
    - 通过 doc_ids 标记不同曲子，在 FlexAttention 中隔离注意力
    """

    def __init__(self, target_length: int = 16384, bucket_size: int = 512):
        """
        Args:
            target_length: 目标序列长度 (拼接后的最大长度)
            bucket_size: 分桶大小 (用于减少 FlexAttention 重编译)
        """
        self.target_length = target_length
        self.bucket_size = bucket_size

    def __call__(self, batch: List[Optional[Dict]]) -> Optional[Dict[str, torch.Tensor]]:
        # 过滤 None 样本
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None

        # 按长度降序排序 (贪心打包效果更好)
        batch = sorted(batch, key=lambda x: x['length'], reverse=True)

        # 贪心打包: 尽量填满每个 pack
        packed_sequences = []
        current_pack = {'items': [], 'total_len': 0}

        for item in batch:
            length = item['length']

            # 如果单个样本超过 target_length，单独成 pack
            if length > self.target_length:
                # 先保存当前 pack
                if current_pack['items']:
                    packed_sequences.append(current_pack)
                # 截断超长样本
                truncated_item = {
                    k: v[:self.target_length] if isinstance(v, torch.Tensor) else v
                    for k, v in item.items()
                }
                truncated_item['length'] = self.target_length
                packed_sequences.append({'items': [truncated_item], 'total_len': self.target_length})
                current_pack = {'items': [], 'total_len': 0}
            elif current_pack['total_len'] + length <= self.target_length:
                # 可以放入当前 pack
                current_pack['items'].append(item)
                current_pack['total_len'] += length
            else:
                # 当前 pack 已满，开启新 pack
                if current_pack['items']:
                    packed_sequences.append(current_pack)
                current_pack = {'items': [item], 'total_len': length}

        # 保存最后一个 pack
        if current_pack['items']:
            packed_sequences.append(current_pack)

        # 构建输出 tensor
        batch_size = len(packed_sequences)
        max_len = self._bucket_length(max(p['total_len'] for p in packed_sequences))

        # 预分配张量
        token_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        doc_ids = torch.zeros(batch_size, max_len, dtype=torch.long)  # 关键: 文档 ID
        chord_ids = torch.full((batch_size, max_len), -1, dtype=torch.long)
        position_ids = torch.full((batch_size, max_len), -1, dtype=torch.long)
        instrument_ids = torch.full((batch_size, max_len), 129, dtype=torch.long)
        token_type_ids = torch.full((batch_size, max_len), -1, dtype=torch.long)
        note_ids = torch.full((batch_size, max_len), -1, dtype=torch.long)
        lengths = torch.zeros(batch_size, dtype=torch.long)

        # 填充每个 pack
        for b_idx, pack in enumerate(packed_sequences):
            offset = 0
            for doc_idx, item in enumerate(pack['items']):
                length = item['length']
                end_pos = offset + length

                # 填充各字段
                token_ids[b_idx, offset:end_pos] = item['token_ids'][:length]
                labels[b_idx, offset:end_pos] = item['token_ids'][:length]
                doc_ids[b_idx, offset:end_pos] = doc_idx  # 关键: 分配文档 ID
                chord_ids[b_idx, offset:end_pos] = item['chord_ids'][:length]
                instrument_ids[b_idx, offset:end_pos] = item['instrument_ids'][:length]

                if 'position_ids' in item:
                    position_ids[b_idx, offset:end_pos] = item['position_ids'][:length]
                if 'token_type_ids' in item:
                    token_type_ids[b_idx, offset:end_pos] = item['token_type_ids'][:length]
                if 'note_ids' in item:
                    note_ids[b_idx, offset:end_pos] = item['note_ids'][:length]

                offset = end_pos

            lengths[b_idx] = pack['total_len']

        return {
            'token_ids': token_ids,
            'labels': labels,
            'doc_ids': doc_ids,  # 新增
            'chord_ids': chord_ids,
            'position_ids': position_ids,
            'instrument_ids': instrument_ids,
            'token_type_ids': token_type_ids,
            'note_ids': note_ids,
            'lengths': lengths,
        }

    def _bucket_length(self, length: int) -> int:
        """将长度向上取整到最近的桶边界

        v1.3 更新: 静态编译模式下返回固定 MAX_SEQ_LEN
        """
        if is_static_compile_mode():
            return MAX_SEQ_LEN
        bucketed = ((length + self.bucket_size - 1) // self.bucket_size) * self.bucket_size
        return min(bucketed, self.target_length)


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
    """
    余弦退火 + 线性预热
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class H800Trainer:
    """
    H800 优化训练器
    """

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

        # 设备
        self.device = torch.device(f'cuda:{self.local_rank}')

        # 模型
        self.model = model.to(self.device)

        # Gradient Checkpointing
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            if self.is_main:
                print("[✓] Gradient Checkpointing enabled")

        # torch.compile (PyTorch 2.0+) — FlexAttention 稀疏加速必需
        if args.compile and not args.no_compile and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model, mode='reduce-overhead')
            if self.is_main:
                print("[✓] torch.compile enabled (FlexAttention sparse acceleration)")
        elif self.is_main:
            print("[!] torch.compile disabled — FlexAttention will fallback to eager (no sparse speedup)")

        # DDP
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=False,
            )

        # 数据加载器 (三阶段动态优化)
        # Phase 1 (Step 1-10): batch_size=1, num_workers=0
        #   - 避免多进程 + 编译冲突
        # Phase 2 (Step 11-100): batch_size=1, num_workers=8
        #   - 加速数据加载，GPU 利用率提升
        #   - 继续用小 batch 避免编译时 OOM
        # Phase 3 (Step 101+): batch_size=目标值, num_workers=8
        #   - 全速训练
        self._phase = 1  # 当前阶段
        self._compilation_batch_size = 1  # 编译阶段最小 batch
        self._target_batch_size = args.batch_size  # 训练阶段目标 batch
        self._target_num_workers = args.num_workers if args.num_workers > 0 else 8

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # 序列打包配置
        self._use_packing = getattr(args, 'use_packing', False)
        if self._use_packing:
            # 使用 PackingCollator 将多首短曲拼接
            self._collate_fn = PackingCollator(
                target_length=min(args.max_seq_len, 16384),  # 打包目标长度
                bucket_size=512
            )
            if self.is_main:
                print(f"[✓] 序列打包已启用 (target_length={min(args.max_seq_len, 16384)})")
        else:
            self._collate_fn = efficient_collate_fn

        train_sampler = DistributedSampler(train_dataset, shuffle=True) if self.world_size > 1 else None
        self.train_sampler = train_sampler
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self._compilation_batch_size,  # Phase 1: 从 batch_size=1 开始
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=0,  # Phase 1: 编译时用 0 避免 OOM
            collate_fn=self._collate_fn,
            pin_memory=True,
            prefetch_factor=None,
            persistent_workers=False,
        )

        if val_dataset is not None:
            val_sampler = DistributedSampler(val_dataset, shuffle=False) if self.world_size > 1 else None
            self.val_sampler = val_sampler
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self._compilation_batch_size,  # Phase 1: 验证集也用最小 batch
                shuffle=False,
                sampler=val_sampler,
                num_workers=0,
                collate_fn=self._collate_fn,  # 使用相同的 collate 函数
                pin_memory=True,
            )
        else:
            self.val_loader = None

        # 优化器 (使用 fused Adam)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
            fused=True,  # H800 优化
        )

        # 学习率调度器
        total_steps = len(self.train_loader) * args.epochs // args.gradient_accumulation_steps
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps,
        )

        # 混合精度配置
        # BF16: H800/A100 原生支持，更稳定，不需要 GradScaler
        # FP16: 需要 GradScaler 防止梯度下溢 (H800 不推荐)
        if args.fp16:
            self.scaler = torch.amp.GradScaler('cuda')
            self.autocast_dtype = torch.float16
            if self.is_main:
                print("[✓] FP16 混合精度 (GradScaler enabled)")
                print("[⚠] 警告: H800 推荐使用 BF16 而非 FP16")
        elif args.bf16:
            self.scaler = None  # BF16 不需要 GradScaler
            self.autocast_dtype = torch.bfloat16
            if self.is_main:
                print("[✓] BF16 混合精度 (H800 推荐)")
        else:
            self.scaler = None
            self.autocast_dtype = torch.float32
            if self.is_main:
                print("[✓] FP32 精度")

        # TF32 状态 (H800 优化)
        if self.is_main:
            tf32_matmul = torch.backends.cuda.matmul.allow_tf32
            tf32_cudnn = torch.backends.cudnn.allow_tf32
            print(f"[✓] TF32: matmul={tf32_matmul}, cudnn={tf32_cudnn}")

        # FlexAttention + Gradient Checkpointing + 混合精度兼容性处理
        # 将 autocast_dtype 传递给模型，用于选择性 checkpoint
        if args.gradient_checkpointing:
            base_model = self.model.module if hasattr(self.model, 'module') else self.model
            base_model._autocast_dtype = self.autocast_dtype

        # 状态
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # 日志
        self.log_dir = Path(args.output_dir) / 'logs'
        self.checkpoint_dir = Path(args.output_dir) / 'checkpoints'

        if self.is_main:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # WandB
            if args.use_wandb:
                import wandb
                wandb.init(
                    project=args.wandb_project,
                    name=args.run_name or f"meloformer-h800-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                    config=vars(args),
                )

    def _switch_to_phase_3(self):
        """Phase 3: 切换到最大 batch_size + 多 worker 模式"""
        if self._phase >= 3:
            return

        self._phase = 3
        if self.is_main:
            print(f"\n{'='*60}")
            print(f"🚀 切换到 Phase 3 (高速训练模式)")
            print(f"📊 batch_size: {self._compilation_batch_size} → {self._target_batch_size}")
            print(f"👷 num_workers: 0 → {self._target_num_workers}")
            print(f"⚡ 预期加速: {self._target_batch_size / self._compilation_batch_size:.1f}x")
            print(f"{'='*60}\n")

        # 重建 train_loader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self._target_batch_size,  # 切换到目标 batch_size
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
            num_workers=self._target_num_workers,  # 切换到多 worker
            collate_fn=self._collate_fn,
            pin_memory=True,
            prefetch_factor=4 if self._target_num_workers > 0 else None,
            persistent_workers=self._target_num_workers > 0,
        )

        # 重建 val_loader
        if self.val_dataset is not None:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self._target_batch_size,  # 验证集也切换
                shuffle=False,
                sampler=self.val_sampler,
                num_workers=self._target_num_workers,
                collate_fn=self._collate_fn,
                pin_memory=True,
            )

    def train_epoch(self) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0

        start_time = time.time()

        if self.world_size > 1:
            self.train_loader.sampler.set_epoch(self.epoch)

        self.optimizer.zero_grad(set_to_none=True)  # 更高效

        for batch_idx, batch in enumerate(self.train_loader):
            # 跳过空 batch (所有样本都超过 max_bars)
            if batch is None:
                continue

            # 移动数据到设备 (non_blocking)
            token_ids = batch['token_ids'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            chord_ids = batch['chord_ids'].to(self.device, non_blocking=True)
            instrument_ids = batch['instrument_ids'].to(self.device, non_blocking=True)
            token_type_ids = batch['token_type_ids'].to(self.device, non_blocking=True)
            note_ids = batch['note_ids'].to(self.device, non_blocking=True)

            # 序列打包: 获取 doc_ids (如果存在)
            doc_ids = None
            if 'doc_ids' in batch:
                doc_ids = batch['doc_ids'].to(self.device, non_blocking=True)

            # 前向传播 + 反向传播 (BF16/FP16)
            # 注意: FlexAttention 要求 forward 和 backward 都在相同的 autocast 上下文中
            with torch.autocast(device_type='cuda', dtype=self.autocast_dtype):
                logits = self.model(
                    token_ids,
                    chord_ids=chord_ids,
                    instrument_ids=instrument_ids,
                    token_type_ids=token_type_ids,
                    note_ids=note_ids,
                    doc_ids=doc_ids,  # 序列打包支持
                )

                # 计算损失
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                loss = loss / self.args.gradient_accumulation_steps

                # 反向传播 (支持 GradScaler)
                # 必须在 autocast 上下文内，否则 FlexAttention 会报 dtype 错误
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
                    # FP16: 使用 GradScaler
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # BF16/FP32: 直接优化
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                # 三阶段动态切换（优化版）
                # Phase 1 (Step 1-10): batch_size=1, num_workers=0 (编译初期，避免多进程冲突)
                # Phase 2 (Step 11-100): batch_size=1, num_workers=8 (编译中期，加速数据加载)
                # Phase 3 (Step 101+): batch_size=6, num_workers=8 (训练阶段，全速)
                if self.global_step == 10:
                    # 切换到 num_workers=8，但保持 batch_size=1
                    if self.is_main:
                        print(f"\n{'='*60}")
                        print(f"🔄 Phase 2: 切换数据加载")
                        print(f"👷 num_workers: 0 → {self._target_num_workers}")
                        print(f"📊 batch_size: {self._compilation_batch_size} (保持)")
                        print(f"⚡ 预期: GPU 利用率提升，编译继续...")
                        print(f"{'='*60}\n")

                    self._phase = 2

                    # 重建 DataLoader（只改 num_workers）
                    self.train_loader = DataLoader(
                        self.train_dataset,
                        batch_size=self._compilation_batch_size,  # 保持 batch_size=1
                        shuffle=(self.train_sampler is None),
                        sampler=self.train_sampler,
                        num_workers=self._target_num_workers,  # 切换到 8 workers
                        collate_fn=self._collate_fn,
                        pin_memory=True,
                        prefetch_factor=4 if self._target_num_workers > 0 else None,
                        persistent_workers=self._target_num_workers > 0,
                    )

                elif self.global_step == 100:
                    # 切换到最大 batch_size
                    self._switch_to_phase_3()

                # 日志
                if self.is_main and self.global_step % self.args.log_interval == 0:
                    elapsed = time.time() - start_time
                    avg_loss = total_loss / num_batches
                    tokens_per_sec = total_tokens / elapsed
                    lr = self.scheduler.get_last_lr()[0]

                    print(f"Step {self.global_step} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"LR: {lr:.2e} | "
                          f"Tokens/s: {tokens_per_sec:.0f}")

                    if self.args.use_wandb:
                        import wandb
                        wandb.log({
                            'train/loss': avg_loss,
                            'train/learning_rate': lr,
                            'train/tokens_per_sec': tokens_per_sec,
                            'train/step': self.global_step,
                        })

                # 保存检查点
                if self.is_main and self.global_step % self.args.save_interval == 0:
                    self.save_checkpoint(f'step_{self.global_step}')

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
            token_ids = batch['token_ids'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            chord_ids = batch['chord_ids'].to(self.device, non_blocking=True)
            instrument_ids = batch['instrument_ids'].to(self.device, non_blocking=True)
            token_type_ids = batch['token_type_ids'].to(self.device, non_blocking=True)
            note_ids = batch['note_ids'].to(self.device, non_blocking=True)
            doc_ids = batch.get('doc_ids')
            if doc_ids is not None:
                doc_ids = doc_ids.to(self.device, non_blocking=True)

            with torch.autocast(device_type='cuda', dtype=self.autocast_dtype):
                logits = self.model(
                    token_ids,
                    chord_ids=chord_ids,
                    instrument_ids=instrument_ids,
                    token_type_ids=token_type_ids,
                    note_ids=note_ids,
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

        # 聚合分布式结果
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
            print(f"批大小 (per GPU): {self.args.batch_size}")
            print(f"梯度累积: {self.args.gradient_accumulation_steps}")
            print(f"有效批大小: {self.args.batch_size * self.world_size * self.args.gradient_accumulation_steps}")
            print(f"Epochs: {self.args.epochs}")
            print(f"学习率: {self.args.learning_rate}")
            print(f"混合精度: {'BF16' if self.args.bf16 else ('FP16' if self.args.fp16 else 'FP32')}")
            print(f"TF32: {torch.backends.cuda.matmul.allow_tf32}")
            print(f"静态编译: {is_static_compile_mode()}")
            print(f"Flash Attention: {'Yes' if HAS_FLASH_ATTN else 'No (using SDPA)'}")
            print("=" * 60 + "\n")

        for epoch in range(self.args.epochs):
            self.epoch = epoch

            if self.is_main:
                print(f"\n{'='*50}")
                print(f"Epoch {epoch + 1}/{self.args.epochs}")
                print(f"{'='*50}")

            epoch_start = time.time()

            # 训练
            train_metrics = self.train_epoch()

            # 验证
            val_metrics = self.validate()

            epoch_time = time.time() - epoch_start

            if self.is_main:
                print(f"Epoch {epoch + 1} | "
                      f"Train Loss: {train_metrics['loss']:.4f} | "
                      f"Time: {epoch_time:.1f}s", end='')

                if val_metrics:
                    print(f" | Val Loss: {val_metrics['val_loss']:.4f}", end='')

                    if val_metrics['val_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val_loss']
                        self.save_checkpoint('best')
                        print(" (best)", end='')

                print()

                if self.args.use_wandb:
                    import wandb
                    wandb.log({
                        'epoch': epoch + 1,
                        'epoch_time': epoch_time,
                        **{f'train/{k}': v for k, v in train_metrics.items()},
                        **{f'val/{k}': v for k, v in val_metrics.items()},
                    })

            # 保存 epoch 检查点
            if self.is_main:
                self.save_checkpoint(f'epoch_{epoch + 1}')

        if self.is_main:
            print("\n训练完成！")
            self.save_checkpoint('final')

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
    parser.add_argument('--max_bars', type=int, default=2048,
                        help='最大 bar 数量 (默认 2048，覆盖 99.8%% 样本)')
    # 注：已移除 --attention_mode，统一使用 FlexAttention (PyTorch 2.5+)

    # v1.4 高级优化
    parser.add_argument('--num_kv_heads', type=int, default=None,
                        help='GQA KV 头数 (默认: 根据 model_size 自动配置)')
    parser.add_argument('--use_rms_norm', action='store_true', default=True,
                        help='使用 RMSNorm (v1.4 默认开启)')
    parser.add_argument('--no_rms_norm', action='store_true',
                        help='禁用 RMSNorm，使用 LayerNorm')
    parser.add_argument('--use_2d_rope', action='store_true', default=True,
                        help='使用 2D RoPE 分层位置编码 (v1.4 默认开启)')
    parser.add_argument('--no_2d_rope', action='store_true',
                        help='禁用 2D RoPE，使用标准 1D RoPE')

    # 训练
    parser.add_argument('--epochs', type=int, default=50, help='训练 epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='批大小 (per GPU)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='梯度累积步数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='权重衰减')
    parser.add_argument('--warmup_steps', type=int, default=2000, help='预热步数')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪')

    # 序列打包优化 (v1.1.0)
    parser.add_argument('--use_packing', action='store_true',
                        help='启用序列打包 (Sequence Packing)，将多首短曲拼接成目标长度，提升训练效率 1.5-2x')

    # 静态编译优化 (v1.3)
    parser.add_argument('--static_compile', action='store_true',
                        help='启用静态编译模式: 所有输入 Padding 到固定长度，消除 FlexAttention 重编译，提升速度 15x+')
    parser.add_argument('--static_max_seq_len', type=int, default=16384,
                        help='静态编译模式的固定序列长度 (默认 16384)')
    parser.add_argument('--static_max_bars', type=int, default=256,
                        help='静态编译模式的固定 bar 数量 (默认 256)')

    # H800 优化
    parser.add_argument('--bf16', action='store_true', default=True, help='使用 BF16 (推荐)')
    parser.add_argument('--fp16', action='store_true', help='使用 FP16')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True, help='使用梯度检查点')
    parser.add_argument('--compile', action='store_true', default=True, help='使用 torch.compile (默认开启，FlexAttention 稀疏加速必需)')
    parser.add_argument('--no_compile', action='store_true', help='禁用 torch.compile')

    # 输出
    parser.add_argument('--output_dir', type=str, default='./outputs_h800', help='输出目录')
    parser.add_argument('--log_interval', type=int, default=50, help='日志间隔')
    parser.add_argument('--save_interval', type=int, default=2000, help='保存间隔')

    # 其他
    parser.add_argument('--num_workers', type=int, default=16, help='数据加载工作进程数 (H800 推荐 16+)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')

    # WandB
    parser.add_argument('--use_wandb', action='store_true', help='使用 WandB 记录')
    parser.add_argument('--wandb_project', type=str, default='meloformer-h800', help='WandB 项目名')
    parser.add_argument('--run_name', type=str, default=None, help='运行名称')

    return parser.parse_args()


def main():
    args = parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 静态编译模式 (v1.3)
    if args.static_compile:
        set_static_compile_mode(
            enabled=True,
            max_seq_len=args.static_max_seq_len,
            max_sum_len=args.static_max_bars,
        )
        # 静态编译模式下，max_seq_len 必须与 static_max_seq_len 一致
        if args.max_seq_len != args.static_max_seq_len:
            print(f"[静态编译] 将 max_seq_len 从 {args.max_seq_len} 调整为 {args.static_max_seq_len}")
            args.max_seq_len = args.static_max_seq_len
        if args.max_bars != args.static_max_bars:
            print(f"[静态编译] 将 max_bars 从 {args.max_bars} 调整为 {args.static_max_bars}")
            args.max_bars = args.static_max_bars

    # 检测混合精度可用性
    if args.bf16:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            pass  # BF16 可用
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
            max_bars=args.max_bars,
            max_samples=args.max_samples,
        )
    else:
        full_dataset = PreprocessedDataset(
            args.data_dir,
            max_seq_len=args.max_seq_len,
            max_bars=args.max_bars,
            max_samples=args.max_samples,
        )
    print(f"max_bars={args.max_bars} (超过此值的样本将被跳过)")

    # 划分训练/验证集
    total_size = len(full_dataset)
    val_size = int(total_size * args.val_split)
    train_size = total_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")

    # v1.4 参数处理
    use_rms_norm = args.use_rms_norm and not args.no_rms_norm
    use_2d_rope = args.use_2d_rope and not args.no_2d_rope

    # 创建模型 (FlexAttention + GQA + RMSNorm + 2D RoPE)
    print(f"\n创建模型 (size={args.model_size}, max_bars={args.max_bars}, attention=FlexAttention)...")
    print(f"  v1.4 特性: GQA={'auto' if args.num_kv_heads is None else args.num_kv_heads}, "
          f"RMSNorm={use_rms_norm}, 2D-RoPE={use_2d_rope}")
    model = create_model(
        vocab_size=tokenizer.vocab_size,
        model_size=args.model_size,
        max_seq_len=args.max_seq_len,
        max_bars=args.max_bars,
        chord_start_id=tokenizer.chord_start_id,
        chord_end_id=tokenizer.chord_end_id,
        # v1.4 参数
        num_kv_heads=args.num_kv_heads,  # None 表示使用默认 GQA 配置
        use_rms_norm=use_rms_norm,
        use_2d_rope=use_2d_rope,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params / 1e6:.2f}M")

    # 创建训练器
    trainer = H800Trainer(model, train_dataset, val_dataset, args)

    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # 开始训练
    try:
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()
