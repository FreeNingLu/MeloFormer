#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MeloFormer v2.1 FIM (Fill-In-the-Middle) 训练脚本

适配 MeloFormer v2.1 的 Summary Token + FlexAttention 架构:
- Two-phase attention: Phase 1 (Summarize) SDPA + Phase 2 (Updating) FlexAttention
- chord_ids: 每个 token 映射到一个 chord 段 (Summary Token 注意力依赖)
- instrument_ids, token_type_ids, note_ids: 额外元数据
- batch_size=1 约束 (FlexAttention captured tensor mask_mod 是 batch 共享的)

FIM 模式:
- TRACK_MASK (25%): 遮蔽整个乐器轨道，用于自动编曲/声部生成
- TIME_FIM (20%): 遮蔽单轨道的一个时间段，生成该时间段内容
- SKELETON_FIM (25%): 给定骨架（乐器+和弦），生成音符
- PRETRAIN (30%): 保持原样 (Summary Token 目标)，防止灾难性遗忘

关键适配:
- FIM 重排后 chord_ids 必须重新编号为连续序列 (0,1,2,...)
- FIM 特殊 token 的元数据: chord_id=-1, instrument_id=129, token_type_id=-1, note_id=-1
- 所有元数据 (chord_ids, instrument_ids, token_type_ids, note_ids) 与 token_ids 同步重排
"""

import os
import sys
import math
import time
import json
import random
import hashlib
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

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
WARMUP_PHASE_STEPS = 10
DEFAULT_NUM_WORKERS = 16
BATCH_SIZE = 1                # v2.1 constraint: captured tensor mask_mod requires batch_size=1
BUCKET_SIZE = 512
SHARD_CACHE_SIZE = 30
SHARD_PROGRESS_INTERVAL = 500

# Collate 字段
TENSOR_FIELDS = ['token_ids', 'chord_ids', 'instrument_ids', 'token_type_ids', 'note_ids', 'position_ids']

# --- Torch configuration ---
def configure_torch_for_training():
    """Configure torch._dynamo cache and TF32 settings at startup."""
    torch._dynamo.config.cache_size_limit = DYNAMO_CACHE_SIZE
    torch._dynamo.config.accumulated_cache_size_limit = DYNAMO_ACCUM_CACHE_SIZE
    torch._dynamo.config.suppress_errors = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# --- Optional dependencies ---
try:
    from datasets import load_from_disk
    HAS_HF_DATASETS = True
except ImportError:
    HAS_HF_DATASETS = False

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    HAS_LIGER = True
except ImportError:
    HAS_LIGER = False


# ==================== FIM Token 定义 ====================

class FIMTokens:
    """FIM 特殊 Token 定义

    原始 vocab_size = 643 (ID 0-642 已被使用)
    FIM Token 使用 643-650，扩展后 vocab_size = 651
    """
    # Track Mask: 遮蔽整个轨道，生成完整乐器轨道
    TRK_PRE = 643
    TRK_SUF = 644
    TRK_MID = 645

    # Time FIM: 遮蔽单轨道的一个时间段
    TIME_PRE = 646
    TIME_SUF = 647
    TIME_MID = 648

    # Skeleton FIM: 给定骨架（乐器+和弦），生成音符
    SKE     = 649
    SKE_MID = 650

    NUM_SPECIAL = 8

    # FIM 特殊 token 的元数据默认值
    SPECIAL_CHORD_ID = -1
    SPECIAL_INSTRUMENT_ID = 129
    SPECIAL_TOKEN_TYPE_ID = -1
    SPECIAL_NOTE_ID = -1


class FIMMode(Enum):
    """FIM 模式"""
    PRETRAIN = 0
    TRACK_MASK = 1
    SKELETON_FIM = 2
    TIME_FIM = 3


# ==================== 数据集 ====================

class ArrowDataset(Dataset):
    """
    HuggingFace Arrow 格式数据集 - 零拷贝内存映射

    返回: token_ids, chord_ids, instrument_ids, token_type_ids, note_ids, position_ids, length
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

        cache_dir = data_path_obj.parent / (data_path_obj.name + '_merged_cache')
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

        num_chords = int(sample['chord_ids'].max().item()) + 1
        if num_chords > self.max_chords:
            return None
        sample['num_chords'] = num_chords

        if sample['length'] > self.max_seq_len:
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    sample[k] = v[:self.max_seq_len]
            sample['length'] = self.max_seq_len

        return sample


class PreprocessedDataset(Dataset):
    """
    加载预处理数据 - 支持分片格式和单文件格式

    返回: token_ids, chord_ids, instrument_ids, token_type_ids, note_ids, position_ids, length
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
        """分片格式 - 使用懒加载"""
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

        self._shard_cache = {}
        self._cache_order = []
        self._max_cache_size = SHARD_CACHE_SIZE

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
        """单文件格式 (旧版兼容)"""
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
            sample['num_chords'] = num_chords

        if sample['length'] > self.max_seq_len:
            sample = {k: v[:self.max_seq_len] if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
            sample['length'] = self.max_seq_len

        return sample


# ==================== FIM 处理器 ====================

class FIMProcessor:
    """
    FIM 数据处理器 (MeloFormer v2.1 适配版)

    与 MIDILM 版本的关键区别:
    1. 处理完整的元数据: chord_ids, instrument_ids, token_type_ids, note_ids
    2. FIM 重排后必须 renumber chord_ids 为连续序列 (0,1,2,...)
    3. FIM 特殊 token 使用固定元数据值

    轨道 Mask (Vertical): 遮蔽整个乐器轨道
    时间填空 (TIME_FIM): 遮蔽单轨道的时间段
    骨架 FIM (SKELETON_FIM): 根据乐器和弦骨架填写音符
    """

    # Token ID 范围 (HID tokenizer_v2.py)
    TRACK_TOKEN_START = 367  # #D (ID 367)
    TRACK_TOKEN_END = 496    # #P0-#P127 (ID 368-495), 不含 496

    EOS_TOKEN = 2

    CHORD_TOKEN_START = 6
    CHORD_TOKEN_END = 223
    T_TOKEN_START = 223
    T_TOKEN_END = 239
    P_TOKEN_START = 239
    P_TOKEN_END = 367
    L_TOKEN_START = 547
    L_TOKEN_END = 611
    V_TOKEN_START = 611
    V_TOKEN_END = 643

    def __init__(
        self,
        fim_ratio: float = 0.7,
        min_prefix_len: int = 10,
        min_suffix_len: int = 10,
        min_middle_len: int = 20,
        curriculum_learning: bool = False,
        curriculum_start_ratios: Tuple[float, ...] = (0.35, 0.15, 0.35, 0.15),
        curriculum_end_ratios: Tuple[float, ...] = (0.20, 0.30, 0.30, 0.20),
    ):
        """
        Args:
            fim_ratio: FIM 数据比例 (1 - fim_ratio 为 Causal)
            min_prefix_len: 最小前缀长度
            min_suffix_len: 最小后缀长度
            min_middle_len: 最小中间长度
            curriculum_learning: 是否启用课程学习
            curriculum_start_ratios: 课程学习初始比例 (Causal, TrackMask, SkeletonFIM, TimeFIM)
            curriculum_end_ratios: 课程学习结束比例 (Causal, TrackMask, SkeletonFIM, TimeFIM)
        """
        self.fim_ratio = fim_ratio
        self.min_prefix_len = min_prefix_len
        self.min_suffix_len = min_suffix_len
        self.min_middle_len = min_middle_len

        self.pretrain_ratio = 0.30  # PRETRAIN: 保持预训练目标，防止灾难性遗忘
        self.track_mask_ratio = 0.25
        self.time_fim_ratio = 0.20
        self.skeleton_fim_ratio = 0.25

        self.curriculum_learning = curriculum_learning
        self.curriculum_start_ratios = curriculum_start_ratios
        self.curriculum_end_ratios = curriculum_end_ratios
        if curriculum_learning:
            self.set_curriculum_ratios(*curriculum_start_ratios)

    def set_curriculum_ratios(self, pretrain: float, track_mask: float, skeleton_fim: float, time_fim: float):
        """手动设置模式比例"""
        total = pretrain + track_mask + skeleton_fim + time_fim
        self.pretrain_ratio = pretrain / total
        self.track_mask_ratio = track_mask / total
        self.skeleton_fim_ratio = skeleton_fim / total
        self.time_fim_ratio = time_fim / total

    def update_curriculum(self, progress: float):
        """根据训练进度更新模式比例 (线性插值)"""
        if not self.curriculum_learning:
            return
        progress = max(0.0, min(1.0, progress))
        pretrain = self.curriculum_start_ratios[0] + (self.curriculum_end_ratios[0] - self.curriculum_start_ratios[0]) * progress
        track_mask = self.curriculum_start_ratios[1] + (self.curriculum_end_ratios[1] - self.curriculum_start_ratios[1]) * progress
        skeleton_fim = self.curriculum_start_ratios[2] + (self.curriculum_end_ratios[2] - self.curriculum_start_ratios[2]) * progress
        time_fim = self.curriculum_start_ratios[3] + (self.curriculum_end_ratios[3] - self.curriculum_start_ratios[3]) * progress
        self.set_curriculum_ratios(pretrain, track_mask, skeleton_fim, time_fim)

    # --- Helper methods ---

    def _strip_trailing_eos(self, *tensors):
        """剥离末尾 EOS token（及对应的辅助 tensor），确保每个 FIM 序列只有末尾一个 EOS"""
        if len(tensors) == 0:
            return tensors
        first = tensors[0]
        if len(first) > 0 and first[-1].item() == self.EOS_TOKEN:
            return tuple(t[:-1] for t in tensors)
        return tensors

    def _find_track_boundaries(self, token_ids: torch.Tensor) -> List[int]:
        """找到轨道标记 Token 的位置（向量化）"""
        mask = (token_ids >= self.TRACK_TOKEN_START) & (token_ids < self.TRACK_TOKEN_END)
        return mask.nonzero(as_tuple=False).squeeze(-1).tolist()

    def _is_chord_token(self, tid: int) -> bool:
        return self.CHORD_TOKEN_START <= tid < self.CHORD_TOKEN_END

    def _is_note_token(self, tid: int) -> bool:
        return (
            (self.T_TOKEN_START <= tid < self.T_TOKEN_END) or
            (self.P_TOKEN_START <= tid < self.P_TOKEN_END) or
            (self.L_TOKEN_START <= tid < self.L_TOKEN_END) or
            (self.V_TOKEN_START <= tid < self.V_TOKEN_END)
        )

    def _find_chord_positions_in_track(self, track_tokens: torch.Tensor) -> List[int]:
        """返回单轨道内每个和弦 token 的相对位置列表"""
        positions = []
        for i, tid in enumerate(track_tokens.tolist()):
            if self._is_chord_token(tid):
                positions.append(i)
        return positions

    def _extract_skeleton_and_notes_with_meta(
        self, track_tokens: torch.Tensor, track_chord_ids: torch.Tensor,
        track_instrument_ids: torch.Tensor, track_token_type_ids: torch.Tensor,
        track_note_ids: torch.Tensor,
    ) -> Tuple[
        List[int], List[int], List[int], List[int], List[int],
        List[Tuple[int, List[int], List[int], List[int], List[int], List[int]]]
    ]:
        """
        从轨道提取骨架和音符，同时保留元数据

        Returns:
            skeleton_tokens, skeleton_chord_ids, skeleton_instrument_ids,
            skeleton_token_type_ids, skeleton_note_ids,
            chord_notes_with_meta: [(chord_tid, [note_tids], [note_cids], [note_iids],
                                     [note_ttids], [note_nids]), ...]
        """
        skel_tids = []
        skel_cids = []
        skel_iids = []
        skel_ttids = []
        skel_nids = []

        chord_notes_meta = []
        cur_chord_tid = None
        cur_note_tids = []
        cur_note_cids = []
        cur_note_iids = []
        cur_note_ttids = []
        cur_note_nids = []

        tids = track_tokens.tolist()
        cids = track_chord_ids.tolist()
        iids = track_instrument_ids.tolist()
        ttids = track_token_type_ids.tolist()
        nids = track_note_ids.tolist()

        for i, tid in enumerate(tids):
            if self.TRACK_TOKEN_START <= tid < self.TRACK_TOKEN_END:
                skel_tids.append(tid)
                skel_cids.append(cids[i])
                skel_iids.append(iids[i])
                skel_ttids.append(ttids[i])
                skel_nids.append(nids[i])
            elif self._is_chord_token(tid):
                if cur_chord_tid is not None and cur_note_tids:
                    chord_notes_meta.append((
                        cur_chord_tid, cur_note_tids, cur_note_cids,
                        cur_note_iids, cur_note_ttids, cur_note_nids,
                    ))
                cur_chord_tid = tid
                cur_note_tids = []
                cur_note_cids = []
                cur_note_iids = []
                cur_note_ttids = []
                cur_note_nids = []
                skel_tids.append(tid)
                skel_cids.append(cids[i])
                skel_iids.append(iids[i])
                skel_ttids.append(ttids[i])
                skel_nids.append(nids[i])
            elif self._is_note_token(tid):
                cur_note_tids.append(tid)
                cur_note_cids.append(cids[i])
                cur_note_iids.append(iids[i])
                cur_note_ttids.append(ttids[i])
                cur_note_nids.append(nids[i])

        if cur_chord_tid is not None and cur_note_tids:
            chord_notes_meta.append((
                cur_chord_tid, cur_note_tids, cur_note_cids,
                cur_note_iids, cur_note_ttids, cur_note_nids,
            ))

        return skel_tids, skel_cids, skel_iids, skel_ttids, skel_nids, chord_notes_meta

    def _renumber_chord_ids(self, chord_ids: torch.Tensor) -> torch.Tensor:
        """
        Renumber chord_ids to be sequential 0,1,2,...

        FIM 重排后 chord_ids 可能不连续 (如 [0,0,3,3,1,1,2,2] for PRE+SUF+MID)，
        Summary Token 注意力要求 chord_ids 必须是连续的。
        FIM 特殊 token (chord_id=-1) 保持不变。
        """
        id_map = {}
        new_ids = torch.full_like(chord_ids, -1)
        next_id = 0
        for i, cid in enumerate(chord_ids.tolist()):
            if cid < 0:  # FIM special tokens
                continue
            if cid not in id_map:
                id_map[cid] = next_id
                next_id += 1
            new_ids[i] = id_map[cid]
        return new_ids

    def _make_special_meta(self, n: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """创建 n 个 FIM 特殊 token 的元数据"""
        return (
            torch.full((n,), FIMTokens.SPECIAL_CHORD_ID, dtype=torch.long),
            torch.full((n,), FIMTokens.SPECIAL_INSTRUMENT_ID, dtype=torch.long),
            torch.full((n,), FIMTokens.SPECIAL_TOKEN_TYPE_ID, dtype=torch.long),
            torch.full((n,), FIMTokens.SPECIAL_NOTE_ID, dtype=torch.long),
        )

    # --- Main process entry ---

    def process(self, sample: Dict) -> Tuple[Dict, FIMMode]:
        """
        处理单个样本，应用 FIM 转换

        Args:
            sample: {'token_ids': Tensor, 'chord_ids': Tensor, 'instrument_ids': Tensor,
                     'token_type_ids': Tensor, 'note_ids': Tensor, 'length': int}

        Returns:
            new_sample: 同 keys + 'position_ids' + 'labels'
            mode: FIMMode
        """
        r = random.random()
        if r < self.pretrain_ratio:
            return self._process_pretrain(sample)
        elif r < self.pretrain_ratio + self.track_mask_ratio:
            return self._process_track_mask(sample)
        elif r < self.pretrain_ratio + self.track_mask_ratio + self.time_fim_ratio:
            return self._process_time_fim(sample)
        else:
            return self._process_skeleton_fim(sample)

    # --- PRETRAIN (保持预训练目标，防止灾难性遗忘) ---

    def _process_pretrain(self, sample: Dict) -> Tuple[Dict, FIMMode]:
        """Causal 模式: 保持原样"""
        token_ids = sample['token_ids']
        seq_len = len(token_ids)

        new_sample = {
            'token_ids': token_ids,
            'chord_ids': sample['chord_ids'],
            'instrument_ids': sample['instrument_ids'],
            'token_type_ids': sample['token_type_ids'],
            'note_ids': sample['note_ids'],
            'position_ids': torch.arange(seq_len, dtype=torch.long),
            'labels': token_ids.clone(),
            'length': seq_len,
        }
        return new_sample, FIMMode.PRETRAIN

    # --- TRACK_MASK ---

    def _process_track_mask(self, sample: Dict) -> Tuple[Dict, FIMMode]:
        """
        轨道 Mask 模式 (Vertical) - 遮蔽整个轨道

        chord_ids: 保持原始 chord 结构，重排后 renumber
        instrument_ids, token_type_ids, note_ids: 与 token_ids 同步重排
        FIM 特殊 token: chord_id=-1, instrument_id=129, token_type_id=-1, note_id=-1
        """
        token_ids = sample['token_ids']
        seq_len = len(token_ids)
        boundaries = self._find_track_boundaries(token_ids)

        if len(boundaries) < 2:
            return self._process_pretrain(sample)

        track_idx = random.randint(0, len(boundaries) - 1)
        middle_start = boundaries[track_idx]
        middle_end = boundaries[track_idx + 1] if track_idx + 1 < len(boundaries) else seq_len

        prefix_end = middle_start
        prefix_len = prefix_end
        middle_len = middle_end - middle_start

        if prefix_len < self.min_prefix_len or middle_len < self.min_middle_len:
            return self._process_pretrain(sample)

        return self._build_track_fim_sequence(sample, prefix_end, middle_start, middle_end)

    def _build_track_fim_sequence(
        self, sample: Dict, prefix_end: int, middle_start: int, middle_end: int,
    ) -> Tuple[Dict, FIMMode]:
        """
        构建 Track Mask FIM 序列

        物理重排: <TRK_PRE> prefix <TRK_SUF> suffix <TRK_MID> middle <EOS>
        所有元数据同步重排，chord_ids 重新编号
        """
        token_ids = sample['token_ids']
        chord_ids = sample['chord_ids']
        instrument_ids = sample['instrument_ids']
        token_type_ids = sample['token_type_ids']
        note_ids = sample['note_ids']
        seq_len = len(token_ids)

        prefix_t = token_ids[:prefix_end]
        middle_t = token_ids[middle_start:middle_end]
        suffix_t = token_ids[middle_end:]

        prefix_c = chord_ids[:prefix_end]
        middle_c = chord_ids[middle_start:middle_end]
        suffix_c = chord_ids[middle_end:]

        prefix_i = instrument_ids[:prefix_end]
        middle_i = instrument_ids[middle_start:middle_end]
        suffix_i = instrument_ids[middle_end:]

        prefix_tt = token_type_ids[:prefix_end]
        middle_tt = token_type_ids[middle_start:middle_end]
        suffix_tt = token_type_ids[middle_end:]

        prefix_n = note_ids[:prefix_end]
        middle_n = note_ids[middle_start:middle_end]
        suffix_n = note_ids[middle_end:]

        # 剥离 suffix 末尾的原始 EOS（FIM 序列末尾会统一添加一个 EOS）
        suffix_t, suffix_c, suffix_i, suffix_tt, suffix_n = self._strip_trailing_eos(
            suffix_t, suffix_c, suffix_i, suffix_tt, suffix_n)

        len_pre = len(prefix_t)
        len_mid = len(middle_t)
        len_suf = len(suffix_t)

        sp1_c, sp1_i, sp1_tt, sp1_n = self._make_special_meta(1)
        eos_c, eos_i, eos_tt, eos_n = self._make_special_meta(1)

        # token_ids: <TRK_PRE> prefix <TRK_SUF> suffix <TRK_MID> middle <EOS>
        new_token_ids = torch.cat([
            torch.tensor([FIMTokens.TRK_PRE], dtype=torch.long),
            prefix_t,
            torch.tensor([FIMTokens.TRK_SUF], dtype=torch.long),
            suffix_t,
            torch.tensor([FIMTokens.TRK_MID], dtype=torch.long),
            middle_t,
            torch.tensor([self.EOS_TOKEN], dtype=torch.long),
        ])

        new_chord_ids = torch.cat([sp1_c, prefix_c, sp1_c.clone(), suffix_c, sp1_c.clone(), middle_c, eos_c])
        new_instrument_ids = torch.cat([sp1_i, prefix_i, sp1_i.clone(), suffix_i, sp1_i.clone(), middle_i, eos_i])
        new_token_type_ids = torch.cat([sp1_tt, prefix_tt, sp1_tt.clone(), suffix_tt, sp1_tt.clone(), middle_tt, eos_tt])
        new_note_ids = torch.cat([sp1_n, prefix_n, sp1_n.clone(), suffix_n, sp1_n.clone(), middle_n, eos_n])

        # Renumber chord_ids
        new_chord_ids = self._renumber_chord_ids(new_chord_ids)

        # Position IDs: PRE -> SUF -> MID -> EOS (continuous within each segment)
        pre_positions = list(range(0, len_pre + 1))
        suf_positions = list(range(len_pre + 1, len_pre + len_suf + 2))
        mid_start_pos = len_pre + len_suf + 2
        mid_positions = list(range(mid_start_pos, mid_start_pos + len_mid + 1))
        eos_position = [mid_start_pos + len_mid + 1]
        position_ids = torch.tensor(pre_positions + suf_positions + mid_positions + eos_position, dtype=torch.long)

        # Labels: 只有 middle + EOS 部分有效
        labels = torch.full_like(new_token_ids, -100)
        mid_start_in_new = 1 + len_pre + 1 + len_suf + 1  # <PRE> + prefix + <SUF> + suffix + <MID>
        mid_end_in_new = mid_start_in_new + len_mid
        labels[mid_start_in_new:mid_end_in_new] = middle_t
        labels[mid_end_in_new] = self.EOS_TOKEN

        new_sample = {
            'token_ids': new_token_ids,
            'chord_ids': new_chord_ids,
            'instrument_ids': new_instrument_ids,
            'token_type_ids': new_token_type_ids,
            'note_ids': new_note_ids,
            'position_ids': position_ids,
            'labels': labels,
            'length': len(new_token_ids),
        }
        return new_sample, FIMMode.TRACK_MASK

    # --- TIME_FIM ---

    def _process_time_fim(self, sample: Dict) -> Tuple[Dict, FIMMode]:
        """
        时间填空模式 (TIME_FIM) - 单轨道时间段填空

        随机选一个轨道，遮蔽其中一个和弦区间（时间段），其他轨道保持完整。

        序列结构:
          prefix = [header] [Track A 完整] [#B Cmaj...notes...]
          middle = [Gmaj...notes...]
          suffix = [Amin...notes...] [Track C 完整]
          FIM: <TIME_PRE> prefix <TIME_SUF> #B suffix <TIME_MID> #B middle <EOS>
        """
        token_ids = sample['token_ids']
        seq_len = len(token_ids)
        boundaries = self._find_track_boundaries(token_ids)

        if len(boundaries) < 1:
            return self._process_track_mask(sample)

        track_idx = random.randint(0, len(boundaries) - 1)
        track_start = boundaries[track_idx]
        track_end = boundaries[track_idx + 1] if track_idx + 1 < len(boundaries) else seq_len

        inst_token = token_ids[track_start].item()

        track_tokens = token_ids[track_start:track_end]
        chord_pos = self._find_chord_positions_in_track(track_tokens)

        if len(chord_pos) < 3:
            return self._process_track_mask(sample)

        num_chords = len(chord_pos)
        chord_start = random.randint(1, num_chords - 2)
        max_span = max(1, num_chords // 3)
        chord_end_idx = random.randint(chord_start + 1, min(chord_start + max_span, num_chords - 1))

        middle_start_rel = chord_pos[chord_start]
        middle_end_rel = chord_pos[chord_end_idx]

        prefix_end = track_start + middle_start_rel
        middle_start = track_start + middle_start_rel
        middle_end = track_start + middle_end_rel

        if prefix_end < self.min_prefix_len or (middle_end - middle_start) < self.min_middle_len:
            return self._process_track_mask(sample)

        return self._build_time_fim_sequence(sample, prefix_end, middle_start, middle_end, inst_token)

    def _build_time_fim_sequence(
        self, sample: Dict, prefix_end: int, middle_start: int, middle_end: int, inst_token: int,
    ) -> Tuple[Dict, FIMMode]:
        """
        构建时间填空 FIM 序列

        物理重排: <TIME_PRE> prefix <TIME_SUF> #inst suffix <TIME_MID> #inst middle <EOS>
        Labels: middle + EOS 有效（#inst 不参与 loss）
        """
        token_ids = sample['token_ids']
        chord_ids = sample['chord_ids']
        instrument_ids = sample['instrument_ids']
        token_type_ids = sample['token_type_ids']
        note_ids = sample['note_ids']
        seq_len = len(token_ids)

        prefix_t = token_ids[:prefix_end]
        middle_t = token_ids[middle_start:middle_end]
        suffix_t = token_ids[middle_end:]

        prefix_c = chord_ids[:prefix_end]
        middle_c = chord_ids[middle_start:middle_end]
        suffix_c = chord_ids[middle_end:]

        prefix_i = instrument_ids[:prefix_end]
        middle_i = instrument_ids[middle_start:middle_end]
        suffix_i = instrument_ids[middle_end:]

        prefix_tt = token_type_ids[:prefix_end]
        middle_tt = token_type_ids[middle_start:middle_end]
        suffix_tt = token_type_ids[middle_end:]

        prefix_n = note_ids[:prefix_end]
        middle_n = note_ids[middle_start:middle_end]
        suffix_n = note_ids[middle_end:]

        # 剥离 suffix 末尾的原始 EOS（FIM 序列末尾会统一添加一个 EOS）
        suffix_t, suffix_c, suffix_i, suffix_tt, suffix_n = self._strip_trailing_eos(
            suffix_t, suffix_c, suffix_i, suffix_tt, suffix_n)

        len_pre = len(prefix_t)
        len_mid = len(middle_t)
        len_suf = len(suffix_t)

        # inst token 元数据: 使用 FIM special defaults
        inst_tid_tensor = torch.tensor([inst_token], dtype=torch.long)
        inst_cid = torch.tensor([FIMTokens.SPECIAL_CHORD_ID], dtype=torch.long)
        inst_iid = torch.tensor([FIMTokens.SPECIAL_INSTRUMENT_ID], dtype=torch.long)
        inst_ttid = torch.tensor([FIMTokens.SPECIAL_TOKEN_TYPE_ID], dtype=torch.long)
        inst_nid = torch.tensor([FIMTokens.SPECIAL_NOTE_ID], dtype=torch.long)

        sp1_c, sp1_i, sp1_tt, sp1_n = self._make_special_meta(1)
        eos_c, eos_i, eos_tt, eos_n = self._make_special_meta(1)

        # token_ids: <TIME_PRE> prefix <TIME_SUF> #inst suffix <TIME_MID> #inst middle <EOS>
        new_token_ids = torch.cat([
            torch.tensor([FIMTokens.TIME_PRE], dtype=torch.long),
            prefix_t,
            torch.tensor([FIMTokens.TIME_SUF], dtype=torch.long),
            inst_tid_tensor,
            suffix_t,
            torch.tensor([FIMTokens.TIME_MID], dtype=torch.long),
            inst_tid_tensor.clone(),
            middle_t,
            torch.tensor([self.EOS_TOKEN], dtype=torch.long),
        ])

        new_chord_ids = torch.cat([
            sp1_c, prefix_c,
            sp1_c.clone(), inst_cid, suffix_c,
            sp1_c.clone(), inst_cid.clone(), middle_c,
            eos_c,
        ])
        new_instrument_ids = torch.cat([
            sp1_i, prefix_i,
            sp1_i.clone(), inst_iid, suffix_i,
            sp1_i.clone(), inst_iid.clone(), middle_i,
            eos_i,
        ])
        new_token_type_ids = torch.cat([
            sp1_tt, prefix_tt,
            sp1_tt.clone(), inst_ttid, suffix_tt,
            sp1_tt.clone(), inst_ttid.clone(), middle_tt,
            eos_tt,
        ])
        new_note_ids = torch.cat([
            sp1_n, prefix_n,
            sp1_n.clone(), inst_nid, suffix_n,
            sp1_n.clone(), inst_nid.clone(), middle_n,
            eos_n,
        ])

        # Renumber chord_ids
        new_chord_ids = self._renumber_chord_ids(new_chord_ids)

        # Position IDs
        suf_total = 1 + len_suf   # #inst(1) + suffix
        mid_total = 1 + len_mid   # #inst(1) + middle

        pre_positions = list(range(0, len_pre + 1))
        suf_positions = list(range(len_pre + 1, len_pre + 1 + suf_total + 1))
        mid_start_pos = len_pre + suf_total + 2
        mid_positions = list(range(mid_start_pos, mid_start_pos + mid_total + 1))
        eos_position = [mid_start_pos + mid_total + 1]
        position_ids = torch.tensor(pre_positions + suf_positions + mid_positions + eos_position, dtype=torch.long)

        # Labels: 只有 middle + EOS 有效 (#inst 不参与 loss)
        labels = torch.full_like(new_token_ids, -100)
        mid_content_start = 1 + len_pre + 1 + 1 + len_suf + 1 + 1  # <TIME_PRE> prefix <TIME_SUF> #inst suffix <TIME_MID> #inst
        labels[mid_content_start:mid_content_start + len_mid] = middle_t
        labels[mid_content_start + len_mid] = self.EOS_TOKEN

        new_sample = {
            'token_ids': new_token_ids,
            'chord_ids': new_chord_ids,
            'instrument_ids': new_instrument_ids,
            'token_type_ids': new_token_type_ids,
            'note_ids': new_note_ids,
            'position_ids': position_ids,
            'labels': labels,
            'length': len(new_token_ids),
        }
        return new_sample, FIMMode.TIME_FIM

    # --- SKELETON_FIM ---

    def _process_skeleton_fim(self, sample: Dict) -> Tuple[Dict, FIMMode]:
        """
        骨架 FIM 模式 (SKELETON_FIM)

        从一个轨道提取骨架（乐器+和弦序列），预测音符部分。

        序列结构:
            <SKE> [其他轨道] [骨架] [后续轨道] <SKE_MID> [chord1+notes1 chord2+notes2 ...] <EOS>
        """
        token_ids = sample['token_ids']
        seq_len = len(token_ids)
        boundaries = self._find_track_boundaries(token_ids)

        if len(boundaries) < 1:
            return self._process_pretrain(sample)

        track_idx = random.randint(0, len(boundaries) - 1)
        track_start = boundaries[track_idx]
        track_end = boundaries[track_idx + 1] if track_idx + 1 < len(boundaries) else seq_len

        track_tokens = token_ids[track_start:track_end]
        track_chord_ids = sample['chord_ids'][track_start:track_end]
        track_instrument_ids = sample['instrument_ids'][track_start:track_end]
        track_token_type_ids = sample['token_type_ids'][track_start:track_end]
        track_note_ids = sample['note_ids'][track_start:track_end]

        skel_tids, skel_cids, skel_iids, skel_ttids, skel_nids, chord_notes_meta = \
            self._extract_skeleton_and_notes_with_meta(
                track_tokens, track_chord_ids, track_instrument_ids,
                track_token_type_ids, track_note_ids,
            )

        if len(chord_notes_meta) == 0 or len(skel_tids) < 2:
            return self._process_track_mask(sample)

        return self._build_skeleton_fim_sequence(sample, track_start, track_end,
                                                  skel_tids, skel_cids, skel_iids,
                                                  skel_ttids, skel_nids, chord_notes_meta)

    def _build_skeleton_fim_sequence(
        self, sample: Dict, track_start: int, track_end: int,
        skel_tids: List[int], skel_cids: List[int], skel_iids: List[int],
        skel_ttids: List[int], skel_nids: List[int],
        chord_notes_meta: List[Tuple],
    ) -> Tuple[Dict, FIMMode]:
        """
        构建骨架 FIM 序列

        物理序列: <SKE> prefix skeleton suffix <SKE_MID> [chord1 notes1 ...] <EOS>
        Position IDs: 直接递增 (条件 -> 生成)
        Labels: SKE_MID 部分 + EOS 有效
        """
        token_ids = sample['token_ids']
        chord_ids = sample['chord_ids']
        instrument_ids = sample['instrument_ids']
        token_type_ids = sample['token_type_ids']
        note_ids = sample['note_ids']

        prefix_t = token_ids[:track_start]
        suffix_t = token_ids[track_end:]

        prefix_c = chord_ids[:track_start]
        suffix_c = chord_ids[track_end:]

        prefix_i = instrument_ids[:track_start]
        suffix_i = instrument_ids[track_end:]

        prefix_tt = token_type_ids[:track_start]
        suffix_tt = token_type_ids[track_end:]

        prefix_n = note_ids[:track_start]
        suffix_n = note_ids[track_end:]

        # 剥离 suffix 末尾的原始 EOS（FIM 序列末尾会统一添加一个 EOS）
        suffix_t, suffix_c, suffix_i, suffix_tt, suffix_n = self._strip_trailing_eos(
            suffix_t, suffix_c, suffix_i, suffix_tt, suffix_n)

        len_pre = len(prefix_t)
        len_skeleton = len(skel_tids)
        len_suf = len(suffix_t)

        # MID content: [chord1, notes1..., chord2, notes2..., ...]
        mid_tids = []
        mid_cids = []
        mid_iids = []
        mid_ttids = []
        mid_nids = []
        for entry in chord_notes_meta:
            chord_tid, note_tids, note_cids, note_iids, note_ttids_e, note_nids_e = entry
            # 和弦 token: 使用其下属音符的 chord_id (共享)，token_type=-1
            if note_cids:
                mid_tids.append(chord_tid)
                mid_cids.append(note_cids[0])
                mid_iids.append(note_iids[0])
                mid_ttids.append(-1)
                mid_nids.append(-1)
            mid_tids.extend(note_tids)
            mid_cids.extend(note_cids)
            mid_iids.extend(note_iids)
            mid_ttids.extend(note_ttids_e)
            mid_nids.extend(note_nids_e)

        len_mid = len(mid_tids)

        sp1_c, sp1_i, sp1_tt, sp1_n = self._make_special_meta(1)
        eos_c, eos_i, eos_tt, eos_n = self._make_special_meta(1)

        # token_ids: <SKE> prefix skeleton suffix <SKE_MID> mid_content <EOS>
        new_token_ids = torch.cat([
            torch.tensor([FIMTokens.SKE], dtype=torch.long),
            prefix_t,
            torch.tensor(skel_tids, dtype=torch.long),
            suffix_t,
            torch.tensor([FIMTokens.SKE_MID], dtype=torch.long),
            torch.tensor(mid_tids, dtype=torch.long),
            torch.tensor([self.EOS_TOKEN], dtype=torch.long),
        ])

        new_chord_ids = torch.cat([
            sp1_c, prefix_c,
            torch.tensor(skel_cids, dtype=torch.long),
            suffix_c,
            sp1_c.clone(),
            torch.tensor(mid_cids, dtype=torch.long),
            eos_c,
        ])
        new_instrument_ids = torch.cat([
            sp1_i, prefix_i,
            torch.tensor(skel_iids, dtype=torch.long),
            suffix_i,
            sp1_i.clone(),
            torch.tensor(mid_iids, dtype=torch.long),
            eos_i,
        ])
        new_token_type_ids = torch.cat([
            sp1_tt, prefix_tt,
            torch.tensor(skel_ttids, dtype=torch.long),
            suffix_tt,
            sp1_tt.clone(),
            torch.tensor(mid_ttids, dtype=torch.long),
            eos_tt,
        ])
        new_note_ids = torch.cat([
            sp1_n, prefix_n,
            torch.tensor(skel_nids, dtype=torch.long),
            suffix_n,
            sp1_n.clone(),
            torch.tensor(mid_nids, dtype=torch.long),
            eos_n,
        ])

        # Renumber chord_ids
        new_chord_ids = self._renumber_chord_ids(new_chord_ids)

        # Position IDs: 直接递增
        total_len = len(new_token_ids)
        position_ids = torch.arange(total_len, dtype=torch.long)

        # Labels: SKE_MID 部分 + EOS 有效
        labels = torch.full_like(new_token_ids, -100)
        mid_start_in_new = 1 + len_pre + len_skeleton + len_suf + 1  # <SKE> prefix skeleton suffix <SKE_MID>
        for i in range(len_mid):
            labels[mid_start_in_new + i] = mid_tids[i]
        labels[mid_start_in_new + len_mid] = self.EOS_TOKEN

        new_sample = {
            'token_ids': new_token_ids,
            'chord_ids': new_chord_ids,
            'instrument_ids': new_instrument_ids,
            'token_type_ids': new_token_type_ids,
            'note_ids': new_note_ids,
            'position_ids': position_ids,
            'labels': labels,
            'length': total_len,
        }
        return new_sample, FIMMode.SKELETON_FIM


# ==================== FIM Collator ====================

class FIMCollator:
    """FIM 批次整理器 - MeloFormer v2.1 版本 (batch_size=1)"""

    LENGTH_BUCKETS = [512, 1024, 2048, 4096, 8192, 16384, 24576]

    def __init__(
        self,
        fim_processor: FIMProcessor,
        max_seq_len: int = 24576,
    ):
        self.fim_processor = fim_processor
        self.max_seq_len = max_seq_len

    def _get_bucket_length(self, length: int) -> int:
        for bucket in self.LENGTH_BUCKETS:
            if length <= bucket:
                return min(bucket, self.max_seq_len)
        return self.max_seq_len

    def _create_dummy_batch(self) -> Dict[str, torch.Tensor]:
        """Dummy batch for DDP sync (all labels = -100)"""
        dummy_len = self.LENGTH_BUCKETS[0]
        return {
            'token_ids': torch.zeros(1, dummy_len, dtype=torch.long),
            'chord_ids': torch.full((1, dummy_len), -1, dtype=torch.long),
            'instrument_ids': torch.full((1, dummy_len), 129, dtype=torch.long),
            'token_type_ids': torch.full((1, dummy_len), -1, dtype=torch.long),
            'note_ids': torch.full((1, dummy_len), -1, dtype=torch.long),
            'position_ids': torch.arange(dummy_len, dtype=torch.long).unsqueeze(0),
            'labels': torch.full((1, dummy_len), -100, dtype=torch.long),
            'lengths': torch.tensor([dummy_len], dtype=torch.long),
            'modes': torch.zeros(1, dtype=torch.long),
        }

    def __call__(self, batch: List[Optional[Dict]]) -> Dict[str, torch.Tensor]:
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return self._create_dummy_batch()

        processed = []
        for item in batch:
            if item.get('length', 0) < 50:
                continue

            new_sample, mode = self.fim_processor.process(item)

            # Truncate to max_seq_len
            length = min(new_sample['length'], self.max_seq_len)
            for k, v in new_sample.items():
                if isinstance(v, torch.Tensor) and v.dim() == 1:
                    new_sample[k] = v[:length]
            new_sample['length'] = length

            # Clamp position_ids
            new_sample['position_ids'] = torch.clamp(new_sample['position_ids'], max=self.max_seq_len - 1)

            processed.append((new_sample, mode.value))

        if len(processed) == 0:
            return self._create_dummy_batch()

        # Sort by length descending
        processed = sorted(processed, key=lambda x: x[0]['length'], reverse=True)
        max_len = self._get_bucket_length(processed[0][0]['length'])
        batch_size = len(processed)

        token_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        chord_ids = torch.full((batch_size, max_len), -1, dtype=torch.long)
        instrument_ids = torch.full((batch_size, max_len), 129, dtype=torch.long)
        token_type_ids = torch.full((batch_size, max_len), -1, dtype=torch.long)
        note_ids = torch.full((batch_size, max_len), -1, dtype=torch.long)
        position_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        lengths = torch.zeros(batch_size, dtype=torch.long)
        modes = torch.zeros(batch_size, dtype=torch.long)

        max_num_chords = 0
        for i, (sample, mode_val) in enumerate(processed):
            length = sample['length']
            token_ids[i, :length] = sample['token_ids'][:length]
            chord_ids[i, :length] = sample['chord_ids'][:length]
            instrument_ids[i, :length] = sample['instrument_ids'][:length]
            token_type_ids[i, :length] = sample['token_type_ids'][:length]
            note_ids[i, :length] = sample['note_ids'][:length]
            position_ids[i, :length] = sample['position_ids'][:length]
            labels[i, :length] = sample['labels'][:length]
            lengths[i] = length
            modes[i] = mode_val
            # chord_ids 在 FIMProcessor 中已重编号为连续序列，直接取 max+1
            nc = int(sample['chord_ids'][:length].max().item()) + 1 if length > 0 else 1
            max_num_chords = max(max_num_chords, nc)

        return {
            'token_ids': token_ids,
            'chord_ids': chord_ids,
            'instrument_ids': instrument_ids,
            'token_type_ids': token_type_ids,
            'note_ids': note_ids,
            'position_ids': position_ids,
            'labels': labels,
            'lengths': lengths,
            'modes': modes,
            'num_chords': max_num_chords,
        }


# ==================== 训练工具 ====================

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl', init_method='env://',
            timeout=timedelta(minutes=30),
        )

    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_wsd_schedule(optimizer, num_warmup_steps, num_stable_steps, num_decay_steps, min_lr_ratio=0.01):
    """WSD (Warmup-Stable-Decay) 学习率调度器"""
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        if step < num_warmup_steps + num_stable_steps:
            return 1.0
        decay_step = step - num_warmup_steps - num_stable_steps
        decay_progress = min(float(decay_step) / float(max(1, num_decay_steps)), 1.0)
        return max(min_lr_ratio, min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * decay_progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ==================== FIM 训练器 ====================

class FIMTrainer:
    """MeloFormer v2.1 FIM 训练器"""

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        args: argparse.Namespace = None,
        rank: int = 0,
        world_size: int = 1,
        local_rank: int = 0,
        device: torch.device = None,
    ):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_main = rank == 0
        self.device = device or torch.device(f'cuda:{local_rank}')
        self.model = model

        # DDP
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=False)

        # torch.compile: 同 train.py，不在模型层面使用 torch.compile(model)
        # FlexAttention 内部已独立编译，双层 compile 导致持续 graph break + 重编译
        if self.is_main:
            print("[FIM] FlexAttention kernel-level compile active")

        # Mode weights
        self.mode_weights = [float(x) for x in args.mode_weights.split(',')]
        assert len(self.mode_weights) == 4

        # Curriculum
        curriculum_start = tuple(float(x) for x in args.curriculum_start_ratios.split(','))
        curriculum_end = tuple(float(x) for x in args.curriculum_end_ratios.split(','))

        # FIM Processor
        self.fim_processor = FIMProcessor(
            fim_ratio=args.fim_ratio,
            curriculum_learning=args.curriculum_learning,
            curriculum_start_ratios=curriculum_start,
            curriculum_end_ratios=curriculum_end,
        )

        # DataLoader
        collate = FIMCollator(fim_processor=self.fim_processor, max_seq_len=args.max_seq_len)
        train_sampler = DistributedSampler(train_dataset, shuffle=True) if self.world_size > 1 else None
        self.train_sampler = train_sampler
        self.train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE,
            shuffle=(train_sampler is None), sampler=train_sampler,
            num_workers=args.num_workers, collate_fn=collate,
            pin_memory=True,
            prefetch_factor=4 if args.num_workers > 0 else None,
            persistent_workers=args.num_workers > 0,
        )

        if val_dataset is not None:
            val_sampler = DistributedSampler(val_dataset, shuffle=False) if self.world_size > 1 else None
            self.val_loader = DataLoader(
                val_dataset, batch_size=BATCH_SIZE,
                shuffle=False, sampler=val_sampler,
                num_workers=args.num_workers, collate_fn=collate, pin_memory=True,
            )
        else:
            self.val_loader = None

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=args.learning_rate,
            weight_decay=args.weight_decay, betas=(0.9, 0.95), eps=1e-6,
            fused=torch.cuda.is_available(),
        )

        # Scheduler
        total_steps = len(self.train_loader) * args.epochs // args.gradient_accumulation_steps
        self.total_steps = total_steps

        if args.scheduler == 'wsd':
            warmup = args.warmup_steps
            stable = int(total_steps * args.wsd_stable_ratio)
            decay = total_steps - warmup - stable
            self.scheduler = get_wsd_schedule(self.optimizer, warmup, stable, decay)
            if self.is_main:
                print(f"[FIM] WSD: warmup={warmup}, stable={stable}, decay={decay}")
        else:
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, args.warmup_steps, total_steps)

        # Mixed precision
        if args.bf16:
            self.autocast_dtype = torch.bfloat16
        elif args.fp16:
            self.autocast_dtype = torch.float16
        else:
            self.autocast_dtype = torch.float32

        # GC + autocast dtype propagation
        if args.gradient_checkpointing:
            base_model = self.model.module if hasattr(self.model, 'module') else self.model
            base_model._autocast_dtype = self.autocast_dtype

        # State
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self._step_log_rows: List[Dict] = []
        self._epoch_log_rows: List[Dict] = []

        self.checkpoint_dir = Path(args.output_dir) / 'checkpoints'
        self.log_dir = Path(args.output_dir) / 'logs'
        if self.is_main:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)

            run_ts = datetime.now().strftime('%Y%m%d-%H%M%S')
            log_file = self.log_dir / f'train_fim_{run_ts}.log'
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setFormatter(logging.Formatter('%(asctime)s %(message)s', datefmt='%H:%M:%S'))
            self._file_logger = logging.getLogger('meloformer_fim')
            self._file_logger.setLevel(logging.INFO)
            self._file_logger.addHandler(fh)
            self._log_file = log_file
            print(f'[+] 训练日志: {log_file}')

            if args.use_wandb:
                import wandb
                wandb.init(
                    project=args.wandb_project,
                    name=args.run_name or f"meloformer-fim-{run_ts}",
                    config=vars(args),
                )

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        start_time = time.time()
        step_start = time.time()

        mode_losses = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        mode_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        mode_names = {0: 'Causal', 1: 'TrackMask', 2: 'SkeletonFIM', 3: 'TimeFIM'}

        start_time = time.time()

        if self.world_size > 1:
            self.train_loader.sampler.set_epoch(self.epoch)

        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {self.epoch+1} [FIM]',
            disable=not self.is_main,
            dynamic_ncols=True,
            leave=False,
        )

        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue

            token_ids = batch['token_ids'].to(self.device, non_blocking=True)
            chord_ids = batch['chord_ids'].to(self.device, non_blocking=True)
            instrument_ids = batch['instrument_ids'].to(self.device, non_blocking=True)
            token_type_ids = batch['token_type_ids'].to(self.device, non_blocking=True)
            note_ids = batch['note_ids'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            modes = batch['modes'].to(self.device, non_blocking=True)

            batch_tokens = (labels != -100).sum().item()
            is_dummy = (batch_tokens == 0)

            num_chords = batch.get('num_chords', None)
            doc_ids = batch.get('doc_ids', None)
            if doc_ids is not None:
                doc_ids = doc_ids.to(self.device, non_blocking=True)

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

                # Per-sample weighted loss
                per_token_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction='none',
                ).view(shift_labels.shape)

                bs = shift_labels.size(0)
                loss_parts = []
                for i in range(bs):
                    mask = shift_labels[i] != -100
                    n_valid = mask.sum()
                    if n_valid == 0:
                        continue
                    sample_loss = per_token_loss[i][mask].sum() / n_valid
                    mode_idx = modes[i].item()
                    loss_parts.append(sample_loss * self.mode_weights[mode_idx])

                    with torch.no_grad():
                        mode_losses[mode_idx] += sample_loss.item()
                        mode_counts[mode_idx] += 1

                if loss_parts:
                    loss = torch.stack(loss_parts).mean()
                else:
                    loss = per_token_loss.sum() * 0

                loss = loss / self.args.gradient_accumulation_steps

            # NaN detection
            if torch.isnan(loss) or torch.isinf(loss):
                self.optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()

            if not is_dummy:
                total_loss += loss.item() * self.args.gradient_accumulation_steps
                total_tokens += batch_tokens
                num_batches += 1

            if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                # Curriculum
                if self.args.curriculum_learning:
                    progress = self.global_step / max(self.total_steps, 1)
                    self.fim_processor.update_curriculum(progress)

                # Log
                if self.is_main and self.global_step % self.args.log_interval == 0:
                    now = time.time()
                    elapsed = now - start_time
                    ms_per_step = (now - step_start) * 1000
                    step_start = now
                    avg_loss = total_loss / max(num_batches, 1)
                    tokens_per_sec = total_tokens / max(elapsed, 1)
                    lr = self.scheduler.get_last_lr()[0]
                    mem_gb = torch.cuda.max_memory_allocated(self.device) / 1024**3

                    # per-mode loss strings for tqdm
                    mode_strs = []
                    for m in [0, 1, 2, 3]:
                        if mode_counts[m] > 0:
                            mode_strs.append(f"{mode_names[m][:4]}={mode_losses[m]/mode_counts[m]:.4f}")
                    mode_line = ' '.join(mode_strs)

                    log_line = (
                        f"[FIM] step={self.global_step:>6}  epoch={self.epoch+1}  "
                        f"loss={avg_loss:.4f}  lr={lr:.2e}  "
                        f"tok/s={tokens_per_sec:.0f}  ms/step={ms_per_step:.0f}  "
                        f"mem={mem_gb:.2f}G  | {mode_line}"
                    )
                    tqdm.write(log_line)
                    self._file_logger.info(log_line)
                    pbar.set_postfix(loss=f'{avg_loss:.4f}', lr=f'{lr:.2e}',
                                     tok_s=f'{tokens_per_sec:.0f}', mem=f'{mem_gb:.1f}G')

                    # record for summary
                    self._step_log_rows.append({
                        'step': self.global_step, 'epoch': self.epoch + 1,
                        'loss': avg_loss, 'lr': lr,
                        'tok/s': tokens_per_sec, 'ms/step': ms_per_step, 'mem_gb': mem_gb,
                        **{f'loss_{mode_names[m].lower()}': mode_losses[m] / mode_counts[m]
                           if mode_counts[m] > 0 else float('nan') for m in [0, 1, 2, 3]},
                    })

                    if self.args.use_wandb:
                        import wandb
                        log_dict = {
                            'fim/loss': avg_loss, 'fim/lr': lr,
                            'fim/step': self.global_step,
                            'fim/tokens_per_sec': tokens_per_sec,
                            'fim/ms_per_step': ms_per_step,
                            'fim/mem_gb': mem_gb,
                        }
                        for m in [0, 1, 2, 3]:
                            if mode_counts[m] > 0:
                                log_dict[f'fim/loss_{mode_names[m].lower()}'] = mode_losses[m] / mode_counts[m]
                        wandb.log(log_dict)

                if self.is_main and self.global_step % self.args.save_interval == 0:
                    self.save_checkpoint(f'step_{self.global_step}')

        pbar.close()
        return {'loss': total_loss / max(num_batches, 1)}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            if batch is None:
                continue

            token_ids = batch['token_ids'].to(self.device, non_blocking=True)
            chord_ids = batch['chord_ids'].to(self.device, non_blocking=True)
            instrument_ids = batch['instrument_ids'].to(self.device, non_blocking=True)
            token_type_ids = batch['token_type_ids'].to(self.device, non_blocking=True)
            note_ids = batch['note_ids'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)

            num_chords = batch.get('num_chords', None)
            doc_ids = batch.get('doc_ids', None)
            if doc_ids is not None:
                doc_ids = doc_ids.to(self.device, non_blocking=True)

            with torch.autocast(device_type='cuda', dtype=self.autocast_dtype):
                logits = self.model(
                    token_ids, chord_ids=chord_ids, instrument_ids=instrument_ids,
                    token_type_ids=token_type_ids, note_ids=note_ids,
                    num_chords=num_chords, doc_ids=doc_ids,
                )
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1), ignore_index=-100,
                )

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        if self.world_size > 1:
            loss_tensor = torch.tensor([avg_loss], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / self.world_size

        return {'val_loss': avg_loss}

    def _print_summary(self):
        """训练结束后打印汇总表格并写入日志文件"""
        MODE_NAMES = ['Causal', 'TrackMask', 'SkeletonFIM', 'TimeFIM']
        lines = []
        lines.append('')
        lines.append('=' * 90)
        lines.append('  FIM 训练汇总 - Per-Step 统计')
        lines.append('=' * 90)
        hdr = (f"{'Step':>7}  {'Ep':>3}  {'Loss':>8}  {'LR':>9}  {'Tok/s':>8}  "
               f"{'ms/stp':>7}  {'Mem(G)':>7}  "
               f"{'Causal':>8}  {'TrackMsk':>9}  {'SkelFIM':>8}  {'TimeFIM':>8}")
        lines.append(hdr)
        lines.append('-' * 90)
        for r in self._step_log_rows:
            lines.append(
                f"{r['step']:>7}  {r['epoch']:>3}  {r['loss']:>8.4f}  {r['lr']:>9.2e}  "
                f"{r['tok/s']:>8.0f}  {r['ms/step']:>7.0f}  {r['mem_gb']:>7.2f}  "
                f"{r.get('loss_pretrain', float('nan')):>8.4f}  "
                f"{r.get('loss_trackmask', float('nan')):>9.4f}  "
                f"{r.get('loss_skeletonfim', float('nan')):>8.4f}  "
                f"{r.get('loss_timefim', float('nan')):>8.4f}"
            )
        if self._step_log_rows:
            avg_loss = sum(r['loss']    for r in self._step_log_rows) / len(self._step_log_rows)
            avg_toks = sum(r['tok/s']   for r in self._step_log_rows) / len(self._step_log_rows)
            avg_ms   = sum(r['ms/step'] for r in self._step_log_rows) / len(self._step_log_rows)
            avg_mem  = sum(r['mem_gb']  for r in self._step_log_rows) / len(self._step_log_rows)
            def _avg_mode(key):
                vals = [r.get(key, float('nan')) for r in self._step_log_rows]
                vals = [v for v in vals if v == v]  # filter nan
                return sum(vals) / len(vals) if vals else float('nan')
            lines.append('-' * 90)
            lines.append(
                f"{'avg':>7}  {'':>3}  {avg_loss:>8.4f}  {'':>9}  {avg_toks:>8.0f}  "
                f"{avg_ms:>7.0f}  {avg_mem:>7.2f}  "
                f"{_avg_mode('loss_pretrain'):>8.4f}  "
                f"{_avg_mode('loss_trackmask'):>9.4f}  "
                f"{_avg_mode('loss_skeletonfim'):>8.4f}  "
                f"{_avg_mode('loss_timefim'):>8.4f}"
            )
        lines.append('')
        lines.append('=' * 90)
        lines.append('  FIM 训练汇总 - Per-Epoch 统计')
        lines.append('=' * 90)
        lines.append(f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>9}  {'Time(s)':>8}")
        lines.append('-' * 40)
        for r in self._epoch_log_rows:
            val_str = f"{r['val_loss']:>9.4f}" if r['val_loss'] == r['val_loss'] else '        -'
            lines.append(f"{r['epoch']:>6}  {r['train_loss']:>10.4f}  {val_str}  {r['time_s']:>8.1f}")
        lines.append('=' * 90)
        lines.append('')

        summary = '\n'.join(lines)
        tqdm.write(summary)
        self._file_logger.info(summary)
        summary_path = self.log_dir / 'summary.txt'
        summary_path.write_text(summary, encoding='utf-8')
        tqdm.write(f'[+] 汇总写入: {summary_path}')

    def train(self):
        if self.is_main:
            print("\n" + "=" * 60, flush=True)
            print("MeloFormer v2.1 FIM 微调", flush=True)
            print("=" * 60, flush=True)
            print(f"设备: {self.device}", flush=True)
            print(f"批大小: {BATCH_SIZE} (v2.1 constraint)", flush=True)
            print(f"梯度累积: {self.args.gradient_accumulation_steps}", flush=True)
            print(f"有效批大小: {BATCH_SIZE * self.world_size * self.args.gradient_accumulation_steps}", flush=True)
            print(f"FIM 比例: {self.args.fim_ratio:.0%}", flush=True)
            print(f"Mode Weights (C/T/S/TF): {self.mode_weights}", flush=True)
            print("=" * 60 + "\n", flush=True)

        for epoch in range(self.args.epochs):
            self.epoch = epoch
            if self.is_main:
                print(f"\n[FIM] Epoch {epoch + 1}/{self.args.epochs}", flush=True)

            epoch_start = time.time()
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            epoch_time = time.time() - epoch_start

            if self.is_main:
                val_loss = val_metrics.get('val_loss', float('nan')) if val_metrics else float('nan')
                self._epoch_log_rows.append({
                    'epoch': epoch + 1,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_loss,
                    'time_s': epoch_time,
                })
                msg = (f"[FIM] Epoch {epoch + 1} | "
                       f"Train Loss: {train_metrics['loss']:.4f} | "
                       f"Time: {epoch_time:.1f}s")
                if val_metrics:
                    msg += f" | Val Loss: {val_loss:.4f}"
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint('best')
                        msg += " (best)"
                tqdm.write(msg)
                self._file_logger.info(msg)
                self.save_checkpoint(f'epoch_{epoch + 1}')

        if self.is_main:
            tqdm.write("\n[FIM] 训练完成！")
            self.save_checkpoint('final')
            self._print_summary()

    def save_checkpoint(self, name: str):
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'args': vars(self.args),
            'fim_mode': True,
        }
        path = self.checkpoint_dir / f'{name}.pt'
        torch.save(checkpoint, path)
        if self.is_main:
            print(f"[FIM] 检查点已保存: {path}")


# ==================== 预训练权重加载 ====================

def _load_pretrained_checkpoint(model: nn.Module, path: str, device: torch.device, is_main: bool):
    """加载预训练检查点 (在 DDP 包装之前调用)，处理 vocab 扩展 643→651"""
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    old_state = checkpoint['model_state_dict']
    new_state = model.state_dict()

    loaded_count = 0
    for key in old_state:
        if key not in new_state:
            continue
        if old_state[key].shape == new_state[key].shape:
            new_state[key] = old_state[key]
            loaded_count += 1
        elif 'embedding' in key or 'lm_head' in key:
            old_vocab = old_state[key].shape[0]
            new_state[key][:old_vocab] = old_state[key]
            loaded_count += 1
            if is_main:
                print(f"[FIM] 扩展权重: {key} ({old_vocab} -> {new_state[key].shape[0]})")

    model.load_state_dict(new_state)

    # FIM token 语义初始化
    with torch.no_grad():
        emb = model.token_embedding.weight.data
        bos_vec = emb[1].clone()       # BOS
        chord_n_vec = emb[6].clone()   # chord 'N'

        emb[FIMTokens.TRK_PRE]  = bos_vec
        emb[FIMTokens.TRK_SUF]  = bos_vec
        emb[FIMTokens.TRK_MID]  = chord_n_vec
        emb[FIMTokens.TIME_PRE] = bos_vec
        emb[FIMTokens.TIME_SUF] = bos_vec
        emb[FIMTokens.TIME_MID] = chord_n_vec
        emb[FIMTokens.SKE]      = bos_vec
        emb[FIMTokens.SKE_MID]  = chord_n_vec

    if is_main:
        print(f"[FIM] 预训练模型已加载: {path}")
        print(f"[FIM] 成功加载 {loaded_count} 个权重")
        print(f"[FIM] 8 个 FIM tokens 语义初始化完成 (vocab 643→651)")


# ==================== 参数解析 ====================

def parse_args():
    parser = argparse.ArgumentParser(description='MeloFormer v2.1 FIM 微调')

    # 数据
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录')
    parser.add_argument('--val_split', type=float, default=0.05)
    parser.add_argument('--max_seq_len', type=int, default=24576)
    parser.add_argument('--max_chords', type=int, default=2048)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--use_arrow', action='store_true')

    # 模型
    parser.add_argument('--model_size', type=str, default='base',
                        choices=['8m', '62m', '177m', '366m', '600m', '800m', '1.5b',
                                 'small', 'base', 'large', 'xlarge'])
    parser.add_argument('--pretrained', type=str, required=True, help='预训练模型检查点路径')

    # FIM
    parser.add_argument('--fim_ratio', type=float, default=0.7)
    parser.add_argument('--mode_weights', type=str, default='1.0,1.0,1.0,1.0',
                        help='Causal/TrackMask/SkeletonFIM/TimeFIM 的 Loss 权重')
    parser.add_argument('--curriculum_learning', action='store_true')
    parser.add_argument('--curriculum_start_ratios', type=str, default='0.35,0.15,0.35,0.15')
    parser.add_argument('--curriculum_end_ratios', type=str, default='0.20,0.30,0.30,0.20')

    # 训练
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'wsd'])
    parser.add_argument('--wsd_stable_ratio', type=float, default=0.3)

    # H800 优化
    parser.add_argument('--bf16', action='store_true', default=True)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True)
    # --compile 已废弃，无实际效果（见 train.py 说明）
    parser.add_argument('--compile', action='store_true', default=False, help='(已废弃，无效果)')
    parser.add_argument('--warmup_compile', action='store_true', default=False,
                        help='训练前预编译实际会遇到的 FlexAttention kernel 桶组合')
    parser.add_argument('--flex_backend', type=str, default='TRITON', choices=['TRITON', 'FA4'],
                        help='FlexAttention 后端: TRITON (默认) 或 FA4 (需要 PyTorch nightly + FA4)')

    # v3.0 架构优化开关
    parser.add_argument('--sub_attn_bias', action='store_true', default=False,
                        help='启用子注意力 per-head additive bias (v3.1)')
    parser.add_argument('--k2v2_gate', action='store_true', default=False,
                        help='启用 K2/V2 需求门控 (v3.0)')
    parser.add_argument('--depth_selector', action='store_true', default=False,
                        help='启用 DepthSelector 跨层聚合 (v3.0)')

    # 输出
    parser.add_argument('--output_dir', type=str, default='./runs_fim')
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=1000)

    # 其他
    parser.add_argument('--num_workers', type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument('--seed', type=int, default=42)

    # WandB
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='meloformer-fim')
    parser.add_argument('--run_name', type=str, default=None)

    return parser.parse_args()


# ==================== Main ====================

def main():
    configure_torch_for_training()
    args = parse_args()

    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.bf16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        if is_main:
            print("[Warning] BF16 不支持，回退到 FP32")
        args.bf16 = False
    if args.fp16 and args.bf16:
        args.bf16 = False

    # 加载数据集
    if is_main:
        print("[FIM] 加载数据集...")
    if args.use_arrow:
        full_dataset = ArrowDataset(
            args.data_dir, max_seq_len=args.max_seq_len,
            max_chords=args.max_chords, max_samples=args.max_samples,
        )
    else:
        full_dataset = PreprocessedDataset(
            args.data_dir, max_seq_len=args.max_seq_len,
            max_chords=args.max_chords, max_samples=args.max_samples,
        )

    total_size = len(full_dataset)
    val_size = int(total_size * args.val_split)
    train_size = total_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    if is_main:
        print(f"[FIM] 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")

    # 创建模型 (FIM 扩展词表)
    set_flex_backend(args.flex_backend)
    tokenizer = HIDTokenizerV2()
    model = create_model(
        vocab_size=tokenizer.fim_vocab_size,
        model_size=args.model_size,
        max_seq_len=args.max_seq_len,
        max_chords=args.max_chords,
        chord_start_id=tokenizer.chord_start_id,
        chord_end_id=tokenizer.chord_end_id,
        fim=True,
        sub_attn_bias=getattr(args, 'sub_attn_bias', False),
        k2v2_gate=getattr(args, 'k2v2_gate', False),
        depth_selector=getattr(args, 'depth_selector', False),
    )

    total_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"[FIM] 模型参数量: {total_params / 1e6:.2f}M")

    device = torch.device(f'cuda:{local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)

    # 加载预训练权重
    _load_pretrained_checkpoint(model, args.pretrained, device, is_main)

    # Gradient Checkpointing
    if args.gradient_checkpointing:
        gc_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
        model.gradient_checkpointing_enable(autocast_dtype=gc_dtype)

    if world_size > 1:
        dist.barrier(device_ids=[local_rank])

    # 训练
    trainer = FIMTrainer(
        model, train_dataset, val_dataset, args,
        rank=rank, world_size=world_size, local_rank=local_rank, device=device,
    )

    # FlexAttention 预编译 warm-up
    if getattr(args, 'warmup_compile', False):
        from train import warmup_flex_compile
        warmup_flex_compile(
            model, device, args.max_seq_len, args.max_chords,
            is_main=is_main,
        )

    try:
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()
