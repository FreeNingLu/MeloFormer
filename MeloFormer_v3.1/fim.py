#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIM (Fill-In-the-Middle) 数据处理模块

提供 FIM 训练的核心组件:
- FIMTokens: FIM 特殊 token 定义
- FIMMode: FIM 训练模式枚举
- FIMProcessor: FIM 数据变换处理器
- load_pretrained_for_fim: 预训练权重加载 (vocab 扩展 643→651)
"""

import random
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from enum import Enum


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
            fim_ratio: FIM 数据比例 (1 - fim_ratio 为 Pretrain)
            min_prefix_len: 最小前缀长度
            min_suffix_len: 最小后缀长度
            min_middle_len: 最小中间长度
            curriculum_learning: 是否启用课程学习
            curriculum_start_ratios: 课程学习初始比例 (Pretrain, TrackMask, SkeletonFIM, TimeFIM)
            curriculum_end_ratios: 课程学习结束比例 (Pretrain, TrackMask, SkeletonFIM, TimeFIM)
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
        Renumber chord_ids to be sequential 0,1,2,... by first-occurrence order.
        FIM special tokens (chord_id=-1) are preserved.
        """
        valid = chord_ids >= 0
        if not valid.any():
            return chord_ids.clone()

        new_ids = torch.full_like(chord_ids, -1)
        vals = chord_ids[valid]

        # 按首次出现顺序编号: 用 unique + first-occurrence 排名
        unique_vals, inverse = vals.unique(return_inverse=True)
        # unique_vals 是排序的，但我们需要按首次出现顺序
        # 找每个 unique 值的最小位置索引
        num_unique = unique_vals.size(0)
        first_pos = torch.full((num_unique,), len(vals), dtype=torch.long)
        # scatter_reduce amin: 对每个 unique 值找最小的出现位置
        first_pos.scatter_reduce_(0, inverse, torch.arange(len(vals)), reduce='amin')
        # 按 first_pos 排序得到首次出现顺序的排名
        rank = torch.zeros(num_unique, dtype=torch.long)
        rank[first_pos.argsort()] = torch.arange(num_unique)

        new_ids[valid] = rank[inverse]
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
        """Pretrain 模式: 保持原样"""
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


# ==================== 预训练权重加载 ====================

def load_pretrained_for_fim(model: nn.Module, path: str, device: torch.device, is_main: bool):
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
