#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlexAttention with Summary Token - MeloFormer 核心创新

v1.7 更新:
- 恢复 repeat_interleave: FlexAttention enable_gqa=True 触发 sdpa_dense_backward (O(N²) 稠密回退)
- 稀疏反向传播不支持非连续内存，物理复制虽多占显存但避免 12GB OOM
- 保留 token_in_bar_ids 预计算优化

v1.6 更新 (已回退):
- 移除 GQA repeat_interleave: 使用 FlexAttention enable_gqa=True 原生支持
- ⚠️ 发现问题: 触发稠密回退，导致反向传播 OOM

v1.4.2 更新:
- 修复 RoPE dtype 问题: cos/sin 现在在源头转换为输入 dtype
- 避免 BF16 → FP32 隐式提升导致显存翻倍
- GC 显存节省恢复到正常水平 (~40-60%)

v1.4.1 更新:
- 移除 FP32 强制转换，FlexAttention 在 BF16 下工作
- 大幅减少显存占用 (反向传播内存减半)

实现 Summary Token 机制：
- ss (Summary → Summary): 粗粒度跨 bar 交互
- sr (Summary ← Regular): 信息压缩，S 聚合同 bar 的 R
- rs (Regular → Summary): 获取远距离上下文，R 读取已完成 bar 的 S
- rr (Regular → Regular): 细粒度近距离交互

信息流：
1. Summarize 阶段：SS + SR → sum_x2
2. 二次投影：sum_x2 → K2, V2
3. Updating 阶段：RS + RR → reg_output

参考：
- PyTorch FlexAttention: https://pytorch.org/blog/flexattention/
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
from functools import partial

from .attention_flex import (
    RotaryPositionEmbedding,
    HierarchicalRoPE,
    compute_token_in_bar_ids_fast,
    apply_rotary_pos_emb,
    FlexFCAttentionMask,
    TOKEN_TYPE_VISIBILITY,
    FLEX_ATTENTION_AVAILABLE,
    check_flex_attention_available,
)
from .rms_norm import RMSNorm

if FLEX_ATTENTION_AVAILABLE:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask


# =============================================================================
# H800 静态编译优化常量 (v1.3)
# =============================================================================
# 这些常量确保 FlexAttention 的 create_block_mask 只需编译一次
# 消除闭包捕获动态变量导致的重编译问题
#
# 关键点:
# 1. 所有输入必须 Padding 到固定长度
# 2. mask_mod 只使用这些常量，不捕获动态变量
# 3. 通过 functools.partial 传递 Tensor 参数
MAX_SEQ_LEN = 16384  # 固定序列长度 (与 --max_seq_len 一致)
MAX_SUM_LEN = 256    # 固定最大 bar 数 (与 --max_bars 一致)
PAD_TOKEN_ID = 0     # Padding token ID
IGNORE_INDEX = -100  # Loss 计算忽略的 index

# 是否启用静态编译模式 (通过 set_static_compile_mode 设置)
_STATIC_COMPILE_MODE = False


def set_static_compile_mode(enabled: bool, max_seq_len: int = 16384, max_sum_len: int = 256):
    """
    设置静态编译模式

    Args:
        enabled: 是否启用静态编译
        max_seq_len: 固定序列长度
        max_sum_len: 固定 Summary (bar) 数量
    """
    global _STATIC_COMPILE_MODE, MAX_SEQ_LEN, MAX_SUM_LEN
    _STATIC_COMPILE_MODE = enabled
    MAX_SEQ_LEN = max_seq_len
    MAX_SUM_LEN = max_sum_len
    if enabled:
        print(f"[静态编译] 已启用: MAX_SEQ_LEN={MAX_SEQ_LEN}, MAX_SUM_LEN={MAX_SUM_LEN}")


def is_static_compile_mode() -> bool:
    """检查是否启用静态编译模式"""
    return _STATIC_COMPILE_MODE


class SummaryTokenEmbedding(nn.Module):
    """
    Summary Token 嵌入层

    为每个 bar 生成一个 Summary Token 嵌入。
    支持：
    1. 可学习嵌入（每个 bar 位置一个）
    2. 共享嵌入（所有 bar 使用同一嵌入）
    """

    def __init__(
        self,
        embed_dim: int,
        max_bars: int = 256,
        learnable: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_bars = max_bars
        self.learnable = learnable

        if learnable:
            # 可学习嵌入：每个 bar 位置一个
            self.embedding = nn.Embedding(max_bars, embed_dim)
            nn.init.normal_(self.embedding.weight, std=0.02)
        else:
            # 共享嵌入：所有 bar 使用同一向量
            self.embedding = nn.Parameter(torch.zeros(1, embed_dim))
            nn.init.normal_(self.embedding, std=0.02)

    def forward(
        self,
        num_bars: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        生成 Summary Token 嵌入

        Args:
            num_bars: bar 数量
            batch_size: batch 大小
            device: 设备

        Returns:
            embeddings: (batch, num_bars, embed_dim)
        """
        # 边界检查
        if num_bars > self.max_bars:
            raise ValueError(
                f"num_bars ({num_bars}) > max_bars ({self.max_bars}). "
                f"请使用 --max_bars {num_bars} 或更大的值"
            )

        if self.learnable:
            bar_indices = torch.arange(num_bars, device=device)
            embeddings = self.embedding(bar_indices)  # (num_bars, embed_dim)
            embeddings = embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            embeddings = self.embedding.expand(batch_size, num_bars, -1)

        return embeddings.to(device)


class FlexSummaryAttentionMask:
    """
    Summary Token 注意力掩码生成器 (FlexAttention 版本)

    生成两个 mask_mod 函数：
    1. summarize_mask_mod: SS + SR (Summary 聚合阶段)
    2. updating_mask_mod: RS + RR (Regular 更新阶段)
    """

    def __init__(
        self,
        rr_mask_generator: Optional[FlexFCAttentionMask] = None,
        rs_look_back: int = -1,
    ):
        """
        Args:
            rr_mask_generator: 用于生成 rr 掩码的 FlexFCAttentionMask 实例
            rs_look_back: Regular 能看多少个 bar 前的 Summary
                         -1 表示所有已完成的 bar（因果）
                         正数 N 表示只看最近 N 个 bar 的 Summary
        """
        self.rr_mask_generator = rr_mask_generator or FlexFCAttentionMask()
        self.rs_look_back = rs_look_back

    def create_summarize_mask_mod(
        self,
        bar_ids: torch.Tensor,  # (seq_len,) 每个 Regular token 的 bar ID
        num_bars: int,
        seq_len: int,
    ) -> Callable:
        """
        创建 Summarize 阶段的 mask_mod (SS + SR)

        Query: Summary tokens [0, num_bars)
        Key/Value: [Summary tokens; Regular tokens] = [0, num_bars + seq_len)

        SS: S_i attend to S_j where j <= i (因果)
        SR: S_i attend to R_j where bar(R_j) == i

        v1.3 更新: 支持静态编译模式
        - 静态模式: 使用全局常量 MAX_SUM_LEN, MAX_SEQ_LEN
        - 动态模式: 保持原有逻辑 (闭包捕获)
        """
        bar_ids = bar_ids.contiguous()

        if _STATIC_COMPILE_MODE:
            # ============================================
            # 静态编译模式: 使用固定常量，避免重编译
            # ============================================
            # 注意: 这里 bar_ids 已经 Padding 到 MAX_SEQ_LEN

            def mask_mod_static(b, h, q_idx, kv_idx):
                # 使用全局常量
                sum_len = MAX_SUM_LEN
                total_kv_len = MAX_SUM_LEN + MAX_SEQ_LEN

                # 安全处理越界索引
                q_idx_safe = q_idx.clamp(0, sum_len - 1)
                kv_idx_safe = kv_idx.clamp(0, total_kv_len - 1)

                # 判断 kv 是 Summary 还是 Regular
                is_kv_summary = kv_idx_safe < sum_len
                is_kv_regular = ~is_kv_summary

                # SS: Summary → Summary，因果注意力
                ss_mask = is_kv_summary & (q_idx_safe >= kv_idx_safe)

                # SR: Summary ← Regular
                reg_kv_idx = (kv_idx_safe - sum_len).clamp(0, MAX_SEQ_LEN - 1)
                reg_bar = bar_ids[reg_kv_idx].clamp(0, sum_len - 1)
                sr_mask = is_kv_regular & (reg_bar == q_idx_safe)

                return ss_mask | sr_mask

            return mask_mod_static

        else:
            # ============================================
            # 动态模式: 原有逻辑 (闭包捕获变量)
            # ============================================
            sum_len = max(num_bars, 1)
            reg_len = max(seq_len, 1)
            total_kv_len = sum_len + reg_len
            bar_ids_len = bar_ids.size(0)

            def mask_mod_dynamic(b, h, q_idx, kv_idx):
                q_idx_safe = q_idx.clamp(0, sum_len - 1)
                kv_idx_safe = kv_idx.clamp(0, total_kv_len - 1)

                is_kv_summary = kv_idx_safe < sum_len
                is_kv_regular = ~is_kv_summary

                ss_mask = is_kv_summary & (q_idx_safe >= kv_idx_safe)

                reg_kv_idx = (kv_idx_safe - sum_len).clamp(0, bar_ids_len - 1)
                reg_bar = bar_ids[reg_kv_idx].clamp(0, sum_len - 1)
                sr_mask = is_kv_regular & (reg_bar == q_idx_safe)

                return ss_mask | sr_mask

            return mask_mod_dynamic

    def create_updating_mask_mod(
        self,
        bar_ids: torch.Tensor,           # (seq_len,) 每个 Regular token 的 bar ID
        chord_ids: torch.Tensor,         # (seq_len,) 每个 Regular token 的 chord ID
        instrument_ids: torch.Tensor,    # (seq_len,) 每个 Regular token 的乐器 ID
        num_bars: int,
        seq_len: int,
        token_type_ids: Optional[torch.Tensor] = None,
        note_ids: Optional[torch.Tensor] = None,
        doc_ids: Optional[torch.Tensor] = None,  # 新增: 文档 ID (用于序列打包)
    ) -> Callable:
        """
        创建 Updating 阶段的 mask_mod (RS + RR)

        Query: Regular tokens [0, seq_len)
        Key/Value: [Summary K2/V2; Regular K/V] = [0, num_bars + seq_len)

        RS: R_i attend to S_j where bar(R_i) > j (只看已完成 bar 的 Summary)
        RR: 使用 FlexFCAttentionMask 的逻辑

        v1.3 更新: 支持静态编译模式
        """
        bar_ids = bar_ids.contiguous()
        chord_ids = chord_ids.contiguous()
        instrument_ids = instrument_ids.contiguous()

        rs_look_back = self.rs_look_back

        # Token 类型稀疏
        use_token_type_sparsity = token_type_ids is not None and note_ids is not None
        if use_token_type_sparsity:
            token_type_ids = token_type_ids.contiguous()
            note_ids = note_ids.contiguous()
            type_visibility = TOKEN_TYPE_VISIBILITY.to(bar_ids.device)
        else:
            type_visibility = None

        # RR mask 参数
        cross_inst_full_range = self.rr_mask_generator.cross_inst_full_range
        cross_inst_far_offsets = self.rr_mask_generator.cross_inst_far_offsets

        # 序列打包
        use_doc_mask = doc_ids is not None
        if use_doc_mask:
            doc_ids = doc_ids.contiguous()

        if _STATIC_COMPILE_MODE:
            # ============================================
            # 静态编译模式: 使用固定常量
            # ============================================
            def mask_mod_static(b, h, q_idx, kv_idx):
                # 使用全局常量
                sum_len = MAX_SUM_LEN
                total_kv_len = MAX_SUM_LEN + MAX_SEQ_LEN

                # 安全处理越界索引
                q_idx_safe = q_idx.clamp(0, MAX_SEQ_LEN - 1)
                kv_idx_safe = kv_idx.clamp(0, total_kv_len - 1)

                # 判断 kv 是 Summary 还是 Regular
                is_kv_summary = kv_idx_safe < sum_len
                is_kv_regular = ~is_kv_summary

                # Query 的 bar
                q_bar = bar_ids[q_idx_safe].clamp(0, sum_len - 1)

                # === RS: Regular → Summary ===
                if rs_look_back == -1:
                    rs_mask = is_kv_summary & (q_bar > kv_idx_safe)
                else:
                    start_bar = (q_bar - rs_look_back).clamp(0)
                    rs_mask = is_kv_summary & (kv_idx_safe >= start_bar) & (kv_idx_safe < q_bar)

                # === RR: Regular → Regular ===
                reg_kv_idx = (kv_idx_safe - sum_len).clamp(0, MAX_SEQ_LEN - 1)

                # 因果性
                rr_causal = q_idx_safe >= reg_kv_idx

                # 获取 token 信息
                q_chord = chord_ids[q_idx_safe]
                k_chord = chord_ids[reg_kv_idx]
                q_inst = instrument_ids[q_idx_safe]
                k_inst = instrument_ids[reg_kv_idx]

                # 全局 token
                is_global_k = (k_chord == -1) | (k_inst == 129)

                # 同乐器
                same_inst = (q_inst == k_inst) & (q_inst < 129) & (k_inst < 129)

                # 跨乐器近距离
                chord_diff = q_chord - k_chord
                diff_inst = (q_inst != k_inst) & (q_inst < 129) & (k_inst < 129)
                cross_near = diff_inst & (chord_diff >= 0) & (chord_diff <= cross_inst_full_range)

                # 跨乐器远距离
                cross_far = torch.zeros_like(rr_causal)
                for offset in cross_inst_far_offsets:
                    cross_far = cross_far | (diff_inst & (chord_diff == offset))

                bar_mask = is_global_k | same_inst | cross_near | cross_far

                # Token 类型级稀疏
                if use_token_type_sparsity:
                    q_type = token_type_ids[q_idx_safe]
                    k_type = token_type_ids[reg_kv_idx]
                    q_note = note_ids[q_idx_safe]
                    k_note = note_ids[reg_kv_idx]

                    same_note = (q_note == k_note) & (q_note >= 0) & (k_note >= 0)
                    q_type_safe = q_type.clamp(0, 3)
                    k_type_safe = k_type.clamp(0, 3)
                    type_visible = type_visibility[q_type_safe, k_type_safe]
                    is_global_type = (q_type == -1) | (k_type == -1)
                    is_global_note = (q_note == -1) | (k_note == -1)
                    token_type_mask = same_note | type_visible | is_global_type | is_global_note

                    rr_mask = is_kv_regular & rr_causal & bar_mask & token_type_mask
                else:
                    rr_mask = is_kv_regular & rr_causal & bar_mask

                # 序列打包
                if use_doc_mask:
                    q_doc = doc_ids[q_idx_safe]
                    k_doc = doc_ids[reg_kv_idx]
                    same_doc = (q_doc == k_doc)
                    rr_mask = rr_mask & same_doc

                return rs_mask | rr_mask

            return mask_mod_static

        else:
            # ============================================
            # 动态模式: 原有逻辑 (闭包捕获变量)
            # ============================================
            sum_len = max(num_bars, 1)
            reg_len = max(seq_len, 1)
            total_kv_len = sum_len + reg_len

            bar_ids_len = bar_ids.size(0)
            chord_ids_len = chord_ids.size(0)
            instrument_ids_len = instrument_ids.size(0)
            token_type_ids_len = token_type_ids.size(0) if use_token_type_sparsity else 0
            note_ids_len = note_ids.size(0) if use_token_type_sparsity else 0
            doc_ids_len = doc_ids.size(0) if use_doc_mask else 0

            def mask_mod_dynamic(b, h, q_idx, kv_idx):
                q_idx_safe = q_idx.clamp(0, bar_ids_len - 1)
                kv_idx_safe = kv_idx.clamp(0, total_kv_len - 1)

                is_kv_summary = kv_idx_safe < sum_len
                is_kv_regular = ~is_kv_summary

                q_bar = bar_ids[q_idx_safe].clamp(0, sum_len - 1)

                if rs_look_back == -1:
                    rs_mask = is_kv_summary & (q_bar > kv_idx_safe)
                else:
                    start_bar = (q_bar - rs_look_back).clamp(0)
                    rs_mask = is_kv_summary & (kv_idx_safe >= start_bar) & (kv_idx_safe < q_bar)

                reg_kv_idx = (kv_idx_safe - sum_len).clamp(0, bar_ids_len - 1)
                rr_causal = q_idx_safe >= reg_kv_idx

                q_chord = chord_ids[q_idx_safe.clamp(0, chord_ids_len - 1)]
                k_chord = chord_ids[reg_kv_idx.clamp(0, chord_ids_len - 1)]
                q_inst = instrument_ids[q_idx_safe.clamp(0, instrument_ids_len - 1)]
                k_inst = instrument_ids[reg_kv_idx.clamp(0, instrument_ids_len - 1)]

                is_global_k = (k_chord == -1) | (k_inst == 129)
                same_inst = (q_inst == k_inst) & (q_inst < 129) & (k_inst < 129)

                chord_diff = q_chord - k_chord
                diff_inst = (q_inst != k_inst) & (q_inst < 129) & (k_inst < 129)
                cross_near = diff_inst & (chord_diff >= 0) & (chord_diff <= cross_inst_full_range)

                cross_far = torch.zeros_like(rr_causal)
                for offset in cross_inst_far_offsets:
                    cross_far = cross_far | (diff_inst & (chord_diff == offset))

                bar_mask = is_global_k | same_inst | cross_near | cross_far

                if use_token_type_sparsity:
                    q_type = token_type_ids[q_idx_safe.clamp(0, token_type_ids_len - 1)]
                    k_type = token_type_ids[reg_kv_idx.clamp(0, token_type_ids_len - 1)]
                    q_note = note_ids[q_idx_safe.clamp(0, note_ids_len - 1)]
                    k_note = note_ids[reg_kv_idx.clamp(0, note_ids_len - 1)]

                    same_note = (q_note == k_note) & (q_note >= 0) & (k_note >= 0)
                    q_type_safe = q_type.clamp(0, 3)
                    k_type_safe = k_type.clamp(0, 3)
                    type_visible = type_visibility[q_type_safe, k_type_safe]
                    is_global_type = (q_type == -1) | (k_type == -1)
                    is_global_note = (q_note == -1) | (k_note == -1)
                    token_type_mask = same_note | type_visible | is_global_type | is_global_note

                    rr_mask = is_kv_regular & rr_causal & bar_mask & token_type_mask
                else:
                    rr_mask = is_kv_regular & rr_causal & bar_mask

                if use_doc_mask:
                    q_doc = doc_ids[q_idx_safe.clamp(0, doc_ids_len - 1)]
                    k_doc = doc_ids[reg_kv_idx.clamp(0, doc_ids_len - 1)]
                    same_doc = (q_doc == k_doc)
                    rr_mask = rr_mask & same_doc

                return rs_mask | rr_mask

            return mask_mod_dynamic

    def create_block_masks(
        self,
        bar_ids: torch.Tensor,
        chord_ids: torch.Tensor,
        instrument_ids: torch.Tensor,
        num_bars: int,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        token_type_ids: Optional[torch.Tensor] = None,
        note_ids: Optional[torch.Tensor] = None,
        doc_ids: Optional[torch.Tensor] = None,  # 新增: 文档 ID (用于序列打包)
    ) -> Tuple:
        """
        创建 Summarize 和 Updating 两个阶段的 BlockMask

        Args:
            doc_ids: 文档 ID tensor，用于序列打包时隔离不同文档的注意力

        Returns:
            (summarize_block_mask, updating_block_mask)

        v1.3 更新: 静态编译模式下使用固定维度
        """
        check_flex_attention_available()

        # 如果是 batch 格式，取第一个
        if bar_ids.dim() == 2:
            bar_ids_flat = bar_ids[0]
            chord_ids_flat = chord_ids[0]
            instrument_ids_flat = instrument_ids[0]
        else:
            bar_ids_flat = bar_ids
            chord_ids_flat = chord_ids
            instrument_ids_flat = instrument_ids

        bar_ids_flat = bar_ids_flat.to(device)
        chord_ids_flat = chord_ids_flat.to(device)
        instrument_ids_flat = instrument_ids_flat.to(device)

        # 静态编译模式: 使用固定的 MAX_SEQ_LEN
        if _STATIC_COMPILE_MODE:
            target_len = MAX_SEQ_LEN
            target_sum_len = MAX_SUM_LEN
        else:
            target_len = seq_len
            target_sum_len = num_bars

        # 确保 tensor 长度与目标长度一致
        if bar_ids_flat.size(0) < target_len:
            pad_len = target_len - bar_ids_flat.size(0)
            bar_ids_flat = F.pad(bar_ids_flat, (0, pad_len), value=bar_ids_flat[-1].item() if bar_ids_flat.numel() > 0 else 0)
            chord_ids_flat = F.pad(chord_ids_flat, (0, pad_len), value=chord_ids_flat[-1].item() if chord_ids_flat.numel() > 0 else 0)
            instrument_ids_flat = F.pad(instrument_ids_flat, (0, pad_len), value=129)

        # Token 类型
        token_type_ids_flat = None
        note_ids_flat = None
        if token_type_ids is not None:
            token_type_ids_flat = token_type_ids[0] if token_type_ids.dim() == 2 else token_type_ids
            token_type_ids_flat = token_type_ids_flat.to(device)
            if token_type_ids_flat.size(0) < target_len:
                pad_len = target_len - token_type_ids_flat.size(0)
                token_type_ids_flat = F.pad(token_type_ids_flat, (0, pad_len), value=-1)
        if note_ids is not None:
            note_ids_flat = note_ids[0] if note_ids.dim() == 2 else note_ids
            note_ids_flat = note_ids_flat.to(device)
            if note_ids_flat.size(0) < target_len:
                pad_len = target_len - note_ids_flat.size(0)
                note_ids_flat = F.pad(note_ids_flat, (0, pad_len), value=-1)

        # 文档 ID (序列打包)
        doc_ids_flat = None
        if doc_ids is not None:
            doc_ids_flat = doc_ids[0] if doc_ids.dim() == 2 else doc_ids
            doc_ids_flat = doc_ids_flat.to(device)
            if doc_ids_flat.size(0) < target_len:
                pad_len = target_len - doc_ids_flat.size(0)
                doc_ids_flat = F.pad(doc_ids_flat, (0, pad_len),
                                     value=doc_ids_flat[-1].item() if doc_ids_flat.numel() > 0 else 0)

        # Summarize 阶段: Q=Summary, KV=[Summary; Regular]
        summarize_mask_mod = self.create_summarize_mask_mod(
            bar_ids_flat, num_bars, seq_len
        )

        # 静态模式使用固定维度，动态模式使用实际维度
        if _STATIC_COMPILE_MODE:
            sum_q_len = MAX_SUM_LEN
            sum_kv_len = MAX_SUM_LEN + MAX_SEQ_LEN
            upd_q_len = MAX_SEQ_LEN
            upd_kv_len = MAX_SUM_LEN + MAX_SEQ_LEN
        else:
            sum_q_len = num_bars
            sum_kv_len = num_bars + seq_len
            upd_q_len = seq_len
            upd_kv_len = num_bars + seq_len

        summarize_block_mask = create_block_mask(
            summarize_mask_mod,
            B=batch_size,
            H=None,
            Q_LEN=sum_q_len,
            KV_LEN=sum_kv_len,
            device=device,
            _compile=True,
        )

        # Updating 阶段: Q=Regular, KV=[Summary; Regular]
        updating_mask_mod = self.create_updating_mask_mod(
            bar_ids_flat, chord_ids_flat, instrument_ids_flat,
            num_bars, seq_len,
            token_type_ids=token_type_ids_flat,
            note_ids=note_ids_flat,
            doc_ids=doc_ids_flat,
        )
        updating_block_mask = create_block_mask(
            updating_mask_mod,
            B=batch_size,
            H=None,
            Q_LEN=upd_q_len,
            KV_LEN=upd_kv_len,
            device=device,
            _compile=True,
        )

        return summarize_block_mask, updating_block_mask


class FlexSummaryAttention(nn.Module):
    """
    FlexAttention with Summary Token + GQA + 2D RoPE

    v1.4 更新:
    - GQA (Grouped Query Attention): 减少 KV Cache
    - 2D RoPE: 分层位置编码 (bar + token-in-bar)

    实现两阶段注意力：
    1. Summarize: Summary 聚合 Summary + Regular
    2. Updating: Regular 聚合 Summary(K2,V2) + Regular
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int = None,  # v1.4: GQA - KV heads 数量
        dropout: float = 0.1,
        bias: bool = True,
        max_seq_len: int = 32768,
        max_bars: int = 256,
        use_rope: bool = True,
        use_2d_rope: bool = True,  # v1.4: 是否使用 2D RoPE
    ):
        super().__init__()

        check_flex_attention_available()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_rope = use_rope
        self.use_2d_rope = use_2d_rope
        self.dropout_p = dropout

        # GQA: num_kv_heads 默认等于 num_heads (标准 MHA)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.kv_dim = self.head_dim * self.num_kv_heads

        assert embed_dim % num_heads == 0
        assert num_heads % self.num_kv_heads == 0, \
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"

        # Summary 投影 (Q 用 num_heads, K/V 用 num_kv_heads)
        self.sum_q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.sum_k_proj = nn.Linear(embed_dim, self.kv_dim, bias=bias)  # GQA
        self.sum_v_proj = nn.Linear(embed_dim, self.kv_dim, bias=bias)  # GQA
        self.sum_out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Regular 投影 (Q 用 num_heads, K/V 用 num_kv_heads)
        self.reg_q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.reg_k_proj = nn.Linear(embed_dim, self.kv_dim, bias=bias)  # GQA
        self.reg_v_proj = nn.Linear(embed_dim, self.kv_dim, bias=bias)  # GQA
        self.reg_out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # K2, V2 二次投影 (Summary 输出 → RS 的 K/V)
        self.sum_k2_proj = nn.Linear(embed_dim, self.kv_dim, bias=bias)  # GQA
        self.sum_v2_proj = nn.Linear(embed_dim, self.kv_dim, bias=bias)  # GQA

        # RoPE
        if use_rope:
            if use_2d_rope:
                # 2D RoPE: 分层位置编码
                self.rope = HierarchicalRoPE(
                    self.head_dim,
                    max_bars=max_bars,
                    max_tokens_per_bar=256,
                )
                self.rope_1d = RotaryPositionEmbedding(self.head_dim, max_seq_len)  # 用于 Summary
            else:
                self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len)
                self.rope_1d = self.rope
        else:
            self.rope = None
            self.rope_1d = None

    def forward(
        self,
        sum_x: torch.Tensor,           # (batch, num_bars, embed_dim)
        reg_x: torch.Tensor,           # (batch, seq_len, embed_dim)
        summarize_block_mask,          # Summarize 阶段的 BlockMask
        updating_block_mask,           # Updating 阶段的 BlockMask
        bar_ids: torch.Tensor = None,  # v1.4: (batch, seq_len) 用于 2D RoPE
        token_in_bar_ids: torch.Tensor = None,  # v1.6: 预计算的 token_in_bar_ids
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            sum_x: Summary token 嵌入 (batch, num_bars, embed_dim)
            reg_x: Regular token 嵌入 (batch, seq_len, embed_dim)
            summarize_block_mask: SS + SR 的 BlockMask
            updating_block_mask: RS + RR 的 BlockMask
            bar_ids: (batch, seq_len) 每个 token 的 bar ID，用于 2D RoPE
            token_in_bar_ids: (batch, seq_len) 预计算的 token-in-bar ID，避免每层重复计算

        Returns:
            sum_output: (batch, num_bars, embed_dim)
            reg_output: (batch, seq_len, embed_dim)
        """
        batch_size = sum_x.size(0)
        sum_len = sum_x.size(1)
        reg_len = reg_x.size(1)

        # === 投影 ===
        # Summary Q, K, V
        sum_q = self.sum_q_proj(sum_x)
        sum_k = self.sum_k_proj(sum_x)
        sum_v = self.sum_v_proj(sum_x)

        # Regular Q, K, V
        reg_q = self.reg_q_proj(reg_x)
        reg_k = self.reg_k_proj(reg_x)
        reg_v = self.reg_v_proj(reg_x)

        # 重塑为多头: Q 用 num_heads, K/V 用 num_kv_heads
        sum_q = sum_q.view(batch_size, sum_len, self.num_heads, self.head_dim).transpose(1, 2)
        sum_k = sum_k.view(batch_size, sum_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        sum_v = sum_v.view(batch_size, sum_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        reg_q = reg_q.view(batch_size, reg_len, self.num_heads, self.head_dim).transpose(1, 2)
        reg_k = reg_k.view(batch_size, reg_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        reg_v = reg_v.view(batch_size, reg_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 应用 RoPE
        if self.rope is not None:
            # Summary 使用 1D RoPE (bar 位置)
            sum_cos, sum_sin = self.rope_1d(sum_x, sum_len)
            sum_q, sum_k = apply_rotary_pos_emb(sum_q, sum_k, sum_cos, sum_sin)

            # Regular 使用 2D RoPE 或 1D RoPE
            if self.use_2d_rope and bar_ids is not None:
                # 2D RoPE: 使用预计算的 token_in_bar_ids (v1.6 优化)
                if token_in_bar_ids is None:
                    # 兼容性: 如果没有预计算，则在此计算
                    token_in_bar_ids = compute_token_in_bar_ids_fast(bar_ids)
                reg_cos, reg_sin = self.rope(reg_x, bar_ids, token_in_bar_ids)
                reg_q, reg_k = apply_rotary_pos_emb(reg_q, reg_k, reg_cos, reg_sin)
            else:
                # 1D RoPE: Regular 使用 token 位置 (从 sum_len 开始)
                total_len = sum_len + reg_len
                cos, sin = self.rope_1d(reg_x, total_len)
                reg_cos = cos[:, :, sum_len:, :]
                reg_sin = sin[:, :, sum_len:, :]
                reg_q, reg_k = apply_rotary_pos_emb(reg_q, reg_k, reg_cos, reg_sin)

        # v1.7: 恢复 repeat_interleave - FlexAttention 稀疏反向传播不支持非连续内存
        # enable_gqa=True 会触发 sdpa_dense_backward (O(N²) 稠密回退)，导致 12GB OOM
        # 物理复制虽然多占一点显存，但能保证稀疏反向传播正常工作
        if self.num_kv_groups > 1:
            sum_k = sum_k.repeat_interleave(self.num_kv_groups, dim=1)
            sum_v = sum_v.repeat_interleave(self.num_kv_groups, dim=1)
            reg_k = reg_k.repeat_interleave(self.num_kv_groups, dim=1)
            reg_v = reg_v.repeat_interleave(self.num_kv_groups, dim=1)

        # v1.4.1: 确保所有 tensor dtype 一致 (RoPE cos/sin 可能是 FP32)
        target_dtype = sum_q.dtype
        if sum_k.dtype != target_dtype:
            sum_k = sum_k.to(target_dtype)
        if sum_v.dtype != target_dtype:
            sum_v = sum_v.to(target_dtype)
        if reg_k.dtype != target_dtype:
            reg_k = reg_k.to(target_dtype)
        if reg_v.dtype != target_dtype:
            reg_v = reg_v.to(target_dtype)

        # === 阶段 1: Summarize (SS + SR) ===
        # v1.4.1: 移除 FP32 强制转换，让 FlexAttention 在 BF16 下工作
        # PyTorch 2.5+ FlexAttention 已支持 BF16 反向传播
        # 拼接 K, V: [Summary; Regular]
        cat_k_sum = torch.cat([sum_k, reg_k], dim=2)
        cat_v_sum = torch.cat([sum_v, reg_v], dim=2)

        # FlexAttention: Q=Summary, KV=[Summary; Regular]
        # v1.7: 移除 enable_gqa，使用物理复制后的等头数 KV
        sum_attn_out = flex_attention(sum_q, cat_k_sum, cat_v_sum, block_mask=summarize_block_mask)

        # 重塑
        sum_attn_out = sum_attn_out.transpose(1, 2).contiguous().view(batch_size, sum_len, self.embed_dim)

        # === 阶段 2: K2, V2 二次投影 ===
        sum_k2 = self.sum_k2_proj(sum_attn_out)
        sum_v2 = self.sum_v2_proj(sum_attn_out)

        # 重塑为多头 (K2/V2 也使用 num_kv_heads)
        sum_k2 = sum_k2.view(batch_size, sum_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        sum_v2 = sum_v2.view(batch_size, sum_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # v1.7: 恢复 repeat_interleave (K2/V2 也需要)
        if self.num_kv_groups > 1:
            sum_k2 = sum_k2.repeat_interleave(self.num_kv_groups, dim=1)
            sum_v2 = sum_v2.repeat_interleave(self.num_kv_groups, dim=1)

        # v1.4.1: 确保 K2/V2 dtype 与 reg_q 一致
        target_dtype = reg_q.dtype
        if sum_k2.dtype != target_dtype:
            sum_k2 = sum_k2.to(target_dtype)
        if sum_v2.dtype != target_dtype:
            sum_v2 = sum_v2.to(target_dtype)

        # === 阶段 3: Updating (RS + RR) ===
        # v1.4.1: 移除 FP32 强制转换，让 FlexAttention 在 BF16 下工作
        # 拼接 K, V: [Summary K2/V2; Regular K/V]
        cat_k_reg = torch.cat([sum_k2, reg_k], dim=2)
        cat_v_reg = torch.cat([sum_v2, reg_v], dim=2)

        # FlexAttention: Q=Regular, KV=[Summary K2; Regular]
        # v1.7: 移除 enable_gqa，使用物理复制后的等头数 KV
        reg_attn_out = flex_attention(reg_q, cat_k_reg, cat_v_reg, block_mask=updating_block_mask)

        # 重塑
        reg_attn_out = reg_attn_out.transpose(1, 2).contiguous().view(batch_size, reg_len, self.embed_dim)

        # === 输出投影 ===
        sum_output = self.sum_out_proj(sum_attn_out)
        reg_output = self.reg_out_proj(reg_attn_out)

        return sum_output, reg_output


class FlexSummaryAttentionBlock(nn.Module):
    """
    FlexAttention Summary Block (Pre-LN/RMSNorm + RoPE/2D-RoPE + SwiGLU + GQA)

    v1.7.4 更新:
    - 亚层级 Gradient Checkpointing: Attention 和 FFN 分别 checkpoint
    - 显存节省: 72GB → ~50GB (预计)

    v1.4 更新:
    - GQA (Grouped Query Attention)
    - RMSNorm 替代 LayerNorm
    - 2D RoPE 分层位置编码

    整合 FlexSummaryAttention 和 FFN：
    - Summary 和 Regular 各自有独立的 FFN
    - Pre-Norm 架构
    - 残差连接
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        activation: str = 'swiglu',
        max_seq_len: int = 32768,
        max_bars: int = 256,
        use_rope: bool = True,
        use_2d_rope: bool = True,  # v1.4
        share_ffn: bool = False,
        num_kv_heads: int = None,  # v1.4: GQA
        use_rms_norm: bool = True,  # v1.4: 使用 RMSNorm
    ):
        super().__init__()

        self.use_2d_rope = use_2d_rope

        # Summary Attention with GQA
        self.attention = FlexSummaryAttention(
            embed_dim, num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            max_bars=max_bars,
            use_rope=use_rope,
            use_2d_rope=use_2d_rope,
        )

        # Normalization (RMSNorm or LayerNorm)
        NormLayer = RMSNorm if use_rms_norm else nn.LayerNorm
        self.sum_attn_norm = NormLayer(embed_dim)
        self.reg_attn_norm = NormLayer(embed_dim)

        # FFN for Summary (SwiGLU)
        self.use_swiglu = activation == 'swiglu'
        if self.use_swiglu:
            self.sum_ffn_gate = nn.Linear(embed_dim, ffn_dim, bias=False)
            self.sum_ffn_up = nn.Linear(embed_dim, ffn_dim, bias=False)
            self.sum_ffn_down = nn.Linear(ffn_dim, embed_dim, bias=False)
        else:
            self.sum_ffn = nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, embed_dim),
                nn.Dropout(dropout),
            )

        # FFN for Regular
        self.share_ffn = share_ffn
        if not share_ffn:
            if self.use_swiglu:
                self.reg_ffn_gate = nn.Linear(embed_dim, ffn_dim, bias=False)
                self.reg_ffn_up = nn.Linear(embed_dim, ffn_dim, bias=False)
                self.reg_ffn_down = nn.Linear(ffn_dim, embed_dim, bias=False)
            else:
                self.reg_ffn = nn.Sequential(
                    nn.Linear(embed_dim, ffn_dim),
                    nn.GELU() if activation == 'gelu' else nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_dim, embed_dim),
                    nn.Dropout(dropout),
                )

        self.sum_ffn_norm = NormLayer(embed_dim)
        self.reg_ffn_norm = NormLayer(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # v1.7.4: 亚层级 Gradient Checkpointing 标志
        self._gradient_checkpointing = False
        self._autocast_dtype = None

    def _apply_ffn(self, x: torch.Tensor, is_summary: bool) -> torch.Tensor:
        """应用 FFN"""
        if self.use_swiglu:
            if is_summary or self.share_ffn:
                return self.sum_ffn_down(F.silu(self.sum_ffn_gate(x)) * self.sum_ffn_up(x))
            else:
                return self.reg_ffn_down(F.silu(self.reg_ffn_gate(x)) * self.reg_ffn_up(x))
        else:
            if is_summary or self.share_ffn:
                return self.sum_ffn(x)
            else:
                return self.reg_ffn(x)

    def _run_attention(
        self,
        sum_x: torch.Tensor,
        reg_x: torch.Tensor,
        summarize_block_mask,
        updating_block_mask,
        bar_ids: torch.Tensor,
        token_in_bar_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        v1.7.4: Attention 子层 (用于亚层级 checkpoint)
        """
        sum_x_norm = self.sum_attn_norm(sum_x)
        reg_x_norm = self.reg_attn_norm(reg_x)

        sum_attn_out, reg_attn_out = self.attention(
            sum_x_norm, reg_x_norm,
            summarize_block_mask, updating_block_mask,
            bar_ids=bar_ids,
            token_in_bar_ids=token_in_bar_ids,
        )

        sum_attn_out = self.dropout(sum_attn_out)
        reg_attn_out = self.dropout(reg_attn_out)

        return sum_attn_out, reg_attn_out

    def _run_ffn(
        self,
        sum_x: torch.Tensor,
        reg_x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        v1.7.4: FFN 子层 (用于亚层级 checkpoint)
        """
        sum_x_norm = self.sum_ffn_norm(sum_x)
        reg_x_norm = self.reg_ffn_norm(reg_x)

        sum_ffn_out = self._apply_ffn(sum_x_norm, is_summary=True)
        reg_ffn_out = self._apply_ffn(reg_x_norm, is_summary=False)

        sum_ffn_out = self.dropout(sum_ffn_out)
        reg_ffn_out = self.dropout(reg_ffn_out)

        return sum_ffn_out, reg_ffn_out

    def forward(
        self,
        sum_x: torch.Tensor,
        reg_x: torch.Tensor,
        summarize_block_mask,
        updating_block_mask,
        bar_ids: torch.Tensor = None,  # v1.4: 用于 2D RoPE
        token_in_bar_ids: torch.Tensor = None,  # v1.6: 预计算的 token_in_bar_ids
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        v1.7.4: 支持亚层级 Gradient Checkpointing
        - Attention 和 FFN 分别 checkpoint
        - 计算 Attention 时释放 FFN 中间变量，反之亦然
        - 显存节省预计: 72GB → ~50GB

        Args:
            sum_x: Summary token 嵌入 (batch, num_bars, embed_dim)
            reg_x: Regular token 嵌入 (batch, seq_len, embed_dim)
            summarize_block_mask: SS + SR 的 BlockMask
            updating_block_mask: RS + RR 的 BlockMask
            bar_ids: (batch, seq_len) 每个 token 的 bar ID，用于 2D RoPE
            token_in_bar_ids: (batch, seq_len) 预计算的 token-in-bar ID

        Returns:
            sum_output: (batch, num_bars, embed_dim)
            reg_output: (batch, seq_len, embed_dim)
        """
        # v1.7.4: 亚层级 Gradient Checkpointing
        if self._gradient_checkpointing and self.training:
            autocast_dtype = self._autocast_dtype

            # === Checkpoint 1: Attention ===
            def run_attn_ckpt(s_x, r_x):
                if autocast_dtype is not None:
                    with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                        return self._run_attention(
                            s_x, r_x,
                            summarize_block_mask, updating_block_mask,
                            bar_ids, token_in_bar_ids
                        )
                else:
                    return self._run_attention(
                        s_x, r_x,
                        summarize_block_mask, updating_block_mask,
                        bar_ids, token_in_bar_ids
                    )

            sum_attn_out, reg_attn_out = torch.utils.checkpoint.checkpoint(
                run_attn_ckpt, sum_x, reg_x,
                use_reentrant=False,
                preserve_rng_state=True,
            )

            # Attention 残差连接
            sum_x = sum_x + sum_attn_out
            reg_x = reg_x + reg_attn_out

            # === Checkpoint 2: FFN ===
            # 此时 Attention 的中间显存已被释放
            def run_ffn_ckpt(s_x, r_x):
                if autocast_dtype is not None:
                    with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                        return self._run_ffn(s_x, r_x)
                else:
                    return self._run_ffn(s_x, r_x)

            sum_ffn_out, reg_ffn_out = torch.utils.checkpoint.checkpoint(
                run_ffn_ckpt, sum_x, reg_x,
                use_reentrant=False,
                preserve_rng_state=True,
            )

            # FFN 残差连接
            sum_x = sum_x + sum_ffn_out
            reg_x = reg_x + reg_ffn_out

            return sum_x, reg_x

        else:
            # === 非 Checkpoint 模式: 原有逻辑 ===
            # Attention with residual (Pre-Norm)
            sum_residual = sum_x
            reg_residual = reg_x

            sum_x = self.sum_attn_norm(sum_x)
            reg_x = self.reg_attn_norm(reg_x)

            sum_x, reg_x = self.attention(
                sum_x, reg_x,
                summarize_block_mask, updating_block_mask,
                bar_ids=bar_ids,
                token_in_bar_ids=token_in_bar_ids,
            )

            sum_x = self.dropout(sum_x)
            reg_x = self.dropout(reg_x)

            sum_x = sum_residual + sum_x
            reg_x = reg_residual + reg_x

            # FFN with residual (Pre-LN)
            sum_residual = sum_x
            reg_residual = reg_x

            sum_x = self.sum_ffn_norm(sum_x)
            reg_x = self.reg_ffn_norm(reg_x)

            sum_x = self._apply_ffn(sum_x, is_summary=True)
            reg_x = self._apply_ffn(reg_x, is_summary=False)

            sum_x = self.dropout(sum_x)
            reg_x = self.dropout(reg_x)

            sum_x = sum_residual + sum_x
            reg_x = reg_residual + reg_x

            return sum_x, reg_x


if __name__ == '__main__':
    print("=== Testing FlexAttention Summary Token Module ===\n")

    print(f"PyTorch version: {torch.__version__}")
    print(f"FlexAttention available: {FLEX_ATTENTION_AVAILABLE}")

    if not FLEX_ATTENTION_AVAILABLE:
        print("\nFlexAttention 不可用，需要 PyTorch 2.5+")
        exit(0)

    # 测试参数
    batch_size = 2
    num_bars = 8
    seq_len = 256
    embed_dim = 256
    num_heads = 4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 创建测试数据
    reg_x = torch.randn(batch_size, seq_len, embed_dim, device=device)

    # bar_ids: 模拟每个 token 属于哪个 bar
    bar_ids = torch.repeat_interleave(
        torch.arange(num_bars, device=device),
        seq_len // num_bars
    )[:seq_len].unsqueeze(0).expand(batch_size, -1)

    # chord_ids: 与 bar_ids 相同
    chord_ids = bar_ids.clone()

    # instrument_ids: 交替 0, 1, 2, 3
    instrument_ids = (torch.arange(seq_len, device=device) % 4).unsqueeze(0).expand(batch_size, -1)

    print(f"\nTest data:")
    print(f"  reg_x shape: {reg_x.shape}")
    print(f"  bar_ids shape: {bar_ids.shape}")
    print(f"  num_bars: {num_bars}")

    # 创建 Summary Token 嵌入
    sum_embedding = SummaryTokenEmbedding(embed_dim).to(device)
    sum_x = sum_embedding(num_bars, batch_size, device)
    print(f"  sum_x shape: {sum_x.shape}")

    # 创建掩码
    print("\nCreating BlockMasks...")
    mask_generator = FlexSummaryAttentionMask()
    summarize_mask, updating_mask = mask_generator.create_block_masks(
        bar_ids, chord_ids, instrument_ids,
        num_bars, batch_size, seq_len, device
    )
    print(f"  Summarize BlockMask created")
    print(f"  Updating BlockMask created")

    # 创建 Summary Attention Block
    print("\nCreating FlexSummaryAttentionBlock...")
    block = FlexSummaryAttentionBlock(
        embed_dim, num_heads, embed_dim * 4,
        max_seq_len=seq_len
    ).to(device)

    params = sum(p.numel() for p in block.parameters())
    print(f"  Parameters: {params / 1e6:.2f}M")

    # 前向传播
    print("\nForward pass...")
    with torch.no_grad():
        sum_out, reg_out = block(sum_x, reg_x, summarize_mask, updating_mask)

    print(f"  Summary input: {sum_x.shape} -> output: {sum_out.shape}")
    print(f"  Regular input: {reg_x.shape} -> output: {reg_out.shape}")

    # 验证
    assert sum_out.shape == sum_x.shape
    assert reg_out.shape == reg_x.shape

    print("\n=== All Tests Passed ===")
