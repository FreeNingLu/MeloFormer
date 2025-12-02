#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlexAttention 基础模块

提供给 attention_flex_summary.py 使用：
- TOKEN_TYPE_VISIBILITY: Token 类型可见性矩阵
- RotaryPositionEmbedding: RoPE 位置编码
- FlexFCAttentionMask: RR 掩码生成器
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable

# ============================================
# Token 类型可见性矩阵 (基于 NMI 分析)
# ============================================

# Token 类型: 0=T(Onset), 1=P(Pitch), 2=D(Duration), 3=V(Velocity), -1=全局
# T↔T, T↔P, P↔P: 可见 | V↔V: 可见 | D 相关: 不可见
TOKEN_TYPE_VISIBILITY = torch.tensor([
    # Key:  T      P      D      V
    [True,  True,  False, False],  # Query T
    [True,  True,  False, False],  # Query P
    [False, False, False, False],  # Query D
    [False, False, False, True ],  # Query V
], dtype=torch.bool)


# ============================================
# RoPE (Rotary Position Embedding)
# ============================================

class RotaryPositionEmbedding(nn.Module):
    """RoPE 位置编码 - 更好的长序列外推能力"""

    def __init__(self, dim: int, max_seq_len: int = 32768, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        return (
            self.cos_cached[:, :, :seq_len, :].to(x.device),
            self.sin_cached[:, :, :seq_len, :].to(x.device)
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


# ============================================
# FlexAttention 检查
# ============================================

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    flex_attention = None
    create_block_mask = None


def check_flex_attention_available():
    if not FLEX_ATTENTION_AVAILABLE:
        raise RuntimeError(f"需要 PyTorch 2.5+ (当前: {torch.__version__})")


# ============================================
# FlexFCAttentionMask - RR 掩码生成器
# ============================================

class FlexFCAttentionMask:
    """
    FC-Attention 掩码生成器 (用于 RR block)

    规则:
    1. 全局 token: 所有 token 都能看到
    2. 同乐器: 全连接
    3. 跨乐器近距离 (offset ≤ 2): 全连接
    4. 跨乐器远距离: 只看 offset=4
    5. Token 类型稀疏: 基于 NMI 分析
    """

    CROSS_INST_FAR_OFFSETS = (4,)
    CROSS_INST_FULL_RANGE = 2

    def __init__(self, cross_inst_full_range: int = 2, cross_inst_far_offsets: Tuple[int, ...] = None):
        self.cross_inst_full_range = cross_inst_full_range
        self.cross_inst_far_offsets = cross_inst_far_offsets or self.CROSS_INST_FAR_OFFSETS

    def create_mask_mod(
        self,
        chord_ids: torch.Tensor,
        instrument_ids: torch.Tensor,
        seq_len: int,
        token_type_ids: Optional[torch.Tensor] = None,
        note_ids: Optional[torch.Tensor] = None,
    ) -> Callable:
        """创建 mask_mod 闭包"""
        chord_ids = chord_ids.contiguous()
        instrument_ids = instrument_ids.contiguous()

        use_token_type_sparsity = token_type_ids is not None and note_ids is not None
        if use_token_type_sparsity:
            token_type_ids = token_type_ids.contiguous()
            note_ids = note_ids.contiguous()
            type_visibility = TOKEN_TYPE_VISIBILITY.to(chord_ids.device)

        cross_inst_full_range = self.cross_inst_full_range
        cross_inst_far_offsets = self.cross_inst_far_offsets
        actual_seq_len = seq_len

        def mask_mod(b, h, q_idx, kv_idx):
            q_idx_safe = q_idx.clamp(0, actual_seq_len - 1)
            kv_idx_safe = kv_idx.clamp(0, actual_seq_len - 1)

            # 因果性
            causal = q_idx >= kv_idx

            q_chord = chord_ids[q_idx_safe]
            k_chord = chord_ids[kv_idx_safe]
            q_inst = instrument_ids[q_idx_safe]
            k_inst = instrument_ids[kv_idx_safe]

            # 全局 token
            is_global_k = (k_chord == -1) | (k_inst == 129)

            # 同乐器
            same_inst = (q_inst == k_inst) & (q_inst < 129) & (k_inst < 129)

            # 跨乐器近距离
            chord_diff = q_chord - k_chord
            diff_inst = (q_inst != k_inst) & (q_inst < 129) & (k_inst < 129)
            cross_near = diff_inst & (chord_diff >= 0) & (chord_diff <= cross_inst_full_range)

            # 跨乐器远距离
            cross_far = torch.zeros_like(causal)
            for offset in cross_inst_far_offsets:
                cross_far = cross_far | (diff_inst & (chord_diff == offset))

            bar_mask = is_global_k | same_inst | cross_near | cross_far

            # Token 类型稀疏
            if use_token_type_sparsity:
                q_type = token_type_ids[q_idx_safe]
                k_type = token_type_ids[kv_idx_safe]
                q_note = note_ids[q_idx_safe]
                k_note = note_ids[kv_idx_safe]

                same_note = (q_note == k_note) & (q_note >= 0) & (k_note >= 0)
                q_type_safe = q_type.clamp(0, 3)
                k_type_safe = k_type.clamp(0, 3)
                type_visible = type_visibility[q_type_safe, k_type_safe]
                is_global_type = (q_type == -1) | (k_type == -1)
                is_global_note = (q_note == -1) | (k_note == -1)
                token_type_mask = same_note | type_visible | is_global_type | is_global_note

                return causal & bar_mask & token_type_mask
            else:
                return causal & bar_mask

        return mask_mod

    def create_block_mask_for_batch(
        self,
        chord_ids: torch.Tensor,
        instrument_ids: torch.Tensor,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        device: torch.device,
        token_type_ids: Optional[torch.Tensor] = None,
        note_ids: Optional[torch.Tensor] = None,
    ):
        """创建 BlockMask"""
        check_flex_attention_available()

        if chord_ids.dim() == 2:
            chord_ids_flat = chord_ids[0]
            instrument_ids_flat = instrument_ids[0]
        else:
            chord_ids_flat = chord_ids
            instrument_ids_flat = instrument_ids

        token_type_ids_flat = None
        note_ids_flat = None
        if token_type_ids is not None:
            token_type_ids_flat = token_type_ids[0] if token_type_ids.dim() == 2 else token_type_ids
            token_type_ids_flat = token_type_ids_flat.to(device)
        if note_ids is not None:
            note_ids_flat = note_ids[0] if note_ids.dim() == 2 else note_ids
            note_ids_flat = note_ids_flat.to(device)

        chord_ids_flat = chord_ids_flat.to(device)
        instrument_ids_flat = instrument_ids_flat.to(device)

        mask_mod = self.create_mask_mod(
            chord_ids_flat, instrument_ids_flat, seq_len,
            token_type_ids=token_type_ids_flat,
            note_ids=note_ids_flat
        )

        return create_block_mask(
            mask_mod,
            B=batch_size,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
            _compile=True,
        )
