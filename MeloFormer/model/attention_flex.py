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
        # v1.4.2: 确保 cos/sin 与输入 dtype 一致，避免 BF16 → FP32 提升导致显存翻倍
        cos = self.cos_cached[:, :, :seq_len, :].to(device=x.device, dtype=x.dtype)
        sin = self.sin_cached[:, :, :seq_len, :].to(device=x.device, dtype=x.dtype)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


# ============================================
# 2D RoPE (Hierarchical Position Embedding)
# ============================================

class HierarchicalRoPE(nn.Module):
    """
    分层 RoPE: 同时编码 bar 位置和 token-in-bar 位置

    音乐的二维结构:
    - Bar 维度: 第几小节 (粗粒度)
    - Token 维度: 小节内第几个 token (细粒度)

    实现方式: 将 head_dim 分成两半
    - 前半维度: 编码 bar_id
    - 后半维度: 编码 token_in_bar 位置

    收益:
    - 第 100 小节的第 1 个音符 和 第 1 小节的第 1 个音符
      在 token-in-bar 维度上距离很近，有利于学习结构
    """

    def __init__(
        self,
        dim: int,
        max_bars: int = 256,
        max_tokens_per_bar: int = 256,
        base: int = 10000,
    ):
        """
        Args:
            dim: 总维度 (必须是偶数)
            max_bars: 最大 bar 数量
            max_tokens_per_bar: 每个 bar 内最大 token 数
            base: RoPE base 频率
        """
        super().__init__()
        assert dim % 2 == 0, f"dim must be even, got {dim}"

        self.dim = dim
        self.half_dim = dim // 2
        self.max_bars = max_bars
        self.max_tokens_per_bar = max_tokens_per_bar

        # Bar 维度的频率 (前半维度)
        inv_freq_bar = 1.0 / (base ** (torch.arange(0, self.half_dim, 2).float() / self.half_dim))
        self.register_buffer('inv_freq_bar', inv_freq_bar)

        # Token-in-bar 维度的频率 (后半维度)
        inv_freq_token = 1.0 / (base ** (torch.arange(0, self.half_dim, 2).float() / self.half_dim))
        self.register_buffer('inv_freq_token', inv_freq_token)

        # 预计算缓存
        self._build_cache()

    def _build_cache(self):
        """预计算 cos/sin 缓存"""
        # Bar 维度
        bar_pos = torch.arange(self.max_bars).float()
        bar_freqs = torch.outer(bar_pos, self.inv_freq_bar)
        bar_emb = torch.cat([bar_freqs, bar_freqs], dim=-1)  # [max_bars, half_dim]
        self.register_buffer('bar_cos', bar_emb.cos(), persistent=False)
        self.register_buffer('bar_sin', bar_emb.sin(), persistent=False)

        # Token-in-bar 维度
        token_pos = torch.arange(self.max_tokens_per_bar).float()
        token_freqs = torch.outer(token_pos, self.inv_freq_token)
        token_emb = torch.cat([token_freqs, token_freqs], dim=-1)  # [max_tokens, half_dim]
        self.register_buffer('token_cos', token_emb.cos(), persistent=False)
        self.register_buffer('token_sin', token_emb.sin(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        bar_ids: torch.Tensor,
        token_in_bar_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取 2D RoPE 的 cos/sin

        Args:
            x: [batch, heads, seq_len, head_dim] 输入张量 (用于设备/dtype)
            bar_ids: [batch, seq_len] 或 [seq_len] 每个 token 的 bar ID
            token_in_bar_ids: [batch, seq_len] 或 [seq_len] 每个 token 在 bar 内的位置

        Returns:
            cos, sin: [batch, 1, seq_len, head_dim]
        """
        device = x.device
        batch_size = x.shape[0]
        seq_len = bar_ids.shape[-1]

        # 确保是 [batch, seq_len]
        if bar_ids.dim() == 1:
            bar_ids = bar_ids.unsqueeze(0).expand(batch_size, -1)
        if token_in_bar_ids.dim() == 1:
            token_in_bar_ids = token_in_bar_ids.unsqueeze(0).expand(batch_size, -1)

        # Clamp 到有效范围
        bar_ids = bar_ids.clamp(0, self.max_bars - 1)
        token_in_bar_ids = token_in_bar_ids.clamp(0, self.max_tokens_per_bar - 1)

        # 索引 cos/sin
        # bar_cos: [max_bars, half_dim] -> [batch, seq_len, half_dim]
        bar_cos = self.bar_cos.to(device)[bar_ids]  # [batch, seq_len, half_dim]
        bar_sin = self.bar_sin.to(device)[bar_ids]
        token_cos = self.token_cos.to(device)[token_in_bar_ids]  # [batch, seq_len, half_dim]
        token_sin = self.token_sin.to(device)[token_in_bar_ids]

        # 拼接: [batch, seq_len, dim]
        cos = torch.cat([bar_cos, token_cos], dim=-1)  # [batch, seq_len, dim]
        sin = torch.cat([bar_sin, token_sin], dim=-1)

        # 添加 head 维度: [batch, 1, seq_len, dim]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        # v1.4.2: 确保 cos/sin 与输入 dtype 一致，避免 BF16 → FP32 提升导致显存翻倍
        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)

        return cos, sin


def compute_token_in_bar_ids(chord_ids: torch.Tensor) -> torch.Tensor:
    """
    从 chord_ids 计算每个 token 在其 bar 内的位置

    Args:
        chord_ids: [batch, seq_len] 或 [seq_len] 每个 token 的 bar/chord ID

    Returns:
        token_in_bar_ids: 同形状，每个 token 在其 bar 内的位置 (从 0 开始)
    """
    if chord_ids.dim() == 1:
        chord_ids = chord_ids.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size, seq_len = chord_ids.shape
    device = chord_ids.device

    token_in_bar_ids = torch.zeros_like(chord_ids)

    for b in range(batch_size):
        current_bar = -1
        position_in_bar = 0
        for i in range(seq_len):
            bar_id = chord_ids[b, i].item()
            if bar_id != current_bar:
                current_bar = bar_id
                position_in_bar = 0
            token_in_bar_ids[b, i] = position_in_bar
            position_in_bar += 1

    if squeeze_output:
        token_in_bar_ids = token_in_bar_ids.squeeze(0)

    return token_in_bar_ids


def compute_token_in_bar_ids_fast(chord_ids: torch.Tensor) -> torch.Tensor:
    """
    快速计算 token_in_bar_ids (向量化版本)

    Args:
        chord_ids: [batch, seq_len] 或 [seq_len]

    Returns:
        token_in_bar_ids: 同形状
    """
    if chord_ids.dim() == 1:
        chord_ids = chord_ids.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size, seq_len = chord_ids.shape
    device = chord_ids.device

    # 检测 bar 变化点
    # shifted: 将 chord_ids 右移一位，第一个位置填 -1
    shifted = torch.cat([
        torch.full((batch_size, 1), -1, device=device, dtype=chord_ids.dtype),
        chord_ids[:, :-1]
    ], dim=1)

    # bar_change: 在 bar 变化的位置为 True
    bar_change = (chord_ids != shifted)  # [batch, seq_len]

    # 使用 cumsum 计算每个 bar 的起始位置
    # 然后用当前位置减去起始位置得到 token_in_bar
    bar_starts = torch.zeros_like(chord_ids, dtype=torch.long)

    # 对每个 batch 计算
    for b in range(batch_size):
        bar_change_indices = torch.where(bar_change[b])[0]
        if len(bar_change_indices) == 0:
            bar_starts[b] = 0
        else:
            # 每个位置找到最近的 bar 起始点
            for i, start_idx in enumerate(bar_change_indices):
                end_idx = bar_change_indices[i + 1] if i + 1 < len(bar_change_indices) else seq_len
                bar_starts[b, start_idx:end_idx] = start_idx

    # token_in_bar = position - bar_start
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    token_in_bar_ids = positions - bar_starts

    if squeeze_output:
        token_in_bar_ids = token_in_bar_ids.squeeze(0)

    return token_in_bar_ids


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
