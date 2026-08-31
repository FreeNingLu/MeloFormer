#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlexAttention 基础模块 (v2.1)

提供给 attention_flex_summary.py 使用：
- TOKEN_TYPE_VISIBILITY: Token 类型可见性矩阵
- RotaryPositionEmbedding: 1D RoPE 位置编码
- apply_rotary_pos_emb: RoPE 应用函数

v2.1 清理:
- 移除 HierarchicalRoPE (2D RoPE 已证伪)
- 移除 compute_token_in_bar_ids / compute_token_in_bar_ids_fast (v2.1 不使用)
- 保留 FlexFCAttentionMask (可选的 RR 细粒度掩码生成器，供未来 v2.2 使用)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable

# ============================================
# Token 类型可见性矩阵 (基于 NMI 分析)
# ============================================

# Token 类型: 0=T(Onset), 1=P(Pitch), 2=D(Duration), 3=V(Velocity), -1=全局
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
        cos = self.cos_cached[:, :, :seq_len, :].to(device=x.device, dtype=x.dtype)
        sin = self.sin_cached[:, :, :seq_len, :].to(device=x.device, dtype=x.dtype)
        return cos, sin


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
