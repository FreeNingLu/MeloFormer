#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlexAttention with Summary Token - MeloFormer 核心创新

v2.0 更新 (WSA 风格重写):
- 核心修复: mask_mod 全部改为纯索引算术，消除动态张量捕获
- 借鉴 WSA (Windowed Sink Attention) 的成功经验
- 每个 bar 的 token 序列 padding 到固定长度 BAR_LEN
- bar 归属通过 idx // BAR_LEN 算术推导，不再查表
- 编译一次，永久复用，不再触发 Triton kernel 重编译
- 移除 GQA repeat_interleave，改用 enable_gqa=True (PyTorch 2.8+)

v1.7 更新:
- 恢复 repeat_interleave: FlexAttention enable_gqa=True 触发 sdpa_dense_backward (O(N²) 稠密回退)
- [v2.0 已修复] PyTorch 2.8 不再触发 sdpa_dense_backward

实现 Summary Token 机制：
- ss (Summary → Summary): 粗粒度跨 bar 交互
- sr (Summary ← Regular): 信息压缩，S 聚合同 bar 的 R
- rs (Regular → Summary): 获取远距离上下文，R 读取已完成 bar 的 S
- rr (Regular → Regular): 细粒度近距离交互 (v2.0: 同 bar 内因果)

信息流：
1. Summarize 阶段：SS + SR → sum_x2
2. 二次投影：sum_x2 → K2, V2
3. Updating 阶段：RS + RR → reg_output

参考：
- PyTorch FlexAttention: https://pytorch.org/blog/flexattention/
- WSA: arXiv:2510.25745
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
from functools import partial

from .attention_flex import (
    RotaryPositionEmbedding,
    apply_rotary_pos_emb,
    FLEX_ATTENTION_AVAILABLE,
    check_flex_attention_available,
)
from .rms_norm import RMSNorm

if FLEX_ATTENTION_AVAILABLE:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask


# =============================================================================
# v2.0 WSA 风格常量 — 纯算术 mask_mod 的基础
# =============================================================================
# 核心思想 (借鉴 WSA):
#   bar_id = token_idx // BAR_LEN  (纯整数除法，不捕获任何外部张量)
#
# 所有 mask_mod 只使用以下常量 + 函数参数 (b, h, q_idx, kv_idx)
# 不捕获任何 GPU 张量 → 编译一次，永久复用

BAR_LEN = 128        # 每个 bar padding 到的固定 token 数
MAX_BARS = 256       # 最大 bar 数
MAX_SEQ_LEN = BAR_LEN * MAX_BARS  # 固定序列长度 = 128 × 256 = 32768
PAD_TOKEN_ID = 0
IGNORE_INDEX = -100

# 静态编译模式
_STATIC_COMPILE_MODE = False
MAX_SUM_LEN = MAX_BARS  # Summary token 序列长度 = bar 数


def is_static_compile_mode() -> bool:
    return _STATIC_COMPILE_MODE


def set_static_compile_mode(enabled: bool = True):
    global _STATIC_COMPILE_MODE
    _STATIC_COMPILE_MODE = enabled


def set_bar_len(bar_len: int, max_bars: int = 256):
    """
    设置 bar padding 长度

    Args:
        bar_len: 每个 bar 的固定 token 数 (建议 64/128/256)
        max_bars: 最大 bar 数
    """
    global BAR_LEN, MAX_BARS, MAX_SEQ_LEN
    BAR_LEN = bar_len
    MAX_BARS = max_bars
    MAX_SEQ_LEN = bar_len * max_bars
    print(f"[v2.0] BAR_LEN={BAR_LEN}, MAX_BARS={MAX_BARS}, MAX_SEQ_LEN={MAX_SEQ_LEN}")


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
    Summary Token 注意力掩码生成器 (v2.0 WSA 风格)

    v2.0 核心改动:
    - mask_mod 全部使用纯索引算术，不捕获任何 GPU 张量
    - bar 归属通过 idx // BAR_LEN 推导
    - RR 阶段简化为"同 bar 内因果"（去除乐器间稀疏规则）
    - 编译一次，永久复用

    生成两个 mask_mod 函数：
    1. summarize_mask_mod: SS + SR (Summary 聚合阶段)
    2. updating_mask_mod: RS + RR (Regular 更新阶段)
    """

    def __init__(self, rs_look_back: int = -1):
        """
        Args:
            rs_look_back: Regular 能看多少个 bar 前的 Summary
                         -1 表示所有已完成的 bar（因果）
                         正数 N 表示只看最近 N 个 bar 的 Summary
        """
        self.rs_look_back = rs_look_back

    @staticmethod
    def create_summarize_mask_mod() -> Callable:
        """
        创建 Summarize 阶段的 mask_mod (SS + SR)

        v2.0: 纯算术版本，不捕获任何 GPU 张量
        - Q: Summary tokens [0, MAX_BARS)
        - KV: [Summary; Regular] = [0, MAX_BARS + MAX_SEQ_LEN)
        - SS: S_i attend to S_j where j <= i (因果)
        - SR: S_i attend to R_j where j // BAR_LEN == i
        """
        # 捕获的全部是 Python int 常量
        sum_len = MAX_BARS
        bar_len = BAR_LEN

        def mask_mod(b, h, q_idx, kv_idx):
            is_kv_summary = kv_idx < sum_len
            is_kv_regular = ~is_kv_summary

            # SS: Summary → Summary，因果
            ss_mask = is_kv_summary & (q_idx >= kv_idx)

            # SR: Summary_i ← Regular_j，其中 bar(j) == i
            reg_kv_offset = kv_idx - sum_len
            reg_bar = reg_kv_offset // bar_len  # ← 纯算术！WSA 风格
            sr_mask = is_kv_regular & (reg_bar == q_idx)

            return ss_mask | sr_mask

        return mask_mod

    def create_updating_mask_mod(self) -> Callable:
        """
        创建 Updating 阶段的 mask_mod (RS + RR)

        v2.0: 纯算术版本，不捕获任何 GPU 张量
        - Q: Regular tokens [0, MAX_SEQ_LEN)
        - KV: [Summary K2; Regular] = [0, MAX_BARS + MAX_SEQ_LEN)
        - RS: R_i attend to S_j where bar(i) > j
        - RR: 同 bar 内因果 (简化版，去除乐器间稀疏规则)
        """
        sum_len = MAX_BARS
        bar_len = BAR_LEN
        rs_look_back = self.rs_look_back

        def mask_mod(b, h, q_idx, kv_idx):
            is_kv_summary = kv_idx < sum_len
            is_kv_regular = ~is_kv_summary

            # Query 的 bar (纯算术)
            q_bar = q_idx // bar_len

            # RS: Regular_i → Summary_j，bar(i) > j
            if rs_look_back == -1:
                rs_mask = is_kv_summary & (q_bar > kv_idx)
            else:
                start_bar = (q_bar - rs_look_back)
                # clamp to 0 using max with 0
                start_bar = start_bar * (start_bar > 0)
                rs_mask = is_kv_summary & (kv_idx >= start_bar) & (kv_idx < q_bar)

            # RR: 同 bar 内因果
            reg_kv_offset = kv_idx - sum_len
            kv_bar = reg_kv_offset // bar_len  # ← 纯算术
            same_bar = (q_bar == kv_bar)
            rr_causal = q_idx >= reg_kv_offset
            rr_mask = is_kv_regular & same_bar & rr_causal

            return rs_mask | rr_mask

        return mask_mod

    def create_block_masks(
        self,
        num_bars: int,
        batch_size: int,
        device: torch.device,
    ) -> Tuple:
        """
        创建 Summarize 和 Updating 两个阶段的 BlockMask

        v2.0: 大幅简化，不再需要传入 bar_ids/chord_ids/instrument_ids
        所有信息通过 BAR_LEN 算术推导

        Args:
            num_bars: 实际 bar 数量 (用于确定序列长度)
            batch_size: batch 大小
            device: 设备

        Returns:
            (summarize_block_mask, updating_block_mask)
        """
        check_flex_attention_available()

        seq_len = num_bars * BAR_LEN  # 固定序列长度

        # Summarize: Q=Summary(MAX_BARS), KV=[Summary; Regular](MAX_BARS + MAX_SEQ_LEN)
        summarize_mask_mod = self.create_summarize_mask_mod()
        summarize_block_mask = create_block_mask(
            summarize_mask_mod,
            B=batch_size,
            H=None,
            Q_LEN=MAX_BARS,
            KV_LEN=MAX_BARS + MAX_SEQ_LEN,
            device=device,
            _compile=True,
        )

        # Updating: Q=Regular(MAX_SEQ_LEN), KV=[Summary; Regular](MAX_BARS + MAX_SEQ_LEN)
        updating_mask_mod = self.create_updating_mask_mod()
        updating_block_mask = create_block_mask(
            updating_mask_mod,
            B=batch_size,
            H=None,
            Q_LEN=MAX_SEQ_LEN,
            KV_LEN=MAX_BARS + MAX_SEQ_LEN,
            device=device,
            _compile=True,
        )

        return summarize_block_mask, updating_block_mask


class FlexSummaryAttention(nn.Module):
    """
    FlexAttention with Summary Token + GQA + 1D RoPE

    v2.0 更新:
    - 移除 2D RoPE (已证实长序列表现不佳)，统一使用 1D RoPE
    - 不再需要 bar_ids / token_in_bar_ids 参数
    - bar padding 后位置信息由序列位置隐含

    实现两阶段注意力：
    1. Summarize: Summary 聚合 Summary + Regular
    2. Updating: Regular 聚合 Summary(K2,V2) + Regular
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int = None,
        dropout: float = 0.1,
        bias: bool = True,
        max_seq_len: int = 32768,
        max_bars: int = 256,
    ):
        super().__init__()

        check_flex_attention_available()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout

        # GQA
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.kv_dim = self.head_dim * self.num_kv_heads

        assert embed_dim % num_heads == 0
        assert num_heads % self.num_kv_heads == 0

        # Summary 投影
        self.sum_q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.sum_k_proj = nn.Linear(embed_dim, self.kv_dim, bias=bias)
        self.sum_v_proj = nn.Linear(embed_dim, self.kv_dim, bias=bias)
        self.sum_out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Regular 投影
        self.reg_q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.reg_k_proj = nn.Linear(embed_dim, self.kv_dim, bias=bias)
        self.reg_v_proj = nn.Linear(embed_dim, self.kv_dim, bias=bias)
        self.reg_out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # K2, V2 二次投影
        self.sum_k2_proj = nn.Linear(embed_dim, self.kv_dim, bias=bias)
        self.sum_v2_proj = nn.Linear(embed_dim, self.kv_dim, bias=bias)

        # 1D RoPE (Summary 和 Regular 各自独立)
        self.sum_rope = RotaryPositionEmbedding(self.head_dim, max_bars)
        self.reg_rope = RotaryPositionEmbedding(self.head_dim, max_seq_len)

    def forward(
        self,
        sum_x: torch.Tensor,
        reg_x: torch.Tensor,
        summarize_block_mask,
        updating_block_mask,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        v2.0 简化版 forward — 不再需要 bar_ids / token_in_bar_ids

        Args:
            sum_x: (batch, num_bars, embed_dim)
            reg_x: (batch, seq_len, embed_dim)
            summarize_block_mask: SS + SR 的 BlockMask
            updating_block_mask: RS + RR 的 BlockMask

        Returns:
            sum_output: (batch, num_bars, embed_dim)
            reg_output: (batch, seq_len, embed_dim)
        """
        batch_size = sum_x.size(0)
        sum_len = sum_x.size(1)
        reg_len = reg_x.size(1)

        # === 投影 ===
        sum_q = self.sum_q_proj(sum_x)
        sum_k = self.sum_k_proj(sum_x)
        sum_v = self.sum_v_proj(sum_x)

        reg_q = self.reg_q_proj(reg_x)
        reg_k = self.reg_k_proj(reg_x)
        reg_v = self.reg_v_proj(reg_x)

        # 重塑为多头
        sum_q = sum_q.view(batch_size, sum_len, self.num_heads, self.head_dim).transpose(1, 2)
        sum_k = sum_k.view(batch_size, sum_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        sum_v = sum_v.view(batch_size, sum_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        reg_q = reg_q.view(batch_size, reg_len, self.num_heads, self.head_dim).transpose(1, 2)
        reg_k = reg_k.view(batch_size, reg_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        reg_v = reg_v.view(batch_size, reg_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 1D RoPE
        sum_cos, sum_sin = self.sum_rope(sum_x, sum_len)
        sum_q, sum_k = apply_rotary_pos_emb(sum_q, sum_k, sum_cos, sum_sin)

        reg_cos, reg_sin = self.reg_rope(reg_x, reg_len)
        reg_q, reg_k = apply_rotary_pos_emb(reg_q, reg_k, reg_cos, reg_sin)

        # GQA: 使用 enable_gqa=True (PyTorch 2.8+, 不再需要 repeat_interleave)
        # dtype 对齐: RoPE 可能将 Q/K 提升到 FP32，V 仍为 BF16
        target_dtype = sum_q.dtype
        if sum_v.dtype != target_dtype:
            sum_v = sum_v.to(target_dtype)
        if reg_v.dtype != target_dtype:
            reg_v = reg_v.to(target_dtype)

        # === 阶段 1: Summarize (SS + SR) ===
        cat_k_sum = torch.cat([sum_k, reg_k], dim=2)
        cat_v_sum = torch.cat([sum_v, reg_v], dim=2)
        sum_attn_out = flex_attention(
            sum_q, cat_k_sum, cat_v_sum,
            block_mask=summarize_block_mask,
            enable_gqa=True,
        )
        sum_attn_out = sum_attn_out.transpose(1, 2).contiguous().view(batch_size, sum_len, self.embed_dim)

        # === 阶段 2: K2, V2 二次投影 ===
        sum_k2 = self.sum_k2_proj(sum_attn_out)
        sum_v2 = self.sum_v2_proj(sum_attn_out)
        sum_k2 = sum_k2.view(batch_size, sum_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        sum_v2 = sum_v2.view(batch_size, sum_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if sum_k2.dtype != target_dtype:
            sum_k2 = sum_k2.to(target_dtype)
            sum_v2 = sum_v2.to(target_dtype)

        # === 阶段 3: Updating (RS + RR) ===
        cat_k_reg = torch.cat([sum_k2, reg_k], dim=2)
        cat_v_reg = torch.cat([sum_v2, reg_v], dim=2)
        reg_attn_out = flex_attention(
            reg_q, cat_k_reg, cat_v_reg,
            block_mask=updating_block_mask,
            enable_gqa=True,
        )
        reg_attn_out = reg_attn_out.transpose(1, 2).contiguous().view(batch_size, reg_len, self.embed_dim)

        # === 输出投影 ===
        sum_output = self.sum_out_proj(sum_attn_out)
        reg_output = self.reg_out_proj(reg_attn_out)

        return sum_output, reg_output


class FlexSummaryAttentionBlock(nn.Module):
    """
    FlexAttention Summary Block (v2.0)

    v2.0 更新:
    - 移除 2D RoPE 参数 (统一 1D RoPE)
    - 移除 bar_ids / token_in_bar_ids 参数
    - 保留亚层级 Gradient Checkpointing
    - 保留 GQA, RMSNorm, SwiGLU

    结构: Pre-Norm + 残差连接
    - Summary 和 Regular 各自有独立的 FFN
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
        share_ffn: bool = False,
        num_kv_heads: int = None,
        use_rms_norm: bool = True,
    ):
        super().__init__()

        self.attention = FlexSummaryAttention(
            embed_dim, num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            max_bars=max_bars,
        )

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

        self._gradient_checkpointing = False
        self._autocast_dtype = None

    def _apply_ffn(self, x: torch.Tensor, is_summary: bool) -> torch.Tensor:
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sum_x_norm = self.sum_attn_norm(sum_x)
        reg_x_norm = self.reg_attn_norm(reg_x)
        sum_attn_out, reg_attn_out = self.attention(
            sum_x_norm, reg_x_norm,
            summarize_block_mask, updating_block_mask,
        )
        return self.dropout(sum_attn_out), self.dropout(reg_attn_out)

    def _run_ffn(
        self,
        sum_x: torch.Tensor,
        reg_x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sum_ffn_out = self._apply_ffn(self.sum_ffn_norm(sum_x), is_summary=True)
        reg_ffn_out = self._apply_ffn(self.reg_ffn_norm(reg_x), is_summary=False)
        return self.dropout(sum_ffn_out), self.dropout(reg_ffn_out)

    def forward(
        self,
        sum_x: torch.Tensor,
        reg_x: torch.Tensor,
        summarize_block_mask,
        updating_block_mask,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        v2.0 简化版 forward — 不再需要 bar_ids / token_in_bar_ids

        Args:
            sum_x: (batch, num_bars, embed_dim)
            reg_x: (batch, seq_len, embed_dim)  seq_len = num_bars * BAR_LEN
            summarize_block_mask: SS + SR 的 BlockMask
            updating_block_mask: RS + RR 的 BlockMask
        """
        if self._gradient_checkpointing and self.training:
            autocast_dtype = self._autocast_dtype

            def run_attn_ckpt(s_x, r_x):
                if autocast_dtype is not None:
                    with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                        return self._run_attention(s_x, r_x, summarize_block_mask, updating_block_mask)
                return self._run_attention(s_x, r_x, summarize_block_mask, updating_block_mask)

            sum_attn_out, reg_attn_out = torch.utils.checkpoint.checkpoint(
                run_attn_ckpt, sum_x, reg_x, use_reentrant=False, preserve_rng_state=True,
            )
            sum_x = sum_x + sum_attn_out
            reg_x = reg_x + reg_attn_out

            def run_ffn_ckpt(s_x, r_x):
                if autocast_dtype is not None:
                    with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                        return self._run_ffn(s_x, r_x)
                return self._run_ffn(s_x, r_x)

            sum_ffn_out, reg_ffn_out = torch.utils.checkpoint.checkpoint(
                run_ffn_ckpt, sum_x, reg_x, use_reentrant=False, preserve_rng_state=True,
            )
            sum_x = sum_x + sum_ffn_out
            reg_x = reg_x + reg_ffn_out
            return sum_x, reg_x

        else:
            # Attention + residual
            sum_residual, reg_residual = sum_x, reg_x
            sum_attn_out, reg_attn_out = self._run_attention(
                sum_x, reg_x, summarize_block_mask, updating_block_mask,
            )
            sum_x = sum_residual + sum_attn_out
            reg_x = reg_residual + reg_attn_out

            # FFN + residual
            sum_residual, reg_residual = sum_x, reg_x
            sum_ffn_out, reg_ffn_out = self._run_ffn(sum_x, reg_x)
            sum_x = sum_residual + sum_ffn_out
            reg_x = reg_residual + reg_ffn_out

            return sum_x, reg_x


if __name__ == '__main__':
    """
    v2.0 测试脚本

    测试内容:
    1. 前向传播正确性
    2. 反向传播正确性 (梯度存在且有限)
    3. 重编译检测 (不同 batch 不应触发重编译)

    在潞晨云上运行: python -m MeloFormer.model.attention_flex_summary
    """
    import time

    print("=== Testing FlexAttention Summary Token v2.0 (WSA-style) ===\n")
    print(f"PyTorch version: {torch.__version__}")
    print(f"FlexAttention available: {FLEX_ATTENTION_AVAILABLE}")

    if not FLEX_ATTENTION_AVAILABLE:
        print("\nFlexAttention 不可用，需要 PyTorch 2.5+")
        exit(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # === 配置 ===
    num_bars = 8
    embed_dim = 256
    num_heads = 4
    batch_size = 2

    set_bar_len(bar_len=32, max_bars=num_bars)  # 测试用小 BAR_LEN
    seq_len = num_bars * BAR_LEN  # 8 * 32 = 256

    print(f"\nConfig: BAR_LEN={BAR_LEN}, MAX_BARS={MAX_BARS}, seq_len={seq_len}")

    # === 创建测试数据 (bar padding 格式) ===
    reg_x = torch.randn(batch_size, seq_len, embed_dim, device=device, requires_grad=True)

    sum_embedding = SummaryTokenEmbedding(embed_dim, max_bars=MAX_BARS).to(device)
    sum_x = sum_embedding(num_bars, batch_size, device)

    print(f"sum_x: {sum_x.shape}, reg_x: {reg_x.shape}")

    # === 创建 BlockMask (只需 num_bars，不需要 bar_ids!) ===
    print("\nCreating BlockMasks (v2.0 pure arithmetic)...")
    mask_gen = FlexSummaryAttentionMask()
    t0 = time.time()
    sum_mask, upd_mask = mask_gen.create_block_masks(num_bars, batch_size, device)
    t1 = time.time()
    print(f"  BlockMask created in {t1-t0:.2f}s")

    # === 创建 Block ===
    block = FlexSummaryAttentionBlock(
        embed_dim, num_heads, embed_dim * 4,
        max_seq_len=seq_len, max_bars=MAX_BARS,
    ).to(device)

    params = sum(p.numel() for p in block.parameters())
    print(f"  Block params: {params / 1e6:.2f}M")

    # === 测试 1: 前向传播 ===
    print("\n[Test 1] Forward pass...")
    sum_out, reg_out = block(sum_x, reg_x, sum_mask, upd_mask)
    print(f"  sum: {sum_x.shape} -> {sum_out.shape}")
    print(f"  reg: {reg_x.shape} -> {reg_out.shape}")
    assert sum_out.shape == sum_x.shape
    assert reg_out.shape == reg_x.shape
    print("  PASSED")

    # === 测试 2: 反向传播 ===
    print("\n[Test 2] Backward pass...")
    loss = sum_out.sum() + reg_out.sum()
    loss.backward()
    assert reg_x.grad is not None, "reg_x.grad is None!"
    assert torch.isfinite(reg_x.grad).all(), "reg_x.grad has non-finite values!"
    print(f"  reg_x.grad norm: {reg_x.grad.norm().item():.4f}")
    print("  PASSED")

    # === 测试 3: 重编译检测 ===
    print("\n[Test 3] Recompilation check (3 batches with different data)...")
    times = []
    for i in range(3):
        reg_x_new = torch.randn(batch_size, seq_len, embed_dim, device=device)
        sum_x_new = sum_embedding(num_bars, batch_size, device)

        t0 = time.time()
        with torch.no_grad():
            block(sum_x_new, reg_x_new, sum_mask, upd_mask)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.time()
        times.append(t1 - t0)
        print(f"  Batch {i+1}: {times[-1]*1000:.1f}ms")

    # 第一次可能慢 (编译)，后续应该快且稳定
    if len(times) >= 3 and times[2] < times[0] * 0.5:
        print("  WARNING: 后续 batch 显著变快，可能第一次触发了编译 (正常)")
    if len(times) >= 3 and times[2] > times[1] * 3:
        print("  ERROR: 后续 batch 变慢，可能触发了重编译!")
    else:
        print("  PASSED (无重编译迹象)")

    print("\n=== All Tests Passed ===")
