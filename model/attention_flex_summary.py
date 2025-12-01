#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlexAttention with Summary Token - MuseFormer 核心创新

实现 MuseFormer 论文的 Summary Token 机制：
- ss (Summary → Summary): 粗粒度跨 bar 交互
- sr (Summary ← Regular): 信息压缩，S 聚合同 bar 的 R
- rs (Regular → Summary): 获取远距离上下文，R 读取已完成 bar 的 S
- rr (Regular → Regular): 细粒度近距离交互

信息流：
1. Summarize 阶段：SS + SR → sum_x2
2. 二次投影：sum_x2 → K2, V2
3. Updating 阶段：RS + RR → reg_output

参考：
- MuseFormer: https://arxiv.org/abs/2210.10349
- PyTorch FlexAttention: https://pytorch.org/blog/flexattention/
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable

from .attention_flex import (
    RotaryPositionEmbedding,
    apply_rotary_pos_emb,
    FlexFCAttentionMask,
    TOKEN_TYPE_VISIBILITY,
    FLEX_ATTENTION_AVAILABLE,
    check_flex_attention_available,
)

if FLEX_ATTENTION_AVAILABLE:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask


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
        """
        bar_ids = bar_ids.contiguous()
        sum_len = max(num_bars, 1)  # 确保至少为 1
        reg_len = max(seq_len, 1)   # 确保至少为 1
        total_kv_len = sum_len + reg_len
        # bar_ids 的长度 (已在 create_block_masks 中填充到 seq_len)
        bar_ids_len = bar_ids.size(0)

        def mask_mod(b, h, q_idx, kv_idx):
            # 安全处理越界索引
            q_idx_safe = q_idx.clamp(0, sum_len - 1)
            kv_idx_safe = kv_idx.clamp(0, total_kv_len - 1)

            # 判断 kv 是 Summary 还是 Regular
            is_kv_summary = kv_idx_safe < sum_len
            is_kv_regular = ~is_kv_summary

            # SS: Summary → Summary，因果注意力
            ss_mask = is_kv_summary & (q_idx_safe >= kv_idx_safe)

            # SR: Summary ← Regular
            # S_i 只看 bar i 的 Regular tokens
            # 注意: bar_ids 已填充到 seq_len，但仍需 clamp 以确保安全
            reg_kv_idx = (kv_idx_safe - sum_len).clamp(0, bar_ids_len - 1)
            reg_bar = bar_ids[reg_kv_idx].clamp(0, sum_len - 1)  # 处理负数 bar_id (全局 token)
            sr_mask = is_kv_regular & (reg_bar == q_idx_safe)

            return ss_mask | sr_mask

        return mask_mod

    def create_updating_mask_mod(
        self,
        bar_ids: torch.Tensor,           # (seq_len,) 每个 Regular token 的 bar ID
        chord_ids: torch.Tensor,         # (seq_len,) 每个 Regular token 的 chord ID
        instrument_ids: torch.Tensor,    # (seq_len,) 每个 Regular token 的乐器 ID
        num_bars: int,
        seq_len: int,
        token_type_ids: Optional[torch.Tensor] = None,
        note_ids: Optional[torch.Tensor] = None,
    ) -> Callable:
        """
        创建 Updating 阶段的 mask_mod (RS + RR)

        Query: Regular tokens [0, seq_len)
        Key/Value: [Summary K2/V2; Regular K/V] = [0, num_bars + seq_len)

        RS: R_i attend to S_j where bar(R_i) > j (只看已完成 bar 的 Summary)
        RR: 使用 FlexFCAttentionMask 的逻辑
        """
        bar_ids = bar_ids.contiguous()
        chord_ids = chord_ids.contiguous()
        instrument_ids = instrument_ids.contiguous()

        sum_len = max(num_bars, 1)  # 确保至少为 1
        reg_len = max(seq_len, 1)   # 确保至少为 1
        total_kv_len = sum_len + reg_len
        rs_look_back = self.rs_look_back

        # 获取各 tensor 的长度 (已在 create_block_masks 中填充到 seq_len)
        bar_ids_len = bar_ids.size(0)
        chord_ids_len = chord_ids.size(0)
        instrument_ids_len = instrument_ids.size(0)

        # Token 类型稀疏
        use_token_type_sparsity = token_type_ids is not None and note_ids is not None
        token_type_ids_len = 0
        note_ids_len = 0
        if use_token_type_sparsity:
            token_type_ids = token_type_ids.contiguous()
            note_ids = note_ids.contiguous()
            token_type_ids_len = token_type_ids.size(0)
            note_ids_len = note_ids.size(0)
            type_visibility = TOKEN_TYPE_VISIBILITY.to(bar_ids.device)

        # RR mask 参数
        cross_inst_full_range = self.rr_mask_generator.cross_inst_full_range
        cross_inst_far_offsets = self.rr_mask_generator.cross_inst_far_offsets

        def mask_mod(b, h, q_idx, kv_idx):
            # 安全处理越界索引 - 使用实际 tensor 长度来 clamp
            q_idx_safe = q_idx.clamp(0, bar_ids_len - 1)
            kv_idx_safe = kv_idx.clamp(0, total_kv_len - 1)

            # 判断 kv 是 Summary 还是 Regular
            is_kv_summary = kv_idx_safe < sum_len
            is_kv_regular = ~is_kv_summary

            # Query 的 bar (处理负数 bar_id - 全局 token 映射到 bar 0)
            q_bar = bar_ids[q_idx_safe].clamp(0, sum_len - 1)

            # === RS: Regular → Summary (K2, V2) ===
            # R 只能看已完成 bar 的 Summary (bar(R) > S_idx)
            if rs_look_back == -1:
                # 看所有已完成 bar
                rs_mask = is_kv_summary & (q_bar > kv_idx_safe)
            else:
                # 只看最近 N 个 bar
                start_bar = (q_bar - rs_look_back).clamp(0)
                rs_mask = is_kv_summary & (kv_idx_safe >= start_bar) & (kv_idx_safe < q_bar)

            # === RR: Regular → Regular ===
            # 使用 FlexFCAttentionMask 的逻辑

            # Regular K/V 的实际索引 - 使用实际 tensor 长度来 clamp
            reg_kv_idx = (kv_idx_safe - sum_len).clamp(0, bar_ids_len - 1)

            # 因果性
            rr_causal = q_idx_safe >= reg_kv_idx

            # 获取 token 信息 - 使用安全索引
            q_chord = chord_ids[q_idx_safe.clamp(0, chord_ids_len - 1)]
            k_chord = chord_ids[reg_kv_idx.clamp(0, chord_ids_len - 1)]
            q_inst = instrument_ids[q_idx_safe.clamp(0, instrument_ids_len - 1)]
            k_inst = instrument_ids[reg_kv_idx.clamp(0, instrument_ids_len - 1)]

            # 全局 token
            is_global_k = (k_chord == -1) | (k_inst == 129)

            # 同乐器：全连接
            same_inst = (q_inst == k_inst) & (q_inst < 129) & (k_inst < 129)

            # 跨乐器近距离
            chord_diff = q_chord - k_chord
            diff_inst = (q_inst != k_inst) & (q_inst < 129) & (k_inst < 129)
            cross_near = diff_inst & (chord_diff >= 0) & (chord_diff <= cross_inst_full_range)

            # 跨乐器远距离
            cross_far = torch.zeros_like(rr_causal)
            for offset in cross_inst_far_offsets:
                cross_far = cross_far | (diff_inst & (chord_diff == offset))

            # Bar 级掩码
            bar_mask = is_global_k | same_inst | cross_near | cross_far

            # Token 类型级稀疏
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

            return rs_mask | rr_mask

        return mask_mod

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
    ) -> Tuple:
        """
        创建 Summarize 和 Updating 两个阶段的 BlockMask

        Returns:
            (summarize_block_mask, updating_block_mask)
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

        # 确保 tensor 长度与 seq_len 一致，避免 create_block_mask 遍历时越界
        # 这是 CUDA 上 indexSelectLargeIndex 错误的根本原因
        if bar_ids_flat.size(0) < seq_len:
            pad_len = seq_len - bar_ids_flat.size(0)
            # 用最后一个值填充 (或 0)
            bar_ids_flat = F.pad(bar_ids_flat, (0, pad_len), value=bar_ids_flat[-1].item() if bar_ids_flat.numel() > 0 else 0)
            chord_ids_flat = F.pad(chord_ids_flat, (0, pad_len), value=chord_ids_flat[-1].item() if chord_ids_flat.numel() > 0 else 0)
            instrument_ids_flat = F.pad(instrument_ids_flat, (0, pad_len), value=129)  # 129 = 全局 token

        # Token 类型
        token_type_ids_flat = None
        note_ids_flat = None
        if token_type_ids is not None:
            token_type_ids_flat = token_type_ids[0] if token_type_ids.dim() == 2 else token_type_ids
            token_type_ids_flat = token_type_ids_flat.to(device)
            if token_type_ids_flat.size(0) < seq_len:
                pad_len = seq_len - token_type_ids_flat.size(0)
                token_type_ids_flat = F.pad(token_type_ids_flat, (0, pad_len), value=-1)
        if note_ids is not None:
            note_ids_flat = note_ids[0] if note_ids.dim() == 2 else note_ids
            note_ids_flat = note_ids_flat.to(device)
            if note_ids_flat.size(0) < seq_len:
                pad_len = seq_len - note_ids_flat.size(0)
                note_ids_flat = F.pad(note_ids_flat, (0, pad_len), value=-1)

        # Summarize 阶段: Q=Summary, KV=[Summary; Regular]
        summarize_mask_mod = self.create_summarize_mask_mod(
            bar_ids_flat, num_bars, seq_len
        )
        summarize_block_mask = create_block_mask(
            summarize_mask_mod,
            B=batch_size,
            H=None,
            Q_LEN=num_bars,
            KV_LEN=num_bars + seq_len,
            device=device,
            _compile=True,
        )

        # Updating 阶段: Q=Regular, KV=[Summary; Regular]
        updating_mask_mod = self.create_updating_mask_mod(
            bar_ids_flat, chord_ids_flat, instrument_ids_flat,
            num_bars, seq_len,
            token_type_ids=token_type_ids_flat,
            note_ids=note_ids_flat,
        )
        updating_block_mask = create_block_mask(
            updating_mask_mod,
            B=batch_size,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=num_bars + seq_len,
            device=device,
            _compile=True,
        )

        return summarize_block_mask, updating_block_mask


class FlexSummaryAttention(nn.Module):
    """
    FlexAttention with Summary Token

    实现两阶段注意力：
    1. Summarize: Summary 聚合 Summary + Regular
    2. Updating: Regular 聚合 Summary(K2,V2) + Regular
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        max_seq_len: int = 32768,
        use_rope: bool = True,
    ):
        super().__init__()

        check_flex_attention_available()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_rope = use_rope
        self.dropout_p = dropout

        assert embed_dim % num_heads == 0

        # Summary 投影
        self.sum_q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.sum_k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.sum_v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.sum_out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Regular 投影
        self.reg_q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.reg_k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.reg_v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.reg_out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # K2, V2 二次投影 (Summary 输出 → RS 的 K/V)
        self.sum_k2_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.sum_v2_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # RoPE
        if use_rope:
            self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len)
        else:
            self.rope = None

    def forward(
        self,
        sum_x: torch.Tensor,           # (batch, num_bars, embed_dim)
        reg_x: torch.Tensor,           # (batch, seq_len, embed_dim)
        summarize_block_mask,          # Summarize 阶段的 BlockMask
        updating_block_mask,           # Updating 阶段的 BlockMask
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            sum_x: Summary token 嵌入 (batch, num_bars, embed_dim)
            reg_x: Regular token 嵌入 (batch, seq_len, embed_dim)
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
        # Summary Q, K, V
        sum_q = self.sum_q_proj(sum_x)
        sum_k = self.sum_k_proj(sum_x)
        sum_v = self.sum_v_proj(sum_x)

        # Regular Q, K, V
        reg_q = self.reg_q_proj(reg_x)
        reg_k = self.reg_k_proj(reg_x)
        reg_v = self.reg_v_proj(reg_x)

        # 重塑为多头: (batch, num_heads, len, head_dim)
        sum_q = sum_q.view(batch_size, sum_len, self.num_heads, self.head_dim).transpose(1, 2)
        sum_k = sum_k.view(batch_size, sum_len, self.num_heads, self.head_dim).transpose(1, 2)
        sum_v = sum_v.view(batch_size, sum_len, self.num_heads, self.head_dim).transpose(1, 2)

        reg_q = reg_q.view(batch_size, reg_len, self.num_heads, self.head_dim).transpose(1, 2)
        reg_k = reg_k.view(batch_size, reg_len, self.num_heads, self.head_dim).transpose(1, 2)
        reg_v = reg_v.view(batch_size, reg_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 应用 RoPE
        if self.rope is not None:
            # Summary 使用 bar 位置
            sum_cos, sum_sin = self.rope(sum_x, sum_len)
            sum_q, sum_k = apply_rotary_pos_emb(sum_q, sum_k, sum_cos, sum_sin)

            # Regular 使用 token 位置 (从 sum_len 开始，避免冲突)
            total_len = sum_len + reg_len
            cos, sin = self.rope(reg_x, total_len)
            reg_cos = cos[:, :, sum_len:, :]
            reg_sin = sin[:, :, sum_len:, :]
            reg_q, reg_k = apply_rotary_pos_emb(reg_q, reg_k, reg_cos, reg_sin)

        # === 阶段 1: Summarize (SS + SR) ===
        # FlexAttention 在混合精度下的反向传播要求所有 tensor 保持一致的 dtype
        # 禁用 autocast，使用 FP32 计算 attention（更稳定）
        with torch.autocast(device_type='cuda', enabled=False):
            # 转换为 FP32
            sum_q_fp32 = sum_q.float()
            sum_k_fp32 = sum_k.float()
            sum_v_fp32 = sum_v.float()
            reg_k_fp32 = reg_k.float()
            reg_v_fp32 = reg_v.float()

            # 拼接 K, V: [Summary; Regular]
            cat_k_sum = torch.cat([sum_k_fp32, reg_k_fp32], dim=2)
            cat_v_sum = torch.cat([sum_v_fp32, reg_v_fp32], dim=2)

            # FlexAttention: Q=Summary, KV=[Summary; Regular]
            sum_attn_out = flex_attention(sum_q_fp32, cat_k_sum, cat_v_sum, block_mask=summarize_block_mask)

            # 转回原始 dtype
            sum_attn_out = sum_attn_out.to(sum_q.dtype)

        # 重塑
        sum_attn_out = sum_attn_out.transpose(1, 2).contiguous().view(batch_size, sum_len, self.embed_dim)

        # === 阶段 2: K2, V2 二次投影 ===
        sum_k2 = self.sum_k2_proj(sum_attn_out)
        sum_v2 = self.sum_v2_proj(sum_attn_out)

        # 重塑为多头
        sum_k2 = sum_k2.view(batch_size, sum_len, self.num_heads, self.head_dim).transpose(1, 2)
        sum_v2 = sum_v2.view(batch_size, sum_len, self.num_heads, self.head_dim).transpose(1, 2)

        # === 阶段 3: Updating (RS + RR) ===
        # FlexAttention 在混合精度下的反向传播要求所有 tensor 保持一致的 dtype
        # 禁用 autocast，使用 FP32 计算 attention（更稳定）
        with torch.autocast(device_type='cuda', enabled=False):
            # 转换为 FP32
            reg_q_fp32 = reg_q.float()
            sum_k2_fp32 = sum_k2.float()
            sum_v2_fp32 = sum_v2.float()
            reg_k_fp32 = reg_k.float()
            reg_v_fp32 = reg_v.float()

            # 拼接 K, V: [Summary K2/V2; Regular K/V]
            cat_k_reg = torch.cat([sum_k2_fp32, reg_k_fp32], dim=2)
            cat_v_reg = torch.cat([sum_v2_fp32, reg_v_fp32], dim=2)

            # FlexAttention: Q=Regular, KV=[Summary K2; Regular]
            reg_attn_out = flex_attention(reg_q_fp32, cat_k_reg, cat_v_reg, block_mask=updating_block_mask)

            # 转回原始 dtype
            reg_attn_out = reg_attn_out.to(reg_q.dtype)

        # 重塑
        reg_attn_out = reg_attn_out.transpose(1, 2).contiguous().view(batch_size, reg_len, self.embed_dim)

        # === 输出投影 ===
        sum_output = self.sum_out_proj(sum_attn_out)
        reg_output = self.reg_out_proj(reg_attn_out)

        return sum_output, reg_output


class FlexSummaryAttentionBlock(nn.Module):
    """
    FlexAttention Summary Block (Pre-LN + RoPE + SwiGLU)

    整合 FlexSummaryAttention 和 FFN：
    - Summary 和 Regular 各自有独立的 FFN
    - Pre-LN 架构
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
        use_rope: bool = True,
        share_ffn: bool = False,
    ):
        super().__init__()

        # Summary Attention
        self.attention = FlexSummaryAttention(
            embed_dim, num_heads, dropout,
            max_seq_len=max_seq_len, use_rope=use_rope,
        )

        # Layer Norm
        self.sum_attn_norm = nn.LayerNorm(embed_dim)
        self.reg_attn_norm = nn.LayerNorm(embed_dim)

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

        self.sum_ffn_norm = nn.LayerNorm(embed_dim)
        self.reg_ffn_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

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

    def forward(
        self,
        sum_x: torch.Tensor,
        reg_x: torch.Tensor,
        summarize_block_mask,
        updating_block_mask,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            sum_x: Summary token 嵌入 (batch, num_bars, embed_dim)
            reg_x: Regular token 嵌入 (batch, seq_len, embed_dim)
            summarize_block_mask: SS + SR 的 BlockMask
            updating_block_mask: RS + RR 的 BlockMask

        Returns:
            sum_output: (batch, num_bars, embed_dim)
            reg_output: (batch, seq_len, embed_dim)
        """
        # === Attention with residual (Pre-LN) ===
        sum_residual = sum_x
        reg_residual = reg_x

        sum_x = self.sum_attn_norm(sum_x)
        reg_x = self.reg_attn_norm(reg_x)

        sum_x, reg_x = self.attention(
            sum_x, reg_x,
            summarize_block_mask, updating_block_mask
        )

        sum_x = self.dropout(sum_x)
        reg_x = self.dropout(reg_x)

        sum_x = sum_residual + sum_x
        reg_x = reg_residual + reg_x

        # === FFN with residual (Pre-LN) ===
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
