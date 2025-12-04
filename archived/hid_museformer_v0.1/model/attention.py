#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FC-Attention 模块 - 适配 HID 编码的 Fine-Coarse Attention

实现 MuseFormer 论文 (NeurIPS 2022) 的 FC-Attention:
https://arxiv.org/abs/2210.10349

核心思想：
1. 当前小节内：全连接（跨乐器，同一小节内所有 token 互相可见）
2. Fine-grained：能看前 1, 2, 4, 8, 12, 16, 24, 32 小节的所有 token（跨乐器）
3. 因果性：只能看过去的 token

注意：HID 编码按乐器分组，但 FC-Attention 基于小节/和弦，不做乐器隔离。
同一时间段（同一小节）的不同乐器音符应该能互相看到。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class RotaryPositionEmbedding(nn.Module):
    """
    RoPE (Rotary Position Embedding)

    比可学习位置编码更好的长序列外推能力
    参考: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, dim: int, max_seq_len: int = 32768, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # 预计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # 预计算 cos/sin 缓存
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """构建 cos/sin 缓存"""
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入张量 (用于获取 device)
            seq_len: 序列长度

        Returns:
            cos, sin: (1, 1, seq_len, dim)
        """
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len

        return (
            self.cos_cached[:, :, :seq_len, :].to(x.device),
            self.sin_cached[:, :, :seq_len, :].to(x.device)
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """将张量的后半部分旋转到前半部分"""
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    应用 RoPE 到 Q 和 K

    Args:
        q, k: (batch, num_heads, seq_len, head_dim)
        cos, sin: (1, 1, seq_len, head_dim)
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class FCAttentionMask:
    """
    FC-Attention 掩码生成器 (基于跨乐器相似度分析优化)

    为 HID 编码生成注意力掩码：

    HID 编码结构:
    [BOS] [BPM] [TS] [#P0] [SEP] [Chord] T P T P ... [Chord] T P ... [#P1] [SEP] ...

    掩码规则 (基于 450K MIDI 统计分析):

    1. 全局 token (BOS, BPM, TS, 乐器标记, SEP): 所有 token 都能看到

    2. 同乐器注意力 (Same Instrument):
       - 全连接！即使最远的 offset=29 相似度 (0.389) 仍高于跨乐器最高 (0.263)
       - 音乐主题回顾、动机发展可能出现在任意位置

    3. 跨乐器近距离 (Cross Instrument, offset ≤ 2):
       - 2 小节内全连接（含同小节）
       - 和声协调 + 过渡衔接的基础需求

    4. 跨乐器远距离 (Cross Instrument, offset > 2):
       - 只看 offset=4（乐段边界参考）
       - 统计显示跨乐器相似度平坦 (~0.25)，无周期性

    5. 因果性: 只能看过去的 token

    统计依据 (来自 cross_instrument_similarity.py):
    - 同乐器: 0.389-0.528，有 4 小节周期性
    - 跨乐器: 0.23-0.26，平坦无周期
    """

    # 跨乐器远距离: 只看 4 小节前（乐段边界）
    CROSS_INST_FAR_OFFSETS = (-4,)
    # 跨乐器近距离全连接范围
    CROSS_INST_FULL_RANGE = 2  # offset <= 2 全连接

    # Token 类型可见性矩阵（不同音符之间）
    # 基于 NMI 统计分析：T→T (0.193), P→P (0.144), V→V (0.152) 必看
    # T↔P (~0.09) 保留，D 相关 (< 0.07) 不看
    # True = 可见, False = 不可见
    # 行 = Query 类型, 列 = Key 类型
    # 类型: 0=T, 1=P, 2=D(L), 3=V
    TOKEN_TYPE_VISIBILITY = torch.tensor([
        # Key:  T      P      D      V
        [True,  True,  False, False],  # Query T: 看 T, P
        [True,  True,  False, False],  # Query P: 看 T, P
        [False, False, False, False],  # Query D: 不看其他音符的任何 token
        [False, False, False, True ],  # Query V: 只看 V
    ], dtype=torch.bool)

    def __init__(
        self,
        same_inst_full_connection: bool = True,  # 同乐器全连接
        cross_inst_full_range: int = 2,  # 跨乐器全连接范围 (offset <= N)
        cross_inst_far_offsets: Tuple[int, ...] = None,  # 跨乐器远距离偏移
    ):
        """
        Args:
            same_inst_full_connection: 同乐器是否全连接 (默认 True)
            cross_inst_full_range: 跨乐器全连接范围，offset <= N 时全连接 (默认 2)
            cross_inst_far_offsets: 跨乐器远距离看的小节偏移 (负数表示往前看)
                                   默认: (-4,) 只看 4 小节前
        """
        self.same_inst_full_connection = same_inst_full_connection
        self.cross_inst_full_range = cross_inst_full_range
        self.cross_inst_far_offsets = cross_inst_far_offsets or self.CROSS_INST_FAR_OFFSETS

    def create_mask(
        self,
        chord_ids: torch.Tensor,
        instrument_ids: torch.Tensor,
        seq_len: int,
        device: torch.device,
        is_chord_token: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        note_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        创建注意力掩码

        Args:
            chord_ids: (batch, seq_len) 每个 token 的和弦/小节索引
                       -1 表示全局 token (BOS, BPM, TS, 乐器标记, SEP)
            instrument_ids: (batch, seq_len) 每个 token 的乐器 ID（用于识别全局 token）
                       129 表示全局 token
            seq_len: 序列长度
            device: 设备
            is_chord_token: (batch, seq_len) 是否是和弦 token (可选)
                           用于让同一小节内的 token 强制看到该小节的和弦 token
            token_type_ids: (batch, seq_len) Token 类型 (可选)
                           0=T, 1=P, 2=D(L), 3=V, -1=全局/其他
            note_ids: (batch, seq_len) 音符 ID (可选)
                     同一音符的 T, P, L, V token 有相同 ID，-1 表示非音符 token

        Returns:
            mask: (batch, seq_len, seq_len) True = 可注意, False = 不可注意
        """
        batch_size = chord_ids.size(0)

        # 基础因果掩码 (只能看过去)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()

        # 初始化掩码
        mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool, device=device)

        for b in range(batch_size):
            batch_chord_ids = chord_ids[b]
            batch_inst_ids = instrument_ids[b]

            # 1. 全局 token (chord_id == -1 或 instrument_id == 129)
            is_global = (batch_chord_ids == -1) | (batch_inst_ids == 129)
            # 所有 token 都能看到全局 token
            global_visible = is_global.unsqueeze(0).expand(seq_len, -1)

            # 2. 同乐器掩码 (Same Instrument)
            # 同乐器全连接，因为即使远距离相似度 (0.389) 仍高于跨乐器 (0.263)
            same_inst = (batch_inst_ids.unsqueeze(0) == batch_inst_ids.unsqueeze(1))
            same_inst = same_inst & (batch_inst_ids.unsqueeze(0) < 129)  # query 非全局
            same_inst = same_inst & (batch_inst_ids.unsqueeze(1) < 129)  # key 非全局

            if self.same_inst_full_connection:
                # 同乐器全连接
                same_inst_mask = same_inst
            else:
                # 可选：同乐器也只看特定偏移（未来可扩展）
                same_inst_mask = same_inst

            # 3. 跨乐器掩码 (Cross Instrument)
            diff_inst = (batch_inst_ids.unsqueeze(0) != batch_inst_ids.unsqueeze(1))
            diff_inst = diff_inst & (batch_inst_ids.unsqueeze(0) < 129)  # query 非全局
            diff_inst = diff_inst & (batch_inst_ids.unsqueeze(1) < 129)  # key 非全局

            # 3.1 跨乐器近距离：offset <= cross_inst_full_range 全连接
            # chord_offset = chord_ids[i] - chord_ids[j]
            chord_diff = batch_chord_ids.unsqueeze(0) - batch_chord_ids.unsqueeze(1)  # (seq, seq)
            cross_inst_near = diff_inst & (chord_diff >= 0) & (chord_diff <= self.cross_inst_full_range)

            # 3.2 跨乐器远距离：只看特定偏移 (如 offset=4)
            cross_inst_far = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
            for offset in self.cross_inst_far_offsets:
                # offset 是负数，如 -4 表示看 4 小节前
                target_diff = -offset  # chord_ids[i] - chord_ids[j] == -offset
                cross_inst_far = cross_inst_far | (diff_inst & (chord_diff == target_diff))

            # 4. 和弦锚点：确保每个小节内的 token 能看到该小节的和弦 token
            chord_anchor_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
            if is_chord_token is not None:
                batch_is_chord = is_chord_token[b]
                # 同一小节内的 token 能看到该小节的和弦 token
                same_chord = (batch_chord_ids.unsqueeze(0) == batch_chord_ids.unsqueeze(1))
                same_chord = same_chord & (batch_chord_ids.unsqueeze(0) >= 0)
                same_chord = same_chord & (batch_chord_ids.unsqueeze(1) >= 0)
                # key 是和弦 token
                is_chord_key = batch_is_chord.unsqueeze(0).expand(seq_len, -1)
                chord_anchor_mask = same_chord & is_chord_key

            # 组合掩码
            batch_mask = (
                global_visible |      # 全局 token 对所有可见
                same_inst_mask |      # 同乐器：全连接
                cross_inst_near |     # 跨乐器近距离 (offset<=2)：全连接
                cross_inst_far |      # 跨乐器远距离 (offset=4)：稀疏连接
                chord_anchor_mask     # 和弦锚点
            ) & causal_mask

            # 5. Token 类型级稀疏（可选）
            # 只影响不同音符之间的连接，同音符内保持全连接
            if token_type_ids is not None and note_ids is not None:
                batch_token_types = token_type_ids[b]
                batch_note_ids = note_ids[b]

                # 同音符掩码：同 note_id 的 token 之间全连接
                # note_id = -1 表示非音符 token（全局 token），不参与此过滤
                same_note = (batch_note_ids.unsqueeze(0) == batch_note_ids.unsqueeze(1))
                same_note = same_note & (batch_note_ids.unsqueeze(0) >= 0)
                same_note = same_note & (batch_note_ids.unsqueeze(1) >= 0)

                # 不同音符之间：按 TOKEN_TYPE_VISIBILITY 过滤
                # 将 -1 (全局 token) 映射到 0 以便索引（但稍后会被覆盖）
                q_types = batch_token_types.clamp(min=0)
                k_types = batch_token_types.clamp(min=0)
                type_visible = self.TOKEN_TYPE_VISIBILITY.to(device)[
                    q_types.unsqueeze(1), k_types.unsqueeze(0)
                ]

                # 构建 token 类型掩码：同音符全连接 OR 类型可见
                token_type_mask = same_note | type_visible

                # 全局 token (token_type = -1 或 note_id = -1) 不受此限制
                is_global_token = (batch_token_types == -1) | (batch_note_ids == -1)
                is_global_q = is_global_token.unsqueeze(1)
                is_global_k = is_global_token.unsqueeze(0)
                token_type_mask = token_type_mask | is_global_q | is_global_k

                # 应用 token 类型级掩码
                batch_mask = batch_mask & token_type_mask

            mask[b] = batch_mask

        return mask


class MultiHeadFCAttention(nn.Module):
    """
    多头 FC-Attention with RoPE

    支持：
    - 标准注意力
    - 稀疏注意力掩码
    - RoPE 位置编码
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        max_seq_len: int = 32768,
        use_rope: bool = True,
        use_flash_attention: bool = True,  # 新增: Flash Attention 开关
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.use_rope = use_rope
        self.dropout_p = dropout

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

        # RoPE
        if use_rope:
            self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len)
        else:
            self.rope = None

        # Flash Attention 检测
        self.use_flash_attention = use_flash_attention and hasattr(F, 'scaled_dot_product_attention')
        if self.use_flash_attention:
            print(f"[MultiHeadFCAttention] Using Flash Attention (PyTorch SDPA)")
        else:
            print(f"[MultiHeadFCAttention] Using standard attention (高显存警告!)")

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            attention_mask: (batch, seq_len, seq_len) or (seq_len, seq_len)
            key_padding_mask: (batch, seq_len) True = pad

        Returns:
            output: (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape

        # 投影
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 重塑为多头
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 应用 RoPE
        if self.rope is not None:
            cos, sin = self.rope(x, seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Flash Attention 路径 (显存优化)
        if self.use_flash_attention:
            # 构建 attention mask for SDPA
            # SDPA 需要 (batch, heads, q_len, k_len) 或 (q_len, k_len) 格式
            # True = 不参与注意力, False = 参与注意力 (与我们相反)
            attn_mask = None

            if attention_mask is not None:
                # 我们的 mask: True = 可注意, False = 屏蔽
                # SDPA 需要: float mask, -inf = 屏蔽
                if attention_mask.dim() == 2:
                    attn_mask = attention_mask.unsqueeze(0).unsqueeze(0)
                elif attention_mask.dim() == 3:
                    attn_mask = attention_mask.unsqueeze(1)
                # 转换: True -> 0, False -> -inf
                attn_mask = torch.where(attn_mask, torch.zeros_like(attn_mask, dtype=q.dtype),
                                        torch.full_like(attn_mask, float('-inf'), dtype=q.dtype))

            # 合并 key_padding_mask
            if key_padding_mask is not None:
                # key_padding_mask: (batch, seq_len), True = pad
                padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
                padding_mask = torch.where(padding_mask,
                                          torch.full_like(padding_mask, float('-inf'), dtype=q.dtype),
                                          torch.zeros_like(padding_mask, dtype=q.dtype))
                if attn_mask is not None:
                    attn_mask = attn_mask + padding_mask
                else:
                    attn_mask = padding_mask

            # 使用 PyTorch 的 scaled_dot_product_attention (自动选择 Flash/Memory-Efficient)
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False,  # 我们用自定义 mask
            )
        else:
            # 标准 Attention 路径 (高显存)
            # 注意力分数
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

            # 应用注意力掩码
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
                elif attention_mask.dim() == 3:
                    attention_mask = attention_mask.unsqueeze(1)

                # True -> 可注意, False -> 屏蔽
                attn_weights = attn_weights.masked_fill(~attention_mask, float('-inf'))

            # Key padding mask
            if key_padding_mask is not None:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float('-inf')
                )

            # Softmax
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # 加权求和
            attn_output = torch.matmul(attn_weights, v)

        # 重塑回原始维度
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # 输出投影
        output = self.out_proj(attn_output)

        return output


class FCAttentionBlock(nn.Module):
    """
    FC-Attention Block (Pre-LN + RoPE)

    包含：
    - Multi-Head FC-Attention with RoPE
    - Feed-Forward Network (支持 SwiGLU)
    - Layer Normalization (Pre-LN)
    - Residual Connection
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        activation: str = 'swiglu',  # 'gelu', 'relu', 'swiglu'
        max_seq_len: int = 32768,
        use_rope: bool = True,
    ):
        super().__init__()

        self.attention = MultiHeadFCAttention(
            embed_dim, num_heads, dropout,
            max_seq_len=max_seq_len, use_rope=use_rope
        )
        self.attention_norm = nn.LayerNorm(embed_dim)

        # FFN: 支持 SwiGLU (LLaMA 风格) 或 GELU
        if activation == 'swiglu':
            # SwiGLU: 需要两个门控投影
            self.ffn_gate = nn.Linear(embed_dim, ffn_dim, bias=False)
            self.ffn_up = nn.Linear(embed_dim, ffn_dim, bias=False)
            self.ffn_down = nn.Linear(ffn_dim, embed_dim, bias=False)
            self.use_swiglu = True
        else:
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, embed_dim),
                nn.Dropout(dropout),
            )
            self.use_swiglu = False

        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            attention_mask: (batch, seq_len, seq_len)
            key_padding_mask: (batch, seq_len)

        Returns:
            output: (batch, seq_len, embed_dim)
        """
        # Self-Attention with residual (Pre-LN)
        residual = x
        x = self.attention_norm(x)
        x = self.attention(x, attention_mask, key_padding_mask)
        x = self.dropout(x)
        x = residual + x

        # FFN with residual (Pre-LN)
        residual = x
        x = self.ffn_norm(x)

        if self.use_swiglu:
            # SwiGLU: gate * silu(up)
            x = self.ffn_down(F.silu(self.ffn_gate(x)) * self.ffn_up(x))
            x = self.dropout(x)
        else:
            x = self.ffn(x)

        x = residual + x

        return x


class SummaryAttentionMask:
    """
    Summary Token 注意力掩码生成器

    生成 ss, sr, rs, rr 四种掩码，用于 FC-Attention with Summary Token。

    四种注意力块：
    - ss (Summary → Summary): S 之间互相看（因果），粗粒度跨 bar 交互
    - sr (Summary ← Regular): S 聚合同 bar 内的 R，信息压缩
    - rs (Regular → Summary): R 读取已完成 bar 的 S，获取远距离上下文
    - rr (Regular → Regular): R 之间互相看，细粒度近距离交互
    """

    def __init__(
        self,
        rr_mask_generator: Optional[FCAttentionMask] = None,
        rs_look_back: int = -1,  # R 能看多少个 bar 前的 S，-1 表示全部
    ):
        """
        Args:
            rr_mask_generator: 用于生成 rr 掩码的 FCAttentionMask 实例
                              如果为 None，则创建默认实例
            rs_look_back: Regular 能看多少个 bar 前的 Summary
                         -1 表示所有已完成的 bar（因果）
                         正数 N 表示只看最近 N 个 bar 的 Summary
        """
        self.rr_mask_generator = rr_mask_generator or FCAttentionMask()
        self.rs_look_back = rs_look_back

    def create_masks(
        self,
        bar_ids: torch.Tensor,  # (batch, reg_len) 每个 Regular token 的 bar ID
        instrument_ids: torch.Tensor,  # (batch, reg_len) 每个 Regular token 的乐器 ID
        num_bars: int,  # bar 数量（也是 Summary token 数量）
        device: torch.device,
        is_chord_token: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        note_ids: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        创建 ss, sr, rs, rr 四种掩码

        Args:
            bar_ids: (batch, reg_len) 每个 Regular token 的 bar 索引
            instrument_ids: (batch, reg_len) 每个 Regular token 的乐器 ID
            num_bars: bar 数量，也是 Summary token 数量
            device: 设备
            is_chord_token: (batch, reg_len) 是否是和弦 token（用于 rr）
            token_type_ids: (batch, reg_len) Token 类型（用于 rr）
            note_ids: (batch, reg_len) 音符 ID（用于 rr）

        Returns:
            dict: {
                'ss': (batch, num_bars, num_bars),      # Summary → Summary
                'sr': (batch, num_bars, reg_len),       # Summary ← Regular
                'rs': (batch, reg_len, num_bars),       # Regular → Summary
                'rr': (batch, reg_len, reg_len),        # Regular → Regular
            }
            True = 可注意, False = 不可注意
        """
        batch_size = bar_ids.size(0)
        reg_len = bar_ids.size(1)
        sum_len = num_bars

        # 初始化掩码
        ss_mask = torch.zeros(batch_size, sum_len, sum_len, dtype=torch.bool, device=device)
        sr_mask = torch.zeros(batch_size, sum_len, reg_len, dtype=torch.bool, device=device)
        rs_mask = torch.zeros(batch_size, reg_len, sum_len, dtype=torch.bool, device=device)

        # === SS 掩码：Summary 之间因果注意力 ===
        # S_i 可以看 S_j 当且仅当 j <= i（因果）
        ss_causal = torch.tril(torch.ones(sum_len, sum_len, device=device)).bool()
        ss_mask = ss_causal.unsqueeze(0).expand(batch_size, -1, -1)

        for b in range(batch_size):
            batch_bar_ids = bar_ids[b]

            # === SR 掩码：Summary 聚合同 bar 的 Regular ===
            # S_i 可以看 R_j 当且仅当 R_j 属于 bar i
            for bar_idx in range(num_bars):
                # 找到属于这个 bar 的 Regular token
                in_bar = (batch_bar_ids == bar_idx)
                sr_mask[b, bar_idx] = in_bar

            # === RS 掩码：Regular 读取已完成 bar 的 Summary ===
            # R_j 可以看 S_i 当且仅当 bar(R_j) > i（只看已完成 bar 的 Summary）
            # 注意：如果 R_j 在 bar k，它只能看 S_0, S_1, ..., S_{k-1}
            for j in range(reg_len):
                r_bar = batch_bar_ids[j].item()
                if r_bar >= 0:
                    if self.rs_look_back == -1:
                        # 看所有已完成 bar 的 Summary
                        rs_mask[b, j, :r_bar] = True
                    else:
                        # 只看最近 N 个 bar 的 Summary
                        start_bar = max(0, r_bar - self.rs_look_back)
                        rs_mask[b, j, start_bar:r_bar] = True

        # === RR 掩码：使用现有的 FCAttentionMask ===
        rr_mask = self.rr_mask_generator.create_mask(
            bar_ids, instrument_ids, reg_len, device,
            is_chord_token=is_chord_token,
            token_type_ids=token_type_ids,
            note_ids=note_ids,
        )

        return {
            'ss': ss_mask,
            'sr': sr_mask,
            'rs': rs_mask,
            'rr': rr_mask,
        }


class SummaryAttention(nn.Module):
    """
    FC-Attention with Summary Token

    实现 MuseFormer 的 Summary Token 机制：
    - ss: Summary → Summary（粗粒度跨 bar 交互）
    - sr: Summary ← Regular（信息压缩）
    - rs: Regular → Summary（获取远距离上下文）
    - rr: Regular → Regular（细粒度近距离交互）

    信息流：
    1. Summarize 阶段：SS + SR → sum_x2
    2. 二次投影：sum_x2 → K2, V2
    3. Updating 阶段：RS + RR → reg_output
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        max_seq_len: int = 32768,
        use_rope: bool = True,
        share_kv_proj: bool = False,  # Summary 和 Regular 是否共享 K、V 投影
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.use_rope = use_rope
        self.share_kv_proj = share_kv_proj

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Summary Token 投影
        self.sum_q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.sum_k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.sum_v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.sum_out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Regular Token 投影
        self.reg_q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        if share_kv_proj:
            # 共享 K、V 投影
            self.reg_k_proj = self.sum_k_proj
            self.reg_v_proj = self.sum_v_proj
        else:
            self.reg_k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.reg_v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.reg_out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # K2、V2 二次投影（SR 后用于 RS）
        self.sum_k2_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.sum_v2_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.dropout_p = dropout

        # RoPE
        if use_rope:
            self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len)
        else:
            self.rope = None

        # Flash Attention 检测
        self.use_flash_attention = hasattr(F, 'scaled_dot_product_attention')

    def _attention(
        self,
        q: torch.Tensor,  # (batch, num_heads, q_len, head_dim)
        k: torch.Tensor,  # (batch, num_heads, k_len, head_dim)
        v: torch.Tensor,  # (batch, num_heads, k_len, head_dim)
        mask: torch.Tensor,  # (batch, q_len, k_len) or (batch, 1, q_len, k_len)
    ) -> torch.Tensor:
        """
        核心注意力计算 (支持 Flash Attention)

        Returns:
            output: (batch, num_heads, q_len, head_dim)
        """
        if self.use_flash_attention:
            # Flash Attention 路径
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch, 1, q_len, k_len)
            # 转换 mask: True -> 0, False -> -inf
            attn_mask = torch.where(mask, torch.zeros_like(mask, dtype=q.dtype),
                                    torch.full_like(mask, float('-inf'), dtype=q.dtype))
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False,
            )
        else:
            # 标准 Attention 路径
            # 注意力分数
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

            # 应用掩码
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch, 1, q_len, k_len)
            attn_weights = attn_weights.masked_fill(~mask, float('-inf'))

            # Softmax
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # 加权求和
            output = torch.matmul(attn_weights, v)
        return output

    def forward(
        self,
        sum_x: torch.Tensor,  # (batch, sum_len, embed_dim)
        reg_x: torch.Tensor,  # (batch, reg_len, embed_dim)
        ss_mask: torch.Tensor,  # (batch, sum_len, sum_len)
        sr_mask: torch.Tensor,  # (batch, sum_len, reg_len)
        rs_mask: torch.Tensor,  # (batch, reg_len, sum_len)
        rr_mask: torch.Tensor,  # (batch, reg_len, reg_len)
        sum_positions: Optional[torch.Tensor] = None,  # (batch, sum_len) Summary 的位置
        reg_positions: Optional[torch.Tensor] = None,  # (batch, reg_len) Regular 的位置
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            sum_x: Summary token 嵌入 (batch, sum_len, embed_dim)
            reg_x: Regular token 嵌入 (batch, reg_len, embed_dim)
            ss_mask: Summary → Summary 掩码
            sr_mask: Summary ← Regular 掩码
            rs_mask: Regular → Summary 掩码
            rr_mask: Regular → Regular 掩码
            sum_positions: Summary token 的位置（用于 RoPE）
            reg_positions: Regular token 的位置（用于 RoPE）

        Returns:
            sum_output: (batch, sum_len, embed_dim)
            reg_output: (batch, reg_len, embed_dim)
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

        # 重塑为多头
        sum_q = sum_q.view(batch_size, sum_len, self.num_heads, self.head_dim).transpose(1, 2)
        sum_k = sum_k.view(batch_size, sum_len, self.num_heads, self.head_dim).transpose(1, 2)
        sum_v = sum_v.view(batch_size, sum_len, self.num_heads, self.head_dim).transpose(1, 2)

        reg_q = reg_q.view(batch_size, reg_len, self.num_heads, self.head_dim).transpose(1, 2)
        reg_k = reg_k.view(batch_size, reg_len, self.num_heads, self.head_dim).transpose(1, 2)
        reg_v = reg_v.view(batch_size, reg_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 应用 RoPE（如果使用）
        if self.rope is not None:
            total_len = sum_len + reg_len
            cos, sin = self.rope(sum_x, total_len)

            # Summary 使用前 sum_len 个位置（或自定义位置）
            sum_cos = cos[:, :, :sum_len, :]
            sum_sin = sin[:, :, :sum_len, :]
            sum_q, sum_k = apply_rotary_pos_emb(sum_q, sum_k, sum_cos, sum_sin)

            # Regular 使用后 reg_len 个位置（或自定义位置）
            # 为简化，这里假设 Regular 位置从 sum_len 开始
            # 实际使用时可能需要更精细的位置编码策略
            reg_cos = cos[:, :, sum_len:sum_len+reg_len, :]
            reg_sin = sin[:, :, sum_len:sum_len+reg_len, :]
            reg_q, reg_k = apply_rotary_pos_emb(reg_q, reg_k, reg_cos, reg_sin)

        # === 第一阶段：Summarize (SS + SR) ===
        # Summary 从 Summary 和 Regular 聚合
        # 拼接 K, V
        src_k_sum = torch.cat([sum_k, reg_k], dim=2)  # (batch, heads, sum_len+reg_len, head_dim)
        src_v_sum = torch.cat([sum_v, reg_v], dim=2)

        # 拼接掩码
        sx_mask = torch.cat([ss_mask, sr_mask], dim=-1)  # (batch, sum_len, sum_len+reg_len)

        # 计算 Summary 的注意力输出
        sum_attn_out = self._attention(sum_q, src_k_sum, src_v_sum, sx_mask)
        sum_attn_out = sum_attn_out.transpose(1, 2).contiguous().view(batch_size, sum_len, self.embed_dim)

        # === 第二阶段：K2、V2 二次投影 ===
        sum_k2 = self.sum_k2_proj(sum_attn_out)
        sum_v2 = self.sum_v2_proj(sum_attn_out)

        # 重塑为多头
        sum_k2 = sum_k2.view(batch_size, sum_len, self.num_heads, self.head_dim).transpose(1, 2)
        sum_v2 = sum_v2.view(batch_size, sum_len, self.num_heads, self.head_dim).transpose(1, 2)

        # === 第三阶段：Updating (RS + RR) ===
        # Regular 从 Summary K2 和 Regular 聚合
        src_k_reg = torch.cat([sum_k2, reg_k], dim=2)  # (batch, heads, sum_len+reg_len, head_dim)
        src_v_reg = torch.cat([sum_v2, reg_v], dim=2)

        # 拼接掩码
        rx_mask = torch.cat([rs_mask, rr_mask], dim=-1)  # (batch, reg_len, sum_len+reg_len)

        # 计算 Regular 的注意力输出
        reg_attn_out = self._attention(reg_q, src_k_reg, src_v_reg, rx_mask)
        reg_attn_out = reg_attn_out.transpose(1, 2).contiguous().view(batch_size, reg_len, self.embed_dim)

        # === 输出投影 ===
        sum_output = self.sum_out_proj(sum_attn_out)
        reg_output = self.reg_out_proj(reg_attn_out)

        return sum_output, reg_output


class SummaryAttentionBlock(nn.Module):
    """
    Summary Attention Block (Pre-LN + RoPE)

    整合 SummaryAttention 和 FFN：
    - Summary 和 Regular 各自有独立的 FFN
    - Pre-LN 架构
    - 残差连接
    - SwiGLU 激活（可选）
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
        share_kv_proj: bool = False,
        share_ffn: bool = False,  # Summary 和 Regular 是否共享 FFN
    ):
        super().__init__()

        # Summary Attention
        self.attention = SummaryAttention(
            embed_dim, num_heads, dropout,
            max_seq_len=max_seq_len, use_rope=use_rope,
            share_kv_proj=share_kv_proj,
        )

        # Layer Norm for attention
        self.sum_attn_norm = nn.LayerNorm(embed_dim)
        self.reg_attn_norm = nn.LayerNorm(embed_dim)

        # FFN for Summary
        if activation == 'swiglu':
            self.sum_ffn_gate = nn.Linear(embed_dim, ffn_dim, bias=False)
            self.sum_ffn_up = nn.Linear(embed_dim, ffn_dim, bias=False)
            self.sum_ffn_down = nn.Linear(ffn_dim, embed_dim, bias=False)
            self.use_swiglu = True
        else:
            self.sum_ffn = nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, embed_dim),
                nn.Dropout(dropout),
            )
            self.use_swiglu = False

        # FFN for Regular
        if share_ffn:
            if self.use_swiglu:
                self.reg_ffn_gate = self.sum_ffn_gate
                self.reg_ffn_up = self.sum_ffn_up
                self.reg_ffn_down = self.sum_ffn_down
            else:
                self.reg_ffn = self.sum_ffn
        else:
            if activation == 'swiglu':
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
        self.share_ffn = share_ffn

    def _apply_ffn(self, x: torch.Tensor, is_summary: bool) -> torch.Tensor:
        """应用 FFN"""
        if self.use_swiglu:
            if is_summary or self.share_ffn:
                gate = self.sum_ffn_gate
                up = self.sum_ffn_up
                down = self.sum_ffn_down
            else:
                gate = self.reg_ffn_gate
                up = self.reg_ffn_up
                down = self.reg_ffn_down
            return down(F.silu(gate(x)) * up(x))
        else:
            if is_summary or self.share_ffn:
                return self.sum_ffn(x)
            else:
                return self.reg_ffn(x)

    def forward(
        self,
        sum_x: torch.Tensor,
        reg_x: torch.Tensor,
        ss_mask: torch.Tensor,
        sr_mask: torch.Tensor,
        rs_mask: torch.Tensor,
        rr_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            sum_x: Summary token 嵌入 (batch, sum_len, embed_dim)
            reg_x: Regular token 嵌入 (batch, reg_len, embed_dim)
            ss_mask, sr_mask, rs_mask, rr_mask: 四种掩码

        Returns:
            sum_output: (batch, sum_len, embed_dim)
            reg_output: (batch, reg_len, embed_dim)
        """
        # === Attention with residual (Pre-LN) ===
        sum_residual = sum_x
        reg_residual = reg_x

        sum_x = self.sum_attn_norm(sum_x)
        reg_x = self.reg_attn_norm(reg_x)

        sum_x, reg_x = self.attention(
            sum_x, reg_x,
            ss_mask, sr_mask, rs_mask, rr_mask
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


class SummaryTokenEmbedding(nn.Module):
    """
    Summary Token 嵌入层

    为每个 bar 生成一个 Summary Token 嵌入。
    嵌入可以是：
    1. 可学习的嵌入（每个 bar 位置一个）
    2. 固定的嵌入（所有 bar 共享）
    3. 基于 bar 内容的嵌入（需要额外计算）
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
        else:
            # 固定嵌入：所有 bar 共享
            self.register_buffer(
                'embedding',
                torch.zeros(1, embed_dim)
            )

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
        if self.learnable:
            bar_indices = torch.arange(num_bars, device=device)
            embeddings = self.embedding(bar_indices)  # (num_bars, embed_dim)
            embeddings = embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            embeddings = self.embedding.expand(batch_size, num_bars, -1)

        return embeddings


if __name__ == '__main__':
    # 测试
    batch_size = 2
    seq_len = 128
    embed_dim = 512
    num_heads = 8

    # 创建测试输入
    x = torch.randn(batch_size, seq_len, embed_dim)
    bar_ids = torch.randint(-1, 10, (batch_size, seq_len))
    instrument_ids = torch.randint(-1, 5, (batch_size, seq_len))

    # 创建掩码
    mask_generator = FCAttentionMask()
    mask = mask_generator.create_mask(bar_ids, instrument_ids, seq_len, x.device)

    # 创建注意力块
    attention_block = FCAttentionBlock(embed_dim, num_heads, embed_dim * 4)

    # 前向传播
    output = attention_block(x, mask)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Mask shape: {mask.shape}")

    # === 测试 Summary Token 机制 ===
    print("\n=== Testing Summary Token Mechanism ===")

    num_bars = 10
    reg_len = 100  # Regular token 数量
    sum_len = num_bars  # Summary token 数量

    # 创建测试数据
    reg_x = torch.randn(batch_size, reg_len, embed_dim)
    reg_bar_ids = torch.randint(0, num_bars, (batch_size, reg_len))
    reg_instrument_ids = torch.randint(0, 4, (batch_size, reg_len))

    # 创建 Summary Token 嵌入
    sum_embedding = SummaryTokenEmbedding(embed_dim, max_bars=256)
    sum_x = sum_embedding(num_bars, batch_size, reg_x.device)
    print(f"Summary embedding shape: {sum_x.shape}")

    # 创建四种掩码
    sum_mask_generator = SummaryAttentionMask()
    masks = sum_mask_generator.create_masks(
        reg_bar_ids, reg_instrument_ids, num_bars, reg_x.device
    )
    print(f"SS mask shape: {masks['ss'].shape}")
    print(f"SR mask shape: {masks['sr'].shape}")
    print(f"RS mask shape: {masks['rs'].shape}")
    print(f"RR mask shape: {masks['rr'].shape}")

    # 创建 Summary Attention Block
    sum_attn_block = SummaryAttentionBlock(embed_dim, num_heads, embed_dim * 4)

    # 前向传播
    sum_output, reg_output = sum_attn_block(
        sum_x, reg_x,
        masks['ss'], masks['sr'], masks['rs'], masks['rr']
    )
    print(f"Summary output shape: {sum_output.shape}")
    print(f"Regular output shape: {reg_output.shape}")

    # 验证掩码正确性
    print("\n=== Verifying Mask Correctness ===")

    # SS 应该是因果的
    ss_is_causal = torch.all(masks['ss'] == torch.tril(torch.ones_like(masks['ss'][0])).unsqueeze(0).expand_as(masks['ss']))
    print(f"SS mask is causal: {ss_is_causal.item()}")

    # SR 应该只让 S_i 看 bar i 的 R
    sr_correct = True
    for b in range(batch_size):
        for bar_idx in range(num_bars):
            expected = (reg_bar_ids[b] == bar_idx)
            actual = masks['sr'][b, bar_idx]
            if not torch.all(expected == actual):
                sr_correct = False
                break
    print(f"SR mask is correct: {sr_correct}")

    # RS 应该只让 R 看已完成 bar 的 S
    rs_correct = True
    for b in range(batch_size):
        for j in range(reg_len):
            r_bar = reg_bar_ids[b, j].item()
            expected = torch.zeros(num_bars, dtype=torch.bool)
            expected[:r_bar] = True
            actual = masks['rs'][b, j]
            if not torch.all(expected == actual):
                rs_correct = False
                break
    print(f"RS mask is correct: {rs_correct}")

    print("\n=== All Tests Passed ===" if ss_is_causal and sr_correct and rs_correct else "\n=== Some Tests Failed ===")
