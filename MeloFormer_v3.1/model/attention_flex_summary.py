#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlexAttention with Summary Token - MeloFormer v3.0

v2.1 更新:
- 恢复 captured tensor mask_mod (PoC Level 4 证明 PyTorch 2.10 不再触发重编译)
- 恢复变长 chord (自然 token 数，不再 padding 到固定 BAR_LEN)
- 恢复 Token Type Visibility (T/P/D/V 因果先验)
- Phase 1 (Summarize) 改用 F.scaled_dot_product_attention
- Phase 2 (Updating) 保持 FlexAttention + captured tensor mask_mod
- enable_gqa=True (PyTorch 2.10, 不再需要 repeat_interleave)
- 1D RoPE (2D HierarchicalRoPE 已证伪)

v2.1.1 FA4 兼容性更新:
- num_chords 分桶: CHORD_BUCKET_SIZE 控制, 限制 (Q_LEN, KV_LEN) 组合数
- 固定 buffer captured tensor: 预分配固定大小 int32 buffer, 每次 copy_ 更新
  避免 torch.compile(dynamic=False) 因 tensor identity/shape 变化触发重编译
- 移除错误的 mask_cache: block_mask sparsity 每个样本不同, 不可按 shape 缓存
- SDPA summarize_mask 仍然按 (bucketed_seq_len, bucketed_num_chords) 缓存 (无编译问题)

实现 Summary Token 机制：
- ss (Summary → Summary): 粗粒度跨 chord 交互
- sr (Summary ← Regular): 信息压缩，S 聚合同 chord 的 R
- rs (Regular → Summary): 获取远距离上下文，R 读取已完成 chord 的 S
- rr (Regular → Regular): 细粒度同 chord 内交互 + Token Type Visibility
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable

from .attention_flex import (
    RotaryPositionEmbedding,
    apply_rotary_pos_emb,
    TOKEN_TYPE_VISIBILITY,
    FLEX_ATTENTION_AVAILABLE,
    check_flex_attention_available,
)
from .rms_norm import RMSNorm

if FLEX_ATTENTION_AVAILABLE:
    from functools import partial
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    # -------------------------------------------------------------------------
    # FlexAttention 后端配置
    #
    # TRITON (默认): dynamic=True，支持变长序列，无需额外依赖
    # FA4:           dynamic=False + kernel_options={"BACKEND": "FLASH"}，
    #                需要 PyTorch nightly + FlashAttention-4 安装
    #                性能更高 (3-6x vs Triton)，但要求序列长度固定
    #                (BUCKET_SIZE + CHORD_BUCKET_SIZE 分桶可减少重编译次数)
    # -------------------------------------------------------------------------
    _FLEX_BACKEND: str = "TRITON"  # 运行时可通过 set_flex_backend() 修改

    _compiled_flex_attention = torch.compile(flex_attention, dynamic=True)
    _compiled_create_block_mask = torch.compile(create_block_mask, dynamic=True)

# 序列长度分桶: 与 train.py/train_fim.py 的 BUCKET_SIZE=512 对齐
# num_chords 分桶: 32
#
# 总 (Q_LEN, KV_LEN) 组合数:
#   Q_LEN  种类 = max_seq_len / SEQ_BUCKET_SIZE  = 16384/512 = 32
#   KV_LEN 种类 = max_chords  / CHORD_BUCKET_SIZE = 256/32   = 8
#   总组合 = 32 * 8 = 256 种
#   每种需要 fwd + bwd 两次编译, 但编译有缓存, 热启动后不再触发
SEQ_BUCKET_SIZE = 512
CHORD_BUCKET_SIZE = 32


def _bucket(value: int, bucket_size: int) -> int:
    """向上取整到最近的桶边界"""
    return ((value + bucket_size - 1) // bucket_size) * bucket_size


def set_flex_backend(backend: str) -> None:
    """
    设置 FlexAttention 后端。需在模型创建前调用。

    Args:
        backend: "TRITON" (默认) 或 "FA4"
            - TRITON: dynamic=True，支持变长序列，无额外依赖
            - FA4:    dynamic=False + FLASH kernel，需要 PyTorch nightly + FA4
                      性能更高但要求 FlashAttention-4 已安装
    """
    import sys
    this = sys.modules[__name__]
    if not FLEX_ATTENTION_AVAILABLE:
        return

    backend = backend.upper()
    if backend not in ("TRITON", "FA4"):
        raise ValueError(f"flex_backend 必须是 'TRITON' 或 'FA4'，收到: {backend!r}")

    from functools import partial
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    if backend == "FA4":
        # FA4: 需要 CUTE flash attention library (FlashAttention-4)
        # 通过小规模测试验证是否真正可用，不可用时自动 fallback 到 TRITON
        try:
            _fa4_test = partial(flex_attention, kernel_options={"BACKEND": "FLASH"})
            _fa4_compiled = torch.compile(_fa4_test, dynamic=False)
            # 用小 tensor 触发实际编译，检测 CUTE 库是否存在
            _q = torch.randn(1, 1, 16, 16, device='cuda', dtype=torch.bfloat16)
            _fa4_compiled(_q, _q, _q)
            # 通过，正式设置
            fa4_flex = partial(flex_attention, kernel_options={"BACKEND": "FLASH"})
            this._compiled_flex_attention = torch.compile(fa4_flex, dynamic=False)
            this._compiled_create_block_mask = torch.compile(create_block_mask, dynamic=False)
            this._FLEX_BACKEND = "FA4"
            print("[FlexAttention] 后端切换为: FA4")
        except Exception as e:
            print(f"[FlexAttention] FA4 不可用 ({type(e).__name__}: CUTE library missing)，自动 fallback 到 TRITON")
            this._compiled_flex_attention = torch.compile(flex_attention, dynamic=True)
            this._compiled_create_block_mask = torch.compile(create_block_mask, dynamic=True)
            this._FLEX_BACKEND = "TRITON"
    else:
        # TRITON: dynamic=True，通用，无额外依赖
        this._compiled_flex_attention = torch.compile(flex_attention, dynamic=True)
        this._compiled_create_block_mask = torch.compile(create_block_mask, dynamic=True)
        this._FLEX_BACKEND = "TRITON"
        print("[FlexAttention] 后端切换为: TRITON")


# =============================================================================
# v2.1 常量
# =============================================================================
PAD_TOKEN_ID = 0
IGNORE_INDEX = -100


class SummaryTokenEmbedding(nn.Module):
    """
    Summary Token 嵌入层

    为每个 chord 生成一个 Summary Token 嵌入。
    支持：
    1. 可学习嵌入（每个 chord 位置一个）
    2. 共享嵌入（所有 chord 使用同一嵌入）
    """

    def __init__(
        self,
        embed_dim: int,
        max_chords: int = 256,
        learnable: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_chords = max_chords
        self.learnable = learnable

        if learnable:
            self.embedding = nn.Embedding(max_chords, embed_dim)
            nn.init.normal_(self.embedding.weight, std=0.02)
        else:
            self.embedding = nn.Parameter(torch.zeros(1, embed_dim))
            nn.init.normal_(self.embedding, std=0.02)

    def forward(self, num_chords: int, batch_size: int, device: torch.device) -> torch.Tensor:
        if num_chords > self.max_chords:
            raise ValueError(f"num_chords ({num_chords}) > max_chords ({self.max_chords})")

        if self.learnable:
            chord_indices = torch.arange(num_chords, device=device)
            embeddings = self.embedding(chord_indices).unsqueeze(0).expand(batch_size, -1, -1)
        else:
            embeddings = self.embedding.expand(batch_size, num_chords, -1)

        return embeddings.to(device)


class FlexSummaryAttentionMask:
    """
    Summary Token 注意力掩码生成器 (v2.1.1)

    v2.1.1 FA4 兼容性改动:
    - captured tensor 使用固定大小 buffer + copy_, 不创建新 tensor
    - num_chords 分桶, 限制 (Q_LEN, KV_LEN) 组合数
    - block_mask 不缓存 (每个样本的 sparsity pattern 不同)
    - summarize_sdpa_mask 按 (bucketed_seq_len, bucketed_num_chords) 缓存
    """

    def __init__(self, rs_look_back: int = -1, max_seq_len: int = 16384, max_chords: int = 256):
        self.rs_look_back = rs_look_back

        # 固定大小 captured tensor buffer (int32 for FA4 CuTeDSL)
        # 预分配到最大可能尺寸, 每次 copy_ 更新内容
        # 这些 buffer 的 identity 和 shape 永远不变 → 不触发重编译
        self._chord_ids_buf = torch.zeros(max_seq_len, dtype=torch.int32)
        self._token_type_ids_buf = torch.full((max_seq_len,), -1, dtype=torch.int32)
        self._note_ids_buf = torch.full((max_seq_len,), -1, dtype=torch.int32)
        self._doc_ids_buf = torch.zeros(max_seq_len, dtype=torch.int32)          # 每个 regular token 的文档 ID
        self._sum_doc_ids_buf = torch.zeros(max_chords, dtype=torch.int32)        # 每个 summary token 的文档 ID
        self._type_visibility_buf = TOKEN_TYPE_VISIBILITY.clone()
        self._buffers_on_device = False

    def _ensure_buffers_on_device(self, device: torch.device):
        """首次调用时将 buffer 移到 GPU"""
        if not self._buffers_on_device:
            self._chord_ids_buf = self._chord_ids_buf.to(device)
            self._token_type_ids_buf = self._token_type_ids_buf.to(device)
            self._note_ids_buf = self._note_ids_buf.to(device)
            self._doc_ids_buf = self._doc_ids_buf.to(device)
            self._sum_doc_ids_buf = self._sum_doc_ids_buf.to(device)
            self._type_visibility_buf = self._type_visibility_buf.to(device)
            self._buffers_on_device = True

    def _build_mask_mod(
        self,
        bucketed_seq_len: int,
        bucketed_num_chords: int,
    ) -> Callable:
        """
        构建 mask_mod 闭包, capture 的是固定 buffer 引用 (identity 不变)。

        运行时语义由 buffer 内容 + bucketed_seq_len/bucketed_num_chords 共同决定。
        torch.compile guard 只看 captured tensor identity/shape + Python scalar 值,
        所以只要 (bucketed_seq_len, bucketed_num_chords) 相同就不会重编译。

        跨文档隔离:
        - RS: Regular → Summary 增加 doc_ids[q] == sum_doc_ids[kv] 检查
        - RR: same_chord 条件已天然隔离 (chord_ids 打包时已累积偏移)
        """
        sum_len = bucketed_num_chords
        rs_look_back = self.rs_look_back
        chord_ids_len = bucketed_seq_len

        # 引用固定 buffer — identity 永远不变
        chord_ids = self._chord_ids_buf
        token_type_ids = self._token_type_ids_buf
        note_ids = self._note_ids_buf
        type_visibility = self._type_visibility_buf
        doc_ids = self._doc_ids_buf
        sum_doc_ids = self._sum_doc_ids_buf

        def mask_mod(b, h, q_idx, kv_idx):
            q_safe = q_idx.clamp(0, chord_ids_len - 1)
            kv_safe = kv_idx.clamp(0, sum_len + chord_ids_len - 1)

            is_kv_summary = kv_safe < sum_len
            is_kv_regular = ~is_kv_summary

            # Query 的 chord 和文档 (captured tensor indexing)
            q_chord = chord_ids[q_safe].clamp(0, sum_len - 1)
            q_doc = doc_ids[q_safe]

            # === RS: Regular → Summary (已完成 chord + 同文档) ===
            kv_sum_doc = sum_doc_ids[kv_safe.clamp(0, sum_len - 1)]
            same_doc_rs = (q_doc == kv_sum_doc)

            if rs_look_back == -1:
                rs_mask = is_kv_summary & (q_chord > kv_safe) & same_doc_rs
            else:
                start_chord = (q_chord - rs_look_back).clamp(0)
                rs_mask = is_kv_summary & (kv_safe >= start_chord) & (kv_safe < q_chord) & same_doc_rs

            # === RR: 同 chord 内因果 + Token Type Visibility ===
            # chord_ids 打包时已累积偏移, same_chord 天然隔离跨文档
            reg_kv = (kv_safe - sum_len).clamp(0, chord_ids_len - 1)
            kv_chord = chord_ids[reg_kv].clamp(0, sum_len - 1)
            same_chord = (q_chord == kv_chord)
            rr_causal = q_safe >= reg_kv

            q_type = token_type_ids[q_safe].clamp(0, 3)
            k_type = token_type_ids[reg_kv].clamp(0, 3)
            q_note = note_ids[q_safe]
            k_note = note_ids[reg_kv]

            same_note = (q_note == k_note) & (q_note >= 0) & (k_note >= 0)
            type_visible = type_visibility[q_type, k_type]
            is_global_type = (token_type_ids[q_safe] == -1) | (token_type_ids[reg_kv] == -1)
            is_global_note = (q_note == -1) | (k_note == -1)
            token_mask = same_note | type_visible | is_global_type | is_global_note

            rr_mask = is_kv_regular & same_chord & rr_causal & token_mask

            return rs_mask | rr_mask

        return mask_mod

    def create_masks(
        self,
        chord_ids: torch.Tensor,
        num_chords: int,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        token_type_ids: Optional[torch.Tensor] = None,
        note_ids: Optional[torch.Tensor] = None,
        doc_ids: Optional[torch.Tensor] = None,
    ) -> Tuple:
        """
        创建两个阶段的 mask。

        v2.2: 支持序列打包 (doc_ids 跨文档隔离)。
        非打包模式下 doc_ids=None → 全 0 → 隔离检查变成 no-op。

        Returns:
            (summarize_sdpa_mask, updating_block_mask, bucketed_num_chords, bucketed_seq_len)
        """
        check_flex_attention_available()
        self._ensure_buffers_on_device(device)

        # 分桶: 限制 (Q_LEN, KV_LEN) 组合数, 避免 FA4 (dynamic=False) 过度重编译
        bucketed_num_chords = _bucket(num_chords, CHORD_BUCKET_SIZE)
        bucketed_seq_len = _bucket(seq_len, SEQ_BUCKET_SIZE)

        # --- 更新固定 buffer 内容 (copy_, 不改变 identity/shape) ---
        self._chord_ids_buf[:seq_len].copy_(chord_ids[:seq_len].to(torch.int32))
        if seq_len < self._chord_ids_buf.shape[0]:
            self._chord_ids_buf[seq_len:].fill_(num_chords - 1)

        if token_type_ids is not None:
            self._token_type_ids_buf[:seq_len].copy_(token_type_ids[:seq_len].to(torch.int32))
            if seq_len < self._token_type_ids_buf.shape[0]:
                self._token_type_ids_buf[seq_len:].fill_(-1)

        if note_ids is not None:
            self._note_ids_buf[:seq_len].copy_(note_ids[:seq_len].to(torch.int32))
            if seq_len < self._note_ids_buf.shape[0]:
                self._note_ids_buf[seq_len:].fill_(-1)

        # doc_ids buffer: 非打包模式全 0 (同一文档), 打包模式由 collator 提供
        if doc_ids is not None:
            self._doc_ids_buf[:seq_len].copy_(doc_ids[:seq_len].to(torch.int32))
        else:
            self._doc_ids_buf[:seq_len].fill_(0)
        if seq_len < self._doc_ids_buf.shape[0]:
            self._doc_ids_buf[seq_len:].fill_(-1)  # padding 区域文档 ID=-1

        # sum_doc_ids: 每个 chord 的文档归属
        # chord c → 取该 chord 中第一个 token 的 doc_id
        # 向量化: scatter_reduce 找每个 chord 的最小位置索引
        chord_ids_int = self._chord_ids_buf[:seq_len]
        doc_ids_int = self._doc_ids_buf[:seq_len]
        positions = torch.arange(seq_len, dtype=torch.long, device=device)
        first_pos = torch.full((num_chords,), seq_len, dtype=torch.long, device=device)
        first_pos.scatter_reduce_(
            0, chord_ids_int[:seq_len].long().clamp(0, num_chords - 1),
            positions, reduce='amin',
        )
        valid = first_pos < seq_len
        self._sum_doc_ids_buf[:num_chords].fill_(-1)
        self._sum_doc_ids_buf[:num_chords][valid] = doc_ids_int[first_pos[valid].clamp(0, seq_len - 1)]
        if num_chords < self._sum_doc_ids_buf.shape[0]:
            self._sum_doc_ids_buf[num_chords:].fill_(-1)

        # --- Phase 1: SDPA summarize_mask ---
        bM = bucketed_num_chords
        bN = bucketed_seq_len
        M = num_chords
        N = seq_len
        summarize_mask = torch.full((bM, bM + bN), float('-inf'), device=device)

        # SS: 有效 Summary → 有效 Summary 因果 + 同文档隔离
        # ss_causal[i, j] = 0.0 iff i >= j AND same_doc(i, j)
        sum_doc = self._sum_doc_ids_buf[:M]  # (M,)
        same_doc_ss = (sum_doc.unsqueeze(1) == sum_doc.unsqueeze(0))  # (M, M)
        ss_causal = torch.tril(torch.ones(M, M, dtype=torch.bool, device=device))
        ss_allowed = ss_causal & same_doc_ss
        summarize_mask[:M, :M] = torch.where(
            ss_allowed,
            torch.tensor(0.0, device=device),
            torch.tensor(float('-inf'), device=device),
        )

        # SR: Summary_i ← chord_i 的 Regular tokens
        # chord_ids 已累积偏移, same_chord 天然保证同文档
        sr_bool = (chord_ids[:N].unsqueeze(0) == torch.arange(M, device=device).unsqueeze(1))
        summarize_mask[:M, bM:bM + N] = torch.where(
            sr_bool,
            torch.tensor(0.0, device=device),
            torch.tensor(float('-inf'), device=device),
        )

        summarize_mask = summarize_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, bM, bM+bN)

        # --- Phase 2: FlexAttention block_mask (每次重建) ---
        mask_mod = self._build_mask_mod(bucketed_seq_len, bucketed_num_chords)
        updating_block_mask = _compiled_create_block_mask(
            mask_mod,
            B=batch_size,
            H=None,
            Q_LEN=bucketed_seq_len,
            KV_LEN=bucketed_num_chords + bucketed_seq_len,
            device=device,
        )

        return summarize_mask, updating_block_mask, bucketed_num_chords, bucketed_seq_len


class FlexSummaryAttention(nn.Module):
    """
    FlexAttention with Summary Token + GQA + 1D RoPE (v2.1)

    v2.1:
    - Phase 1 (Summarize): SDPA with explicit mask
    - Phase 2 (Updating): FlexAttention with captured tensor mask_mod
    - enable_gqa=True (no repeat_interleave)
    - 1D RoPE
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int = None,
        dropout: float = 0.1,
        bias: bool = True,
        max_seq_len: int = 32768,
        max_chords: int = 256,
        # v3.0 注意力优化开关
        sub_attn_bias: bool = False,
        k2v2_gate: bool = False,
    ):
        super().__init__()

        check_flex_attention_available()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout

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

        # 1D RoPE
        self.sum_rope = RotaryPositionEmbedding(self.head_dim, max_chords)
        self.reg_rope = RotaryPositionEmbedding(self.head_dim, max_seq_len)

        # ── v3.0→v3.1: 子注意力 per-head additive bias ──
        # v3.0 原版用 shape [1] 标量，触发 Triton codegen ndim 断言
        # v3.1 改为 per-head [num_heads]，通过 h 索引访问，同时增强表达力
        self.sub_attn_bias = sub_attn_bias
        if sub_attn_bias:
            self.ss_bias = nn.Parameter(torch.zeros(num_heads))
            self.sr_bias = nn.Parameter(torch.zeros(num_heads))
            self.rs_bias = nn.Parameter(torch.zeros(num_heads))
            self.rr_bias = nn.Parameter(torch.zeros(num_heads))

        # ── v3.0: K2/V2 需求门控 ──
        self.k2v2_gate_enabled = k2v2_gate
        if k2v2_gate:
            # 初始化使 sigmoid 输出 ≈ 1.0（不改变原始行为）
            self.k2v2_gate_scalar = nn.Parameter(torch.ones(1))
            self.k2v2_gate_bias = nn.Parameter(torch.tensor(4.0))  # sigmoid(4) ≈ 0.98

    def forward(
        self,
        sum_x: torch.Tensor,
        reg_x: torch.Tensor,
        summarize_sdpa_mask: torch.Tensor,
        updating_block_mask,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        # dtype 对齐
        target_dtype = sum_q.dtype
        if sum_v.dtype != target_dtype:
            sum_v = sum_v.to(target_dtype)
        if reg_v.dtype != target_dtype:
            reg_v = reg_v.to(target_dtype)

        # === Phase 1: Summarize (SDPA) ===
        # cat K/V for Phase 1: Summary queries attend to all Summary + Regular K/V
        cat_k_sum = torch.cat([sum_k, reg_k], dim=2)  # (B, kv_heads, M+N, head_dim)
        cat_v_sum = torch.cat([sum_v, reg_v], dim=2)

        # v3.1: 子注意力 per-head additive bias (Phase 1: SS/SR)
        if self.sub_attn_bias:
            # summarize_sdpa_mask shape: (B, 1, M, M+N) — head 维度=1 用于广播
            # 展开到 (B, num_heads, M, M+N) 以支持 per-head bias
            bias_mask = summarize_sdpa_mask.expand(-1, self.num_heads, -1, -1).clone()
            ss = self.ss_bias.view(1, -1, 1, 1)  # (1, H, 1, 1)
            sr = self.sr_bias.view(1, -1, 1, 1)
            bias_mask[:, :, :, :sum_len] = bias_mask[:, :, :, :sum_len] + ss
            bias_mask[:, :, :, sum_len:] = bias_mask[:, :, :, sum_len:] + sr
        else:
            bias_mask = summarize_sdpa_mask

        sum_attn_out = F.scaled_dot_product_attention(
            sum_q, cat_k_sum, cat_v_sum,
            attn_mask=bias_mask,
            enable_gqa=True,
        )
        # reshape 替代 contiguous().view()，避免显式内存拷贝
        sum_attn_out = sum_attn_out.transpose(1, 2).reshape(batch_size, sum_len, self.embed_dim)

        # === Phase 1→2: K2/V2 投影 ===
        sum_k2 = self.sum_k2_proj(sum_attn_out)
        sum_v2 = self.sum_v2_proj(sum_attn_out)

        # v3.0: K2/V2 需求门控
        if self.k2v2_gate_enabled:
            reg_energy = reg_x.detach().norm(dim=-1, keepdim=True).mean(dim=1, keepdim=True)  # (1, 1, 1)
            gate = torch.sigmoid(self.k2v2_gate_scalar * reg_energy + self.k2v2_gate_bias)
            sum_k2 = sum_k2 * gate
            sum_v2 = sum_v2 * gate

        sum_k2 = sum_k2.reshape(batch_size, sum_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        sum_v2 = sum_v2.reshape(batch_size, sum_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if sum_k2.dtype != target_dtype:
            sum_k2 = sum_k2.to(target_dtype)
            sum_v2 = sum_v2.to(target_dtype)

        # === Phase 2: Updating (FlexAttention) ===
        # cat K2/V2(Summary, padded to bucketed_num_chords) + reg K/V for Phase 2
        cat_k_reg = torch.cat([sum_k2, reg_k], dim=2)  # (B, kv_heads, M+N, head_dim)
        cat_v_reg = torch.cat([sum_v2, reg_v], dim=2)

        # v3.1: 子注意力 per-head additive bias (Phase 2: RS/RR via score_mod)
        if self.sub_attn_bias:
            _rs_bias = self.rs_bias   # shape [num_heads]
            _rr_bias = self.rr_bias   # shape [num_heads]
            _sum_len = sum_len

            def score_mod(score, b, h, q_idx, kv_idx):
                is_summary = kv_idx < _sum_len
                return torch.where(is_summary, score + _rs_bias[h], score + _rr_bias[h])

            reg_attn_out = _compiled_flex_attention(
                reg_q, cat_k_reg, cat_v_reg,
                block_mask=updating_block_mask,
                score_mod=score_mod,
                enable_gqa=True,
            )
        else:
            reg_attn_out = _compiled_flex_attention(
                reg_q, cat_k_reg, cat_v_reg,
                block_mask=updating_block_mask,
                enable_gqa=True,
            )
        reg_attn_out = reg_attn_out.transpose(1, 2).reshape(batch_size, reg_len, self.embed_dim)

        # === 输出投影 ===
        sum_output = self.sum_out_proj(sum_attn_out)
        reg_output = self.reg_out_proj(reg_attn_out)

        return sum_output, reg_output


class FlexSummaryAttentionBlock(nn.Module):
    """
    FlexAttention Summary Block (v2.1)

    Pre-Norm + 残差连接
    Summary 和 Regular 各自有独立的 FFN (SwiGLU)
    亚层级 Gradient Checkpointing
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        activation: str = 'swiglu',
        max_seq_len: int = 32768,
        max_chords: int = 256,
        share_ffn: bool = False,
        num_kv_heads: int = None,
        use_rms_norm: bool = True,
        # v3.0 注意力优化开关
        sub_attn_bias: bool = False,
        k2v2_gate: bool = False,
    ):
        super().__init__()

        self.attention = FlexSummaryAttention(
            embed_dim, num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            max_chords=max_chords,
            sub_attn_bias=sub_attn_bias,
            k2v2_gate=k2v2_gate,
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

    def _run_attention(self, sum_x, reg_x, summarize_mask, updating_mask):
        sum_x_norm = self.sum_attn_norm(sum_x)
        reg_x_norm = self.reg_attn_norm(reg_x)
        sum_attn_out, reg_attn_out = self.attention(
            sum_x_norm, reg_x_norm, summarize_mask, updating_mask,
        )
        return self.dropout(sum_attn_out), self.dropout(reg_attn_out)

    def _run_ffn(self, sum_x, reg_x):
        sum_ffn_out = self._apply_ffn(self.sum_ffn_norm(sum_x), is_summary=True)
        reg_ffn_out = self._apply_ffn(self.reg_ffn_norm(reg_x), is_summary=False)
        return self.dropout(sum_ffn_out), self.dropout(reg_ffn_out)

    def forward(
        self,
        sum_x: torch.Tensor,
        reg_x: torch.Tensor,
        summarize_mask,
        updating_mask,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._gradient_checkpointing and self.training:
            autocast_dtype = self._autocast_dtype

            def run_attn_ckpt(s_x, r_x):
                if autocast_dtype is not None:
                    with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                        return self._run_attention(s_x, r_x, summarize_mask, updating_mask)
                return self._run_attention(s_x, r_x, summarize_mask, updating_mask)

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
            sum_residual, reg_residual = sum_x, reg_x
            sum_attn_out, reg_attn_out = self._run_attention(
                sum_x, reg_x, summarize_mask, updating_mask,
            )
            sum_x = sum_residual + sum_attn_out
            reg_x = reg_residual + reg_attn_out

            sum_residual, reg_residual = sum_x, reg_x
            sum_ffn_out, reg_ffn_out = self._run_ffn(sum_x, reg_x)
            sum_x = sum_residual + sum_ffn_out
            reg_x = reg_residual + reg_ffn_out

            return sum_x, reg_x
