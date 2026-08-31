#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MeloFormer v3.0

Summary Token + FlexAttention 稀疏注意力符号音乐生成模型
GQA + RMSNorm + SwiGLU + 1D RoPE

v2.1 更新:
- Phase 1 (Summarize) 改用 SDPA
- 恢复 captured tensor mask_mod (变长 chord + Token Type Visibility)
- 移除 2D RoPE 遗留参数
- 亚层级 Gradient Checkpointing 保留

v2.2 更新:
- FA4 兼容性（固定 buffer + 分桶）
- 序列打包 + 跨文档隔离

v3.0 更新:
- 子注意力 additive bias（SS/SR/RS/RR 可学习自适应）
- K2/V2 需求门控（reg_x 反馈驱动的流间交互）
- DepthSelector（sum_x 跨层自适应聚合，Full AttnRes）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List

from .attention_flex import FLEX_ATTENTION_AVAILABLE
from .attention_flex_summary import (
    FlexSummaryAttentionBlock,
    FlexSummaryAttentionMask,
    SummaryTokenEmbedding,
)
from .rms_norm import RMSNorm


class DepthSelector(nn.Module):
    """sum_x 的深度自适应聚合（Full AttnRes）。

    每层一个实例。从所有历史层的输出增量中，根据 content-dependent 权重选择性聚合，
    替代标准残差的均匀累加。

    参考：AttnRes (Kimi, arXiv:2603.15031)，直接用 Full 版本（不用 Block）。
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(embed_dim))       # 层级查询向量，零初始化
        self.norm = RMSNorm(embed_dim)                       # key 归一化
        self.scale = 1.0 / math.sqrt(embed_dim)             # 缩放因子
        self.mag_scale = nn.Parameter(torch.ones(1))         # 可学习 magnitude 缩放

    def forward(self, history: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            history: [v_0, v_1, ..., v_{l-1}]
                     v_0 = embedding, v_i = 第 i 层的输出增量 (detached)
        Returns:
            (B, M, d) 加权聚合结果
        """
        V = torch.stack(history, dim=0)                      # (L, B, M, d)
        K = self.norm(V)                                     # (L, B, M, d)
        logits = torch.einsum('c,lbmc->lbm', self.w, K) * self.scale
        weights = logits.softmax(dim=0)                      # 深度维度 softmax
        out = torch.einsum('lbm,lbmc->bmc', weights, V)     # (B, M, d)
        return out * self.mag_scale


class MeloFormer(nn.Module):
    """
    MeloFormer v3.0: Summary Token + FlexAttention + GQA + RMSNorm + SwiGLU

    注意力机制:
    - Phase 1 (Summarize): SDPA
      - SS: Summary → Summary (粗粒度跨 chord)
      - SR: Summary ← Regular (信息压缩)
    - Phase 2 (Updating): FlexAttention + captured tensor mask_mod
      - RS: Regular → Summary (远距离上下文)
      - RR: Regular → Regular (同 chord 内因果 + Token Type Visibility)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        num_kv_heads: int = None,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 24576,
        max_chords: int = 256,
        chord_start_id: int = 6,
        chord_end_id: int = 222,
        use_rms_norm: bool = True,
        # v3.0 注意力优化开关
        sub_attn_bias: bool = False,
        depth_selector: bool = False,
        k2v2_gate: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.max_seq_len = max_seq_len
        self.max_chords = max_chords
        self.chord_start_id = chord_start_id
        self.chord_end_id = chord_end_id
        self.use_rms_norm = use_rms_norm

        # v3.0 注意力优化配置
        self.sub_attn_bias = sub_attn_bias
        self.depth_selector = depth_selector
        self.k2v2_gate = k2v2_gate

        if not FLEX_ATTENTION_AVAILABLE:
            raise RuntimeError(f"需要 PyTorch 2.5+ (当前: {torch.__version__})")

        # 嵌入层
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.instrument_embedding = nn.Embedding(130, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Summary Token
        self.summary_embedding = SummaryTokenEmbedding(embed_dim, max_chords)
        self.mask_generator = FlexSummaryAttentionMask(
            max_seq_len=max_seq_len, max_chords=max_chords,
        )

        # Transformer 层
        self.layers = nn.ModuleList([
            FlexSummaryAttentionBlock(
                embed_dim, num_heads, ffn_dim, dropout,
                activation='swiglu',
                max_seq_len=max_seq_len,
                max_chords=max_chords,
                num_kv_heads=self.num_kv_heads,
                use_rms_norm=use_rms_norm,
                sub_attn_bias=sub_attn_bias,
                k2v2_gate=k2v2_gate,
            )
            for _ in range(num_layers)
        ])

        # 输出层
        self.output_norm = RMSNorm(embed_dim) if use_rms_norm else nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        # v3.0: DepthSelector（每层一个独立实例）
        if depth_selector:
            self.depth_selectors = nn.ModuleList([
                DepthSelector(embed_dim) for _ in range(num_layers)
            ])

        self._gradient_checkpointing = False
        self._autocast_dtype = None
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            nn.init.ones_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)

    def gradient_checkpointing_enable(self, autocast_dtype=None):
        """启用亚层级 Gradient Checkpointing"""
        self._gradient_checkpointing = True
        self._autocast_dtype = autocast_dtype
        for layer in self.layers:
            layer._gradient_checkpointing = True
            layer._autocast_dtype = autocast_dtype

    def forward(
        self,
        token_ids: torch.Tensor,
        chord_ids: Optional[torch.Tensor] = None,
        instrument_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        note_ids: Optional[torch.Tensor] = None,
        num_chords: Optional[int] = None,
        doc_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len)
            chord_ids: (batch, seq_len) 和弦 索引，每个 token 归属的和弦段 ID
            instrument_ids: (batch, seq_len) 乐器 ID
            token_type_ids: (batch, seq_len) Token 类型 0=T,1=P,2=D,3=V,-1=全局
            note_ids: (batch, seq_len) 音符 ID
            num_chords: chord 数量
            doc_ids: (batch, seq_len) 文档 ID (打包模式, None=单文档)

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        # 推断 chord 数量
        if num_chords is None:
            num_chords = int(chord_ids.max().item()) + 1 if chord_ids is not None else 1

        # 嵌入
        x = self.token_embedding(token_ids)
        if instrument_ids is not None:
            x = x + self.instrument_embedding(instrument_ids.clamp(0, 129))
        x = self.dropout(x)

        # Summary Token
        sum_x = self.summary_embedding(num_chords, batch_size, device)

        # v2.1: FlexAttention mask_mod 的 captured tensor 是 batch 共享的，
        # 要求 batch_size=1（使用 gradient_accumulation 代替大 batch）。
        if batch_size != 1:
            raise RuntimeError(
                f"v2.1 要求 batch_size=1（当前 batch_size={batch_size}）。"
                f"请使用 --batch_size 1 --gradient_accumulation_steps N 代替。"
            )
        chord_ids_1d = chord_ids[0] if chord_ids is not None else torch.zeros(seq_len, dtype=torch.long, device=device)
        tt_ids = token_type_ids[0] if token_type_ids is not None else None
        n_ids = note_ids[0] if note_ids is not None else None
        d_ids = doc_ids[0] if doc_ids is not None else None

        summarize_mask, updating_mask, bucketed_num_chords, bucketed_seq_len = self.mask_generator.create_masks(
            chord_ids=chord_ids_1d.to(device),
            num_chords=num_chords,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            token_type_ids=tt_ids.to(device) if tt_ids is not None else None,
            note_ids=n_ids.to(device) if n_ids is not None else None,
            doc_ids=d_ids.to(device) if d_ids is not None else None,
        )

        # Pad summary tokens to bucketed_num_chords for FA4 shape consistency
        if bucketed_num_chords > num_chords:
            pad_size = bucketed_num_chords - num_chords
            sum_x = F.pad(sum_x, (0, 0, 0, pad_size))  # (B, bucketed_M, D)

        # Pad regular tokens to bucketed_seq_len for FA4 Q_LEN consistency
        if bucketed_seq_len > seq_len:
            pad_size = bucketed_seq_len - seq_len
            x = F.pad(x, (0, 0, 0, pad_size))  # (B, bucketed_N, D)

        # Transformer
        if self.depth_selector:
            # v3.0: DepthSelector 模式 — sum_x 通过跨层自适应聚合
            history: List[torch.Tensor] = [sum_x.clone()]  # v_0 = embedding

            for i, layer in enumerate(self.layers):
                sum_in = self.depth_selectors[i](history)

                # 层内计算：用 sum_in 替代 sum_x 作为输入
                if layer._gradient_checkpointing and layer.training:
                    def _make_attn_fn(ly):
                        def run_attn(s, r):
                            if ly._autocast_dtype is not None:
                                with torch.autocast(device_type='cuda', dtype=ly._autocast_dtype):
                                    return ly._run_attention(s, r, summarize_mask, updating_mask)
                            return ly._run_attention(s, r, summarize_mask, updating_mask)
                        return run_attn
                    sum_attn_out, reg_attn_out = torch.utils.checkpoint.checkpoint(
                        _make_attn_fn(layer), sum_in, x,
                        use_reentrant=False, preserve_rng_state=True)
                else:
                    sum_attn_out, reg_attn_out = layer._run_attention(
                        sum_in, x, summarize_mask, updating_mask)

                sum_h = sum_in + sum_attn_out     # _run_attention 内部已 dropout
                x = x + reg_attn_out

                if layer._gradient_checkpointing and layer.training:
                    def _make_ffn_fn(ly):
                        def run_ffn(s, r):
                            if ly._autocast_dtype is not None:
                                with torch.autocast(device_type='cuda', dtype=ly._autocast_dtype):
                                    return ly._run_ffn(s, r)
                            return ly._run_ffn(s, r)
                        return run_ffn
                    sum_ffn_out, reg_ffn_out = torch.utils.checkpoint.checkpoint(
                        _make_ffn_fn(layer), sum_h, x,
                        use_reentrant=False, preserve_rng_state=True)
                else:
                    sum_ffn_out, reg_ffn_out = layer._run_ffn(sum_h, x)

                sum_x = sum_h + sum_ffn_out       # _run_ffn 内部已 dropout
                x = x + reg_ffn_out

                # 记录本层增量到 history（detached，不积累计算图）
                delta = (sum_attn_out + sum_ffn_out).detach()
                history.append(delta)
        else:
            # v2.2 原始路径
            for layer in self.layers:
                sum_x, x = layer(sum_x, x, summarize_mask, updating_mask)

        # 输出: 截断回原始 seq_len (去掉 padding)
        x = x[:, :seq_len, :]
        x = self.output_norm(x)
        return self.lm_head(x)


def create_model(vocab_size: int = 643, model_size: str = 'base', fim: bool = False, **kwargs) -> MeloFormer:
    """
    创建模型

    Args:
        vocab_size: 基础词表大小 (默认 643)
        model_size: 模型大小
        fim: 是否启用 FIM 模式 (扩展词表到 651)

    GQA ratio 必须是 2 的幂次 (FlexAttention enable_gqa=True 约束)
    """
    if fim:
        vocab_size = max(vocab_size, 651)  # 643 + 8 FIM tokens
    configs = {
        '8m':   {'embed_dim': 256,  'num_layers': 6,  'num_heads': 4,  'num_kv_heads': 2,  'ffn_dim': 1408},
        '62m':  {'embed_dim': 512,  'num_layers': 12, 'num_heads': 8,  'num_kv_heads': 4,  'ffn_dim': 2816},
        '177m': {'embed_dim': 768,  'num_layers': 16, 'num_heads': 12, 'num_kv_heads': 6,  'ffn_dim': 4096},
        '366m': {'embed_dim': 1024, 'num_layers': 24, 'num_heads': 16, 'num_kv_heads': 4,  'ffn_dim': 4096},
        '600m': {'embed_dim': 1280, 'num_layers': 28, 'num_heads': 20, 'num_kv_heads': 10, 'ffn_dim': 5120},
        '800m': {'embed_dim': 1408, 'num_layers': 30, 'num_heads': 22, 'num_kv_heads': 11, 'ffn_dim': 5632},
        '1.5b': {'embed_dim': 1536, 'num_layers': 32, 'num_heads': 24, 'num_kv_heads': 12, 'ffn_dim': 5632},
        # 兼容旧命名
        'small':  {'embed_dim': 256,  'num_layers': 6,  'num_heads': 4,  'num_kv_heads': 2,  'ffn_dim': 1408},
        'base':   {'embed_dim': 512,  'num_layers': 12, 'num_heads': 8,  'num_kv_heads': 4,  'ffn_dim': 2816},
        'large':  {'embed_dim': 768,  'num_layers': 16, 'num_heads': 12, 'num_kv_heads': 6,  'ffn_dim': 4096},
        'xlarge': {'embed_dim': 1024, 'num_layers': 24, 'num_heads': 16, 'num_kv_heads': 4,  'ffn_dim': 4096},
    }
    config = configs.get(model_size, configs['base'])
    config.update(kwargs)
    return MeloFormer(vocab_size=vocab_size, **config)
