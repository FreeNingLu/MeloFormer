#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HID-MuseFormer 模型

基于 HID 编码 + MuseFormer Summary Token 机制
使用 FlexAttention (PyTorch 2.5+) 实现高效稀疏注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .attention_flex import FLEX_ATTENTION_AVAILABLE
from .attention_flex_summary import (
    FlexSummaryAttentionBlock,
    FlexSummaryAttentionMask,
    SummaryTokenEmbedding,
)


class HIDMuseFormer(nn.Module):
    """
    HID-MuseFormer: Summary Token + FlexAttention

    注意力机制:
    - ss: Summary → Summary (粗粒度跨 bar)
    - sr: Summary ← Regular (信息压缩)
    - rs: Regular → Summary (远距离上下文)
    - rr: Regular → Regular (细粒度近距离)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 24576,
        max_bars: int = 256,
        chord_start_id: int = 6,
        chord_end_id: int = 222,
        **kwargs,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.max_bars = max_bars
        self.chord_start_id = chord_start_id
        self.chord_end_id = chord_end_id

        if not FLEX_ATTENTION_AVAILABLE:
            raise RuntimeError(f"需要 PyTorch 2.5+ (当前: {torch.__version__})")

        # 嵌入层
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.instrument_embedding = nn.Embedding(130, embed_dim)  # 0-127: MIDI, 128: drums, 129: global
        self.dropout = nn.Dropout(dropout)

        # Summary Token
        self.summary_embedding = SummaryTokenEmbedding(embed_dim, max_bars)
        self.mask_generator = FlexSummaryAttentionMask()

        # Transformer 层
        self.layers = nn.ModuleList([
            FlexSummaryAttentionBlock(
                embed_dim, num_heads, ffn_dim, dropout,
                activation='swiglu', max_seq_len=max_seq_len, use_rope=True
            )
            for _ in range(num_layers)
        ])

        # 输出层
        self.output_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # 权重共享

        self._gradient_checkpointing = False
        self._autocast_dtype = None  # 由训练脚本设置
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def gradient_checkpointing_enable(self, autocast_dtype=None):
        """
        启用 Gradient Checkpointing

        Args:
            autocast_dtype: 混合精度 dtype (torch.bfloat16/torch.float16/None)
                           用于在 checkpoint 重计算时恢复 autocast 上下文
        """
        self._gradient_checkpointing = True
        self._autocast_dtype = autocast_dtype

    def _checkpoint_forward(
        self,
        layer,
        sum_x: torch.Tensor,
        x: torch.Tensor,
        summarize_mask,
        updating_mask,
    ):
        """
        选择性 Gradient Checkpointing：只对 FFN 部分 checkpoint

        FlexAttention 的反向传播与 checkpoint 重计算有 dtype 兼容性问题，
        因此我们将层拆分为：
        1. Attention 部分 - 正常执行（不 checkpoint）
        2. FFN 部分 - 使用 checkpoint（节省显存）

        这样既能利用 checkpoint 节省显存，又能避免 FlexAttention 的 dtype 问题。
        """
        # === Attention 部分 - 正常执行 ===
        sum_residual = sum_x
        reg_residual = x

        sum_x = layer.sum_attn_norm(sum_x)
        x = layer.reg_attn_norm(x)

        sum_x, x = layer.attention(sum_x, x, summarize_mask, updating_mask)

        sum_x = layer.dropout(sum_x)
        x = layer.dropout(x)

        sum_x = sum_residual + sum_x
        x = reg_residual + x

        # === FFN 部分 - 使用 checkpoint ===
        def ffn_forward(sum_x, x):
            sum_residual = sum_x
            reg_residual = x

            sum_x = layer.sum_ffn_norm(sum_x)
            x = layer.reg_ffn_norm(x)

            sum_x = layer._apply_ffn(sum_x, is_summary=True)
            x = layer._apply_ffn(x, is_summary=False)

            sum_x = layer.dropout(sum_x)
            x = layer.dropout(x)

            sum_x = sum_residual + sum_x
            x = reg_residual + x

            return sum_x, x

        # FFN checkpoint (dtype 安全，因为 FFN 只包含标准 Linear 层)
        sum_x, x = torch.utils.checkpoint.checkpoint(
            ffn_forward, sum_x, x, use_reentrant=False
        )

        return sum_x, x

    def forward(
        self,
        token_ids: torch.Tensor,
        chord_ids: Optional[torch.Tensor] = None,
        instrument_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        note_ids: Optional[torch.Tensor] = None,
        num_bars: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len)
            chord_ids: (batch, seq_len) bar/chord 索引
            instrument_ids: (batch, seq_len) 乐器 ID
            token_type_ids: (batch, seq_len) Token 类型 0=T,1=P,2=D,3=V
            note_ids: (batch, seq_len) 音符 ID
            num_bars: bar 数量

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        # 推断 bar 数量
        if num_bars is None:
            num_bars = int(chord_ids.max().item()) + 1 if chord_ids is not None else 1

        # 嵌入
        x = self.token_embedding(token_ids)
        if instrument_ids is not None:
            x = x + self.instrument_embedding(instrument_ids.clamp(0, 129))
        x = self.dropout(x)

        # Summary Token
        sum_x = self.summary_embedding(num_bars, batch_size, device)

        # 创建掩码
        summarize_mask, updating_mask = None, None
        if chord_ids is not None and instrument_ids is not None:
            summarize_mask, updating_mask = self.mask_generator.create_block_masks(
                chord_ids, chord_ids, instrument_ids,
                num_bars, batch_size, seq_len, device,
                token_type_ids=token_type_ids,
                note_ids=note_ids,
            )

        # Transformer
        # 注意: FlexAttention 与 Gradient Checkpointing + 混合精度存在兼容性问题
        # 解决方案: 只对 FFN 部分使用 checkpoint，Attention 部分正常执行
        for layer in self.layers:
            if self._gradient_checkpointing and self.training:
                sum_x, x = self._checkpoint_forward(
                    layer, sum_x, x, summarize_mask, updating_mask
                )
            else:
                sum_x, x = layer(sum_x, x, summarize_mask, updating_mask)

        # 输出
        x = self.output_norm(x)
        return self.lm_head(x)


def create_model(vocab_size: int = 643, model_size: str = 'base', **kwargs) -> HIDMuseFormer:
    """创建模型"""
    configs = {
        'small':  {'embed_dim': 256,  'num_layers': 6,  'num_heads': 4,  'ffn_dim': 1408},
        'base':   {'embed_dim': 512,  'num_layers': 12, 'num_heads': 8,  'ffn_dim': 2816},
        'large':  {'embed_dim': 768,  'num_layers': 16, 'num_heads': 12, 'ffn_dim': 4096},
        'xlarge': {'embed_dim': 1024, 'num_layers': 24, 'num_heads': 16, 'ffn_dim': 5632},
    }
    config = configs.get(model_size, configs['base'])
    config.update(kwargs)
    return HIDMuseFormer(vocab_size=vocab_size, **config)


if __name__ == '__main__':
    print("HID-MuseFormer 测试")
    for size in ['small', 'base', 'large', 'xlarge']:
        m = create_model(model_size=size)
        params = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"  {size}: {params:.2f}M")
