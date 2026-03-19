#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MeloFormer v2.0

Summary Token + FlexAttention 稀疏注意力符号音乐生成模型
GQA + RMSNorm + SwiGLU + 1D RoPE
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
    is_static_compile_mode,
    set_static_compile_mode,
    MAX_SEQ_LEN,
    MAX_SUM_LEN,
)
from .rms_norm import RMSNorm


class MeloFormer(nn.Module):
    """
    MeloFormer v2.0: Summary Token + FlexAttention + GQA + RMSNorm + SwiGLU

    注意力机制 (WSA 风格):
    - SS: Summary → Summary (粗粒度跨 bar)
    - SR: Summary ← Regular (信息压缩)
    - RS: Regular → Summary (远距离上下文)
    - RR: Regular → Regular (同 bar 内因果)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        num_kv_heads: int = None,  # v1.4: GQA
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 24576,
        max_bars: int = 256,
        chord_start_id: int = 6,
        chord_end_id: int = 222,
        use_rms_norm: bool = True,  # v1.4: RMSNorm
        use_2d_rope: bool = True,   # v1.4: 2D RoPE
        **kwargs,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.max_seq_len = max_seq_len
        self.max_bars = max_bars
        self.chord_start_id = chord_start_id
        self.chord_end_id = chord_end_id
        self.use_rms_norm = use_rms_norm
        self.use_2d_rope = use_2d_rope

        if not FLEX_ATTENTION_AVAILABLE:
            raise RuntimeError(f"需要 PyTorch 2.5+ (当前: {torch.__version__})")

        # 嵌入层
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.instrument_embedding = nn.Embedding(130, embed_dim)  # 0-127: MIDI, 128: drums, 129: global
        self.dropout = nn.Dropout(dropout)

        # Summary Token
        self.summary_embedding = SummaryTokenEmbedding(embed_dim, max_bars)
        self.mask_generator = FlexSummaryAttentionMask()

        # Transformer 层 (v2.0: GQA + RMSNorm + 1D RoPE)
        self.layers = nn.ModuleList([
            FlexSummaryAttentionBlock(
                embed_dim, num_heads, ffn_dim, dropout,
                activation='swiglu',
                max_seq_len=max_seq_len,
                max_bars=max_bars,
                num_kv_heads=self.num_kv_heads,
                use_rms_norm=use_rms_norm,
            )
            for _ in range(num_layers)
        ])

        # 输出层 (RMSNorm or LayerNorm)
        self.output_norm = RMSNorm(embed_dim) if use_rms_norm else nn.LayerNorm(embed_dim)
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
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            nn.init.ones_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)

    def gradient_checkpointing_enable(self, autocast_dtype=None):
        """
        启用 Gradient Checkpointing

        v1.7.4: 使用亚层级 Gradient Checkpointing
        - Attention 和 FFN 分别做 checkpoint
        - 计算 Attention 时释放 FFN 中间变量，反之亦然
        - 显存节省: 72GB → ~50GB (16k 序列)

        Args:
            autocast_dtype: 混合精度 dtype (torch.bfloat16/torch.float16/None)
                           用于在 checkpoint 重计算时恢复 autocast 上下文
        """
        self._gradient_checkpointing = True
        self._autocast_dtype = autocast_dtype

        # v1.7.4: 在每个 Block 中启用亚层级 checkpoint
        for layer in self.layers:
            layer._gradient_checkpointing = True
            layer._autocast_dtype = autocast_dtype

    def _checkpoint_forward(
        self,
        layer,
        sum_x: torch.Tensor,
        x: torch.Tensor,
        summarize_mask,
        updating_mask,
        bar_ids: torch.Tensor = None,  # v1.4: 2D RoPE
        token_in_bar_ids: torch.Tensor = None,  # v1.6: 预计算的 token_in_bar_ids
    ):
        """
        全量 Gradient Checkpointing：对整个 layer 做 checkpoint

        v1.0.3 更新：从选择性 checkpoint (只 FFN) 改为全量 checkpoint
        - 显存节省：从 ~30% 提升到 ~60-70%
        - 解决方案：在 checkpoint 内部恢复 autocast 上下文

        v1.6 更新：传递预计算的 token_in_bar_ids
        """
        autocast_dtype = self._autocast_dtype

        def layer_forward(sum_x, x):
            # 在 checkpoint 重计算时恢复 autocast 上下文
            # 这是解决 FlexAttention dtype 不匹配的关键
            if autocast_dtype is not None:
                with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                    return layer(sum_x, x, summarize_mask, updating_mask,
                                bar_ids=bar_ids, token_in_bar_ids=token_in_bar_ids)
            else:
                return layer(sum_x, x, summarize_mask, updating_mask,
                            bar_ids=bar_ids, token_in_bar_ids=token_in_bar_ids)

        return torch.utils.checkpoint.checkpoint(
            layer_forward, sum_x, x,
            use_reentrant=False,
            preserve_rng_state=True,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        chord_ids: Optional[torch.Tensor] = None,
        instrument_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        note_ids: Optional[torch.Tensor] = None,
        doc_ids: Optional[torch.Tensor] = None,  # 新增: 文档 ID (用于序列打包)
        num_bars: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len)
            chord_ids: (batch, seq_len) bar/chord 索引
            instrument_ids: (batch, seq_len) 乐器 ID
            token_type_ids: (batch, seq_len) Token 类型 0=T,1=P,2=D,3=V
            note_ids: (batch, seq_len) 音符 ID
            doc_ids: (batch, seq_len) 文档 ID，用于序列打包时隔离不同文档的注意力
            num_bars: bar 数量

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        # 静态编译模式: 验证输入已 Padding 到固定长度
        if is_static_compile_mode():
            from .attention_flex_summary import MAX_SEQ_LEN as _MAX_SEQ_LEN
            assert seq_len == _MAX_SEQ_LEN, (
                f"[静态编译] 输入长度 {seq_len} != MAX_SEQ_LEN {_MAX_SEQ_LEN}. "
                f"请确保 collate 函数将所有输入 Padding 到 {_MAX_SEQ_LEN}"
            )

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

        # 创建掩码 (v2.0: 只需 num_bars, batch_size, device)
        summarize_mask, updating_mask = self.mask_generator.create_block_masks(
            num_bars, batch_size, device,
        )

        # Transformer (v2.0: 不再需要 bar_ids / token_in_bar_ids)
        for layer in self.layers:
            sum_x, x = layer(sum_x, x, summarize_mask, updating_mask)

        # 输出
        x = self.output_norm(x)
        return self.lm_head(x)


def create_model(vocab_size: int = 643, model_size: str = 'base', **kwargs) -> MeloFormer:
    """
    创建模型

    模型配置 (包含 GQA):
    - 8m:    ~8M params   (embed=256,  layers=6,  heads=4,  kv=2)
    - 62m:   ~62M params  (embed=512,  layers=12, heads=8,  kv=4)
    - 177m:  ~177M params (embed=768,  layers=16, heads=12, kv=4)
    - 366m:  ~366M params (embed=1024, layers=24, heads=16, kv=4)
    - 600m:  ~600M params (embed=1280, layers=28, heads=20, kv=4)
    - 800m:  ~800M params (embed=1408, layers=30, heads=22, kv=2)
    - 1.5b:  ~1.5B params (embed=1536, layers=32, heads=24, kv=8)

    兼容旧名称: small/base/large/xlarge
    """
    configs = {
        # 新命名 (带参数量后缀) — GQA ratio 必须是 2 的幂 (FlexAttention Triton kernel 要求)
        '8m':   {'embed_dim': 256,  'num_layers': 6,  'num_heads': 4,  'num_kv_heads': 2,  'ffn_dim': 1408},   # 2:1
        '62m':  {'embed_dim': 512,  'num_layers': 12, 'num_heads': 8,  'num_kv_heads': 4,  'ffn_dim': 2816},   # 2:1
        '177m': {'embed_dim': 768,  'num_layers': 16, 'num_heads': 12, 'num_kv_heads': 6,  'ffn_dim': 4096},   # 2:1
        '366m': {'embed_dim': 1024, 'num_layers': 24, 'num_heads': 16, 'num_kv_heads': 4,  'ffn_dim': 4096},   # 4:1
        '600m': {'embed_dim': 1280, 'num_layers': 28, 'num_heads': 20, 'num_kv_heads': 10, 'ffn_dim': 5120},   # 2:1
        '800m': {'embed_dim': 1408, 'num_layers': 30, 'num_heads': 22, 'num_kv_heads': 11, 'ffn_dim': 5632},   # 2:1
        '1.5b': {'embed_dim': 1536, 'num_layers': 32, 'num_heads': 24, 'num_kv_heads': 12, 'ffn_dim': 5632},   # 2:1

        # 兼容旧命名
        'small':  {'embed_dim': 256,  'num_layers': 6,  'num_heads': 4,  'num_kv_heads': 2,  'ffn_dim': 1408},
        'base':   {'embed_dim': 512,  'num_layers': 12, 'num_heads': 8,  'num_kv_heads': 4,  'ffn_dim': 2816},
        'large':  {'embed_dim': 768,  'num_layers': 16, 'num_heads': 12, 'num_kv_heads': 6,  'ffn_dim': 4096},
        'xlarge': {'embed_dim': 1024, 'num_layers': 24, 'num_heads': 16, 'num_kv_heads': 4,  'ffn_dim': 4096},
    }
    config = configs.get(model_size, configs['base'])
    config.update(kwargs)
    return MeloFormer(vocab_size=vocab_size, **config)


if __name__ == '__main__':
    print("MeloFormer 测试")
    print("=" * 50)
    for size in ['8m', '62m', '177m', '366m', '600m', '800m', '1.5b']:
        m = create_model(model_size=size)
        params = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"  {size}: {params:.2f}M")
    print()
    # 兼容旧名称
    print("兼容旧名称:")
    for size in ['small', 'base', 'large', 'xlarge']:
        m = create_model(model_size=size)
        params = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"  {size}: {params:.2f}M")
