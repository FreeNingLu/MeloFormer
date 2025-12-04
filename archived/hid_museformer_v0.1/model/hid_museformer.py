#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HID-MuseFormer 模型

基于 HID (Hierarchical Instrument-aware Duration-free) 编码的 MuseFormer 变体

主要特点：
1. 和弦 token 替代 BAR token，携带和声信息
2. FC-Attention 适配 HID 的乐器分组结构
3. 乐器间通过 SEP token 交互
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict

from .attention import FCAttentionBlock, FCAttentionMask


class PositionalEncoding(nn.Module):
    """
    可学习的位置编码 (备用，RoPE 已在 attention 中)

    注意: 现在主要使用 RoPE，这个类保留用于向后兼容
    """

    def __init__(self, embed_dim: int, max_seq_len: int = 8192, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim

        # 可学习的位置编码 (备用)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        # 初始化
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor, use_pos_embed: bool = False) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            use_pos_embed: 是否使用位置编码 (默认 False，因为用 RoPE)
        """
        if use_pos_embed:
            seq_len = x.size(1)
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            pos_embed = self.position_embedding(positions)
            return self.dropout(x + pos_embed)
        else:
            return self.dropout(x)


class ChordEmbedding(nn.Module):
    """
    和弦感知的嵌入层

    为和弦 token 提供额外的和声信息嵌入
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        chord_start_id: int,
        chord_end_id: int,
        num_roots: int = 12,
        num_types: int = 7,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.chord_start_id = chord_start_id
        self.chord_end_id = chord_end_id

        # 基础 token 嵌入
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # 和弦根音嵌入 (C, C#, D, ... B)
        self.root_embedding = nn.Embedding(num_roots + 1, embed_dim)  # +1 for N chord

        # 和弦类型嵌入 (maj, min, dim, 7, maj7, min7, sus4)
        self.type_embedding = nn.Embedding(num_types + 1, embed_dim)  # +1 for N chord

        # 融合层
        self.chord_fusion = nn.Linear(embed_dim * 3, embed_dim)

        # 初始化
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.root_embedding.weight, std=0.02)
        nn.init.normal_(self.type_embedding.weight, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,
        chord_root_ids: Optional[torch.Tensor] = None,
        chord_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len) token IDs
            chord_root_ids: (batch, seq_len) 和弦根音 ID (0-11, 12=N)
            chord_type_ids: (batch, seq_len) 和弦类型 ID (0-6, 7=N)

        Returns:
            embeddings: (batch, seq_len, embed_dim)
        """
        # 基础嵌入
        token_embed = self.token_embedding(token_ids)

        # 如果没有提供和弦信息，直接返回
        if chord_root_ids is None or chord_type_ids is None:
            return token_embed

        # 检查哪些位置是和弦 token
        is_chord = (token_ids >= self.chord_start_id) & (token_ids <= self.chord_end_id)

        # 获取和弦嵌入
        root_embed = self.root_embedding(chord_root_ids)
        type_embed = self.type_embedding(chord_type_ids)

        # 融合和弦信息
        chord_combined = torch.cat([token_embed, root_embed, type_embed], dim=-1)
        chord_embed = self.chord_fusion(chord_combined)

        # 只对和弦 token 使用融合嵌入
        output = torch.where(
            is_chord.unsqueeze(-1).expand_as(token_embed),
            chord_embed,
            token_embed
        )

        return output


class InstrumentEmbedding(nn.Module):
    """
    乐器感知嵌入层

    为每个乐器提供独立的嵌入
    """

    def __init__(self, embed_dim: int, num_instruments: int = 130):
        super().__init__()

        # 乐器嵌入 (0-127: MIDI program, 128: drums, 129: global)
        self.instrument_embedding = nn.Embedding(num_instruments, embed_dim)
        nn.init.normal_(self.instrument_embedding.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        instrument_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            instrument_ids: (batch, seq_len) 乐器 ID

        Returns:
            output: (batch, seq_len, embed_dim)
        """
        inst_embed = self.instrument_embedding(instrument_ids)
        return x + inst_embed


class HIDMuseFormer(nn.Module):
    """
    HID-MuseFormer 主模型

    结构：
    1. 嵌入层：Token + Position + Instrument + Chord
    2. FC-Attention 层 × N
    3. 输出层：LM Head
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 24576,  # H800 80GB 可支持 24K+
        chord_start_id: int = 6,
        chord_end_id: int = 90,
        fine_grained_bars: Tuple[int, ...] = (-1, -2, -4, -8, -12, -16, -24, -32),
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # 嵌入层
        self.chord_embedding = ChordEmbedding(
            vocab_size, embed_dim, chord_start_id, chord_end_id
        )
        self.position_encoding = PositionalEncoding(embed_dim, max_seq_len, dropout)
        self.instrument_embedding = InstrumentEmbedding(embed_dim)

        # Transformer 层 (Pre-LN + RoPE + SwiGLU)
        self.layers = nn.ModuleList([
            FCAttentionBlock(
                embed_dim, num_heads, ffn_dim, dropout,
                activation='swiglu',
                max_seq_len=max_seq_len,
                use_rope=True
            )
            for _ in range(num_layers)
        ])

        # 输出层
        self.output_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # 权重共享
        self.lm_head.weight = self.chord_embedding.token_embedding.weight

        # 注意力掩码生成器
        self.mask_generator = FCAttentionMask(fine_grained_bars)

        # Gradient Checkpointing
        self._gradient_checkpointing = False

        # 初始化
        self.apply(self._init_weights)

    def gradient_checkpointing_enable(self):
        """启用 Gradient Checkpointing (显存换速度)"""
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """禁用 Gradient Checkpointing"""
        self._gradient_checkpointing = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        token_ids: torch.Tensor,
        chord_ids: Optional[torch.Tensor] = None,
        instrument_ids: Optional[torch.Tensor] = None,
        chord_root_ids: Optional[torch.Tensor] = None,
        chord_type_ids: Optional[torch.Tensor] = None,
        is_chord_token: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len) token IDs
            chord_ids: (batch, seq_len) 和弦索引 (用于 FC-Attention)
            instrument_ids: (batch, seq_len) 乐器 ID
            chord_root_ids: (batch, seq_len) 和弦根音 ID
            chord_type_ids: (batch, seq_len) 和弦类型 ID
            is_chord_token: (batch, seq_len) 是否是和弦 token
            attention_mask: (batch, seq_len, seq_len) 预计算的注意力掩码
            key_padding_mask: (batch, seq_len) padding mask

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = token_ids.shape

        # 自动检测和弦 token (如果没有提供)
        if is_chord_token is None:
            # 和弦 token ID 在 chord_start_id 到 chord_end_id 之间
            is_chord_token = (token_ids >= self.chord_embedding.chord_start_id) & \
                            (token_ids <= self.chord_embedding.chord_end_id)

        # 嵌入
        x = self.chord_embedding(token_ids, chord_root_ids, chord_type_ids)
        x = self.position_encoding(x)

        if instrument_ids is not None:
            # 确保 instrument_ids 在有效范围内
            instrument_ids = instrument_ids.clamp(0, 129)
            x = self.instrument_embedding(x, instrument_ids)

        # 生成或使用注意力掩码
        if attention_mask is None and chord_ids is not None and instrument_ids is not None:
            attention_mask = self.mask_generator.create_mask(
                chord_ids, instrument_ids, seq_len, token_ids.device, is_chord_token
            )

        # Transformer 层
        for layer in self.layers:
            if self._gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, attention_mask, key_padding_mask,
                    use_reentrant=False
                )
            else:
                x = layer(x, attention_mask, key_padding_mask)

        # 输出
        x = self.output_norm(x)
        logits = self.lm_head(x)

        return logits

    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_length: int = 2048,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_id: int = 2,
        chord_ids: Optional[torch.Tensor] = None,
        instrument_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        自回归生成

        Args:
            prompt_ids: (batch, prompt_len) 提示 token IDs
            max_length: 最大生成长度
            temperature: 温度
            top_k: Top-K 采样
            top_p: Top-P (nucleus) 采样
            eos_id: EOS token ID
            chord_ids: 初始和弦索引
            instrument_ids: 初始乐器 ID

        Returns:
            generated_ids: (batch, generated_len)
        """
        self.eval()
        batch_size = prompt_ids.size(0)
        device = prompt_ids.device

        # 初始化
        generated = prompt_ids.clone()

        # 跟踪当前和弦和乐器
        if chord_ids is None:
            current_chord_ids = torch.zeros(batch_size, prompt_ids.size(1), dtype=torch.long, device=device)
        else:
            current_chord_ids = chord_ids.clone()

        if instrument_ids is None:
            current_inst_ids = torch.full((batch_size, prompt_ids.size(1)), 129, dtype=torch.long, device=device)
        else:
            current_inst_ids = instrument_ids.clone()

        with torch.no_grad():
            for _ in range(max_length - prompt_ids.size(1)):
                # 前向传播
                logits = self.forward(
                    generated,
                    chord_ids=current_chord_ids,
                    instrument_ids=current_inst_ids,
                )

                # 获取最后一个位置的 logits
                next_logits = logits[:, -1, :] / temperature

                # Top-K 过滤
                if top_k > 0:
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[indices_to_remove] = float('-inf')

                # Top-P 过滤
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_logits[indices_to_remove] = float('-inf')

                # 采样
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # 添加到生成序列
                generated = torch.cat([generated, next_token], dim=1)

                # 更新 chord_ids 和 instrument_ids
                # 简化处理：新 token 继承当前状态
                new_chord_id = current_chord_ids[:, -1:].clone()
                new_inst_id = current_inst_ids[:, -1:].clone()

                current_chord_ids = torch.cat([current_chord_ids, new_chord_id], dim=1)
                current_inst_ids = torch.cat([current_inst_ids, new_inst_id], dim=1)

                # 检查 EOS
                if (next_token == eos_id).all():
                    break

        return generated


def create_model(
    vocab_size: int = 415,
    model_size: str = 'base',
    **kwargs
) -> HIDMuseFormer:
    """
    创建模型的便捷函数

    Args:
        vocab_size: 词汇表大小
        model_size: 模型大小 ('small', 'base', 'large', 'xlarge')

    Returns:
        model: HIDMuseFormer 实例
    """
    # FFN 使用 LLaMA 风格的 8/3 × embed_dim × 2 ≈ 5.3× 比例
    # SwiGLU 实际上有 3 个矩阵 (gate, up, down)，所以需要更大的 FFN
    configs = {
        'small': {
            'embed_dim': 256,
            'num_layers': 6,
            'num_heads': 4,
            'ffn_dim': 1408,   # 5.5× (原 4×)
        },
        'base': {
            'embed_dim': 512,
            'num_layers': 12,
            'num_heads': 8,
            'ffn_dim': 2816,   # 5.5× (原 4×)
        },
        'large': {
            'embed_dim': 768,
            'num_layers': 16,
            'num_heads': 12,
            'ffn_dim': 4096,   # 5.3× (原 4×)
        },
        'xlarge': {
            'embed_dim': 1024,
            'num_layers': 24,
            'num_heads': 16,
            'ffn_dim': 5632,   # 5.5× (原 4×)
        },
    }

    config = configs.get(model_size, configs['base'])
    config.update(kwargs)

    return HIDMuseFormer(vocab_size=vocab_size, **config)


if __name__ == '__main__':
    # 测试
    print("HID-MuseFormer 模型测试")
    print("=" * 50)

    # 创建模型
    model = create_model(vocab_size=415, model_size='base')

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"模型大小: base")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"参数量 (M): {total_params / 1e6:.2f}M")

    # 测试前向传播
    batch_size = 2
    seq_len = 256

    token_ids = torch.randint(0, 415, (batch_size, seq_len))
    chord_ids = torch.randint(-1, 10, (batch_size, seq_len))
    instrument_ids = torch.randint(0, 130, (batch_size, seq_len))

    print(f"\n测试前向传播:")
    print(f"  输入形状: {token_ids.shape}")

    logits = model(token_ids, chord_ids, instrument_ids)
    print(f"  输出形状: {logits.shape}")

    # 测试不同大小的模型
    print(f"\n不同模型大小的参数量:")
    for size in ['small', 'base', 'large', 'xlarge']:
        m = create_model(vocab_size=415, model_size=size)
        params = sum(p.numel() for p in m.parameters())
        print(f"  {size}: {params / 1e6:.2f}M")
