#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-Embedding 文本编码器

封装 Qwen3-Embedding-0.6B 用于文本到 Summary Token 的桥接
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Dict
from transformers import AutoModel, AutoTokenizer


class QwenTextEncoder(nn.Module):
    """
    Qwen3-Embedding 文本编码器

    将文本编码为固定维度的嵌入向量，用于 Diffusion Bridge 的输入。

    特点:
    - 使用预训练的 Qwen3-Embedding-0.6B (1024 维输出)
    - 可选的可学习投影层 (1024 → target_dim)
    - 支持冻结/解冻基础模型
    - 支持 mean pooling 或 CLS token
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        target_dim: int = 512,
        freeze_backbone: bool = True,
        pooling: str = "mean",  # "mean" or "cls"
        use_projection: bool = True,
    ):
        """
        Args:
            model_name: HuggingFace 模型名称或本地路径
            target_dim: 输出嵌入维度 (Summary Token 维度)
            freeze_backbone: 是否冻结 Qwen 基础模型
            pooling: 池化方式 ("mean" 或 "cls")
            use_projection: 是否使用投影层
        """
        super().__init__()

        self.model_name = model_name
        self.target_dim = target_dim
        self.freeze_backbone = freeze_backbone
        self.pooling = pooling
        self.use_projection = use_projection

        # 加载 Qwen 模型
        self.backbone = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.hidden_dim = self.backbone.config.hidden_size  # 1024 for 0.6B

        # 冻结 backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 投影层
        if use_projection:
            self.proj = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, target_dim),
            )
        else:
            self.proj = nn.Identity() if self.hidden_dim == target_dim else nn.Linear(self.hidden_dim, target_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            input_ids: (batch, seq_len) Token IDs
            attention_mask: (batch, seq_len) 注意力掩码

        Returns:
            embeddings: (batch, target_dim) 文本嵌入
        """
        # Backbone forward
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        else:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

        # Pooling
        if self.pooling == "cls":
            # 使用第一个 token
            pooled = hidden_states[:, 0, :]
        elif self.pooling == "mean":
            # Mean pooling (考虑 attention mask)
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_embeddings / sum_mask
            else:
                pooled = hidden_states.mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # 投影
        embeddings = self.proj(pooled)

        return embeddings

    def encode_text(
        self,
        texts: Union[str, list],
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = 512,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        便捷方法：直接从文本字符串编码

        Args:
            texts: 单个文本或文本列表
            tokenizer: 可选的 tokenizer
            max_length: 最大序列长度
            device: 设备

        Returns:
            embeddings: (batch, target_dim)
        """
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        if isinstance(texts, str):
            texts = [texts]

        # Tokenize
        inputs = tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        elif hasattr(self, 'proj') and hasattr(self.proj, 'weight'):
            device = next(self.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

        return self.forward(inputs['input_ids'], inputs.get('attention_mask'))

    def unfreeze_backbone(self, unfreeze_layers: int = -1):
        """
        解冻部分 backbone 层用于微调

        Args:
            unfreeze_layers: 解冻最后 N 层，-1 表示全部解冻
        """
        if unfreeze_layers == -1:
            for param in self.backbone.parameters():
                param.requires_grad = True
            self.freeze_backbone = False
        else:
            # 只解冻最后 N 层
            layers = list(self.backbone.encoder.layer)
            for layer in layers[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True


class QwenTextEncoderWithTokenizer(nn.Module):
    """
    带 Tokenizer 的文本编码器 (便捷封装)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        target_dim: int = 512,
        freeze_backbone: bool = True,
        max_length: int = 512,
    ):
        super().__init__()

        self.encoder = QwenTextEncoder(
            model_name=model_name,
            target_dim=target_dim,
            freeze_backbone=freeze_backbone,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.max_length = max_length

    def forward(self, texts: Union[str, list]) -> torch.Tensor:
        """
        Args:
            texts: 文本或文本列表

        Returns:
            embeddings: (batch, target_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        device = next(self.encoder.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        return self.encoder(inputs['input_ids'], inputs['attention_mask'])


if __name__ == "__main__":
    print("=== Testing QwenTextEncoder ===\n")

    # 注意: 需要先下载模型
    # 测试时可以用较小的模型或 mock
    try:
        encoder = QwenTextEncoder(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            target_dim=512,
            freeze_backbone=True,
        )

        params_total = sum(p.numel() for p in encoder.parameters()) / 1e6
        params_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad) / 1e6

        print(f"Total params: {params_total:.2f}M")
        print(f"Trainable params: {params_trainable:.2f}M")

        # 测试 encode
        texts = [
            "A melodic piano piece in C major",
            "繁星点缀的夜空，钢琴声缓缓流淌",
        ]

        embeddings = encoder.encode_text(texts)
        print(f"\nInput: {len(texts)} texts")
        print(f"Output shape: {embeddings.shape}")  # Expected: (2, 512)

    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please download the model first or adjust the model path")
