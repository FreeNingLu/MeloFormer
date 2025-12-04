#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text2Summary 完整模型

整合:
- QwenTextEncoder: 文本编码
- FlowMatchingBridge: 文本 → Summary Token 桥接
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from transformers import AutoTokenizer

from .text_encoder import QwenTextEncoder
from .flow_matching import FlowMatchingBridge, TransformerFlowMatchingBridge


class Text2SummaryModel(nn.Module):
    """
    Text2Summary: 文本到 Summary Token 生成模型

    架构:
    1. QwenTextEncoder: 文本 → 嵌入 (1024 → 512)
    2. FlowMatchingBridge: 嵌入 → Summary Token (512)

    训练:
    - 输入: 文本 + Summary Token 真值
    - 输出: Flow Matching 损失

    推理:
    - 输入: 文本
    - 输出: 生成的 Summary Token (num_bars, 512)
    """

    def __init__(
        self,
        text_encoder_name: str = "Qwen/Qwen3-Embedding-0.6B",
        summary_dim: int = 512,
        bridge_hidden_dim: int = 1024,
        bridge_num_layers: int = 6,
        freeze_text_encoder: bool = True,
        use_transformer_bridge: bool = False,
    ):
        """
        Args:
            text_encoder_name: 文本编码器模型名
            summary_dim: Summary Token 维度
            bridge_hidden_dim: Bridge 隐藏层维度
            bridge_num_layers: Bridge 层数
            freeze_text_encoder: 是否冻结文本编码器
            use_transformer_bridge: 使用 Transformer 版本的 Bridge
        """
        super().__init__()

        self.summary_dim = summary_dim

        # 文本编码器
        self.text_encoder = QwenTextEncoder(
            model_name=text_encoder_name,
            target_dim=summary_dim,
            freeze_backbone=freeze_text_encoder,
        )

        # Flow Matching Bridge
        if use_transformer_bridge:
            self.bridge = TransformerFlowMatchingBridge(
                input_dim=summary_dim,
                cond_dim=summary_dim,
                hidden_dim=summary_dim,
                num_layers=bridge_num_layers,
                num_heads=8,
            )
        else:
            self.bridge = FlowMatchingBridge(
                input_dim=summary_dim,
                cond_dim=summary_dim,
                hidden_dim=bridge_hidden_dim,
                num_layers=bridge_num_layers,
            )

        # 用于 CFG 的空条件嵌入
        self.null_cond = nn.Parameter(torch.randn(1, summary_dim) * 0.01)

        # 归一化统计
        self.register_buffer('summary_mean', torch.zeros(summary_dim))
        self.register_buffer('summary_std', torch.ones(summary_dim))

    def set_normalization_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """设置归一化统计"""
        self.summary_mean.copy_(mean)
        self.summary_std.copy_(std)

    def normalize_summary(self, summary: torch.Tensor) -> torch.Tensor:
        """归一化 Summary Token"""
        return (summary - self.summary_mean) / (self.summary_std + 1e-6)

    def denormalize_summary(self, summary: torch.Tensor) -> torch.Tensor:
        """反归一化 Summary Token"""
        return summary * (self.summary_std + 1e-6) + self.summary_mean

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        编码文本

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            text_emb: (batch, summary_dim)
        """
        return self.text_encoder(input_ids, attention_mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        summary_tokens: torch.Tensor,
        num_bars: Optional[torch.Tensor] = None,
        cfg_dropout: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        训练前向传播

        Args:
            input_ids: (batch, seq_len) 文本 token IDs
            attention_mask: (batch, seq_len) 注意力掩码
            summary_tokens: (batch, num_bars, 512) Summary Token 真值
            num_bars: (batch,) 有效 bar 数量
            cfg_dropout: CFG dropout 概率

        Returns:
            {'loss': Tensor, 'text_emb': Tensor}
        """
        batch_size = input_ids.size(0)

        # 文本编码
        text_emb = self.encode_text(input_ids, attention_mask)  # (batch, 512)

        # CFG: 随机丢弃条件
        if self.training and cfg_dropout > 0:
            mask = torch.rand(batch_size, device=text_emb.device) < cfg_dropout
            text_emb = torch.where(
                mask.unsqueeze(-1),
                self.null_cond.expand(batch_size, -1),
                text_emb,
            )

        # 归一化 summary (如果设置了统计)
        summary_norm = self.normalize_summary(summary_tokens)

        # 起点: 文本嵌入扩展到 num_bars
        actual_num_bars = summary_tokens.size(1)
        x_0 = text_emb.unsqueeze(1).expand(-1, actual_num_bars, -1)

        # 终点: 归一化的 Summary Token
        x_1 = summary_norm

        # Flow Matching 损失
        loss = self.bridge.compute_loss(x_0, x_1, text_emb)

        return {
            'loss': loss,
            'text_emb': text_emb,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        text_emb: Optional[torch.Tensor] = None,
        num_bars: int = 16,
        steps: int = 20,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        生成 Summary Token

        Args:
            input_ids: 文本 token IDs (和 text_emb 二选一)
            attention_mask: 注意力掩码
            text_emb: 预计算的文本嵌入
            num_bars: 生成的 bar 数量
            steps: ODE 采样步数
            cfg_scale: CFG 强度

        Returns:
            summary_tokens: (batch, num_bars, 512)
        """
        # 获取文本嵌入
        if text_emb is None:
            text_emb = self.encode_text(input_ids, attention_mask)

        batch_size = text_emb.size(0)

        # 采样
        if cfg_scale > 1.0:
            null_cond = self.null_cond.expand(batch_size, -1)
            samples = self.bridge.sample_with_cfg(
                text_emb, num_bars, steps, cfg_scale, null_cond
            )
        else:
            samples = self.bridge.sample(text_emb, num_bars, steps)

        # 反归一化
        samples = self.denormalize_summary(samples)

        return samples

    def generate_from_text(
        self,
        texts: list,
        tokenizer,
        num_bars: int = 16,
        steps: int = 20,
        cfg_scale: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        便捷方法：直接从文本生成

        Args:
            texts: 文本列表
            tokenizer: 文本 tokenizer
            num_bars: bar 数量
            steps: 采样步数
            cfg_scale: CFG 强度
            device: 设备

        Returns:
            summary_tokens: (batch, num_bars, 512)
        """
        if device is None:
            device = next(self.parameters()).device

        # Tokenize
        inputs = tokenizer(
            texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        return self.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            num_bars=num_bars,
            steps=steps,
            cfg_scale=cfg_scale,
        )


class Text2SummaryTrainer:
    """训练器"""

    def __init__(
        self,
        model: Text2SummaryModel,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device: torch.device = None,
        grad_clip: float = 1.0,
        use_amp: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.grad_clip = grad_clip
        self.use_amp = use_amp

        self.model.to(self.device)

        if use_amp:
            self.scaler = torch.GradScaler()

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """单步训练"""
        self.model.train()
        self.optimizer.zero_grad()

        # 移动到设备
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        summary_tokens = batch['summary_tokens'].to(self.device)

        # 前向传播
        if self.use_amp:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = self.model(input_ids, attention_mask, summary_tokens)
                loss = outputs['loss']

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(input_ids, attention_mask, summary_tokens)
            loss = outputs['loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return {'loss': loss.item()}

    @torch.no_grad()
    def eval_step(self, batch: Dict) -> Dict[str, float]:
        """评估步"""
        self.model.eval()

        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        summary_tokens = batch['summary_tokens'].to(self.device)

        outputs = self.model(input_ids, attention_mask, summary_tokens)

        return {'loss': outputs['loss'].item()}


def create_model(
    text_encoder_name: str = "Qwen/Qwen3-Embedding-0.6B",
    use_transformer_bridge: bool = False,
    **kwargs,
) -> Text2SummaryModel:
    """便捷函数：创建模型"""
    return Text2SummaryModel(
        text_encoder_name=text_encoder_name,
        use_transformer_bridge=use_transformer_bridge,
        **kwargs,
    )


if __name__ == "__main__":
    print("=== Testing Text2SummaryModel ===\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    try:
        # 创建模型
        model = Text2SummaryModel(
            text_encoder_name="Qwen/Qwen3-Embedding-0.6B",
            summary_dim=512,
            bridge_hidden_dim=1024,
            bridge_num_layers=6,
            freeze_text_encoder=True,
            use_transformer_bridge=False,
        )

        model = model.to(device)

        # 统计参数
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

        print(f"Total params: {total_params:.2f}M")
        print(f"Trainable params: {trainable_params:.2f}M")

        # 测试生成
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)

        texts = [
            "A melodic piano piece in C major",
            "繁星点缀的夜空，钢琴声缓缓流淌",
        ]

        samples = model.generate_from_text(texts, tokenizer, num_bars=16, steps=20)
        print(f"\nGenerated samples shape: {samples.shape}")

    except Exception as e:
        print(f"Error: {e}")
        print("Please check if Qwen model is available")
