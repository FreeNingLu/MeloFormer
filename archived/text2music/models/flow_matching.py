#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flow Matching Bridge - Diffusion Bridge 实现

使用 Rectified Flow / Flow Matching 实现文本到 Summary Token 的桥接。

核心思想:
- 不是从噪声到数据的扩散，而是从 text embedding 到 summary token 的插值
- 学习速度场 v(x_t, t, condition)
- ODE 积分完成采样，比 SDE 更快更稳定

参考:
- Rectified Flow (ICLR 2023)
- Flow Matching for Generative Modeling (ICLR 2023)
- I2SB: Image-to-Image Schrödinger Bridge (NeurIPS 2023)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
from tqdm import tqdm


class SinusoidalPositionEmbedding(nn.Module):
    """正弦位置编码用于时间嵌入"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (batch,) 时间步 [0, 1]

        Returns:
            emb: (batch, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class FlowMatchingBridge(nn.Module):
    """
    Flow Matching Bridge: 文本 → Summary Token

    架构:
    - 时间嵌入: Sinusoidal + MLP
    - 条件注入: 拼接或加性
    - 速度场网络: MLP with residual connections

    训练:
    - 损失: MSE(v_pred, v_true) where v_true = x_1 - x_0
    - x_t = (1-t) * x_0 + t * x_1 (线性插值)

    采样:
    - ODE 积分: dx/dt = v(x, t, cond)
    - 使用 Euler 或 RK4 求解
    """

    def __init__(
        self,
        input_dim: int = 512,      # x_t 维度 (Summary Token)
        cond_dim: int = 512,        # 条件维度 (Text Embedding)
        hidden_dim: int = 1024,     # 隐藏层维度
        num_layers: int = 6,        # 层数
        dropout: float = 0.1,
        time_embed_dim: Optional[int] = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim

        time_embed_dim = time_embed_dim or hidden_dim

        # 时间嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(hidden_dim),
            nn.Linear(hidden_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # 条件投影
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 速度场网络 (with residual connections)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                ResidualBlock(hidden_dim, time_embed_dim, dropout)
            )

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(
        self,
        x_t: torch.Tensor,      # (batch, input_dim) 或 (batch, num_bars, input_dim)
        t: torch.Tensor,        # (batch,) 时间 [0, 1]
        cond: torch.Tensor,     # (batch, cond_dim) 或 (batch, num_bars, cond_dim)
    ) -> torch.Tensor:
        """
        预测速度场 v(x_t, t, cond)

        Args:
            x_t: 当前状态
            t: 时间步
            cond: 条件 (text embedding)

        Returns:
            v: 预测的速度场，与 x_t 同形状
        """
        # 处理不同维度情况
        original_shape = x_t.shape
        if x_t.dim() == 3:
            # (batch, num_bars, dim) -> (batch * num_bars, dim)
            batch_size, num_bars, dim = x_t.shape
            x_t = x_t.view(-1, dim)
            t = t.repeat_interleave(num_bars)
            if cond.dim() == 2:
                cond = cond.unsqueeze(1).expand(-1, num_bars, -1)
            cond = cond.view(-1, cond.size(-1))
        else:
            batch_size = x_t.size(0)
            num_bars = None

        # 时间嵌入
        t_emb = self.time_embed(t)  # (batch, time_embed_dim)

        # 输入和条件投影
        h = self.input_proj(x_t)  # (batch, hidden_dim)
        c = self.cond_proj(cond)  # (batch, hidden_dim)

        # 融合条件
        h = h + c

        # 通过残差层
        for layer in self.layers:
            h = layer(h, t_emb)

        # 输出
        v = self.output_proj(h)

        # 恢复原始形状
        if num_bars is not None:
            v = v.view(batch_size, num_bars, -1)

        return v

    def compute_loss(
        self,
        x_0: torch.Tensor,      # 起点 (text embedding extended)
        x_1: torch.Tensor,      # 终点 (Summary Token, 真值)
        cond: torch.Tensor,     # 条件
        weights: Optional[torch.Tensor] = None,  # 可选的样本权重
    ) -> torch.Tensor:
        """
        计算 Flow Matching 损失

        Args:
            x_0: 起点 (batch, [num_bars,] dim)
            x_1: 终点 (batch, [num_bars,] dim)
            cond: 条件 (batch, cond_dim)
            weights: 可选的损失权重

        Returns:
            loss: 标量损失
        """
        batch_size = x_0.size(0)
        device = x_0.device

        # 随机采样时间 t ~ U(0, 1)
        t = torch.rand(batch_size, device=device)

        # 线性插值: x_t = (1-t) * x_0 + t * x_1
        if x_0.dim() == 3:
            # (batch, num_bars, dim)
            t_expand = t.view(-1, 1, 1)
        else:
            t_expand = t.view(-1, 1)

        x_t = (1 - t_expand) * x_0 + t_expand * x_1

        # 真实速度: v = x_1 - x_0 (因为 dx/dt = x_1 - x_0 对于线性插值)
        v_true = x_1 - x_0

        # 预测速度
        v_pred = self.forward(x_t, t, cond)

        # MSE 损失
        if weights is not None:
            loss = (weights * (v_pred - v_true).pow(2)).mean()
        else:
            loss = F.mse_loss(v_pred, v_true)

        return loss

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,     # (batch, cond_dim) 条件
        num_bars: int = 16,     # 生成的 bar 数量
        steps: int = 20,        # ODE 步数
        method: str = "euler",  # 积分方法: "euler" 或 "heun"
        noise_scale: float = 0.1,  # 初始噪声缩放
    ) -> torch.Tensor:
        """
        ODE 采样

        Args:
            cond: 条件嵌入 (batch, cond_dim)
            num_bars: 生成的 bar 数量
            steps: ODE 积分步数
            method: 积分方法
            noise_scale: 初始噪声缩放

        Returns:
            samples: (batch, num_bars, input_dim)
        """
        batch_size = cond.size(0)
        device = cond.device

        # 初始化 x_0: 从条件扩展 + 噪声
        x = cond.unsqueeze(1).expand(-1, num_bars, -1).clone()  # (batch, num_bars, cond_dim)

        # 投影到 input_dim (如果需要)
        if self.cond_dim != self.input_dim:
            x = F.pad(x, (0, self.input_dim - self.cond_dim))[:, :, :self.input_dim]

        # 添加噪声
        x = x + torch.randn_like(x) * noise_scale

        # ODE 积分
        dt = 1.0 / steps

        for i in range(steps):
            t = torch.full((batch_size,), i / steps, device=device)

            if method == "euler":
                # Euler 方法
                v = self.forward(x, t, cond)
                x = x + v * dt

            elif method == "heun":
                # Heun 方法 (改进的 Euler，二阶精度)
                v1 = self.forward(x, t, cond)
                x_pred = x + v1 * dt

                t_next = torch.full((batch_size,), (i + 1) / steps, device=device)
                v2 = self.forward(x_pred, t_next, cond)

                x = x + (v1 + v2) * dt / 2

            else:
                raise ValueError(f"Unknown method: {method}")

        return x

    @torch.no_grad()
    def sample_with_cfg(
        self,
        cond: torch.Tensor,
        num_bars: int = 16,
        steps: int = 20,
        cfg_scale: float = 1.0,
        null_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Classifier-Free Guidance 采样

        Args:
            cond: 条件嵌入
            num_bars: bar 数量
            steps: 步数
            cfg_scale: CFG 强度 (1.0 = 无 CFG)
            null_cond: 空条件 (用于 CFG)

        Returns:
            samples: (batch, num_bars, input_dim)
        """
        if cfg_scale == 1.0 or null_cond is None:
            return self.sample(cond, num_bars, steps)

        batch_size = cond.size(0)
        device = cond.device

        # 初始化
        x = cond.unsqueeze(1).expand(-1, num_bars, -1).clone()
        if self.cond_dim != self.input_dim:
            x = F.pad(x, (0, self.input_dim - self.cond_dim))[:, :, :self.input_dim]
        x = x + torch.randn_like(x) * 0.1

        dt = 1.0 / steps

        for i in range(steps):
            t = torch.full((batch_size,), i / steps, device=device)

            # 条件速度
            v_cond = self.forward(x, t, cond)

            # 无条件速度
            v_uncond = self.forward(x, t, null_cond.expand(batch_size, -1))

            # CFG 组合
            v = v_uncond + cfg_scale * (v_cond - v_uncond)

            x = x + v * dt

        return x


class ResidualBlock(nn.Module):
    """残差块 with 时间调制"""

    def __init__(
        self,
        hidden_dim: int,
        time_embed_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        # 时间调制
        self.time_proj = nn.Linear(time_embed_dim, hidden_dim * 2)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,        # (batch, hidden_dim)
        t_emb: torch.Tensor,    # (batch, time_embed_dim)
    ) -> torch.Tensor:
        # 时间调制参数
        t_out = self.time_proj(t_emb)
        scale, shift = t_out.chunk(2, dim=-1)

        # 第一层
        h = self.norm1(x)
        h = h * (1 + scale) + shift  # AdaLN 风格调制
        h = self.linear1(h)
        h = F.silu(h)
        h = self.dropout(h)

        # 第二层
        h = self.norm2(h)
        h = self.linear2(h)
        h = self.dropout(h)

        return x + h


class TransformerFlowMatchingBridge(nn.Module):
    """
    使用 Transformer 架构的 Flow Matching Bridge

    适用于需要处理 bar 序列依赖关系的场景
    """

    def __init__(
        self,
        input_dim: int = 512,
        cond_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_bars: int = 256,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim

        # 时间嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 位置编码
        self.pos_embed = nn.Embedding(max_bars, hidden_dim)

        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 条件投影
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        # Transformer 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(
        self,
        x_t: torch.Tensor,      # (batch, num_bars, input_dim)
        t: torch.Tensor,        # (batch,)
        cond: torch.Tensor,     # (batch, cond_dim)
    ) -> torch.Tensor:
        """预测速度场"""
        batch_size, num_bars, _ = x_t.shape
        device = x_t.device

        # 时间嵌入
        t_emb = self.time_embed(t)  # (batch, hidden_dim)

        # 输入投影
        h = self.input_proj(x_t)  # (batch, num_bars, hidden_dim)

        # 位置编码
        pos = torch.arange(num_bars, device=device)
        h = h + self.pos_embed(pos).unsqueeze(0)

        # 条件投影并广播
        c = self.cond_proj(cond).unsqueeze(1)  # (batch, 1, hidden_dim)
        h = h + c

        # 时间嵌入广播
        h = h + t_emb.unsqueeze(1)

        # Transformer
        h = self.transformer(h)

        # 输出
        v = self.output_proj(h)

        return v

    def compute_loss(self, x_0, x_1, cond, weights=None):
        """计算损失 (与 FlowMatchingBridge 相同)"""
        batch_size = x_0.size(0)
        device = x_0.device

        t = torch.rand(batch_size, device=device)
        t_expand = t.view(-1, 1, 1)

        x_t = (1 - t_expand) * x_0 + t_expand * x_1
        v_true = x_1 - x_0
        v_pred = self.forward(x_t, t, cond)

        if weights is not None:
            loss = (weights * (v_pred - v_true).pow(2)).mean()
        else:
            loss = F.mse_loss(v_pred, v_true)

        return loss

    @torch.no_grad()
    def sample(self, cond, num_bars=16, steps=20, method="euler", noise_scale=0.1):
        """ODE 采样"""
        batch_size = cond.size(0)
        device = cond.device

        # 初始化
        x = cond.unsqueeze(1).expand(-1, num_bars, -1).clone()
        if self.cond_dim != self.input_dim:
            x = F.pad(x, (0, self.input_dim - self.cond_dim))[:, :, :self.input_dim]
        x = x + torch.randn_like(x) * noise_scale

        dt = 1.0 / steps

        for i in range(steps):
            t = torch.full((batch_size,), i / steps, device=device)
            v = self.forward(x, t, cond)
            x = x + v * dt

        return x


if __name__ == "__main__":
    print("=== Testing Flow Matching Bridge ===\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 测试 MLP 版本
    print("\n--- MLP Bridge ---")
    bridge = FlowMatchingBridge(
        input_dim=512,
        cond_dim=512,
        hidden_dim=1024,
        num_layers=6,
    ).to(device)

    params = sum(p.numel() for p in bridge.parameters()) / 1e6
    print(f"Parameters: {params:.2f}M")

    # 测试 forward
    batch_size = 4
    num_bars = 16
    x_t = torch.randn(batch_size, num_bars, 512, device=device)
    t = torch.rand(batch_size, device=device)
    cond = torch.randn(batch_size, 512, device=device)

    v = bridge(x_t, t, cond)
    print(f"Input: x_t {x_t.shape}, t {t.shape}, cond {cond.shape}")
    print(f"Output: v {v.shape}")

    # 测试损失
    x_0 = cond.unsqueeze(1).expand(-1, num_bars, -1)
    x_1 = torch.randn(batch_size, num_bars, 512, device=device)
    loss = bridge.compute_loss(x_0, x_1, cond)
    print(f"Loss: {loss.item():.4f}")

    # 测试采样
    samples = bridge.sample(cond, num_bars=16, steps=20)
    print(f"Samples: {samples.shape}")

    # 测试 Transformer 版本
    print("\n--- Transformer Bridge ---")
    bridge_tf = TransformerFlowMatchingBridge(
        input_dim=512,
        cond_dim=512,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
    ).to(device)

    params_tf = sum(p.numel() for p in bridge_tf.parameters()) / 1e6
    print(f"Parameters: {params_tf:.2f}M")

    v_tf = bridge_tf(x_t, t, cond)
    print(f"Output: v {v_tf.shape}")

    samples_tf = bridge_tf.sample(cond, num_bars=16, steps=20)
    print(f"Samples: {samples_tf.shape}")

    print("\n=== All Tests Passed ===")
