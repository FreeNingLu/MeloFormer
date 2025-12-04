#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RMSNorm (Root Mean Square Layer Normalization)

比 LayerNorm 更简单高效:
- 去掉了 mean 计算
- BF16 下数值更稳定
- Llama, PaLM 等大模型标配

参考: https://arxiv.org/abs/1910.07467
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    公式: x * rsqrt(mean(x^2) + eps) * weight

    相比 LayerNorm:
    - 无 mean 中心化 (减少计算)
    - 无 bias (减少参数)
    - BF16 下更稳定
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Args:
            dim: 归一化维度
            eps: 数值稳定性常数
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """计算 RMS 归一化"""
        # x^2 的均值，然后取 rsqrt
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., dim) 输入张量

        Returns:
            归一化后的张量，形状不变
        """
        # 为了数值稳定性，在 float32 下计算归一化
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self) -> str:
        return f'{self.dim}, eps={self.eps}'


if __name__ == '__main__':
    # 简单测试
    print("RMSNorm 测试")

    # 创建测试数据
    batch, seq, dim = 2, 10, 64
    x = torch.randn(batch, seq, dim)

    # RMSNorm
    rms_norm = RMSNorm(dim)
    y_rms = rms_norm(x)

    # LayerNorm 对比
    layer_norm = nn.LayerNorm(dim)
    y_ln = layer_norm(x)

    print(f"输入: {x.shape}")
    print(f"RMSNorm 输出: {y_rms.shape}, mean={y_rms.mean():.4f}, std={y_rms.std():.4f}")
    print(f"LayerNorm 输出: {y_ln.shape}, mean={y_ln.mean():.4f}, std={y_ln.std():.4f}")

    # 参数对比
    rms_params = sum(p.numel() for p in rms_norm.parameters())
    ln_params = sum(p.numel() for p in layer_norm.parameters())
    print(f"\n参数量: RMSNorm={rms_params}, LayerNorm={ln_params}")

    # BF16 测试
    x_bf16 = x.bfloat16()
    rms_norm_bf16 = rms_norm.bfloat16()
    y_bf16 = rms_norm_bf16(x_bf16)
    print(f"\nBF16 输出: {y_bf16.dtype}, mean={y_bf16.float().mean():.4f}")

    print("\n测试通过!")
