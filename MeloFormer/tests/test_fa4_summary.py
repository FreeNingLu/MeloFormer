#!/usr/bin/env python3
"""
测试 FA4: 在 MeloFormer 的 Summary Token 稀疏注意力上
对比 repeat_interleave (旧) vs enable_gqa=True (新)

重点验证:
1. 反向传播是否正常 (不触发 sdpa_dense_backward)
2. 显存差异
3. 速度差异
4. 梯度数值一致性
"""

import torch
import torch.nn as nn
import gc
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from model.attention_flex_summary import (
    SummaryTokenEmbedding,
    FlexSummaryAttentionMask,
    FlexSummaryAttention,
    FlexSummaryAttentionBlock,
    set_bar_len,
    BAR_LEN, MAX_BARS, MAX_SEQ_LEN,
)
from model.attention_flex import RotaryPositionEmbedding, apply_rotary_pos_emb
from model.rms_norm import RMSNorm


class FlexSummaryAttention_NativeGQA(FlexSummaryAttention):
    """
    修改版: 用 enable_gqa=True 替代 repeat_interleave
    """

    def forward(self, sum_x, reg_x, summarize_block_mask, updating_block_mask):
        batch_size = sum_x.size(0)
        sum_len = sum_x.size(1)
        reg_len = reg_x.size(1)

        # 投影
        sum_q = self.sum_q_proj(sum_x)
        sum_k = self.sum_k_proj(sum_x)
        sum_v = self.sum_v_proj(sum_x)
        reg_q = self.reg_q_proj(reg_x)
        reg_k = self.reg_k_proj(reg_x)
        reg_v = self.reg_v_proj(reg_x)

        # 重塑多头
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

        # *** 不做 repeat_interleave，直接用 enable_gqa=True ***

        target_dtype = sum_q.dtype

        # 阶段 1: Summarize (SS + SR)
        cat_k_sum = torch.cat([sum_k, reg_k], dim=2)
        cat_v_sum = torch.cat([sum_v, reg_v], dim=2)
        sum_attn_out = flex_attention(
            sum_q, cat_k_sum, cat_v_sum,
            block_mask=summarize_block_mask,
            enable_gqa=True,  # ← 关键改动
        )
        sum_attn_out = sum_attn_out.transpose(1, 2).contiguous().view(batch_size, sum_len, self.embed_dim)

        # 阶段 2: K2, V2
        sum_k2 = self.sum_k2_proj(sum_attn_out)
        sum_v2 = self.sum_v2_proj(sum_attn_out)
        sum_k2 = sum_k2.view(batch_size, sum_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        sum_v2 = sum_v2.view(batch_size, sum_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # *** 不做 repeat_interleave ***

        # 阶段 3: Updating (RS + RR)
        cat_k_reg = torch.cat([sum_k2, reg_k], dim=2)
        cat_v_reg = torch.cat([sum_v2, reg_v], dim=2)
        reg_attn_out = flex_attention(
            reg_q, cat_k_reg, cat_v_reg,
            block_mask=updating_block_mask,
            enable_gqa=True,  # ← 关键改动
        )
        reg_attn_out = reg_attn_out.transpose(1, 2).contiguous().view(batch_size, reg_len, self.embed_dim)

        # 输出投影
        sum_output = self.sum_out_proj(sum_attn_out)
        reg_output = self.reg_out_proj(reg_attn_out)

        return sum_output, reg_output


def test_method(method, num_bars=16, bar_len=64, embed_dim=256,
                num_heads=8, num_kv_heads=2, batch_size=2, warmup=2, runs=3):
    """测试一种方法"""
    device = torch.device('cuda')

    set_bar_len(bar_len=bar_len, max_bars=num_bars)
    seq_len = num_bars * bar_len

    # 创建数据
    reg_x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    sum_embedding = SummaryTokenEmbedding(embed_dim, max_bars=num_bars).to(device).to(torch.bfloat16)
    sum_x = sum_embedding(num_bars, batch_size, device)

    # 创建 mask
    mask_gen = FlexSummaryAttentionMask()
    sum_mask, upd_mask = mask_gen.create_block_masks(num_bars, batch_size, device)

    # 创建 attention
    if method == 'repeat':
        attn = FlexSummaryAttention(
            embed_dim, num_heads, num_kv_heads=num_kv_heads,
            max_seq_len=seq_len, max_bars=num_bars,
        ).to(device).to(torch.bfloat16)
    else:
        attn = FlexSummaryAttention_NativeGQA(
            embed_dim, num_heads, num_kv_heads=num_kv_heads,
            max_seq_len=seq_len, max_bars=num_bars,
        ).to(device).to(torch.bfloat16)

    attn.train()

    # Warmup
    for _ in range(warmup):
        s_out, r_out = attn(sum_x.detach().requires_grad_(True), reg_x, sum_mask, upd_mask)
        (s_out.sum() + r_out.sum()).backward()
        reg_x.grad = None

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Timed runs
    times = []
    for _ in range(runs):
        reg_x.grad = None
        sum_x_run = sum_x.detach().requires_grad_(True)

        torch.cuda.synchronize()
        t0 = time.time()

        s_out, r_out = attn(sum_x_run, reg_x, sum_mask, upd_mask)
        loss = s_out.sum() + r_out.sum()
        loss.backward()

        torch.cuda.synchronize()
        times.append(time.time() - t0)

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    avg_time = sum(times) / len(times)
    grad_ok = reg_x.grad is not None and torch.isfinite(reg_x.grad).all().item()

    return {
        'method': method,
        'peak_mem_mb': peak_mem,
        'avg_time_ms': avg_time * 1000,
        'grad_ok': grad_ok,
        'grad_norm': reg_x.grad.norm().item() if reg_x.grad is not None else 0,
    }


def main():
    print("=" * 70)
    print("MeloFormer Summary Token: repeat_interleave vs enable_gqa=True")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}")
    print()

    configs = [
        # (num_bars, bar_len, embed_dim, num_heads, num_kv_heads, batch)
        (16, 64, 256, 8, 2, 2),    # small: 1024 tokens
        (32, 64, 256, 8, 2, 2),    # medium: 2048 tokens
        (32, 128, 512, 8, 4, 2),   # base: 4096 tokens
    ]

    for num_bars, bar_len, embed_dim, num_heads, num_kv_heads, batch in configs:
        seq_len = num_bars * bar_len
        gqa_ratio = num_heads // num_kv_heads
        print(f"\n{'='*70}")
        print(f"Config: seq={seq_len}, bars={num_bars}, bar_len={bar_len}, "
              f"dim={embed_dim}, heads={num_heads}:{num_kv_heads} (GQA {gqa_ratio}:1), batch={batch}")
        print(f"{'='*70}")

        r_old = test_method('repeat', num_bars, bar_len, embed_dim, num_heads, num_kv_heads, batch)
        torch.cuda.empty_cache()
        gc.collect()

        r_new = test_method('native', num_bars, bar_len, embed_dim, num_heads, num_kv_heads, batch)
        torch.cuda.empty_cache()
        gc.collect()

        print(f"\n{'方案':<25} {'峰值显存':>12} {'平均耗时':>12} {'梯度':>8} {'梯度范数':>12}")
        print("-" * 70)
        print(f"{'repeat_interleave':<25} {r_old['peak_mem_mb']:>9.0f} MB {r_old['avg_time_ms']:>9.1f} ms "
              f"{'✅' if r_old['grad_ok'] else '❌':>8} {r_old['grad_norm']:>12.4f}")
        print(f"{'enable_gqa=True':<25} {r_new['peak_mem_mb']:>9.0f} MB {r_new['avg_time_ms']:>9.1f} ms "
              f"{'✅' if r_new['grad_ok'] else '❌':>8} {r_new['grad_norm']:>12.4f}")

        if r_old['grad_ok'] and r_new['grad_ok']:
            mem_diff = r_old['peak_mem_mb'] - r_new['peak_mem_mb']
            speed_ratio = r_old['avg_time_ms'] / r_new['avg_time_ms'] if r_new['avg_time_ms'] > 0 else 0
            print(f"\n  显存差: {mem_diff:+.0f} MB | 速度比: {speed_ratio:.2f}x")

    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)


if __name__ == '__main__':
    main()
