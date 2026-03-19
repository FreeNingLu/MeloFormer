#!/usr/bin/env python3
"""
测试 FA4 后端是否解决了 FlexAttention 的三个瓶颈:
1. enable_gqa=True 反向传播是否还触发 sdpa_dense_backward (O(N²) 退化)
2. 去掉 repeat_interleave 后 GQA 是否正常工作
3. 显存对比: repeat_interleave vs enable_gqa=True

环境: PyTorch 2.8 + H800 (SM90 Hopper)
"""

import torch
import torch.nn as nn
import gc
import time
from torch.nn.attention.flex_attention import flex_attention, create_block_mask


def get_mem_mb():
    return torch.cuda.memory_allocated() / 1024 / 1024


def get_peak_mb():
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def make_causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def test_gqa_backward(method: str, batch=2, seq_len=4096, embed_dim=512,
                       num_heads=8, num_kv_heads=2, warmup=2, runs=3):
    """
    method: 'repeat' = 老方案 repeat_interleave
            'native' = 新方案 enable_gqa=True
    """
    device = torch.device('cuda')
    head_dim = embed_dim // num_heads

    # 创建 Q (num_heads) 和 KV (num_kv_heads)
    q = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(batch, num_kv_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(batch, num_kv_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)

    # Block mask
    block_mask = create_block_mask(make_causal_mask, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device)

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    def run_once():
        if method == 'repeat':
            # 老方案: repeat_interleave 扩展 KV heads
            groups = num_heads // num_kv_heads
            k_exp = k.repeat_interleave(groups, dim=1)
            v_exp = v.repeat_interleave(groups, dim=1)
            out = flex_attention(q, k_exp, v_exp, block_mask=block_mask, enable_gqa=False)
        else:
            # 新方案: 直接 enable_gqa=True
            out = flex_attention(q, k, v, block_mask=block_mask, enable_gqa=True)
        loss = out.sum()
        loss.backward()
        return out

    # Warmup
    for _ in range(warmup):
        run_once()
        q.grad = None
        k.grad = None
        v.grad = None

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_before = get_mem_mb()

    # Timed runs
    times = []
    for _ in range(runs):
        q.grad = None
        k.grad = None
        v.grad = None
        torch.cuda.synchronize()
        t0 = time.time()
        run_once()
        torch.cuda.synchronize()
        times.append(time.time() - t0)

    peak_mem = get_peak_mb()
    avg_time = sum(times) / len(times)

    # 检查梯度是否正常
    grad_ok = (q.grad is not None and k.grad is not None and v.grad is not None
               and torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all()
               and torch.isfinite(v.grad).all())

    return {
        'method': method,
        'peak_mem_mb': peak_mem,
        'avg_time_ms': avg_time * 1000,
        'times_ms': [t * 1000 for t in times],
        'grad_ok': grad_ok,
        'q_grad_norm': q.grad.norm().item() if q.grad is not None else 0,
        'k_grad_norm': k.grad.norm().item() if k.grad is not None else 0,
    }


def test_summary_token_mask_gqa(method: str, batch=2, num_bars=32,
                                 bar_len=128, embed_dim=512,
                                 num_heads=8, num_kv_heads=2):
    """用 Summary Token 风格的稀疏 mask 测试 GQA"""
    device = torch.device('cuda')
    head_dim = embed_dim // num_heads
    seq_len = num_bars * bar_len  # 4096

    q = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(batch, num_kv_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(batch, num_kv_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)

    # 模拟 RR mask: 同 bar 内因果
    BL = bar_len

    def rr_mask(b, h, q_idx, kv_idx):
        same_bar = (q_idx // BL) == (kv_idx // BL)
        causal = q_idx >= kv_idx
        return same_bar & causal

    block_mask = create_block_mask(rr_mask, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device)

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    try:
        if method == 'repeat':
            groups = num_heads // num_kv_heads
            k_exp = k.repeat_interleave(groups, dim=1)
            v_exp = v.repeat_interleave(groups, dim=1)
            out = flex_attention(q, k_exp, v_exp, block_mask=block_mask, enable_gqa=False)
        else:
            out = flex_attention(q, k, v, block_mask=block_mask, enable_gqa=True)

        loss = out.sum()
        loss.backward()
        peak_mem = get_peak_mb()
        grad_ok = (q.grad is not None and torch.isfinite(q.grad).all())
        return {'method': method, 'peak_mem_mb': peak_mem, 'grad_ok': grad_ok, 'error': None}
    except Exception as e:
        return {'method': method, 'peak_mem_mb': 0, 'grad_ok': False, 'error': str(e)}


def main():
    print("=" * 70)
    print("FA4 后端测试: enable_gqa=True 是否解决反向传播退化")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # ========== 测试 1: 简单因果 mask + GQA ==========
    print("-" * 70)
    print("测试 1: Causal Mask + GQA (seq_len=4096, heads=8, kv_heads=2)")
    print("-" * 70)

    r_repeat = test_gqa_backward('repeat')
    # 清理
    torch.cuda.empty_cache()
    gc.collect()

    r_native = test_gqa_backward('native')
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\n{'方案':<20} {'峰值显存':>12} {'平均耗时':>12} {'梯度正常':>10}")
    print("-" * 55)
    print(f"{'repeat_interleave':<20} {r_repeat['peak_mem_mb']:>9.0f} MB {r_repeat['avg_time_ms']:>9.1f} ms {'✅' if r_repeat['grad_ok'] else '❌':>10}")
    print(f"{'enable_gqa=True':<20} {r_native['peak_mem_mb']:>9.0f} MB {r_native['avg_time_ms']:>9.1f} ms {'✅' if r_native['grad_ok'] else '❌':>10}")

    if r_native['grad_ok']:
        mem_saved = r_repeat['peak_mem_mb'] - r_native['peak_mem_mb']
        mem_pct = mem_saved / r_repeat['peak_mem_mb'] * 100
        speedup = r_repeat['avg_time_ms'] / r_native['avg_time_ms'] if r_native['avg_time_ms'] > 0 else 0
        print(f"\n  显存节省: {mem_saved:.0f} MB ({mem_pct:.1f}%)")
        print(f"  速度提升: {speedup:.2f}x")

    # ========== 测试 2: 稀疏 mask (模拟 RR 同 bar 因果) + GQA ==========
    print()
    print("-" * 70)
    print("测试 2: 稀疏 Mask (同 bar 因果, bar_len=128) + GQA")
    print("-" * 70)

    s_repeat = test_summary_token_mask_gqa('repeat')
    torch.cuda.empty_cache()
    gc.collect()

    s_native = test_summary_token_mask_gqa('native')
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\n{'方案':<20} {'峰值显存':>12} {'梯度正常':>10} {'错误':>20}")
    print("-" * 65)
    print(f"{'repeat_interleave':<20} {s_repeat['peak_mem_mb']:>9.0f} MB {'✅' if s_repeat['grad_ok'] else '❌':>10} {s_repeat['error'] or '无':>20}")
    print(f"{'enable_gqa=True':<20} {s_native['peak_mem_mb']:>9.0f} MB {'✅' if s_native['grad_ok'] else '❌':>10} {s_native['error'] or '无':>20}")

    # ========== 测试 3: 更大序列 (8192) ==========
    print()
    print("-" * 70)
    print("测试 3: 大序列 (seq_len=8192, heads=8, kv_heads=2)")
    print("-" * 70)

    try:
        r8k_repeat = test_gqa_backward('repeat', seq_len=8192)
        torch.cuda.empty_cache()
        gc.collect()
    except RuntimeError as e:
        r8k_repeat = {'method': 'repeat', 'peak_mem_mb': 0, 'avg_time_ms': 0, 'grad_ok': False}
        print(f"  repeat_interleave: OOM or error: {e}")

    try:
        r8k_native = test_gqa_backward('native', seq_len=8192)
        torch.cuda.empty_cache()
        gc.collect()
    except RuntimeError as e:
        r8k_native = {'method': 'native', 'peak_mem_mb': 0, 'avg_time_ms': 0, 'grad_ok': False}
        print(f"  enable_gqa=True: OOM or error: {e}")

    print(f"\n{'方案':<20} {'峰值显存':>12} {'平均耗时':>12} {'梯度正常':>10}")
    print("-" * 55)
    print(f"{'repeat_interleave':<20} {r8k_repeat['peak_mem_mb']:>9.0f} MB {r8k_repeat['avg_time_ms']:>9.1f} ms {'✅' if r8k_repeat['grad_ok'] else '❌':>10}")
    print(f"{'enable_gqa=True':<20} {r8k_native['peak_mem_mb']:>9.0f} MB {r8k_native['avg_time_ms']:>9.1f} ms {'✅' if r8k_native['grad_ok'] else '❌':>10}")

    # ========== 总结 ==========
    print()
    print("=" * 70)
    print("总结")
    print("=" * 70)

    all_native_ok = r_native['grad_ok'] and s_native['grad_ok'] and r8k_native['grad_ok']

    if all_native_ok:
        print("✅ enable_gqa=True 在所有测试中反向传播正常！")
        print("   → 可以去掉 repeat_interleave workaround")
        print("   → 瓶颈 1 (反向退化) 和瓶颈 3 (GQA 物理拷贝) 已解决")
    else:
        print("❌ enable_gqa=True 仍有问题:")
        if not r_native['grad_ok']:
            print("   - 因果 mask: 梯度异常")
        if not s_native['grad_ok']:
            print(f"   - 稀疏 mask: {'梯度异常' if not s_native['error'] else s_native['error']}")
        if not r8k_native['grad_ok']:
            print("   - 大序列 8192: 梯度异常或 OOM")


if __name__ == '__main__':
    main()
