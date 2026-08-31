#!/usr/bin/env python3
"""
v3.0 Layer 0: score_mod 编译兼容性测试

验证 FlexAttention 的 score_mod 中捕获的 nn.Parameter 值变化时，
是否触发 Triton 重编译。

预期结果：不触发（与 mask_mod 中 captured tensor 行为一致）。
如果触发：线 1a 的 Phase 2 bias 需要改为在 FlexAttention 之外施加。

用法：在有 GPU 的环境上运行
    python tests/test_score_mod_compile.py
"""

import torch
import torch.nn as nn
import time

def test_score_mod_no_recompile():
    """测试 score_mod 中的 Parameter 更新不触发重编译"""

    try:
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    except ImportError:
        print("SKIP: FlexAttention 不可用 (需要 PyTorch 2.5+)")
        return

    if not torch.cuda.is_available():
        print("SKIP: 需要 CUDA")
        return

    device = torch.device('cuda')
    B, H, N, D = 1, 4, 128, 64

    # 模拟 rs_bias / rr_bias
    rs_bias = nn.Parameter(torch.zeros(1, device=device))
    rr_bias = nn.Parameter(torch.zeros(1, device=device))
    split_point = N // 2  # 前半是 "summary"，后半是 "regular"

    def score_mod(score, b, h, q_idx, kv_idx):
        is_summary = kv_idx < split_point
        return torch.where(is_summary, score + rs_bias, score + rr_bias)

    def mask_mod(b, h, q_idx, kv_idx):
        return q_idx >= 0  # 全可见

    q = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)

    block_mask = create_block_mask(mask_mod, B, H, N, N, device=device)

    compiled_flex = torch.compile(flex_attention)

    # ── 第一次调用（触发编译，预期较慢）──
    print("第 1 次调用（初始编译）...")
    torch.cuda.synchronize()
    t0 = time.time()
    out1 = compiled_flex(q, k, v, block_mask=block_mask, score_mod=score_mod)
    torch.cuda.synchronize()
    t1 = time.time() - t0
    print(f"  耗时: {t1:.3f}s")

    # ── 第二次调用（相同参数值，应走缓存）──
    print("第 2 次调用（相同参数值）...")
    torch.cuda.synchronize()
    t0 = time.time()
    out2 = compiled_flex(q, k, v, block_mask=block_mask, score_mod=score_mod)
    torch.cuda.synchronize()
    t2 = time.time() - t0
    print(f"  耗时: {t2:.3f}s")

    # ── 修改参数值 ──
    with torch.no_grad():
        rs_bias.fill_(0.5)
        rr_bias.fill_(-0.3)

    # ── 第三次调用（参数值变化，关键测试）──
    print("第 3 次调用（参数值已变化）...")
    torch.cuda.synchronize()
    t0 = time.time()
    out3 = compiled_flex(q, k, v, block_mask=block_mask, score_mod=score_mod)
    torch.cuda.synchronize()
    t3 = time.time() - t0
    print(f"  耗时: {t3:.3f}s")

    # ── 再多跑几次确认稳定 ──
    times = []
    for i in range(5):
        with torch.no_grad():
            rs_bias.fill_(float(i) * 0.1)
        torch.cuda.synchronize()
        t0 = time.time()
        _ = compiled_flex(q, k, v, block_mask=block_mask, score_mod=score_mod)
        torch.cuda.synchronize()
        times.append(time.time() - t0)

    avg_subsequent = sum(times) / len(times)
    print(f"\n后续 5 次平均耗时: {avg_subsequent:.3f}s")

    # ── 判定 ──
    # 编译通常需要 5-30s，缓存命中 < 0.1s
    # 如果第 3 次耗时接近第 2 次（而非第 1 次），则说明没有重编译
    recompile_threshold = t1 * 0.3  # 如果超过首次编译时间的 30%，认为重编译了

    if t3 < recompile_threshold:
        print(f"\n✓ 测试通过：score_mod 中 Parameter 值变化不触发重编译")
        print(f"  首次编译: {t1:.3f}s, 参数变化后: {t3:.3f}s")
        return True
    else:
        print(f"\n✗ 测试失败：score_mod 中 Parameter 值变化触发了重编译！")
        print(f"  首次编译: {t1:.3f}s, 参数变化后: {t3:.3f}s")
        print(f"  → 线 1a Phase 2 需要改为输出后处理方案")
        return False

    # 验证输出不同（确认 bias 确实生效）
    if not torch.allclose(out1, out3):
        print("✓ 确认：不同 bias 值产生不同输出（bias 生效）")
    else:
        print("⚠ 警告：不同 bias 值产生相同输出（bias 可能未生效）")


if __name__ == '__main__':
    test_score_mod_no_recompile()
