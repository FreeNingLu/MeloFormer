#!/usr/bin/env python3
"""
层数对比测试 - 验证是否真正全量 GC

原理：
- 真·全量 GC：显存与层数几乎无关（只多一点权重）
- 假·全量 GC：显存随层数线性增长

运行方法：
    python test_gc_layers.py
"""

import torch
import torch.nn as nn
import gc
from model.meloformer import MeloFormer


def get_gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def test_layers(num_layers, batch_size=1, seq_len=4096, num_bars=64, use_gc=True):
    """测试指定层数的显存占用"""
    device = torch.device('cuda')

    # 创建模型 (固定 hidden_dim=512，只改层数)
    model = MeloFormer(
        vocab_size=643,
        embed_dim=512,
        num_layers=num_layers,
        num_heads=8,
        ffn_dim=2816,
        max_seq_len=seq_len,
    ).to(device)

    if use_gc:
        model._gradient_checkpointing = True
        model._autocast_dtype = torch.bfloat16
    else:
        model._gradient_checkpointing = False

    model.train()

    # 计算参数量
    params_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024

    # 创建假数据
    token_ids = torch.randint(0, 643, (batch_size, seq_len), device=device)
    chord_ids = torch.repeat_interleave(
        torch.arange(num_bars, device=device),
        seq_len // num_bars
    )[:seq_len].unsqueeze(0).expand(batch_size, -1)
    instrument_ids = (torch.arange(seq_len, device=device) % 4).unsqueeze(0).expand(batch_size, -1)
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    note_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1) % 128

    # 清理
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    # 前向 + 反向
    # 注意: FlexAttention 要求前向和反向都在相同的 autocast 上下文中
    try:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(
                token_ids, chord_ids, instrument_ids,
                token_type_ids=token_type_ids, note_ids=note_ids,
                num_bars=num_bars
            )
            loss = logits.mean()
            loss.backward()  # 必须在 autocast 上下文内

        peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        activation_mem = peak_mem - params_mb

        del model, logits, loss
        torch.cuda.empty_cache()
        gc.collect()

        return peak_mem, params_mb, activation_mem

    except RuntimeError as e:
        if "out of memory" in str(e):
            del model
            torch.cuda.empty_cache()
            gc.collect()
            return float('inf'), params_mb, float('inf')
        raise


def main():
    if not torch.cuda.is_available():
        print("需要 CUDA GPU")
        return

    print("=" * 70)
    print("层数对比测试 - 验证全量 GC")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()
    print("原理：")
    print("  - 真·全量 GC：激活显存与层数几乎无关")
    print("  - 假·全量 GC：激活显存随层数线性增长")
    print()

    # 测试不同层数
    layers_to_test = [6, 12, 18, 24]

    print("=" * 70)
    print("测试 1：不使用 GC")
    print("=" * 70)
    print(f"{'层数':<10} {'峰值显存':>12} {'参数显存':>12} {'激活显存':>12}")
    print("-" * 70)

    no_gc_results = []
    for num_layers in layers_to_test:
        peak, params, activation = test_layers(num_layers, use_gc=False)
        if peak != float('inf'):
            print(f"{num_layers:<10} {peak:>10.0f} MB {params:>10.0f} MB {activation:>10.0f} MB")
            no_gc_results.append((num_layers, activation))
        else:
            print(f"{num_layers:<10} {'OOM':>12}")

    print()
    print("=" * 70)
    print("测试 2：使用全量 GC")
    print("=" * 70)
    print(f"{'层数':<10} {'峰值显存':>12} {'参数显存':>12} {'激活显存':>12}")
    print("-" * 70)

    gc_results = []
    for num_layers in layers_to_test:
        peak, params, activation = test_layers(num_layers, use_gc=True)
        if peak != float('inf'):
            print(f"{num_layers:<10} {peak:>10.0f} MB {params:>10.0f} MB {activation:>10.0f} MB")
            gc_results.append((num_layers, activation))
        else:
            print(f"{num_layers:<10} {'OOM':>12}")

    # 分析
    print()
    print("=" * 70)
    print("分析结果")
    print("=" * 70)

    if len(gc_results) >= 2:
        # 计算激活显存增长率
        layers1, act1 = gc_results[0]
        layers2, act2 = gc_results[-1]

        layer_ratio = layers2 / layers1
        act_ratio = act2 / act1

        print(f"\n层数从 {layers1} 增加到 {layers2}（{layer_ratio:.1f}x）")
        print(f"激活显存从 {act1:.0f}MB 增加到 {act2:.0f}MB（{act_ratio:.2f}x）")

        if act_ratio > layer_ratio * 0.8:
            print("\n❌ 诊断：激活显存随层数线性增长")
            print("   结论：GC 可能没有完全生效，或存在显存泄漏")
            print("   可能原因：")
            print("   1. FlexAttention 的 FP32 workaround 占用额外显存")
            print("   2. Summary Token (sum_x) 的计算图没有正确断开")
            print("   3. Block Mask 在每层都缓存了一份")
        elif act_ratio < layer_ratio * 0.3:
            print("\n✅ 诊断：激活显存与层数几乎无关")
            print("   结论：真·全量 GC 生效！")
        else:
            print("\n⚠️  诊断：部分 GC 生效")
            print("   结论：可能是选择性 GC（只 checkpoint 了部分层）")

    # 对比无 GC 的增长率
    if len(no_gc_results) >= 2:
        layers1, act1 = no_gc_results[0]
        layers2, act2 = no_gc_results[-1]
        no_gc_ratio = act2 / act1
        print(f"\n对照组（无 GC）：激活显存增长 {no_gc_ratio:.2f}x")


if __name__ == '__main__':
    main()
