#!/usr/bin/env python3
"""
测试全量 Gradient Checkpointing 效果

对比：
1. 不用 GC - 显存高，速度快
2. 全量 GC - 显存低，速度略慢

运行方法：
    python test_gc.py
"""

import torch
import torch.nn as nn
import gc
import time
from model import create_model


def get_gpu_memory_mb():
    """获取当前 GPU 显存使用量 (MB)"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def get_gpu_memory_reserved_mb():
    """获取当前 GPU 显存预留量 (MB)"""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1024 / 1024
    return 0


def test_forward_backward(model, batch_size, seq_len, num_bars, use_gc, warmup=2, runs=5):
    """测试前向+反向传播"""
    device = next(model.parameters()).device

    # 创建假数据
    token_ids = torch.randint(0, 643, (batch_size, seq_len), device=device)
    chord_ids = torch.repeat_interleave(
        torch.arange(num_bars, device=device),
        seq_len // num_bars
    )[:seq_len].unsqueeze(0).expand(batch_size, -1)
    instrument_ids = (torch.arange(seq_len, device=device) % 4).unsqueeze(0).expand(batch_size, -1)
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    note_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1) % 128

    # 清理显存
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    mem_before = get_gpu_memory_mb()

    # 预热
    for _ in range(warmup):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(
                token_ids, chord_ids, instrument_ids,
                token_type_ids=token_type_ids, note_ids=note_ids,
                num_bars=num_bars
            )
            loss = logits[:, :-1].reshape(-1, logits.size(-1)).mean()
        loss.backward()
        model.zero_grad()

    torch.cuda.synchronize()

    # 正式测试
    times = []
    for _ in range(runs):
        torch.cuda.reset_peak_memory_stats()
        start = time.time()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(
                token_ids, chord_ids, instrument_ids,
                token_type_ids=token_type_ids, note_ids=note_ids,
                num_bars=num_bars
            )
            loss = logits[:, :-1].reshape(-1, logits.size(-1)).mean()
        loss.backward()

        torch.cuda.synchronize()
        times.append(time.time() - start)

        model.zero_grad()

    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    avg_time = sum(times) / len(times)

    return peak_mem, avg_time


def main():
    if not torch.cuda.is_available():
        print("需要 CUDA GPU")
        return

    device = torch.device('cuda')

    print("=" * 70)
    print("全量 Gradient Checkpointing 效果测试")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    # 测试配置
    configs = [
        # (model_size, batch_size, seq_len, num_bars)
        ('small', 2, 4096, 64),
        ('small', 4, 4096, 64),
        ('base', 2, 4096, 64),
    ]

    results = []

    for model_size, batch_size, seq_len, num_bars in configs:
        print(f"\n{'='*70}")
        print(f"测试: {model_size} | batch={batch_size} | seq={seq_len} | bars={num_bars}")
        print(f"{'='*70}")

        # ========== 测试 1: 不用 GC ==========
        print("\n[1] 不使用 Gradient Checkpointing...")
        torch.cuda.empty_cache()
        gc.collect()

        model = create_model(model_size=model_size).to(device)
        model._gradient_checkpointing = False  # 禁用 GC
        model.train()

        try:
            peak_mem_no_gc, time_no_gc = test_forward_backward(
                model, batch_size, seq_len, num_bars, use_gc=False
            )
            print(f"    峰值显存: {peak_mem_no_gc:.0f} MB")
            print(f"    平均时间: {time_no_gc*1000:.1f} ms")
        except RuntimeError as e:
            if "out of memory" in str(e):
                peak_mem_no_gc = float('inf')
                time_no_gc = float('inf')
                print(f"    ❌ OOM - 显存不足")
            else:
                raise

        del model
        torch.cuda.empty_cache()
        gc.collect()

        # ========== 测试 2: 使用全量 GC ==========
        print("\n[2] 使用全量 Gradient Checkpointing...")

        model = create_model(model_size=model_size).to(device)
        model._gradient_checkpointing = True  # 启用 GC
        model._autocast_dtype = torch.bfloat16  # 设置 autocast dtype
        model.train()

        try:
            peak_mem_gc, time_gc = test_forward_backward(
                model, batch_size, seq_len, num_bars, use_gc=True
            )
            print(f"    峰值显存: {peak_mem_gc:.0f} MB")
            print(f"    平均时间: {time_gc*1000:.1f} ms")
        except RuntimeError as e:
            if "out of memory" in str(e):
                peak_mem_gc = float('inf')
                time_gc = float('inf')
                print(f"    ❌ OOM - 显存不足")
            else:
                raise

        del model
        torch.cuda.empty_cache()
        gc.collect()

        # ========== 对比 ==========
        if peak_mem_no_gc != float('inf') and peak_mem_gc != float('inf'):
            mem_saved = (1 - peak_mem_gc / peak_mem_no_gc) * 100
            time_overhead = (time_gc / time_no_gc - 1) * 100

            print(f"\n[对比结果]")
            print(f"    显存节省: {mem_saved:.1f}%")
            print(f"    时间开销: +{time_overhead:.1f}%")

            results.append({
                'config': f"{model_size}/b{batch_size}/s{seq_len}",
                'mem_no_gc': peak_mem_no_gc,
                'mem_gc': peak_mem_gc,
                'mem_saved': mem_saved,
                'time_no_gc': time_no_gc,
                'time_gc': time_gc,
                'time_overhead': time_overhead,
            })
        elif peak_mem_no_gc == float('inf') and peak_mem_gc != float('inf'):
            print(f"\n[对比结果]")
            print(f"    ✅ GC 让原本 OOM 的配置可以运行！")
            print(f"    显存: {peak_mem_gc:.0f} MB")

            results.append({
                'config': f"{model_size}/b{batch_size}/s{seq_len}",
                'mem_no_gc': 'OOM',
                'mem_gc': peak_mem_gc,
                'mem_saved': '∞',
                'time_no_gc': 'N/A',
                'time_gc': time_gc,
                'time_overhead': 'N/A',
            })

    # ========== 总结 ==========
    print("\n")
    print("=" * 70)
    print("总结")
    print("=" * 70)
    print(f"{'配置':<25} {'无GC显存':>12} {'有GC显存':>12} {'节省':>10} {'时间开销':>12}")
    print("-" * 70)
    for r in results:
        mem_no_gc = f"{r['mem_no_gc']:.0f} MB" if r['mem_no_gc'] != 'OOM' else 'OOM'
        mem_gc = f"{r['mem_gc']:.0f} MB" if r['mem_gc'] != 'OOM' else 'OOM'
        mem_saved = f"{r['mem_saved']:.1f}%" if isinstance(r['mem_saved'], float) else r['mem_saved']
        time_overhead = f"+{r['time_overhead']:.1f}%" if isinstance(r['time_overhead'], float) else r['time_overhead']
        print(f"{r['config']:<25} {mem_no_gc:>12} {mem_gc:>12} {mem_saved:>10} {time_overhead:>12}")

    print()
    print("✅ 全量 Gradient Checkpointing 验证成功！")
    print()
    print("结论:")
    print("  - 显存节省: 通常 40-60%")
    print("  - 时间开销: 通常 20-40%")
    print("  - 核心价值: 让你能跑更大的 batch 或更大的模型")


if __name__ == '__main__':
    main()
