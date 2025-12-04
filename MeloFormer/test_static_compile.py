#!/usr/bin/env python3
"""
测试静态编译效果

对比：
1. 动态模式 - 每个 batch 不同 seq_len 导致 FlexAttention 重编译
2. 静态模式 - 固定 seq_len，只编译一次

运行方法：
    python test_static_compile.py

预期效果:
- 动态模式: 每个 batch 都可能触发重编译，速度波动大
- 静态模式: 第 1 个 batch 编译，后续 batch 快 10-15x
"""

import torch
import torch.nn as nn
import gc
import time
from model import create_model
from model.attention_flex_summary import (
    set_static_compile_mode,
    is_static_compile_mode,
    MAX_SEQ_LEN,
    MAX_SUM_LEN,
)


def get_gpu_memory_mb():
    """获取当前 GPU 显存使用量 (MB)"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def create_batch(batch_size, seq_len, num_bars, device):
    """创建测试 batch"""
    token_ids = torch.randint(0, 643, (batch_size, seq_len), device=device)
    chord_ids = torch.repeat_interleave(
        torch.arange(num_bars, device=device),
        seq_len // num_bars
    )[:seq_len].unsqueeze(0).expand(batch_size, -1)
    instrument_ids = (torch.arange(seq_len, device=device) % 4).unsqueeze(0).expand(batch_size, -1)
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    note_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1) % 128

    return {
        'token_ids': token_ids,
        'chord_ids': chord_ids,
        'instrument_ids': instrument_ids,
        'token_type_ids': token_type_ids,
        'note_ids': note_ids,
        'num_bars': num_bars,
    }


def test_forward_backward(model, batch, warmup=0, runs=1):
    """测试前向+反向传播，返回时间列表"""
    device = next(model.parameters()).device

    # 预热
    for _ in range(warmup):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(
                batch['token_ids'],
                chord_ids=batch['chord_ids'],
                instrument_ids=batch['instrument_ids'],
                token_type_ids=batch['token_type_ids'],
                note_ids=batch['note_ids'],
                num_bars=batch['num_bars'],
            )
            loss = logits.mean()
        loss.backward()
        model.zero_grad()

    torch.cuda.synchronize()

    # 正式测试
    times = []
    for _ in range(runs):
        start = time.time()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(
                batch['token_ids'],
                chord_ids=batch['chord_ids'],
                instrument_ids=batch['instrument_ids'],
                token_type_ids=batch['token_type_ids'],
                note_ids=batch['note_ids'],
                num_bars=batch['num_bars'],
            )
            loss = logits.mean()
        loss.backward()

        torch.cuda.synchronize()
        times.append(time.time() - start)

        model.zero_grad()

    return times


def test_dynamic_mode():
    """测试动态模式 (不同 seq_len)"""
    print("\n" + "=" * 70)
    print("测试 1: 动态模式 (每个 batch 不同 seq_len)")
    print("=" * 70)

    # 确保关闭静态模式
    set_static_compile_mode(False)
    assert not is_static_compile_mode()

    device = torch.device('cuda')
    model = create_model(model_size='small').to(device)
    model._gradient_checkpointing = True
    model._autocast_dtype = torch.bfloat16
    model.train()

    # 模拟不同长度的 batch
    seq_lens = [1024, 2048, 3072, 4096, 2048, 1024, 4096, 3072]
    batch_size = 1

    print(f"\n序列长度序列: {seq_lens}")
    print(f"每个不同的 seq_len 可能触发 FlexAttention 重编译\n")

    times = []
    for i, seq_len in enumerate(seq_lens):
        num_bars = max(seq_len // 64, 1)

        # 清理显存
        torch.cuda.empty_cache()
        gc.collect()

        batch = create_batch(batch_size, seq_len, num_bars, device)

        t = test_forward_backward(model, batch, warmup=0, runs=1)[0]
        times.append(t)

        print(f"  Batch {i+1}: seq_len={seq_len:5d}, num_bars={num_bars:3d}, time={t*1000:8.1f}ms")

    avg_time = sum(times) / len(times)
    print(f"\n平均时间: {avg_time*1000:.1f}ms")
    print(f"时间波动: {min(times)*1000:.1f}ms - {max(times)*1000:.1f}ms")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return times


def test_static_mode():
    """测试静态模式 (固定 seq_len)"""
    print("\n" + "=" * 70)
    print("测试 2: 静态模式 (固定 seq_len)")
    print("=" * 70)

    # 启用静态模式
    static_seq_len = 16384
    static_num_bars = 256
    set_static_compile_mode(True, max_seq_len=static_seq_len, max_sum_len=static_num_bars)
    assert is_static_compile_mode()

    device = torch.device('cuda')
    model = create_model(model_size='small').to(device)
    model._gradient_checkpointing = True
    model._autocast_dtype = torch.bfloat16
    model.train()

    print(f"\n静态配置: MAX_SEQ_LEN={static_seq_len}, MAX_SUM_LEN={static_num_bars}")
    print(f"所有 batch 都 Padding 到固定长度，只编译一次\n")

    batch_size = 1
    num_batches = 8

    # 清理显存
    torch.cuda.empty_cache()
    gc.collect()

    # 创建固定长度的 batch
    batch = create_batch(batch_size, static_seq_len, static_num_bars, device)

    times = []
    for i in range(num_batches):
        t = test_forward_backward(model, batch, warmup=0, runs=1)[0]
        times.append(t)

        print(f"  Batch {i+1}: seq_len={static_seq_len:5d}, num_bars={static_num_bars:3d}, time={t*1000:8.1f}ms")

    avg_time = sum(times) / len(times)
    avg_after_compile = sum(times[1:]) / len(times[1:]) if len(times) > 1 else times[0]

    print(f"\n平均时间 (全部): {avg_time*1000:.1f}ms")
    print(f"平均时间 (第2个起): {avg_after_compile*1000:.1f}ms")
    print(f"编译开销: {times[0]*1000:.1f}ms (仅第1个 batch)")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    # 关闭静态模式
    set_static_compile_mode(False)

    return times


def main():
    if not torch.cuda.is_available():
        print("需要 CUDA GPU")
        return

    print("=" * 70)
    print("FlexAttention 静态编译效果测试")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 测试动态模式
    dynamic_times = test_dynamic_mode()

    # 测试静态模式
    static_times = test_static_mode()

    # 对比结果
    print("\n" + "=" * 70)
    print("对比结果")
    print("=" * 70)

    dynamic_avg = sum(dynamic_times) / len(dynamic_times)
    static_avg = sum(static_times) / len(static_times)
    static_avg_after = sum(static_times[1:]) / len(static_times[1:]) if len(static_times) > 1 else static_times[0]

    print(f"\n动态模式:")
    print(f"  平均时间: {dynamic_avg*1000:.1f}ms")
    print(f"  时间范围: {min(dynamic_times)*1000:.1f}ms - {max(dynamic_times)*1000:.1f}ms")

    print(f"\n静态模式:")
    print(f"  平均时间 (全部): {static_avg*1000:.1f}ms")
    print(f"  平均时间 (第2个起): {static_avg_after*1000:.1f}ms")
    print(f"  编译开销: {static_times[0]*1000:.1f}ms")

    if static_avg_after < dynamic_avg:
        speedup = dynamic_avg / static_avg_after
        print(f"\n{'='*70}")
        print(f"静态编译提速: {speedup:.1f}x (相对于动态模式)")
        print(f"{'='*70}")
    else:
        print(f"\n注意: 静态模式未表现出预期的速度优势")
        print(f"可能原因:")
        print(f"  1. 测试 batch 数量太少，编译开销占主导")
        print(f"  2. 动态模式缓存命中率较高")
        print(f"  3. 需要更多 batch 才能摊薄编译成本")

    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print("""
静态编译模式的核心价值:
1. 消除重编译: 只在第1个 batch 编译，后续复用 Kernel
2. 稳定性能: 每个 batch 的耗时稳定一致
3. 大规模训练: 训练数十万 batch 时，编译开销可忽略不计

使用方法:
    python train.py --static_compile --static_max_seq_len 8192 --static_max_bars 256 ...
""")


if __name__ == '__main__':
    main()
