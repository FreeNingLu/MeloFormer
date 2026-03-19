#!/usr/bin/env python3
"""
MeloFormer 1.5B 训练速度基准测试

测试配置:
- 模型: 1.5B (embed=1536, layers=32, heads=24, kv=8, ffn=5632)
- GPU: H800 80GB
- torch.compile + enable_gqa=True + Summary Token 稀疏注意力
- Gradient Checkpointing + BF16

测试不同 seq_len 和 batch_size 组合的:
- 吞吐量 (tokens/sec)
- 显存占用
- 单步耗时
"""

import torch
import torch.nn as nn
import gc
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.attention_flex_summary import set_bar_len, FlexSummaryAttentionMask, SummaryTokenEmbedding
from model.meloformer import create_model


def test_config(model_size, seq_len, num_bars, bar_len, batch_size, use_compile=True, use_gc=True):
    """测试一个配置"""
    device = torch.device('cuda')

    set_bar_len(bar_len=bar_len, max_bars=num_bars)

    # 创建模型
    model = create_model(model_size=model_size, max_seq_len=seq_len, max_bars=num_bars).to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e9

    if use_gc:
        model.gradient_checkpointing_enable()

    model.train()

    if use_compile:
        model = torch.compile(model, mode='default')

    # 创建假数据
    token_ids = torch.randint(0, 643, (batch_size, seq_len), device=device)
    chord_ids = torch.arange(seq_len, device=device).div(bar_len, rounding_mode='floor').clamp(max=num_bars-1)
    chord_ids = chord_ids.unsqueeze(0).expand(batch_size, -1)
    instrument_ids = (torch.arange(seq_len, device=device) % 4).unsqueeze(0).expand(batch_size, -1)
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    note_ids = (torch.arange(seq_len, device=device) % 128).unsqueeze(0).expand(batch_size, -1)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup (includes compilation)
    print(f'    Warming up (compilation)...', end='', flush=True)
    warmup_start = time.time()
    for i in range(3):
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(token_ids, chord_ids, instrument_ids,
                          token_type_ids=token_type_ids, note_ids=note_ids,
                          num_bars=num_bars)
            loss = logits[:, :-1].reshape(-1, logits.size(-1))
            loss = loss.mean()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    warmup_time = time.time() - warmup_start
    print(f' done ({warmup_time:.1f}s)')

    # Timed runs
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    num_steps = 10
    times = []
    for step in range(num_steps):
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t0 = time.time()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(token_ids, chord_ids, instrument_ids,
                          token_type_ids=token_type_ids, note_ids=note_ids,
                          num_bars=num_bars)
            loss = logits[:, :-1].reshape(-1, logits.size(-1))
            loss = loss.mean()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        times.append(time.time() - t0)

    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    avg_time = sum(times) / len(times)
    tokens_per_step = batch_size * seq_len
    tokens_per_sec = tokens_per_step / avg_time

    result = {
        'params_b': params,
        'seq_len': seq_len,
        'batch_size': batch_size,
        'num_bars': num_bars,
        'peak_mem_gb': peak_mem,
        'avg_step_ms': avg_time * 1000,
        'tokens_per_sec': tokens_per_sec,
        'compile_time_s': warmup_time,
    }

    del model, optimizer, token_ids, chord_ids, instrument_ids, logits, loss
    torch.cuda.empty_cache()
    gc.collect()

    return result


def main():
    print("=" * 70)
    print("MeloFormer 1.5B 训练速度基准测试")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    # 先确认参数量
    from model.meloformer import create_model as _cm
    m = _cm(model_size='1.5b')
    params = sum(p.numel() for p in m.parameters()) / 1e9
    print(f"1.5B 模型实际参数量: {params:.3f}B")
    del m

    configs = [
        # (seq_len, num_bars, bar_len, batch_size, use_gc)
        (4096,  32,  128, 1, False),
        (4096,  32,  128, 2, False),
        (4096,  32,  128, 4, False),
        (8192,  64,  128, 1, False),
        (8192,  64,  128, 2, False),
        (16384, 128, 128, 1, False),
    ]

    results = []
    for seq_len, num_bars, bar_len, batch_size, use_gc in configs:
        print(f"\n{'='*70}")
        print(f"Config: seq={seq_len}, bars={num_bars}, batch={batch_size}, GC={'on' if use_gc else 'off'}")
        print(f"{'='*70}")

        try:
            r = test_config('1.5b', seq_len, num_bars, bar_len, batch_size, use_gc=use_gc)
            results.append(r)
            print(f"    峰值显存: {r['peak_mem_gb']:.1f} GB")
            print(f"    单步耗时: {r['avg_step_ms']:.0f} ms")
            print(f"    吞吐量:   {r['tokens_per_sec']:.0f} tokens/sec")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"    ❌ OOM")
                torch.cuda.empty_cache()
                gc.collect()
            else:
                raise

    # 总结
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print(f"\n{'seq_len':>8} {'batch':>6} {'显存':>8} {'单步':>10} {'吞吐量':>15}")
    print("-" * 50)
    for r in results:
        print(f"{r['seq_len']:>8} {r['batch_size']:>6} {r['peak_mem_gb']:>6.1f}GB {r['avg_step_ms']:>8.0f}ms {r['tokens_per_sec']:>12.0f} tok/s")


if __name__ == '__main__':
    main()
