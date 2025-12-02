#!/usr/bin/env python3
"""
MeloFormer 显存测试和动态配置生成工具
测试不同阶段的显存需求，自动生成最优配置
"""

import os
import sys
import time
import json
import torch
import psutil
from pathlib import Path

# 增加 dynamo cache size 避免 FlexAttention 重编译问题
import torch._dynamo
torch._dynamo.config.cache_size_limit = 2048
torch._dynamo.config.accumulated_cache_size_limit = 4096

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / 'hid_museformer_v1.0'))

from model.hid_museformer import HIDMuseFormer
from data.tokenizer_v2 import HIDTokenizerV2


def get_memory_usage():
    """获取当前内存和显存使用情况"""
    # CPU 内存
    process = psutil.Process()
    ram_gb = process.memory_info().rss / 1024**3

    # GPU 显存
    if torch.cuda.is_available():
        gpu_gb = torch.cuda.memory_allocated() / 1024**3
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3
    else:
        gpu_gb = 0
        gpu_reserved = 0

    return {
        'ram_gb': round(ram_gb, 2),
        'gpu_allocated_gb': round(gpu_gb, 2),
        'gpu_reserved_gb': round(gpu_reserved, 2),
    }


def test_compilation_phase(model, batch_size, seq_len):
    """测试编译阶段的显存需求"""
    print(f"\n{'='*60}")
    print(f"测试编译阶段: batch_size={batch_size}, seq_len={seq_len}")
    print(f"{'='*60}")

    model.train()
    torch.cuda.empty_cache()

    # 准备假数据
    device = next(model.parameters()).device
    token_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    chord_ids = torch.randint(0, 100, (batch_size, seq_len // 8), device=device)
    instrument_ids = torch.randint(0, 10, (batch_size, seq_len), device=device)
    token_type_ids = torch.randint(0, 5, (batch_size, seq_len), device=device)
    note_ids = torch.randint(0, 128, (batch_size, seq_len), device=device)

    mem_before = get_memory_usage()
    print(f"前向传播前: {mem_before}")

    # 前向传播 (触发编译)
    start_time = time.time()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits = model(
            token_ids,
            chord_ids=chord_ids,
            instrument_ids=instrument_ids,
            token_type_ids=token_type_ids,
            note_ids=note_ids,
        )

    mem_after_forward = get_memory_usage()
    compile_time = time.time() - start_time

    print(f"前向传播后: {mem_after_forward}")
    print(f"编译时间: {compile_time:.2f}s")

    # 反向传播
    labels = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
    )
    loss.backward()

    mem_after_backward = get_memory_usage()
    print(f"反向传播后: {mem_after_backward}")

    # 清理
    del logits, loss, token_ids, labels
    torch.cuda.empty_cache()

    return {
        'batch_size': batch_size,
        'seq_len': seq_len,
        'before': mem_before,
        'after_forward': mem_after_forward,
        'after_backward': mem_after_backward,
        'compile_time': round(compile_time, 2),
    }


def test_training_phase(model, batch_size, seq_len):
    """测试训练阶段的显存需求（编译后）"""
    print(f"\n{'='*60}")
    print(f"测试训练阶段: batch_size={batch_size}, seq_len={seq_len}")
    print(f"{'='*60}")

    model.train()
    torch.cuda.empty_cache()

    device = next(model.parameters()).device

    # 多次迭代测试稳定性
    results = []
    for i in range(3):
        token_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        chord_ids = torch.randint(0, 100, (batch_size, seq_len // 8), device=device)
        instrument_ids = torch.randint(0, 10, (batch_size, seq_len), device=device)
        token_type_ids = torch.randint(0, 5, (batch_size, seq_len), device=device)
        note_ids = torch.randint(0, 128, (batch_size, seq_len), device=device)
        labels = torch.randint(0, 1000, (batch_size, seq_len), device=device)

        mem_before = get_memory_usage()

        # 前向 + 反向
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(
                token_ids,
                chord_ids=chord_ids,
                instrument_ids=instrument_ids,
                token_type_ids=token_type_ids,
                note_ids=note_ids,
            )

        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
        )
        loss.backward()

        mem_after = get_memory_usage()

        print(f"迭代 {i+1}: {mem_after}")

        results.append(mem_after)

        # 清理
        del logits, loss, token_ids, labels
        torch.cuda.empty_cache()

    return results


def generate_optimal_config(test_results, total_ram_gb, total_gpu_gb):
    """根据测试结果生成最优配置"""
    print(f"\n{'='*60}")
    print("生成最优动态配置")
    print(f"{'='*60}")

    # 分析编译阶段
    compile_results = [r for r in test_results if 'compile_time' in r]

    # 找到在 GPU 限制内的最大 batch_size（编译阶段）
    compile_safe_configs = []
    for r in compile_results:
        gpu_peak = max(
            r['after_forward']['gpu_reserved_gb'],
            r['after_backward']['gpu_reserved_gb']
        )
        if gpu_peak < total_gpu_gb * 0.8:  # 留 20% 余量
            compile_safe_configs.append({
                'batch_size': r['batch_size'],
                'gpu_gb': gpu_peak,
            })

    if not compile_safe_configs:
        print("❌ 错误: 所有配置都超出 GPU 限制")
        return None

    # 编译阶段最优配置（最小 batch）
    compile_batch = min(c['batch_size'] for c in compile_safe_configs)

    # 训练阶段最优配置（最大 batch）
    train_batch = max(c['batch_size'] for c in compile_safe_configs)

    config = {
        'hardware': {
            'total_ram_gb': total_ram_gb,
            'total_gpu_gb': total_gpu_gb,
        },
        'phase_1_compilation': {
            'step_range': '1-10',
            'batch_size': compile_batch,
            'num_workers': 0,
            'seq_len': 8192,
            'expected_gpu_gb': compile_safe_configs[0]['gpu_gb'],
        },
        'phase_2_transition': {
            'step_range': '11-50',
            'batch_size': compile_batch,
            'num_workers': 0,
            'seq_len': 8192,
        },
        'phase_3_training': {
            'step_range': '51+',
            'batch_size': train_batch,
            'num_workers': 8,
            'seq_len': 8192,
            'expected_gpu_gb': max(c['gpu_gb'] for c in compile_safe_configs),
        },
    }

    print("\n✅ 最优配置:")
    print(json.dumps(config, indent=2, ensure_ascii=False))

    return config


def main():
    print("="*60)
    print("MeloFormer 显存测试工具")
    print("="*60)

    # 检查 GPU
    if not torch.cuda.is_available():
        print("❌ 错误: 未检测到 GPU")
        return

    device = torch.device('cuda:0')
    total_gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    total_ram_gb = psutil.virtual_memory().total / 1024**3

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU 显存: {total_gpu_gb:.2f} GB")
    print(f"CPU 内存: {total_ram_gb:.2f} GB")

    # 创建模型
    print("\n创建模型...")
    vocab_size = 2000
    model = HIDMuseFormer(
        vocab_size=vocab_size,
        embed_dim=256,
        num_layers=6,
        num_heads=4,
        ffn_dim=1408,
        max_seq_len=8192,
        max_bars=2048,
        use_gradient_checkpointing=True,
    ).to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 创建优化器（模拟真实训练）
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # 测试配置
    test_configs = [
        {'batch_size': 1, 'seq_len': 8192},
        {'batch_size': 2, 'seq_len': 8192},
        {'batch_size': 4, 'seq_len': 8192},
        {'batch_size': 6, 'seq_len': 8192},
        {'batch_size': 8, 'seq_len': 8192},
    ]

    test_results = []

    # 测试编译阶段
    for config in test_configs:
        try:
            result = test_compilation_phase(
                model,
                config['batch_size'],
                config['seq_len']
            )
            test_results.append(result)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"❌ OOM: batch_size={config['batch_size']}")
                torch.cuda.empty_cache()
                break
            else:
                raise

    # 生成配置
    if test_results:
        optimal_config = generate_optimal_config(
            test_results,
            total_ram_gb,
            total_gpu_gb
        )

        # 保存配置
        if optimal_config:
            output_file = Path(__file__).parent / 'optimal_config.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(optimal_config, f, indent=2, ensure_ascii=False)
            print(f"\n✅ 配置已保存到: {output_file}")

    print("\n" + "="*60)
    print("测试完成")
    print("="*60)


if __name__ == '__main__':
    main()
