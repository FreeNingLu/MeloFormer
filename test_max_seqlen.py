#!/usr/bin/env python3
"""
测试不同模型规格在 batch_size=1 时的最大序列长度
用于确定各模型的极限配置
"""

import os
import sys
import torch
import psutil
from pathlib import Path

# 增加 dynamo cache size
import torch._dynamo
torch._dynamo.config.cache_size_limit = 2048
torch._dynamo.config.accumulated_cache_size_limit = 4096

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / 'hid_museformer_v0.9'))

from model.hid_museformer import HIDMuseFormer


def get_gpu_memory():
    """获取 GPU 显存使用"""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1024**3
    return 0


def test_max_seqlen(model_config, start_len=8192, max_len=32768, step=2048):
    """
    测试给定模型配置下的最大序列长度

    Args:
        model_config: 模型配置字典
        start_len: 起始序列长度
        max_len: 最大测试序列长度
        step: 每次增加的长度
    """
    print(f"\n{'='*60}")
    print(f"测试模型: {model_config['name']}")
    print(f"参数量: {model_config['params']}")
    print(f"配置: embed_dim={model_config['embed_dim']}, "
          f"num_layers={model_config['num_layers']}, "
          f"num_heads={model_config['num_heads']}")
    print(f"{'='*60}\n")

    device = torch.device('cuda:0')
    vocab_size = 2000

    # 创建模型
    model = HIDMuseFormer(
        vocab_size=vocab_size,
        embed_dim=model_config['embed_dim'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        ffn_dim=model_config['ffn_dim'],
        max_seq_len=max_len,
        max_bars=max_len // 4,
        use_gradient_checkpointing=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    print(f"模型创建完成，实际参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M\n")

    results = []
    last_success_len = 0

    # 二分搜索最大序列长度
    seq_len = start_len

    while seq_len <= max_len:
        try:
            torch.cuda.empty_cache()
            model.train()

            # 准备数据 (batch_size=1)
            token_ids = torch.randint(0, 1000, (1, seq_len), device=device)
            chord_ids = torch.randint(0, 100, (1, seq_len // 8), device=device)
            instrument_ids = torch.randint(0, 10, (1, seq_len), device=device)
            token_type_ids = torch.randint(0, 5, (1, seq_len), device=device)
            note_ids = torch.randint(0, 128, (1, seq_len), device=device)
            labels = torch.randint(0, 1000, (1, seq_len), device=device)

            mem_before = get_gpu_memory()

            # 前向传播
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(
                    token_ids,
                    chord_ids=chord_ids,
                    instrument_ids=instrument_ids,
                    token_type_ids=token_type_ids,
                    note_ids=note_ids,
                )

            mem_after_forward = get_gpu_memory()

            # 反向传播
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
            )
            loss.backward()

            mem_after_backward = get_gpu_memory()

            optimizer.zero_grad()

            result = {
                'seq_len': seq_len,
                'forward_gpu_gb': round(mem_after_forward, 2),
                'backward_gpu_gb': round(mem_after_backward, 2),
                'success': True,
            }
            results.append(result)
            last_success_len = seq_len

            print(f"✅ seq_len={seq_len:5d} | "
                  f"前向: {mem_after_forward:5.2f}GB | "
                  f"反向: {mem_after_backward:5.2f}GB")

            # 清理
            del logits, loss, token_ids, labels
            torch.cuda.empty_cache()

            # 继续测试更长序列
            seq_len += step

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"❌ seq_len={seq_len:5d} | OOM")
                torch.cuda.empty_cache()
                break
            else:
                raise

    print(f"\n{'='*60}")
    print(f"✅ 最大序列长度: {last_success_len}")
    print(f"{'='*60}\n")

    return {
        'model': model_config['name'],
        'max_seq_len': last_success_len,
        'results': results,
    }


def main():
    print("="*60)
    print("MeloFormer 最大序列长度测试")
    print("="*60)

    # 检查 GPU
    if not torch.cuda.is_available():
        print("❌ 错误: 未检测到 GPU")
        return

    device = torch.device('cuda:0')
    total_gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU 显存: {total_gpu_gb:.2f} GB\n")

    # 模型配置
    model_configs = [
        {
            'name': 'small',
            'params': '17M',
            'embed_dim': 256,
            'num_layers': 6,
            'num_heads': 4,
            'ffn_dim': 1408,
        },
        {
            'name': 'base',
            'params': '85M',
            'embed_dim': 512,
            'num_layers': 12,
            'num_heads': 8,
            'ffn_dim': 2816,
        },
        {
            'name': 'large',
            'params': '200M',
            'embed_dim': 768,
            'num_layers': 16,
            'num_heads': 12,
            'ffn_dim': 4224,
        },
        {
            'name': 'xlarge',
            'params': '450M',
            'embed_dim': 1024,
            'num_layers': 24,
            'num_heads': 16,
            'ffn_dim': 5632,
        },
    ]

    all_results = []

    for config in model_configs:
        result = test_max_seqlen(
            config,
            start_len=8192,
            max_len=32768,
            step=2048
        )
        all_results.append(result)

    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print(f"\n{'模型':<10} {'参数量':<10} {'最大序列长度':<15} {'推荐配置':<20}")
    print("-"*60)

    for result in all_results:
        config = next(c for c in model_configs if c['name'] == result['model'])
        max_len = result['max_seq_len']

        # 推荐配置：留 20% 余量
        recommended_len = int(max_len * 0.8)

        print(f"{result['model']:<10} {config['params']:<10} "
              f"{max_len:<15} {recommended_len} (安全值)")

    print("\n" + "="*60)
    print("说明")
    print("="*60)
    print("- 以上测试基于 batch_size=1 + gradient_checkpointing")
    print("- 推荐配置留 20% 显存余量，确保稳定性")
    print("- 如需更大 batch_size，需相应减少 seq_len")
    print("- 例如: batch_size=2 时，max_seq_len 约减半")
    print("="*60)


if __name__ == '__main__':
    main()
