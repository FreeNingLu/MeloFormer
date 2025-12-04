#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 .pt 分片转换为 HuggingFace Arrow 格式

Arrow 格式优势:
1. 零拷贝内存映射 - 不需要反序列化
2. O(1) 随机访问 - 直接定位到任意样本
3. 多进程安全 - num_workers 可以开很大
4. 自动分桶 - 使用 sort + batch 实现长度分组

使用方法:
    python convert_to_arrow.py --input ~/data/processed_data --output ~/data/arrow_data
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset, Features, Sequence, Value
import gc


def get_seq_len(sample):
    """获取样本的序列长度"""
    if 'length' in sample:
        length = sample['length']
        # 确保是 int
        if isinstance(length, torch.Tensor):
            return int(length.item())
        return int(length)
    if 'token_ids' in sample:
        t = sample['token_ids']
        return t.shape[0] if hasattr(t, 'shape') else len(t)
    return 0


def tensor_to_list(t):
    """将 tensor 转为 list of int"""
    if t is None:
        return []
    if isinstance(t, torch.Tensor):
        return t.tolist()
    elif isinstance(t, np.ndarray):
        return t.tolist()
    elif isinstance(t, list):
        return t
    return []


def safe_get_list(sample, key):
    """安全获取列表字段，确保返回 list of int"""
    val = sample.get(key)
    if val is None:
        return []
    result = tensor_to_list(val)
    # 确保是 list
    if not isinstance(result, list):
        return []
    return result


def main():
    parser = argparse.ArgumentParser(description='转换为 HuggingFace Arrow 格式')
    parser.add_argument('--input', type=str, required=True, help='输入目录 (包含 shard_*.pt)')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--sort_by_length', action='store_true', default=True, help='按长度排序 (减少 padding)')
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_files = sorted(input_dir.glob('shard_*.pt'))
    if not shard_files:
        print(f"错误: 在 {input_dir} 中未找到 shard_*.pt 文件")
        return

    print(f"=" * 60)
    print(f"转换为 HuggingFace Arrow 格式")
    print(f"=" * 60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"源分片数: {len(shard_files)}")
    print(f"按长度排序: {args.sort_by_length}")
    print()

    # 显式定义 schema，避免类型推断问题
    features = Features({
        'token_ids': Sequence(Value('int64')),
        'chord_ids': Sequence(Value('int64')),
        'instrument_ids': Sequence(Value('int64')),
        'token_type_ids': Sequence(Value('int64')),
        'note_ids': Sequence(Value('int64')),
        'length': Value('int64'),
    })

    # 使用生成器逐个加载，避免内存爆炸
    error_count = 0
    processed_count = 0

    def generate_samples():
        nonlocal error_count, processed_count
        for shard_idx, shard_file in enumerate(tqdm(shard_files, desc="读取分片")):
            try:
                shard_data = torch.load(shard_file, weights_only=False)
            except Exception as e:
                print(f"\n警告: 加载 {shard_file} 失败: {e}")
                error_count += 1
                continue

            for sample_idx, sample in enumerate(shard_data):
                try:
                    processed_count += 1
                    yield {
                        'token_ids': safe_get_list(sample, 'token_ids'),
                        'chord_ids': safe_get_list(sample, 'chord_ids'),
                        'instrument_ids': safe_get_list(sample, 'instrument_ids'),
                        'token_type_ids': safe_get_list(sample, 'token_type_ids'),
                        'note_ids': safe_get_list(sample, 'note_ids'),
                        'length': get_seq_len(sample),
                    }
                except Exception as e:
                    print(f"\n警告: shard {shard_idx} sample {sample_idx} 处理失败: {e}")
                    error_count += 1
                    continue

            del shard_data
            gc.collect()

    print("创建 Arrow Dataset...")
    dataset = Dataset.from_generator(generate_samples, features=features)

    if error_count > 0:
        print(f"\n警告: 共有 {error_count} 个错误，已跳过")

    print(f"总样本数: {len(dataset):,}")

    # 按长度排序（可选，但强烈推荐）
    if args.sort_by_length:
        print("按序列长度排序...")
        dataset = dataset.sort('length')

    # 保存为 Arrow 格式
    print(f"保存到 {output_dir}...")
    dataset.save_to_disk(str(output_dir))

    # 打印统计
    lengths = dataset['length']
    print()
    print(f"=" * 60)
    print(f"完成!")
    print(f"=" * 60)
    print(f"样本数: {len(dataset):,}")
    print(f"长度范围: {min(lengths)} - {max(lengths)}")
    print(f"平均长度: {sum(lengths) / len(lengths):.0f}")
    print()
    print("使用方法:")
    print("  from datasets import load_from_disk")
    print(f"  dataset = load_from_disk('{output_dir}')")
    print()
    print("训练时直接用:")
    print(f"  python train.py --data_dir {output_dir} --use_arrow ...")


if __name__ == '__main__':
    main()
