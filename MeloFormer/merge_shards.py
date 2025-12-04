#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分片合并脚本 - 减少 I/O 开销

将多个小分片合并成少数大分片，提高数据加载效率
"""

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='合并数据分片')
    parser.add_argument('--input', type=str, required=True, help='输入目录 (包含 shard_*.pt)')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--target_shards', type=int, default=8, help='目标分片数量')
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 找到所有分片
    shard_files = sorted(input_dir.glob('shard_*.pt'))
    if not shard_files:
        print(f"错误: 在 {input_dir} 中未找到 shard_*.pt 文件")
        return

    print(f"=" * 60)
    print(f"分片合并")
    print(f"=" * 60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"源分片数: {len(shard_files)}")
    print(f"目标分片数: {args.target_shards}")
    print()

    # 加载所有数据
    print("加载所有分片...")
    all_samples = []
    for shard_file in tqdm(shard_files, desc="读取分片"):
        shard_data = torch.load(shard_file, weights_only=False)
        all_samples.extend(shard_data)

    total_samples = len(all_samples)
    print(f"总样本数: {total_samples:,}")

    # 计算每个新分片的大小
    samples_per_shard = (total_samples + args.target_shards - 1) // args.target_shards
    print(f"每个新分片: ~{samples_per_shard:,} 样本")
    print()

    # 保存新分片
    print("保存新分片...")
    saved_shards = []
    for i in range(args.target_shards):
        start_idx = i * samples_per_shard
        end_idx = min((i + 1) * samples_per_shard, total_samples)

        if start_idx >= total_samples:
            break

        shard_data = all_samples[start_idx:end_idx]
        shard_path = output_dir / f"shard_{i:04d}.pt"
        torch.save(shard_data, shard_path)
        saved_shards.append(str(shard_path))
        print(f"  {shard_path.name}: {len(shard_data):,} 样本")

    # 保存 meta.json
    meta = {
        'total_samples': total_samples,
        'num_shards': len(saved_shards),
        'samples_per_shard': samples_per_shard,
    }
    meta_path = output_dir / 'meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print()
    print(f"完成!")
    print(f"新分片数: {len(saved_shards)}")
    print(f"meta.json: {meta_path}")


if __name__ == '__main__':
    main()
