#!/usr/bin/env python3
"""
测试数据预处理 -> MIDI 解码

从 processed_data 中取一个样本，解码成 MIDI 文件
验证数据预处理是否正确
"""

import torch
import argparse
from pathlib import Path
from data import HIDTokenizerV2
from data.txt_to_midi import txt_to_midi
import tempfile


def main():
    parser = argparse.ArgumentParser(description='测试数据解码为 MIDI')
    parser.add_argument('--data_dir', type=str, required=True, help='预处理数据目录')
    parser.add_argument('--sample_idx', type=int, default=0, help='样本索引')
    parser.add_argument('--shard_idx', type=int, default=0, help='分片索引')
    parser.add_argument('--output', type=str, default='test_output.mid', help='输出 MIDI 文件')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # 加载分片
    shard_files = sorted(data_dir.glob('shard_*.pt'))
    if not shard_files:
        print(f"错误: 在 {data_dir} 中未找到 shard_*.pt 文件")
        return

    print(f"找到 {len(shard_files)} 个分片")

    if args.shard_idx >= len(shard_files):
        print(f"错误: 分片索引 {args.shard_idx} 超出范围 (0-{len(shard_files)-1})")
        return

    shard_file = shard_files[args.shard_idx]
    print(f"加载分片: {shard_file}")

    shard_data = torch.load(shard_file, weights_only=False)
    print(f"分片包含 {len(shard_data)} 个样本")

    if args.sample_idx >= len(shard_data):
        print(f"错误: 样本索引 {args.sample_idx} 超出范围 (0-{len(shard_data)-1})")
        return

    sample = shard_data[args.sample_idx]

    # 打印样本信息
    print(f"\n样本信息:")
    print(f"  - token_ids: {sample['token_ids'].shape}")
    print(f"  - chord_ids: {sample['chord_ids'].shape}")
    print(f"  - instrument_ids: {sample['instrument_ids'].shape}")
    print(f"  - length: {sample['length']}")

    if 'token_type_ids' in sample:
        print(f"  - token_type_ids: {sample['token_type_ids'].shape}")
    if 'note_ids' in sample:
        print(f"  - note_ids: {sample['note_ids'].shape}")

    # 创建 tokenizer
    tokenizer = HIDTokenizerV2()

    # 解码 token_ids
    token_ids = sample['token_ids'].tolist()
    print(f"\n前 50 个 token IDs: {token_ids[:50]}")

    # 解码为文本
    print(f"\n解码 tokens...")
    tokens = tokenizer.decode(token_ids)
    print(f"解码得到 {len(tokens)} 个 tokens")
    print(f"前 30 个 tokens: {tokens[:30]}")

    # 转换为 MIDI
    print(f"\n转换为 MIDI: {args.output}")
    try:
        # tokens 是字符串，需要先写入临时 txt 文件
        # decode 返回的是原始文本格式（带换行符）
        txt_content = tokens if isinstance(tokens, str) else '\n'.join(tokens)

        # 写入临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(txt_content)
            tmp_txt = f.name

        print(f"临时 TXT 文件: {tmp_txt}")

        # 转换为 MIDI
        txt_to_midi(tmp_txt, args.output)
        print(f"✅ MIDI 文件已保存: {args.output}")

        # 清理临时文件
        Path(tmp_txt).unlink()

    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()

    # 统计 token 分布
    print(f"\n=== Token 统计 ===")
    token_counts = {}
    for token in tokens:
        prefix = token.split('_')[0] if '_' in token else token[:2]
        token_counts[prefix] = token_counts.get(prefix, 0) + 1

    for prefix, count in sorted(token_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {prefix}: {count}")


if __name__ == '__main__':
    main()
