#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIDI 数据预处理脚本 - 分片格式

将 MIDI 文件批量转换为分片 .pt 格式，减少训练时 I/O 开销

输出格式:
    processed_data/
    ├── shard_0000.pt   # 每个分片包含 ~10000 个样本
    ├── shard_0001.pt
    ├── ...
    └── meta.json

用法:
    python preprocess_data.py --input midi_files.txt --output processed_data/ --workers 32

H800 推荐:
    python preprocess_data.py --input midi_files.txt --output processed_data/ --workers 64 --shard-size 10000
"""

import os
import sys
import argparse
import torch
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import io
from contextlib import redirect_stdout
import tempfile
import queue
import threading

# 添加父目录到 path
sys.path.insert(0, str(Path(__file__).parent))

from data.tokenizer_v2 import HIDTokenizerV2, TokenInfo
from data.midi_to_txt import midi_to_txt


def process_single_midi(args):
    """处理单个 MIDI 文件 (用于多进程)，返回数据而不保存"""
    midi_path, max_seq_len = args

    try:
        # 创建 tokenizer (每个进程独立)
        tokenizer = HIDTokenizerV2()

        # MIDI -> TXT
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=True, mode='w') as tmp:
            with redirect_stdout(io.StringIO()):
                midi_to_txt(midi_path, tmp.name, quantize=True, detect_chords=True)

            # 读取转换后的文本
            txt_content = open(tmp.name, 'r').read()

        # Tokenize
        token_ids, token_infos = tokenizer.encode_with_info(txt_content, add_special=True)

        # 检查是否有效
        if len(token_ids) <= 2:  # 只有 BOS + EOS
            return None, midi_path, "Empty sequence"

        # 截断
        if len(token_ids) > max_seq_len:
            token_ids = token_ids[:max_seq_len-1] + [tokenizer.eos_id]
            token_infos = token_infos[:max_seq_len-1] + [token_infos[-1]]

        # 构建数据
        data = {
            'token_ids': torch.tensor(token_ids, dtype=torch.long),
            'chord_ids': torch.tensor([info.chord_idx for info in token_infos], dtype=torch.long),
            'position_ids': torch.tensor([info.position for info in token_infos], dtype=torch.long),
            'instrument_ids': torch.tensor([
                info.instrument_id if info.instrument_id >= 0 else 129
                for info in token_infos
            ], dtype=torch.long),
            'token_type_ids': torch.tensor([info.token_type for info in token_infos], dtype=torch.long),
            'note_ids': torch.tensor([info.note_id for info in token_infos], dtype=torch.long),
            'length': len(token_ids),
        }

        return data, midi_path, None

    except Exception as e:
        return None, midi_path, str(e)


def save_shard(samples, shard_idx, output_dir):
    """保存一个分片"""
    shard_path = output_dir / f"shard_{shard_idx:04d}.pt"
    torch.save(samples, shard_path)
    return shard_path


def main():
    parser = argparse.ArgumentParser(description='MIDI 数据预处理 (分片格式)')
    parser.add_argument('--input', type=str, required=True, help='输入 MIDI 目录或文件列表')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--workers', type=int, default=32, help='并行工作进程数')
    parser.add_argument('--max-seq-len', type=int, default=24576, help='最大序列长度')
    parser.add_argument('--shard-size', type=int, default=10000, help='每个分片的样本数')
    parser.add_argument('--limit', type=int, default=None, help='限制处理文件数量 (测试用)')

    args = parser.parse_args()

    # 收集 MIDI 文件
    input_path = Path(args.input)
    if input_path.is_file():
        with open(input_path, 'r') as f:
            midi_files = [line.strip() for line in f if line.strip()]
    else:
        midi_files = []
        for ext in ['*.mid', '*.midi', '*.MID', '*.MIDI']:
            midi_files.extend(input_path.rglob(ext))
        midi_files = [str(f) for f in midi_files]

    if args.limit:
        midi_files = midi_files[:args.limit]

    print(f"=" * 60)
    print(f"MIDI 预处理 (分片格式)")
    print(f"=" * 60)
    print(f"输入文件数: {len(midi_files):,}")
    print(f"输出目录: {args.output}")
    print(f"工作进程: {args.workers}")
    print(f"分片大小: {args.shard_size}")
    print(f"最大序列长度: {args.max_seq_len}")
    print(f"预计分片数: {len(midi_files) // args.shard_size + 1}")
    print()

    # 准备输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 准备任务
    tasks = [(midi_file, args.max_seq_len) for midi_file in midi_files]

    # 统计
    success = 0
    failed = 0
    total_tokens = 0
    failed_files = []

    # 当前分片
    current_shard = []
    shard_idx = 0
    saved_shards = []

    # 多进程处理 - 单个 Executor，滑动窗口提交
    max_pending = args.workers * 50  # 最大待处理任务数

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        task_iter = iter(tasks)
        submitted = 0

        with tqdm(total=len(tasks), desc="处理中") as pbar:
            # 初始提交
            for _ in range(min(max_pending, len(tasks))):
                task = next(task_iter, None)
                if task is None:
                    break
                futures[executor.submit(process_single_midi, task)] = task
                submitted += 1

            # 滑动窗口处理
            while futures:
                # 等待任意一个完成
                done_futures = []
                for future in list(futures.keys()):
                    if future.done():
                        done_futures.append(future)

                if not done_futures:
                    # 没有完成的，等一下
                    import time
                    time.sleep(0.01)
                    continue

                for future in done_futures:
                    del futures[future]
                    data, midi_path, error = future.result()

                    if data is not None:
                        success += 1
                        total_tokens += data['length']
                        current_shard.append(data)

                        # 分片满了，保存
                        if len(current_shard) >= args.shard_size:
                            shard_path = save_shard(current_shard, shard_idx, output_dir)
                            saved_shards.append(str(shard_path))
                            tqdm.write(f"  保存分片: {shard_path.name} ({len(current_shard)} 样本)")
                            current_shard = []
                            shard_idx += 1
                    else:
                        failed += 1
                        if error:
                            failed_files.append((midi_path, error))

                    pbar.update(1)
                    pbar.set_postfix({
                        'success': success,
                        'failed': failed,
                        'shards': shard_idx,
                        'avg_tokens': total_tokens // max(success, 1)
                    })

                    # 提交新任务
                    if submitted < len(tasks):
                        task = next(task_iter, None)
                        if task is not None:
                            futures[executor.submit(process_single_midi, task)] = task
                            submitted += 1

    # 保存最后一个分片
    if current_shard:
        shard_path = save_shard(current_shard, shard_idx, output_dir)
        saved_shards.append(str(shard_path))
        print(f"  保存分片: {shard_path.name} ({len(current_shard)} 样本)")
        shard_idx += 1

    # 统计
    print()
    print("=" * 60)
    print("预处理完成")
    print("=" * 60)
    print(f"成功: {success:,}")
    print(f"失败: {failed}")
    print(f"分片数: {shard_idx}")
    print(f"总 token 数: {total_tokens:,}")
    print(f"平均 token/样本: {total_tokens // max(success, 1):,}")

    # 保存元数据
    meta = {
        'total_samples': success,
        'total_shards': shard_idx,
        'total_tokens': total_tokens,
        'shard_size': args.shard_size,
        'max_seq_len': args.max_seq_len,
        'vocab_size': HIDTokenizerV2().vocab_size,
        'shards': [Path(s).name for s in saved_shards],
    }

    meta_path = output_dir / 'meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\n元数据已保存: {meta_path}")

    # 保存失败列表
    if failed_files:
        failed_path = output_dir / 'failed.txt'
        with open(failed_path, 'w') as f:
            for path, error in failed_files[:1000]:  # 只保存前 1000 个
                f.write(f"{path}\t{error}\n")
        print(f"失败列表已保存: {failed_path} (前 {min(len(failed_files), 1000)} 个)")

    # 打印分片信息
    print(f"\n分片文件:")
    for shard in saved_shards[:5]:
        print(f"  {Path(shard).name}")
    if len(saved_shards) > 5:
        print(f"  ... 共 {len(saved_shards)} 个分片")


if __name__ == '__main__':
    main()
