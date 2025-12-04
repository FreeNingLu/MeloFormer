#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIDI 数据预处理脚本

将 MIDI 文件批量转换为 .pt 格式，训练时直接加载，提升 10-100x 速度

用法:
    python preprocess_data.py --input MIDI/ --output processed_data/ --workers 16

H800 推荐:
    python preprocess_data.py --input MIDI/ --output processed_data/ --workers 32 --max-seq-len 24576
"""

import os
import sys
import argparse
import torch
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import traceback
import io
from contextlib import redirect_stdout
import tempfile

# 添加父目录到 path
sys.path.insert(0, str(Path(__file__).parent))

from data.tokenizer_v2 import HIDTokenizerV2, TokenInfo
from data.midi_to_txt import midi_to_txt


def process_single_midi(args):
    """处理单个 MIDI 文件 (用于多进程)"""
    midi_path, output_path, max_seq_len = args

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
            'is_chord_token': torch.tensor([info.is_chord for info in token_infos], dtype=torch.bool),
            'length': len(token_ids),
        }

        # 保存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(data, output_path)

        return len(token_ids), midi_path, None

    except Exception as e:
        return None, midi_path, str(e)


def main():
    parser = argparse.ArgumentParser(description='MIDI 数据预处理')
    parser.add_argument('--input', type=str, required=True, help='输入 MIDI 目录或文件列表')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--workers', type=int, default=8, help='并行工作进程数')
    parser.add_argument('--max-seq-len', type=int, default=24576, help='最大序列长度 (H800: 24576)')
    parser.add_argument('--limit', type=int, default=None, help='限制处理文件数量 (用于测试)')

    args = parser.parse_args()

    # 收集 MIDI 文件
    input_path = Path(args.input)
    if input_path.is_file():
        # 文件列表
        with open(input_path, 'r') as f:
            midi_files = [line.strip() for line in f if line.strip()]
    else:
        # 目录
        midi_files = []
        for ext in ['*.mid', '*.midi', '*.MID', '*.MIDI']:
            midi_files.extend(input_path.rglob(ext))
        midi_files = [str(f) for f in midi_files]

    if args.limit:
        midi_files = midi_files[:args.limit]

    print(f"找到 {len(midi_files)} 个 MIDI 文件")
    print(f"输出目录: {args.output}")
    print(f"工作进程: {args.workers}")
    print(f"最大序列长度: {args.max_seq_len}")
    print()

    # 准备任务
    output_dir = Path(args.output)
    tasks = []
    for midi_file in midi_files:
        # 构建输出路径
        rel_path = Path(midi_file).stem + '.pt'
        # 用 hash 避免文件名冲突
        file_hash = hex(hash(midi_file) & 0xFFFFFFFF)[2:]
        output_path = output_dir / f"{rel_path[:-3]}_{file_hash}.pt"
        tasks.append((midi_file, str(output_path), args.max_seq_len))

    # 多进程处理
    success = 0
    failed = 0
    total_tokens = 0
    failed_files = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_midi, task): task for task in tasks}

        with tqdm(total=len(tasks), desc="处理中") as pbar:
            for future in as_completed(futures):
                result, midi_path, error = future.result()

                if result is not None:
                    success += 1
                    total_tokens += result
                else:
                    failed += 1
                    failed_files.append((midi_path, error))

                pbar.update(1)
                pbar.set_postfix({
                    'success': success,
                    'failed': failed,
                    'avg_tokens': total_tokens // max(success, 1)
                })

    # 统计
    print()
    print("=" * 60)
    print("预处理完成")
    print("=" * 60)
    print(f"成功: {success}")
    print(f"失败: {failed}")
    print(f"总 token 数: {total_tokens:,}")
    print(f"平均 token/文件: {total_tokens // max(success, 1):,}")

    # 保存元数据
    meta = {
        'total_files': success,
        'total_tokens': total_tokens,
        'max_seq_len': args.max_seq_len,
        'vocab_size': HIDTokenizerV2().vocab_size,
    }

    meta_path = output_dir / 'meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\n元数据已保存: {meta_path}")

    # 保存失败列表
    if failed_files:
        failed_path = output_dir / 'failed.txt'
        with open(failed_path, 'w') as f:
            for path, error in failed_files:
                f.write(f"{path}\t{error}\n")
        print(f"失败列表已保存: {failed_path}")

    # 生成文件列表
    pt_files = list(output_dir.glob('*.pt'))
    list_path = output_dir / 'train_files.txt'
    with open(list_path, 'w') as f:
        for pt_file in pt_files:
            f.write(str(pt_file) + '\n')
    print(f"文件列表已保存: {list_path} ({len(pt_files)} 个文件)")


if __name__ == '__main__':
    main()
