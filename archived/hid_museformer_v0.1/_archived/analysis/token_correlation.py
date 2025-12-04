#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音符内部 Token 相关性分析

分析 HID 编码中不同 token 类型之间的相关性：
- T (Time): 时间位置
- P (Pitch): 音高
- L (Duration): 持续时间
- V (Velocity): 力度
- C (Chord): 和弦

目标：确定是否可以在 token 类型级别做注意力优化
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
import math

try:
    import mido
    from tqdm import tqdm
    import numpy as np
except ImportError as e:
    print(f"请安装依赖: pip install mido tqdm numpy")
    sys.exit(1)


def extract_token_sequences(midi_path: str) -> Optional[Dict]:
    """
    提取每个乐器的 token 序列

    Returns:
        {
            instrument_id: {
                'times': [t1, t2, ...],      # 时间位置序列
                'pitches': [p1, p2, ...],    # 音高序列
                'durations': [d1, d2, ...],  # 持续时间序列
                'velocities': [v1, v2, ...], # 力度序列
            }
        }
    """
    try:
        mid = mido.MidiFile(midi_path)
    except:
        return None

    ticks_per_beat = mid.ticks_per_beat

    # 默认 4/4 拍
    numerator, denominator = 4, 4
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'time_signature':
                numerator = msg.numerator
                denominator = msg.denominator
                break

    ticks_per_bar = ticks_per_beat * numerator * 4 // denominator
    positions_per_bar = 16
    ticks_per_position = ticks_per_bar // positions_per_bar

    result = defaultdict(lambda: {
        'times': [],
        'pitches': [],
        'durations': [],
        'velocities': [],
        'bars': [],
    })

    for track_idx, track in enumerate(mid.tracks):
        abs_time = 0
        current_notes = {}

        instrument = track_idx
        for msg in track:
            if msg.type == 'program_change':
                instrument = msg.program
                break

        for msg in track:
            abs_time += msg.time

            if msg.type == 'note_on' and msg.velocity > 0:
                bar_idx = abs_time // ticks_per_bar
                pos_in_bar = (abs_time % ticks_per_bar) // ticks_per_position
                pos_in_bar = min(pos_in_bar, 15)

                current_notes[(msg.channel, msg.note)] = (abs_time, msg.velocity, bar_idx, pos_in_bar)

            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in current_notes:
                    start_time, velocity, bar_idx, pos_in_bar = current_notes.pop(key)
                    duration_ticks = abs_time - start_time
                    duration_units = max(1, min(64, duration_ticks // ticks_per_position))

                    result[instrument]['times'].append(pos_in_bar)
                    result[instrument]['pitches'].append(msg.note)
                    result[instrument]['durations'].append(duration_units)
                    result[instrument]['velocities'].append(velocity // 4)  # 量化到 32 级
                    result[instrument]['bars'].append(bar_idx)

    # 过滤掉音符太少的乐器
    result = {k: v for k, v in result.items() if len(v['times']) >= 20}

    return dict(result) if result else None


def compute_conditional_entropy(x_seq: List[int], y_seq: List[int], n_bins_x: int, n_bins_y: int) -> float:
    """
    计算条件熵 H(Y|X)

    H(Y|X) = H(X,Y) - H(X)
    """
    if len(x_seq) != len(y_seq) or len(x_seq) < 10:
        return 0.0

    # 联合分布
    joint_counts = defaultdict(int)
    x_counts = defaultdict(int)

    for x, y in zip(x_seq, y_seq):
        joint_counts[(x, y)] += 1
        x_counts[x] += 1

    total = len(x_seq)

    # H(X,Y)
    h_xy = 0.0
    for count in joint_counts.values():
        if count > 0:
            p = count / total
            h_xy -= p * math.log2(p)

    # H(X)
    h_x = 0.0
    for count in x_counts.values():
        if count > 0:
            p = count / total
            h_x -= p * math.log2(p)

    # H(Y|X) = H(X,Y) - H(X)
    h_y_given_x = h_xy - h_x

    return max(0.0, h_y_given_x)


def compute_entropy(seq: List[int]) -> float:
    """计算熵 H(X)"""
    if len(seq) < 10:
        return 0.0

    counts = defaultdict(int)
    for x in seq:
        counts[x] += 1

    total = len(seq)
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy


def compute_mutual_info(x_seq: List[int], y_seq: List[int]) -> float:
    """
    计算互信息 I(X;Y) = H(Y) - H(Y|X)
    """
    if len(x_seq) != len(y_seq) or len(x_seq) < 10:
        return 0.0

    h_y = compute_entropy(y_seq)
    h_y_given_x = compute_conditional_entropy(x_seq, y_seq, 128, 128)

    return max(0.0, h_y - h_y_given_x)


def compute_nmi(x_seq: List[int], y_seq: List[int]) -> float:
    """归一化互信息"""
    if len(x_seq) != len(y_seq) or len(x_seq) < 10:
        return 0.0

    h_x = compute_entropy(x_seq)
    h_y = compute_entropy(y_seq)

    if h_x == 0 or h_y == 0:
        return 0.0

    mi = compute_mutual_info(x_seq, y_seq)
    nmi = mi / math.sqrt(h_x * h_y)

    return min(1.0, nmi)


def analyze_single_file(midi_path: str) -> Optional[Dict]:
    """分析单个文件的 token 相关性"""
    data = extract_token_sequences(midi_path)
    if not data:
        return None

    result = {
        'same_note': defaultdict(list),      # 同一音符内 token 相关性
        'adjacent_note': defaultdict(list),  # 相邻音符间 token 相关性
        'same_bar': defaultdict(list),       # 同小节内 token 相关性
        'cross_bar': defaultdict(list),      # 跨小节 token 相关性
    }

    token_types = ['times', 'pitches', 'durations', 'velocities']

    for inst, inst_data in data.items():
        n_notes = len(inst_data['times'])
        if n_notes < 20:
            continue

        # 1. 同一音符内的 token 相关性 (T-P, T-D, T-V, P-D, P-V, D-V)
        for i, t1 in enumerate(token_types):
            for t2 in token_types[i+1:]:
                seq1 = inst_data[t1]
                seq2 = inst_data[t2]
                nmi = compute_nmi(seq1, seq2)
                result['same_note'][f'{t1[0].upper()}-{t2[0].upper()}'].append(nmi)

        # 2. 相邻音符间的 token 相关性
        for t1 in token_types:
            for t2 in token_types:
                seq1 = inst_data[t1][:-1]  # 前一个音符
                seq2 = inst_data[t2][1:]   # 当前音符
                nmi = compute_nmi(seq1, seq2)
                result['adjacent_note'][f'{t1[0].upper()}→{t2[0].upper()}'].append(nmi)

        # 3. 同小节 vs 跨小节的相关性
        bars = inst_data['bars']
        for t1 in token_types:
            for t2 in token_types:
                same_bar_pairs = []
                cross_bar_pairs = []

                for i in range(1, n_notes):
                    if bars[i] == bars[i-1]:
                        same_bar_pairs.append((inst_data[t1][i-1], inst_data[t2][i]))
                    else:
                        cross_bar_pairs.append((inst_data[t1][i-1], inst_data[t2][i]))

                if len(same_bar_pairs) >= 10:
                    seq1 = [p[0] for p in same_bar_pairs]
                    seq2 = [p[1] for p in same_bar_pairs]
                    nmi = compute_nmi(seq1, seq2)
                    result['same_bar'][f'{t1[0].upper()}→{t2[0].upper()}'].append(nmi)

                if len(cross_bar_pairs) >= 10:
                    seq1 = [p[0] for p in cross_bar_pairs]
                    seq2 = [p[1] for p in cross_bar_pairs]
                    nmi = compute_nmi(seq1, seq2)
                    result['cross_bar'][f'{t1[0].upper()}→{t2[0].upper()}'].append(nmi)

    return result


def main():
    parser = argparse.ArgumentParser(description='Token 相关性分析')
    parser.add_argument('--input', type=str, required=True,
                        help='输入文件列表 (valid_files.txt)')
    parser.add_argument('--output', type=str, default='token_correlation_stats.json',
                        help='输出 JSON 文件')
    parser.add_argument('--max-files', type=int, default=2000,
                        help='最多分析文件数')
    parser.add_argument('--workers', type=int, default=8,
                        help='并行工作进程数')
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        files = [line.strip() for line in f if line.strip()]

    files = files[:args.max_files]
    print(f"分析 {len(files)} 个 MIDI 文件...")

    # 聚合结果
    aggregated = {
        'same_note': defaultdict(list),
        'adjacent_note': defaultdict(list),
        'same_bar': defaultdict(list),
        'cross_bar': defaultdict(list),
    }

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(analyze_single_file, f): f for f in files}

        for future in tqdm(as_completed(futures), total=len(futures), desc="分析中"):
            result = future.result()
            if result:
                for category in aggregated:
                    for key, values in result[category].items():
                        aggregated[category][key].extend(values)

    # 计算统计量
    stats = {}

    print("\n" + "=" * 60)
    print("Token 相关性分析结果 (NMI)")
    print("=" * 60)

    for category, data in aggregated.items():
        print(f"\n【{category}】")
        print("-" * 40)
        stats[category] = {}

        for key in sorted(data.keys()):
            values = data[key]
            if values:
                mean = sum(values) / len(values)
                std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
                stats[category][key] = {'mean': mean, 'std': std, 'count': len(values)}
                print(f"  {key:8s}: NMI = {mean:.4f} ± {std:.4f} (n={len(values)})")

    # 保存结果
    with open(args.output, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
