#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token 跨类型相关性分析 (按小节距离)

分析维度：
1. 同小节内 T↔P, T↔D, T↔V, P↔D, P↔V, D↔V
2. 不同小节距离 (offset=1,2,3,...) 的跨类型相关性
3. 同类型 vs 跨类型对比

Token 类型：
- T (Time): 时间位置 (0-15)
- P (Pitch): 音高 (0-127)
- D (Duration): 持续时间
- V (Velocity): 力度
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


def extract_bar_features(midi_path: str) -> Optional[Dict]:
    """
    提取每个小节的 token 特征

    Returns:
        {
            instrument_id: {
                bar_idx: {
                    'times': [t1, t2, ...],
                    'pitches': [p1, p2, ...],
                    'durations': [d1, d2, ...],
                    'velocities': [v1, v2, ...]
                }
            }
        }
    """
    try:
        mid = mido.MidiFile(midi_path)
    except:
        return None

    ticks_per_beat = mid.ticks_per_beat
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

    result = defaultdict(lambda: defaultdict(lambda: {
        'times': [], 'pitches': [], 'durations': [], 'velocities': []
    }))

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

                    result[instrument][bar_idx]['times'].append(pos_in_bar)
                    result[instrument][bar_idx]['pitches'].append(msg.note)
                    result[instrument][bar_idx]['durations'].append(duration_units)
                    result[instrument][bar_idx]['velocities'].append(velocity // 4)  # 量化到32级

    # 过滤：至少有10个小节
    filtered = {}
    for inst, bars in result.items():
        if len(bars) >= 10:
            filtered[inst] = dict(bars)

    return filtered if filtered else None


def compute_nmi(x_seq: List[int], y_seq: List[int]) -> float:
    """计算归一化互信息"""
    if len(x_seq) != len(y_seq) or len(x_seq) < 5:
        return 0.0

    def entropy(seq):
        counts = defaultdict(int)
        for x in seq:
            counts[x] += 1
        total = len(seq)
        h = 0.0
        for c in counts.values():
            if c > 0:
                p = c / total
                h -= p * math.log2(p)
        return h

    def joint_entropy(seq1, seq2):
        counts = defaultdict(int)
        for x, y in zip(seq1, seq2):
            counts[(x, y)] += 1
        total = len(seq1)
        h = 0.0
        for c in counts.values():
            if c > 0:
                p = c / total
                h -= p * math.log2(p)
        return h

    h_x = entropy(x_seq)
    h_y = entropy(y_seq)
    h_xy = joint_entropy(x_seq, y_seq)

    if h_x == 0 or h_y == 0:
        return 0.0

    mi = h_x + h_y - h_xy
    nmi = mi / math.sqrt(h_x * h_y)
    return min(1.0, max(0.0, nmi))


def aggregate_bar_features(bar_data: Dict) -> Dict[str, int]:
    """将小节内的多个音符聚合为单个特征向量（用于小节级别的相关性分析）"""
    if not bar_data['times']:
        return None

    # 聚合策略：使用统计特征
    # Time: 使用最常见的位置 (mode)
    # Pitch: 使用平均值的区间 (分成12个半音组)
    # Duration: 使用平均值的区间
    # Velocity: 使用平均值的区间

    from collections import Counter

    times = bar_data['times']
    pitches = bar_data['pitches']
    durations = bar_data['durations']
    velocities = bar_data['velocities']

    # Time: mode (最常见位置)
    time_mode = Counter(times).most_common(1)[0][0]

    # Pitch: 平均值离散化到12个区间
    pitch_avg = sum(pitches) / len(pitches)
    pitch_bin = min(11, int(pitch_avg // 10.67))  # 0-127 -> 0-11

    # Duration: 平均值离散化到8个区间
    dur_avg = sum(durations) / len(durations)
    dur_bin = min(7, int(dur_avg // 8))  # 1-64 -> 0-7

    # Velocity: 平均值离散化到8个区间
    vel_avg = sum(velocities) / len(velocities)
    vel_bin = min(7, int(vel_avg // 4))  # 0-31 -> 0-7

    return {
        'T': time_mode,
        'P': pitch_bin,
        'D': dur_bin,
        'V': vel_bin
    }


def analyze_single_file(midi_path: str) -> Optional[Dict]:
    """分析单个文件"""
    data = extract_bar_features(midi_path)
    if not data:
        return None

    token_types = ['T', 'P', 'D', 'V']
    max_offset = 16  # 最多看16小节

    result = {
        # 同小节内跨类型
        'same_bar_cross_type': defaultdict(list),  # {'T→P': [NMI], ...}

        # 按小节距离的跨类型相关性
        'by_offset': {},  # {offset: {'T→P': [NMI], 'T→T': [NMI], ...}}
    }

    for offset in range(1, max_offset + 1):
        result['by_offset'][offset] = defaultdict(list)

    for inst, bars in data.items():
        if len(bars) < 10:
            continue

        # 聚合每个小节的特征
        bar_indices = sorted(bars.keys())
        aggregated = {}
        for bar_idx in bar_indices:
            agg = aggregate_bar_features(bars[bar_idx])
            if agg:
                aggregated[bar_idx] = agg

        if len(aggregated) < 10:
            continue

        bar_list = sorted(aggregated.keys())

        # 1. 同小节内跨类型相关性
        for i, t1 in enumerate(token_types):
            for t2 in token_types[i+1:]:
                seq1 = [aggregated[b][t1] for b in bar_list]
                seq2 = [aggregated[b][t2] for b in bar_list]
                nmi = compute_nmi(seq1, seq2)
                key = f'{t1}→{t2}'
                result['same_bar_cross_type'][key].append(nmi)

        # 2. 按小节距离的相关性
        for offset in range(1, max_offset + 1):
            # 找到有效的小节对
            valid_pairs = []
            for bar_idx in bar_list:
                prev_idx = bar_idx - offset
                if prev_idx in aggregated:
                    valid_pairs.append((prev_idx, bar_idx))

            if len(valid_pairs) < 5:
                continue

            # 计算所有 token 类型组合
            for t1 in token_types:
                for t2 in token_types:
                    seq_prev = [aggregated[p[0]][t1] for p in valid_pairs]
                    seq_curr = [aggregated[p[1]][t2] for p in valid_pairs]
                    nmi = compute_nmi(seq_prev, seq_curr)
                    key = f'{t1}→{t2}'
                    result['by_offset'][offset][key].append(nmi)

    return result


def main():
    parser = argparse.ArgumentParser(description='Token 跨类型相关性分析 (按小节)')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='token_cross_type_by_bar.json')
    parser.add_argument('--max-files', type=int, default=2000)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        files = [line.strip() for line in f if line.strip()]
    files = files[:args.max_files]
    print(f"分析 {len(files)} 个文件...")

    # 聚合
    agg = {
        'same_bar_cross_type': defaultdict(list),
        'by_offset': {i: defaultdict(list) for i in range(1, 17)},
    }

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(analyze_single_file, f): f for f in files}
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                for key, vals in result['same_bar_cross_type'].items():
                    agg['same_bar_cross_type'][key].extend(vals)
                for offset, data in result['by_offset'].items():
                    for key, vals in data.items():
                        agg['by_offset'][offset][key].extend(vals)

    # 统计
    stats = {
        'same_bar_cross_type': {},
        'by_offset': {},
    }

    token_types = ['T', 'P', 'D', 'V']

    print("\n" + "=" * 70)
    print("Token 跨类型相关性分析 (按小节距离)")
    print("=" * 70)

    # 同小节内跨类型
    print("\n【同小节内 跨类型相关性】")
    print("-" * 40)
    for key in sorted(agg['same_bar_cross_type'].keys()):
        vals = agg['same_bar_cross_type'][key]
        if vals:
            mean = sum(vals) / len(vals)
            stats['same_bar_cross_type'][key] = mean
            print(f"  {key}: NMI = {mean:.4f}")

    # 按小节距离
    print("\n【按小节距离的 Token 相关性】")
    print("-" * 70)

    # 表头
    all_pairs = [f'{t1}→{t2}' for t1 in token_types for t2 in token_types]
    same_type_pairs = ['T→T', 'P→P', 'D→D', 'V→V']
    cross_type_pairs = ['T→P', 'P→T', 'T→D', 'D→T', 'T→V', 'V→T', 'P→D', 'D→P', 'P→V', 'V→P', 'D→V', 'V→D']

    # 先打印同类型
    print("\n同类型 (Same Type):")
    header = f"{'Offset':>6} |" + "".join(f"{p:>8}" for p in same_type_pairs)
    print(header)
    print("-" * len(header))

    for offset in range(1, 17):
        stats['by_offset'][offset] = {}
        row = f"{offset:>6} |"
        for pair in same_type_pairs:
            vals = agg['by_offset'][offset].get(pair, [])
            if vals:
                mean = sum(vals) / len(vals)
                stats['by_offset'][offset][pair] = mean
                row += f"{mean:>8.4f}"
            else:
                row += f"{'N/A':>8}"
        print(row)

    # 再打印关键跨类型
    print("\n跨类型 (Cross Type) - 关键组合:")
    key_cross = ['T→P', 'P→T', 'T→D', 'P→D', 'T→V', 'P→V']
    header = f"{'Offset':>6} |" + "".join(f"{p:>8}" for p in key_cross)
    print(header)
    print("-" * len(header))

    for offset in range(1, 17):
        row = f"{offset:>6} |"
        for pair in key_cross:
            vals = agg['by_offset'][offset].get(pair, [])
            if vals:
                mean = sum(vals) / len(vals)
                stats['by_offset'][offset][pair] = mean
                row += f"{mean:>8.4f}"
            else:
                row += f"{'N/A':>8}"
        print(row)

    # 结论
    print("\n" + "=" * 70)
    print("分析结论")
    print("=" * 70)

    # 计算平均值
    tt_avg = sum(stats['by_offset'][i].get('T→T', 0) for i in range(1, 5)) / 4
    pp_avg = sum(stats['by_offset'][i].get('P→P', 0) for i in range(1, 5)) / 4
    tp_avg = sum(stats['by_offset'][i].get('T→P', 0) for i in range(1, 5)) / 4
    td_avg = sum(stats['by_offset'][i].get('T→D', 0) for i in range(1, 5)) / 4
    tv_avg = sum(stats['by_offset'][i].get('T→V', 0) for i in range(1, 5)) / 4

    print(f"\n近距离 (offset 1-4) 平均 NMI:")
    print(f"  同类型: T→T={tt_avg:.4f}, P→P={pp_avg:.4f}")
    print(f"  跨类型: T→P={tp_avg:.4f}, T→D={td_avg:.4f}, T→V={tv_avg:.4f}")

    if tt_avg > 0:
        print(f"\n跨类型相对于同类型的比例:")
        print(f"  T→P / T→T = {tp_avg/tt_avg*100:.1f}%")
        print(f"  T→D / T→T = {td_avg/tt_avg*100:.1f}%")
        print(f"  T→V / T→T = {tv_avg/tt_avg*100:.1f}%")

    # 保存
    with open(args.output, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
