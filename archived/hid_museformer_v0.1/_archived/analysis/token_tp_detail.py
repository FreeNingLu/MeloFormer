#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细研究 T→P, P→T 跨类型 token 相关性

分析维度：
1. 不同距离（相邻、隔1个、隔2个音符...）
2. 同小节 vs 跨小节
3. 不同节奏位置的影响
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


def extract_note_sequences(midi_path: str) -> Optional[Dict]:
    """提取音符序列"""
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

    result = defaultdict(list)

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
                    result[instrument].append({
                        'time': pos_in_bar,
                        'pitch': msg.note,
                        'bar': bar_idx,
                        'abs_time': start_time,
                    })

    # 按时间排序
    for inst in result:
        result[inst].sort(key=lambda x: x['abs_time'])

    result = {k: v for k, v in result.items() if len(v) >= 30}
    return dict(result) if result else None


def compute_nmi(x_seq: List[int], y_seq: List[int]) -> float:
    """计算归一化互信息"""
    if len(x_seq) != len(y_seq) or len(x_seq) < 10:
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


def analyze_single_file(midi_path: str) -> Optional[Dict]:
    """分析单个文件"""
    data = extract_note_sequences(midi_path)
    if not data:
        return None

    result = {
        # 按音符距离
        'T_to_P_by_distance': defaultdict(list),  # {distance: [NMI]}
        'P_to_T_by_distance': defaultdict(list),

        # 同小节 vs 跨小节
        'T_to_P_same_bar': [],
        'T_to_P_cross_bar': [],
        'P_to_T_same_bar': [],
        'P_to_T_cross_bar': [],

        # 按节奏位置分组
        'T_to_P_by_position': defaultdict(list),  # {(pos_prev, pos_curr): [NMI]}

        # 同类型对比
        'T_to_T_by_distance': defaultdict(list),
        'P_to_P_by_distance': defaultdict(list),
    }

    for inst, notes in data.items():
        n = len(notes)
        if n < 30:
            continue

        times = [note['time'] for note in notes]
        pitches = [note['pitch'] for note in notes]
        bars = [note['bar'] for note in notes]

        # 1. 按音符距离分析 (distance = 1, 2, 3, ... 10)
        for dist in range(1, 11):
            if n <= dist:
                continue

            # T[i-dist] → P[i]
            t_prev = times[:-dist]
            p_curr = pitches[dist:]
            if len(t_prev) >= 10:
                nmi = compute_nmi(t_prev, p_curr)
                result['T_to_P_by_distance'][dist].append(nmi)

            # P[i-dist] → T[i]
            p_prev = pitches[:-dist]
            t_curr = times[dist:]
            if len(p_prev) >= 10:
                nmi = compute_nmi(p_prev, t_curr)
                result['P_to_T_by_distance'][dist].append(nmi)

            # 对比: T→T, P→P
            t_prev_t = times[:-dist]
            t_curr_t = times[dist:]
            if len(t_prev_t) >= 10:
                nmi = compute_nmi(t_prev_t, t_curr_t)
                result['T_to_T_by_distance'][dist].append(nmi)

            p_prev_p = pitches[:-dist]
            p_curr_p = pitches[dist:]
            if len(p_prev_p) >= 10:
                nmi = compute_nmi(p_prev_p, p_curr_p)
                result['P_to_P_by_distance'][dist].append(nmi)

        # 2. 同小节 vs 跨小节 (相邻音符)
        same_bar_tp = []
        cross_bar_tp = []
        same_bar_pt = []
        cross_bar_pt = []

        for i in range(1, n):
            if bars[i] == bars[i-1]:
                same_bar_tp.append((times[i-1], pitches[i]))
                same_bar_pt.append((pitches[i-1], times[i]))
            else:
                cross_bar_tp.append((times[i-1], pitches[i]))
                cross_bar_pt.append((pitches[i-1], times[i]))

        if len(same_bar_tp) >= 10:
            t_seq = [x[0] for x in same_bar_tp]
            p_seq = [x[1] for x in same_bar_tp]
            result['T_to_P_same_bar'].append(compute_nmi(t_seq, p_seq))

        if len(cross_bar_tp) >= 10:
            t_seq = [x[0] for x in cross_bar_tp]
            p_seq = [x[1] for x in cross_bar_tp]
            result['T_to_P_cross_bar'].append(compute_nmi(t_seq, p_seq))

        if len(same_bar_pt) >= 10:
            p_seq = [x[0] for x in same_bar_pt]
            t_seq = [x[1] for x in same_bar_pt]
            result['P_to_T_same_bar'].append(compute_nmi(p_seq, t_seq))

        if len(cross_bar_pt) >= 10:
            p_seq = [x[0] for x in cross_bar_pt]
            t_seq = [x[1] for x in cross_bar_pt]
            result['P_to_T_cross_bar'].append(compute_nmi(p_seq, t_seq))

        # 3. 按节奏位置分组
        # 强拍: 0, 4, 8, 12  弱拍: 其他
        def is_strong_beat(pos):
            return pos in [0, 4, 8, 12]

        for i in range(1, n):
            prev_pos = times[i-1]
            curr_pos = times[i]
            prev_strong = is_strong_beat(prev_pos)
            curr_strong = is_strong_beat(curr_pos)
            key = (prev_strong, curr_strong)
            # 这里简单记录，后续聚合

    return result


def main():
    parser = argparse.ArgumentParser(description='T↔P 详细相关性分析')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='token_tp_detail_stats.json')
    parser.add_argument('--max-files', type=int, default=2000)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        files = [line.strip() for line in f if line.strip()]
    files = files[:args.max_files]
    print(f"分析 {len(files)} 个文件...")

    # 聚合
    agg = {
        'T_to_P_by_distance': defaultdict(list),
        'P_to_T_by_distance': defaultdict(list),
        'T_to_P_same_bar': [],
        'T_to_P_cross_bar': [],
        'P_to_T_same_bar': [],
        'P_to_T_cross_bar': [],
        'T_to_T_by_distance': defaultdict(list),
        'P_to_P_by_distance': defaultdict(list),
    }

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(analyze_single_file, f): f for f in files}
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                for dist, vals in result['T_to_P_by_distance'].items():
                    agg['T_to_P_by_distance'][dist].extend(vals)
                for dist, vals in result['P_to_T_by_distance'].items():
                    agg['P_to_T_by_distance'][dist].extend(vals)
                for dist, vals in result['T_to_T_by_distance'].items():
                    agg['T_to_T_by_distance'][dist].extend(vals)
                for dist, vals in result['P_to_P_by_distance'].items():
                    agg['P_to_P_by_distance'][dist].extend(vals)
                agg['T_to_P_same_bar'].extend(result['T_to_P_same_bar'])
                agg['T_to_P_cross_bar'].extend(result['T_to_P_cross_bar'])
                agg['P_to_T_same_bar'].extend(result['P_to_T_same_bar'])
                agg['P_to_T_cross_bar'].extend(result['P_to_T_cross_bar'])

    # 统计
    stats = {}

    print("\n" + "=" * 60)
    print("T→P, P→T 跨类型相关性详细分析")
    print("=" * 60)

    print("\n【按音符距离】")
    print("-" * 60)
    print(f"{'Dist':>4} | {'T→P':>8} | {'P→T':>8} | {'T→T':>8} | {'P→P':>8}")
    print("-" * 60)

    stats['by_distance'] = {}
    for dist in range(1, 11):
        tp = agg['T_to_P_by_distance'].get(dist, [])
        pt = agg['P_to_T_by_distance'].get(dist, [])
        tt = agg['T_to_T_by_distance'].get(dist, [])
        pp = agg['P_to_P_by_distance'].get(dist, [])

        tp_mean = sum(tp) / len(tp) if tp else 0
        pt_mean = sum(pt) / len(pt) if pt else 0
        tt_mean = sum(tt) / len(tt) if tt else 0
        pp_mean = sum(pp) / len(pp) if pp else 0

        stats['by_distance'][dist] = {
            'T_to_P': tp_mean,
            'P_to_T': pt_mean,
            'T_to_T': tt_mean,
            'P_to_P': pp_mean,
        }
        print(f"{dist:>4} | {tp_mean:>8.4f} | {pt_mean:>8.4f} | {tt_mean:>8.4f} | {pp_mean:>8.4f}")

    print("\n【同小节 vs 跨小节】(相邻音符)")
    print("-" * 40)

    def calc_mean(lst):
        return sum(lst) / len(lst) if lst else 0

    tp_same = calc_mean(agg['T_to_P_same_bar'])
    tp_cross = calc_mean(agg['T_to_P_cross_bar'])
    pt_same = calc_mean(agg['P_to_T_same_bar'])
    pt_cross = calc_mean(agg['P_to_T_cross_bar'])

    stats['same_vs_cross'] = {
        'T_to_P_same_bar': tp_same,
        'T_to_P_cross_bar': tp_cross,
        'P_to_T_same_bar': pt_same,
        'P_to_T_cross_bar': pt_cross,
    }

    print(f"T→P 同小节: {tp_same:.4f}")
    print(f"T→P 跨小节: {tp_cross:.4f}")
    print(f"P→T 同小节: {pt_same:.4f}")
    print(f"P→T 跨小节: {pt_cross:.4f}")

    # 结论
    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)

    # 找衰减点
    tp_dist1 = stats['by_distance'][1]['T_to_P']
    tp_dist5 = stats['by_distance'][5]['T_to_P']
    tp_dist10 = stats['by_distance'][10]['T_to_P']

    print(f"\nT→P 相关性衰减:")
    print(f"  距离=1: {tp_dist1:.4f}")
    print(f"  距离=5: {tp_dist5:.4f} (衰减 {(1-tp_dist5/tp_dist1)*100:.1f}%)")
    print(f"  距离=10: {tp_dist10:.4f} (衰减 {(1-tp_dist10/tp_dist1)*100:.1f}%)")

    tt_dist1 = stats['by_distance'][1]['T_to_T']
    pp_dist1 = stats['by_distance'][1]['P_to_P']

    print(f"\n对比同类型 (距离=1):")
    print(f"  T→T: {tt_dist1:.4f}")
    print(f"  P→P: {pp_dist1:.4f}")
    print(f"  T→P: {tp_dist1:.4f} (是 T→T 的 {tp_dist1/tt_dist1*100:.1f}%)")

    print(f"\n同小节 vs 跨小节:")
    print(f"  T→P 同小节: {tp_same:.4f}")
    print(f"  T→P 跨小节: {tp_cross:.4f}")
    if tp_same > 0:
        print(f"  跨小节相对衰减: {(1-tp_cross/tp_same)*100:.1f}%")

    # 保存
    with open(args.output, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
