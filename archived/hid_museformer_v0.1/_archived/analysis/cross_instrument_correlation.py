#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨乐器相关性分析 - 用于 FC-Attention 掩码设计

相关性 vs 相似度:
- 相似度: 两个小节内容有多像 (Jaccard)
- 相关性: 知道 A 能帮助预测 B 的程度 (Mutual Information)

分析维度:
1. 同乐器不同 offset 的互信息
2. 跨乐器不同 offset 的互信息
3. 条件熵分析

输出:
- 互信息矩阵 (乐器 x offset)
- 相关性热力图
- 掩码设计建议
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Set, Optional
import math

try:
    import mido
    from tqdm import tqdm
    import numpy as np
except ImportError as e:
    print(f"请安装依赖: pip install mido tqdm numpy")
    sys.exit(1)


def extract_bar_features(midi_path: str, ticks_per_bar: int = None) -> Dict[int, Dict[int, Dict]]:
    """
    提取每个乐器每个小节的特征

    Returns:
        {instrument_id: {bar_idx: {
            'pitches': set of pitches,
            'pitch_classes': set of pitch classes (0-11),
            'onsets': list of relative onset positions (0-1),
            'velocities': list of velocities,
            'durations': list of durations,
            'note_count': int,
            'pitch_histogram': [12] int array  # 用于互信息计算
        }}}
    """
    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        return {}

    # 计算 ticks_per_bar
    ticks_per_beat = mid.ticks_per_beat
    if ticks_per_bar is None:
        # 默认 4/4 拍
        numerator, denominator = 4, 4
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'time_signature':
                    numerator = msg.numerator
                    denominator = msg.denominator
                    break
        ticks_per_bar = ticks_per_beat * numerator * 4 // denominator

    # 提取音符
    result = defaultdict(lambda: defaultdict(lambda: {
        'pitches': set(),
        'pitch_classes': set(),
        'onsets': [],
        'velocities': [],
        'durations': [],
        'note_count': 0,
        'pitch_histogram': [0] * 12
    }))

    for track_idx, track in enumerate(mid.tracks):
        abs_time = 0
        current_notes = {}  # {(channel, pitch): start_time}

        # 尝试确定乐器
        instrument = track_idx  # 默认用轨道索引
        for msg in track:
            if msg.type == 'program_change':
                instrument = msg.program
                break

        for msg in track:
            abs_time += msg.time

            if msg.type == 'note_on' and msg.velocity > 0:
                bar_idx = abs_time // ticks_per_bar
                relative_onset = (abs_time % ticks_per_bar) / ticks_per_bar

                current_notes[(msg.channel, msg.note)] = (abs_time, msg.velocity)

                bar_data = result[instrument][bar_idx]
                bar_data['pitches'].add(msg.note)
                bar_data['pitch_classes'].add(msg.note % 12)
                bar_data['onsets'].append(relative_onset)
                bar_data['velocities'].append(msg.velocity)
                bar_data['note_count'] += 1
                bar_data['pitch_histogram'][msg.note % 12] += 1

            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in current_notes:
                    start_time, velocity = current_notes.pop(key)
                    duration = abs_time - start_time
                    bar_idx = start_time // ticks_per_bar
                    result[instrument][bar_idx]['durations'].append(duration / ticks_per_bar)

    return dict(result)


def discretize_features(bar_data: Dict, n_bins: int = 8) -> Tuple[int, ...]:
    """
    将小节特征离散化为符号序列，用于互信息计算

    Returns:
        tuple of discrete symbols representing the bar
    """
    if bar_data['note_count'] == 0:
        return (0,)  # 空小节

    # 特征1: 音符数量 (离散化)
    note_count_bin = min(bar_data['note_count'] // 4, n_bins - 1)

    # 特征2: 音高范围
    if bar_data['pitches']:
        pitch_range = max(bar_data['pitches']) - min(bar_data['pitches'])
        pitch_range_bin = min(pitch_range // 12, n_bins - 1)
    else:
        pitch_range_bin = 0

    # 特征3: 主要音高类 (出现最多的)
    hist = bar_data['pitch_histogram']
    dominant_pc = hist.index(max(hist)) if max(hist) > 0 else 0

    # 特征4: 节奏密度 (onset 分布)
    if bar_data['onsets']:
        avg_onset = sum(bar_data['onsets']) / len(bar_data['onsets'])
        rhythm_bin = int(avg_onset * n_bins) % n_bins
    else:
        rhythm_bin = 0

    return (note_count_bin, pitch_range_bin, dominant_pc, rhythm_bin)


def compute_entropy(symbols: List[Tuple]) -> float:
    """计算符号序列的熵 H(X)"""
    if not symbols:
        return 0.0

    counts = defaultdict(int)
    for s in symbols:
        counts[s] += 1

    total = len(symbols)
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy


def compute_joint_entropy(symbols_x: List[Tuple], symbols_y: List[Tuple]) -> float:
    """计算联合熵 H(X, Y)"""
    if len(symbols_x) != len(symbols_y):
        return 0.0

    joint_counts = defaultdict(int)
    for x, y in zip(symbols_x, symbols_y):
        joint_counts[(x, y)] += 1

    total = len(symbols_x)
    entropy = 0.0
    for count in joint_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy


def compute_mutual_information(symbols_x: List[Tuple], symbols_y: List[Tuple]) -> float:
    """
    计算互信息 I(X; Y) = H(X) + H(Y) - H(X, Y)

    互信息表示知道 X 后 Y 的不确定性减少多少
    """
    h_x = compute_entropy(symbols_x)
    h_y = compute_entropy(symbols_y)
    h_xy = compute_joint_entropy(symbols_x, symbols_y)

    mi = h_x + h_y - h_xy
    return max(0.0, mi)  # 数值误差可能导致微小负值


def compute_normalized_mi(symbols_x: List[Tuple], symbols_y: List[Tuple]) -> float:
    """
    计算归一化互信息 NMI = I(X; Y) / sqrt(H(X) * H(Y))

    范围 [0, 1]，更容易比较
    """
    h_x = compute_entropy(symbols_x)
    h_y = compute_entropy(symbols_y)

    if h_x == 0 or h_y == 0:
        return 0.0

    mi = compute_mutual_information(symbols_x, symbols_y)
    nmi = mi / math.sqrt(h_x * h_y)

    return min(1.0, nmi)  # 确保不超过 1


def analyze_single_file(args: Tuple[str, int]) -> Optional[Dict]:
    """分析单个 MIDI 文件的相关性"""
    midi_path, max_offset = args

    try:
        features = extract_bar_features(midi_path)
    except Exception as e:
        return None

    if not features:
        return None

    # 至少需要 2 个乐器
    instruments = list(features.keys())
    if len(instruments) < 2:
        return None

    # 找到所有乐器共有的小节范围
    all_bars = set()
    for inst_data in features.values():
        all_bars.update(inst_data.keys())

    if len(all_bars) < max_offset + 2:
        return None

    min_bar = min(all_bars)
    max_bar = max(all_bars)

    result = {
        'same_inst': defaultdict(list),  # {offset: [NMI values]}
        'cross_inst': defaultdict(list),  # {offset: [NMI values]}
        'cross_inst_same_bar': [],  # 跨乐器同小节
    }

    # 离散化所有小节
    discrete_bars = {}
    for inst, bars in features.items():
        discrete_bars[inst] = {}
        for bar_idx, bar_data in bars.items():
            discrete_bars[inst][bar_idx] = discretize_features(bar_data)

    # 1. 同乐器相关性
    for inst in instruments:
        inst_bars = discrete_bars.get(inst, {})
        bar_indices = sorted(inst_bars.keys())

        for offset in range(1, max_offset + 1):
            symbols_current = []
            symbols_history = []

            for bar_idx in bar_indices:
                history_idx = bar_idx - offset
                if history_idx in inst_bars:
                    symbols_current.append(inst_bars[bar_idx])
                    symbols_history.append(inst_bars[history_idx])

            if len(symbols_current) >= 10:  # 至少 10 对才有统计意义
                nmi = compute_normalized_mi(symbols_history, symbols_current)
                result['same_inst'][offset].append(nmi)

    # 2. 跨乐器相关性
    for i, inst1 in enumerate(instruments):
        for inst2 in instruments[i+1:]:
            bars1 = discrete_bars.get(inst1, {})
            bars2 = discrete_bars.get(inst2, {})

            # 2.1 跨乐器同小节
            common_bars = set(bars1.keys()) & set(bars2.keys())
            if len(common_bars) >= 10:
                symbols1 = [bars1[b] for b in sorted(common_bars)]
                symbols2 = [bars2[b] for b in sorted(common_bars)]
                nmi = compute_normalized_mi(symbols1, symbols2)
                result['cross_inst_same_bar'].append(nmi)

            # 2.2 跨乐器不同小节
            for offset in range(1, max_offset + 1):
                # inst2 当前小节 vs inst1 历史小节
                symbols_current = []
                symbols_history = []

                for bar_idx in bars2.keys():
                    history_idx = bar_idx - offset
                    if history_idx in bars1:
                        symbols_current.append(bars2[bar_idx])
                        symbols_history.append(bars1[history_idx])

                if len(symbols_current) >= 10:
                    nmi = compute_normalized_mi(symbols_history, symbols_current)
                    result['cross_inst'][offset].append(nmi)

    return result


def main():
    parser = argparse.ArgumentParser(description='跨乐器相关性分析')
    parser.add_argument('--input', type=str, required=True,
                        help='输入文件列表 (valid_files.txt)')
    parser.add_argument('--output', type=str, default='correlation_stats.json',
                        help='输出 JSON 文件')
    parser.add_argument('--max-files', type=int, default=5000,
                        help='最多分析文件数')
    parser.add_argument('--max-offset', type=int, default=32,
                        help='最大小节偏移')
    parser.add_argument('--workers', type=int, default=8,
                        help='并行工作进程数')
    args = parser.parse_args()

    # 读取文件列表
    with open(args.input, 'r') as f:
        files = [line.strip() for line in f if line.strip()]

    files = files[:args.max_files]
    print(f"分析 {len(files)} 个 MIDI 文件...")

    # 聚合结果
    aggregated = {
        'same_inst': defaultdict(list),
        'cross_inst': defaultdict(list),
        'cross_inst_same_bar': [],
    }

    # 并行处理
    tasks = [(f, args.max_offset) for f in files]
    processed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(analyze_single_file, task): task for task in tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc="分析中"):
            result = future.result()
            if result:
                for offset, values in result['same_inst'].items():
                    aggregated['same_inst'][offset].extend(values)
                for offset, values in result['cross_inst'].items():
                    aggregated['cross_inst'][offset].extend(values)
                aggregated['cross_inst_same_bar'].extend(result['cross_inst_same_bar'])
            processed += 1

    # 计算统计量
    stats = {
        'same_inst': {},
        'cross_inst': {},
        'cross_inst_same_bar': {},
        'summary': {}
    }

    # 同乐器统计
    print("\n同乐器相关性 (Normalized Mutual Information):")
    print("-" * 50)
    for offset in sorted(aggregated['same_inst'].keys()):
        values = aggregated['same_inst'][offset]
        if values:
            mean_nmi = sum(values) / len(values)
            std_nmi = (sum((v - mean_nmi) ** 2 for v in values) / len(values)) ** 0.5
            stats['same_inst'][offset] = {
                'mean': mean_nmi,
                'std': std_nmi,
                'count': len(values)
            }
            print(f"  offset={offset:2d}: NMI={mean_nmi:.4f} ± {std_nmi:.4f} (n={len(values)})")

    # 跨乐器同小节统计
    print("\n跨乐器同小节相关性:")
    print("-" * 50)
    values = aggregated['cross_inst_same_bar']
    if values:
        mean_nmi = sum(values) / len(values)
        std_nmi = (sum((v - mean_nmi) ** 2 for v in values) / len(values)) ** 0.5
        stats['cross_inst_same_bar'] = {
            'mean': mean_nmi,
            'std': std_nmi,
            'count': len(values)
        }
        print(f"  offset=0: NMI={mean_nmi:.4f} ± {std_nmi:.4f} (n={len(values)})")

    # 跨乐器不同小节统计
    print("\n跨乐器不同小节相关性:")
    print("-" * 50)
    for offset in sorted(aggregated['cross_inst'].keys()):
        values = aggregated['cross_inst'][offset]
        if values:
            mean_nmi = sum(values) / len(values)
            std_nmi = (sum((v - mean_nmi) ** 2 for v in values) / len(values)) ** 0.5
            stats['cross_inst'][offset] = {
                'mean': mean_nmi,
                'std': std_nmi,
                'count': len(values)
            }
            print(f"  offset={offset:2d}: NMI={mean_nmi:.4f} ± {std_nmi:.4f} (n={len(values)})")

    # 关键发现
    print("\n" + "=" * 50)
    print("关键发现:")
    print("=" * 50)

    # 找同乐器最高/最低
    if stats['same_inst']:
        same_inst_nmis = [(o, s['mean']) for o, s in stats['same_inst'].items()]
        max_same = max(same_inst_nmis, key=lambda x: x[1])
        min_same = min(same_inst_nmis, key=lambda x: x[1])
        print(f"同乐器最高: offset={max_same[0]}, NMI={max_same[1]:.4f}")
        print(f"同乐器最低: offset={min_same[0]}, NMI={min_same[1]:.4f}")
        stats['summary']['same_inst_max'] = {'offset': max_same[0], 'nmi': max_same[1]}
        stats['summary']['same_inst_min'] = {'offset': min_same[0], 'nmi': min_same[1]}

    # 找跨乐器最高/最低
    if stats['cross_inst']:
        cross_inst_nmis = [(o, s['mean']) for o, s in stats['cross_inst'].items()]
        max_cross = max(cross_inst_nmis, key=lambda x: x[1])
        min_cross = min(cross_inst_nmis, key=lambda x: x[1])
        print(f"跨乐器不同小节最高: offset={max_cross[0]}, NMI={max_cross[1]:.4f}")
        print(f"跨乐器不同小节最低: offset={min_cross[0]}, NMI={min_cross[1]:.4f}")
        stats['summary']['cross_inst_max'] = {'offset': max_cross[0], 'nmi': max_cross[1]}
        stats['summary']['cross_inst_min'] = {'offset': min_cross[0], 'nmi': min_cross[1]}

    if stats['cross_inst_same_bar']:
        print(f"跨乐器同小节: NMI={stats['cross_inst_same_bar']['mean']:.4f}")
        stats['summary']['cross_inst_same_bar'] = stats['cross_inst_same_bar']['mean']

    # 掩码设计建议
    print("\n" + "=" * 50)
    print("掩码设计建议 (基于相关性):")
    print("=" * 50)

    # 找同乐器的"有效"offset (NMI > 阈值)
    threshold = 0.1  # NMI 阈值
    effective_same_inst = [o for o, s in stats['same_inst'].items() if s['mean'] > threshold]
    effective_cross_inst = [o for o, s in stats['cross_inst'].items() if s['mean'] > threshold]

    print(f"同乐器有效 offset (NMI > {threshold}): {effective_same_inst[:16]}...")
    print(f"跨乐器有效 offset (NMI > {threshold}): {effective_cross_inst[:8]}...")

    stats['summary']['threshold'] = threshold
    stats['summary']['effective_same_inst'] = effective_same_inst
    stats['summary']['effective_cross_inst'] = effective_cross_inst

    # 保存结果
    with open(args.output, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
