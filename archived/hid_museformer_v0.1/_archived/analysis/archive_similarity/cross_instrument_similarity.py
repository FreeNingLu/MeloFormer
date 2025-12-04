#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨乐器相关性分析

基于 MIDI 数据统计：
1. 同乐器不同小节的相似度（验证 MuseFormer 的发现）
2. 跨乐器同小节的相似度（和声协调）
3. 跨乐器不同小节的相似度（长距离跨乐器依赖）

用于指导 FC-Attention 掩码设计
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import multiprocessing as mp
from functools import partial
import argparse

try:
    import mido
except ImportError:
    print("请安装 mido: pip install mido")
    sys.exit(1)


def extract_bar_features(midi_path: str, ticks_per_bar: int = None) -> Dict:
    """
    从 MIDI 文件提取每个乐器每个小节的特征

    Returns:
        {
            instrument_id: {
                bar_idx: {
                    'pitches': set of pitches,
                    'pitch_classes': set of pitch classes (0-11),
                    'onset_positions': list of relative positions in bar,
                    'note_count': int,
                    'velocity_mean': float,
                }
            }
        }
    """
    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        return None

    ticks_per_beat = mid.ticks_per_beat
    if ticks_per_bar is None:
        ticks_per_bar = ticks_per_beat * 4  # 假设 4/4 拍

    # 收集所有音符
    instrument_notes = defaultdict(list)  # {program: [(start_tick, pitch, velocity, duration)]}

    for track in mid.tracks:
        current_tick = 0
        current_program = 0
        note_on_times = {}  # {pitch: (start_tick, velocity)}

        for msg in track:
            current_tick += msg.time

            if msg.type == 'program_change':
                current_program = msg.program
            elif msg.type == 'note_on' and msg.velocity > 0:
                note_on_times[msg.note] = (current_tick, msg.velocity)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in note_on_times:
                    start_tick, velocity = note_on_times.pop(msg.note)
                    duration = current_tick - start_tick
                    instrument_notes[current_program].append((start_tick, msg.note, velocity, duration))

    if not instrument_notes:
        return None

    # 按小节整理特征
    features = {}

    for program, notes in instrument_notes.items():
        if not notes:
            continue

        bar_features = defaultdict(lambda: {
            'pitches': set(),
            'pitch_classes': set(),
            'onset_positions': [],
            'velocities': [],
            'note_count': 0,
        })

        for start_tick, pitch, velocity, duration in notes:
            bar_idx = start_tick // ticks_per_bar
            pos_in_bar = (start_tick % ticks_per_bar) / ticks_per_bar  # 0-1

            bar_features[bar_idx]['pitches'].add(pitch)
            bar_features[bar_idx]['pitch_classes'].add(pitch % 12)
            bar_features[bar_idx]['onset_positions'].append(pos_in_bar)
            bar_features[bar_idx]['velocities'].append(velocity)
            bar_features[bar_idx]['note_count'] += 1

        # 计算统计量
        for bar_idx in bar_features:
            vels = bar_features[bar_idx]['velocities']
            bar_features[bar_idx]['velocity_mean'] = np.mean(vels) if vels else 0
            del bar_features[bar_idx]['velocities']

        features[program] = dict(bar_features)

    return features


def jaccard_similarity(set1: Set, set2: Set) -> float:
    """Jaccard 相似度"""
    if not set1 and not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def rhythm_similarity(onsets1: List[float], onsets2: List[float], bins: int = 16) -> float:
    """
    节奏相似度：将 onset 位置量化到 bins 个格子，计算重叠度
    """
    if not onsets1 or not onsets2:
        return 0.0

    # 量化到 bins
    def quantize(onsets):
        hist = set()
        for pos in onsets:
            hist.add(int(pos * bins) % bins)
        return hist

    q1 = quantize(onsets1)
    q2 = quantize(onsets2)

    return jaccard_similarity(q1, q2)


def bar_similarity(feat1: Dict, feat2: Dict) -> Dict[str, float]:
    """
    计算两个小节的多维度相似度
    """
    return {
        'pitch': jaccard_similarity(feat1['pitches'], feat2['pitches']),
        'pitch_class': jaccard_similarity(feat1['pitch_classes'], feat2['pitch_classes']),
        'rhythm': rhythm_similarity(feat1['onset_positions'], feat2['onset_positions']),
    }


def analyze_single_file(midi_path: str, max_offset: int = 32) -> Dict:
    """
    分析单个 MIDI 文件的相关性

    Returns:
        {
            'same_inst': {offset: [similarities]},  # 同乐器不同小节
            'cross_inst_same_bar': {(inst1, inst2): [similarities]},  # 跨乐器同小节
            'cross_inst_diff_bar': {(inst1, inst2, offset): [similarities]},  # 跨乐器不同小节
        }
    """
    features = extract_bar_features(midi_path)
    if features is None or len(features) < 1:
        return None

    results = {
        'same_inst': defaultdict(list),
        'cross_inst_same_bar': defaultdict(list),
        'cross_inst_diff_bar': defaultdict(list),
    }

    instruments = list(features.keys())

    # 1. 同乐器不同小节
    for inst in instruments:
        bars = sorted(features[inst].keys())
        for i, bar1 in enumerate(bars):
            for bar2 in bars[:i]:
                offset = bar1 - bar2
                if offset <= max_offset:
                    sim = bar_similarity(features[inst][bar1], features[inst][bar2])
                    results['same_inst'][offset].append(sim)

    # 2. 跨乐器
    for i, inst1 in enumerate(instruments):
        for inst2 in instruments[i+1:]:
            bars1 = set(features[inst1].keys())
            bars2 = set(features[inst2].keys())

            # 同小节
            common_bars = bars1 & bars2
            for bar in common_bars:
                sim = bar_similarity(features[inst1][bar], features[inst2][bar])
                pair = (min(inst1, inst2), max(inst1, inst2))
                results['cross_inst_same_bar'][pair].append(sim)

            # 不同小节
            for bar1 in bars1:
                for bar2 in bars2:
                    offset = abs(bar1 - bar2)
                    if 0 < offset <= max_offset:
                        sim = bar_similarity(features[inst1][bar1], features[inst2][bar2])
                        pair = (min(inst1, inst2), max(inst1, inst2))
                        results['cross_inst_diff_bar'][(pair, offset)].append(sim)

    return results


def process_file(midi_path: str, max_offset: int = 32) -> Dict:
    """包装函数，处理异常"""
    try:
        return analyze_single_file(midi_path, max_offset)
    except Exception as e:
        return None


def aggregate_results(all_results: List[Dict]) -> Dict:
    """聚合所有文件的结果"""

    aggregated = {
        'same_inst': defaultdict(lambda: {'pitch': [], 'pitch_class': [], 'rhythm': []}),
        'cross_inst_same_bar': defaultdict(lambda: {'pitch': [], 'pitch_class': [], 'rhythm': []}),
        'cross_inst_diff_bar': defaultdict(lambda: {'pitch': [], 'pitch_class': [], 'rhythm': []}),
    }

    for result in all_results:
        if result is None:
            continue

        # 同乐器
        for offset, sims in result['same_inst'].items():
            for sim in sims:
                for key in ['pitch', 'pitch_class', 'rhythm']:
                    aggregated['same_inst'][offset][key].append(sim[key])

        # 跨乐器同小节
        for pair, sims in result['cross_inst_same_bar'].items():
            for sim in sims:
                for key in ['pitch', 'pitch_class', 'rhythm']:
                    aggregated['cross_inst_same_bar'][pair][key].append(sim[key])

        # 跨乐器不同小节
        for (pair, offset), sims in result['cross_inst_diff_bar'].items():
            for sim in sims:
                for key in ['pitch', 'pitch_class', 'rhythm']:
                    aggregated['cross_inst_diff_bar'][(pair, offset)][key].append(sim[key])

    return aggregated


def compute_statistics(aggregated: Dict) -> Dict:
    """计算统计量"""

    stats = {
        'same_inst': {},
        'cross_inst_same_bar': {},
        'cross_inst_diff_bar': {},
    }

    # 同乐器
    for offset, data in aggregated['same_inst'].items():
        stats['same_inst'][offset] = {
            key: {
                'mean': float(np.mean(values)) if values else 0,
                'std': float(np.std(values)) if values else 0,
                'count': len(values),
            }
            for key, values in data.items()
        }

    # 跨乐器同小节
    for pair, data in aggregated['cross_inst_same_bar'].items():
        pair_str = f"{pair[0]}-{pair[1]}"
        stats['cross_inst_same_bar'][pair_str] = {
            key: {
                'mean': float(np.mean(values)) if values else 0,
                'std': float(np.std(values)) if values else 0,
                'count': len(values),
            }
            for key, values in data.items()
        }

    # 跨乐器不同小节：按乐器对聚合
    cross_diff_by_pair = defaultdict(lambda: defaultdict(lambda: {'pitch': [], 'pitch_class': [], 'rhythm': []}))
    for (pair, offset), data in aggregated['cross_inst_diff_bar'].items():
        for key, values in data.items():
            cross_diff_by_pair[pair][offset][key].extend(values)

    for pair, offset_data in cross_diff_by_pair.items():
        pair_str = f"{pair[0]}-{pair[1]}"
        stats['cross_inst_diff_bar'][pair_str] = {}
        for offset, data in offset_data.items():
            stats['cross_inst_diff_bar'][pair_str][offset] = {
                key: {
                    'mean': float(np.mean(values)) if values else 0,
                    'std': float(np.std(values)) if values else 0,
                    'count': len(values),
                }
                for key, values in data.items()
            }

    return stats


def get_instrument_name(program: int) -> str:
    """获取乐器名称"""
    # GM 乐器分类
    categories = {
        (0, 7): 'Piano',
        (8, 15): 'Chromatic Percussion',
        (16, 23): 'Organ',
        (24, 31): 'Guitar',
        (32, 39): 'Bass',
        (40, 47): 'Strings',
        (48, 55): 'Ensemble',
        (56, 63): 'Brass',
        (64, 71): 'Reed',
        (72, 79): 'Pipe',
        (80, 87): 'Synth Lead',
        (88, 95): 'Synth Pad',
        (96, 103): 'Synth Effects',
        (104, 111): 'Ethnic',
        (112, 119): 'Percussive',
        (120, 127): 'Sound Effects',
    }

    for (start, end), name in categories.items():
        if start <= program <= end:
            return name
    return f'Program_{program}'


def print_summary(stats: Dict):
    """打印分析摘要"""

    print("\n" + "=" * 70)
    print("跨乐器相关性分析结果")
    print("=" * 70)

    # 1. 同乐器不同小节
    print("\n【1. 同乐器不同小节的相似度】")
    print("验证 MuseFormer 的发现：前 1, 2, 4, 8... 小节相似度高")
    print("-" * 50)
    print(f"{'Offset':<10} {'Pitch':<12} {'PitchClass':<12} {'Rhythm':<12} {'Count':<10}")
    print("-" * 50)

    # 只显示关键 offset
    key_offsets = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32]
    for offset in key_offsets:
        if offset in stats['same_inst']:
            data = stats['same_inst'][offset]
            pitch_mean = data['pitch']['mean']
            pc_mean = data['pitch_class']['mean']
            rhythm_mean = data['rhythm']['mean']
            count = data['pitch']['count']

            # 高亮 4 的倍数
            marker = " *" if offset in [4, 8, 12, 16, 24, 32] else ""
            print(f"{offset:<10} {pitch_mean:<12.4f} {pc_mean:<12.4f} {rhythm_mean:<12.4f} {count:<10}{marker}")

    # 2. 跨乐器同小节
    print("\n【2. 跨乐器同小节的相似度】")
    print("衡量不同乐器在同一时间的协调程度")
    print("-" * 60)
    print(f"{'Instrument Pair':<25} {'PitchClass':<12} {'Rhythm':<12} {'Count':<10}")
    print("-" * 60)

    # 按 pitch_class 相似度排序
    sorted_pairs = sorted(
        stats['cross_inst_same_bar'].items(),
        key=lambda x: x[1]['pitch_class']['mean'],
        reverse=True
    )

    for pair_str, data in sorted_pairs[:15]:  # Top 15
        parts = pair_str.split('-')
        inst1, inst2 = int(parts[0]), int(parts[1])
        name1 = get_instrument_name(inst1)
        name2 = get_instrument_name(inst2)
        pair_name = f"{name1[:10]}-{name2[:10]}"

        pc_mean = data['pitch_class']['mean']
        rhythm_mean = data['rhythm']['mean']
        count = data['pitch_class']['count']

        print(f"{pair_name:<25} {pc_mean:<12.4f} {rhythm_mean:<12.4f} {count:<10}")

    # 3. 跨乐器不同小节的衰减
    print("\n【3. 跨乐器不同小节的相似度衰减】")
    print("检验跨乐器长距离依赖是否必要")
    print("-" * 60)

    # 聚合所有乐器对的跨小节相似度
    cross_diff_aggregated = defaultdict(lambda: {'pitch_class': [], 'rhythm': []})
    for pair_str, offset_data in stats['cross_inst_diff_bar'].items():
        for offset, data in offset_data.items():
            cross_diff_aggregated[offset]['pitch_class'].append(data['pitch_class']['mean'])
            cross_diff_aggregated[offset]['rhythm'].append(data['rhythm']['mean'])

    print(f"{'Offset':<10} {'PitchClass (avg)':<18} {'Rhythm (avg)':<15}")
    print("-" * 45)

    for offset in key_offsets:
        if offset in cross_diff_aggregated:
            pc_values = cross_diff_aggregated[offset]['pitch_class']
            rhythm_values = cross_diff_aggregated[offset]['rhythm']
            pc_mean = np.mean(pc_values) if pc_values else 0
            rhythm_mean = np.mean(rhythm_values) if rhythm_values else 0
            print(f"{offset:<10} {pc_mean:<18.4f} {rhythm_mean:<15.4f}")

    # 4. 结论建议
    print("\n【4. FC-Attention 掩码设计建议】")
    print("-" * 60)

    # 基于数据给出建议
    same_inst_1 = stats['same_inst'].get(1, {}).get('pitch_class', {}).get('mean', 0)
    same_inst_4 = stats['same_inst'].get(4, {}).get('pitch_class', {}).get('mean', 0)

    print(f"同乐器 offset=1 相似度: {same_inst_1:.4f}")
    print(f"同乐器 offset=4 相似度: {same_inst_4:.4f}")

    if 1 in cross_diff_aggregated:
        cross_1 = np.mean(cross_diff_aggregated[1]['pitch_class'])
        print(f"跨乐器 offset=1 相似度: {cross_1:.4f}")

    if 4 in cross_diff_aggregated:
        cross_4 = np.mean(cross_diff_aggregated[4]['pitch_class'])
        print(f"跨乐器 offset=4 相似度: {cross_4:.4f}")

    print("\n建议的注意力掩码策略：")
    print("  - 同乐器: 保留 offset = 1, 2, 4, 8, 12, 16, 24, 32")
    print("  - 跨乐器同小节: 全连接")
    print("  - 跨乐器不同小节: 根据上述统计决定是否保留")


def main():
    parser = argparse.ArgumentParser(description='跨乐器相关性分析')
    parser.add_argument('--input', type=str, required=True, help='MIDI 文件列表或目录')
    parser.add_argument('--output', type=str, default='similarity_stats.json', help='输出文件')
    parser.add_argument('--max-files', type=int, default=10000, help='最大处理文件数')
    parser.add_argument('--max-offset', type=int, default=32, help='最大小节偏移')
    parser.add_argument('--workers', type=int, default=8, help='并行进程数')

    args = parser.parse_args()

    # 收集 MIDI 文件
    if os.path.isfile(args.input):
        # 文件列表
        with open(args.input, 'r') as f:
            midi_files = [line.strip() for line in f if line.strip()]
    else:
        # 目录
        midi_files = []
        for ext in ['*.mid', '*.midi', '*.MID', '*.MIDI']:
            midi_files.extend(Path(args.input).rglob(ext))
        midi_files = [str(f) for f in midi_files]

    # 限制数量
    if len(midi_files) > args.max_files:
        import random
        random.shuffle(midi_files)
        midi_files = midi_files[:args.max_files]

    print(f"分析 {len(midi_files)} 个 MIDI 文件...")

    # 并行处理
    process_func = partial(process_file, max_offset=args.max_offset)

    with mp.Pool(args.workers) as pool:
        results = list(pool.imap_unordered(process_func, midi_files, chunksize=100))

    # 过滤有效结果
    valid_results = [r for r in results if r is not None]
    print(f"有效结果: {len(valid_results)} / {len(midi_files)}")

    # 聚合
    print("聚合统计...")
    aggregated = aggregate_results(valid_results)

    # 计算统计量
    stats = compute_statistics(aggregated)

    # 保存
    # 需要转换 tuple keys 为 string
    def convert_keys(obj):
        if isinstance(obj, dict):
            return {str(k): convert_keys(v) for k, v in obj.items()}
        return obj

    stats_serializable = convert_keys(stats)

    with open(args.output, 'w') as f:
        json.dump(stats_serializable, f, indent=2)

    print(f"统计结果已保存到: {args.output}")

    # 打印摘要
    print_summary(stats)


if __name__ == '__main__':
    main()
