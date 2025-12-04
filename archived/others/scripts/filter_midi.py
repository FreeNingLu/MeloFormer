#!/usr/bin/env python3
"""
MIDI 数据筛选脚本

根据 MuseFormer 论文的筛选规则，对 MIDI 数据集进行质量筛选。

筛选规则：
1. 时长筛选：16 <= n_beats <= 10000
2. 音符数筛选：n_notes >= min_n_notes
3. 音高多样性：n_pitches >= min_n_pitches
4. 跨小节率：cross_bar_rate <= 0.15
5. 拍号筛选：可选只保留 4/4 拍
6. 乐器筛选：可选只保留有效乐器

用法：
    python filter_midi.py --input-dir /path/to/midi --output-dir /path/to/output
"""

import os
import sys
import json
import argparse
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict
import traceback

import mido
from tqdm import tqdm


@dataclass
class FilterConfig:
    """筛选配置"""
    # 时长限制
    min_n_beats: int = 16          # 最少拍数（约4小节）
    max_n_beats: int = 10000       # 最多拍数

    # 音符限制
    min_n_notes: int = 32          # 最少音符数
    max_n_notes: int = 100000      # 最多音符数

    # 音高多样性
    min_n_pitches: int = 8         # 最少不同音高数

    # 跨小节率
    max_cross_bar_rate: float = 0.15

    # 拍号筛选
    filter_time_signature: bool = False  # 是否只保留特定拍号
    allowed_time_signatures: tuple = ((4, 4),)  # 允许的拍号

    # 时长限制（秒）
    min_duration: float = 10.0     # 最短时长
    max_duration: float = 1800.0   # 最长时长（30分钟）

    # 音轨限制
    min_tracks: int = 1            # 最少音轨数
    max_tracks: int = 50           # 最多音轨数

    # 去重
    deduplicate: bool = True       # 是否去重


@dataclass
class MidiInfo:
    """MIDI 文件信息"""
    path: str
    filename: str
    md5: str = ""

    # 基本信息
    n_tracks: int = 0
    n_notes: int = 0
    n_pitches: int = 0
    n_beats: int = 0
    duration: float = 0.0
    ticks_per_beat: int = 480

    # 拍号信息
    time_signatures: List[Tuple[int, int]] = None
    has_time_signature_change: bool = False

    # 速度信息
    tempo_mean: float = 120.0
    tempo_changes: int = 0

    # 质量指标
    cross_bar_rate: float = 0.0
    note_density: float = 0.0  # 每拍音符数

    # 乐器信息
    instruments: List[int] = None
    has_drums: bool = False

    # 筛选结果
    is_valid: bool = True
    reject_reason: str = ""

    def __post_init__(self):
        if self.time_signatures is None:
            self.time_signatures = []
        if self.instruments is None:
            self.instruments = []


def calculate_md5(file_path: str) -> str:
    """计算文件 MD5"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def analyze_midi(file_path: str, config: FilterConfig) -> Optional[MidiInfo]:
    """
    分析单个 MIDI 文件

    返回 MidiInfo 对象，包含分析结果和筛选状态
    """
    info = MidiInfo(
        path=file_path,
        filename=os.path.basename(file_path)
    )

    try:
        mid = mido.MidiFile(file_path)
    except Exception as e:
        info.is_valid = False
        info.reject_reason = f"无法读取: {str(e)[:50]}"
        return info

    # 基本信息
    info.ticks_per_beat = mid.ticks_per_beat
    info.duration = mid.length

    # 收集所有音符和事件
    all_notes = []
    all_pitches = set()
    instruments = set()
    has_drums = False

    # 收集拍号和速度变化
    time_signatures = []
    tempo_changes = []

    for track in mid.tracks:
        current_time = 0
        current_program = 0
        is_drum_track = False

        for msg in track:
            current_time += msg.time

            if msg.type == 'program_change':
                current_program = msg.program
            elif msg.type == 'note_on':
                # 检查是否有必要的属性（mido 用 'note' 而不是 'pitch'）
                if not hasattr(msg, 'velocity') or not hasattr(msg, 'note'):
                    continue
                if msg.velocity <= 0:
                    continue

                pitch = msg.note

                # 判断是否是鼓轨道（channel 9 或 10，MIDI 标准）
                if hasattr(msg, 'channel') and msg.channel in (9, 10):
                    is_drum_track = True
                    has_drums = True
                else:
                    instruments.add(current_program)

                all_notes.append({
                    'pitch': pitch,
                    'start': current_time,
                    'velocity': msg.velocity,
                    'is_drum': is_drum_track
                })
                if not is_drum_track:
                    all_pitches.add(pitch)

            elif msg.type == 'time_signature':
                if hasattr(msg, 'numerator') and hasattr(msg, 'denominator'):
                    time_signatures.append((msg.numerator, msg.denominator))
            elif msg.type == 'set_tempo':
                if hasattr(msg, 'tempo'):
                    tempo_changes.append(mido.tempo2bpm(msg.tempo))

    # 统计信息
    info.n_tracks = len([t for t in mid.tracks if any(m.type == 'note_on' for m in t)])
    info.n_notes = len(all_notes)
    info.n_pitches = len(all_pitches)
    info.instruments = list(instruments)
    info.has_drums = has_drums

    # 拍号信息
    if time_signatures:
        info.time_signatures = list(set(time_signatures))
        info.has_time_signature_change = len(set(time_signatures)) > 1
    else:
        info.time_signatures = [(4, 4)]  # 默认 4/4

    # 速度信息
    if tempo_changes:
        info.tempo_mean = sum(tempo_changes) / len(tempo_changes)
        info.tempo_changes = len(tempo_changes)

    # 计算拍数
    if info.ticks_per_beat > 0 and all_notes:
        max_tick = max(n['start'] for n in all_notes)
        info.n_beats = max_tick // info.ticks_per_beat + 1

    # 计算跨小节率
    if info.n_notes > 0 and info.n_beats > 0:
        # 简化计算：检查音符是否跨越小节边界
        ts = info.time_signatures[0] if info.time_signatures else (4, 4)
        beats_per_bar = ts[0]
        ticks_per_bar = info.ticks_per_beat * beats_per_bar

        cross_bar_count = 0
        for note in all_notes:
            start_bar = note['start'] // ticks_per_bar
            # 简单估算：如果音符在小节后半部分开始，可能跨小节
            pos_in_bar = note['start'] % ticks_per_bar
            if pos_in_bar > ticks_per_bar * 0.75:
                cross_bar_count += 1

        info.cross_bar_rate = cross_bar_count / info.n_notes if info.n_notes > 0 else 0

    # 计算音符密度
    if info.n_beats > 0:
        info.note_density = info.n_notes / info.n_beats

    # ========== 筛选检查 ==========

    # 1. 检查是否能读取
    if info.n_notes == 0:
        info.is_valid = False
        info.reject_reason = "无音符"
        return info

    # 2. 时长检查
    if info.n_beats < config.min_n_beats:
        info.is_valid = False
        info.reject_reason = f"太短: {info.n_beats} 拍 < {config.min_n_beats}"
        return info

    if info.n_beats > config.max_n_beats:
        info.is_valid = False
        info.reject_reason = f"太长: {info.n_beats} 拍 > {config.max_n_beats}"
        return info

    # 3. 秒数检查
    if info.duration < config.min_duration:
        info.is_valid = False
        info.reject_reason = f"时长太短: {info.duration:.1f}s < {config.min_duration}s"
        return info

    if info.duration > config.max_duration:
        info.is_valid = False
        info.reject_reason = f"时长太长: {info.duration:.1f}s > {config.max_duration}s"
        return info

    # 4. 音符数检查
    if info.n_notes < config.min_n_notes:
        info.is_valid = False
        info.reject_reason = f"音符太少: {info.n_notes} < {config.min_n_notes}"
        return info

    if info.n_notes > config.max_n_notes:
        info.is_valid = False
        info.reject_reason = f"音符太多: {info.n_notes} > {config.max_n_notes}"
        return info

    # 5. 音高多样性检查
    if info.n_pitches < config.min_n_pitches:
        info.is_valid = False
        info.reject_reason = f"音高太少: {info.n_pitches} < {config.min_n_pitches}"
        return info

    # 6. 跨小节率检查
    if info.cross_bar_rate > config.max_cross_bar_rate:
        info.is_valid = False
        info.reject_reason = f"跨小节率过高: {info.cross_bar_rate:.2f} > {config.max_cross_bar_rate}"
        return info

    # 7. 拍号检查
    if config.filter_time_signature:
        valid_ts = False
        for ts in info.time_signatures:
            if ts in config.allowed_time_signatures:
                valid_ts = True
                break
        if not valid_ts:
            info.is_valid = False
            info.reject_reason = f"拍号不符: {info.time_signatures}"
            return info

    # 8. 音轨数检查
    if info.n_tracks < config.min_tracks:
        info.is_valid = False
        info.reject_reason = f"音轨太少: {info.n_tracks} < {config.min_tracks}"
        return info

    if info.n_tracks > config.max_tracks:
        info.is_valid = False
        info.reject_reason = f"音轨太多: {info.n_tracks} > {config.max_tracks}"
        return info

    # 计算 MD5 用于去重
    if config.deduplicate:
        info.md5 = calculate_md5(file_path)

    return info


def process_file(args) -> Optional[MidiInfo]:
    """处理单个文件的包装函数（用于多进程）"""
    file_path, config = args
    try:
        return analyze_midi(file_path, config)
    except Exception as e:
        info = MidiInfo(path=file_path, filename=os.path.basename(file_path))
        info.is_valid = False
        info.reject_reason = f"处理错误: {str(e)[:50]}"
        return info


def find_midi_files(input_dir: str) -> List[str]:
    """递归查找所有 MIDI 文件"""
    midi_files = []
    extensions = {'.mid', '.midi', '.MID', '.MIDI'}

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                midi_files.append(os.path.join(root, file))

    return midi_files


def filter_dataset(
    input_dir: str,
    output_dir: str,
    config: FilterConfig,
    num_workers: int = 8,
    save_stats: bool = True
) -> Dict:
    """
    筛选整个数据集

    Args:
        input_dir: 输入目录
        output_dir: 输出目录（存放结果文件）
        config: 筛选配置
        num_workers: 并行工作进程数
        save_stats: 是否保存统计信息

    Returns:
        统计信息字典
    """
    print("=" * 60)
    print("MIDI 数据筛选")
    print("=" * 60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print()

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 查找所有 MIDI 文件
    print("查找 MIDI 文件...")
    midi_files = find_midi_files(input_dir)
    total_files = len(midi_files)
    print(f"找到 {total_files:,} 个 MIDI 文件")
    print()

    if total_files == 0:
        print("没有找到 MIDI 文件！")
        return {}

    # 并行处理 - 使用 Pool.imap 避免内存问题
    print(f"开始分析（{num_workers} 进程）...")
    results = []

    from multiprocessing import Pool

    # 准备参数
    args_list = [(f, config) for f in midi_files]

    with Pool(processes=num_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_file, args_list, chunksize=100),
                          total=total_files, desc="分析进度"):
            if result:
                results.append(result)

    print()

    # 统计结果
    valid_results = [r for r in results if r.is_valid]
    invalid_results = [r for r in results if not r.is_valid]

    # 去重
    if config.deduplicate:
        print("去重中...")
        seen_md5 = set()
        unique_results = []
        duplicates = 0

        for r in valid_results:
            if r.md5 not in seen_md5:
                seen_md5.add(r.md5)
                unique_results.append(r)
            else:
                duplicates += 1
                r.is_valid = False
                r.reject_reason = "重复文件"
                invalid_results.append(r)

        valid_results = unique_results
        print(f"去除 {duplicates:,} 个重复文件")

    # 统计拒绝原因
    reject_reasons = {}
    for r in invalid_results:
        reason = r.reject_reason.split(':')[0] if ':' in r.reject_reason else r.reject_reason
        reject_reasons[reason] = reject_reasons.get(reason, 0) + 1

    # 打印统计
    print()
    print("=" * 60)
    print("筛选结果统计")
    print("=" * 60)
    print(f"总文件数:     {total_files:,}")
    print(f"有效文件数:   {len(valid_results):,} ({len(valid_results)/total_files*100:.1f}%)")
    print(f"无效文件数:   {len(invalid_results):,} ({len(invalid_results)/total_files*100:.1f}%)")
    print()

    print("拒绝原因分布:")
    for reason, count in sorted(reject_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count:,} ({count/len(invalid_results)*100:.1f}%)")
    print()

    # 有效文件统计
    if valid_results:
        avg_notes = sum(r.n_notes for r in valid_results) / len(valid_results)
        avg_duration = sum(r.duration for r in valid_results) / len(valid_results)
        avg_tracks = sum(r.n_tracks for r in valid_results) / len(valid_results)
        avg_beats = sum(r.n_beats for r in valid_results) / len(valid_results)

        print("有效文件统计:")
        print(f"  平均音符数: {avg_notes:.0f}")
        print(f"  平均时长:   {avg_duration:.0f} 秒")
        print(f"  平均音轨数: {avg_tracks:.1f}")
        print(f"  平均拍数:   {avg_beats:.0f}")

    # 保存结果
    if save_stats:
        # 保存有效文件列表
        valid_list_path = os.path.join(output_dir, 'valid_files.txt')
        with open(valid_list_path, 'w') as f:
            for r in valid_results:
                f.write(r.path + '\n')
        print(f"\n有效文件列表已保存: {valid_list_path}")

        # 保存无效文件列表
        invalid_list_path = os.path.join(output_dir, 'invalid_files.txt')
        with open(invalid_list_path, 'w') as f:
            for r in invalid_results:
                f.write(f"{r.path}\t{r.reject_reason}\n")
        print(f"无效文件列表已保存: {invalid_list_path}")

        # 保存详细统计
        stats = {
            'config': asdict(config),
            'total_files': total_files,
            'valid_files': len(valid_results),
            'invalid_files': len(invalid_results),
            'reject_reasons': reject_reasons,
            'valid_stats': {
                'avg_notes': avg_notes if valid_results else 0,
                'avg_duration': avg_duration if valid_results else 0,
                'avg_tracks': avg_tracks if valid_results else 0,
                'avg_beats': avg_beats if valid_results else 0,
            }
        }

        stats_path = os.path.join(output_dir, 'filter_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"统计信息已保存: {stats_path}")

        # 保存详细信息（用于后续分析）
        details_path = os.path.join(output_dir, 'valid_details.jsonl')
        with open(details_path, 'w') as f:
            for r in valid_results:
                detail = {
                    'path': r.path,
                    'filename': r.filename,
                    'md5': r.md5,
                    'n_tracks': r.n_tracks,
                    'n_notes': r.n_notes,
                    'n_pitches': r.n_pitches,
                    'n_beats': r.n_beats,
                    'duration': r.duration,
                    'time_signatures': r.time_signatures,
                    'tempo_mean': r.tempo_mean,
                    'instruments': r.instruments,
                    'has_drums': r.has_drums,
                }
                f.write(json.dumps(detail, ensure_ascii=False) + '\n')
        print(f"详细信息已保存: {details_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description='MIDI 数据筛选脚本')

    # 输入输出
    parser.add_argument('--input-dir', type=str, required=True,
                        help='输入 MIDI 文件目录')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='输出目录')

    # 筛选参数
    parser.add_argument('--min-beats', type=int, default=16,
                        help='最少拍数 (默认: 16)')
    parser.add_argument('--max-beats', type=int, default=10000,
                        help='最多拍数 (默认: 10000)')
    parser.add_argument('--min-notes', type=int, default=32,
                        help='最少音符数 (默认: 32)')
    parser.add_argument('--max-notes', type=int, default=100000,
                        help='最多音符数 (默认: 100000)')
    parser.add_argument('--min-pitches', type=int, default=8,
                        help='最少不同音高数 (默认: 8)')
    parser.add_argument('--max-cross-bar-rate', type=float, default=0.15,
                        help='最大跨小节率 (默认: 0.15)')
    parser.add_argument('--min-duration', type=float, default=10.0,
                        help='最短时长秒数 (默认: 10)')
    parser.add_argument('--max-duration', type=float, default=1800.0,
                        help='最长时长秒数 (默认: 1800)')
    parser.add_argument('--filter-ts', action='store_true',
                        help='是否只保留 4/4 拍')
    parser.add_argument('--no-deduplicate', action='store_true',
                        help='不进行去重')

    # 运行参数
    parser.add_argument('--workers', type=int, default=8,
                        help='并行进程数 (默认: 8)')

    args = parser.parse_args()

    # 创建配置
    config = FilterConfig(
        min_n_beats=args.min_beats,
        max_n_beats=args.max_beats,
        min_n_notes=args.min_notes,
        max_n_notes=args.max_notes,
        min_n_pitches=args.min_pitches,
        max_cross_bar_rate=args.max_cross_bar_rate,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        filter_time_signature=args.filter_ts,
        deduplicate=not args.no_deduplicate,
    )

    # 打印配置
    print("筛选配置:")
    print(f"  拍数范围: {config.min_n_beats} - {config.max_n_beats}")
    print(f"  音符范围: {config.min_n_notes} - {config.max_n_notes}")
    print(f"  最少音高: {config.min_n_pitches}")
    print(f"  跨小节率: <= {config.max_cross_bar_rate}")
    print(f"  时长范围: {config.min_duration} - {config.max_duration} 秒")
    print(f"  只保留4/4: {config.filter_time_signature}")
    print(f"  去重: {config.deduplicate}")
    print()

    # 运行筛选
    filter_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config=config,
        num_workers=args.workers,
    )


if __name__ == '__main__':
    main()
