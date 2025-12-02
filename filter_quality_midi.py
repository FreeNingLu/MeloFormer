#!/usr/bin/env python3
"""MIDI 智能质量筛选脚本"""

import os
import mido
from multiprocessing import Pool
from tqdm import tqdm

def analyze_midi(filepath):
    """分析单个 MIDI 文件"""
    try:
        file_size = os.path.getsize(filepath)
        mid = mido.MidiFile(filepath)

        total_notes = 0
        instruments = set()
        has_drums = False
        unique_pitches = set()

        for track in mid.tracks:
            current_program = 0
            for msg in track:
                if msg.type == 'program_change':
                    current_program = msg.program
                elif msg.type == 'note_on' and msg.velocity > 0:
                    total_notes += 1
                    unique_pitches.add(msg.note)
                    if msg.channel == 9:
                        has_drums = True
                    else:
                        instruments.add(current_program)

        # 计算时长
        try:
            duration = mid.length
        except:
            duration = -1

        num_instruments = len(instruments) + (1 if has_drums else 0)
        pitch_range = max(unique_pitches) - min(unique_pitches) if unique_pitches else 0
        is_solo = num_instruments == 1 and not has_drums

        return {
            'path': filepath,
            'notes': total_notes,
            'instruments': num_instruments,
            'has_drums': has_drums,
            'duration': duration,
            'pitch_range': pitch_range,
            'is_solo': is_solo,
            'size': file_size,
        }
    except:
        return None

def is_high_quality(info):
    """智能质量判断"""
    if info is None:
        return False

    notes = info['notes']
    duration = info['duration']
    pitch_range = info['pitch_range']
    is_solo = info['is_solo']

    # 基础过滤
    if notes < 50:
        return False
    if 0 < duration < 15:
        return False

    # 独奏：要求音高范围 ≥ 1个八度 且 音符 ≥ 100
    if is_solo:
        return notes >= 100 and pitch_range >= 12

    # 多乐器：音符 ≥ 100
    return notes >= 100

def filter_midi_dataset(input_dir, output_file, workers=8):
    """筛选高质量 MIDI"""
    # 收集所有文件
    midi_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith(('.mid', '.midi')):
                midi_files.append(os.path.join(root, f))

    print(f"总文件数: {len(midi_files)}")

    # 并行分析
    with Pool(workers) as pool:
        results = list(tqdm(
            pool.imap(analyze_midi, midi_files, chunksize=100),
            total=len(midi_files),
            desc="分析中"
        ))

    # 筛选
    high_quality = [r for r in results if is_high_quality(r)]

    # 统计
    solo_count = sum(1 for r in high_quality if r['is_solo'])
    multi_count = len(high_quality) - solo_count

    # 被过滤的统计
    filtered_out = [r for r in results if r is not None and not is_high_quality(r)]
    filtered_notes_low = sum(1 for r in filtered_out if r['notes'] < 50)
    filtered_duration_low = sum(1 for r in filtered_out if 0 < r['duration'] < 15)
    filtered_solo_pitch = sum(1 for r in filtered_out if r['is_solo'] and r['pitch_range'] < 12)
    filtered_solo_notes = sum(1 for r in filtered_out if r['is_solo'] and r['notes'] < 100)
    filtered_multi_notes = sum(1 for r in filtered_out if not r['is_solo'] and r['notes'] < 100)

    failed_count = sum(1 for r in results if r is None)

    print(f"\n筛选结果:")
    print(f"  高质量多乐器: {multi_count}")
    print(f"  高质量独奏:   {solo_count}")
    print(f"  总计:         {len(high_quality)} ({100*len(high_quality)/len(midi_files):.1f}%)")
    print(f"\n过滤原因统计:")
    print(f"  解析失败:     {failed_count}")
    print(f"  音符<50:      {filtered_notes_low}")
    print(f"  时长<15秒:    {filtered_duration_low}")
    print(f"  独奏音高<12:  {filtered_solo_pitch}")
    print(f"  独奏音符<100: {filtered_solo_notes}")
    print(f"  多乐器音符<100: {filtered_multi_notes}")

    # 保存
    with open(output_file, 'w') as f:
        for item in high_quality:
            f.write(item['path'] + '\n')

    print(f"\n结果已保存到: {output_file}")

    return high_quality

if __name__ == '__main__':
    filter_midi_dataset(
        input_dir='/Users/freeninglu/Desktop/MuseFormer/midi_valid',
        output_file='/Users/freeninglu/Desktop/MuseFormer/midi_high_quality.txt',
        workers=8
    )
