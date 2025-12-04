#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TXT 到 MIDI 转换器

支持格式：
- 量化格式 (Bar + 16分音符位置) - 头部 "H ... Q"
- 无损格式 (Bar + Tick) - 头部 "H ... {ticks_per_bar}"
- 旧版 Delta Time 格式 (兼容)
"""

import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage
import re
from typing import List, Tuple, Dict

# 量化参数 (需要与 midi_to_txt.py 一致)
POSITIONS_PER_BAR = 16  # 每小节 16 个位置 (16 分音符)
VELOCITY_BINS = 32      # 力度量化为 32 级


def dequantize_velocity(bin_idx: int) -> int:
    """反量化 velocity"""
    return min(127, (bin_idx * 128 + VELOCITY_BINS // 2) // VELOCITY_BINS)


def txt_to_midi(txt_file_path: str, output_midi_path: str):
    """
    将文本格式转换回 MIDI 文件
    自动检测格式类型
    """
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 检测格式类型
    is_quantized = False
    is_bar_tick = False
    is_ultra = False

    for line in lines[:10]:
        line = line.strip()
        if line.startswith("H "):
            parts = line.split()
            # 量化格式: H 轨道数 ticks_per_beat Q
            if len(parts) == 4 and parts[3] == 'Q':
                is_quantized = True
            # 无损格式: H 轨道数 ticks_per_beat ticks_per_bar (数字)
            elif len(parts) == 4 and parts[3].isdigit():
                is_bar_tick = True
            elif len(parts) == 3:
                is_ultra = True
            break

    if is_quantized:
        return _parse_quantized_format(lines, output_midi_path)
    elif is_bar_tick:
        return _parse_bar_tick_format(lines, output_midi_path)
    elif is_ultra:
        return _parse_ultra_format(lines, output_midi_path)
    else:
        return _parse_basic_format(lines, output_midi_path)


def _parse_bar_tick_format(lines: List[str], output_midi_path: str):
    """解析 Bar + Tick 格式 (无损)"""
    bpm_changes = []  # [(tick, bpm), ...]
    time_sig_changes = []  # [(tick, num, denom), ...]
    # 每个 track 独立存储: [(program, channel, notes), ...]
    tracks_data = []

    ticks_per_beat = 480
    ticks_per_bar = 1920

    current_program = None
    current_channel = 0
    current_notes = None
    current_bar = 0
    last_velocity = 64
    last_duration = 480

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # 解析 Header: H 轨道数 ticks_per_beat ticks_per_bar
        if line.startswith("H "):
            parts = line.split()
            if len(parts) >= 4:
                ticks_per_beat = int(parts[2])
                ticks_per_bar = int(parts[3])

        # 解析 Tempo: T B{bar} T{tick} {bpm}
        elif line.startswith("T ") and not line.startswith("TS "):
            parts = line.split()
            if len(parts) >= 4:
                bar = int(parts[1][1:])  # B{bar}
                tick_in_bar = int(parts[2][1:])  # T{tick}
                bpm = float(parts[3])
                tick = bar * ticks_per_bar + tick_in_bar
                bpm_changes.append((tick, bpm))

        # 解析 Time Signature: TS B{bar} T{tick} {num}/{denom}
        elif line.startswith("TS "):
            parts = line.split()
            if len(parts) >= 4:
                bar = int(parts[1][1:])
                tick_in_bar = int(parts[2][1:])
                ts_parts = parts[3].split('/')
                num = int(ts_parts[0])
                denom = int(ts_parts[1])
                tick = bar * ticks_per_bar + tick_in_bar
                time_sig_changes.append((tick, num, denom))

        # 解析音色标记: #P{program} (simplified, no channel)
        elif line.startswith("#"):
            # 保存前一个 track
            if current_notes is not None and len(current_notes) > 0:
                tracks_data.append((current_program, current_channel, current_notes))

            program_str = line[1:].strip()

            # Drum track: #D (channel 9)
            if program_str == "D":
                current_program = 0  # Drum 不需要 program change
                current_channel = 9
            # 新格式: #P{program} (只区分 program，不区分 channel，类似 REMI)
            elif program_str.startswith("P"):
                current_program = int(program_str[1:])
                current_channel = 0
            # 旧格式兼容: #C{channel}:{program}
            elif program_str.startswith("C") and ":" in program_str:
                parts = program_str[1:].split(":")
                current_channel = int(parts[0])
                current_program = int(parts[1])
            # 旧格式兼容: #D:{program} (Drum channel)
            elif program_str.startswith("D:"):
                current_channel = 9
                current_program = int(program_str[2:])
            # 旧格式兼容: #{program}
            elif program_str.isdigit():
                current_channel = 0
                current_program = int(program_str)
            else:
                match = re.search(r'timbre=(\d+)', line)
                if match:
                    current_program = int(match.group(1))
                    current_channel = 0
                else:
                    continue

            current_notes = []
            current_bar = 0
            last_velocity = 64
            last_duration = 480

        # 解析音符行
        elif current_notes is not None:
            parts = line.split()
            if not parts:
                continue

            # 解析 Bar + Tick 格式的音符
            bar = current_bar
            tick_in_bar = None
            pitch = None
            duration = last_duration
            velocity = last_velocity

            for part in parts:
                if part.startswith('B'):
                    bar = int(part[1:])
                    current_bar = bar
                elif part.startswith('T'):
                    tick_in_bar = int(part[1:])
                elif part.startswith('P'):
                    pitch = int(part[1:])
                elif part.startswith('L'):
                    duration = int(part[1:])
                    last_duration = duration
                elif part.startswith('V'):
                    velocity = int(part[1:])
                    last_velocity = velocity

            if pitch is not None and tick_in_bar is not None:
                tick = bar * ticks_per_bar + tick_in_bar
                current_notes.append((tick, pitch, velocity, duration))

    # 保存最后一个 track
    if current_notes is not None and len(current_notes) > 0:
        tracks_data.append((current_program, current_channel, current_notes))

    # 创建 MIDI 文件
    return _create_midi_file_v2(bpm_changes, tracks_data, ticks_per_beat,
                                output_midi_path, time_sig_changes=time_sig_changes)


def _parse_quantized_format(lines: List[str], output_midi_path: str):
    """解析量化格式 (Bar + 16分音符位置)"""
    bpm_changes = []  # [(tick, bpm), ...]
    time_sig_changes = []  # [(tick, num, denom), ...]
    tracks_data = []  # [(program, channel, notes), ...]

    ticks_per_beat = 480
    # 默认 4/4 拍
    ticks_per_bar = ticks_per_beat * 4
    ticks_per_16th = ticks_per_bar // POSITIONS_PER_BAR

    current_program = None
    current_channel = 0
    current_notes = None
    current_bar = 0
    last_velocity_bin = 16  # 中等力度
    last_duration_units = 4  # 4 个 16 分音符 = 1 拍

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # 解析 Header: H 轨道数 ticks_per_beat Q
        if line.startswith("H "):
            parts = line.split()
            if len(parts) >= 3:
                ticks_per_beat = int(parts[2])
                # 重新计算
                ticks_per_bar = ticks_per_beat * 4
                ticks_per_16th = ticks_per_bar // POSITIONS_PER_BAR

        # 解析 Tempo: T B{bar} T{pos} {bpm}
        elif line.startswith("T ") and not line.startswith("TS "):
            parts = line.split()
            if len(parts) >= 4:
                bar = int(parts[1][1:])  # B{bar}
                pos = int(parts[2][1:])  # T{pos} (0-15)
                bpm = float(parts[3])
                tick = bar * ticks_per_bar + pos * ticks_per_16th
                bpm_changes.append((tick, bpm))

        # 解析 Time Signature: TS B{bar} T{pos} {num}/{denom}
        elif line.startswith("TS "):
            parts = line.split()
            if len(parts) >= 4:
                bar = int(parts[1][1:])
                pos = int(parts[2][1:])
                ts_parts = parts[3].split('/')
                num = int(ts_parts[0])
                denom = int(ts_parts[1])
                tick = bar * ticks_per_bar + pos * ticks_per_16th
                time_sig_changes.append((tick, num, denom))

        # 解析音色标记: #P{program} (simplified, no channel)
        elif line.startswith("#"):
            # 保存前一个 track
            if current_notes is not None and len(current_notes) > 0:
                tracks_data.append((current_program, current_channel, current_notes))

            program_str = line[1:].strip()

            # Drum track: #D (channel 9)
            if program_str == "D":
                current_program = 0
                current_channel = 9
            # 新格式: #P{program} (只区分 program，不区分 channel，类似 REMI)
            elif program_str.startswith("P"):
                current_program = int(program_str[1:])
                current_channel = 0
            # 旧格式兼容: #C{channel}:{program}
            elif program_str.startswith("C") and ":" in program_str:
                parts = program_str[1:].split(":")
                current_channel = int(parts[0])
                current_program = int(parts[1])
            else:
                # 兼容旧格式
                current_channel = 0
                current_program = int(program_str) if program_str.isdigit() else 0

            current_notes = []
            current_bar = 0
            last_pos = None  # 新增：追踪上一个 Position
            last_velocity_bin = 16
            last_duration_units = 4

        # 解析音符行（支持每行多个音符）
        elif current_notes is not None:
            parts = line.split()
            if not parts:
                continue

            # 解析量化格式的音符
            bar = current_bar
            pos = last_pos  # 默认继承上一个位置
            duration_units = last_duration_units
            velocity_bin = last_velocity_bin

            for part in parts:
                if part.startswith('B'):
                    bar = int(part[1:])
                    current_bar = bar
                    last_pos = None  # 新 bar 重置 position
                    pos = None
                elif part.startswith('T'):
                    pos = int(part[1:])
                    last_pos = pos  # 记录当前位置
                elif part.startswith('P'):
                    pitch = int(part[1:])
                    # 遇到 P token 时立即保存音符
                    if pos is not None:
                        tick = bar * ticks_per_bar + pos * ticks_per_16th
                        duration = duration_units * ticks_per_16th
                        velocity = dequantize_velocity(velocity_bin)
                        current_notes.append((tick, pitch, velocity, duration))
                elif part.startswith('L'):
                    duration_units = int(part[1:])
                    last_duration_units = duration_units
                    # 如果已经有待处理的音符，更新其 duration
                    if current_notes and current_notes[-1][0] == bar * ticks_per_bar + (pos if pos else 0) * ticks_per_16th:
                        old_note = current_notes.pop()
                        current_notes.append((old_note[0], old_note[1], old_note[2], duration_units * ticks_per_16th))
                elif part.startswith('V'):
                    velocity_bin = int(part[1:])
                    last_velocity_bin = velocity_bin
                    # 如果已经有待处理的音符，更新其 velocity
                    if current_notes and current_notes[-1][0] == bar * ticks_per_bar + (pos if pos else 0) * ticks_per_16th:
                        old_note = current_notes.pop()
                        current_notes.append((old_note[0], old_note[1], dequantize_velocity(velocity_bin), old_note[3]))

    # 保存最后一个 track
    if current_notes is not None and len(current_notes) > 0:
        tracks_data.append((current_program, current_channel, current_notes))

    # 创建 MIDI 文件
    return _create_midi_file_v2(bpm_changes, tracks_data, ticks_per_beat,
                                output_midi_path, time_sig_changes=time_sig_changes)


def _parse_ultra_format(lines: List[str], output_midi_path: str):
    """解析旧版 Delta Time 格式 (兼容)"""
    bpm_changes = []
    notes_by_program = {}

    ticks_per_beat = 480
    current_program = None
    current_tick = 0
    last_note = None

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # 解析 Header
        if line.startswith("H "):
            parts = line.split()
            if len(parts) >= 3:
                ticks_per_beat = int(parts[2])

        # 解析 Tempo
        elif line.startswith("T "):
            parts = line.split()
            if len(parts) >= 3:
                tick = int(parts[1])
                bpm = float(parts[2])
                bpm_changes.append((tick, bpm))

        # 解析音色标记
        elif line.startswith("#"):
            program_str = line[1:].strip()
            if program_str.isdigit():
                current_program = int(program_str)
            else:
                match = re.search(r'timbre=(\d+)', line)
                if match:
                    current_program = int(match.group(1))
                else:
                    continue

            if current_program not in notes_by_program:
                notes_by_program[current_program] = []
            current_tick = 0
            last_note = None

        # 解析音符行 (Delta Time 格式)
        elif current_program is not None:
            parts = line.split()
            if not parts:
                continue

            parsed = _try_parse_delta_note(parts, last_note)
            if parsed:
                delta_time, pitch, velocity, duration = parsed
                current_tick += delta_time
                last_note = (pitch, velocity, duration)
                notes_by_program[current_program].append((current_tick, pitch, velocity, duration))

    return _create_midi_file(bpm_changes, notes_by_program, ticks_per_beat, output_midi_path)


def _try_parse_delta_note(parts, last_note):
    """尝试解析 Delta Time 格式的音符"""
    if not parts:
        return None

    idx = 0
    delta_time = 0

    # 解析 D{delta}
    if parts[idx].startswith('D'):
        try:
            delta_time = int(parts[idx][1:])
        except ValueError:
            return None
        idx += 1

    if idx >= len(parts) or not parts[idx].startswith('P'):
        return None

    try:
        pitch = int(parts[idx][1:])
    except ValueError:
        return None
    idx += 1

    duration = last_note[2] if last_note else 480
    velocity = last_note[1] if last_note else 64

    while idx < len(parts):
        if parts[idx].startswith('L'):
            try:
                duration = int(parts[idx][1:])
            except ValueError:
                pass
        elif parts[idx].startswith('V'):
            try:
                velocity = int(parts[idx][1:])
            except ValueError:
                pass
        idx += 1

    return delta_time, pitch, velocity, duration


def _parse_basic_format(lines: List[str], output_midi_path: str):
    """解析基础版格式"""
    bpm_changes = []
    notes_by_program = {}

    current_program = None
    parsing_notes = False
    ticks_per_beat = 480

    for line in lines:
        line = line.strip()

        if not line:
            parsing_notes = False
            continue

        if line.startswith("BPM:"):
            bpm_str = line[4:].strip()
            for segment in bpm_str.split(','):
                segment = segment.strip()
                match = re.match(r'([\d.]+)->([\d.]+)', segment)
                if match:
                    beat = float(match.group(1))
                    bpm = float(match.group(2))
                    tick = int(beat * ticks_per_beat)
                    bpm_changes.append((tick, bpm))

        elif line.startswith("音色:") or line.startswith("timbre:"):
            current_program = int(line.split(':')[1].strip())
            notes_by_program[current_program] = []
            parsing_notes = False

        elif line.startswith("节拍") or line.startswith("B"):
            parsing_notes = True

        elif parsing_notes and current_program is not None:
            parts = line.split('\t')
            if len(parts) == 4:
                try:
                    beat = float(parts[0])
                    pitch = int(parts[1])
                    velocity = int(parts[2])
                    duration = float(parts[3])
                    tick = int(beat * ticks_per_beat)
                    dur_ticks = int(duration * ticks_per_beat)
                    notes_by_program[current_program].append((tick, pitch, velocity, dur_ticks))
                except ValueError:
                    continue

    return _create_midi_file(bpm_changes, notes_by_program, ticks_per_beat, output_midi_path)


def _create_midi_file_v2(bpm_changes: List[Tuple[int, float]],
                         tracks_data: List[Tuple[int, int, List[Tuple[int, int, int, int]]]],
                         ticks_per_beat: int,
                         output_midi_path: str,
                         time_sig_changes: List[Tuple[int, int, int]] = None):
    """创建 MIDI 文件 (v2: 保持 track 独立)

    Args:
        tracks_data: [(program, channel, notes), ...]
    """
    mid = MidiFile()
    mid.ticks_per_beat = ticks_per_beat

    # 创建 Tempo Track
    tempo_track = MidiTrack()
    mid.tracks.append(tempo_track)

    # 合并 time_signature 和 tempo 事件
    meta_events = []
    if time_sig_changes:
        for tick, num, denom in time_sig_changes:
            meta_events.append((tick, 'time_sig', num, denom))
    if bpm_changes:
        for tick, bpm in bpm_changes:
            meta_events.append((tick, 'tempo', bpm))

    meta_events.sort(key=lambda x: (x[0], 0 if x[1] == 'time_sig' else 1))

    last_tick = 0
    for event in meta_events:
        tick = event[0]
        event_type = event[1]
        delta_time = tick - last_tick

        if event_type == 'time_sig':
            tempo_track.append(MetaMessage('time_signature',
                                           numerator=event[2],
                                           denominator=event[3],
                                           time=delta_time))
        elif event_type == 'tempo':
            tempo = mido.bpm2tempo(event[2])
            tempo_track.append(MetaMessage('set_tempo', tempo=tempo, time=delta_time))
        last_tick = tick

    # 为每个 track 创建独立的 MIDI track
    for program, channel, notes in tracks_data:
        track = MidiTrack()
        mid.tracks.append(track)

        # 设置音色
        track.append(Message('program_change', program=program, channel=channel, time=0))

        # 创建事件列表
        events = []
        for start_tick, pitch, velocity, duration in notes:
            end_tick = start_tick + duration
            events.append((start_tick, 'note_on', pitch, velocity))
            events.append((end_tick, 'note_off', pitch, 0))

        # 按时间排序
        events.sort(key=lambda x: (x[0], x[1] == 'note_off'))

        # 生成 MIDI 消息
        last_tick = 0
        for event in events:
            tick = event[0]
            event_type = event[1]
            delta_time = tick - last_tick

            if event_type == 'note_on':
                track.append(Message('note_on', note=event[2], velocity=event[3],
                                     channel=channel, time=delta_time))
            elif event_type == 'note_off':
                track.append(Message('note_off', note=event[2], velocity=0,
                                     channel=channel, time=delta_time))
            last_tick = tick

        track.append(MetaMessage('end_of_track', time=0))

    mid.save(output_midi_path)
    print(f"✅ 转换完成: {output_midi_path}")


def _create_midi_file(bpm_changes: List[Tuple[int, float]],
                      notes_by_program: Dict[int, List[Tuple[int, int, int, int]]],
                      ticks_per_beat: int,
                      output_midi_path: str,
                      control_changes: Dict[int, List[Tuple[int, int, int]]] = None,
                      time_sig_changes: List[Tuple[int, int, int]] = None):
    """创建 MIDI 文件"""
    mid = MidiFile()
    mid.ticks_per_beat = ticks_per_beat

    # 创建 Tempo Track
    tempo_track = MidiTrack()
    mid.tracks.append(tempo_track)

    # 合并 time_signature 和 tempo 事件，按时间排序
    meta_events = []

    if time_sig_changes:
        for tick, num, denom in time_sig_changes:
            meta_events.append((tick, 'time_sig', num, denom))

    if bpm_changes:
        for tick, bpm in bpm_changes:
            meta_events.append((tick, 'tempo', bpm))

    # 按时间排序，同时刻 time_sig 优先
    meta_events.sort(key=lambda x: (x[0], 0 if x[1] == 'time_sig' else 1))

    last_tick = 0
    for event in meta_events:
        tick = event[0]
        event_type = event[1]
        delta_time = tick - last_tick

        if event_type == 'time_sig':
            tempo_track.append(MetaMessage('time_signature',
                                           numerator=event[2],
                                           denominator=event[3],
                                           time=delta_time))
        elif event_type == 'tempo':
            tempo = mido.bpm2tempo(event[2])
            tempo_track.append(MetaMessage('set_tempo', tempo=tempo, time=delta_time))

        last_tick = tick

    # 为每个音色创建一个轨道
    for program in sorted(notes_by_program.keys()):
        track = MidiTrack()
        mid.tracks.append(track)

        # 判断是否为鼓组 (Program >= 256 表示 Channel 9)
        if program >= 256:
            channel = 9
            actual_program = program - 256
        else:
            channel = 0
            actual_program = program

        # 设置音色
        track.append(Message('program_change', program=actual_program, channel=channel, time=0))

        notes = notes_by_program[program]

        # 创建事件列表
        events = []

        # 添加 Control Changes
        if control_changes and program in control_changes:
            for tick, cc_num, value in control_changes[program]:
                events.append((tick, 'cc', cc_num, value))

        # 添加音符事件
        for start_tick, pitch, velocity, duration in notes:
            end_tick = start_tick + duration
            events.append((start_tick, 'note_on', pitch, velocity))
            events.append((end_tick, 'note_off', pitch, 0))

        # 按时间排序
        events.sort(key=lambda x: (x[0], x[1] == 'note_off'))

        # 生成 MIDI 消息
        last_tick = 0
        for event in events:
            tick = event[0]
            event_type = event[1]
            delta_time = tick - last_tick

            if event_type == 'cc':
                track.append(Message('control_change', control=event[2], value=event[3],
                                     channel=channel, time=delta_time))
            elif event_type == 'note_on':
                track.append(Message('note_on', note=event[2], velocity=event[3],
                                     channel=channel, time=delta_time))
            elif event_type == 'note_off':
                track.append(Message('note_off', note=event[2], velocity=0,
                                     channel=channel, time=delta_time))

            last_tick = tick

        # 添加结束标记
        track.append(MetaMessage('end_of_track', time=0))

    # 保存 MIDI 文件
    mid.save(output_midi_path)
    print(f"✅ 转换完成: {output_midi_path}")


def compare_midi_files(file1: str, file2: str, tolerance: int = 0) -> Tuple[bool, str]:
    """
    比较两个 MIDI 文件是否相同

    Args:
        file1: 第一个 MIDI 文件路径
        file2: 第二个 MIDI 文件路径
        tolerance: tick 容差 (0 表示精确匹配)

    Returns: (是否匹配, 差异描述)
    """
    try:
        mid1 = MidiFile(file1)
        mid2 = MidiFile(file2)
    except Exception as e:
        return False, f"无法读取文件: {e}"

    def extract_info(mid):
        info = {
            'ticks_per_beat': mid.ticks_per_beat,
            'tempos': [],
            'time_sigs': [],
            'notes_by_program': {}
        }

        for track in mid.tracks:
            current_tick = 0
            current_program = 0
            active_notes = {}  # {note: (start_tick, velocity)}

            for msg in track:
                current_tick += msg.time

                if msg.type == 'set_tempo':
                    bpm = round(mido.tempo2bpm(msg.tempo), 2)
                    info['tempos'].append((current_tick, bpm))

                elif msg.type == 'time_signature':
                    info['time_sigs'].append((current_tick, msg.numerator, msg.denominator))

                elif msg.type == 'program_change':
                    if msg.channel == 9:
                        current_program = 256 + msg.program
                    else:
                        current_program = msg.program

                elif msg.type == 'note_on' and msg.velocity > 0:
                    active_notes[msg.note] = (current_tick, msg.velocity)

                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in active_notes:
                        start_tick, velocity = active_notes[msg.note]
                        duration = current_tick - start_tick
                        if current_program not in info['notes_by_program']:
                            info['notes_by_program'][current_program] = []
                        info['notes_by_program'][current_program].append(
                            (start_tick, msg.note, velocity, duration)
                        )
                        del active_notes[msg.note]

        # 排序
        info['tempos'].sort()
        info['time_sigs'].sort()
        for prog in info['notes_by_program']:
            info['notes_by_program'][prog].sort()

        return info

    info1 = extract_info(mid1)
    info2 = extract_info(mid2)

    # 比较 ticks_per_beat
    if info1['ticks_per_beat'] != info2['ticks_per_beat']:
        return False, f"ticks_per_beat 不同: {info1['ticks_per_beat']} vs {info2['ticks_per_beat']}"

    # 比较 tempo
    if len(info1['tempos']) != len(info2['tempos']):
        return False, f"Tempo 数量不同: {len(info1['tempos'])} vs {len(info2['tempos'])}"

    for i, ((t1, b1), (t2, b2)) in enumerate(zip(info1['tempos'], info2['tempos'])):
        if abs(t1 - t2) > tolerance or abs(b1 - b2) > 0.01:
            return False, f"Tempo #{i} 不同: ({t1}, {b1}) vs ({t2}, {b2})"

    # 比较 time signatures
    if len(info1['time_sigs']) != len(info2['time_sigs']):
        return False, f"Time Signature 数量不同: {len(info1['time_sigs'])} vs {len(info2['time_sigs'])}"

    for i, (ts1, ts2) in enumerate(zip(info1['time_sigs'], info2['time_sigs'])):
        if abs(ts1[0] - ts2[0]) > tolerance or ts1[1:] != ts2[1:]:
            return False, f"Time Signature #{i} 不同: {ts1} vs {ts2}"

    # 比较音符
    programs1 = set(info1['notes_by_program'].keys())
    programs2 = set(info2['notes_by_program'].keys())

    if programs1 != programs2:
        return False, f"音色不同: {sorted(programs1)} vs {sorted(programs2)}"

    for prog in programs1:
        notes1 = info1['notes_by_program'][prog]
        notes2 = info2['notes_by_program'][prog]

        if len(notes1) != len(notes2):
            return False, f"Program {prog} 音符数不同: {len(notes1)} vs {len(notes2)}"

        for i, (n1, n2) in enumerate(zip(notes1, notes2)):
            tick1, pitch1, vel1, dur1 = n1
            tick2, pitch2, vel2, dur2 = n2

            if (abs(tick1 - tick2) > tolerance or
                pitch1 != pitch2 or
                vel1 != vel2 or
                abs(dur1 - dur2) > tolerance):
                return False, f"Program {prog} 音符 #{i} 不同:\n  原始: tick={tick1}, pitch={pitch1}, vel={vel1}, dur={dur1}\n  转换: tick={tick2}, pitch={pitch2}, vel={vel2}, dur={dur2}"

    return True, "完全匹配"


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("用法: python txt_to_midi.py <input.txt> <output.mid>")
        sys.exit(1)

    txt_file = sys.argv[1]
    output_midi = sys.argv[2]

    txt_to_midi(txt_file, output_midi)
