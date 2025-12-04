#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MIDI 到 TXT 转换器

支持两种模式：
1. 无损模式 (quantize=False): 使用精确 tick 值
2. 量化模式 (quantize=True): 使用 16 分音符网格 (0-15)

时间表示方式：
- B{bar} - 小节号 (从0开始)
- T{pos} - 小节内位置
  - 无损模式: 精确 tick 偏移 (0 到 ticks_per_bar-1)
  - 量化模式: 16 分音符位置 (0-15)
- P{pitch} - 音高 (0-127)
- L{duration} - 持续时间
  - 无损模式: 精确 ticks
  - 量化模式: 16 分音符数量 (1-64)
- V{velocity} - 力度 (0-127，量化为 32 bins)

Track 处理：
- 按 program 分组，用 #P{program} 标记(不区分 channel，类似 REMI)
- 支持 Type 0 MIDI (单 track 多 channel)
- 支持 Type 1 MIDI (多 track，包括单 track 多 channel)
"""

import mido
from collections import defaultdict
from pathlib import Path
import sys

# 量化参数
POSITIONS_PER_BAR = 16  # 每小节 16 个位置 (16 分音符)
VELOCITY_BINS = 32      # 力度量化为 32 级
MAX_DURATION_UNITS = 64 # 最大持续时间 (16分音符数)


def quantize_velocity(velocity: int) -> int:
    """量化 velocity 到 32 bins"""
    return min(VELOCITY_BINS - 1, velocity * VELOCITY_BINS // 128)


def dequantize_velocity(bin_idx: int) -> int:
    """反量化 velocity"""
    return min(127, (bin_idx * 128 + VELOCITY_BINS // 2) // VELOCITY_BINS)


def midi_to_txt(midi_file_path: str, output_file_path: str, quantize: bool = True):
    """
    MIDI -> TXT 转换

    Args:
        midi_file_path: 输入 MIDI 文件路径
        output_file_path: 输出 TXT 文件路径
        quantize: 是否使用 16 分音符量化 (默认 True)
    """
    try:
        mid = mido.MidiFile(midi_file_path)
    except Exception as e:
        raise Exception(f"无法读取 MIDI 文件: {e}")

    ticks_per_beat = mid.ticks_per_beat
    is_type0 = mid.type == 0

    tempo_changes = []
    time_signatures = []
    track_data = []

    # 收集音符数据
    if is_type0:
        channel_programs = defaultdict(int)
        active_notes = defaultdict(lambda: defaultdict(list))
        notes_by_channel_program = defaultdict(list)

        for track in mid.tracks:
            current_tick = 0
            for msg in track:
                current_tick += msg.time

                if msg.type == 'set_tempo':
                    bpm = mido.tempo2bpm(msg.tempo)
                    tempo_changes.append((current_tick, bpm))
                elif msg.type == 'time_signature':
                    time_signatures.append((current_tick, msg.numerator, msg.denominator))
                elif msg.type == 'program_change':
                    channel_programs[msg.channel] = msg.program
                elif msg.type == 'note_on' and msg.velocity > 0:
                    channel = msg.channel
                    active_notes[channel][msg.note].append((current_tick, msg.velocity))
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    channel = msg.channel
                    if active_notes[channel][msg.note]:
                        start_tick, velocity = active_notes[channel][msg.note].pop(0)
                        duration = current_tick - start_tick
                        program = channel_programs[channel]
                        notes_by_channel_program[(channel, program)].append(
                            (start_tick, msg.note, velocity, duration)
                        )

        for (channel, program), notes in sorted(notes_by_channel_program.items()):
            if notes:
                notes.sort(key=lambda x: x[0])
                track_data.append((channel, program, notes))
    else:
        for track_idx, track in enumerate(mid.tracks):
            current_tick = 0
            channel_programs = defaultdict(int)
            active_notes = defaultdict(lambda: defaultdict(list))
            notes_by_channel = defaultdict(list)

            for msg in track:
                current_tick += msg.time

                if msg.type == 'set_tempo':
                    bpm = mido.tempo2bpm(msg.tempo)
                    tempo_changes.append((current_tick, bpm))
                elif msg.type == 'time_signature':
                    time_signatures.append((current_tick, msg.numerator, msg.denominator))
                elif msg.type == 'program_change':
                    channel_programs[msg.channel] = msg.program
                elif msg.type == 'note_on' and msg.velocity > 0:
                    channel = msg.channel
                    active_notes[channel][msg.note].append((current_tick, msg.velocity))
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    channel = msg.channel
                    if active_notes[channel][msg.note]:
                        start_tick, velocity = active_notes[channel][msg.note].pop(0)
                        duration = current_tick - start_tick
                        notes_by_channel[channel].append((start_tick, msg.note, velocity, duration))

            for channel, notes in sorted(notes_by_channel.items()):
                if notes:
                    notes.sort(key=lambda x: x[0])
                    program = channel_programs[channel]
                    track_data.append((channel, program, notes))

    tempo_changes = sorted(set(tempo_changes), key=lambda x: x[0])
    time_signatures = sorted(set(time_signatures), key=lambda x: x[0])

    # 计算每小节的 ticks (默认 4/4 拍)
    if time_signatures:
        first_ts = time_signatures[0]
        beats_per_bar = first_ts[1]
        beat_unit = first_ts[2]
        ticks_per_bar = int(beats_per_bar * ticks_per_beat * 4 / beat_unit)
    else:
        ticks_per_bar = ticks_per_beat * 4

    # 每个 16 分音符的 ticks
    ticks_per_16th = ticks_per_bar // POSITIONS_PER_BAR

    # 写入文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        if quantize:
            # 量化模式头部: H 轨道数 ticks_per_beat (标记为量化模式)
            f.write(f"H {len(track_data)} {ticks_per_beat} Q\n")
        else:
            # 无损模式头部: H 轨道数 ticks_per_beat ticks_per_bar
            f.write(f"H {len(track_data)} {ticks_per_beat} {ticks_per_bar}\n")

        # Tempo
        if tempo_changes:
            for tick, bpm in tempo_changes:
                bar = tick // ticks_per_bar
                if quantize:
                    pos = round((tick % ticks_per_bar) * POSITIONS_PER_BAR / ticks_per_bar)
                    pos = min(pos, POSITIONS_PER_BAR - 1)
                    f.write(f"T B{bar} T{pos} {bpm:.0f}\n")
                else:
                    tick_in_bar = tick % ticks_per_bar
                    f.write(f"T B{bar} T{tick_in_bar} {bpm:.2f}\n")

        # Time Signatures
        if time_signatures:
            for tick, num, denom in time_signatures:
                bar = tick // ticks_per_bar
                if quantize:
                    pos = round((tick % ticks_per_bar) * POSITIONS_PER_BAR / ticks_per_bar)
                    pos = min(pos, POSITIONS_PER_BAR - 1)
                    f.write(f"TS B{bar} T{pos} {num}/{denom}\n")
                else:
                    tick_in_bar = tick % ticks_per_bar
                    f.write(f"TS B{bar} T{tick_in_bar} {num}/{denom}\n")

        f.write("\n")

        # 每个 (channel, program) 独立输出
        # Channel 9 是 MIDI drum channel，用 #D 标记
        # 其他 channel 只用 program，用 #P{program} 标记
        for channel, program, notes in track_data:
            if channel == 9:
                f.write(f"#D\n")  # Drum track (channel 9)
            else:
                f.write(f"#P{program}\n")

            last_bar = None
            last_pos = None  # 新增：追踪上一个位置
            last_velocity = None
            last_duration = None

            for tick, note, velocity, duration in notes:
                bar = tick // ticks_per_bar

                if quantize:
                    # 量化位置到 16 分音符
                    pos = round((tick % ticks_per_bar) * POSITIONS_PER_BAR / ticks_per_bar)
                    pos = min(pos, POSITIONS_PER_BAR - 1)

                    # 量化持续时间到 16 分音符数量
                    dur_units = max(1, round(duration / ticks_per_16th))
                    dur_units = min(dur_units, MAX_DURATION_UNITS)

                    # 量化力度
                    vel_bin = quantize_velocity(velocity)
                else:
                    pos = tick % ticks_per_bar
                    dur_units = duration
                    vel_bin = velocity

                tokens = []

                # Bar (只在变化时输出)
                if bar != last_bar:
                    tokens.append(f"B{bar}")
                    last_bar = bar
                    last_pos = None  # 新 bar 时重置 position 状态

                # Position (只在变化时输出，和弦音省略)
                if pos != last_pos:
                    tokens.append(f"T{pos}")
                    last_pos = pos

                # Pitch (始终输出)
                tokens.append(f"P{note}")

                # Duration (差分编码)
                if dur_units != last_duration:
                    tokens.append(f"L{dur_units}")
                    last_duration = dur_units

                # Velocity (差分编码)
                if vel_bin != last_velocity:
                    tokens.append(f"V{vel_bin}")
                    last_velocity = vel_bin

                f.write(" ".join(tokens) + "\n")

            f.write("\n")

    mode = "量化" if quantize else "无损"
    print(f"✅ {mode}转换完成: {midi_file_path} -> {output_file_path}")
    txt_size = Path(output_file_path).stat().st_size
    print(f"   TXT 大小: {txt_size:,} 字节")
    print(f"   轨道数: {len(track_data)}")
    if quantize:
        print(f"   位置分辨率: {POSITIONS_PER_BAR} 分音符/小节")
    else:
        print(f"   ticks_per_bar: {ticks_per_bar}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MIDI 到 TXT 转换器")
    parser.add_argument('midi_file', help='输入 MIDI 文件')
    parser.add_argument('output_txt', help='输出 TXT 文件')
    parser.add_argument('--lossless', action='store_true',
                       help='使用无损模式 (默认使用 16 分音符量化)')

    args = parser.parse_args()

    midi_to_txt(args.midi_file, args.output_txt, quantize=not args.lossless)


if __name__ == "__main__":
    main()
