#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HID-MuseFormer 完整流程测试

测试流程：
1. MIDI → 和弦识别
2. MIDI → HID Token 化 (和弦版)
3. Token → FC-Attention 掩码生成
4. 验证 FC-Attention 正确捕捉和弦/小节位置

HID 编码特点：
- 按乐器分组 (不同于 REMI 的时间交错)
- 和弦 token 替代 BAR，标记小节边界
"""

import torch
import mido
from pathlib import Path
from typing import List, Dict, Tuple

# 导入模块
from .data.chord_detector_music21 import Music21ChordDetector
from .data.tokenizer_v2 import HIDTokenizerV2, TokenInfo
from .model.attention import FCAttentionMask


def parse_midi(midi_path: str) -> Tuple[Dict[int, List[Dict]], int, float, str]:
    """
    解析 MIDI 文件

    Returns:
        notes_by_instrument: {inst_id: [{'pitch', 'start', 'duration', 'velocity'}, ...]}
        ticks_per_beat: int
        tempo: float (BPM)
        time_signature: str
    """
    mid = mido.MidiFile(midi_path)
    ticks_per_beat = mid.ticks_per_beat

    # 默认值
    tempo = 120.0
    time_signature = '4/4'

    notes_by_instrument: Dict[int, List[Dict]] = {}

    for track_idx, track in enumerate(mid.tracks):
        current_time = 0
        current_program = 0
        is_drum = False
        active_notes = {}  # {(channel, pitch): start_time}

        for msg in track:
            current_time += msg.time

            if msg.type == 'set_tempo':
                tempo = 60_000_000 / msg.tempo
            elif msg.type == 'time_signature':
                time_signature = f'{msg.numerator}/{msg.denominator}'
            elif msg.type == 'program_change':
                current_program = msg.program
            elif msg.type == 'note_on' and msg.velocity > 0:
                channel = msg.channel
                is_drum = (channel == 9)
                inst_id = 128 if is_drum else current_program

                key = (channel, msg.note)
                active_notes[key] = {
                    'pitch': msg.note,
                    'start': current_time,
                    'velocity': msg.velocity,
                    'inst_id': inst_id,
                }
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in active_notes:
                    note_info = active_notes.pop(key)
                    duration = current_time - note_info['start']
                    inst_id = note_info['inst_id']

                    if inst_id not in notes_by_instrument:
                        notes_by_instrument[inst_id] = []

                    notes_by_instrument[inst_id].append({
                        'pitch': note_info['pitch'],
                        'start': note_info['start'],
                        'duration': max(1, duration),
                        'velocity': note_info['velocity'],
                    })

    return notes_by_instrument, ticks_per_beat, tempo, time_signature


def test_chord_detection(notes_by_instrument: Dict, ticks_per_beat: int):
    """测试和弦识别"""
    print("\n" + "=" * 60)
    print("1. 和弦识别测试")
    print("=" * 60)

    detector = Music21ChordDetector()

    # 合并所有非鼓音符
    all_notes = []
    for inst_id, notes in notes_by_instrument.items():
        if inst_id != 128:  # 排除鼓
            all_notes.extend(notes)

    print(f"总音符数: {len(all_notes)}")

    # 检测和弦序列
    chord_sequence = detector.detect_chord_sequence(
        all_notes,
        ticks_per_beat=ticks_per_beat,
        beats_per_chord=2
    )

    print(f"检测到 {len(chord_sequence)} 个和弦:")
    for i, (tick, chord) in enumerate(chord_sequence[:10]):
        beat = tick / ticks_per_beat
        print(f"  [{i}] Beat {beat:.1f}: {chord}")

    if len(chord_sequence) > 10:
        print(f"  ... (共 {len(chord_sequence)} 个)")

    return chord_sequence


def test_tokenization(notes_by_instrument: Dict, ticks_per_beat: int, tempo: float, time_sig: str):
    """测试 HID Token 化"""
    print("\n" + "=" * 60)
    print("2. HID Token 化测试 (和弦版)")
    print("=" * 60)

    tokenizer = HIDTokenizerV2()

    print(f"词汇表大小: {tokenizer.vocab_size}")
    print(f"和弦 token 范围: {tokenizer.chord_start_id} - {tokenizer.chord_end_id}")

    # 编码
    token_ids, token_infos = tokenizer.encode_midi_with_chords(
        notes_by_instrument,
        ticks_per_beat=ticks_per_beat,
        beats_per_chord=2,
        tempo=tempo,
        time_signature=time_sig,
    )

    print(f"\n生成 {len(token_ids)} 个 tokens")

    # 显示 token 序列结构
    print("\nToken 序列结构:")
    current_inst = None
    current_chord = None
    line_tokens = []

    for i, (tid, info) in enumerate(zip(token_ids[:100], token_infos[:100])):
        token = tokenizer[tid]

        # 乐器变化
        if token.startswith('#P') or token == '#D':
            if line_tokens:
                print(f"    {' '.join(line_tokens)}")
                line_tokens = []
            print(f"\n  [{token}] 乐器 {info.instrument_id}")
            current_inst = token
            continue

        # SEP
        if token == 'SEP':
            if line_tokens:
                print(f"    {' '.join(line_tokens)}")
                line_tokens = []
            continue

        # 和弦 (替代 BAR)
        if info.is_chord:
            if line_tokens:
                print(f"    {' '.join(line_tokens)}")
                line_tokens = []
            print(f"    [{token}] <- 和弦/小节 {info.chord_idx}")
            current_chord = token
            continue

        # 特殊 token
        if token in ['BOS', 'EOS', 'PAD']:
            print(f"  <{token}>")
            continue

        if token.startswith('BPM_') or token.startswith('TS_'):
            print(f"  <{token}>")
            continue

        # 音符 token
        line_tokens.append(token)
        if len(line_tokens) >= 10:
            print(f"    {' '.join(line_tokens)}")
            line_tokens = []

    if line_tokens:
        print(f"    {' '.join(line_tokens)}")

    if len(token_ids) > 100:
        print(f"\n  ... (共 {len(token_ids)} tokens)")

    return token_ids, token_infos, tokenizer


def test_fc_attention_mask(token_ids: List[int], token_infos: List[TokenInfo], tokenizer):
    """测试 FC-Attention 掩码生成"""
    print("\n" + "=" * 60)
    print("3. FC-Attention 掩码测试")
    print("=" * 60)

    # 构建 chord_ids 和 instrument_ids
    seq_len = len(token_ids)
    chord_ids = torch.zeros(1, seq_len, dtype=torch.long)
    instrument_ids = torch.zeros(1, seq_len, dtype=torch.long)

    for i, info in enumerate(token_infos):
        chord_ids[0, i] = info.chord_idx
        instrument_ids[0, i] = info.instrument_id if info.instrument_id >= 0 else 129

    print(f"序列长度: {seq_len}")
    print(f"和弦索引范围: {chord_ids.min().item()} - {chord_ids.max().item()}")
    print(f"乐器 ID 范围: {instrument_ids.min().item()} - {instrument_ids.max().item()}")

    # 创建掩码 (使用论文默认配置: -1, -2, -4, -8, -12, -16, -24, -32)
    mask_generator = FCAttentionMask()

    mask = mask_generator.create_mask(
        chord_ids, instrument_ids, seq_len, torch.device('cpu')
    )

    print(f"掩码形状: {mask.shape}")

    # 分析掩码
    # 找几个关键位置
    print("\n掩码分析:")

    # 找第一个和弦 token 的位置
    chord_positions = [i for i, info in enumerate(token_infos) if info.is_chord]
    pitch_positions = [i for i, info in enumerate(token_infos) if info.is_pitch]

    print(f"  和弦 token 位置: {chord_positions[:5]}...")
    print(f"  音高 token 位置: {pitch_positions[:5]}...")

    if len(chord_positions) >= 2 and len(pitch_positions) >= 1:
        # 检查：同和弦内的音符是否互相可见
        chord_0_start = chord_positions[0]
        chord_1_start = chord_positions[1] if len(chord_positions) > 1 else seq_len

        # 找和弦 0 内的音符
        notes_in_chord_0 = [i for i in pitch_positions if chord_0_start < i < chord_1_start]

        if len(notes_in_chord_0) >= 2:
            n1, n2 = notes_in_chord_0[0], notes_in_chord_0[1]
            can_see = mask[0, n2, n1].item()
            print(f"\n  同和弦内音符可见性: 位置 {n2} 能看到 {n1}? {can_see}")

        # 检查：不同和弦的音符 (非 fine-grained 范围)
        if len(chord_positions) >= 2:
            notes_in_chord_1 = [i for i in pitch_positions if i > chord_positions[1]]
            if notes_in_chord_1:
                n_later = notes_in_chord_1[0]
                n_earlier = notes_in_chord_0[0] if notes_in_chord_0 else chord_0_start
                can_see = mask[0, n_later, n_earlier].item()
                print(f"  不同和弦音符可见性: 位置 {n_later} 能看到 {n_earlier}? {can_see}")

        # 检查：Fine-grained 注意力 (和弦 2 能看和弦 1)
        if len(chord_positions) >= 3:
            # 找和弦 2 内的音符
            chord_2_start = chord_positions[2]
            chord_3_start = chord_positions[3] if len(chord_positions) > 3 else seq_len
            notes_in_chord_2 = [i for i in pitch_positions if chord_2_start < i < chord_3_start]

            # 找和弦 1 内的音符
            chord_1_start = chord_positions[1]
            notes_in_chord_1_range = [i for i in pitch_positions if chord_1_start < i < chord_2_start]

            if notes_in_chord_2 and notes_in_chord_1_range:
                n_chord2 = notes_in_chord_2[0]
                n_chord1 = notes_in_chord_1_range[0]
                can_see = mask[0, n_chord2, n_chord1].item()
                chord2_idx = token_infos[n_chord2].chord_idx
                chord1_idx = token_infos[n_chord1].chord_idx
                inst2 = token_infos[n_chord2].instrument_id
                inst1 = token_infos[n_chord1].instrument_id
                # 检查 chord_ids tensor 中的实际值
                actual_chord2 = chord_ids[0, n_chord2].item()
                actual_chord1 = chord_ids[0, n_chord1].item()
                print(f"  Fine-grained (和弦{chord2_idx}→和弦{chord1_idx}, 乐器{inst2}→{inst1}): 位置 {n_chord2} 能看到 {n_chord1}? {can_see}")
                print(f"    chord_ids 实际值: query={actual_chord2}, key={actual_chord1}, diff={actual_chord2 - actual_chord1}")
                print(f"    instrument_ids: query={instrument_ids[0, n_chord2].item()}, key={instrument_ids[0, n_chord1].item()}")
                print(f"    因果性检查: query位置({n_chord2}) > key位置({n_chord1})? {n_chord2 > n_chord1}")

        # 检查更多 Fine-grained 距离 (和弦 8 能看和弦 4, 即 -4 offset)
        if len(chord_positions) >= 9:
            print("\n  更多 Fine-grained 验证:")

            # 辅助函数：找指定和弦内的音符
            def find_notes_in_chord(target_chord_idx):
                return [i for i, info in enumerate(token_infos)
                        if info.is_pitch and info.chord_idx == target_chord_idx]

            # 测试各个 offset
            test_cases = [
                (8, 7, -1),   # 和弦8 看 和弦7 (offset=-1)
                (8, 6, -2),   # 和弦8 看 和弦6 (offset=-2)
                (8, 4, -4),   # 和弦8 看 和弦4 (offset=-4)
                (8, 0, -8),   # 和弦8 看 和弦0 (offset=-8)
                (8, 5, -3),   # 和弦8 看 和弦5 (offset=-3, 不在 fine-grained 中)
            ]

            for query_chord, key_chord, expected_offset in test_cases:
                notes_query = find_notes_in_chord(query_chord)
                notes_key = find_notes_in_chord(key_chord)

                if notes_query and notes_key:
                    # 确保是同乐器
                    query_inst = token_infos[notes_query[0]].instrument_id
                    same_inst_keys = [n for n in notes_key if token_infos[n].instrument_id == query_inst]

                    if same_inst_keys:
                        n_q = notes_query[0]
                        n_k = same_inst_keys[0]
                        can_see = mask[0, n_q, n_k].item()
                        should_see = expected_offset in (-1, -2, -4, -8, -12, -16, -24, -32)
                        status = "✓" if can_see == should_see else "✗"
                        print(f"    {status} 和弦{query_chord}→和弦{key_chord} (offset={expected_offset}): {can_see} (预期: {should_see})")

    # 检查跨乐器注意力（同一小节内不同乐器应该互相可见）
    print("\n跨乐器注意力检查（同小节应可见）:")
    inst_groups = {}
    for i, info in enumerate(token_infos):
        inst = info.instrument_id
        if inst >= 0:
            if inst not in inst_groups:
                inst_groups[inst] = []
            inst_groups[inst].append(i)

    print(f"  发现 {len(inst_groups)} 个乐器组")
    for inst, positions in list(inst_groups.items())[:3]:
        print(f"    乐器 {inst}: {len(positions)} 个 token")

    # 验证：同一小节内不同乐器的 token 应该互相可见
    if len(inst_groups) >= 2:
        instruments = list(inst_groups.keys())
        inst1, inst2 = instruments[0], instruments[1]

        # 找同一小节的不同乐器 token
        for chord_idx in range(min(5, chord_ids.max().item() + 1)):
            tokens_inst1 = [i for i in inst_groups[inst1]
                           if token_infos[i].chord_idx == chord_idx and token_infos[i].is_pitch]
            tokens_inst2 = [i for i in inst_groups[inst2]
                           if token_infos[i].chord_idx == chord_idx and token_infos[i].is_pitch]

            if tokens_inst1 and tokens_inst2:
                t1, t2 = tokens_inst1[0], tokens_inst2[0]
                # 确保 t2 > t1 (因果性)
                if t2 > t1:
                    can_see = mask[0, t2, t1].item()
                    status = "✓" if can_see else "✗"
                    print(f"  {status} 同小节{chord_idx}跨乐器: 乐器{inst2}位置{t2} 能看 乐器{inst1}位置{t1}? {can_see}")
                elif t1 > t2:
                    can_see = mask[0, t1, t2].item()
                    status = "✓" if can_see else "✗"
                    print(f"  {status} 同小节{chord_idx}跨乐器: 乐器{inst1}位置{t1} 能看 乐器{inst2}位置{t2}? {can_see}")
                break

    return mask, chord_ids, instrument_ids


def test_model_forward(token_ids: List[int], chord_ids: torch.Tensor, instrument_ids: torch.Tensor):
    """测试模型前向传播"""
    print("\n" + "=" * 60)
    print("4. 模型前向传播测试")
    print("=" * 60)

    from .model.hid_museformer import create_model

    # 创建小模型测试
    model = create_model(vocab_size=415, model_size='small')

    params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {params / 1e6:.2f}M")

    # 准备输入 (限制长度)
    max_len = min(256, len(token_ids))
    input_ids = torch.tensor([token_ids[:max_len]])
    chord_ids_input = chord_ids[:, :max_len]
    inst_ids_input = instrument_ids[:, :max_len]

    print(f"输入形状: {input_ids.shape}")

    # 前向传播
    model.eval()
    with torch.no_grad():
        logits = model(
            input_ids,
            chord_ids=chord_ids_input,
            instrument_ids=inst_ids_input,
        )

    print(f"输出形状: {logits.shape}")
    print(f"输出范围: [{logits.min():.2f}, {logits.max():.2f}]")

    # 检查预测
    predictions = logits.argmax(dim=-1)
    print(f"预测 token 示例: {predictions[0, :10].tolist()}")

    return model, logits


def visualize_attention_pattern(mask: torch.Tensor, token_infos: List[TokenInfo], tokenizer, max_show: int = 50):
    """可视化注意力模式"""
    print("\n" + "=" * 60)
    print("5. 注意力模式可视化 (前 {} tokens)".format(max_show))
    print("=" * 60)

    n = min(max_show, mask.shape[1])

    # 创建简化的 token 标签
    labels = []
    for i, info in enumerate(token_infos[:n]):
        token = tokenizer[info.token_id]
        if info.is_chord:
            labels.append(f"[{token[:4]}]")
        elif info.is_pitch:
            labels.append(token)
        elif token == 'SEP':
            labels.append('|')
        elif token.startswith('#'):
            labels.append(token[:3])
        else:
            labels.append(token[:3])

    # 打印掩码矩阵 (简化版)
    print("\n注意力掩码 (1=可见, 0=不可见):")
    print("    " + " ".join(f"{l:>5}" for l in labels[:20]))

    for i in range(min(20, n)):
        row = mask[0, i, :20]
        row_str = " ".join(f"{'█' if v else '·':>5}" for v in row.tolist())
        print(f"{labels[i]:>3} {row_str}")


def main():
    print("=" * 60)
    print("HID-MuseFormer 完整流程测试")
    print("=" * 60)

    # 查找测试 MIDI 文件
    midi_dirs = [
        Path("/Users/freeninglu/Desktop/MIDI"),
        Path("/Users/freeninglu/Desktop/MuseFormer"),
    ]

    midi_file = None
    for d in midi_dirs:
        if d.exists():
            midis = list(d.glob("**/*.mid"))[:1] + list(d.glob("**/*.midi"))[:1]
            if midis:
                midi_file = midis[0]
                break

    if midi_file is None:
        print("未找到 MIDI 文件，使用合成数据测试")
        # 创建合成数据
        notes_by_instrument = {
            0: [  # Piano
                {'pitch': 60, 'start': 0, 'duration': 480, 'velocity': 80},
                {'pitch': 64, 'start': 0, 'duration': 480, 'velocity': 80},
                {'pitch': 67, 'start': 0, 'duration': 480, 'velocity': 80},
                {'pitch': 62, 'start': 480, 'duration': 480, 'velocity': 80},
                {'pitch': 65, 'start': 480, 'duration': 480, 'velocity': 80},
                {'pitch': 69, 'start': 480, 'duration': 480, 'velocity': 80},
                {'pitch': 64, 'start': 960, 'duration': 480, 'velocity': 80},
                {'pitch': 67, 'start': 960, 'duration': 480, 'velocity': 80},
                {'pitch': 72, 'start': 960, 'duration': 480, 'velocity': 80},
            ],
            33: [  # Bass
                {'pitch': 36, 'start': 0, 'duration': 960, 'velocity': 100},
                {'pitch': 38, 'start': 960, 'duration': 960, 'velocity': 100},
            ],
        }
        ticks_per_beat = 480
        tempo = 120.0
        time_sig = '4/4'
    else:
        print(f"使用 MIDI 文件: {midi_file}")
        notes_by_instrument, ticks_per_beat, tempo, time_sig = parse_midi(str(midi_file))

    print(f"\n基本信息:")
    print(f"  Ticks/Beat: {ticks_per_beat}")
    print(f"  Tempo: {tempo:.1f} BPM")
    print(f"  Time Signature: {time_sig}")
    print(f"  乐器数量: {len(notes_by_instrument)}")
    for inst_id, notes in notes_by_instrument.items():
        inst_name = "Drums" if inst_id == 128 else f"Program {inst_id}"
        print(f"    {inst_name}: {len(notes)} 音符")

    # 1. 和弦识别
    chord_sequence = test_chord_detection(notes_by_instrument, ticks_per_beat)

    # 2. Token 化
    token_ids, token_infos, tokenizer = test_tokenization(
        notes_by_instrument, ticks_per_beat, tempo, time_sig
    )

    # 3. FC-Attention 掩码
    mask, chord_ids, instrument_ids = test_fc_attention_mask(
        token_ids, token_infos, tokenizer
    )

    # 4. 模型前向传播
    model, logits = test_model_forward(token_ids, chord_ids, instrument_ids)

    # 5. 可视化
    visualize_attention_pattern(mask, token_infos, tokenizer)

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
