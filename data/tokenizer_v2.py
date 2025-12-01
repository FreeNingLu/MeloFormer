#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HID Tokenizer V2 - 使用和弦替代 BAR

改进：
- BAR token 替换为和弦 token (Cmaj, Amin, G7 等)
- 保留原有的位置、音高、乐器编码
- 和弦既表示小节边界，又携带和声信息
"""

import pickle
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .chord_detector_music21 import Music21ChordDetector as ChordDetector, get_chord_vocab, NOTE_NAMES


@dataclass
class TokenInfo:
    """Token 元信息"""
    token_id: int
    token_str: str
    chord_idx: int = -1        # 所属和弦索引 (-1 表示非音符 token)
    position: int = -1         # 小节内位置 (0-15)
    instrument_id: int = -1    # 所属乐器 (-1 表示全局 token)
    is_sep: bool = False
    is_chord: bool = False     # 是否是和弦 token
    is_pitch: bool = False
    token_type: int = -1       # Token 类型: 0=T, 1=P, 2=D(L), 3=V, -1=全局/其他
    note_id: int = -1          # 音符 ID (同一音符的 T, P, L, V 有相同 ID)

    @property
    def bar_id(self) -> int:
        """向后兼容: bar_id 等同于 chord_idx"""
        return self.chord_idx


class HIDTokenizerV2:
    """
    HID Tokenizer V2 - 和弦版

    词汇表组成:
    - 特殊 tokens: PAD, BOS, EOS, MASK, SEP, UNK (6)
    - 和弦 tokens: N, Cmaj, Cmin, ..., Bsus4 (85)
    - 位置 tokens: T0-T15 (16)
    - 音高 tokens: P0-P127 (128)
    - 乐器 tokens: #D, #P0-#P127 (129)
    - 速度 tokens: BPM_0-BPM_31 (32)
    - 拍号 tokens: TS_* (19)
    - 持续时间 tokens: L1-L64 (64)
    - 力度 tokens: V0-V31 (32)

    总计: 6 + 85 + 16 + 128 + 129 + 32 + 19 + 64 + 32 = 511 tokens
    """

    POSITIONS_PER_BAR = 16
    TEMPO_BINS = 32
    MIN_BPM = 40
    MAX_BPM = 250

    def __init__(self):
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self.chord_detector = ChordDetector()
        self._build_tempo_table()
        self._build_vocab()

    def _build_tempo_table(self):
        """构建 tempo 量化表"""
        self._tempo_values = []
        for i in range(self.TEMPO_BINS):
            ratio = i / (self.TEMPO_BINS - 1)
            bpm = self.MIN_BPM * math.exp(ratio * math.log(self.MAX_BPM / self.MIN_BPM))
            self._tempo_values.append(round(bpm))
        self._tempo_values[0] = self.MIN_BPM
        self._tempo_values[-1] = self.MAX_BPM

    def quantize_tempo(self, bpm: float) -> int:
        bpm = max(self.MIN_BPM, min(self.MAX_BPM, bpm))
        min_diff = float('inf')
        best_bin = 0
        for i, val in enumerate(self._tempo_values):
            diff = abs(val - bpm)
            if diff < min_diff:
                min_diff = diff
                best_bin = i
        return best_bin

    def dequantize_tempo(self, bin_idx: int) -> int:
        return self._tempo_values[max(0, min(self.TEMPO_BINS - 1, bin_idx))]

    def _build_vocab(self):
        """构建词汇表"""
        tokens = []

        # 1. 特殊 tokens (6)
        tokens.extend(['PAD', 'BOS', 'EOS', 'MASK', 'SEP', 'UNK'])

        # 2. 和弦 tokens (85) - 替代原来的 BAR
        chord_vocab = get_chord_vocab()
        tokens.extend(chord_vocab)

        # 3. 位置 tokens: T0-T15 (16)
        for i in range(self.POSITIONS_PER_BAR):
            tokens.append(f'T{i}')

        # 4. 音高 tokens: P0-P127 (128)
        for i in range(128):
            tokens.append(f'P{i}')

        # 5. 乐器 tokens (129)
        tokens.append('#D')  # Drum
        for program in range(128):
            tokens.append(f'#P{program}')

        # 6. 速度 tokens: BPM_0-BPM_31 (32)
        for i in range(self.TEMPO_BINS):
            tokens.append(f'BPM_{i}')

        # 7. 拍号 tokens (19)
        time_sigs = [
            '1/4', '2/4', '3/4', '4/4', '5/4', '6/4', '7/4',
            '2/2', '3/2', '4/2',
            '3/8', '5/8', '6/8', '7/8', '9/8', '12/8',
            '2/8', '4/8', '8/8',
        ]
        for ts in time_sigs:
            tokens.append(f'TS_{ts}')

        # 8. Duration tokens: L1-L64 (64)
        for i in range(1, 65):
            tokens.append(f'L{i}')

        # 9. Velocity tokens: V0-V31 (32)
        for i in range(32):
            tokens.append(f'V{i}')

        # 构建映射
        for i, token in enumerate(tokens):
            self.token2id[token] = i
            self.id2token[i] = token

        # 记录和弦 token 范围
        self.chord_start_id = self.token2id['N']
        self.chord_end_id = self.token2id[chord_vocab[-1]]

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    @property
    def pad_id(self) -> int:
        return self.token2id['PAD']

    @property
    def bos_id(self) -> int:
        return self.token2id['BOS']

    @property
    def eos_id(self) -> int:
        return self.token2id['EOS']

    @property
    def sep_id(self) -> int:
        return self.token2id['SEP']

    def is_chord_token(self, token_id: int) -> bool:
        """判断是否是和弦 token"""
        return self.chord_start_id <= token_id <= self.chord_end_id

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.id2token.get(key, 'UNK')
        return self.token2id.get(key, self.token2id['UNK'])

    def encode_with_info(
        self,
        txt_content: str,
        add_special: bool = True,
    ) -> Tuple[List[int], List['TokenInfo']]:
        """
        从 TXT 格式编码为 token 序列

        Args:
            txt_content: TXT 格式的音乐内容
            add_special: 是否添加 BOS/EOS

        Returns:
            (token_ids, token_infos)
        """
        import re

        token_ids = []
        token_infos = []

        if add_special:
            token_ids.append(self.bos_id)
            token_infos.append(TokenInfo(self.bos_id, 'BOS'))

        current_bar = 0
        current_instrument = -1
        current_position = 0
        current_note_id = 0  # 音符 ID 计数器

        # 解析和弦映射
        chord_map = {}  # {(bar, pos): chord_name}

        lines = txt_content.strip().split('\n')

        # 先扫描 CHORDS 行
        for line in lines:
            line_strip = line.strip()
            if line_strip.startswith('CHORDS '):
                parts = line_strip[7:].split()
                for part in parts:
                    if ':' in part:
                        bar_pos, chord = part.split(':', 1)
                        if '.' in bar_pos:
                            bar_str, pos_str = bar_pos.split('.', 1)
                            bar = int(bar_str)
                            pos = int(pos_str)
                        else:
                            bar = int(bar_pos)
                            pos = 0
                        chord_map[(bar, pos)] = chord
                break

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Header: H 4 384 Q
            if line.startswith('H '):
                continue

            # Tempo: T B0 T0 120
            if line.startswith('T ') and 'B' in line:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        bpm = int(parts[-1])
                        bpm_bin = self.quantize_tempo(bpm)
                        bpm_token = f'BPM_{bpm_bin}'
                        if bpm_token in self.token2id:
                            token_ids.append(self.token2id[bpm_token])
                            token_infos.append(TokenInfo(self.token2id[bpm_token], bpm_token))
                    except ValueError:
                        pass
                continue

            # Time signature: TS B0 T0 4/4
            if line.startswith('TS '):
                parts = line.split()
                if len(parts) >= 4:
                    ts = parts[-1]
                    ts_token = f'TS_{ts}'
                    if ts_token in self.token2id:
                        token_ids.append(self.token2id[ts_token])
                        token_infos.append(TokenInfo(self.token2id[ts_token], ts_token))
                continue

            # Instrument: #P0, #P25, #D
            if line.startswith('#'):
                inst_token = line.strip()
                if inst_token in self.token2id:
                    if inst_token == '#D':
                        current_instrument = 128
                    else:
                        try:
                            current_instrument = int(inst_token[2:])
                        except ValueError:
                            current_instrument = 0

                    token_ids.append(self.token2id[inst_token])
                    token_infos.append(TokenInfo(
                        self.token2id[inst_token], inst_token,
                        instrument_id=current_instrument
                    ))

                    # SEP after instrument
                    token_ids.append(self.sep_id)
                    token_infos.append(TokenInfo(
                        self.sep_id, 'SEP',
                        instrument_id=current_instrument,
                        is_sep=True
                    ))
                continue

            # Skip CHORDS line (already parsed)
            if line.startswith('CHORDS '):
                continue

            # Note line: B0 T0 P60 L4 V80 or T0 P60
            tokens_in_line = line.split()
            for token in tokens_in_line:
                # Bar: B0, B1, ...
                if token.startswith('B') and token[1:].isdigit():
                    current_bar = int(token[1:])
                    # 查找和弦映射，优先用 pos=0 的和弦
                    chord_token = chord_map.get((current_bar, 0), 'N')
                    # 确保和弦 token 在词表中
                    if chord_token not in self.token2id:
                        chord_token = 'N'
                    if chord_token in self.token2id:
                        token_ids.append(self.token2id[chord_token])
                        token_infos.append(TokenInfo(
                            self.token2id[chord_token], chord_token,
                            chord_idx=current_bar,
                            instrument_id=current_instrument,
                            is_chord=True
                        ))
                    continue

                # Position: T0-T15
                if token.startswith('T') and token[1:].isdigit():
                    pos = int(token[1:])
                    current_position = min(pos, self.POSITIONS_PER_BAR - 1)
                    pos_token = f'T{current_position}'
                    # 每遇到新的 T token，note_id 递增（新音符开始）
                    current_note_id += 1
                    if pos_token in self.token2id:
                        token_ids.append(self.token2id[pos_token])
                        token_infos.append(TokenInfo(
                            self.token2id[pos_token], pos_token,
                            chord_idx=current_bar,
                            position=current_position,
                            instrument_id=current_instrument,
                            token_type=0,  # T = 0
                            note_id=current_note_id
                        ))
                    continue

                # Pitch: P0-P127
                if token.startswith('P') and token[1:].isdigit():
                    pitch = int(token[1:])
                    pitch = max(0, min(127, pitch))
                    pitch_token = f'P{pitch}'
                    if pitch_token in self.token2id:
                        token_ids.append(self.token2id[pitch_token])
                        token_infos.append(TokenInfo(
                            self.token2id[pitch_token], pitch_token,
                            chord_idx=current_bar,
                            position=current_position,
                            instrument_id=current_instrument,
                            is_pitch=True,
                            token_type=1,  # P = 1
                            note_id=current_note_id
                        ))
                    continue

                # Duration: L1-L64
                if token.startswith('L') and token[1:].isdigit():
                    dur = int(token[1:])
                    dur = max(1, min(64, dur))
                    dur_token = f'L{dur}'
                    if dur_token in self.token2id:
                        token_ids.append(self.token2id[dur_token])
                        token_infos.append(TokenInfo(
                            self.token2id[dur_token], dur_token,
                            chord_idx=current_bar,
                            position=current_position,
                            instrument_id=current_instrument,
                            token_type=2,  # D(L) = 2
                            note_id=current_note_id
                        ))
                    continue

                # Velocity: V0-V31
                if token.startswith('V') and token[1:].isdigit():
                    vel = int(token[1:])
                    vel = max(0, min(31, vel))
                    vel_token = f'V{vel}'
                    if vel_token in self.token2id:
                        token_ids.append(self.token2id[vel_token])
                        token_infos.append(TokenInfo(
                            self.token2id[vel_token], vel_token,
                            chord_idx=current_bar,
                            position=current_position,
                            instrument_id=current_instrument,
                            token_type=3,  # V = 3
                            note_id=current_note_id
                        ))
                    continue

        if add_special:
            token_ids.append(self.eos_id)
            token_infos.append(TokenInfo(self.eos_id, 'EOS'))

        return token_ids, token_infos

    def encode_midi_with_chords(
        self,
        notes_by_instrument: Dict[int, List[Dict]],
        ticks_per_beat: int = 480,
        beats_per_chord: int = 2,
        tempo: float = 120.0,
        time_signature: str = '4/4',
    ) -> Tuple[List[int], List[TokenInfo]]:
        """
        将 MIDI 音符编码为带和弦的 token 序列

        Args:
            notes_by_instrument: {instrument_id: [{'pitch', 'start', 'duration', 'velocity'}, ...]}
            ticks_per_beat: 每拍 tick 数
            beats_per_chord: 每个和弦跨越的拍数
            tempo: BPM
            time_signature: 拍号

        Returns:
            (token_ids, token_infos)
        """
        token_ids = []
        token_infos = []

        # BOS
        token_ids.append(self.bos_id)
        token_infos.append(TokenInfo(self.bos_id, 'BOS'))

        # Tempo
        bpm_bin = self.quantize_tempo(tempo)
        bpm_token = f'BPM_{bpm_bin}'
        token_ids.append(self.token2id[bpm_token])
        token_infos.append(TokenInfo(self.token2id[bpm_token], bpm_token))

        # Time Signature
        ts_token = f'TS_{time_signature}'
        if ts_token in self.token2id:
            token_ids.append(self.token2id[ts_token])
            token_infos.append(TokenInfo(self.token2id[ts_token], ts_token))

        # 合并所有乐器的音符用于和弦检测
        all_notes = []
        for inst_id, notes in notes_by_instrument.items():
            if inst_id != 128:  # 排除鼓
                for note in notes:
                    all_notes.append(note)

        # 检测和弦序列
        chord_sequence = self.chord_detector.detect_chord_sequence(
            all_notes, ticks_per_beat, beats_per_chord
        )

        window_ticks = ticks_per_beat * beats_per_chord

        # 按乐器编码
        for inst_id, notes in sorted(notes_by_instrument.items()):
            if not notes:
                continue

            # 乐器标记
            if inst_id == 128:
                inst_token = '#D'
            else:
                inst_token = f'#P{inst_id}'

            token_ids.append(self.token2id[inst_token])
            token_infos.append(TokenInfo(
                self.token2id[inst_token], inst_token,
                instrument_id=inst_id
            ))

            # SEP
            token_ids.append(self.sep_id)
            token_infos.append(TokenInfo(
                self.sep_id, 'SEP',
                instrument_id=inst_id,
                is_sep=True
            ))

            # 按和弦窗口分组音符
            notes_sorted = sorted(notes, key=lambda x: x['start'])
            chord_idx = 0
            last_chord_idx = -1  # 跟踪上一个输出的和弦索引 (每个乐器独立)

            for note in notes_sorted:
                note_start = note['start']

                # 找到当前音符所属的和弦窗口
                while (chord_idx + 1 < len(chord_sequence) and
                       chord_sequence[chord_idx + 1][0] <= note_start):
                    chord_idx += 1

                chord_start_tick = chord_sequence[chord_idx][0] if chord_idx < len(chord_sequence) else 0
                chord_name = chord_sequence[chord_idx][1] if chord_idx < len(chord_sequence) else 'N'

                # 如果进入新和弦，添加和弦 token (每个和弦只输出一次)
                if chord_idx != last_chord_idx:
                    last_chord_idx = chord_idx
                    chord_token = chord_name

                    token_ids.append(self.token2id[chord_token])
                    token_infos.append(TokenInfo(
                        self.token2id[chord_token], chord_token,
                        chord_idx=chord_idx,
                        instrument_id=inst_id,
                        is_chord=True
                    ))

                # 计算小节内位置 (0-15)
                pos_in_window = note_start - chord_start_tick
                pos_16th = int(pos_in_window * self.POSITIONS_PER_BAR / window_ticks)
                pos_16th = max(0, min(self.POSITIONS_PER_BAR - 1, pos_16th))

                # Position token
                pos_token = f'T{pos_16th}'
                token_ids.append(self.token2id[pos_token])
                token_infos.append(TokenInfo(
                    self.token2id[pos_token], pos_token,
                    chord_idx=chord_idx,
                    position=pos_16th,
                    instrument_id=inst_id
                ))

                # Pitch token
                pitch = note['pitch']
                pitch_token = f'P{pitch}'
                token_ids.append(self.token2id[pitch_token])
                token_infos.append(TokenInfo(
                    self.token2id[pitch_token], pitch_token,
                    chord_idx=chord_idx,
                    position=pos_16th,
                    instrument_id=inst_id,
                    is_pitch=True
                ))

        # EOS
        token_ids.append(self.eos_id)
        token_infos.append(TokenInfo(self.eos_id, 'EOS'))

        return token_ids, token_infos

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """解码为文本格式"""
        lines = []
        current_track = None
        current_bar = -1  # bar 计数器（每个乐器独立）
        note_line = []

        header = "H 1 480 Q"
        lines.append(header)

        for token_id in token_ids:
            token = self.id2token.get(token_id, 'UNK')

            if skip_special and token in ['PAD', 'BOS', 'EOS', 'MASK', 'UNK']:
                continue

            if token == 'SEP':
                if note_line:
                    lines.append(' '.join(note_line))
                    note_line = []
                current_bar = -1  # 重置 bar 计数器
                continue

            if token.startswith('BPM_'):
                bpm_bin = int(token[4:])
                actual_bpm = self.dequantize_tempo(bpm_bin)
                lines.insert(1, f'T B0 T0 {actual_bpm}')
                continue

            if token.startswith('TS_'):
                ts = token[3:]
                lines.insert(1, f'TS B0 T0 {ts}')
                continue

            if token.startswith('#P') or token == '#D':
                if note_line:
                    lines.append(' '.join(note_line))
                    note_line = []
                if current_track is not None:
                    lines.append("")
                current_track = token
                lines.append(token)
                current_bar = -1  # 重置 bar 计数器
                continue

            # 和弦 token → B{bar} 格式（与 txt_to_midi 兼容）
            if self.is_chord_token(token_id):
                if note_line:
                    lines.append(' '.join(note_line))
                current_bar += 1
                note_line = [f'B{current_bar}']  # 输出 B{bar} 格式
                continue

            if token.startswith('T') and token[1:].isdigit():
                note_line.append(token)
                continue

            if token.startswith('P') and token[1:].isdigit():
                note_line.append(token)
                continue

            # Duration: L1-L64
            if token.startswith('L') and token[1:].isdigit():
                note_line.append(token)
                continue

            # Velocity: V0-V31
            if token.startswith('V') and token[1:].isdigit():
                note_line.append(token)
                continue

        if note_line:
            lines.append(' '.join(note_line))

        return '\n'.join(lines)

    def save(self, path: str):
        """保存 tokenizer"""
        with open(path, 'wb') as f:
            pickle.dump({
                'token2id': self.token2id,
                'id2token': self.id2token,
                'chord_start_id': self.chord_start_id,
                'chord_end_id': self.chord_end_id,
            }, f)

    @classmethod
    def load(cls, path: str) -> 'HIDTokenizerV2':
        """加载 tokenizer"""
        tokenizer = cls.__new__(cls)
        with open(path, 'rb') as f:
            data = pickle.load(f)
        tokenizer.token2id = data['token2id']
        tokenizer.id2token = data['id2token']
        tokenizer.chord_start_id = data['chord_start_id']
        tokenizer.chord_end_id = data['chord_end_id']
        tokenizer.chord_detector = ChordDetector()
        tokenizer._build_tempo_table()
        return tokenizer


if __name__ == '__main__':
    tokenizer = HIDTokenizerV2()

    print(f"HID Tokenizer V2 (和弦版)")
    print(f"=" * 50)
    print(f"词汇表大小: {tokenizer.vocab_size}")
    print(f"")
    print(f"词汇表组成:")
    print(f"  特殊 tokens: 6 (PAD, BOS, EOS, MASK, SEP, UNK)")
    print(f"  和弦 tokens: 85 (N, Cmaj, Cmin, ..., Bsus4)")
    print(f"  位置 tokens: 16 (T0-T15)")
    print(f"  音高 tokens: 128 (P0-P127)")
    print(f"  乐器 tokens: 129 (#D, #P0-#P127)")
    print(f"  速度 tokens: 32 (BPM_0-BPM_31)")
    print(f"  拍号 tokens: 19")
    print(f"  持续时间 tokens: 64 (L1-L64)")
    print(f"  力度 tokens: 32 (V0-V31)")
    print(f"")
    print(f"和弦 token 范围: {tokenizer.chord_start_id} - {tokenizer.chord_end_id}")

    # 测试编码
    print(f"\n测试编码:")
    test_notes = {
        0: [  # Piano
            {'pitch': 60, 'start': 0, 'duration': 480, 'velocity': 80},
            {'pitch': 64, 'start': 0, 'duration': 480, 'velocity': 80},
            {'pitch': 67, 'start': 0, 'duration': 480, 'velocity': 80},
            {'pitch': 62, 'start': 960, 'duration': 480, 'velocity': 80},
            {'pitch': 65, 'start': 960, 'duration': 480, 'velocity': 80},
            {'pitch': 69, 'start': 960, 'duration': 480, 'velocity': 80},
        ],
    }

    token_ids, token_infos = tokenizer.encode_midi_with_chords(test_notes, beats_per_chord=2)
    print(f"Token 数量: {len(token_ids)}")
    print(f"Tokens: {[tokenizer[tid] for tid in token_ids]}")
