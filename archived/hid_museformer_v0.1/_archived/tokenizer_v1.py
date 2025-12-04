#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HID Tokenizer - 适配 MuseFormer 训练

基于 tokenizer_v2_minimal，添加训练所需的辅助功能
"""

import pickle
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TokenInfo:
    """Token 的元信息，用于构建注意力掩码"""
    token_id: int
    token_str: str
    bar_id: int = -1          # 所属小节 (-1 表示非音符 token)
    position: int = -1         # 小节内位置 (0-15)
    instrument_id: int = -1    # 所属乐器 (-1 表示全局 token)
    is_sep: bool = False       # 是否是 SEP token
    is_bar: bool = False       # 是否是 BAR token
    is_pitch: bool = False     # 是否是音高 token


# 乐器默认参数表 (用于解码)
INSTRUMENT_DEFAULTS = {
    range(0, 8): {'velocity': 20, 'duration': 4},      # Piano
    range(8, 16): {'velocity': 22, 'duration': 2},     # Chromatic Percussion
    range(16, 24): {'velocity': 18, 'duration': 2},    # Organ
    range(24, 32): {'velocity': 20, 'duration': 4},    # Guitar
    range(32, 40): {'velocity': 24, 'duration': 8},    # Bass
    range(40, 48): {'velocity': 18, 'duration': 16},   # Strings
    range(48, 56): {'velocity': 18, 'duration': 16},   # Ensemble
    range(56, 64): {'velocity': 24, 'duration': 4},    # Brass
    range(64, 72): {'velocity': 20, 'duration': 4},    # Reed
    range(72, 80): {'velocity': 18, 'duration': 4},    # Pipe
    range(80, 88): {'velocity': 22, 'duration': 4},    # Synth Lead
    range(88, 96): {'velocity': 16, 'duration': 16},   # Synth Pad
    range(96, 104): {'velocity': 20, 'duration': 4},   # Synth Effects
    range(104, 112): {'velocity': 20, 'duration': 2},  # Ethnic
    range(112, 120): {'velocity': 26, 'duration': 1},  # Percussive
    range(120, 128): {'velocity': 20, 'duration': 2},  # Sound Effects
}
DRUM_DEFAULTS = {'velocity': 24, 'duration': 1}


def get_instrument_defaults(program: int) -> Dict[str, int]:
    """获取乐器的默认 velocity 和 duration"""
    for prog_range, defaults in INSTRUMENT_DEFAULTS.items():
        if program in prog_range:
            return defaults
    return {'velocity': 20, 'duration': 4}


class HIDTokenizer:
    """
    HID Tokenizer for MuseFormer Training

    词汇表 (331 tokens):
    - 特殊: PAD(0), BOS(1), EOS(2), MASK(3), SEP(4), UNK(5)
    - 结构: BAR(6)
    - 位置: T0-T15 (7-22)
    - 音高: P0-P127 (23-150)
    - 乐器: #D(151), #P0-#P127 (152-279)
    - 速度: BPM_0-BPM_31 (280-311)
    - 拍号: TS_* (312-330)
    """

    # 量化参数
    POSITIONS_PER_BAR = 16
    TEMPO_BINS = 32
    MIN_BPM = 40
    MAX_BPM = 250

    def __init__(self):
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
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
        """将 BPM 量化到最近的 bin"""
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
        """将 bin 索引转换回 BPM"""
        return self._tempo_values[max(0, min(self.TEMPO_BINS - 1, bin_idx))]

    def _build_vocab(self):
        """构建词汇表 (331 tokens)"""
        tokens = []

        # 特殊 tokens (6)
        tokens.extend(['PAD', 'BOS', 'EOS', 'MASK', 'SEP', 'UNK'])

        # Bar token (1)
        tokens.append('BAR')

        # Position tokens: T0-T15 (16)
        for i in range(self.POSITIONS_PER_BAR):
            tokens.append(f'T{i}')

        # Pitch tokens: P0-P127 (128)
        for i in range(128):
            tokens.append(f'P{i}')

        # Drum token (1)
        tokens.append('#D')

        # Program tokens: #P0-#P127 (128)
        for program in range(128):
            tokens.append(f'#P{program}')

        # Tempo tokens: BPM_0-BPM_31 (32)
        for i in range(self.TEMPO_BINS):
            tokens.append(f'BPM_{i}')

        # Time Signature tokens (19)
        time_sigs = [
            '1/4', '2/4', '3/4', '4/4', '5/4', '6/4', '7/4',
            '2/2', '3/2', '4/2',
            '3/8', '5/8', '6/8', '7/8', '9/8', '12/8',
            '2/8', '4/8', '8/8',
        ]
        for ts in time_sigs:
            tokens.append(f'TS_{ts}')

        # 构建映射
        for i, token in enumerate(tokens):
            self.token2id[token] = i
            self.id2token[i] = token

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

    @property
    def bar_id(self) -> int:
        return self.token2id['BAR']

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.id2token.get(key, 'UNK')
        return self.token2id.get(key, self.token2id['UNK'])

    def encode_file(self, txt_path: str, add_special: bool = True) -> List[int]:
        """将 TXT 文件编码为 token ID 序列"""
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return self.encode(content, add_special)

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """将文本编码为 token ID 序列"""
        tokens = []

        if add_special:
            tokens.append(self.bos_id)

        for line in text.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('H '):
                continue

            # Tempo 行
            if line.startswith('T ') and not line.startswith('TS '):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        bpm = float(parts[3])
                        bpm_bin = self.quantize_tempo(bpm)
                        tokens.append(self.token2id[f'BPM_{bpm_bin}'])
                    except ValueError:
                        pass
                continue

            # Time Signature 行
            if line.startswith('TS '):
                parts = line.split()
                if len(parts) >= 4:
                    ts_token = f'TS_{parts[3]}'
                    if ts_token in self.token2id:
                        tokens.append(self.token2id[ts_token])
                continue

            # Track 行
            if line.startswith('#'):
                track_token = line.strip()
                if track_token in self.token2id:
                    tokens.append(self.token2id[track_token])
                    tokens.append(self.sep_id)
                continue

            # 音符行
            parts = line.split()
            for part in parts:
                if part.startswith('B') and part[1:].isdigit():
                    tokens.append(self.token2id['BAR'])
                elif part.startswith('T') and part[1:].isdigit():
                    if part in self.token2id:
                        tokens.append(self.token2id[part])
                elif part.startswith('P') and part[1:].isdigit():
                    if part in self.token2id:
                        tokens.append(self.token2id[part])
                # 忽略 L 和 V

        if add_special:
            tokens.append(self.eos_id)

        return tokens

    def encode_with_info(self, text: str, add_special: bool = True) -> Tuple[List[int], List[TokenInfo]]:
        """
        编码并返回每个 token 的元信息
        用于构建注意力掩码
        """
        token_ids = []
        token_infos = []

        current_bar = -1
        current_pos = -1
        current_instrument = -1

        if add_special:
            token_ids.append(self.bos_id)
            token_infos.append(TokenInfo(self.bos_id, 'BOS'))

        for line in text.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('H '):
                continue

            # Tempo
            if line.startswith('T ') and not line.startswith('TS '):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        bpm = float(parts[3])
                        bpm_bin = self.quantize_tempo(bpm)
                        token = f'BPM_{bpm_bin}'
                        token_id = self.token2id[token]
                        token_ids.append(token_id)
                        token_infos.append(TokenInfo(token_id, token, instrument_id=-1))
                    except ValueError:
                        pass
                continue

            # Time Signature
            if line.startswith('TS '):
                parts = line.split()
                if len(parts) >= 4:
                    token = f'TS_{parts[3]}'
                    if token in self.token2id:
                        token_id = self.token2id[token]
                        token_ids.append(token_id)
                        token_infos.append(TokenInfo(token_id, token, instrument_id=-1))
                continue

            # Track
            if line.startswith('#'):
                track_token = line.strip()
                if track_token in self.token2id:
                    # 解析乐器 ID
                    if track_token == '#D':
                        current_instrument = 128  # Drum
                    elif track_token.startswith('#P'):
                        current_instrument = int(track_token[2:])

                    token_id = self.token2id[track_token]
                    token_ids.append(token_id)
                    token_infos.append(TokenInfo(
                        token_id, track_token,
                        instrument_id=current_instrument
                    ))

                    # SEP
                    token_ids.append(self.sep_id)
                    token_infos.append(TokenInfo(
                        self.sep_id, 'SEP',
                        instrument_id=current_instrument,
                        is_sep=True
                    ))

                    current_bar = -1
                    current_pos = -1
                continue

            # 音符行
            parts = line.split()
            for part in parts:
                if part.startswith('B') and part[1:].isdigit():
                    current_bar += 1
                    token_id = self.token2id['BAR']
                    token_ids.append(token_id)
                    token_infos.append(TokenInfo(
                        token_id, 'BAR',
                        bar_id=current_bar,
                        instrument_id=current_instrument,
                        is_bar=True
                    ))
                    current_pos = -1

                elif part.startswith('T') and part[1:].isdigit():
                    if part in self.token2id:
                        current_pos = int(part[1:])
                        token_id = self.token2id[part]
                        token_ids.append(token_id)
                        token_infos.append(TokenInfo(
                            token_id, part,
                            bar_id=current_bar,
                            position=current_pos,
                            instrument_id=current_instrument
                        ))

                elif part.startswith('P') and part[1:].isdigit():
                    if part in self.token2id:
                        token_id = self.token2id[part]
                        token_ids.append(token_id)
                        token_infos.append(TokenInfo(
                            token_id, part,
                            bar_id=current_bar,
                            position=current_pos,
                            instrument_id=current_instrument,
                            is_pitch=True
                        ))

        if add_special:
            token_ids.append(self.eos_id)
            token_infos.append(TokenInfo(self.eos_id, 'EOS'))

        return token_ids, token_infos

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """将 token ID 序列解码为文本"""
        lines = []
        current_track = None
        current_program = 0
        is_drum = False
        current_bar = -1
        note_line = []

        header = "H 1 480 Q"
        lines.append(header)
        lines.append("")

        for token_id in token_ids:
            token = self.id2token.get(token_id, 'UNK')

            if skip_special and token in ['PAD', 'BOS', 'EOS', 'MASK', 'UNK']:
                continue

            if token == 'SEP':
                if note_line:
                    lines.append(' '.join(note_line))
                    note_line = []
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
                current_bar = -1

                if token == '#D':
                    is_drum = True
                else:
                    is_drum = False
                    current_program = int(token[2:])
                continue

            if token == 'BAR':
                if note_line:
                    lines.append(' '.join(note_line))
                current_bar += 1
                note_line = [f'B{current_bar}']
                continue

            if token.startswith('T') and token[1:].isdigit():
                if note_line and any(p.startswith('P') for p in note_line):
                    lines.append(' '.join(note_line))
                    bar_tok = [t for t in note_line if t.startswith('B')]
                    note_line = bar_tok if bar_tok else []
                note_line.append(token)
                continue

            if token.startswith('P') and token[1:].isdigit():
                note_line.append(token)
                # 添加默认 L/V
                if is_drum:
                    defaults = DRUM_DEFAULTS
                else:
                    defaults = get_instrument_defaults(current_program)
                note_line.append(f"L{defaults['duration']}")
                note_line.append(f"V{defaults['velocity']}")
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
            }, f)

    @classmethod
    def load(cls, path: str) -> 'HIDTokenizer':
        """加载 tokenizer"""
        tokenizer = cls.__new__(cls)
        with open(path, 'rb') as f:
            data = pickle.load(f)
        tokenizer.token2id = data['token2id']
        tokenizer.id2token = data['id2token']
        tokenizer._build_tempo_table()
        return tokenizer


if __name__ == '__main__':
    tokenizer = HIDTokenizer()
    print(f"HID Tokenizer")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"PAD={tokenizer.pad_id}, BOS={tokenizer.bos_id}, EOS={tokenizer.eos_id}")
    print(f"SEP={tokenizer.sep_id}, BAR={tokenizer.bar_id}")
