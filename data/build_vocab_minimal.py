#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
极简版 Tokenizer (无 Velocity 和 Duration)

实验版本：完全移除 V 和 L tokens，解码时根据乐器特征推断

Token 类型 (331 tokens):
- 特殊 tokens: PAD, BOS, EOS, MASK, SEP, UNK (6)
- Bar token: BAR (相对小节标记) (1)
- Position tokens: T0-T15 (16分音符位置) (16)
- Pitch tokens: P0-P127 (音高) (128)
- Drum token: #D (MIDI channel 9) (1)
- Program tokens: #P0 - #P127 (128)
- Tempo tokens: BPM_0 - BPM_31 (32级量化) (32)
- Time Signature tokens: TS_1/4 - TS_12/8 (19)

相比完整版 (427 tokens):
- 移除 Duration: -64 tokens
- 移除 Velocity: -32 tokens
- 总计: 331 tokens (减少 22%)

解码策略：
- 根据乐器 (Program) 推断默认 Velocity 和 Duration
- 基于音乐常识和乐器特征
"""

import pickle
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional


# 乐器默认参数表 (基于 General MIDI 乐器分类)
INSTRUMENT_DEFAULTS = {
    # Piano (0-7): 中等力度，中等时值
    range(0, 8): {'velocity': 20, 'duration': 4},  # V20, L4 (1拍)

    # Chromatic Percussion (8-15): 较强力度，短时值
    range(8, 16): {'velocity': 22, 'duration': 2},  # V22, L2

    # Organ (16-23): 中等力度，短时值 (风琴音符通常较短)
    range(16, 24): {'velocity': 18, 'duration': 2},  # V18, L2 (半拍)

    # Guitar (24-31): 中等力度，中等时值
    range(24, 32): {'velocity': 20, 'duration': 4},  # V20, L4

    # Bass (32-39): 较强力度，长时值
    range(32, 40): {'velocity': 24, 'duration': 8},  # V24, L8

    # Strings (40-47): 中等力度，长时值
    range(40, 48): {'velocity': 18, 'duration': 16},  # V18, L16 (4拍)

    # Ensemble (48-55): 中等力度，长时值
    range(48, 56): {'velocity': 18, 'duration': 16},  # V18, L16

    # Brass (56-63): 较强力度，中等时值
    range(56, 64): {'velocity': 24, 'duration': 4},  # V24, L4

    # Reed (64-71): 中等力度，中等时值
    range(64, 72): {'velocity': 20, 'duration': 4},  # V20, L4

    # Pipe (72-79): 中等力度，中等时值
    range(72, 80): {'velocity': 18, 'duration': 4},  # V18, L4

    # Synth Lead (80-87): 较强力度，中等时值
    range(80, 88): {'velocity': 22, 'duration': 4},  # V22, L4

    # Synth Pad (88-95): 较弱力度，长时值
    range(88, 96): {'velocity': 16, 'duration': 16},  # V16, L16

    # Synth Effects (96-103): 中等力度，中等时值
    range(96, 104): {'velocity': 20, 'duration': 4},  # V20, L4

    # Ethnic (104-111): 中等力度，短时值
    range(104, 112): {'velocity': 20, 'duration': 2},  # V20, L2

    # Percussive (112-119): 较强力度，短时值
    range(112, 120): {'velocity': 26, 'duration': 1},  # V26, L1

    # Sound Effects (120-127): 中等力度，短时值
    range(120, 128): {'velocity': 20, 'duration': 2},  # V20, L2
}

# Drum 默认参数
DRUM_DEFAULTS = {'velocity': 24, 'duration': 1}  # V24, L1


def get_instrument_defaults(program: int) -> Dict[str, int]:
    """获取乐器的默认 velocity 和 duration"""
    for prog_range, defaults in INSTRUMENT_DEFAULTS.items():
        if program in prog_range:
            return defaults
    return {'velocity': 20, 'duration': 4}  # 默认值


class MinimalTokenizer:
    """极简版 Tokenizer (无 V/L)"""

    # 量化参数
    MAX_BAR = 1000
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
        """构建 tempo 量化表 (32 级，对数分布)"""
        min_bpm, max_bpm = 40, 250
        self._tempo_values = []
        for i in range(self.TEMPO_BINS):
            ratio = i / (self.TEMPO_BINS - 1)
            bpm = min_bpm * math.exp(ratio * math.log(max_bpm / min_bpm))
            self._tempo_values.append(round(bpm))
        self._tempo_values[0] = min_bpm
        self._tempo_values[-1] = max_bpm

    def quantize_tempo(self, bpm: float) -> int:
        """将 BPM 量化到最近的 bin (0-31)"""
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
        bin_idx = max(0, min(self.TEMPO_BINS - 1, bin_idx))
        return self._tempo_values[bin_idx]

    def _build_vocab(self):
        """构建词汇表 (331 tokens)"""
        tokens = []

        # 特殊 tokens (6)
        tokens.extend(['PAD', 'BOS', 'EOS', 'MASK', 'SEP', 'UNK'])

        # Bar token: BAR (1)
        tokens.append('BAR')

        # Position tokens: T0-T15 (16)
        for i in range(self.POSITIONS_PER_BAR):
            tokens.append(f'T{i}')

        # Pitch tokens: P0-P127 (128)
        for i in range(128):
            tokens.append(f'P{i}')

        # 注意：这里不再有 Duration 和 Velocity tokens

        # Drum token: #D (1)
        tokens.append('#D')

        # Program tokens: #P0-#P127 (128)
        for program in range(128):
            tokens.append(f'#P{program}')

        # Tempo tokens: BPM_0 to BPM_31 (32)
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
        """
        将文本编码为 token ID 序列

        完全忽略 L 和 V tokens
        """
        tokens = []
        self._last_bar = -1

        if add_special:
            tokens.append(self.bos_id)

        for line in text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue

            # 跳过 Header 行
            if line.startswith('H '):
                continue

            # Tempo 行
            if line.startswith('T ') and not line.startswith('TS '):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        bpm = float(parts[3])
                        bpm_bin = self.quantize_tempo(bpm)
                        bpm_token = f'BPM_{bpm_bin}'
                        tokens.append(self.token2id[bpm_token])
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

            # Track 行: #P{program} 或 #D
            if line.startswith('#'):
                track_token = line.strip()
                if track_token in self.token2id:
                    tokens.append(self.token2id[track_token])
                    tokens.append(self.token2id['SEP'])
                    self._last_bar = -1
                continue

            # 音符行: [B{bar}] T{pos} P{pitch} [L{dur}] [V{vel}]
            # 只保留 B, T, P，忽略 L 和 V
            parts = line.split()
            for part in parts:
                # Bar
                if part.startswith('B') and part[1:].isdigit():
                    bar_num = int(part[1:])
                    bar_num = min(bar_num, self.MAX_BAR - 1)
                    bars_to_add = bar_num - self._last_bar
                    for _ in range(bars_to_add):
                        tokens.append(self.token2id['BAR'])
                    self._last_bar = bar_num
                # Position
                elif part.startswith('T') and part[1:].isdigit():
                    tokens.append(self.token2id[part])
                # Pitch
                elif part.startswith('P') and part[1:].isdigit():
                    tokens.append(self.token2id[part])
                # 忽略 L 和 V

        if add_special:
            tokens.append(self.eos_id)

        return tokens

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """
        将 token ID 序列解码为 TXT 文本

        根据当前乐器 (Program) 推断 V 和 L
        """
        lines = []
        current_line_tokens = []
        last_bar = -1
        last_pos = None

        # 默认 header
        header = "H 1 480 Q"
        tempo_lines = []
        ts_lines = []
        track_sections = []
        current_track = None
        current_track_lines = []
        current_program = 0  # 当前乐器
        is_drum = False

        i = 0
        while i < len(token_ids):
            token_id = token_ids[i]
            token = self.id2token.get(token_id, 'UNK')

            # 跳过特殊 tokens
            if skip_special and token in ['PAD', 'BOS', 'EOS', 'MASK', 'UNK']:
                i += 1
                continue

            # SEP token
            if token == 'SEP':
                if current_line_tokens:
                    current_track_lines.append(' '.join(current_line_tokens))
                    current_line_tokens = []
                i += 1
                continue

            # BPM token
            if token.startswith('BPM_'):
                bpm_bin = int(token[4:])
                actual_bpm = self.dequantize_tempo(bpm_bin)
                tempo_lines.append(f'T B0 T0 {actual_bpm}')
                i += 1
                continue

            # Time Signature token
            if token.startswith('TS_'):
                ts = token[3:]
                ts_lines.append(f'TS B0 T0 {ts}')
                i += 1
                continue

            # Track token
            if token.startswith('#P'):
                # 保存上一个 track
                if current_track is not None:
                    if current_line_tokens:
                        current_track_lines.append(' '.join(current_line_tokens))
                        current_line_tokens = []
                    track_sections.append((current_track, current_track_lines))

                current_track = token
                current_track_lines = []
                last_bar = -1
                last_pos = None
                # 解析 program
                current_program = int(token[2:])
                is_drum = False
                i += 1
                continue

            if token == '#D':
                if current_track is not None:
                    if current_line_tokens:
                        current_track_lines.append(' '.join(current_line_tokens))
                        current_line_tokens = []
                    track_sections.append((current_track, current_track_lines))

                current_track = token
                current_track_lines = []
                last_bar = -1
                last_pos = None
                is_drum = True
                i += 1
                continue

            # BAR token
            if token == 'BAR':
                last_bar += 1
                if current_line_tokens:
                    current_track_lines.append(' '.join(current_line_tokens))
                current_line_tokens = [f'B{last_bar}']
                last_pos = None
                i += 1
                continue

            # Position token
            if token.startswith('T') and token[1:].isdigit():
                has_pitch = any(t.startswith('P') for t in current_line_tokens)
                if has_pitch:
                    current_track_lines.append(' '.join(current_line_tokens))
                    bar_token = [t for t in current_line_tokens if t.startswith('B') and t[1:].isdigit()]
                    current_line_tokens = bar_token if bar_token else []
                current_line_tokens.append(token)
                last_pos = token
                i += 1
                continue

            # Pitch token - 添加推断的 V 和 L
            if token.startswith('P') and token[1:].isdigit():
                has_pitch = any(t.startswith('P') for t in current_line_tokens)
                if has_pitch:
                    current_track_lines.append(' '.join(current_line_tokens))
                    bar_token = [t for t in current_line_tokens if t.startswith('B') and t[1:].isdigit()]
                    current_line_tokens = bar_token if bar_token else []

                # 检查是否需要继承 Position
                has_position = any(t.startswith('T') and t[1:].isdigit() for t in current_line_tokens)
                if not has_position and last_pos is not None:
                    current_line_tokens.append(last_pos)

                current_line_tokens.append(token)

                # 根据乐器推断 V 和 L
                if is_drum:
                    defaults = DRUM_DEFAULTS
                else:
                    defaults = get_instrument_defaults(current_program)

                current_line_tokens.append(f"L{defaults['duration']}")
                current_line_tokens.append(f"V{defaults['velocity']}")

                i += 1
                continue

            # 其他 token
            current_line_tokens.append(token)
            i += 1

        # 保存最后的内容
        if current_line_tokens:
            current_track_lines.append(' '.join(current_line_tokens))
        if current_track is not None:
            track_sections.append((current_track, current_track_lines))

        # 组装输出
        output_lines = [header]
        output_lines.extend(tempo_lines)
        output_lines.extend(ts_lines)
        output_lines.append('')

        for track, track_lines in track_sections:
            output_lines.append(track)
            output_lines.extend(track_lines)
            output_lines.append('')

        return '\n'.join(output_lines)

    def decode_to_file(self, token_ids: List[int], output_path: str, skip_special: bool = True):
        """将 token ID 序列解码并保存到文件"""
        text = self.decode(token_ids, skip_special)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)

    def save(self, path: str):
        """保存 tokenizer"""
        with open(path, 'wb') as f:
            pickle.dump({
                'token2id': self.token2id,
                'id2token': self.id2token,
            }, f)
        print(f"Tokenizer 已保存到 {path}")

    @classmethod
    def load(cls, path: str) -> 'MinimalTokenizer':
        """加载 tokenizer"""
        tokenizer = cls.__new__(cls)
        with open(path, 'rb') as f:
            data = pickle.load(f)
        tokenizer.token2id = data['token2id']
        tokenizer.id2token = data['id2token']
        tokenizer._build_tempo_table()
        return tokenizer


def main():
    """测试极简版 tokenizer"""
    tokenizer = MinimalTokenizer()

    print(f"=" * 60)
    print(f"极简版 Tokenizer (无 V/L)")
    print(f"=" * 60)
    print(f"词汇表大小: {tokenizer.vocab_size} tokens")
    print(f"相比完整版 (427): 减少 {427 - tokenizer.vocab_size} tokens ({(427 - tokenizer.vocab_size) / 427 * 100:.1f}%)")

    print(f"\n词汇表组成:")
    print(f"  特殊 tokens: 6 (PAD, BOS, EOS, MASK, SEP, UNK)")
    print(f"  Bar: 1 (BAR)")
    print(f"  Position: 16 (T0-T15)")
    print(f"  Pitch: 128 (P0-P127)")
    print(f"  Duration: 0 (已移除)")
    print(f"  Velocity: 0 (已移除)")
    print(f"  Drum: 1 (#D)")
    print(f"  Program: 128 (#P0-#P127)")
    print(f"  Tempo: 32 (BPM_0-BPM_31)")
    print(f"  Time Sig: 19")
    print(f"  总计: {tokenizer.vocab_size}")

    print(f"\n乐器默认参数 (解码时使用):")
    print(f"  Piano (0-7): V20, L4 (中等力度，1拍)")
    print(f"  Bass (32-39): V24, L8 (较强力度，2拍)")
    print(f"  Strings (40-47): V18, L16 (中等力度，4拍)")
    print(f"  Brass (56-63): V24, L4 (较强力度，1拍)")
    print(f"  Synth Pad (88-95): V16, L16 (较弱力度，4拍)")
    print(f"  Drums: V24, L1 (较强力度，短音)")

    # 保存到主 artifacts 目录
    save_path = Path(__file__).parent.parent / 'artifacts' / 'vocab_minimal.pkl'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(save_path))


if __name__ == '__main__':
    main()
