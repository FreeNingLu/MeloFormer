#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
和弦检测器 - 从 MIDI 音符推断和弦

用于将 BAR 替换为和弦标记
"""

from typing import List, Dict, Tuple, Optional
from collections import Counter
import numpy as np


# 音名映射
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# 和弦模板 (相对于根音的半音间隔)
CHORD_TEMPLATES = {
    'maj':   [0, 4, 7],           # 大三和弦: 1-3-5
    'min':   [0, 3, 7],           # 小三和弦: 1-b3-5
    'dim':   [0, 3, 6],           # 减三和弦: 1-b3-b5
    'aug':   [0, 4, 8],           # 增三和弦: 1-3-#5
    '7':     [0, 4, 7, 10],       # 属七和弦: 1-3-5-b7
    'maj7':  [0, 4, 7, 11],       # 大七和弦: 1-3-5-7
    'min7':  [0, 3, 7, 10],       # 小七和弦: 1-b3-5-b7
    'dim7':  [0, 3, 6, 9],        # 减七和弦: 1-b3-b5-bb7
    'hdim7': [0, 3, 6, 10],       # 半减七和弦: 1-b3-b5-b7
    'sus2':  [0, 2, 7],           # 挂二和弦: 1-2-5
    'sus4':  [0, 5, 7],           # 挂四和弦: 1-4-5
    'add9':  [0, 4, 7, 14],       # 加九和弦: 1-3-5-9
    '6':     [0, 4, 7, 9],        # 六和弦: 1-3-5-6
    'min6':  [0, 3, 7, 9],        # 小六和弦: 1-b3-5-6
}

# 简化版和弦模板 (只用常见的)
SIMPLE_CHORD_TEMPLATES = {
    'maj':   [0, 4, 7],
    'min':   [0, 3, 7],
    'dim':   [0, 3, 6],
    '7':     [0, 4, 7, 10],
    'maj7':  [0, 4, 7, 11],
    'min7':  [0, 3, 7, 10],
    'sus4':  [0, 5, 7],
}


class ChordDetector:
    """
    和弦检测器

    从一组音符中推断最可能的和弦
    """

    def __init__(self, use_simple: bool = True):
        """
        Args:
            use_simple: 是否使用简化的和弦模板
        """
        self.templates = SIMPLE_CHORD_TEMPLATES if use_simple else CHORD_TEMPLATES
        self._build_all_chords()

    def _build_all_chords(self):
        """构建所有可能的和弦 (12根音 × N种类型)"""
        self.all_chords = {}

        for root in range(12):
            for chord_type, intervals in self.templates.items():
                chord_name = f"{NOTE_NAMES[root]}{chord_type}"
                # 和弦包含的音高类 (pitch class, 0-11)
                pitch_classes = frozenset((root + i) % 12 for i in intervals)
                self.all_chords[chord_name] = {
                    'root': root,
                    'type': chord_type,
                    'pitch_classes': pitch_classes,
                    'intervals': intervals,
                }

    def detect_chord(self, pitches: List[int], weights: Optional[List[float]] = None) -> str:
        """
        检测和弦

        Args:
            pitches: 音高列表 (MIDI pitch, 0-127)
            weights: 每个音符的权重 (可选，如时值或力度)

        Returns:
            和弦名称 (如 "Cmaj", "Am", "G7") 或 "N" (无和弦)
        """
        if not pitches:
            return "N"

        # 转换为 pitch class (0-11)
        pitch_classes = [p % 12 for p in pitches]

        if weights is None:
            weights = [1.0] * len(pitches)

        # 统计每个 pitch class 的权重
        pc_weights = Counter()
        for pc, w in zip(pitch_classes, weights):
            pc_weights[pc] += w

        # 获取出现的 pitch class 集合
        present_pcs = set(pc_weights.keys())

        if len(present_pcs) < 2:
            return "N"

        # 计算每个和弦的匹配分数
        best_chord = "N"
        best_score = -1

        for chord_name, chord_info in self.all_chords.items():
            chord_pcs = chord_info['pitch_classes']

            # 计算匹配分数
            # 1. 和弦音符在输入中的覆盖率
            matched = present_pcs & chord_pcs
            coverage = len(matched) / len(chord_pcs)

            # 2. 根音权重 (根音应该是最强的)
            root = chord_info['root']
            root_weight = pc_weights.get(root, 0) / (sum(pc_weights.values()) + 1e-6)

            # 3. 非和弦音惩罚
            extra_notes = present_pcs - chord_pcs
            penalty = len(extra_notes) * 0.1

            # 综合分数
            score = coverage * 0.6 + root_weight * 0.3 - penalty

            if score > best_score:
                best_score = score
                best_chord = chord_name

        # 阈值过滤
        if best_score < 0.3:
            return "N"

        return best_chord

    def detect_chord_from_window(
        self,
        notes: List[Dict],
        start_tick: int,
        end_tick: int,
    ) -> str:
        """
        从时间窗口内的音符检测和弦

        Args:
            notes: 音符列表 [{'pitch': int, 'start': int, 'duration': int, 'velocity': int}, ...]
            start_tick: 窗口开始时间
            end_tick: 窗口结束时间

        Returns:
            和弦名称
        """
        # 筛选窗口内的音符
        window_notes = []
        for note in notes:
            note_start = note['start']
            note_end = note_start + note.get('duration', 1)

            # 音符与窗口有重叠
            if note_start < end_tick and note_end > start_tick:
                # 计算重叠时长作为权重
                overlap_start = max(note_start, start_tick)
                overlap_end = min(note_end, end_tick)
                overlap = overlap_end - overlap_start

                window_notes.append({
                    'pitch': note['pitch'],
                    'weight': overlap * note.get('velocity', 64) / 64,
                })

        if not window_notes:
            return "N"

        pitches = [n['pitch'] for n in window_notes]
        weights = [n['weight'] for n in window_notes]

        return self.detect_chord(pitches, weights)

    def detect_chord_sequence(
        self,
        notes: List[Dict],
        ticks_per_beat: int = 480,
        beats_per_chord: int = 2,  # 每几拍检测一个和弦
    ) -> List[Tuple[int, str]]:
        """
        检测整首曲子的和弦序列

        Args:
            notes: 音符列表
            ticks_per_beat: 每拍的 tick 数
            beats_per_chord: 每个和弦跨越的拍数

        Returns:
            [(tick, chord_name), ...] 和弦序列
        """
        if not notes:
            return []

        # 找到最大时间
        max_tick = max(n['start'] + n.get('duration', 0) for n in notes)

        # 每个和弦窗口的 tick 数
        window_ticks = ticks_per_beat * beats_per_chord

        chord_sequence = []
        current_tick = 0

        while current_tick < max_tick:
            chord = self.detect_chord_from_window(
                notes,
                current_tick,
                current_tick + window_ticks
            )
            chord_sequence.append((current_tick, chord))
            current_tick += window_ticks

        # 合并连续相同的和弦
        merged = []
        for tick, chord in chord_sequence:
            if not merged or merged[-1][1] != chord:
                merged.append((tick, chord))

        return merged


def get_chord_vocab() -> List[str]:
    """
    获取和弦词表

    Returns:
        和弦 token 列表
    """
    chords = ['N']  # 无和弦

    for root in NOTE_NAMES:
        for chord_type in SIMPLE_CHORD_TEMPLATES.keys():
            chords.append(f"{root}{chord_type}")

    return chords


if __name__ == '__main__':
    # 测试
    detector = ChordDetector()

    # C 大三和弦: C E G (60, 64, 67)
    chord = detector.detect_chord([60, 64, 67])
    print(f"C-E-G: {chord}")  # 应该是 Cmaj

    # A 小三和弦: A C E (57, 60, 64)
    chord = detector.detect_chord([57, 60, 64])
    print(f"A-C-E: {chord}")  # 应该是 Amin

    # G 属七和弦: G B D F (55, 59, 62, 65)
    chord = detector.detect_chord([55, 59, 62, 65])
    print(f"G-B-D-F: {chord}")  # 应该是 G7

    # 打印词表
    vocab = get_chord_vocab()
    print(f"\n和弦词表大小: {len(vocab)}")
    print(f"前 20 个: {vocab[:20]}")
