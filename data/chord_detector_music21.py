#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
和弦检测器 - 基于 Music21

使用 MIT 开发的 music21 库进行专业级和弦识别
"""

from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import re

from music21 import chord, pitch, key, roman, harmony


# 和弦类型映射：music21 名称 -> 简化名称
CHORD_TYPE_MAP = {
    # 三和弦
    'major triad': 'maj',
    'minor triad': 'min',
    'diminished triad': 'dim',
    'augmented triad': 'aug',

    # 七和弦
    'dominant seventh chord': '7',
    'major seventh chord': 'maj7',
    'minor seventh chord': 'min7',
    'diminished seventh chord': 'dim7',
    'half-diminished seventh chord': 'hdim7',
    'minor-major seventh chord': 'minmaj7',
    'augmented seventh chord': 'aug7',
    'augmented major seventh chord': 'augmaj7',

    # 挂留和弦
    'suspended fourth triad': 'sus4',
    'suspended second triad': 'sus2',
    'suspended fourth seventh chord': '7sus4',

    # 六和弦
    'major sixth chord': '6',
    'minor sixth chord': 'min6',

    # 九和弦
    'dominant ninth chord': '9',
    'major ninth chord': 'maj9',
    'minor ninth chord': 'min9',

    # 加音和弦
    'added-note': 'add',

    # 其他
    'power chord': '5',
    'Italian augmented sixth chord': 'It6',
    'German augmented sixth chord': 'Ger6',
    'French augmented sixth chord': 'Fr6',
}

# 音名映射
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTE_MAP = {
    'C': 'C', 'C#': 'C#', 'D-': 'C#',
    'D': 'D', 'D#': 'D#', 'E-': 'D#',
    'E': 'E', 'F-': 'E',
    'F': 'F', 'E#': 'F', 'F#': 'F#', 'G-': 'F#',
    'G': 'G', 'G#': 'G#', 'A-': 'G#',
    'A': 'A', 'A#': 'A#', 'B-': 'A#',
    'B': 'B', 'C-': 'B',
}


def normalize_note_name(name: str) -> str:
    """标准化音名 (去除降号，转为升号)"""
    # 提取基本音名
    base = name[0].upper()
    accidental = name[1:] if len(name) > 1 else ''

    full_name = base + accidental
    return NOTE_MAP.get(full_name, full_name)


class Music21ChordDetector:
    """
    基于 Music21 的和弦检测器
    """

    def __init__(self):
        # 构建支持的和弦词表
        self._build_chord_vocab()

    def _build_chord_vocab(self):
        """构建和弦词表"""
        chord_types = [
            'maj', 'min', 'dim', 'aug',
            '7', 'maj7', 'min7', 'dim7', 'hdim7',
            'sus2', 'sus4', '7sus4',
            '6', 'min6',
            '9', 'maj9', 'min9',
            '5',  # power chord
        ]

        self.chord_vocab = ['N']  # 无和弦
        for root in NOTE_NAMES:
            for ctype in chord_types:
                self.chord_vocab.append(f"{root}{ctype}")

        self.chord_set = set(self.chord_vocab)

    def detect_chord(self, pitches: List[int], weights: Optional[List[float]] = None) -> str:
        """
        检测和弦

        Args:
            pitches: MIDI 音高列表 (0-127)
            weights: 权重列表 (可选)

        Returns:
            和弦名称 (如 "Cmaj", "Amin", "G7")
        """
        if not pitches or len(pitches) < 2:
            return "N"

        # 去重
        unique_pitches = list(set(pitches))

        if len(unique_pitches) < 2:
            return "N"

        try:
            # 先尝试基于间隔的精确匹配
            interval_result = self._detect_by_intervals(unique_pitches, weights)
            if interval_result != "N" and interval_result in self.chord_set:
                return interval_result

            # 再尝试 music21
            pitch_objects = [pitch.Pitch(midi=p) for p in unique_pitches]
            c = chord.Chord(pitch_objects)
            chord_name = self._parse_chord(c)
            return chord_name

        except Exception as e:
            # 出错时返回 N
            return "N"

    def _detect_by_intervals(self, pitches: List[int], weights: Optional[List[float]] = None) -> str:
        """基于音程间隔的精确检测"""
        # 转换为 pitch class
        pcs = list(set(p % 12 for p in pitches))

        if len(pcs) < 2:
            return "N"

        # 和弦模板 (按优先级排序，更具体的在前)
        templates = [
            # 七和弦
            ([0, 4, 7, 10], '7'),      # 属七
            ([0, 4, 7, 11], 'maj7'),   # 大七
            ([0, 3, 7, 10], 'min7'),   # 小七
            ([0, 3, 6, 9], 'dim7'),    # 减七
            ([0, 3, 6, 10], 'hdim7'),  # 半减七
            # 三和弦
            ([0, 4, 7], 'maj'),        # 大三
            ([0, 3, 7], 'min'),        # 小三
            ([0, 3, 6], 'dim'),        # 减三
            ([0, 4, 8], 'aug'),        # 增三
            # 挂留和弦
            ([0, 5, 7], 'sus4'),       # 挂四
            ([0, 2, 7], 'sus2'),       # 挂二
            # Power chord
            ([0, 7], '5'),             # 五度
        ]

        # 计算权重 (如果提供)
        pc_weights = {}
        if weights:
            for p, w in zip(pitches, weights):
                pc = p % 12
                pc_weights[pc] = pc_weights.get(pc, 0) + w
        else:
            for p in pitches:
                pc = p % 12
                pc_weights[pc] = pc_weights.get(pc, 0) + 1

        # 找最低音作为候选根音的起点
        lowest_pc = min(p % 12 for p in pitches)

        best_match = "N"
        best_score = -1

        # 尝试每个 pitch class 作为根音
        for root in pcs:
            intervals = tuple(sorted((pc - root) % 12 for pc in pcs))

            for template, chord_type in templates:
                template_set = set(template)
                interval_set = set(intervals)

                # 检查模板是否是当前间隔的子集
                if template_set <= interval_set:
                    # 计算匹配分数
                    match_ratio = len(template_set) / len(interval_set)
                    root_weight = pc_weights.get(root, 0) / sum(pc_weights.values())

                    # 加分：根音是最低音
                    bass_bonus = 0.2 if root == lowest_pc else 0

                    score = len(template) * match_ratio + root_weight * 0.5 + bass_bonus

                    if score > best_score:
                        best_score = score
                        root_name = NOTE_NAMES[root]
                        best_match = f"{root_name}{chord_type}"
                        break  # 找到第一个匹配的模板就停止（优先级高的在前）

        return best_match

    def _parse_chord(self, c: chord.Chord) -> str:
        """解析 music21 Chord 对象"""
        try:
            # 方法1: 使用 pitchedCommonName
            common_name = c.pitchedCommonName

            if common_name and 'chord' not in common_name.lower() or 'triad' in common_name.lower():
                parsed = self._parse_common_name(common_name)
                if parsed != "N" and parsed in self.chord_set:
                    return parsed

            # 方法2: 使用 root + quality
            root = c.root()
            if root:
                root_name = normalize_note_name(root.name)

                # 获取和弦性质
                quality = c.quality

                chord_type = self._map_quality(quality, c)
                result = f"{root_name}{chord_type}"

                if result in self.chord_set:
                    return result

                # 尝试简化
                simple_type = self._simplify_chord_type(chord_type)
                result = f"{root_name}{simple_type}"
                if result in self.chord_set:
                    return result

            # 方法3: 尝试识别常见模式
            return self._fallback_detection(c)

        except Exception:
            return "N"

    def _parse_common_name(self, name: str) -> str:
        """解析 pitchedCommonName"""
        if not name:
            return "N"

        # 例如: "C-major triad" -> "Cmaj"
        # "A-minor seventh chord" -> "Amin7"

        parts = name.split('-')
        if len(parts) < 2:
            return "N"

        root = normalize_note_name(parts[0])
        rest = '-'.join(parts[1:]).lower()

        # 匹配和弦类型
        for full_name, short_name in CHORD_TYPE_MAP.items():
            if full_name in rest:
                return f"{root}{short_name}"

        # 简单匹配
        if 'major' in rest and 'seventh' in rest:
            return f"{root}maj7"
        elif 'minor' in rest and 'seventh' in rest:
            return f"{root}min7"
        elif 'dominant' in rest and 'seventh' in rest:
            return f"{root}7"
        elif 'diminished' in rest and 'seventh' in rest:
            return f"{root}dim7"
        elif 'major' in rest:
            return f"{root}maj"
        elif 'minor' in rest:
            return f"{root}min"
        elif 'diminished' in rest:
            return f"{root}dim"
        elif 'augmented' in rest:
            return f"{root}aug"
        elif 'suspended' in rest and 'fourth' in rest:
            return f"{root}sus4"
        elif 'suspended' in rest and 'second' in rest:
            return f"{root}sus2"

        return "N"

    def _map_quality(self, quality: str, c: chord.Chord) -> str:
        """映射 music21 quality 到简化名称"""
        if not quality:
            return "maj"

        quality_lower = quality.lower()

        # 检查是否是七和弦
        is_seventh = c.isSeventh() if hasattr(c, 'isSeventh') else False

        if 'major' in quality_lower:
            return 'maj7' if is_seventh else 'maj'
        elif 'minor' in quality_lower:
            return 'min7' if is_seventh else 'min'
        elif 'diminished' in quality_lower:
            return 'dim7' if is_seventh else 'dim'
        elif 'augmented' in quality_lower:
            return 'aug7' if is_seventh else 'aug'
        elif 'dominant' in quality_lower:
            return '7'
        elif 'half-diminished' in quality_lower:
            return 'hdim7'
        elif 'suspended' in quality_lower:
            return 'sus4'
        else:
            return 'maj'

    def _simplify_chord_type(self, chord_type: str) -> str:
        """简化和弦类型"""
        simplify_map = {
            'augmaj7': 'aug',
            'minmaj7': 'min7',
            'aug7': 'aug',
            '7sus4': 'sus4',
            'hdim7': 'min7',
            '9': '7',
            'maj9': 'maj7',
            'min9': 'min7',
            'add': 'maj',
        }
        return simplify_map.get(chord_type, chord_type)

    def _fallback_detection(self, c: chord.Chord) -> str:
        """后备检测方法"""
        try:
            # 使用 intervalVector 分析
            pitches = [p.midi for p in c.pitches]
            pcs = list(set(p % 12 for p in pitches))

            if len(pcs) < 2:
                return "N"

            # 尝试每个音作为根音
            best_match = "N"
            best_score = 0

            for root in pcs:
                intervals = sorted((pc - root) % 12 for pc in pcs)

                # 匹配常见模式
                patterns = {
                    (0, 4, 7): 'maj',
                    (0, 3, 7): 'min',
                    (0, 3, 6): 'dim',
                    (0, 4, 8): 'aug',
                    (0, 4, 7, 10): '7',
                    (0, 4, 7, 11): 'maj7',
                    (0, 3, 7, 10): 'min7',
                    (0, 5, 7): 'sus4',
                    (0, 2, 7): 'sus2',
                    (0, 7): '5',
                }

                for pattern, chord_type in patterns.items():
                    if set(intervals) >= set(pattern):
                        score = len(pattern)
                        if score > best_score:
                            best_score = score
                            root_name = NOTE_NAMES[root]
                            best_match = f"{root_name}{chord_type}"

            return best_match if best_match in self.chord_set else "N"

        except Exception:
            return "N"

    def detect_chord_from_window(
        self,
        notes: List[Dict],
        start_tick: int,
        end_tick: int,
    ) -> str:
        """
        从时间窗口检测和弦

        Args:
            notes: 音符列表 [{'pitch': int, 'start': int, 'duration': int}, ...]
            start_tick: 窗口开始
            end_tick: 窗口结束

        Returns:
            和弦名称
        """
        window_pitches = []
        window_weights = []

        for note in notes:
            note_start = note['start']
            note_end = note_start + note.get('duration', 1)

            # 检查重叠
            if note_start < end_tick and note_end > start_tick:
                overlap = min(note_end, end_tick) - max(note_start, start_tick)
                velocity = note.get('velocity', 64)

                window_pitches.append(note['pitch'])
                window_weights.append(overlap * velocity)

        return self.detect_chord(window_pitches, window_weights)

    def detect_chord_sequence(
        self,
        notes: List[Dict],
        ticks_per_beat: int = 480,
        beats_per_chord: int = 2,
    ) -> List[Tuple[int, str]]:
        """
        检测和弦序列

        Args:
            notes: 音符列表
            ticks_per_beat: 每拍 tick 数
            beats_per_chord: 每个和弦的拍数

        Returns:
            [(tick, chord_name), ...]
        """
        if not notes:
            return [(0, 'N')]

        max_tick = max(n['start'] + n.get('duration', 0) for n in notes)
        window_ticks = ticks_per_beat * beats_per_chord

        chord_sequence = []
        current_tick = 0

        while current_tick < max_tick:
            chord_name = self.detect_chord_from_window(
                notes, current_tick, current_tick + window_ticks
            )
            chord_sequence.append((current_tick, chord_name))
            current_tick += window_ticks

        # 合并连续相同和弦
        merged = []
        for tick, chord_name in chord_sequence:
            if not merged or merged[-1][1] != chord_name:
                merged.append((tick, chord_name))

        return merged if merged else [(0, 'N')]

    def get_vocab(self) -> List[str]:
        """获取和弦词表"""
        return self.chord_vocab.copy()


def get_chord_vocab() -> List[str]:
    """获取和弦词表"""
    detector = Music21ChordDetector()
    return detector.get_vocab()


if __name__ == '__main__':
    detector = Music21ChordDetector()

    print("=" * 60)
    print("Music21 和弦检测测试")
    print("=" * 60)

    # 基础三和弦
    print("\n【基础三和弦】")
    print(f"C-E-G (60,64,67):     {detector.detect_chord([60, 64, 67])}")
    print(f"A-C-E (57,60,64):     {detector.detect_chord([57, 60, 64])}")
    print(f"G-B-D (55,59,62):     {detector.detect_chord([55, 59, 62])}")

    # 七和弦
    print("\n【七和弦】")
    print(f"G-B-D-F (55,59,62,65):  {detector.detect_chord([55, 59, 62, 65])}")
    print(f"C-E-G-B (60,64,67,71):  {detector.detect_chord([60, 64, 67, 71])}")
    print(f"A-C-E-G (57,60,64,67):  {detector.detect_chord([57, 60, 64, 67])}")

    # 转位
    print("\n【转位和弦】")
    print(f"E-G-C (64,67,72):     {detector.detect_chord([64, 67, 72])}")
    print(f"G-C-E (67,72,76):     {detector.detect_chord([67, 72, 76])}")

    # 复杂和弦
    print("\n【复杂和弦】")
    print(f"C-Eb-Gb (60,63,66):   {detector.detect_chord([60, 63, 66])}")  # Cdim
    print(f"C-E-G# (60,64,68):    {detector.detect_chord([60, 64, 68])}")  # Caug
    print(f"C-F-G (60,65,67):     {detector.detect_chord([60, 65, 67])}")  # Csus4
    print(f"C-D-G (60,62,67):     {detector.detect_chord([60, 62, 67])}")  # Csus2

    # 带额外音
    print("\n【带额外音】")
    print(f"C-E-G-C' (60,64,67,72): {detector.detect_chord([60, 64, 67, 72])}")
    print(f"C-D-E-G (60,62,64,67):  {detector.detect_chord([60, 62, 64, 67])}")

    # 边界情况
    print("\n【边界情况】")
    print(f"单音 (60):            {detector.detect_chord([60])}")
    print(f"空:                   {detector.detect_chord([])}")
    print(f"C-G power (60,67):    {detector.detect_chord([60, 67])}")

    print(f"\n和弦词表大小: {len(detector.get_vocab())}")
