#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MusicFeatureExtractor - 音乐特征提取器

将 token 序列转换为小节级特征向量，用于 TDA 分析。

特征组成 (36 维):
- 音高直方图 (12维) - pitch class 分布
- 节奏模式 (16维) - 16 分音符位置
- 统计特征 (8维) - mean, std, range 等

注: 此模块与模型无关，可同时用于 MIDILM 和 MeloFormer。
    按和弦 token 切分序列 (两者均使用和弦 token 作为结构边界)。
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np


@dataclass
class BarFeature:
    """小节级特征"""
    bar_index: int
    chord: str

    # 音高特征
    pitch_histogram: np.ndarray  # shape: (12,) - pitch class 分布
    pitch_mean: float
    pitch_std: float
    pitch_range: int

    # 节奏特征
    onset_density: float         # 每拍音符数
    rhythm_pattern: np.ndarray   # shape: (16,) - 16分音符位置

    # 力度特征
    velocity_mean: float
    velocity_std: float

    # 时值特征
    duration_mean: float
    duration_diversity: float    # 时值多样性

    def to_vector(self) -> np.ndarray:
        """转换为向量表示 (用于 TDA 点云)"""
        return np.concatenate([
            self.pitch_histogram,                  # 12
            [self.pitch_mean / 127],               # 1
            [self.pitch_std / 20],                 # 1
            [min(self.pitch_range / 48, 1.0)],     # 1
            [min(self.onset_density / 4, 1.0)],    # 1
            self.rhythm_pattern,                    # 16
            [self.velocity_mean / 127],            # 1
            [self.velocity_std / 40],              # 1
            [self.duration_mean / 16],             # 1
            [self.duration_diversity],             # 1
        ])  # 总共 36 维


class MusicFeatureExtractor:
    """
    音乐特征提取器

    将 token 序列转换为小节级特征向量，用于 TDA 分析。
    按和弦 token 切分序列，每个和弦段落提取一个 BarFeature。
    """

    def __init__(self, tokenizer):
        """
        Args:
            tokenizer: HIDTokenizerV2 实例
        """
        self.tokenizer = tokenizer

    def extract_bar_features(self, token_ids: List[int]) -> List[BarFeature]:
        """
        提取小节级特征

        按和弦 token 切分序列，每个和弦段落提取一个 BarFeature。

        Args:
            token_ids: token ID 序列

        Returns:
            小节特征列表
        """
        bars = []
        current_bar_tokens = []
        current_chord = 'N'
        bar_index = 0

        for tid in token_ids:
            # 遇到和弦 token，保存当前小节并开始新小节
            if self.tokenizer.is_chord_token(tid):
                if current_bar_tokens:
                    feature = self._extract_single_bar(
                        current_bar_tokens,
                        current_chord,
                        bar_index
                    )
                    if feature is not None:
                        bars.append(feature)
                    bar_index += 1

                current_chord = self.tokenizer[tid]
                current_bar_tokens = []
            else:
                current_bar_tokens.append(tid)

        # 处理最后一个小节
        if current_bar_tokens:
            feature = self._extract_single_bar(
                current_bar_tokens,
                current_chord,
                bar_index
            )
            if feature is not None:
                bars.append(feature)

        return bars

    def _extract_single_bar(
        self,
        token_ids: List[int],
        chord: str,
        bar_index: int,
    ) -> Optional[BarFeature]:
        """提取单个小节的特征"""
        notes = self._parse_notes_from_tokens(token_ids)

        if not notes:
            return BarFeature(
                bar_index=bar_index,
                chord=chord,
                pitch_histogram=np.zeros(12),
                pitch_mean=60,
                pitch_std=0,
                pitch_range=0,
                onset_density=0,
                rhythm_pattern=np.zeros(16),
                velocity_mean=64,
                velocity_std=0,
                duration_mean=4,
                duration_diversity=0,
            )

        pitches = [n['pitch'] for n in notes]
        positions = [n['position'] for n in notes]
        velocities = [n.get('velocity', 80) for n in notes]
        durations = [n.get('duration', 4) for n in notes]

        # 音高直方图 (pitch class)
        pitch_histogram = np.zeros(12)
        for p in pitches:
            pitch_histogram[p % 12] += 1
        if pitch_histogram.sum() > 0:
            pitch_histogram = pitch_histogram / pitch_histogram.sum()

        # 节奏模式
        rhythm_pattern = np.zeros(16)
        for pos in positions:
            if 0 <= pos < 16:
                rhythm_pattern[pos] += 1
        if rhythm_pattern.sum() > 0:
            rhythm_pattern = rhythm_pattern / rhythm_pattern.sum()

        # 统计特征
        pitch_mean = np.mean(pitches)
        pitch_std = np.std(pitches) if len(pitches) > 1 else 0
        pitch_range = max(pitches) - min(pitches)

        velocity_mean = np.mean(velocities)
        velocity_std = np.std(velocities) if len(velocities) > 1 else 0

        duration_mean = np.mean(durations)
        duration_diversity = len(set(durations)) / max(len(durations), 1)

        return BarFeature(
            bar_index=bar_index,
            chord=chord,
            pitch_histogram=pitch_histogram,
            pitch_mean=float(pitch_mean),
            pitch_std=float(pitch_std),
            pitch_range=int(pitch_range),
            onset_density=len(notes) / 4,  # 假设每小节 4 拍
            rhythm_pattern=rhythm_pattern,
            velocity_mean=float(velocity_mean),
            velocity_std=float(velocity_std),
            duration_mean=float(duration_mean),
            duration_diversity=float(duration_diversity),
        )

    def _parse_notes_from_tokens(self, token_ids: List[int]) -> List[Dict]:
        """从 token 序列解析音符"""
        notes = []
        current_position = 0
        current_pitch = None
        current_duration = 4
        current_velocity = 2  # V2 = 中等力度

        for tid in token_ids:
            token = self.tokenizer[tid]

            if token.startswith('T') and len(token) > 1 and token[1:].isdigit():
                current_position = int(token[1:])
            elif token.startswith('P') and len(token) > 1 and token[1:].isdigit():
                current_pitch = int(token[1:])
            elif token.startswith('L') and len(token) > 1 and token[1:].isdigit():
                current_duration = int(token[1:])
            elif token.startswith('V') and len(token) > 1 and token[1:].isdigit():
                current_velocity = int(token[1:])

                # V token 标志音符结束，保存音符
                if current_pitch is not None:
                    notes.append({
                        'position': current_position,
                        'pitch': current_pitch,
                        'duration': current_duration,
                        'velocity': current_velocity * 4,  # 反量化
                    })
                    current_pitch = None

        return notes

    def bars_to_point_cloud(self, bars: List[BarFeature]) -> np.ndarray:
        """
        将小节特征列表转换为点云

        Args:
            bars: 小节特征列表

        Returns:
            shape: (n_bars, 36) 点云数组
        """
        if not bars:
            return np.empty((0, 36))

        return np.array([bar.to_vector() for bar in bars])

    def compute_bar_similarity(self, bar1: BarFeature, bar2: BarFeature) -> float:
        """
        计算两个小节的相似度

        Args:
            bar1, bar2: 小节特征

        Returns:
            相似度 (0-1)，越高越相似
        """
        v1 = bar1.to_vector()
        v2 = bar2.to_vector()

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(v1, v2) / (norm1 * norm2))

    def find_similar_bars(
        self,
        bars: List[BarFeature],
        threshold: float = 0.8,
    ) -> List[List[int]]:
        """
        找出相似的小节组

        Args:
            bars: 小节特征列表
            threshold: 相似度阈值

        Returns:
            相似小节的索引组列表
        """
        n = len(bars)
        if n < 2:
            return []

        similarity = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.compute_bar_similarity(bars[i], bars[j])
                similarity[i, j] = sim
                similarity[j, i] = sim

        groups = []
        visited = set()

        for i in range(n):
            if i in visited:
                continue

            group = [i]
            for j in range(i + 1, n):
                if j not in visited and similarity[i, j] >= threshold:
                    group.append(j)

            if len(group) > 1:
                groups.append(group)
                visited.update(group)

        return groups
