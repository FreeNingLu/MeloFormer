#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generator.py - MeloFormer v2.1 核心生成器模块

功能:
1. MusicGenerator 类 - 单次自回归生成
2. 音乐专用重复惩罚 (和弦循环、和弦重复、模式重复)
3. 置信度分析 (generate_with_confidence)
4. Token → MIDI 转换

与 MIDILM 的关键区别:
- MeloFormer 需要维护 chord_ids, instrument_ids, token_type_ids, note_ids
- 每次 forward 需要重新计算 FlexAttention mask (无 KV cache)
- batch_size 固定为 1 (FlexAttention v2.1 约束)
- chord_ids 跟踪: 遇到和弦 token 时 chord counter 递增
- token_type: T=0, P=1, D/L=2, V=3, 其他=-1
- note_id: 每遇到 T token 时递增

使用:
    from inference.generator import MusicGenerator, load_model

    model, tokenizer = load_model(checkpoint_path, device)
    generator = MusicGenerator(model, tokenizer, device)

    # 基础生成
    token_ids = generator.generate(instruments=[128, 33, 0], ...)

    # 带置信度生成
    result = generator.generate_with_confidence(...)

CLI:
    python -m inference.generator --checkpoint model.pt --output out.mid
"""

import time
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm

import torch
import torch.nn.functional as F
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

import sys
from pathlib import Path

# 添加项目根目录到 path
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model import MeloFormer, create_model
from data import HIDTokenizerV2


# ============================================================
# 数据类
# ============================================================

@dataclass
class GenerationStep:
    """单步生成结果"""
    token_id: int
    token_str: str
    confidence: float      # softmax 概率
    entropy: float         # 分布熵
    top_k_tokens: List[Tuple[str, float]]  # [(token, prob), ...]


@dataclass
class GenerationResult:
    """完整生成结果"""
    token_ids: List[int]
    steps: List[GenerationStep]
    avg_confidence: float
    avg_entropy: float
    low_confidence_positions: List[int]  # 置信度低的位置


@dataclass
class GenerationMeta:
    """generate() 的性能元数据"""
    token_ids: List[int]
    n_tokens: int          # 生成的 token 数 (不含 prompt)
    elapsed_ms: float      # 总耗时 (毫秒)
    tokens_per_sec: float  # 吞吐量
    peak_mem_gb: float     # 峰值显存 (GB), CPU 推理时为 0
    n_chords: int          # 生成的 chord 数


# ============================================================
# 辅助: 序列元信息追踪器
# ============================================================

class SequenceTracker:
    """
    追踪生成序列中的 chord_ids, instrument_ids, token_type_ids, note_ids

    MeloFormer v2.1 需要这些辅助 ID 来构建 FlexAttention mask。
    在自回归生成过程中，每生成一个新 token 就需要更新这些 ID。
    """

    def __init__(self, tokenizer: HIDTokenizerV2):
        self.tokenizer = tokenizer
        self.chord_counter: int = 0
        self.current_instrument: int = 0
        self.note_counter: int = 0

        self.chord_ids: List[int] = []
        self.instrument_ids: List[int] = []
        self.token_type_ids: List[int] = []
        self.note_ids: List[int] = []

    def reset(self):
        """重置追踪器状态"""
        self.chord_counter = 0
        self.current_instrument = 0
        self.note_counter = 0
        self.chord_ids = []
        self.instrument_ids = []
        self.token_type_ids = []
        self.note_ids = []

    def _classify_token(self, token_id: int) -> Tuple[int, int, int, int]:
        """
        分类一个 token，返回 (chord_id, instrument_id, token_type, note_id)

        Token type: T=0, P=1, D/L=2, V=3, other=-1
        """
        token_str = self.tokenizer[token_id]

        # 和弦 token: 递增 chord counter
        if self.tokenizer.is_chord_token(token_id):
            self.chord_counter += 1
            return (self.chord_counter - 1, self.current_instrument, -1, -1)

        # 乐器 token: 更新当前乐器
        if token_str == '#D':
            self.current_instrument = 128
            return (self.chord_counter, self.current_instrument, -1, -1)
        if token_str.startswith('#P') and token_str[2:].isdigit():
            self.current_instrument = int(token_str[2:])
            return (self.chord_counter, self.current_instrument, -1, -1)

        # 位置 token (T): 新音符开始
        if token_str.startswith('T') and len(token_str) > 1 and token_str[1:].isdigit():
            self.note_counter += 1
            return (self.chord_counter, self.current_instrument, 0, self.note_counter)

        # 音高 token (P)
        if token_str.startswith('P') and len(token_str) > 1 and token_str[1:].isdigit():
            return (self.chord_counter, self.current_instrument, 1, self.note_counter)

        # 时值 token (L)
        if token_str.startswith('L') and len(token_str) > 1 and token_str[1:].isdigit():
            return (self.chord_counter, self.current_instrument, 2, self.note_counter)

        # 力度 token (V)
        if token_str.startswith('V') and len(token_str) > 1 and token_str[1:].isdigit():
            return (self.chord_counter, self.current_instrument, 3, self.note_counter)

        # 其他 (BOS, EOS, SEP, BPM, TS 等)
        return (self.chord_counter, self.current_instrument, -1, -1)

    def append(self, token_id: int):
        """追踪新生成的 token"""
        chord_id, inst_id, tt, nid = self._classify_token(token_id)
        self.chord_ids.append(chord_id)
        self.instrument_ids.append(inst_id)
        self.token_type_ids.append(tt)
        self.note_ids.append(nid)

    def extend(self, token_ids: List[int]):
        """批量追踪 token 序列"""
        for tid in token_ids:
            self.append(tid)

    def to_tensors(self, device: torch.device) -> Tuple[torch.Tensor, ...]:
        """返回 (chord_ids, instrument_ids, token_type_ids, note_ids) tensors, shape: (1, seq_len)"""
        return (
            torch.tensor([self.chord_ids], dtype=torch.long, device=device),
            torch.tensor([self.instrument_ids], dtype=torch.long, device=device),
            torch.tensor([self.token_type_ids], dtype=torch.long, device=device),
            torch.tensor([self.note_ids], dtype=torch.long, device=device),
        )

    @property
    def num_chords(self) -> int:
        """当前 chord 数量 (chord_counter 表示已出现的 chord 数)"""
        return max(self.chord_counter, 1)


# ============================================================
# 主生成器类
# ============================================================

class MusicGenerator:
    """
    MeloFormer v2.1 音乐生成器

    功能:
    - 基础生成 (generate)
    - 带置信度生成 (generate_with_confidence)
    - 音乐专用重复惩罚 (和弦循环、和弦重复、音符模式)
    - 强制多轨 / 强制和弦
    - MIDI 导出

    与 MIDILM 生成器的区别:
    - 每次 forward 传入完整序列 + 辅助 ID (无 KV cache)
    - batch_size 固定为 1
    - 需要维护 SequenceTracker 追踪辅助 ID
    """

    def __init__(
        self,
        model: MeloFormer,
        tokenizer: HIDTokenizerV2,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    # ========================================
    # 核心生成方法
    # ========================================

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Optional[List[int]] = None,
        max_length: int = 2048,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        # 音乐专用重复惩罚 (推荐)
        use_music_penalty: bool = True,
        bar_loop_penalty: float = 0.5,
        max_same_chord: int = 2,
        pattern_penalty: float = 0.3,
        # 通用重复惩罚 (use_music_penalty=False 时生效)
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        no_repeat_ngram_size: int = 0,
        # 条件生成参数
        instruments: Optional[List[int]] = None,
        tempo: int = 120,
        time_signature: str = '4/4',
        chords: Optional[List[str]] = None,
        force_multi_track: bool = False,
        force_chords: bool = False,
        min_notes_per_track: int = 50,
        block_instrument_switch: bool = False,
        # 其他
        verbose: bool = False,
    ) -> List[int]:
        """
        生成音乐

        Args:
            prompt_ids: token ID 格式的提示
            max_length: 最大生成长度
            temperature: 温度 (越高越随机)
            top_k: Top-K 采样
            top_p: Top-P 采样

            use_music_penalty: 使用音乐专用惩罚 (推荐开启)
            bar_loop_penalty: 检测到和弦循环时的惩罚强度 (0-1, 越小惩罚越强)
            max_same_chord: 允许连续相同和弦的最大数量
            pattern_penalty: 检测到音符模式重复时的惩罚 (0-1)

            repetition_penalty: 通用重复惩罚 (use_music_penalty=False 时)
            frequency_penalty: 通用频率惩罚
            presence_penalty: 通用存在惩罚
            no_repeat_ngram_size: 禁止重复的 n-gram 大小

            instruments: 要使用的乐器列表
            tempo: BPM
            time_signature: 拍号
            chords: 和弦进行
            force_multi_track: 强制多轨生成
            force_chords: 强制按照指定和弦进行
            min_notes_per_track: 强制多轨时每个乐器的最小音符数
            block_instrument_switch: 禁止生成其他乐器标记
            verbose: 是否显示进度

        Returns:
            生成的 token IDs
        """
        # 构建初始 prompt
        if prompt_ids is None:
            prompt_ids = self._build_prompt(
                instruments=instruments,
                tempo=tempo,
                time_signature=time_signature,
                chords=chords,
            )

        generated_ids = list(prompt_ids)
        prompt_len = len(prompt_ids)

        # 性能计时
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        t_start = time.time()

        # 初始化 SequenceTracker
        tracker = SequenceTracker(self.tokenizer)
        tracker.extend(prompt_ids)

        # 统计 token 频率 (用于 frequency_penalty)
        token_counts: Dict[int, int] = {}
        for tid in prompt_ids:
            token_counts[tid] = token_counts.get(tid, 0) + 1

        # 强制和弦相关
        forced_chord_ids = []
        chord_index = 0
        if force_chords and chords:
            for chord in chords:
                if chord in self.tokenizer.token2id:
                    forced_chord_ids.append(self.tokenizer[chord])
            chord_index = 1 if len(forced_chord_ids) > 1 else 0

        # 强制多轨相关
        current_track_idx = 0
        current_track_notes = 0
        instrument_tokens = []
        if force_multi_track and instruments and len(instruments) > 1:
            for inst in instruments:
                if inst == 128:
                    instrument_tokens.append(self.tokenizer['#D'])
                else:
                    instrument_tokens.append(self.tokenizer[f'#P{inst}'])
            if verbose:
                print(f"  [强制多轨] 乐器队列: {[self.tokenizer[tid] for tid in instrument_tokens]}")

        # 生成循环
        pbar = tqdm(
            range(max_length - len(prompt_ids)),
            desc='Generating',
            disable=not verbose,
            dynamic_ncols=True,
            unit='tok',
        )
        for step in pbar:
            # 构建 tensor 并调用模型 (完整序列, 无 KV cache)
            input_ids = torch.tensor([generated_ids], dtype=torch.long, device=self.device)
            chord_ids_t, inst_ids_t, tt_ids_t, note_ids_t = tracker.to_tensors(self.device)

            logits = self.model(
                input_ids,
                chord_ids=chord_ids_t,
                instrument_ids=inst_ids_t,
                token_type_ids=tt_ids_t,
                note_ids=note_ids_t,
                num_chords=tracker.num_chords,
            )

            # 获取最后位置的 logits
            next_logits = logits[0, -1, :].clone()

            # ========================================
            # 应用重复惩罚
            # ========================================
            if use_music_penalty:
                next_logits = self._apply_music_repetition_penalty(
                    next_logits,
                    generated_ids,
                    bar_loop_penalty=bar_loop_penalty,
                    max_same_chord=max_same_chord,
                    pattern_penalty=pattern_penalty,
                )
            else:
                next_logits = self._apply_repetition_penalties(
                    next_logits,
                    generated_ids,
                    token_counts,
                    repetition_penalty=repetition_penalty,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                )
                if no_repeat_ngram_size > 0:
                    next_logits = self._apply_ngram_blocking(
                        next_logits, generated_ids, no_repeat_ngram_size
                    )

            # 禁止乐器切换
            if block_instrument_switch:
                drum_token_id = self.tokenizer.token2id.get('#D')
                if drum_token_id is not None:
                    next_logits[drum_token_id] = float('-inf')
                for prog in range(128):
                    inst_token_id = self.tokenizer.token2id.get(f'#P{prog}')
                    if inst_token_id is not None:
                        next_logits[inst_token_id] = float('-inf')

            # 温度缩放
            next_logits = next_logits / temperature

            # Top-K 过滤
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[-1]] = float('-inf')

            # Top-P 过滤
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[indices_to_remove] = float('-inf')

            # 采样
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 强制和弦替换
            if force_chords and forced_chord_ids and self.tokenizer.is_chord_token(next_token.item()):
                next_token = torch.tensor(
                    [forced_chord_ids[chord_index % len(forced_chord_ids)]],
                    device=self.device,
                )
                chord_index += 1

            # 强制多轨逻辑
            if force_multi_track and instrument_tokens:
                token_str = self.tokenizer[next_token.item()]

                if token_str.startswith('P') and token_str[1:].isdigit():
                    current_track_notes += 1

                should_switch = False

                if next_token.item() == self.tokenizer.eos_id and current_track_idx < len(instrument_tokens) - 1:
                    if current_track_notes >= min_notes_per_track:
                        should_switch = True
                    else:
                        next_logits[self.tokenizer.eos_id] = float('-inf')
                        probs = F.softmax(next_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        token_str = self.tokenizer[next_token.item()]
                        if token_str.startswith('P') and token_str[1:].isdigit():
                            current_track_notes += 1

                if current_track_notes >= min_notes_per_track and current_track_idx < len(instrument_tokens) - 1:
                    last_token_str = self.tokenizer[generated_ids[-1]] if generated_ids else ''
                    if (last_token_str.startswith('V') or
                        last_token_str.startswith('L') or
                        last_token_str.startswith('P') or
                        (generated_ids and self.tokenizer.is_chord_token(generated_ids[-1]))):
                        should_switch = True

                if should_switch:
                    current_track_idx += 1
                    current_track_notes = 0
                    next_inst_token = instrument_tokens[current_track_idx]
                    if verbose:
                        print(f"  [强制多轨] 切换到轨道 {current_track_idx + 1}: "
                              f"{self.tokenizer[next_inst_token]}")

                    extra_tokens = [next_inst_token, self.tokenizer.sep_id]
                    if chords and chords[0] in self.tokenizer.token2id:
                        extra_tokens.append(self.tokenizer[chords[0]])

                    # 追踪额外 token
                    for tid in extra_tokens:
                        generated_ids.append(tid)
                        tracker.append(tid)
                        token_counts[tid] = token_counts.get(tid, 0) + 1
                    continue

            # 添加到生成序列
            token_id = next_token.item()
            generated_ids.append(token_id)
            tracker.append(token_id)
            token_counts[token_id] = token_counts.get(token_id, 0) + 1

            # 更新进度条
            if verbose:
                elapsed = time.time() - t_start
                gen_count = len(generated_ids) - prompt_len
                tok_s = gen_count / max(elapsed, 1e-6)
                pbar.set_postfix(chords=tracker.num_chords, tok_s=f'{tok_s:.0f}')

            # 检查 EOS
            if token_id == self.tokenizer.eos_id:
                break

        pbar.close()

        # 计算性能指标
        elapsed_s = time.time() - t_start
        n_generated = len(generated_ids) - prompt_len
        tok_s = n_generated / max(elapsed_s, 1e-6)
        if self.device.type == 'cuda':
            peak_mem_gb = torch.cuda.max_memory_allocated(self.device) / 1e9
        else:
            peak_mem_gb = 0.0

        if verbose:
            print(f"\n生成完成: {n_generated} tokens | "
                  f"{tok_s:.1f} tok/s | "
                  f"{elapsed_s*1000:.0f} ms | "
                  f"峰值显存 {peak_mem_gb:.2f} GB | "
                  f"{tracker.num_chords} chords")

        return GenerationMeta(
            token_ids=generated_ids,
            n_tokens=n_generated,
            elapsed_ms=elapsed_s * 1000,
            tokens_per_sec=tok_s,
            peak_mem_gb=peak_mem_gb,
            n_chords=tracker.num_chords,
        )

    @torch.no_grad()
    def generate_with_confidence(
        self,
        prompt_ids: Optional[List[int]] = None,
        max_length: int = 1024,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        no_repeat_ngram_size: int = 0,
        confidence_threshold: float = 0.1,
        return_top_k: int = 5,
        # 条件生成参数
        instruments: Optional[List[int]] = None,
        tempo: int = 120,
        time_signature: str = '4/4',
        chords: Optional[List[str]] = None,
    ) -> GenerationResult:
        """
        生成音乐，同时输出每步的置信度信息

        Args:
            ... (同 generate)
            confidence_threshold: 低置信度阈值
            return_top_k: 返回多少个候选

        Returns:
            GenerationResult: 包含 token_ids 和每步置信度
        """
        if prompt_ids is None:
            prompt_ids = self._build_prompt(
                instruments=instruments,
                tempo=tempo,
                time_signature=time_signature,
                chords=chords,
            )

        generated_ids = list(prompt_ids)
        steps: List[GenerationStep] = []
        low_confidence_positions: List[int] = []

        # 初始化 SequenceTracker
        tracker = SequenceTracker(self.tokenizer)
        tracker.extend(prompt_ids)

        token_counts: Dict[int, int] = {}
        for tid in prompt_ids:
            token_counts[tid] = token_counts.get(tid, 0) + 1

        for step in range(max_length - len(prompt_ids)):
            # 构建 tensor 并调用模型
            input_ids = torch.tensor([generated_ids], dtype=torch.long, device=self.device)
            chord_ids_t, inst_ids_t, tt_ids_t, note_ids_t = tracker.to_tensors(self.device)

            logits = self.model(
                input_ids,
                chord_ids=chord_ids_t,
                instrument_ids=inst_ids_t,
                token_type_ids=tt_ids_t,
                note_ids=note_ids_t,
                num_chords=tracker.num_chords,
            )

            next_logits = logits[0, -1, :].clone()

            # 计算原始概率分布 (用于置信度计算)
            raw_probs = F.softmax(next_logits, dim=-1)
            entropy = -torch.sum(raw_probs * torch.log(raw_probs + 1e-10)).item()

            # 应用重复惩罚
            next_logits = self._apply_repetition_penalties(
                next_logits, generated_ids, token_counts,
                repetition_penalty, frequency_penalty, presence_penalty
            )
            if no_repeat_ngram_size > 0:
                next_logits = self._apply_ngram_blocking(
                    next_logits, generated_ids, no_repeat_ngram_size
                )

            next_logits = next_logits / temperature

            # Top-K 过滤
            filtered_logits = next_logits.clone()
            if top_k > 0:
                v, _ = torch.topk(filtered_logits, min(top_k, filtered_logits.size(-1)))
                filtered_logits[filtered_logits < v[-1]] = float('-inf')

            # Top-P 过滤
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                filtered_logits[indices_to_remove] = float('-inf')

            # 采样
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()

            # 置信度 = 采样 token 的原始概率
            confidence = raw_probs[token_id].item()

            # Top-K 候选
            top_k_values, top_k_indices = torch.topk(raw_probs, return_top_k)
            top_k_tokens = [
                (self.tokenizer[idx.item()], prob.item())
                for idx, prob in zip(top_k_indices, top_k_values)
            ]

            step_result = GenerationStep(
                token_id=token_id,
                token_str=self.tokenizer[token_id],
                confidence=confidence,
                entropy=entropy,
                top_k_tokens=top_k_tokens,
            )
            steps.append(step_result)

            if confidence < confidence_threshold:
                low_confidence_positions.append(len(generated_ids))

            generated_ids.append(token_id)
            tracker.append(token_id)
            token_counts[token_id] = token_counts.get(token_id, 0) + 1

            if token_id == self.tokenizer.eos_id:
                break

        avg_confidence = sum(s.confidence for s in steps) / len(steps) if steps else 0
        avg_entropy = sum(s.entropy for s in steps) / len(steps) if steps else 0

        return GenerationResult(
            token_ids=generated_ids,
            steps=steps,
            avg_confidence=avg_confidence,
            avg_entropy=avg_entropy,
            low_confidence_positions=low_confidence_positions,
        )

    # ========================================
    # 音乐专用重复惩罚
    # ========================================

    def _apply_music_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated_ids: List[int],
        bar_loop_window: int = 4,
        bar_loop_penalty: float = 0.5,
        max_same_chord: int = 2,
        chord_repeat_penalty: float = 0.8,
        pattern_window: int = 32,
        pattern_min_length: int = 8,
        pattern_penalty: float = 0.3,
    ) -> torch.Tensor:
        """
        音乐专用重复惩罚

        针对音乐生成特点设计:
        1. 和弦循环检测: 检测是否陷入 2-4 个和弦的无限循环
        2. 连续和弦惩罚: 避免 Cmaj Cmaj Cmaj 这种无意义重复
        3. 音符模式惩罚: 检测短音符序列的重复
        """
        if len(generated_ids) < 20:
            return logits

        # 1. 和弦循环检测
        bar_penalty_applied = self._detect_bar_loop(
            generated_ids, bar_loop_window, bar_loop_penalty, logits
        )

        # 2. 连续和弦惩罚
        self._penalize_consecutive_chords(
            generated_ids, max_same_chord, chord_repeat_penalty, logits
        )

        # 3. 音符模式惩罚
        if not bar_penalty_applied:
            self._penalize_note_pattern(
                generated_ids, pattern_window, pattern_min_length, pattern_penalty, logits
            )

        return logits

    def _detect_bar_loop(
        self,
        generated_ids: List[int],
        window: int,
        penalty: float,
        logits: torch.Tensor,
    ) -> bool:
        """
        检测和弦级循环

        将序列按和弦切分，检测最近几个和弦段是否形成循环。
        """
        chord_positions = []
        for i, tid in enumerate(generated_ids):
            if self.tokenizer.is_chord_token(tid):
                chord_positions.append(i)

        if len(chord_positions) < window * 2:
            return False

        recent_bars = []
        for i in range(len(chord_positions) - 1):
            start = chord_positions[i]
            end = chord_positions[i + 1]
            bar_content = tuple(generated_ids[start:end])
            recent_bars.append(bar_content)

        if len(recent_bars) < window * 2:
            return False

        recent_bars = recent_bars[-(window * 2):]

        for cycle_len in range(1, window + 1):
            if len(recent_bars) >= cycle_len * 2:
                last_cycle = recent_bars[-cycle_len:]
                prev_cycle = recent_bars[-(cycle_len * 2):-cycle_len]

                if last_cycle == prev_cycle:
                    last_chord_id = generated_ids[chord_positions[-1]]

                    if logits[last_chord_id] != float('-inf'):
                        logits[last_chord_id] = logits[last_chord_id] * penalty

                    for tid in range(self.tokenizer.chord_start_id,
                                     self.tokenizer.chord_end_id + 1):
                        if tid != last_chord_id:
                            logits[tid] = logits[tid] * (1.0 + (1.0 - penalty) * 0.5)

                    return True

        return False

    def _penalize_consecutive_chords(
        self,
        generated_ids: List[int],
        max_same: int,
        penalty: float,
        logits: torch.Tensor,
    ):
        """惩罚连续相同和弦"""
        recent_chords = []
        for tid in reversed(generated_ids):
            if self.tokenizer.is_chord_token(tid):
                recent_chords.append(tid)
                if len(recent_chords) >= max_same + 1:
                    break

        recent_chords = list(reversed(recent_chords))

        if len(recent_chords) >= max_same:
            last_chord = recent_chords[-1]
            same_count = 0
            for chord in reversed(recent_chords):
                if chord == last_chord:
                    same_count += 1
                else:
                    break

            if same_count >= max_same:
                logits[last_chord] = logits[last_chord] * penalty

    def _penalize_note_pattern(
        self,
        generated_ids: List[int],
        window: int,
        min_length: int,
        penalty: float,
        logits: torch.Tensor,
    ):
        """检测并惩罚短音符模式重复"""
        recent = generated_ids[-window:] if len(generated_ids) > window else generated_ids

        note_tokens = []
        for tid in recent:
            token_str = self.tokenizer[tid]
            if (token_str.startswith('T') or token_str.startswith('P') or
                token_str.startswith('L') or token_str.startswith('V')):
                note_tokens.append(tid)

        if len(note_tokens) < min_length * 2:
            return

        for pattern_len in range(min_length, len(note_tokens) // 2 + 1):
            last_pattern = tuple(note_tokens[-pattern_len:])
            prev_pattern = tuple(note_tokens[-(pattern_len * 2):-pattern_len])

            if last_pattern == prev_pattern:
                for tid in set(last_pattern):
                    if logits[tid] != float('-inf'):
                        logits[tid] = logits[tid] * penalty
                break

    # ========================================
    # 通用重复惩罚 (向后兼容)
    # ========================================

    def _apply_repetition_penalties(
        self,
        logits: torch.Tensor,
        generated_ids: List[int],
        token_counts: Dict[int, int],
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ) -> torch.Tensor:
        """通用重复惩罚"""
        if repetition_penalty == 1.0 and frequency_penalty == 0.0 and presence_penalty == 0.0:
            return logits

        seen_tokens = set(generated_ids)

        for token_id in seen_tokens:
            if repetition_penalty != 1.0:
                if logits[token_id] > 0:
                    logits[token_id] = logits[token_id] / repetition_penalty
                else:
                    logits[token_id] = logits[token_id] * repetition_penalty

            if frequency_penalty != 0.0:
                count = token_counts.get(token_id, 0)
                logits[token_id] = logits[token_id] - frequency_penalty * count

            if presence_penalty != 0.0:
                logits[token_id] = logits[token_id] - presence_penalty

        return logits

    def _apply_ngram_blocking(
        self,
        logits: torch.Tensor,
        generated_ids: List[int],
        ngram_size: int,
    ) -> torch.Tensor:
        """N-gram blocking"""
        if ngram_size <= 0 or len(generated_ids) < ngram_size:
            return logits

        ngrams = set()
        for i in range(len(generated_ids) - ngram_size + 1):
            ngram = tuple(generated_ids[i:i + ngram_size])
            ngrams.add(ngram)

        if len(generated_ids) >= ngram_size - 1:
            prefix = tuple(generated_ids[-(ngram_size - 1):])
            for token_id in range(logits.size(0)):
                candidate_ngram = prefix + (token_id,)
                if candidate_ngram in ngrams:
                    logits[token_id] = float('-inf')

        return logits

    # ========================================
    # Prompt 构建
    # ========================================

    def _build_prompt(
        self,
        instruments: Optional[List[int]] = None,
        tempo: int = 120,
        time_signature: str = '4/4',
        chords: Optional[List[str]] = None,
    ) -> List[int]:
        """构建初始 prompt"""
        token_ids = []

        # BOS
        token_ids.append(self.tokenizer.bos_id)

        # Tempo
        bpm_bin = self.tokenizer.quantize_tempo(tempo)
        token_ids.append(self.tokenizer[f'BPM_{bpm_bin}'])

        # Time signature
        ts_token = f'TS_{time_signature}'
        if ts_token in self.tokenizer.token2id:
            token_ids.append(self.tokenizer[ts_token])

        # 第一个和弦
        first_chord = 'Cmaj'
        if chords and len(chords) > 0:
            first_chord = chords[0] if chords[0] in self.tokenizer.token2id else 'Cmaj'

        # 只添加第一个乐器
        if instruments:
            first_inst = instruments[0]
            if first_inst == 128:
                token_ids.append(self.tokenizer['#D'])
            else:
                token_ids.append(self.tokenizer[f'#P{first_inst}'])
        else:
            token_ids.append(self.tokenizer['#P0'])

        token_ids.append(self.tokenizer.sep_id)
        if first_chord in self.tokenizer.token2id:
            token_ids.append(self.tokenizer[first_chord])

        return token_ids

    # ========================================
    # 分析方法
    # ========================================

    @torch.no_grad()
    def compute_perplexity(self, token_ids: List[int]) -> float:
        """计算困惑度 (PPL)"""
        tracker = SequenceTracker(self.tokenizer)
        tracker.extend(token_ids)

        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        chord_ids_t, inst_ids_t, tt_ids_t, note_ids_t = tracker.to_tensors(self.device)

        logits = self.model(
            input_tensor,
            chord_ids=chord_ids_t,
            instrument_ids=inst_ids_t,
            token_type_ids=tt_ids_t,
            note_ids=note_ids_t,
            num_chords=tracker.num_chords,
        )

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_tensor[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='mean'
        )

        return torch.exp(loss).item()

    @torch.no_grad()
    def compute_sequence_confidence(
        self,
        token_ids: List[int],
    ) -> Tuple[float, List[float]]:
        """计算给定序列的置信度"""
        tracker = SequenceTracker(self.tokenizer)
        tracker.extend(token_ids)

        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        chord_ids_t, inst_ids_t, tt_ids_t, note_ids_t = tracker.to_tensors(self.device)

        logits = self.model(
            input_tensor,
            chord_ids=chord_ids_t,
            instrument_ids=inst_ids_t,
            token_type_ids=tt_ids_t,
            note_ids=note_ids_t,
            num_chords=tracker.num_chords,
        )
        probs = F.softmax(logits, dim=-1)

        per_token_confidence = []
        for i in range(1, len(token_ids)):
            conf = probs[0, i - 1, token_ids[i]].item()
            per_token_confidence.append(conf)

        avg_confidence = sum(per_token_confidence) / len(per_token_confidence) \
            if per_token_confidence else 0
        return avg_confidence, per_token_confidence

    def analyze_generation(self, result: GenerationResult, verbose: bool = True) -> Dict:
        """分析生成结果"""
        analysis = {
            'total_tokens': len(result.token_ids),
            'generated_tokens': len(result.steps),
            'avg_confidence': result.avg_confidence,
            'avg_entropy': result.avg_entropy,
            'low_confidence_count': len(result.low_confidence_positions),
            'low_confidence_ratio': (
                len(result.low_confidence_positions) / len(result.steps)
                if result.steps else 0
            ),
        }

        if verbose:
            print("=" * 50)
            print("生成分析报告")
            print("=" * 50)
            print(f"总 token 数: {analysis['total_tokens']}")
            print(f"生成 token 数: {analysis['generated_tokens']}")
            print(f"平均置信度: {analysis['avg_confidence']:.4f}")
            print(f"平均熵: {analysis['avg_entropy']:.4f}")
            print(f"低置信度位置数: {analysis['low_confidence_count']} "
                  f"({analysis['low_confidence_ratio'] * 100:.1f}%)")

        return analysis

    # ========================================
    # MIDI 导出
    # ========================================

    def tokens_to_midi(
        self,
        token_ids: List[int],
        output_path: str,
        ticks_per_beat: int = 480,
    ) -> MidiFile:
        """将 token IDs 转换为 MIDI 文件"""
        midi = MidiFile(ticks_per_beat=ticks_per_beat)

        current_instrument = 0
        chord_tick = 0
        current_position = 0
        tempo_bpm = 120
        time_sig_num = 4
        time_sig_denom = 4

        beats_per_chord = 4
        ticks_per_chord = ticks_per_beat * beats_per_chord
        ticks_per_16th = ticks_per_chord // 16

        current_duration = ticks_per_beat // 2
        current_velocity = 80

        tracks_data: Dict[int, List[Dict]] = {}
        pending_note = None

        def dequantize_velocity(bin_idx: int) -> int:
            return min(127, (bin_idx * 128 + 16) // 32)

        for tid in token_ids:
            token = self.tokenizer[tid]

            if token in ['PAD', 'BOS', 'EOS', 'MASK', 'UNK']:
                continue

            if token.startswith('BPM_'):
                bpm_bin = int(token[4:])
                tempo_bpm = self.tokenizer.dequantize_tempo(bpm_bin)
                continue

            if token.startswith('TS_'):
                ts = token[3:]
                parts = ts.split('/')
                if len(parts) == 2:
                    time_sig_num = int(parts[0])
                    time_sig_denom = int(parts[1])
                continue

            if token.startswith('#P'):
                if pending_note is not None and pending_note['instrument'] in tracks_data:
                    tracks_data[pending_note['instrument']].append({
                        'pitch': pending_note['pitch'],
                        'start': pending_note['start'],
                        'duration': pending_note['duration'],
                        'velocity': pending_note['velocity'],
                    })
                    pending_note = None

                current_instrument = int(token[2:])
                if current_instrument not in tracks_data:
                    tracks_data[current_instrument] = []
                continue

            if token == '#D':
                if pending_note is not None and pending_note['instrument'] in tracks_data:
                    tracks_data[pending_note['instrument']].append({
                        'pitch': pending_note['pitch'],
                        'start': pending_note['start'],
                        'duration': pending_note['duration'],
                        'velocity': pending_note['velocity'],
                    })
                    pending_note = None

                current_instrument = 128
                if current_instrument not in tracks_data:
                    tracks_data[current_instrument] = []
                continue

            if token == 'SEP':
                if pending_note is not None and pending_note['instrument'] in tracks_data:
                    tracks_data[pending_note['instrument']].append({
                        'pitch': pending_note['pitch'],
                        'start': pending_note['start'],
                        'duration': pending_note['duration'],
                        'velocity': pending_note['velocity'],
                    })
                    pending_note = None
                chord_tick = 0
                continue

            if self.tokenizer.is_chord_token(tid):
                if pending_note is not None and pending_note['instrument'] in tracks_data:
                    tracks_data[pending_note['instrument']].append({
                        'pitch': pending_note['pitch'],
                        'start': pending_note['start'],
                        'duration': pending_note['duration'],
                        'velocity': pending_note['velocity'],
                    })
                    pending_note = None
                chord_tick += ticks_per_chord
                current_position = 0
                continue

            if token.startswith('T') and token[1:].isdigit():
                if pending_note is not None and pending_note['instrument'] in tracks_data:
                    tracks_data[pending_note['instrument']].append({
                        'pitch': pending_note['pitch'],
                        'start': pending_note['start'],
                        'duration': pending_note['duration'],
                        'velocity': pending_note['velocity'],
                    })
                    pending_note = None
                current_position = int(token[1:])
                continue

            if token.startswith('P') and token[1:].isdigit():
                if pending_note is not None and pending_note['instrument'] in tracks_data:
                    tracks_data[pending_note['instrument']].append({
                        'pitch': pending_note['pitch'],
                        'start': pending_note['start'],
                        'duration': pending_note['duration'],
                        'velocity': pending_note['velocity'],
                    })

                pitch = int(token[1:])
                pos_tick = chord_tick + (current_position * ticks_per_chord) // 16

                pending_note = {
                    'pitch': pitch,
                    'start': pos_tick,
                    'duration': current_duration,
                    'velocity': current_velocity,
                    'instrument': current_instrument,
                }
                continue

            if token.startswith('L') and token[1:].isdigit():
                dur_units = int(token[1:])
                current_duration = dur_units * ticks_per_16th
                if pending_note is not None:
                    pending_note['duration'] = current_duration
                continue

            if token.startswith('V') and token[1:].isdigit():
                vel_bin = int(token[1:])
                current_velocity = dequantize_velocity(vel_bin)
                if pending_note is not None:
                    pending_note['velocity'] = current_velocity
                continue

        # 保存最后的音符
        if pending_note is not None and pending_note['instrument'] in tracks_data:
            tracks_data[pending_note['instrument']].append({
                'pitch': pending_note['pitch'],
                'start': pending_note['start'],
                'duration': pending_note['duration'],
                'velocity': pending_note['velocity'],
            })

        # 创建 MIDI tracks
        meta_track = MidiTrack()
        midi.tracks.append(meta_track)

        tempo_us = int(60_000_000 / tempo_bpm)
        meta_track.append(MetaMessage('set_tempo', tempo=tempo_us, time=0))
        meta_track.append(MetaMessage(
            'time_signature',
            numerator=time_sig_num,
            denominator=time_sig_denom,
            time=0
        ))

        for inst_id, notes in tracks_data.items():
            if not notes:
                continue

            track = MidiTrack()
            midi.tracks.append(track)

            channel = 9 if inst_id == 128 else min(inst_id, 15)
            if inst_id != 128:
                track.append(Message('program_change', program=inst_id,
                                     channel=channel, time=0))

            notes.sort(key=lambda x: x['start'])

            events = []
            for note in notes:
                events.append({
                    'time': note['start'],
                    'type': 'note_on',
                    'note': note['pitch'],
                    'velocity': note['velocity'],
                    'channel': channel,
                })
                events.append({
                    'time': note['start'] + note['duration'],
                    'type': 'note_off',
                    'note': note['pitch'],
                    'velocity': 0,
                    'channel': channel,
                })

            events.sort(key=lambda x: (x['time'], x['type'] == 'note_on'))

            current_time = 0
            for event in events:
                delta = event['time'] - current_time
                current_time = event['time']

                if event['type'] == 'note_on':
                    track.append(Message(
                        'note_on',
                        note=event['note'],
                        velocity=event['velocity'],
                        channel=event['channel'],
                        time=delta
                    ))
                else:
                    track.append(Message(
                        'note_off',
                        note=event['note'],
                        velocity=0,
                        channel=event['channel'],
                        time=delta
                    ))

            track.append(MetaMessage('end_of_track', time=0))

        midi.save(output_path)
        print(f"MIDI 文件已保存: {output_path}")

        return midi


# ============================================================
# 辅助函数
# ============================================================

def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[MeloFormer, HIDTokenizerV2]:
    """
    加载模型和 tokenizer

    Args:
        checkpoint_path: 检查点路径
        device: 设备

    Returns:
        (model, tokenizer)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint.get('args', {})

    tokenizer = HIDTokenizerV2()

    # 从 checkpoint 读取 vocab_size 和模型结构
    state_dict = checkpoint['model_state_dict']
    checkpoint_vocab_size = state_dict['token_embedding.weight'].shape[0]
    checkpoint_dim = state_dict['layers.0.reg_attn_norm.weight'].shape[0]

    # 根据 dim 推断 model_size
    dim_to_size = {
        256: '8m', 512: '62m', 768: '177m', 1024: '366m',
        1280: '600m', 1408: '800m', 1536: '1.5b',
    }
    inferred_size = dim_to_size.get(checkpoint_dim, '62m')
    model_size = args.get('model_size', inferred_size)

    model = create_model(
        vocab_size=checkpoint_vocab_size,
        model_size=model_size,
        max_seq_len=args.get('max_seq_len', 24576),
    )

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()

    print(f"  模型: MeloFormer v2.1 {model_size}, vocab_size={checkpoint_vocab_size}")

    return model, tokenizer


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='MeloFormer v2.1 音乐生成')

    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')

    # 生成参数
    parser.add_argument('--max_length', type=int, default=2048, help='最大生成长度')
    parser.add_argument('--temperature', type=float, default=1.0, help='采样温度')
    parser.add_argument('--top_k', type=int, default=50, help='Top-K 采样')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-P 采样')

    # 音乐专用重复惩罚 (默认开启)
    parser.add_argument('--use_music_penalty', action='store_true', default=True,
                        help='使用音乐专用惩罚 (默认开启)')
    parser.add_argument('--no_music_penalty', dest='use_music_penalty', action='store_false',
                        help='禁用音乐专用惩罚，使用通用惩罚')
    parser.add_argument('--bar_loop_penalty', type=float, default=0.5,
                        help='和弦循环惩罚强度 (0-1, 越小越强)')
    parser.add_argument('--max_same_chord', type=int, default=2,
                        help='允许连续相同和弦的最大数量')
    parser.add_argument('--pattern_penalty', type=float, default=0.3,
                        help='音符模式重复惩罚 (0-1)')

    # 通用重复惩罚 (--no_music_penalty 时生效)
    parser.add_argument('--repetition_penalty', type=float, default=1.2,
                        help='通用重复惩罚 (需要 --no_music_penalty)')
    parser.add_argument('--frequency_penalty', type=float, default=0.0,
                        help='通用频率惩罚')
    parser.add_argument('--presence_penalty', type=float, default=0.0,
                        help='通用存在惩罚')
    parser.add_argument('--no_repeat_ngram_size', type=int, default=0,
                        help='禁止重复的 n-gram 大小')

    # 音乐参数
    parser.add_argument('--tempo', type=int, default=120, help='BPM')
    parser.add_argument('--time_signature', type=str, default='4/4', help='拍号')
    parser.add_argument('--instruments', type=int, nargs='+', default=[0],
                        help='乐器列表 (MIDI program numbers, 128=drums)')
    parser.add_argument('--chords', type=str, nargs='+', default=None,
                        help='和弦进行 (如: Cmaj Amin Fmaj Gmaj)')
    parser.add_argument('--force_multi_track', action='store_true',
                        help='强制多轨生成')
    parser.add_argument('--force_chords', action='store_true',
                        help='强制按照指定和弦进行')
    parser.add_argument('--min_notes_per_track', type=int, default=50,
                        help='强制多轨时每个乐器的最小音符数')

    # 输出
    parser.add_argument('--output', type=str, default='./generated.mid', help='输出文件路径')
    parser.add_argument('--num_samples', type=int, default=1, help='生成样本数')
    parser.add_argument('--with_confidence', action='store_true',
                        help='生成时输出置信度分析')

    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--verbose', action='store_true', help='显示详细信息')

    return parser.parse_args()


def main():
    args = parse_args()

    print("MeloFormer v2.1 音乐生成")
    print("=" * 50)

    device = torch.device(args.device)
    print(f"设备: {device}")

    print(f"加载模型: {args.checkpoint}")
    model, tokenizer = load_model(args.checkpoint, device)

    generator = MusicGenerator(model, tokenizer, device)

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 打印惩罚设置
    print(f"\n惩罚设置:")
    if args.use_music_penalty:
        print(f"  模式: 音乐专用惩罚")
        print(f"  bar_loop_penalty: {args.bar_loop_penalty}")
        print(f"  max_same_chord: {args.max_same_chord}")
        print(f"  pattern_penalty: {args.pattern_penalty}")
    else:
        print(f"  模式: 通用惩罚")
        print(f"  repetition_penalty: {args.repetition_penalty}")
        print(f"  frequency_penalty: {args.frequency_penalty}")
        print(f"  no_repeat_ngram_size: {args.no_repeat_ngram_size}")

    for i in range(args.num_samples):
        print(f"\n生成样本 {i + 1}/{args.num_samples}...")

        if args.with_confidence:
            result = generator.generate_with_confidence(
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                frequency_penalty=args.frequency_penalty,
                presence_penalty=args.presence_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                instruments=args.instruments,
                tempo=args.tempo,
                time_signature=args.time_signature,
                chords=args.chords,
            )
            token_ids = result.token_ids
            generator.analyze_generation(result)
        else:
            meta = generator.generate(
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                use_music_penalty=args.use_music_penalty,
                bar_loop_penalty=args.bar_loop_penalty,
                max_same_chord=args.max_same_chord,
                pattern_penalty=args.pattern_penalty,
                repetition_penalty=args.repetition_penalty,
                frequency_penalty=args.frequency_penalty,
                presence_penalty=args.presence_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                instruments=args.instruments,
                tempo=args.tempo,
                time_signature=args.time_signature,
                chords=args.chords,
                force_multi_track=args.force_multi_track,
                force_chords=args.force_chords,
                min_notes_per_track=args.min_notes_per_track,
                verbose=args.verbose,
            )
            token_ids = meta.token_ids

        print(f"  生成了 {meta.n_tokens} tokens | {meta.tokens_per_sec:.1f} tok/s | "
              f"{meta.elapsed_ms:.0f} ms | 峰值显存 {meta.peak_mem_gb:.2f} GB")

        output_path = args.output if args.num_samples == 1 else \
            str(Path(args.output).parent /
                f"{Path(args.output).stem}_{i + 1}{Path(args.output).suffix}")

        generator.tokens_to_midi(token_ids, output_path)

    print("\n完成!")


if __name__ == '__main__':
    main()
