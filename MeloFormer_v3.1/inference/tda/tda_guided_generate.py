#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TDA 引导的音乐生成器 - MeloFormer v2.1

基于 Gong & Huang 2025 方法，在生成过程中使用拓扑数据分析 (TDA)
来引导生成，使生成的音乐具有期望的重复结构和拓扑特性。

核心思想:
    每一步采样: adjusted_logits = logits + lambda * TDA_bonus

与 MIDILM TDA 引导的区别:
    - 使用 MeloFormer v2.1 的 SequenceTracker 维护辅助 ID
    - 每次 forward 传入完整序列 (无 KV cache)
    - Summary Token 自动聚合跨 chord 拓扑信息

使用:
    python -m inference.tda.tda_guided_generate \
        --checkpoint checkpoints/best.pt \
        --output tda_output.mid \
        --lambda_tda 0.3

    # 分阶段生成
    python -m inference.tda.tda_guided_generate \
        --checkpoint checkpoints/best.pt \
        --staged \
        --lambda_tda 0.3
"""

import argparse
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import numpy as np

# 添加项目根目录
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model import MeloFormer, create_model
from data import HIDTokenizerV2
from inference.generator import MusicGenerator, SequenceTracker, load_model
from .topology_computer import TopologyComputer, PersistenceDiagram
from .feature_extractor import MusicFeatureExtractor, BarFeature


@dataclass
class TDAConfig:
    """TDA 引导配置"""
    # 引导强度
    lambda_tda: float = 0.3           # TDA 奖励权重

    # 计算效率
    eval_every_n_tokens: int = 10     # 每 N 个 token 评估一次 TDA
    top_k_candidates: int = 20        # 评估 top-k 候选 token
    min_bars_for_tda: int = 4         # 最少需要多少小节才启用 TDA

    # 目标拓扑
    target_beta0: int = 1             # 目标连通分量数 (通常为 1)
    target_beta1: int = 2             # 目标循环结构数
    target_persistence: float = 0.5   # 目标持续度


class TDACache:
    """TDA 计算缓存"""

    def __init__(self):
        self.bars: List[BarFeature] = []
        self.point_cloud: Optional[np.ndarray] = None
        self.diagrams: Optional[List[PersistenceDiagram]] = None
        self.last_bar_index: int = -1

    def update(
        self,
        new_bars: List[BarFeature],
        topology_computer: TopologyComputer
    ):
        """增量更新缓存"""
        if not new_bars:
            return

        for bar in new_bars:
            if bar.bar_index > self.last_bar_index:
                self.bars.append(bar)
                self.last_bar_index = bar.bar_index

        if len(self.bars) >= 3:
            self.point_cloud = np.array([b.to_vector() for b in self.bars])
            self.diagrams = topology_computer.compute_persistence(self.point_cloud)

    def reset(self):
        """重置缓存"""
        self.bars = []
        self.point_cloud = None
        self.diagrams = None
        self.last_bar_index = -1


class TDAGuidedGenerator:
    """
    TDA 引导的音乐生成器 (MeloFormer v2.1)

    在自回归生成过程中，使用拓扑数据分析来引导生成，
    使生成的音乐具有期望的重复结构和拓扑特性。
    """

    def __init__(
        self,
        model: MeloFormer,
        tokenizer: HIDTokenizerV2,
        device: torch.device,
        config: Optional[TDAConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config or TDAConfig()

        # 初始化组件
        self.generator = MusicGenerator(model, tokenizer, device)
        self.topology_computer = TopologyComputer(max_dimension=1)
        self.feature_extractor = MusicFeatureExtractor(tokenizer)

        # 缓存
        self.tda_cache = TDACache()

        # 目标拓扑 (可从参考 MIDI 设置)
        self._target_diagrams: Optional[List[PersistenceDiagram]] = None

    def set_reference_midi(self, reference_tokens: List[int]):
        """设置参考 MIDI 作为目标拓扑"""
        bars = self.feature_extractor.extract_bar_features(reference_tokens)
        if len(bars) >= 3:
            point_cloud = self.feature_extractor.bars_to_point_cloud(bars)
            self._target_diagrams = self.topology_computer.compute_persistence(point_cloud)
            print(f"已设置参考拓扑: {len(bars)} 小节")

    @torch.no_grad()
    def generate_with_tda(
        self,
        prompt_ids: Optional[List[int]] = None,
        max_length: int = 2048,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        # TDA 参数
        lambda_tda: Optional[float] = None,
        # 音乐参数
        instruments: Optional[List[int]] = None,
        tempo: int = 120,
        time_signature: str = '4/4',
        chords: Optional[List[str]] = None,
        force_chords: bool = False,
        use_music_penalty: bool = True,
        verbose: bool = False,
    ) -> List[int]:
        """
        TDA 引导的音乐生成

        Args:
            prompt_ids: 初始 prompt token IDs
            max_length: 最大生成长度
            temperature: 采样温度
            top_k: Top-K 采样
            top_p: Top-P (Nucleus) 采样
            lambda_tda: TDA 引导强度 (覆盖 config)
            instruments: 乐器列表 (MIDI program numbers)
            tempo: BPM
            time_signature: 拍号
            chords: 和弦进行列表
            force_chords: 强制按照指定和弦进行循环生成
            use_music_penalty: 使用音乐专用重复惩罚
            verbose: 显示详细信息

        Returns:
            生成的 token ID 列表
        """
        lambda_tda = lambda_tda if lambda_tda is not None else self.config.lambda_tda
        eval_every_n = self.config.eval_every_n_tokens

        # 重置缓存
        self.tda_cache.reset()

        # 构建初始 prompt
        if prompt_ids is None:
            prompt_ids = self._build_prompt(instruments, tempo, time_signature, chords)

        # 强制和弦相关
        forced_chord_ids = []
        chord_index = 0
        if force_chords and chords:
            for chord in chords:
                if chord in self.tokenizer.token2id:
                    forced_chord_ids.append(self.tokenizer[chord])
            chord_index = 1 if len(forced_chord_ids) > 1 else 0

        generated_ids = list(prompt_ids)

        # 初始化 SequenceTracker
        tracker = SequenceTracker(self.tokenizer)
        tracker.extend(prompt_ids)

        # 统计
        tda_evals = 0
        tda_adjustments = 0

        if verbose:
            print(f"开始 TDA 引导生成 (lambda={lambda_tda})")
            print(f"初始 prompt: {len(prompt_ids)} tokens")

        # 生成循环
        for step in range(max_length - len(prompt_ids)):
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

            next_logits = logits[0, -1, :].clone()

            # 应用音乐专用重复惩罚
            if use_music_penalty:
                next_logits = self._apply_music_penalty(next_logits, generated_ids)

            # ========================================
            # TDA 引导: 每隔 N 步评估
            # ========================================
            num_bars = self._count_bars(generated_ids)
            should_eval_tda = (
                step > 0 and
                step % eval_every_n == 0 and
                num_bars >= self.config.min_bars_for_tda
            )

            if should_eval_tda:
                tda_evals += 1

                # 更新 TDA 缓存
                bars = self.feature_extractor.extract_bar_features(generated_ids)
                self.tda_cache.update(bars, self.topology_computer)

                # 计算 TDA 奖励
                tda_bonus = self._compute_tda_bonus(next_logits, generated_ids)

                if tda_bonus is not None:
                    next_logits = next_logits + lambda_tda * tda_bonus
                    tda_adjustments += 1

                    if verbose and tda_adjustments % 10 == 1:
                        if self.tda_cache.diagrams:
                            betti = self.topology_computer.compute_betti_numbers(
                                self.tda_cache.diagrams
                            )
                            print(f"  [Step {step}] TDA: beta_0={betti[0]}, "
                                  f"beta_1={betti[1]}, bars={num_bars}")

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

            token_id = next_token.item()

            # 强制和弦替换
            if force_chords and forced_chord_ids and self.tokenizer.is_chord_token(token_id):
                token_id = forced_chord_ids[chord_index % len(forced_chord_ids)]
                chord_index += 1

            generated_ids.append(token_id)
            tracker.append(token_id)

            # 检查 EOS
            if token_id == self.tokenizer.eos_id:
                break

            # 进度显示
            if verbose and step > 0 and step % 200 == 0:
                print(f"  已生成 {step} tokens, TDA evals: {tda_evals}, "
                      f"adjustments: {tda_adjustments}")

        if verbose:
            print(f"\n生成完成: {len(generated_ids)} tokens")
            print(f"TDA 评估次数: {tda_evals}, 调整次数: {tda_adjustments}")

            # 最终拓扑分析
            final_bars = self.feature_extractor.extract_bar_features(generated_ids)
            if len(final_bars) >= 3:
                final_cloud = self.feature_extractor.bars_to_point_cloud(final_bars)
                final_diags = self.topology_computer.compute_persistence(final_cloud)
                betti = self.topology_computer.compute_betti_numbers(final_diags)
                print(f"最终拓扑: beta_0={betti[0]}, beta_1={betti[1]}, "
                      f"小节数={len(final_bars)}")

        return generated_ids

    def _compute_tda_bonus(
        self,
        logits: torch.Tensor,
        history: List[int],
    ) -> Optional[torch.Tensor]:
        """
        计算 TDA 奖励向量

        对 top-k 候选 token，模拟添加后的拓扑变化，
        计算与目标拓扑的相似度得分作为奖励。

        Returns:
            shape: (vocab_size,) 奖励向量，或 None (如果无法计算)
        """
        if self.tda_cache.diagrams is None:
            return None

        # 获取 top-k 候选
        _, top_indices = torch.topk(logits, self.config.top_k_candidates)
        candidates = top_indices.tolist()

        # 当前拓扑得分
        current_score = self._compute_topology_score(self.tda_cache.diagrams)

        # 计算每个候选的 TDA 奖励
        bonus = torch.zeros_like(logits)

        for token_id in candidates:
            # 只有和弦 token 会显著影响拓扑
            if self.tokenizer.is_chord_token(token_id):
                simulated_ids = history + [token_id]
                simulated_bars = self.feature_extractor.extract_bar_features(simulated_ids)

                if len(simulated_bars) > len(self.tda_cache.bars):
                    point_cloud = self.feature_extractor.bars_to_point_cloud(simulated_bars)

                    if len(point_cloud) >= 3:
                        diagrams = self.topology_computer.compute_persistence(point_cloud)
                        new_score = self._compute_topology_score(diagrams)

                        delta_score = new_score - current_score
                        scale = logits.abs().max().item()
                        bonus[token_id] = delta_score * scale

        return bonus

    def _compute_topology_score(
        self,
        diagrams: List[PersistenceDiagram]
    ) -> float:
        """
        计算拓扑得分 (与目标拓扑的相似度)

        Returns:
            得分 (0-1)，越高越好
        """
        score = 0.0

        betti = self.topology_computer.compute_betti_numbers(diagrams)

        # 1. beta_0 得分 (连通性)
        beta0_diff = abs(betti[0] - self.config.target_beta0)
        beta0_score = max(0, 1 - beta0_diff / max(self.config.target_beta0, 1))
        score += 0.2 * beta0_score

        # 2. beta_1 得分 (循环结构) -- 最重要
        beta1_diff = abs(betti[1] - self.config.target_beta1)
        beta1_score = max(0, 1 - beta1_diff / max(self.config.target_beta1, 1))
        score += 0.5 * beta1_score

        # 3. 持续度得分
        if len(diagrams) > 1:
            h1_persistence = diagrams[1].total_persistence
            persistence_score = min(h1_persistence / self.config.target_persistence, 1.0)
            score += 0.3 * persistence_score

        # 4. 如果有参考目标，计算 Wasserstein 距离
        if self._target_diagrams and len(diagrams) > 1:
            dist = self.topology_computer.compute_distance(
                diagrams[1],
                self._target_diagrams[1]
            )
            dist_score = np.exp(-dist)
            score = 0.5 * score + 0.5 * dist_score

        return score

    def _apply_music_penalty(
        self,
        logits: torch.Tensor,
        history: List[int]
    ) -> torch.Tensor:
        """应用音乐专用重复惩罚"""
        return self.generator._apply_music_repetition_penalty(
            logits, history,
            bar_loop_penalty=0.5,
            max_same_chord=2,
            pattern_penalty=0.3,
        )

    def _count_bars(self, token_ids: List[int]) -> int:
        """统计和弦/小节数量"""
        return sum(1 for tid in token_ids if self.tokenizer.is_chord_token(tid))

    def _build_prompt(
        self,
        instruments: Optional[List[int]] = None,
        tempo: int = 120,
        time_signature: str = '4/4',
        chords: Optional[List[str]] = None,
    ) -> List[int]:
        """构建初始 prompt"""
        if instruments is None:
            instruments = [0]
        if chords is None:
            chords = ['Cmaj']

        prompt = [self.tokenizer.bos_id]

        # Tempo
        tempo_bin = self.tokenizer.quantize_tempo(tempo)
        prompt.append(self.tokenizer[f'BPM_{tempo_bin}'])

        # Time signature
        ts_token = f'TS_{time_signature}'
        if ts_token in self.tokenizer.token2id:
            prompt.append(self.tokenizer[ts_token])

        # Instruments
        for inst in instruments:
            if inst == 128:
                prompt.append(self.tokenizer['#D'])
            else:
                prompt.append(self.tokenizer[f'#P{inst}'])

        # SEP
        prompt.append(self.tokenizer.sep_id)

        # First chord
        if chords[0] in self.tokenizer.token2id:
            prompt.append(self.tokenizer[chords[0]])

        return prompt

    def tokens_to_midi(self, token_ids: List[int], output_path: str):
        """导出 MIDI 文件"""
        self.generator.tokens_to_midi(token_ids, output_path)


# ============================================================
# 分阶段 TDA 生成
# ============================================================

def generate_staged_with_tda(
    checkpoint_path: str,
    output_path: str,
    instruments: Optional[Dict[str, int]] = None,
    chords: Optional[List[str]] = None,
    tempo: int = 120,
    lambda_tda: float = 0.3,
    target_beta1: int = 2,
    max_per_stage: int = 1024,
    temperature: float = 1.1,
    top_k: int = 80,
    top_p: float = 0.95,
    verbose: bool = True,
) -> List[int]:
    """
    TDA 引导的分阶段生成

    流程:
    1. 鼓阶段: 基础 TDA 引导 (建立节奏骨架)
    2. 贝斯阶段: TDA 引导 (确保低音与鼓配合)
    3. 钢琴阶段: TDA 引导 (建立和声)
    4. 吉他阶段: TDA 引导 (丰富织体)
    """
    if instruments is None:
        instruments = {
            'drum': 128,
            'bass': 33,
            'piano': 0,
            'guitar': 25,
        }

    if chords is None:
        chords = ['Cmaj', 'Amin', 'Fmaj', 'Gmaj']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if verbose:
        print("=" * 70)
        print("MeloFormer v2.1 TDA 引导分阶段生成")
        print("=" * 70)
        print(f"设备: {device}")
        print(f"TDA lambda: {lambda_tda}")
        print(f"目标 beta_1: {target_beta1}")
        print(f"和弦进行: {' -> '.join(chords)}")
        print()

    # 加载模型
    model, tokenizer = load_model(checkpoint_path, device)

    config = TDAConfig(
        lambda_tda=lambda_tda,
        target_beta1=target_beta1,
        eval_every_n_tokens=10,
        min_bars_for_tda=4,
    )

    generator = TDAGuidedGenerator(model, tokenizer, device, config)

    gen_params = {
        'temperature': temperature,
        'top_k': top_k,
        'top_p': top_p,
        'use_music_penalty': True,
        'lambda_tda': lambda_tda,
    }

    note_counts = {}
    instrument_order = list(instruments.items())

    # 阶段 1: 第一个乐器
    stage_name, stage_inst = instrument_order[0]
    if verbose:
        print(f"[阶段 1/{len(instrument_order)}] 生成 {stage_name} 轨...")

    current_ids = generator.generate_with_tda(
        instruments=[stage_inst],
        tempo=tempo,
        chords=chords,
        max_length=max_per_stage,
        verbose=verbose,
        **gen_params
    )
    note_counts[stage_name] = _count_notes(current_ids, tokenizer)

    if verbose:
        print(f"  -> {stage_name}: {len(current_ids)} tokens, "
              f"{note_counts[stage_name]} 音符\n")

    # 后续阶段
    for stage_idx in range(1, len(instrument_order)):
        stage_name, stage_inst = instrument_order[stage_idx]
        if verbose:
            print(f"[阶段 {stage_idx + 1}/{len(instrument_order)}] "
                  f"续写 {stage_name}...")

        prev_len = len(current_ids)

        prompt = current_ids.copy()
        if stage_inst == 128:
            prompt.append(tokenizer['#D'])
        else:
            prompt.append(tokenizer[f'#P{stage_inst}'])
        prompt.append(tokenizer.sep_id)
        prompt.append(tokenizer[chords[0]])

        current_ids = generator.generate_with_tda(
            prompt_ids=prompt,
            max_length=len(prompt) + max_per_stage,
            verbose=verbose,
            **gen_params
        )
        note_counts[stage_name] = _count_notes(current_ids[prev_len:], tokenizer)

        if verbose:
            print(f"  -> {stage_name}: +{len(current_ids) - prev_len} tokens, "
                  f"{note_counts[stage_name]} 音符\n")

    # 保存
    generator.tokens_to_midi(current_ids, output_path)

    if verbose:
        total_notes = sum(note_counts.values())
        print("=" * 70)
        print(f"生成完成!")
        print(f"保存位置: {output_path}")
        print(f"总长度: {len(current_ids)} tokens")
        note_summary = ' + '.join(f'{k}{v}' for k, v in note_counts.items())
        print(f"音符统计: {note_summary} = {total_notes}")
        print("=" * 70)

    return current_ids


def _count_notes(token_ids: List[int], tokenizer) -> int:
    """统计音符数量"""
    count = 0
    for tid in token_ids:
        token = tokenizer[tid]
        if token.startswith('P') and len(token) > 1 and token[1:].isdigit():
            count += 1
    return count


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='MeloFormer v2.1 TDA 引导的音乐生成',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础 TDA 引导生成
  python -m inference.tda.tda_guided_generate \\
      --checkpoint checkpoints/best.pt \\
      --output tda_output.mid \\
      --lambda_tda 0.3

  # 分阶段 TDA 生成
  python -m inference.tda.tda_guided_generate \\
      --checkpoint checkpoints/best.pt \\
      --staged \\
      --lambda_tda 0.3 \\
      --target_beta1 2
        """
    )

    # 模型参数
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型 checkpoint 路径')
    parser.add_argument('--output', type=str, default='./tda_generated.mid',
                        help='输出 MIDI 文件路径')

    # TDA 参数
    parser.add_argument('--lambda_tda', type=float, default=0.3,
                        help='TDA 引导强度 (0-1)')
    parser.add_argument('--target_beta1', type=int, default=2,
                        help='目标循环结构数 (beta_1)')
    parser.add_argument('--eval_every', type=int, default=10,
                        help='每隔多少 token 评估 TDA')

    # 生成参数
    parser.add_argument('--max_length', type=int, default=2048,
                        help='最大生成长度')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='采样温度')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-K 采样')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-P (Nucleus) 采样')

    # 音乐参数
    parser.add_argument('--tempo', type=int, default=120,
                        help='BPM')
    parser.add_argument('--instruments', type=int, nargs='+', default=[0],
                        help='乐器列表 (MIDI program numbers)')
    parser.add_argument('--chords', type=str, nargs='+',
                        default=['Cmaj', 'Amin', 'Fmaj', 'Gmaj'],
                        help='和弦进行')

    # 分阶段生成
    parser.add_argument('--staged', action='store_true',
                        help='使用分阶段生成模式')

    parser.add_argument('--verbose', action='store_true',
                        help='显示详细信息')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.staged:
        generate_staged_with_tda(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            chords=args.chords,
            tempo=args.tempo,
            lambda_tda=args.lambda_tda,
            target_beta1=args.target_beta1,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            verbose=args.verbose,
        )
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("=" * 60)
        print("MeloFormer v2.1 TDA 引导音乐生成")
        print("=" * 60)
        print(f"设备: {device}")
        print(f"TDA lambda: {args.lambda_tda}")
        print(f"目标 beta_1: {args.target_beta1}")
        print()

        model, tokenizer = load_model(args.checkpoint, device)

        config = TDAConfig(
            lambda_tda=args.lambda_tda,
            target_beta1=args.target_beta1,
            eval_every_n_tokens=args.eval_every,
        )

        generator = TDAGuidedGenerator(model, tokenizer, device, config)

        token_ids = generator.generate_with_tda(
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            instruments=args.instruments,
            tempo=args.tempo,
            chords=args.chords,
            verbose=args.verbose,
        )

        generator.tokens_to_midi(token_ids, args.output)
        print(f"\n完成! 输出: {args.output}")


if __name__ == '__main__':
    main()
