#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深入训练准备检查脚本

对训练流程进行更加细致的检查，确保万无一失
"""

import os
import sys
import time
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import Counter, defaultdict
import traceback
import json
import random
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from data.tokenizer_v2 import HIDTokenizerV2, TokenInfo
from data.dataset import HIDMusicDataset, collate_fn
from model.attention import (
    FCAttentionMask, MultiHeadFCAttention, FCAttentionBlock,
    SummaryAttention, SummaryAttentionMask, SummaryAttentionBlock,
    SummaryTokenEmbedding, RotaryPositionEmbedding
)
from model.hid_museformer import HIDMuseFormer, create_model


@dataclass
class DeepCheckResult:
    """深入检查结果"""
    name: str
    passed: bool
    message: str
    severity: str = "info"  # info, warning, error, critical
    details: Optional[Dict] = None


class DeepTrainingChecker:
    """深入训练检查器"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[DeepCheckResult] = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    def add_result(self, result: DeepCheckResult):
        self.results.append(result)
        icons = {"info": "ℹ", "warning": "⚠", "error": "✗", "critical": "💀"}
        status = "✓" if result.passed else icons.get(result.severity, "✗")
        self.log(f"  [{status}] {result.name}: {result.message}")

    # ==================== 1. Tokenizer 边界检查 ====================

    def check_tokenizer_edge_cases(self) -> List[DeepCheckResult]:
        """检查 Tokenizer 边界情况"""
        self.log("\n" + "=" * 60)
        self.log("1. TOKENIZER 边界情况检查")
        self.log("=" * 60)

        results = []
        tokenizer = HIDTokenizerV2()

        # 1.1 空输入
        try:
            ids, infos = tokenizer.encode_with_info("", add_special=True)
            has_bos_eos = len(ids) >= 2 and ids[0] == tokenizer.bos_id and ids[-1] == tokenizer.eos_id
            results.append(DeepCheckResult(
                name="空输入处理",
                passed=has_bos_eos,
                message=f"输出 {len(ids)} tokens" + (" (BOS+EOS)" if has_bos_eos else " (缺少特殊token)"),
            ))
        except Exception as e:
            results.append(DeepCheckResult(
                name="空输入处理",
                passed=False,
                message=f"异常: {str(e)}",
                severity="error",
            ))
        self.add_result(results[-1])

        # 1.2 极端音高值
        extreme_pitches = [
            ("P0", "最低音高"),
            ("P127", "最高音高"),
            ("P60", "中央C"),
        ]
        for pitch, desc in extreme_pitches:
            test_content = f"#P0\nB0 T0 {pitch} L4 V16"
            ids, infos = tokenizer.encode_with_info(test_content)
            pitch_found = any(info.token_str == pitch for info in infos)
            results.append(DeepCheckResult(
                name=f"音高边界 ({desc})",
                passed=pitch_found,
                message=f"{pitch} 正确编码" if pitch_found else f"{pitch} 编码失败",
            ))
            self.add_result(results[-1])

        # 1.3 极端持续时间
        extreme_durations = [("L1", "最短"), ("L64", "最长"), ("L32", "中等")]
        for dur, desc in extreme_durations:
            test_content = f"#P0\nB0 T0 P60 {dur} V16"
            ids, infos = tokenizer.encode_with_info(test_content)
            dur_found = any(info.token_str == dur for info in infos)
            results.append(DeepCheckResult(
                name=f"持续时间边界 ({desc})",
                passed=dur_found,
                message=f"{dur} 正确编码" if dur_found else f"{dur} 编码失败",
            ))
            self.add_result(results[-1])

        # 1.4 极端力度
        extreme_velocities = [("V0", "最弱"), ("V31", "最强"), ("V16", "中等")]
        for vel, desc in extreme_velocities:
            test_content = f"#P0\nB0 T0 P60 L4 {vel}"
            ids, infos = tokenizer.encode_with_info(test_content)
            vel_found = any(info.token_str == vel for info in infos)
            results.append(DeepCheckResult(
                name=f"力度边界 ({desc})",
                passed=vel_found,
                message=f"{vel} 正确编码" if vel_found else f"{vel} 编码失败",
            ))
            self.add_result(results[-1])

        # 1.5 所有乐器 token
        instrument_tests = [
            ("#D", "鼓"),
            ("#P0", "钢琴"),
            ("#P127", "音效"),
            ("#P24", "吉他"),
        ]
        for inst, desc in instrument_tests:
            test_content = f"{inst}\nB0 T0 P60 L4 V16"
            ids, infos = tokenizer.encode_with_info(test_content)
            inst_found = any(info.token_str == inst for info in infos)
            results.append(DeepCheckResult(
                name=f"乐器 ({desc})",
                passed=inst_found,
                message=f"{inst} 正确编码" if inst_found else f"{inst} 编码失败",
            ))
            self.add_result(results[-1])

        # 1.6 所有位置 token (T0-T15)
        position_errors = []
        for i in range(16):
            pos_token = f"T{i}"
            test_content = f"#P0\nB0 {pos_token} P60 L4 V16"
            ids, infos = tokenizer.encode_with_info(test_content)
            if not any(info.token_str == pos_token for info in infos):
                position_errors.append(pos_token)

        results.append(DeepCheckResult(
            name="位置 tokens (T0-T15)",
            passed=len(position_errors) == 0,
            message="全部 16 个正确" if not position_errors else f"失败: {position_errors}",
        ))
        self.add_result(results[-1])

        # 1.7 token_type 分配一致性检查
        test_content = """#P0
B0 T0 P60 L4 V16
T4 P64 L4 V16
T8 P67 L4 V16
B1 T0 P72 L8 V20
"""
        ids, infos = tokenizer.encode_with_info(test_content)

        # 检查每个音符的 token_type 序列是否合理 (T -> P -> L -> V)
        type_sequences = []
        current_seq = []
        for info in infos:
            if info.token_type == 0:  # T
                if current_seq:
                    type_sequences.append(current_seq)
                current_seq = [0]
            elif info.token_type in [1, 2, 3]:
                current_seq.append(info.token_type)

        if current_seq:
            type_sequences.append(current_seq)

        # 检查每个序列是否为 [0, 1, 2, 3] 或其子序列
        valid_sequences = all(
            seq == [0, 1, 2, 3] or seq == [0, 1] or seq == [0, 1, 2] or seq == [0, 1, 3]
            for seq in type_sequences
        )

        results.append(DeepCheckResult(
            name="Token 类型序列一致性",
            passed=valid_sequences,
            message=f"检查 {len(type_sequences)} 个音符序列" + (" 正确" if valid_sequences else " 有异常"),
            details={"sequences": type_sequences[:5]},
        ))
        self.add_result(results[-1])

        # 1.8 note_id 连续性检查
        note_ids = [info.note_id for info in infos if info.note_id > 0]
        if note_ids:
            note_id_set = set(note_ids)
            expected_ids = set(range(1, max(note_ids) + 1))
            ids_match = note_id_set == expected_ids
            results.append(DeepCheckResult(
                name="Note ID 连续性",
                passed=ids_match,
                message=f"ID 范围 1-{max(note_ids)}" + (" 连续" if ids_match else " 不连续"),
            ))
            self.add_result(results[-1])

        # 1.9 和弦 token 检查
        chord_content = """H 4 384 Q
T B0 T0 120
TS B0 T0 4/4
CHORDS 0:Cmaj 1:Amin 2:Fmaj 3:G7

#P0
B0 T0 P60 L4 V16
B1 T0 P57 L4 V16
B2 T0 P65 L4 V16
B3 T0 P55 L4 V16
"""
        ids, infos = tokenizer.encode_with_info(chord_content)
        chord_infos = [info for info in infos if info.is_chord]
        expected_chords = ['Cmaj', 'Amin', 'Fmaj', 'G7']
        found_chords = [info.token_str for info in chord_infos]

        chords_match = all(c in found_chords for c in expected_chords)
        results.append(DeepCheckResult(
            name="和弦 token 解析",
            passed=chords_match,
            message=f"找到 {len(chord_infos)} 个和弦: {found_chords[:4]}",
        ))
        self.add_result(results[-1])

        return results

    # ==================== 2. 数据分布检查 ====================

    def check_data_distribution(self, data_path: Optional[str] = None) -> List[DeepCheckResult]:
        """检查数据分布"""
        self.log("\n" + "=" * 60)
        self.log("2. 数据分布检查")
        self.log("=" * 60)

        results = []
        tokenizer = HIDTokenizerV2()

        # 使用测试文件
        test_file = data_path or str(Path(__file__).parent / "data" / "test_output_with_chords.txt")

        if not os.path.exists(test_file):
            results.append(DeepCheckResult(
                name="测试数据文件",
                passed=False,
                message=f"找不到: {test_file}",
                severity="error",
            ))
            self.add_result(results[-1])
            return results

        # 读取并分析数据
        content = open(test_file).read()
        ids, infos = tokenizer.encode_with_info(content)

        # 2.1 序列长度分析
        seq_len = len(ids)
        results.append(DeepCheckResult(
            name="序列长度",
            passed=seq_len > 0,
            message=f"{seq_len} tokens",
            details={"length": seq_len},
        ))
        self.add_result(results[-1])

        # 2.2 Token 类型分布
        type_counts = Counter()
        for info in infos:
            if info.token_type >= 0:
                type_names = {0: 'T', 1: 'P', 2: 'L', 3: 'V'}
                type_counts[type_names.get(info.token_type, 'Other')] += 1
            elif info.is_chord:
                type_counts['Chord'] += 1
            elif info.token_str in ['BOS', 'EOS', 'SEP']:
                type_counts['Special'] += 1
            elif info.token_str.startswith('#'):
                type_counts['Instrument'] += 1
            else:
                type_counts['Other'] += 1

        # 检查 T, P, L, V 数量是否接近
        tplv_counts = [type_counts.get(t, 0) for t in ['T', 'P', 'L', 'V']]
        if all(c > 0 for c in tplv_counts):
            max_diff = max(tplv_counts) - min(tplv_counts)
            avg_count = sum(tplv_counts) / 4
            balance_ratio = max_diff / avg_count if avg_count > 0 else float('inf')
            is_balanced = balance_ratio < 0.5  # 差异不超过平均值的 50%
        else:
            is_balanced = False

        results.append(DeepCheckResult(
            name="Token 类型平衡",
            passed=is_balanced,
            message=f"T:{type_counts.get('T',0)} P:{type_counts.get('P',0)} L:{type_counts.get('L',0)} V:{type_counts.get('V',0)}",
            severity="warning" if not is_balanced else "info",
            details=dict(type_counts),
        ))
        self.add_result(results[-1])

        # 2.3 Bar 分布
        bar_ids = [info.chord_idx for info in infos if info.chord_idx >= 0]
        if bar_ids:
            num_bars = max(bar_ids) + 1
            bar_counts = Counter(bar_ids)
            avg_tokens_per_bar = len(bar_ids) / num_bars

            results.append(DeepCheckResult(
                name="Bar 分布",
                passed=True,
                message=f"{num_bars} bars, 平均 {avg_tokens_per_bar:.1f} tokens/bar",
                details={
                    "num_bars": num_bars,
                    "avg_tokens_per_bar": avg_tokens_per_bar,
                },
            ))
            self.add_result(results[-1])

        # 2.4 乐器分布
        instrument_ids = [info.instrument_id for info in infos if info.instrument_id >= 0]
        if instrument_ids:
            unique_instruments = set(instrument_ids)
            inst_counts = Counter(instrument_ids)

            results.append(DeepCheckResult(
                name="乐器分布",
                passed=len(unique_instruments) > 0,
                message=f"{len(unique_instruments)} 种乐器",
                details={"instruments": list(unique_instruments)[:10]},
            ))
            self.add_result(results[-1])

        # 2.5 音高分布
        pitch_tokens = [info.token_str for info in infos if info.is_pitch]
        if pitch_tokens:
            pitches = [int(p[1:]) for p in pitch_tokens]
            pitch_range = max(pitches) - min(pitches)
            avg_pitch = sum(pitches) / len(pitches)

            results.append(DeepCheckResult(
                name="音高分布",
                passed=True,
                message=f"范围 P{min(pitches)}-P{max(pitches)}, 平均 P{avg_pitch:.1f}",
                details={
                    "min": min(pitches),
                    "max": max(pitches),
                    "avg": avg_pitch,
                },
            ))
            self.add_result(results[-1])

        return results

    # ==================== 3. 梯度和数值稳定性检查 ====================

    def check_gradient_stability(self) -> List[DeepCheckResult]:
        """检查梯度和数值稳定性"""
        self.log("\n" + "=" * 60)
        self.log("3. 梯度和数值稳定性检查")
        self.log("=" * 60)

        results = []
        vocab_size = 643

        # 创建模型
        model = create_model(vocab_size=vocab_size, model_size='small')
        model.to(self.device)
        model.train()

        batch_size = 2
        seq_len = 256

        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(self.device)
        chord_ids = torch.randint(-1, 10, (batch_size, seq_len)).to(self.device)
        instrument_ids = torch.randint(0, 130, (batch_size, seq_len)).to(self.device)
        labels = token_ids.clone()

        # 3.1 前向传播数值检查
        with torch.no_grad():
            logits = model(token_ids, chord_ids, instrument_ids)

            # 检查 NaN
            has_nan = torch.isnan(logits).any()
            # 检查 Inf
            has_inf = torch.isinf(logits).any()
            # 检查数值范围
            logit_max = logits.abs().max().item()
            logit_mean = logits.abs().mean().item()

        results.append(DeepCheckResult(
            name="前向传播 NaN 检查",
            passed=not has_nan,
            message="无 NaN" if not has_nan else "发现 NaN!",
            severity="critical" if has_nan else "info",
        ))
        self.add_result(results[-1])

        results.append(DeepCheckResult(
            name="前向传播 Inf 检查",
            passed=not has_inf,
            message="无 Inf" if not has_inf else "发现 Inf!",
            severity="critical" if has_inf else "info",
        ))
        self.add_result(results[-1])

        results.append(DeepCheckResult(
            name="Logits 数值范围",
            passed=logit_max < 100,
            message=f"max={logit_max:.2f}, mean={logit_mean:.4f}",
            severity="warning" if logit_max > 50 else "info",
        ))
        self.add_result(results[-1])

        # 3.2 梯度检查
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.zero_grad()

        logits = model(token_ids, chord_ids, instrument_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        loss.backward()

        # 检查梯度
        grad_norms = []
        grad_nan_params = []
        grad_inf_params = []
        grad_zero_params = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)

                if torch.isnan(param.grad).any():
                    grad_nan_params.append(name)
                if torch.isinf(param.grad).any():
                    grad_inf_params.append(name)
                if grad_norm == 0:
                    grad_zero_params.append(name)

        results.append(DeepCheckResult(
            name="梯度 NaN 检查",
            passed=len(grad_nan_params) == 0,
            message="无 NaN 梯度" if not grad_nan_params else f"发现 {len(grad_nan_params)} 个",
            severity="critical" if grad_nan_params else "info",
        ))
        self.add_result(results[-1])

        results.append(DeepCheckResult(
            name="梯度 Inf 检查",
            passed=len(grad_inf_params) == 0,
            message="无 Inf 梯度" if not grad_inf_params else f"发现 {len(grad_inf_params)} 个",
            severity="critical" if grad_inf_params else "info",
        ))
        self.add_result(results[-1])

        # 梯度范数分布
        if grad_norms:
            avg_grad_norm = sum(grad_norms) / len(grad_norms)
            max_grad_norm = max(grad_norms)
            min_grad_norm = min(grad_norms)

            results.append(DeepCheckResult(
                name="梯度范数分布",
                passed=0 < avg_grad_norm < 10,
                message=f"avg={avg_grad_norm:.4f}, max={max_grad_norm:.4f}, min={min_grad_norm:.6f}",
                severity="warning" if avg_grad_norm > 5 or avg_grad_norm == 0 else "info",
            ))
            self.add_result(results[-1])

        # 3.3 梯度裁剪效果
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        clipped_norms = []
        for param in model.parameters():
            if param.grad is not None:
                clipped_norms.append(param.grad.norm().item())

        if clipped_norms:
            max_clipped = max(clipped_norms)
            results.append(DeepCheckResult(
                name="梯度裁剪效果",
                passed=max_clipped <= 1.1,  # 允许少量误差
                message=f"裁剪后 max={max_clipped:.4f}",
            ))
            self.add_result(results[-1])

        # 3.4 权重范数检查
        weight_norms = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight_norms.append((name, param.norm().item()))

        if weight_norms:
            avg_weight_norm = sum(n for _, n in weight_norms) / len(weight_norms)
            max_weight = max(weight_norms, key=lambda x: x[1])

            results.append(DeepCheckResult(
                name="权重范数",
                passed=avg_weight_norm < 100,
                message=f"avg={avg_weight_norm:.2f}, max={max_weight[1]:.2f} ({max_weight[0][:30]}...)",
            ))
            self.add_result(results[-1])

        return results

    # ==================== 4. 长序列处理检查 ====================

    def check_long_sequence_handling(self) -> List[DeepCheckResult]:
        """检查长序列处理"""
        self.log("\n" + "=" * 60)
        self.log("4. 长序列处理检查")
        self.log("=" * 60)

        results = []
        vocab_size = 643

        # 4.1 不同序列长度的前向传播
        seq_lengths = [128, 512, 1024, 2048]

        for seq_len in seq_lengths:
            try:
                model = create_model(vocab_size=vocab_size, model_size='small', max_seq_len=max(seq_lengths) + 100)
                model.to(self.device)
                model.eval()

                token_ids = torch.randint(0, vocab_size, (1, seq_len)).to(self.device)
                chord_ids = torch.randint(-1, 10, (1, seq_len)).to(self.device)
                instrument_ids = torch.randint(0, 130, (1, seq_len)).to(self.device)

                with torch.no_grad():
                    start = time.time()
                    logits = model(token_ids, chord_ids, instrument_ids)
                    elapsed = time.time() - start

                shape_correct = logits.shape == (1, seq_len, vocab_size)
                no_nan = not torch.isnan(logits).any()

                results.append(DeepCheckResult(
                    name=f"序列长度 {seq_len}",
                    passed=shape_correct and no_nan,
                    message=f"通过 ({elapsed*1000:.1f}ms)" if (shape_correct and no_nan) else "失败",
                ))
                self.add_result(results[-1])

                del model, token_ids, chord_ids, instrument_ids, logits
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                results.append(DeepCheckResult(
                    name=f"序列长度 {seq_len}",
                    passed=False,
                    message=f"异常: {str(e)[:50]}",
                    severity="error",
                ))
                self.add_result(results[-1])

        # 4.2 RoPE 位置编码检查
        try:
            rope = RotaryPositionEmbedding(64, 4096)
            test_input = torch.randn(2, 4096, 256)

            cos, sin = rope(test_input, 4096)

            # 检查形状
            shape_ok = cos.shape == (1, 1, 4096, 64) and sin.shape == (1, 1, 4096, 64)
            # 检查数值
            no_nan = not (torch.isnan(cos).any() or torch.isnan(sin).any())

            results.append(DeepCheckResult(
                name="RoPE 位置编码 (4096 长度)",
                passed=shape_ok and no_nan,
                message=f"形状正确, 无 NaN" if (shape_ok and no_nan) else "检查失败",
            ))
            self.add_result(results[-1])

        except Exception as e:
            results.append(DeepCheckResult(
                name="RoPE 位置编码",
                passed=False,
                message=f"异常: {str(e)}",
                severity="error",
            ))
            self.add_result(results[-1])

        return results

    # ==================== 5. 模型初始化检查 ====================

    def check_model_initialization(self) -> List[DeepCheckResult]:
        """检查模型初始化"""
        self.log("\n" + "=" * 60)
        self.log("5. 模型初始化检查")
        self.log("=" * 60)

        results = []
        vocab_size = 643

        model = create_model(vocab_size=vocab_size, model_size='base')

        # 5.1 权重初始化分布
        linear_weights = []
        layer_norm_weights = []
        embedding_weights = []

        for name, param in model.named_parameters():
            if 'weight' in name:
                if 'norm' in name.lower():
                    layer_norm_weights.append((name, param))
                elif 'embedding' in name.lower():
                    embedding_weights.append((name, param))
                elif 'proj' in name or 'fc' in name or 'linear' in name or 'lm_head' in name:
                    linear_weights.append((name, param))

        # 检查 Linear 权重 (应该接近 N(0, 0.02))
        if linear_weights:
            means = [p.mean().item() for _, p in linear_weights]
            stds = [p.std().item() for _, p in linear_weights]

            avg_mean = sum(abs(m) for m in means) / len(means)
            avg_std = sum(stds) / len(stds)

            mean_ok = avg_mean < 0.01
            std_ok = 0.015 < avg_std < 0.03

            results.append(DeepCheckResult(
                name="Linear 权重初始化",
                passed=mean_ok and std_ok,
                message=f"mean≈{avg_mean:.4f}, std≈{avg_std:.4f}" +
                        (" (正常)" if mean_ok and std_ok else " (异常)"),
            ))
            self.add_result(results[-1])

        # 检查 LayerNorm 权重 (应该接近 1)
        if layer_norm_weights:
            ln_means = [p.mean().item() for _, p in layer_norm_weights]
            avg_ln_mean = sum(ln_means) / len(ln_means)

            ln_ok = 0.99 < avg_ln_mean < 1.01

            results.append(DeepCheckResult(
                name="LayerNorm 权重初始化",
                passed=ln_ok,
                message=f"mean≈{avg_ln_mean:.4f}" + (" (正常)" if ln_ok else " (应接近1)"),
            ))
            self.add_result(results[-1])

        # 5.2 偏置初始化 (应该为 0)
        biases = [(n, p) for n, p in model.named_parameters() if 'bias' in n]
        if biases:
            bias_means = [p.mean().item() for _, p in biases]
            avg_bias = sum(abs(m) for m in bias_means) / len(bias_means)

            bias_ok = avg_bias < 0.001

            results.append(DeepCheckResult(
                name="Bias 初始化",
                passed=bias_ok,
                message=f"mean≈{avg_bias:.6f}" + (" (接近0)" if bias_ok else " (应为0)"),
            ))
            self.add_result(results[-1])

        # 5.3 嵌入层初始化
        if hasattr(model, 'chord_embedding'):
            embed = model.chord_embedding.token_embedding.weight
            embed_std = embed.std().item()

            embed_ok = 0.015 < embed_std < 0.03

            results.append(DeepCheckResult(
                name="Token 嵌入初始化",
                passed=embed_ok,
                message=f"std≈{embed_std:.4f}" + (" (正常)" if embed_ok else " (异常)"),
            ))
            self.add_result(results[-1])

        # 5.4 参数量统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        results.append(DeepCheckResult(
            name="参数量统计",
            passed=total_params == trainable_params,
            message=f"总计 {total_params/1e6:.2f}M, 可训练 {trainable_params/1e6:.2f}M",
        ))
        self.add_result(results[-1])

        return results

    # ==================== 6. 训练脚本兼容性检查 ====================

    def check_training_script_compatibility(self) -> List[DeepCheckResult]:
        """检查训练脚本兼容性"""
        self.log("\n" + "=" * 60)
        self.log("6. 训练脚本兼容性检查")
        self.log("=" * 60)

        results = []

        # 6.1 检查训练脚本存在
        train_script = Path(__file__).parent / "train.py"
        results.append(DeepCheckResult(
            name="训练脚本存在",
            passed=train_script.exists(),
            message=str(train_script) if train_script.exists() else "找不到 train.py",
        ))
        self.add_result(results[-1])

        if not train_script.exists():
            return results

        # 6.2 检查导入
        try:
            # 尝试导入训练脚本中的模块
            from model import HIDMuseFormer, create_model
            from data import HIDMusicDataset, collate_fn, HIDTokenizerV2

            results.append(DeepCheckResult(
                name="模块导入",
                passed=True,
                message="model 和 data 模块导入成功",
            ))
            self.add_result(results[-1])

        except ImportError as e:
            results.append(DeepCheckResult(
                name="模块导入",
                passed=False,
                message=f"导入失败: {str(e)}",
                severity="error",
            ))
            self.add_result(results[-1])

        # 6.3 检查 Dataset 和 Model 接口兼容性
        try:
            tokenizer = HIDTokenizerV2()
            model = create_model(
                vocab_size=tokenizer.vocab_size,
                model_size='small',
                chord_start_id=tokenizer.chord_start_id,
                chord_end_id=tokenizer.chord_end_id,
            )

            results.append(DeepCheckResult(
                name="Model 创建 (使用 tokenizer 参数)",
                passed=True,
                message=f"vocab_size={tokenizer.vocab_size}",
            ))
            self.add_result(results[-1])

        except Exception as e:
            results.append(DeepCheckResult(
                name="Model 创建",
                passed=False,
                message=f"失败: {str(e)}",
                severity="error",
            ))
            self.add_result(results[-1])

        # 6.4 检查 collate_fn 输出格式
        try:
            # 创建模拟 batch
            tokenizer = HIDTokenizerV2()
            test_content = "#P0\nB0 T0 P60 L4 V16"
            ids, infos = tokenizer.encode_with_info(test_content)

            mock_item = {
                'token_ids': torch.tensor(ids),
                'chord_ids': torch.tensor([i.chord_idx for i in infos]),
                'position_ids': torch.tensor([i.position for i in infos]),
                'instrument_ids': torch.tensor([i.instrument_id if i.instrument_id >= 0 else 129 for i in infos]),
                'is_chord_token': torch.tensor([i.is_chord for i in infos]),
                'length': len(ids),
            }

            batch = collate_fn([mock_item, mock_item])

            required_keys = ['token_ids', 'labels', 'chord_ids', 'instrument_ids', 'key_padding_mask']
            missing = [k for k in required_keys if k not in batch]

            results.append(DeepCheckResult(
                name="collate_fn 输出格式",
                passed=len(missing) == 0,
                message="包含所有必需字段" if not missing else f"缺少: {missing}",
            ))
            self.add_result(results[-1])

        except Exception as e:
            results.append(DeepCheckResult(
                name="collate_fn 测试",
                passed=False,
                message=f"失败: {str(e)}",
                severity="error",
            ))
            self.add_result(results[-1])

        # 6.5 检查混合精度兼容性
        if torch.cuda.is_available():
            try:
                from torch.cuda.amp import autocast, GradScaler

                model = create_model(vocab_size=643, model_size='small').to(self.device)
                scaler = GradScaler()

                token_ids = torch.randint(0, 643, (2, 64)).to(self.device)

                with autocast():
                    logits = model(token_ids)
                    loss = logits.sum()

                scaler.scale(loss).backward()

                results.append(DeepCheckResult(
                    name="混合精度 (AMP)",
                    passed=True,
                    message="autocast + GradScaler 正常",
                ))
                self.add_result(results[-1])

            except Exception as e:
                results.append(DeepCheckResult(
                    name="混合精度 (AMP)",
                    passed=False,
                    message=f"失败: {str(e)}",
                    severity="warning",
                ))
                self.add_result(results[-1])
        else:
            results.append(DeepCheckResult(
                name="混合精度 (AMP)",
                passed=True,
                message="无 GPU, 跳过检查",
                severity="info",
            ))
            self.add_result(results[-1])

        return results

    # ==================== 7. Summary Attention 深入检查 ====================

    def check_summary_attention_deep(self) -> List[DeepCheckResult]:
        """Summary Attention 深入检查"""
        self.log("\n" + "=" * 60)
        self.log("7. SUMMARY ATTENTION 深入检查")
        self.log("=" * 60)

        results = []

        embed_dim = 256
        num_heads = 4
        num_bars = 8
        tokens_per_bar = 32
        batch_size = 2
        reg_len = num_bars * tokens_per_bar
        sum_len = num_bars

        # 7.1 信息流验证: SR → K2V2 → RS
        try:
            sum_attn = SummaryAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=0.0,
                use_rope=False,  # 禁用 RoPE 以便更清楚地看到信息流
            )

            mask_gen = SummaryAttentionMask()
            bar_ids = torch.arange(reg_len).unsqueeze(0).expand(batch_size, -1) // tokens_per_bar
            instrument_ids = torch.zeros(batch_size, reg_len, dtype=torch.long)

            masks = mask_gen.create_masks(
                bar_ids=bar_ids,
                instrument_ids=instrument_ids,
                num_bars=num_bars,
                device='cpu',
            )

            # SR 信息隔离测试: 比较 bar0 有信息和无信息时各 S 的输出差异
            # 基准: bar0 = 0
            reg_x_baseline = torch.zeros(batch_size, reg_len, embed_dim)
            sum_x_test = torch.zeros(batch_size, sum_len, embed_dim)

            with torch.no_grad():
                sum_out_baseline, _ = sum_attn(
                    sum_x_test, reg_x_baseline,
                    masks['ss'], masks['sr'], masks['rs'], masks['rr']
                )

            # 测试: bar0 = 1
            reg_x_with_bar0 = torch.zeros(batch_size, reg_len, embed_dim)
            reg_x_with_bar0[:, :tokens_per_bar, :] = 1.0  # bar 0 的 token 都是 1

            with torch.no_grad():
                sum_out_with_bar0, _ = sum_attn(
                    sum_x_test, reg_x_with_bar0,
                    masks['ss'], masks['sr'], masks['rs'], masks['rr']
                )

            # S0 应该受到 bar0 影响
            s0_diff = (sum_out_baseline[0, 0] - sum_out_with_bar0[0, 0]).abs().sum()
            s0_has_bar0_influence = s0_diff > 0.1

            # S1-S7 不应该受到 bar0 影响 (只受各自 bar 影响)
            s1_7_no_bar0_influence = True
            for s_idx in range(1, num_bars):
                s_diff = (sum_out_baseline[0, s_idx] - sum_out_with_bar0[0, s_idx]).abs().sum()
                if s_diff > 1e-5:
                    s1_7_no_bar0_influence = False
                    break

            results.append(DeepCheckResult(
                name="SR 信息传播 (S0 受 bar0 影响)",
                passed=s0_has_bar0_influence,
                message=f"S0 输出差异={s0_diff:.4f}",
            ))
            self.add_result(results[-1])

            results.append(DeepCheckResult(
                name="SR 信息隔离 (S1-S7 不受 bar0 影响)",
                passed=s1_7_no_bar0_influence,
                message="正确隔离" if s1_7_no_bar0_influence else "信息泄露",
            ))
            self.add_result(results[-1])

            # 7.2 RS 因果性测试: 比较 S0=0 和 S0=1 时各 bar 的输出差异
            # 正确的测试方法：如果 bar0 不能看到 S0，则 S0 的变化不应影响 bar0 的输出

            # 基准: S0 = 0
            sum_x_baseline = torch.zeros(batch_size, sum_len, embed_dim)
            reg_x_test = torch.zeros(batch_size, reg_len, embed_dim)

            with torch.no_grad():
                _, reg_out_baseline = sum_attn(
                    sum_x_baseline, reg_x_test,
                    masks['ss'], masks['sr'], masks['rs'], masks['rr']
                )

            # 测试: S0 = 1
            sum_x_with_s0 = torch.zeros(batch_size, sum_len, embed_dim)
            sum_x_with_s0[:, 0, :] = 1.0  # 只有 S0 有信息

            with torch.no_grad():
                _, reg_out_with_s0 = sum_attn(
                    sum_x_with_s0, reg_x_test,
                    masks['ss'], masks['sr'], masks['rs'], masks['rr']
                )

            # 比较 bar0 的输出差异（因果性测试）
            bar0_diff = (reg_out_baseline[0, :tokens_per_bar] -
                        reg_out_with_s0[0, :tokens_per_bar]).abs().sum()
            bar0_no_s0_influence = bar0_diff < 1e-5

            # 比较 bar1+ 的输出差异（信息传播测试）
            bar1_diff = (reg_out_baseline[0, tokens_per_bar:] -
                        reg_out_with_s0[0, tokens_per_bar:]).abs().sum()
            bar1_has_s0_influence = bar1_diff > 0.1

            results.append(DeepCheckResult(
                name="RS 信息传播 (bar1+ 受 S0 影响)",
                passed=bar1_has_s0_influence,
                message=f"bar1+ 输出差异={bar1_diff:.4f}",
            ))
            self.add_result(results[-1])

            results.append(DeepCheckResult(
                name="RS 因果性 (bar0 不受 S0 影响)",
                passed=bar0_no_s0_influence,
                message=f"bar0 输出差异={bar0_diff:.6f}" if bar0_no_s0_influence else f"因果性破坏: diff={bar0_diff:.4f}",
                severity="critical" if not bar0_no_s0_influence else "info",
            ))
            self.add_result(results[-1])

        except Exception as e:
            results.append(DeepCheckResult(
                name="Summary Attention 信息流检查",
                passed=False,
                message=f"异常: {str(e)}",
                severity="error",
            ))
            self.add_result(results[-1])

        # 7.3 K2V2 二次投影验证
        try:
            # 检查 K2V2 是否真的被使用
            sum_attn = SummaryAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.0)

            # 保存初始 K2V2 权重
            k2_weight_before = sum_attn.sum_k2_proj.weight.clone()
            v2_weight_before = sum_attn.sum_v2_proj.weight.clone()

            # 训练一步
            sum_x = torch.randn(batch_size, sum_len, embed_dim, requires_grad=True)
            reg_x = torch.randn(batch_size, reg_len, embed_dim, requires_grad=True)

            optimizer = torch.optim.Adam(sum_attn.parameters(), lr=0.01)
            optimizer.zero_grad()

            sum_out, reg_out = sum_attn(
                sum_x, reg_x,
                masks['ss'], masks['sr'], masks['rs'], masks['rr']
            )
            loss = sum_out.sum() + reg_out.sum()
            loss.backward()
            optimizer.step()

            # 检查 K2V2 权重是否变化
            k2_changed = not torch.allclose(k2_weight_before, sum_attn.sum_k2_proj.weight)
            v2_changed = not torch.allclose(v2_weight_before, sum_attn.sum_v2_proj.weight)

            results.append(DeepCheckResult(
                name="K2V2 权重更新",
                passed=k2_changed and v2_changed,
                message=f"K2 更新: {k2_changed}, V2 更新: {v2_changed}",
            ))
            self.add_result(results[-1])

        except Exception as e:
            results.append(DeepCheckResult(
                name="K2V2 检查",
                passed=False,
                message=f"异常: {str(e)}",
                severity="error",
            ))
            self.add_result(results[-1])

        return results

    # ==================== 运行所有检查 ====================

    def run_all_checks(self, data_path: Optional[str] = None) -> Dict[str, Any]:
        """运行所有深入检查"""
        self.log("\n" + "=" * 70)
        self.log("HID-MUSEFORMER 深入训练准备检查")
        self.log("=" * 70)

        start_time = time.time()

        all_results = []

        all_results.extend(self.check_tokenizer_edge_cases())
        all_results.extend(self.check_data_distribution(data_path))
        all_results.extend(self.check_gradient_stability())
        all_results.extend(self.check_long_sequence_handling())
        all_results.extend(self.check_model_initialization())
        all_results.extend(self.check_training_script_compatibility())
        all_results.extend(self.check_summary_attention_deep())

        elapsed = time.time() - start_time

        # 统计
        passed = sum(1 for r in all_results if r.passed)
        failed = sum(1 for r in all_results if not r.passed)
        critical = sum(1 for r in all_results if not r.passed and r.severity == "critical")
        errors = sum(1 for r in all_results if not r.passed and r.severity == "error")
        warnings = sum(1 for r in all_results if not r.passed and r.severity == "warning")
        total = len(all_results)

        # 最终报告
        self.log("\n" + "=" * 70)
        self.log("深入检查完成!")
        self.log("=" * 70)
        self.log(f"通过: {passed}/{total}")
        self.log(f"失败: {failed}/{total}")
        if critical > 0:
            self.log(f"  💀 致命问题: {critical}")
        if errors > 0:
            self.log(f"  ✗ 错误: {errors}")
        if warnings > 0:
            self.log(f"  ⚠ 警告: {warnings}")
        self.log(f"耗时: {elapsed:.2f}s")

        if critical > 0:
            self.log("\n💀 发现致命问题! 必须修复后才能训练。")
        elif errors > 0:
            self.log("\n✗ 发现错误! 建议修复后再训练。")
        elif warnings > 0:
            self.log("\n⚠ 发现警告。可以训练，但建议关注。")
        else:
            self.log("\n✓ 所有深入检查通过! 可以安全开始训练。")

        return {
            'passed': passed,
            'failed': failed,
            'critical': critical,
            'errors': errors,
            'warnings': warnings,
            'total': total,
            'elapsed': elapsed,
            'ready_to_train': critical == 0 and errors == 0,
            'results': [
                {
                    'name': r.name,
                    'passed': r.passed,
                    'message': r.message,
                    'severity': r.severity,
                }
                for r in all_results
            ],
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='深入训练准备检查')
    parser.add_argument('--data-path', type=str, help='测试数据文件路径')
    parser.add_argument('--output', type=str, help='输出 JSON 报告路径')
    parser.add_argument('--quiet', action='store_true', help='安静模式')

    args = parser.parse_args()

    checker = DeepTrainingChecker(verbose=not args.quiet)
    report = checker.run_all_checks(data_path=args.data_path)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n报告已保存到: {args.output}")

    sys.exit(0 if report['ready_to_train'] else 1)


if __name__ == '__main__':
    main()
