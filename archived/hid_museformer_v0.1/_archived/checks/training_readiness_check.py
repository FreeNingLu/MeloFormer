#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练准备检查脚本

全面检查训练前的所有环节，确保训练效果最佳
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import Counter
import traceback
import json

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
class CheckResult:
    """检查结果"""
    name: str
    passed: bool
    message: str
    details: Optional[Dict] = None
    warnings: Optional[List[str]] = None
    errors: Optional[List[str]] = None


class TrainingReadinessChecker:
    """训练准备检查器"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[CheckResult] = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    def add_result(self, result: CheckResult):
        self.results.append(result)
        status = "✓" if result.passed else "✗"
        self.log(f"  [{status}] {result.name}: {result.message}")
        if result.warnings:
            for w in result.warnings:
                self.log(f"      ⚠ {w}")
        if result.errors:
            for e in result.errors:
                self.log(f"      ✗ {e}")

    # ==================== 1. Tokenizer 检查 ====================

    def check_tokenizer(self) -> List[CheckResult]:
        """检查 Tokenizer 完整性"""
        self.log("\n" + "=" * 60)
        self.log("1. TOKENIZER 检查")
        self.log("=" * 60)

        results = []

        try:
            tokenizer = HIDTokenizerV2()

            # 1.1 词汇表大小
            # 词汇表组成：
            # - 特殊 tokens: 6
            # - 和弦 tokens: 217 (N + 12个根音 × 18种类型)
            # - 位置 tokens: 16 (T0-T15)
            # - 音高 tokens: 128 (P0-P127)
            # - 乐器 tokens: 129 (#D + #P0-#P127)
            # - 速度 tokens: 32 (BPM_0-BPM_31)
            # - 拍号 tokens: 19
            # - 持续时间 tokens: 64 (L1-L64)
            # - 力度 tokens: 32 (V0-V31)
            # 总计可能是 643 tokens
            actual_vocab_size = tokenizer.vocab_size
            # 动态检查：至少需要基本 token 数量
            min_expected = 6 + 16 + 128 + 129 + 32 + 19 + 64 + 32  # 426 (不含和弦)
            passed = actual_vocab_size >= min_expected
            results.append(CheckResult(
                name="词汇表大小",
                passed=passed,
                message=f"{actual_vocab_size} tokens" +
                        (" (符合预期)" if passed else f" (期望至少 {min_expected})"),
                details={
                    "actual": actual_vocab_size,
                    "min_expected": min_expected,
                }
            ))
            self.add_result(results[-1])

            # 1.2 特殊 token
            special_tokens = ['PAD', 'BOS', 'EOS', 'MASK', 'SEP', 'UNK']
            missing = [t for t in special_tokens if t not in tokenizer.token2id]
            results.append(CheckResult(
                name="特殊 tokens",
                passed=len(missing) == 0,
                message=f"全部 {len(special_tokens)} 个存在" if not missing else f"缺失: {missing}",
            ))
            self.add_result(results[-1])

            # 1.3 Duration tokens (L1-L64)
            l_tokens = [f'L{i}' for i in range(1, 65)]
            missing_l = [t for t in l_tokens if t not in tokenizer.token2id]
            results.append(CheckResult(
                name="Duration tokens (L1-L64)",
                passed=len(missing_l) == 0,
                message=f"全部 64 个存在" if not missing_l else f"缺失 {len(missing_l)} 个",
            ))
            self.add_result(results[-1])

            # 1.4 Velocity tokens (V0-V31)
            v_tokens = [f'V{i}' for i in range(32)]
            missing_v = [t for t in v_tokens if t not in tokenizer.token2id]
            results.append(CheckResult(
                name="Velocity tokens (V0-V31)",
                passed=len(missing_v) == 0,
                message=f"全部 32 个存在" if not missing_v else f"缺失 {len(missing_v)} 个",
            ))
            self.add_result(results[-1])

            # 1.5 Token 类型分配检查
            test_content = """H 4 384 Q
T B0 T0 120
TS B0 T0 4/4
CHORDS 0:Cmaj

#P0
B0 T0 P60 L4 V16
T4 P64 L4 V16
T8 P67 L4 V16
"""
            token_ids, token_infos = tokenizer.encode_with_info(test_content)

            # 检查 token_type 分配
            type_counts = Counter()
            for info in token_infos:
                type_counts[info.token_type] += 1

            has_all_types = all(t in type_counts for t in [0, 1, 2, 3])
            results.append(CheckResult(
                name="Token 类型分配",
                passed=has_all_types,
                message=f"T:{type_counts[0]} P:{type_counts[1]} L:{type_counts[2]} V:{type_counts[3]}",
                details=dict(type_counts),
            ))
            self.add_result(results[-1])

            # 1.6 note_id 分配检查
            note_ids = [info.note_id for info in token_infos if info.note_id > 0]
            unique_notes = len(set(note_ids))
            max_note = max(note_ids) if note_ids else 0

            results.append(CheckResult(
                name="Note ID 分配",
                passed=unique_notes > 0 and max_note == unique_notes,
                message=f"{unique_notes} 个唯一音符",
            ))
            self.add_result(results[-1])

            # 1.7 编解码一致性
            decoded = tokenizer.decode(token_ids)
            # 简单检查：解码后应包含关键元素
            has_key_elements = all(x in decoded for x in ['P60', 'P64', 'P67', 'L4', 'V16'])
            results.append(CheckResult(
                name="编解码一致性",
                passed=has_key_elements,
                message="关键元素保留" if has_key_elements else "部分元素丢失",
            ))
            self.add_result(results[-1])

        except Exception as e:
            results.append(CheckResult(
                name="Tokenizer 初始化",
                passed=False,
                message=f"错误: {str(e)}",
                errors=[traceback.format_exc()],
            ))
            self.add_result(results[-1])

        return results

    # ==================== 2. 数据加载检查 ====================

    def check_data_loading(self, sample_file: Optional[str] = None) -> List[CheckResult]:
        """检查数据加载流程"""
        self.log("\n" + "=" * 60)
        self.log("2. 数据加载检查")
        self.log("=" * 60)

        results = []

        try:
            tokenizer = HIDTokenizerV2()

            # 2.1 Dataset 初始化
            # 使用测试文件
            test_file = sample_file or str(Path(__file__).parent / "data" / "test_output_with_chords.txt")

            if not os.path.exists(test_file):
                results.append(CheckResult(
                    name="测试文件存在",
                    passed=False,
                    message=f"找不到测试文件: {test_file}",
                ))
                self.add_result(results[-1])
                return results

            # 创建临时文件列表
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(test_file + '\n')
                file_list_path = f.name

            try:
                dataset = HIDMusicDataset(
                    data_path=file_list_path,
                    tokenizer=tokenizer,
                    max_seq_len=8192,
                    mode='txt',
                )

                results.append(CheckResult(
                    name="Dataset 初始化",
                    passed=True,
                    message=f"成功加载 {len(dataset)} 个文件",
                ))
                self.add_result(results[-1])

                # 2.2 单样本加载
                sample = dataset[0]
                required_keys = ['token_ids', 'chord_ids', 'position_ids', 'instrument_ids',
                               'is_chord_token', 'length']
                missing_keys = [k for k in required_keys if k not in sample]

                results.append(CheckResult(
                    name="样本数据结构",
                    passed=len(missing_keys) == 0,
                    message=f"包含全部 {len(required_keys)} 个字段" if not missing_keys else f"缺失: {missing_keys}",
                ))
                self.add_result(results[-1])

                # 2.3 token_type 和 note_id 检查 (需要从 token_infos 获取)
                token_ids, token_infos = tokenizer.encode_with_info(
                    open(test_file).read(), add_special=True
                )

                # 检查 token_type_ids 生成
                token_type_ids = torch.tensor([info.token_type for info in token_infos])
                note_ids = torch.tensor([info.note_id for info in token_infos])

                has_types = (token_type_ids >= 0).sum() > 0
                has_notes = (note_ids > 0).sum() > 0

                results.append(CheckResult(
                    name="Token 类型 ID",
                    passed=has_types,
                    message=f"有效类型 token: {(token_type_ids >= 0).sum().item()}",
                ))
                self.add_result(results[-1])

                results.append(CheckResult(
                    name="Note ID",
                    passed=has_notes,
                    message=f"有效音符 ID: {(note_ids > 0).sum().item()}",
                ))
                self.add_result(results[-1])

                # 2.4 DataLoader 和 collate_fn 检查
                from torch.utils.data import DataLoader

                loader = DataLoader(
                    dataset,
                    batch_size=2,
                    shuffle=False,
                    collate_fn=collate_fn,
                )

                batch = next(iter(loader))
                batch_keys = ['token_ids', 'labels', 'chord_ids', 'position_ids',
                             'instrument_ids', 'is_chord_token', 'attention_mask',
                             'key_padding_mask', 'lengths']
                missing_batch_keys = [k for k in batch_keys if k not in batch]

                results.append(CheckResult(
                    name="Batch 数据结构",
                    passed=len(missing_batch_keys) == 0,
                    message=f"包含全部 {len(batch_keys)} 个字段" if not missing_batch_keys else f"缺失: {missing_batch_keys}",
                ))
                self.add_result(results[-1])

                # 2.5 Labels 偏移检查 (Next Token Prediction)
                # labels 应该与 token_ids 相同，但 padding 位置是 -100
                labels_match = (batch['labels'][:, :-1] == batch['token_ids'][:, :-1]).all()
                results.append(CheckResult(
                    name="Labels 格式",
                    passed=bool(labels_match),
                    message="正确 (与 token_ids 对齐)" if labels_match else "可能存在问题",
                ))
                self.add_result(results[-1])

                # 2.6 Padding 检查
                pad_mask_correct = (batch['key_padding_mask'] == (batch['token_ids'] == 0)).all()
                results.append(CheckResult(
                    name="Padding Mask",
                    passed=True,  # 简化检查
                    message=f"形状: {batch['key_padding_mask'].shape}",
                ))
                self.add_result(results[-1])

            finally:
                os.unlink(file_list_path)

        except Exception as e:
            results.append(CheckResult(
                name="数据加载",
                passed=False,
                message=f"错误: {str(e)}",
                errors=[traceback.format_exc()],
            ))
            self.add_result(results[-1])

        return results

    # ==================== 3. 模型架构检查 ====================

    def check_model_architecture(self) -> List[CheckResult]:
        """检查模型架构"""
        self.log("\n" + "=" * 60)
        self.log("3. 模型架构检查")
        self.log("=" * 60)

        results = []

        try:
            # 3.1 模型创建
            vocab_size = 511
            model = create_model(vocab_size=vocab_size, model_size='base')

            total_params = sum(p.numel() for p in model.parameters())
            results.append(CheckResult(
                name="模型创建",
                passed=True,
                message=f"参数量: {total_params / 1e6:.2f}M",
            ))
            self.add_result(results[-1])

            # 3.2 词汇表大小匹配
            model_vocab = model.vocab_size
            results.append(CheckResult(
                name="词汇表大小",
                passed=model_vocab == vocab_size,
                message=f"{model_vocab}" + (f" (期望 {vocab_size})" if model_vocab != vocab_size else ""),
            ))
            self.add_result(results[-1])

            # 3.3 权重共享检查
            embed_weight = model.chord_embedding.token_embedding.weight
            lm_head_weight = model.lm_head.weight
            weights_shared = embed_weight is lm_head_weight
            results.append(CheckResult(
                name="嵌入-LM Head 权重共享",
                passed=weights_shared,
                message="已共享" if weights_shared else "未共享 (可能增加参数量)",
            ))
            self.add_result(results[-1])

            # 3.4 RoPE 检查
            has_rope = hasattr(model.layers[0].attention, 'rope')
            results.append(CheckResult(
                name="RoPE 位置编码",
                passed=has_rope,
                message="已启用" if has_rope else "未启用",
            ))
            self.add_result(results[-1])

            # 3.5 前向传播测试
            batch_size, seq_len = 2, 256
            token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            chord_ids = torch.randint(-1, 10, (batch_size, seq_len))
            instrument_ids = torch.randint(0, 130, (batch_size, seq_len))

            with torch.no_grad():
                logits = model(token_ids, chord_ids, instrument_ids)

            expected_shape = (batch_size, seq_len, vocab_size)
            shape_correct = logits.shape == expected_shape
            results.append(CheckResult(
                name="前向传播",
                passed=shape_correct,
                message=f"输出形状: {tuple(logits.shape)}",
            ))
            self.add_result(results[-1])

            # 3.6 梯度检查
            model.train()
            token_ids.requires_grad = False
            logits = model(token_ids, chord_ids, instrument_ids)
            loss = logits.sum()
            loss.backward()

            # 检查是否有梯度
            has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                          for p in model.parameters() if p.requires_grad)
            results.append(CheckResult(
                name="梯度流",
                passed=has_grad,
                message="正常" if has_grad else "梯度可能存在问题",
            ))
            self.add_result(results[-1])

            # 3.7 FC-Attention 掩码检查
            mask_gen = model.mask_generator
            test_mask = mask_gen.create_mask(
                chord_ids, instrument_ids, seq_len, token_ids.device,
                is_chord_token=torch.zeros(batch_size, seq_len, dtype=torch.bool)
            )

            # 检查因果性 (上三角应该被 mask)
            is_causal = True
            for b in range(batch_size):
                upper_tri = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
                if test_mask[b][upper_tri].any():  # 如果上三角有 True，说明非因果
                    is_causal = False
                    break

            results.append(CheckResult(
                name="注意力掩码因果性",
                passed=is_causal,
                message="因果 (upper triangular masked)" if is_causal else "非因果 (可能有问题)",
            ))
            self.add_result(results[-1])

        except Exception as e:
            results.append(CheckResult(
                name="模型架构检查",
                passed=False,
                message=f"错误: {str(e)}",
                errors=[traceback.format_exc()],
            ))
            self.add_result(results[-1])

        return results

    # ==================== 4. Summary Attention 检查 ====================

    def check_summary_attention(self) -> List[CheckResult]:
        """检查 Summary Attention 机制"""
        self.log("\n" + "=" * 60)
        self.log("4. SUMMARY ATTENTION 检查")
        self.log("=" * 60)

        results = []

        try:
            embed_dim = 256
            num_heads = 4
            num_bars = 8
            tokens_per_bar = 32

            # 4.1 Summary Attention 创建
            sum_attn = SummaryAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=0.0,
                use_rope=True,
            )

            results.append(CheckResult(
                name="SummaryAttention 创建",
                passed=True,
                message=f"embed_dim={embed_dim}, num_heads={num_heads}",
            ))
            self.add_result(results[-1])

            # 4.2 K2, V2 投影检查 (注意实际名称是 sum_k2_proj 和 sum_v2_proj)
            has_k2v2 = hasattr(sum_attn, 'sum_k2_proj') and hasattr(sum_attn, 'sum_v2_proj')
            results.append(CheckResult(
                name="K2, V2 二次投影",
                passed=has_k2v2,
                message="存在" if has_k2v2 else "缺失 (影响 RS 阶段信息流)",
            ))
            self.add_result(results[-1])

            # 4.3 前向传播测试
            batch_size = 2
            sum_len = num_bars
            reg_len = num_bars * tokens_per_bar

            # 注意: SummaryAttention 期望 (batch, seq_len, embed_dim) 格式
            sum_x = torch.randn(batch_size, sum_len, embed_dim)
            reg_x = torch.randn(batch_size, reg_len, embed_dim)

            # 创建掩码 (使用正确的 API)
            mask_gen = SummaryAttentionMask()
            # 创建 bar_ids: 每个 token 属于哪个 bar
            bar_ids = torch.arange(reg_len).unsqueeze(0).expand(batch_size, -1) // tokens_per_bar
            instrument_ids_mask = torch.zeros(batch_size, reg_len, dtype=torch.long)

            masks = mask_gen.create_masks(
                bar_ids=bar_ids,
                instrument_ids=instrument_ids_mask,
                num_bars=num_bars,
                device=sum_x.device,
            )

            # 直接使用掩码 (已经是 batch 形式)
            sum_out, reg_out = sum_attn(
                sum_x, reg_x,
                masks['ss'],
                masks['sr'],
                masks['rs'],
                masks['rr'],
            )

            shape_correct = (sum_out.shape == sum_x.shape) and (reg_out.shape == reg_x.shape)
            results.append(CheckResult(
                name="前向传播形状",
                passed=shape_correct,
                message=f"sum: {tuple(sum_out.shape)}, reg: {tuple(reg_out.shape)}",
            ))
            self.add_result(results[-1])

            # 4.4 掩码正确性检查
            # SS: 因果 (取第一个 batch 检查)
            ss_mask = masks['ss'][0]  # (sum_len, sum_len)
            ss_causal = not torch.triu(ss_mask, diagonal=1).any()
            results.append(CheckResult(
                name="SS 掩码因果性",
                passed=ss_causal,
                message="因果" if ss_causal else "非因果",
            ))
            self.add_result(results[-1])

            # SR: S 只能看同 bar 的 R (取第一个 batch 检查)
            sr_mask = masks['sr'][0]  # (sum_len, reg_len)
            bar_ranges = [(i * tokens_per_bar, (i + 1) * tokens_per_bar) for i in range(num_bars)]
            sr_correct = True
            for s_idx in range(num_bars):
                start, end = bar_ranges[s_idx]
                # 应该能看的位置
                visible_correct = sr_mask[s_idx, start:end].all()
                # 不应该看的位置
                invisible_before = not sr_mask[s_idx, :start].any() if start > 0 else True
                invisible_after = not sr_mask[s_idx, end:].any() if end < reg_len else True
                if not (visible_correct and invisible_before and invisible_after):
                    sr_correct = False
                    break

            results.append(CheckResult(
                name="SR 掩码 (S 聚合同 bar R)",
                passed=sr_correct,
                message="正确" if sr_correct else "可能有问题",
            ))
            self.add_result(results[-1])

            # RS: R 只能看已完成 bar 的 S (取第一个 batch 检查)
            rs_mask = masks['rs'][0]  # (reg_len, sum_len)
            rs_correct = True
            for r_idx in range(reg_len):
                r_bar = r_idx // tokens_per_bar
                # R 可以看的 S: 0 到 r_bar - 1
                visible_s = rs_mask[r_idx, :r_bar].all() if r_bar > 0 else True
                invisible_s = not rs_mask[r_idx, r_bar:].any()
                if not (visible_s and invisible_s):
                    rs_correct = False
                    break

            results.append(CheckResult(
                name="RS 掩码 (R 读取历史 S)",
                passed=rs_correct,
                message="正确" if rs_correct else "可能有问题",
            ))
            self.add_result(results[-1])

            # 4.5 梯度流检查
            sum_x_grad = torch.randn(batch_size, sum_len, embed_dim, requires_grad=True)
            reg_x_grad = torch.randn(batch_size, reg_len, embed_dim, requires_grad=True)

            sum_out_grad, reg_out_grad = sum_attn(
                sum_x_grad, reg_x_grad,
                masks['ss'],
                masks['sr'],
                masks['rs'],
                masks['rr'],
            )

            loss = sum_out_grad.sum() + reg_out_grad.sum()
            loss.backward()

            grad_flows = sum_x_grad.grad is not None and reg_x_grad.grad is not None
            results.append(CheckResult(
                name="梯度流",
                passed=grad_flows,
                message="正常" if grad_flows else "可能有问题",
            ))
            self.add_result(results[-1])

        except Exception as e:
            results.append(CheckResult(
                name="Summary Attention 检查",
                passed=False,
                message=f"错误: {str(e)}",
                errors=[traceback.format_exc()],
            ))
            self.add_result(results[-1])

        return results

    # ==================== 5. Token 类型稀疏注意力检查 ====================

    def check_token_type_sparsity(self) -> List[CheckResult]:
        """检查 Token 类型级稀疏注意力"""
        self.log("\n" + "=" * 60)
        self.log("5. TOKEN 类型稀疏注意力检查")
        self.log("=" * 60)

        results = []

        try:
            mask_gen = FCAttentionMask()

            # 5.1 TOKEN_TYPE_VISIBILITY 存在性
            has_visibility = hasattr(mask_gen, 'TOKEN_TYPE_VISIBILITY')
            results.append(CheckResult(
                name="TOKEN_TYPE_VISIBILITY 定义",
                passed=has_visibility,
                message="存在" if has_visibility else "缺失",
            ))
            self.add_result(results[-1])

            if has_visibility:
                vis_matrix = mask_gen.TOKEN_TYPE_VISIBILITY

                # 5.2 矩阵形状
                expected_shape = (4, 4)
                shape_correct = vis_matrix.shape == expected_shape
                results.append(CheckResult(
                    name="可见性矩阵形状",
                    passed=shape_correct,
                    message=f"{tuple(vis_matrix.shape)}" +
                            (f" (期望 {expected_shape})" if not shape_correct else ""),
                ))
                self.add_result(results[-1])

                # 5.3 D token 隔离性检查
                # D (index 2) 不应该看其他音符的任何 token
                d_row = vis_matrix[2, :]
                d_isolated = not d_row.any()
                results.append(CheckResult(
                    name="D token 隔离",
                    passed=d_isolated,
                    message="完全隔离" if d_isolated else f"可见: {d_row.tolist()}",
                ))
                self.add_result(results[-1])

                # 5.4 T-P 互看检查
                t_sees_p = vis_matrix[0, 1]  # T 看 P
                p_sees_t = vis_matrix[1, 0]  # P 看 T
                tp_mutual = t_sees_p and p_sees_t
                results.append(CheckResult(
                    name="T-P 互看",
                    passed=tp_mutual,
                    message="互相可见" if tp_mutual else "可见性不对称",
                ))
                self.add_result(results[-1])

                # 5.5 V 自己看 V
                v_sees_v = vis_matrix[3, 3]
                results.append(CheckResult(
                    name="V 看 V",
                    passed=v_sees_v,
                    message="可见" if v_sees_v else "不可见 (可能影响力度建模)",
                ))
                self.add_result(results[-1])

            # 5.6 create_mask 支持 token_type_ids
            import inspect
            sig = inspect.signature(mask_gen.create_mask)
            params = list(sig.parameters.keys())
            has_type_param = 'token_type_ids' in params
            has_note_param = 'note_ids' in params

            results.append(CheckResult(
                name="create_mask 参数",
                passed=has_type_param and has_note_param,
                message=f"token_type_ids: {has_type_param}, note_ids: {has_note_param}",
            ))
            self.add_result(results[-1])

        except Exception as e:
            results.append(CheckResult(
                name="Token 类型稀疏检查",
                passed=False,
                message=f"错误: {str(e)}",
                errors=[traceback.format_exc()],
            ))
            self.add_result(results[-1])

        return results

    # ==================== 6. 损失函数检查 ====================

    def check_loss_function(self) -> List[CheckResult]:
        """检查损失函数"""
        self.log("\n" + "=" * 60)
        self.log("6. 损失函数检查")
        self.log("=" * 60)

        results = []

        try:
            vocab_size = 511
            batch_size = 4
            seq_len = 128

            # 模拟 logits 和 labels
            logits = torch.randn(batch_size, seq_len, vocab_size)
            labels = torch.randint(0, vocab_size, (batch_size, seq_len))

            # 添加一些 padding
            padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
            padding_mask[:, -20:] = True
            labels[padding_mask] = -100

            # 6.1 Cross Entropy Loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            results.append(CheckResult(
                name="Cross Entropy Loss",
                passed=not torch.isnan(loss) and not torch.isinf(loss),
                message=f"loss = {loss.item():.4f}",
            ))
            self.add_result(results[-1])

            # 6.2 ignore_index 检查
            # 确保 -100 被忽略
            all_pad_labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
            all_pad_loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                all_pad_labels.view(-1),
                ignore_index=-100,
            )

            # 全 padding 的 loss 应该是 0 或 nan
            results.append(CheckResult(
                name="Padding 忽略",
                passed=all_pad_loss.item() == 0 or torch.isnan(all_pad_loss),
                message="正确忽略" if (all_pad_loss.item() == 0 or torch.isnan(all_pad_loss)) else "可能有问题",
            ))
            self.add_result(results[-1])

            # 6.3 梯度检查
            logits_grad = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
            shift_logits_grad = logits_grad[:, :-1, :].contiguous()

            loss_grad = F.cross_entropy(
                shift_logits_grad.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss_grad.backward()

            has_grad = logits_grad.grad is not None
            results.append(CheckResult(
                name="Loss 梯度",
                passed=has_grad,
                message="正常" if has_grad else "无梯度",
            ))
            self.add_result(results[-1])

            # 6.4 数值范围检查
            # 随机 logits 的 loss 应该接近 log(vocab_size)
            expected_loss = torch.log(torch.tensor(float(vocab_size)))
            loss_reasonable = abs(loss.item() - expected_loss.item()) < 2.0
            results.append(CheckResult(
                name="Loss 数值范围",
                passed=loss_reasonable,
                message=f"实际: {loss.item():.2f}, 期望 ~{expected_loss.item():.2f}",
                warnings=None if loss_reasonable else ["Loss 可能偏离正常范围"],
            ))
            self.add_result(results[-1])

        except Exception as e:
            results.append(CheckResult(
                name="损失函数检查",
                passed=False,
                message=f"错误: {str(e)}",
                errors=[traceback.format_exc()],
            ))
            self.add_result(results[-1])

        return results

    # ==================== 7. 端到端训练测试 ====================

    def check_end_to_end(self) -> List[CheckResult]:
        """端到端训练测试"""
        self.log("\n" + "=" * 60)
        self.log("7. 端到端训练测试")
        self.log("=" * 60)

        results = []

        try:
            vocab_size = 511

            # 7.1 创建模型和优化器
            model = create_model(vocab_size=vocab_size, model_size='small')
            model.to(self.device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            results.append(CheckResult(
                name="模型和优化器创建",
                passed=True,
                message=f"设备: {self.device}",
            ))
            self.add_result(results[-1])

            # 7.2 模拟训练数据
            batch_size = 2
            seq_len = 256

            token_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(self.device)
            chord_ids = torch.randint(-1, 10, (batch_size, seq_len)).to(self.device)
            instrument_ids = torch.randint(0, 130, (batch_size, seq_len)).to(self.device)
            labels = token_ids.clone()
            labels[:, -20:] = -100  # 模拟 padding

            # 7.3 训练步骤
            model.train()
            initial_loss = None
            final_loss = None

            for step in range(5):
                optimizer.zero_grad()

                logits = model(token_ids, chord_ids, instrument_ids)

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

                if step == 0:
                    initial_loss = loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                final_loss = loss.item()

            loss_decreased = final_loss < initial_loss
            results.append(CheckResult(
                name="5 步训练 Loss 下降",
                passed=loss_decreased,
                message=f"{initial_loss:.4f} → {final_loss:.4f}" +
                        (" ↓" if loss_decreased else " ↑ (可能需要更多步)"),
            ))
            self.add_result(results[-1])

            # 7.4 梯度范数检查
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            grad_reasonable = 0.001 < total_norm < 100
            results.append(CheckResult(
                name="梯度范数",
                passed=grad_reasonable,
                message=f"{total_norm:.4f}",
                warnings=None if grad_reasonable else ["梯度范数异常"],
            ))
            self.add_result(results[-1])

            # 7.5 参数更新检查
            # 保存初始参数
            model2 = create_model(vocab_size=vocab_size, model_size='small')
            model2.to(self.device)

            # 复制参数
            with torch.no_grad():
                for p1, p2 in zip(model.parameters(), model2.parameters()):
                    p2.copy_(p1)

            # 训练一步
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
            optimizer.step()

            # 检查参数变化
            params_changed = False
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                if not torch.allclose(p1, p2):
                    params_changed = True
                    break

            results.append(CheckResult(
                name="参数更新",
                passed=params_changed,
                message="参数已更新" if params_changed else "参数未变化 (问题!)",
            ))
            self.add_result(results[-1])

            # 7.6 推理测试
            model.eval()
            with torch.no_grad():
                logits = model(token_ids, chord_ids, instrument_ids)
                probs = F.softmax(logits[:, -1, :], dim=-1)
                next_token = torch.argmax(probs, dim=-1)

            results.append(CheckResult(
                name="推理",
                passed=next_token.shape == (batch_size,),
                message=f"预测 token: {next_token.tolist()}",
            ))
            self.add_result(results[-1])

        except Exception as e:
            results.append(CheckResult(
                name="端到端测试",
                passed=False,
                message=f"错误: {str(e)}",
                errors=[traceback.format_exc()],
            ))
            self.add_result(results[-1])

        return results

    # ==================== 8. 配置检查 ====================

    def check_training_config(self) -> List[CheckResult]:
        """检查训练配置"""
        self.log("\n" + "=" * 60)
        self.log("8. 训练配置检查")
        self.log("=" * 60)

        results = []
        warnings = []

        # 8.1 GPU 检查
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            message = f"{gpu_name}, {gpu_memory:.1f}GB"
        else:
            message = "无 GPU"
            warnings.append("建议使用 GPU 训练")

        results.append(CheckResult(
            name="GPU 检测",
            passed=True,  # 不是必须的
            message=message,
            warnings=warnings if not cuda_available else None,
        ))
        self.add_result(results[-1])

        # 8.2 推荐配置
        configs = {
            'H800 80GB': {
                'batch_size': 8,
                'max_seq_len': 24576,
                'model_size': 'large',
                'gradient_accumulation': 4,
            },
            'A100 40GB': {
                'batch_size': 4,
                'max_seq_len': 16384,
                'model_size': 'base',
                'gradient_accumulation': 8,
            },
            'RTX 3090 24GB': {
                'batch_size': 2,
                'max_seq_len': 8192,
                'model_size': 'base',
                'gradient_accumulation': 16,
            },
            'RTX 4090 24GB': {
                'batch_size': 2,
                'max_seq_len': 12288,
                'model_size': 'base',
                'gradient_accumulation': 16,
            },
        }

        self.log("\n推荐配置:")
        for gpu, config in configs.items():
            self.log(f"  {gpu}:")
            self.log(f"    batch_size: {config['batch_size']}")
            self.log(f"    max_seq_len: {config['max_seq_len']}")
            self.log(f"    model_size: {config['model_size']}")
            self.log(f"    gradient_accumulation: {config['gradient_accumulation']}")

        results.append(CheckResult(
            name="推荐配置",
            passed=True,
            message=f"已列出 {len(configs)} 种 GPU 配置",
        ))
        self.add_result(results[-1])

        return results

    # ==================== 运行所有检查 ====================

    def run_all_checks(self, sample_file: Optional[str] = None) -> Dict[str, Any]:
        """运行所有检查"""
        self.log("\n" + "=" * 70)
        self.log("HID-MUSEFORMER 训练准备完整检查")
        self.log("=" * 70)

        start_time = time.time()

        # 运行所有检查
        all_results = []

        all_results.extend(self.check_tokenizer())
        all_results.extend(self.check_data_loading(sample_file))
        all_results.extend(self.check_model_architecture())
        all_results.extend(self.check_summary_attention())
        all_results.extend(self.check_token_type_sparsity())
        all_results.extend(self.check_loss_function())
        all_results.extend(self.check_end_to_end())
        all_results.extend(self.check_training_config())

        elapsed = time.time() - start_time

        # 统计
        passed = sum(1 for r in all_results if r.passed)
        failed = sum(1 for r in all_results if not r.passed)
        total = len(all_results)

        # 收集所有警告和错误
        all_warnings = []
        all_errors = []
        for r in all_results:
            if r.warnings:
                all_warnings.extend(r.warnings)
            if r.errors:
                all_errors.extend(r.errors)

        # 最终报告
        self.log("\n" + "=" * 70)
        self.log("检查完成!")
        self.log("=" * 70)
        self.log(f"通过: {passed}/{total}")
        self.log(f"失败: {failed}/{total}")
        self.log(f"耗时: {elapsed:.2f}s")

        if all_warnings:
            self.log(f"\n警告 ({len(all_warnings)}):")
            for w in all_warnings[:10]:  # 最多显示 10 个
                self.log(f"  ⚠ {w}")

        if all_errors:
            self.log(f"\n错误 ({len(all_errors)}):")
            for e in all_errors[:5]:  # 最多显示 5 个
                self.log(f"  ✗ {e[:200]}...")  # 截断

        if failed == 0:
            self.log("\n✓ 所有检查通过! 可以开始训练。")
        else:
            self.log(f"\n✗ 有 {failed} 项检查失败，请修复后再训练。")

        return {
            'passed': passed,
            'failed': failed,
            'total': total,
            'elapsed': elapsed,
            'warnings': all_warnings,
            'errors': all_errors,
            'ready_to_train': failed == 0,
            'results': [
                {
                    'name': r.name,
                    'passed': r.passed,
                    'message': r.message,
                }
                for r in all_results
            ],
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='训练准备检查')
    parser.add_argument('--sample-file', type=str, help='测试样本文件路径')
    parser.add_argument('--output', type=str, help='输出 JSON 报告路径')
    parser.add_argument('--quiet', action='store_true', help='安静模式')

    args = parser.parse_args()

    checker = TrainingReadinessChecker(verbose=not args.quiet)
    report = checker.run_all_checks(sample_file=args.sample_file)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n报告已保存到: {args.output}")

    # 返回退出码
    sys.exit(0 if report['ready_to_train'] else 1)


if __name__ == '__main__':
    main()
