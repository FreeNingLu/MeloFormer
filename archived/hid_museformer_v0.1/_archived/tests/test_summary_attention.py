#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary Token 注意力机制综合测试

测试内容：
1. 掩码正确性测试
2. 梯度流动测试
3. 信息传递测试
4. 计算复杂度测试
5. 多层堆叠测试
6. 边界情况测试
7. 因果性验证
8. 数值稳定性测试
"""

import torch
import torch.nn as nn
import time
import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention import (
    SummaryAttention,
    SummaryAttentionBlock,
    SummaryAttentionMask,
    SummaryTokenEmbedding,
    FCAttentionMask,
)


def print_header(title: str):
    """打印测试标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(name: str, passed: bool, details: str = ""):
    """打印测试结果"""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {name}")
    if details:
        print(f"         {details}")


class TestSummaryAttention:
    """Summary Token 注意力机制测试类"""

    def __init__(self, embed_dim=256, num_heads=8, num_bars=8, tokens_per_bar=12):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_bars = num_bars
        self.tokens_per_bar = tokens_per_bar
        self.reg_len = num_bars * tokens_per_bar
        self.sum_len = num_bars

        # 创建模块
        self.sum_embedding = SummaryTokenEmbedding(embed_dim, max_bars=256)
        self.mask_generator = SummaryAttentionMask()
        self.attention = SummaryAttention(embed_dim, num_heads, dropout=0.0)
        self.block = SummaryAttentionBlock(embed_dim, num_heads, embed_dim * 4, dropout=0.0)

    def _create_test_data(self, batch_size=2):
        """创建测试数据"""
        # Regular token 嵌入
        reg_x = torch.randn(batch_size, self.reg_len, self.embed_dim)

        # Bar IDs: 每 tokens_per_bar 个 token 属于一个 bar
        bar_ids = torch.arange(self.num_bars).repeat_interleave(self.tokens_per_bar)
        bar_ids = bar_ids.unsqueeze(0).expand(batch_size, -1)

        # Instrument IDs
        instrument_ids = torch.zeros(batch_size, self.reg_len, dtype=torch.long)

        # Summary token 嵌入
        sum_x = self.sum_embedding(self.num_bars, batch_size, reg_x.device)

        # 创建掩码
        masks = self.mask_generator.create_masks(
            bar_ids, instrument_ids, self.num_bars, reg_x.device
        )

        return reg_x, sum_x, bar_ids, instrument_ids, masks

    def test_mask_correctness(self) -> bool:
        """测试 1: 掩码正确性"""
        print_header("测试 1: 掩码正确性")

        batch_size = 2
        reg_x, sum_x, bar_ids, instrument_ids, masks = self._create_test_data(batch_size)

        all_passed = True

        # 1.1 SS 掩码应该是因果的
        ss_expected = torch.tril(torch.ones(self.sum_len, self.sum_len)).bool()
        ss_expected = ss_expected.unsqueeze(0).expand(batch_size, -1, -1)
        ss_correct = torch.all(masks['ss'] == ss_expected).item()
        print_result("SS 掩码因果性", ss_correct)
        all_passed = all_passed and ss_correct

        # 1.2 SR 掩码: S_i 只能看 bar i 的 R
        sr_correct = True
        for b in range(batch_size):
            for bar_idx in range(self.num_bars):
                expected = (bar_ids[b] == bar_idx)
                actual = masks['sr'][b, bar_idx]
                if not torch.all(expected == actual):
                    sr_correct = False
                    break
        print_result("SR 掩码正确性", sr_correct,
                    f"S_i 只能看 bar i 的 R")
        all_passed = all_passed and sr_correct

        # 1.3 RS 掩码: R 只能看已完成 bar 的 S
        rs_correct = True
        for b in range(batch_size):
            for j in range(self.reg_len):
                r_bar = bar_ids[b, j].item()
                expected = torch.zeros(self.num_bars, dtype=torch.bool)
                expected[:r_bar] = True
                actual = masks['rs'][b, j]
                if not torch.all(expected == actual):
                    rs_correct = False
                    break
        print_result("RS 掩码正确性", rs_correct,
                    f"R 只能看已完成 bar 的 S")
        all_passed = all_passed and rs_correct

        # 1.4 RR 掩码应该是因果的
        rr_causal = torch.all(
            masks['rr'] == (masks['rr'] & torch.tril(torch.ones_like(masks['rr'][0])))
        ).item()
        print_result("RR 掩码因果性", rr_causal)
        all_passed = all_passed and rr_causal

        return all_passed

    def test_gradient_flow(self) -> bool:
        """测试 2: 梯度流动"""
        print_header("测试 2: 梯度流动")

        batch_size = 2
        reg_x, sum_x, bar_ids, instrument_ids, masks = self._create_test_data(batch_size)

        reg_x.requires_grad_(True)
        sum_x = sum_x.detach().requires_grad_(True)

        all_passed = True

        # 2.1 前向传播
        sum_out, reg_out = self.attention(
            sum_x, reg_x,
            masks['ss'], masks['sr'], masks['rs'], masks['rr']
        )

        # 2.2 反向传播
        loss = sum_out.sum() + reg_out.sum()
        loss.backward()

        # 2.3 检查梯度
        sum_grad_exists = sum_x.grad is not None and not torch.all(sum_x.grad == 0)
        reg_grad_exists = reg_x.grad is not None and not torch.all(reg_x.grad == 0)

        print_result("Summary 梯度存在", sum_grad_exists)
        print_result("Regular 梯度存在", reg_grad_exists)
        all_passed = all_passed and sum_grad_exists and reg_grad_exists

        # 2.4 检查梯度是否有 NaN
        sum_grad_valid = not torch.isnan(sum_x.grad).any().item()
        reg_grad_valid = not torch.isnan(reg_x.grad).any().item()

        print_result("Summary 梯度无 NaN", sum_grad_valid)
        print_result("Regular 梯度无 NaN", reg_grad_valid)
        all_passed = all_passed and sum_grad_valid and reg_grad_valid

        return all_passed

    def test_information_flow(self) -> bool:
        """测试 3: 信息传递"""
        print_header("测试 3: 信息传递")

        all_passed = True

        # 3.1 测试 SR: 修改 bar 0 的 R，应该只影响 S_0
        batch_size = 1
        reg_x, sum_x, bar_ids, instrument_ids, masks = self._create_test_data(batch_size)

        # 基线输出
        with torch.no_grad():
            sum_base, reg_base = self.attention(
                sum_x, reg_x,
                masks['ss'], masks['sr'], masks['rs'], masks['rr']
            )

        # 修改 bar 0 的 R
        reg_x_mod = reg_x.clone()
        reg_x_mod[:, :self.tokens_per_bar, :] += 10.0

        with torch.no_grad():
            sum_mod, reg_mod = self.attention(
                sum_x, reg_x_mod,
                masks['ss'], masks['sr'], masks['rs'], masks['rr']
            )

        # S_0 应该改变最多（直接受 bar 0 影响）
        sum_diff = (sum_mod - sum_base).abs().mean(dim=-1)  # (batch, sum_len)
        s0_changed = sum_diff[0, 0] > 0.01
        print_result("SR 信息传递: S_0 受 bar 0 影响", s0_changed.item(),
                    f"S_0 变化量: {sum_diff[0, 0]:.4f}")
        all_passed = all_passed and s0_changed.item()

        # 3.2 测试 RS: 修改 S_0，bar 1+ 的 R 应该受影响
        sum_x_mod = sum_x.clone()
        sum_x_mod[:, 0, :] += 10.0

        with torch.no_grad():
            _, reg_mod2 = self.attention(
                sum_x_mod, reg_x,
                masks['ss'], masks['sr'], masks['rs'], masks['rr']
            )

        reg_diff = (reg_mod2 - reg_base).abs().mean(dim=-1)  # (batch, reg_len)

        # bar 0 的 R 不应该受 S_0 影响（因为 RS 掩码）
        bar0_unchanged = reg_diff[0, :self.tokens_per_bar].mean() < 0.01
        # bar 1+ 的 R 应该受影响
        bar1_plus_changed = reg_diff[0, self.tokens_per_bar:].mean() > 0.01

        print_result("RS 信息传递: bar 0 不受 S_0 影响", bar0_unchanged.item(),
                    f"bar 0 变化量: {reg_diff[0, :self.tokens_per_bar].mean():.4f}")
        print_result("RS 信息传递: bar 1+ 受 S_0 影响", bar1_plus_changed.item(),
                    f"bar 1+ 变化量: {reg_diff[0, self.tokens_per_bar:].mean():.4f}")

        all_passed = all_passed and bar0_unchanged.item() and bar1_plus_changed.item()

        return all_passed

    def test_computational_complexity(self) -> bool:
        """测试 4: 计算复杂度"""
        print_header("测试 4: 计算复杂度")

        all_passed = True

        # 测试不同序列长度的计算时间
        configs = [
            (4, 10),   # 40 tokens
            (8, 15),   # 120 tokens
            (16, 20),  # 320 tokens
            (32, 25),  # 800 tokens
        ]

        times = []
        for num_bars, tokens_per_bar in configs:
            reg_len = num_bars * tokens_per_bar

            # 创建数据
            reg_x = torch.randn(1, reg_len, self.embed_dim)
            bar_ids = torch.arange(num_bars).repeat_interleave(tokens_per_bar).unsqueeze(0)
            instrument_ids = torch.zeros(1, reg_len, dtype=torch.long)
            sum_x = self.sum_embedding(num_bars, 1, reg_x.device)

            # 创建掩码
            mask_gen = SummaryAttentionMask()
            masks = mask_gen.create_masks(bar_ids, instrument_ids, num_bars, reg_x.device)

            # 创建 attention
            attn = SummaryAttention(self.embed_dim, self.num_heads, dropout=0.0)

            # 预热
            with torch.no_grad():
                attn(sum_x, reg_x, masks['ss'], masks['sr'], masks['rs'], masks['rr'])

            # 计时
            start = time.time()
            num_runs = 10
            with torch.no_grad():
                for _ in range(num_runs):
                    attn(sum_x, reg_x, masks['ss'], masks['sr'], masks['rs'], masks['rr'])
            elapsed = (time.time() - start) / num_runs * 1000

            times.append((reg_len, elapsed))
            print_result(f"序列长度 {reg_len}", True, f"平均耗时: {elapsed:.2f} ms")

        # 检查复杂度是否接近线性（相对于 O(n²)）
        # 如果是 O(n²)，时间应该增长 ~(n2/n1)²
        # 如果是 O(n)，时间应该增长 ~(n2/n1)
        ratio_actual = times[-1][1] / times[0][1]
        ratio_len = times[-1][0] / times[0][0]
        ratio_quadratic = ratio_len ** 2

        # FC-Attention 应该比纯 O(n²) 好
        is_subquadratic = ratio_actual < ratio_quadratic * 0.8
        print_result("复杂度优于 O(n²)", is_subquadratic,
                    f"实际增长: {ratio_actual:.1f}x, O(n²)预期: {ratio_quadratic:.1f}x")

        all_passed = all_passed and is_subquadratic

        return all_passed

    def test_multi_layer_stacking(self) -> bool:
        """测试 5: 多层堆叠"""
        print_header("测试 5: 多层堆叠")

        batch_size = 2
        num_layers = 4

        # 创建多层 block
        layers = nn.ModuleList([
            SummaryAttentionBlock(self.embed_dim, self.num_heads, self.embed_dim * 4, dropout=0.0)
            for _ in range(num_layers)
        ])

        reg_x, sum_x, bar_ids, instrument_ids, masks = self._create_test_data(batch_size)
        reg_x.requires_grad_(True)
        sum_x = sum_x.detach().requires_grad_(True)

        all_passed = True

        # 前向传播
        for layer in layers:
            sum_x, reg_x = layer(
                sum_x, reg_x,
                masks['ss'], masks['sr'], masks['rs'], masks['rr']
            )

        # 检查输出
        output_valid = not torch.isnan(sum_x).any() and not torch.isnan(reg_x).any()
        print_result(f"{num_layers} 层堆叠输出有效", output_valid)
        all_passed = all_passed and output_valid

        # 反向传播
        loss = sum_x.sum() + reg_x.sum()
        loss.backward()

        # 检查第一层的梯度
        first_layer_grad_valid = all(
            p.grad is not None and not torch.isnan(p.grad).any()
            for p in layers[0].parameters()
        )
        print_result(f"第一层梯度有效", first_layer_grad_valid)
        all_passed = all_passed and first_layer_grad_valid

        return all_passed

    def test_edge_cases(self) -> bool:
        """测试 6: 边界情况"""
        print_header("测试 6: 边界情况")

        all_passed = True

        # 6.1 单个 bar
        num_bars = 1
        tokens_per_bar = 10
        reg_len = num_bars * tokens_per_bar

        reg_x = torch.randn(1, reg_len, self.embed_dim)
        bar_ids = torch.zeros(1, reg_len, dtype=torch.long)
        instrument_ids = torch.zeros(1, reg_len, dtype=torch.long)
        sum_x = self.sum_embedding(num_bars, 1, reg_x.device)

        mask_gen = SummaryAttentionMask()
        masks = mask_gen.create_masks(bar_ids, instrument_ids, num_bars, reg_x.device)

        attn = SummaryAttention(self.embed_dim, self.num_heads, dropout=0.0)

        try:
            with torch.no_grad():
                sum_out, reg_out = attn(
                    sum_x, reg_x,
                    masks['ss'], masks['sr'], masks['rs'], masks['rr']
                )
            single_bar_ok = not torch.isnan(sum_out).any() and not torch.isnan(reg_out).any()
        except Exception as e:
            single_bar_ok = False
            print(f"         Error: {e}")

        print_result("单 bar 场景", single_bar_ok)
        all_passed = all_passed and single_bar_ok

        # 6.2 空 bar (某些 bar 没有 token)
        # 这种情况在实际中可能不会发生，但应该能处理

        # 6.3 大 batch size
        batch_size = 16
        reg_x, sum_x, bar_ids, instrument_ids, masks = self._create_test_data(batch_size)

        try:
            with torch.no_grad():
                sum_out, reg_out = self.attention(
                    sum_x, reg_x,
                    masks['ss'], masks['sr'], masks['rs'], masks['rr']
                )
            large_batch_ok = not torch.isnan(sum_out).any() and not torch.isnan(reg_out).any()
        except Exception as e:
            large_batch_ok = False
            print(f"         Error: {e}")

        print_result(f"大 batch ({batch_size}) 场景", large_batch_ok)
        all_passed = all_passed and large_batch_ok

        return all_passed

    def test_causality(self) -> bool:
        """测试 7: 因果性验证"""
        print_header("测试 7: 因果性验证")

        all_passed = True

        batch_size = 1
        reg_x, sum_x, bar_ids, instrument_ids, masks = self._create_test_data(batch_size)

        # 7.1 修改未来的 token，不应该影响过去的输出
        with torch.no_grad():
            sum_base, reg_base = self.attention(
                sum_x, reg_x,
                masks['ss'], masks['sr'], masks['rs'], masks['rr']
            )

        # 修改最后一个 bar 的 token
        reg_x_mod = reg_x.clone()
        last_bar_start = (self.num_bars - 1) * self.tokens_per_bar
        reg_x_mod[:, last_bar_start:, :] += 100.0

        with torch.no_grad():
            sum_mod, reg_mod = self.attention(
                sum_x, reg_x_mod,
                masks['ss'], masks['sr'], masks['rs'], masks['rr']
            )

        # 除了最后一个 bar，其他 bar 的 R 输出应该不变
        reg_diff = (reg_mod - reg_base).abs()
        earlier_bars_unchanged = reg_diff[:, :last_bar_start, :].max() < 1e-5

        print_result("因果性: 未来修改不影响过去 R", earlier_bars_unchanged.item(),
                    f"过去 bar 最大变化: {reg_diff[:, :last_bar_start, :].max():.2e}")
        all_passed = all_passed and earlier_bars_unchanged.item()

        # 除了最后一个 S，其他 S 应该不变
        sum_diff = (sum_mod - sum_base).abs()
        earlier_s_unchanged = sum_diff[:, :-1, :].max() < 1e-5

        print_result("因果性: 未来修改不影响过去 S", earlier_s_unchanged.item(),
                    f"过去 S 最大变化: {sum_diff[:, :-1, :].max():.2e}")
        all_passed = all_passed and earlier_s_unchanged.item()

        return all_passed

    def test_numerical_stability(self) -> bool:
        """测试 8: 数值稳定性"""
        print_header("测试 8: 数值稳定性")

        all_passed = True

        batch_size = 2

        # 8.1 大输入值
        reg_x, sum_x, bar_ids, instrument_ids, masks = self._create_test_data(batch_size)
        reg_x_large = reg_x * 100

        with torch.no_grad():
            sum_out, reg_out = self.attention(
                sum_x, reg_x_large,
                masks['ss'], masks['sr'], masks['rs'], masks['rr']
            )

        large_input_stable = not torch.isnan(sum_out).any() and not torch.isnan(reg_out).any()
        large_input_stable = large_input_stable and not torch.isinf(sum_out).any() and not torch.isinf(reg_out).any()

        print_result("大输入值稳定性", large_input_stable)
        all_passed = all_passed and large_input_stable

        # 8.2 小输入值
        reg_x_small = reg_x * 0.001

        with torch.no_grad():
            sum_out, reg_out = self.attention(
                sum_x, reg_x_small,
                masks['ss'], masks['sr'], masks['rs'], masks['rr']
            )

        small_input_stable = not torch.isnan(sum_out).any() and not torch.isnan(reg_out).any()

        print_result("小输入值稳定性", small_input_stable)
        all_passed = all_passed and small_input_stable

        # 8.3 半精度测试 (如果支持)
        if torch.cuda.is_available():
            device = torch.device('cuda')
            reg_x_fp16 = reg_x.to(device).half()
            sum_x_fp16 = sum_x.to(device).half()
            masks_fp16 = {k: v.to(device) for k, v in masks.items()}

            attn_fp16 = SummaryAttention(self.embed_dim, self.num_heads, dropout=0.0).to(device).half()

            with torch.no_grad():
                sum_out, reg_out = attn_fp16(
                    sum_x_fp16, reg_x_fp16,
                    masks_fp16['ss'], masks_fp16['sr'], masks_fp16['rs'], masks_fp16['rr']
                )

            fp16_stable = not torch.isnan(sum_out).any() and not torch.isnan(reg_out).any()
            print_result("FP16 稳定性", fp16_stable)
            all_passed = all_passed and fp16_stable
        else:
            print_result("FP16 稳定性", True, "(跳过，无 GPU)")

        return all_passed

    def test_k2v2_effect(self) -> bool:
        """测试 9: K2/V2 二次投影效果"""
        print_header("测试 9: K2/V2 二次投影效果")

        all_passed = True
        batch_size = 2

        reg_x, sum_x, bar_ids, instrument_ids, masks = self._create_test_data(batch_size)

        # 检查 K2, V2 投影是否起作用
        # 如果 K2, V2 是恒等变换，输出应该不同

        # 创建一个 attention，手动设置 K2, V2 为恒等
        attn_identity = SummaryAttention(self.embed_dim, self.num_heads, dropout=0.0)
        with torch.no_grad():
            # 设置 K2, V2 为恒等变换
            nn.init.eye_(attn_identity.sum_k2_proj.weight)
            nn.init.zeros_(attn_identity.sum_k2_proj.bias)
            nn.init.eye_(attn_identity.sum_v2_proj.weight)
            nn.init.zeros_(attn_identity.sum_v2_proj.bias)

            sum_out_identity, reg_out_identity = attn_identity(
                sum_x, reg_x,
                masks['ss'], masks['sr'], masks['rs'], masks['rr']
            )

        # 创建正常的 attention
        attn_normal = SummaryAttention(self.embed_dim, self.num_heads, dropout=0.0)
        with torch.no_grad():
            sum_out_normal, reg_out_normal = attn_normal(
                sum_x, reg_x,
                masks['ss'], masks['sr'], masks['rs'], masks['rr']
            )

        # 输出应该不同
        outputs_different = not torch.allclose(reg_out_identity, reg_out_normal, atol=1e-3)

        print_result("K2/V2 投影产生不同输出", outputs_different)
        all_passed = all_passed and outputs_different

        return all_passed

    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "="*60)
        print("  Summary Token 注意力机制综合测试")
        print("="*60)
        print(f"\n配置: embed_dim={self.embed_dim}, heads={self.num_heads}, "
              f"bars={self.num_bars}, tokens/bar={self.tokens_per_bar}")

        results = {}

        results['mask'] = self.test_mask_correctness()
        results['gradient'] = self.test_gradient_flow()
        results['info_flow'] = self.test_information_flow()
        results['complexity'] = self.test_computational_complexity()
        results['multi_layer'] = self.test_multi_layer_stacking()
        results['edge_cases'] = self.test_edge_cases()
        results['causality'] = self.test_causality()
        results['stability'] = self.test_numerical_stability()
        results['k2v2'] = self.test_k2v2_effect()

        # 总结
        print_header("测试总结")

        total = len(results)
        passed = sum(results.values())

        for name, result in results.items():
            status = "✓" if result else "✗"
            print(f"  {status} {name}")

        print(f"\n  通过: {passed}/{total}")

        if passed == total:
            print("\n  🎉 所有测试通过!")
        else:
            print(f"\n  ⚠️  {total - passed} 个测试失败")

        return passed == total


def test_with_token_types():
    """测试与 Token 类型级稀疏的集成"""
    print_header("附加测试: Token 类型级稀疏集成")

    embed_dim = 256
    num_heads = 8
    num_bars = 4
    tokens_per_bar = 16  # 每个 bar: 4 notes × 4 tokens (T, P, L, V)
    reg_len = num_bars * tokens_per_bar

    # 创建数据
    batch_size = 2
    reg_x = torch.randn(batch_size, reg_len, embed_dim)

    # Bar IDs
    bar_ids = torch.arange(num_bars).repeat_interleave(tokens_per_bar)
    bar_ids = bar_ids.unsqueeze(0).expand(batch_size, -1)

    # Instrument IDs
    instrument_ids = torch.zeros(batch_size, reg_len, dtype=torch.long)

    # Token types: 0=T, 1=P, 2=D, 3=V, 循环
    token_types = torch.tensor([0, 1, 2, 3] * (tokens_per_bar // 4))
    token_types = token_types.repeat(num_bars)
    token_type_ids = token_types.unsqueeze(0).expand(batch_size, -1)

    # Note IDs: 每 4 个 token 一个 note
    note_ids = torch.arange(reg_len // 4).repeat_interleave(4)
    note_ids = note_ids.unsqueeze(0).expand(batch_size, -1)

    # 创建掩码
    mask_gen = SummaryAttentionMask()
    masks = mask_gen.create_masks(
        bar_ids, instrument_ids, num_bars, reg_x.device,
        token_type_ids=token_type_ids,
        note_ids=note_ids,
    )

    # Summary token
    sum_embedding = SummaryTokenEmbedding(embed_dim)
    sum_x = sum_embedding(num_bars, batch_size, reg_x.device)

    # Attention
    attn = SummaryAttention(embed_dim, num_heads, dropout=0.0)

    with torch.no_grad():
        sum_out, reg_out = attn(
            sum_x, reg_x,
            masks['ss'], masks['sr'], masks['rs'], masks['rr']
        )

    output_valid = not torch.isnan(sum_out).any() and not torch.isnan(reg_out).any()
    print_result("Token 类型级稀疏集成", output_valid)

    # 检查 RR 掩码中 D token 是否被隔离
    # D token (type=2) 只能看同音符的其他 token
    d_indices = (token_type_ids[0] == 2).nonzero().squeeze()
    if d_indices.numel() > 0:
        d_idx = d_indices[4].item()  # 取第5个 D token
        d_mask = masks['rr'][0, d_idx]  # 这个 D token 能看到什么

        # D token 应该只能看到同音符的 token
        note_id = note_ids[0, d_idx].item()
        same_note_mask = (note_ids[0] == note_id)

        # D token 对其他音符的 T, P, D, V 应该都看不到
        other_notes_mask = ~same_note_mask
        d_sees_other_notes = d_mask[other_notes_mask].any()

        print_result("D token 不看其他音符", not d_sees_other_notes.item())

    return output_valid


if __name__ == '__main__':
    # 运行主测试
    tester = TestSummaryAttention()
    all_passed = tester.run_all_tests()

    # 附加测试
    token_type_test = test_with_token_types()

    # 最终结果
    print("\n" + "="*60)
    if all_passed and token_type_test:
        print("  ✅ 所有测试通过！Summary Token 注意力机制实现正确。")
    else:
        print("  ❌ 部分测试失败，请检查实现。")
    print("="*60)
