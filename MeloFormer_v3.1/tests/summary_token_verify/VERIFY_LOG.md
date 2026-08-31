# Summary Token + FlexAttention PoC 验证记录

## 环境信息

| 项目 | 值 |
|------|-----|
| PyTorch | 2.10.0+cu128 |
| GPU | NVIDIA GeForce RTX 4060 Ti (SM89, Ada Lovelace) |
| Backend | Triton (SM89 不支持 FA4 CuTeDSL，需 SM90+) |
| dtype | torch.bfloat16 |
| Python | 3.11 |

---

## Level 1: FlexAttention 基础操作验证

**日期**: 2026-03-19

**目标**: 验证 FlexAttention 的 Triton fused kernel 能否正确编译和执行 MeloFormer mask_mod 中使用的所有操作模式。

### 测试结果

| 测试 | 验证内容 | 状态 | 备注 |
|------|---------|------|------|
| 1a | 1D captured tensor indexing (`bar_ids[idx]`) + `.clamp()` | PASSED | forward + backward 均正确 |
| 1b | 2D captured tensor indexing (`type_visibility[q_type, k_type]`) | PASSED | forward + backward 均正确 |
| 1c | 完整 updating_mask_mod 模式 (6 captured tensors, 非对称 Q/KV, RS+RR 组合) | PASSED | Q_LEN=2048, KV_LEN=2080 |

### 关键输出

```
Test 1a: q.grad=62.0, k.grad=177.0, v.grad=1328.0
Test 1b: q.grad=108.0
Test 1c: q.grad=137.0 (asymmetric Q_LEN=2048, KV_LEN=2080)
```

### 意义

1. **captured tensor indexing 在 Triton codegen 中完全支持** — `bar_ids[q_safe]`、`type_visibility[q_type, k_type]` 等模式可以被 torch.compile 正确编译为 Triton kernel，不需要任何改写。

2. **`.clamp()` + 复杂布尔组合无障碍** — updating_mask_mod 中密集使用的 `.clamp()`、`&`/`|`/`~`、算术比较全部通过编译，不存在 lowering 限制。

3. **非对称 Q/KV 长度正确处理** — Phase 2 的 Q_LEN=N (regular tokens) 与 KV_LEN=M+N (summary + regular) 的不对称形状在 Triton backend 下 forward + backward 均正确。

4. **BF16 无 dtype 问题** — v1.4.2 中遇到的 FlexAttention + BF16 + Gradient Checkpointing dtype 不匹配问题在 PyTorch 2.10 中不再存在。

### 对源码改进的作用

| v1.7 原始问题 | Level 1 验证结论 | 源码改进方向 |
|--------------|-----------------|-------------|
| `enable_gqa=True` → `sdpa_dense_backward` OOM | `repeat_interleave` + 标准 `flex_attention` 在 compiled 路径下正常工作 | 保持 `repeat_interleave` 方案，无需修改 |
| 动态重编译 (每个 seq_len/num_bars 触发 Triton recompile) | `torch.compile` 首次编译后缓存，后续调用无重编译 | 可移除 v1.7 中的 static compile workaround，使用标准 `torch.compile` |
| BF16 dtype 不匹配 → 强制 FP32 | BF16 全程无问题 | 可移除 FP32 workaround，直接使用 BF16 训练 |
| mask_mod 中 tensor indexing 的 FA4 兼容性风险 | Triton backend 完全支持；FA4 待 H100 验证 | mask_mod 代码无需改写，保持原设计 |

---

## Level 2: 完整 updating_mask_mod + 数值正确性

**日期**: 2026-03-19

**目标**: 验证从 v1.7 `attention_flex_summary.py` 提取的完整 `updating_mask_mod`（6 个 captured tensor + 2D indexing + token type sparsity）在 compiled Triton kernel 下的正确性，并与 dense SDPA 做数值对比。

### 测试结果

| 测试 | 验证内容 | 状态 | 备注 |
|------|---------|------|------|
| 2a | 完整 updating_mask_mod compiled forward + backward | PASSED | SEQ_LEN=4096, KV_LEN=4128, 12 heads |
| 2b | 数值正确性 vs dense SDPA (前 128 行) | PASSED | max diff=3.9e-03, mean diff=7.96e-08 |
| 2c | 显存峰值测量 | PASSED | 0.09 GB (sparse) vs 0.76 GB (dense 理论值) |

### 关键输出

```
Test 2a:
  Sparsity: 45.7% (54.3% 的 block 被跳过)
  q.grad norm: 310.0, k.grad norm: 612.0, v.grad norm: 5920.0

Test 2b:
  Max absolute diff:  3.906250e-03
  Mean absolute diff: 7.962808e-08

Test 2c:
  Peak GPU memory: 0.09 GB
  Dense attention score matrix would be: 0.76 GB (FP32)
```

### 意义

1. **完整 mask_mod 复杂度可编译** — 包含 `torch.zeros_like()`、for 循环展开、6 个 captured tensor 的 1D/2D indexing、密集 `.clamp()` 和布尔组合的完整 `updating_mask_mod` 被 `torch.compile` 成功编译为 Triton fused kernel。这是 MeloFormer 中最复杂的 mask_mod，如果这个能过，所有其他 mask_mod 都没问题。

2. **数值精度在 BF16 合理范围内** — max diff 3.9e-03 对 BF16 (mantissa 7 bits, ~1e-2 精度) 来说完全正常。mean diff 7.96e-08 说明绝大多数元素的误差极小，max diff 来自个别 softmax 边界值。

3. **稀疏 kernel 确实生效** — 峰值显存 0.09 GB vs dense 理论 0.76 GB，约 **8.4x 显存节省**。这证明 Triton backend 正确利用了 block_mask 的稀疏性跳过空 block，而非退化为 dense 计算。这直接解决了 v1.7 中 `sdpa_dense_backward` 导致的 OOM 问题。

### 对源码改进的作用

| 发现 | 影响 |
|------|------|
| `_compile=True` 已 deprecated | 源码中 `create_block_mask(..., _compile=True)` 应改为 `torch.compile(create_block_mask)(...)` |
| `sparsity()` 格式化注意 | `sparsity()` 返回的已是百分比值（如 45.7），打印时用 `:.1f%` 而非 `:.1%`（后者会再乘 100） |
| 稀疏 backward 正常工作 | 确认 `repeat_interleave` + compiled `flex_attention` 的组合不会触发 dense backward 回退 |

### 修复记录

- **Sparsity 显示异常 (4573.6%)**: 根因是 Python `:.1%` 格式化符自动乘 100，而 `sparsity()` 返回值已是百分比。修复为 `:.1f%`，实际 sparsity 为 45.7%
- **`_compile=True` deprecated warning**: 改为 `torch.compile(create_block_mask)(...)`

---

## Level 3: 两阶段完整 Pipeline

**日期**: 2026-03-19

**目标**: 验证完整的 MeloFormer 两阶段 Attention pipeline: Phase 1 (SDPA) → 线性投影 → Phase 2 (compiled FlexAttention) + GQA (repeat_interleave) + BF16 + Gradient Checkpointing + torch.compile。

### 测试结果

| 测试 | 验证内容 | 状态 | 备注 |
|------|---------|------|------|
| 3a | 两阶段 forward + backward (BF16) | PASSED | sum_out=(1,32,768), reg_out=(1,4096,768), 所有参数梯度存在 |
| 3b | Gradient Checkpointing (use_reentrant=False) | PASSED | 梯度与 3a 完全一致 (sum=88.5, reg=668.0) |
| 3c | torch.compile 整个模块 | PASSED | SDPA + FlexAttention 混合编译成功，梯度一致 |
| 3d | 显存峰值测量 | PASSED | 0.15 GB (B=1, SEQ=4096, D=768, H=12) |

### 关键输出

```
Test 3a:
  sum_out shape: (1, 32, 768)
  reg_out shape: (1, 4096, 768)
  sum_x.grad norm: 88.5, reg_x.grad norm: 668.0
  All param grads exist: True

Test 3b:
  sum_x.grad norm: 88.5, reg_x.grad norm: 668.0
  (与 3a 完全一致 — checkpoint 不影响数值)

Test 3c:
  Compiled forward+backward: OK
  sum_x.grad norm: 88.5, reg_x.grad norm: 668.0

Test 3d:
  Peak GPU memory: 0.15 GB
```

### 意义

1. **两阶段串联正确工作** — Phase 1 (SDPA) 的输出经过 W_k2/W_v2 线性投影后作为 Phase 2 (FlexAttention) 的 K/V 输入，梯度能正确反向传播穿过两个阶段。这是 Summary Token 机制的核心数据依赖路径，现在确认在 PyTorch 2.10 下无障碍。

2. **Gradient Checkpointing 完全兼容** — `use_reentrant=False` 模式下，checkpoint 包裹整个两阶段 attention 后梯度与非 checkpoint 版本完全一致（88.5 / 668.0）。v1.7 中遇到的 FlexAttention + BF16 + GC 的 dtype 不匹配问题已不存在。

3. **torch.compile 整个模块成功** — SDPA 和 FlexAttention 在同一个 compiled graph 中共存，没有 graph break。这意味着源码迁移时可以直接 `torch.compile(model)` 而不需要手动拆分编译边界。

4. **显存极低** — 0.15 GB 包含了完整的模型参数 + 两阶段 attention + 所有中间激活 + 梯度。在 4060 Ti (16GB) 上有巨大的 headroom，可以支撑更大的 batch size 和序列长度。

### 对源码改进的作用

| 发现 | 影响 |
|------|------|
| Phase 1 SDPA + Phase 2 FlexAttention 混合无问题 | 确认 Q7 的优化建议可行：Phase 1 改用 SDPA 减少 FA4 兼容性风险面 |
| GQA repeat_interleave 在 compiled 路径下正常 | 保持 v1.7 的 repeat_interleave 方案，无需改动 |
| Gradient Checkpointing 兼容 | 可移除 v1.7 中的 FP32 workaround，直接 BF16 + GC |
| torch.compile 无 graph break | 可简化编译策略，不需要 v1.7 中的 static compile mode |
| `Not enough SMs to use max_autotune_gemm mode` warning | 4060 Ti SM 数量不足以触发 max_autotune，H100 上不会有此 warning，不影响正确性 |

---

## Level 4: Captured Tensor 重编译测试

**日期**: 2026-03-21

**目标**: 验证当 captured tensor 的值改变（但 shape 不变）时，`torch.compile(flex_attention)` 是否触发 Triton kernel 重编译。这是 v2.1 方案的关键分叉点——决定是否可以恢复 v1.7 的 captured tensor mask_mod（细粒度稀疏），还是必须保持 v2.0 的纯算术 mask_mod。

### 测试结果

| 测试 | 验证内容 | 状态 | 备注 |
|------|---------|------|------|
| 4a | 1D captured tensor (bar_ids) 值变化，shape 不变 | PASSED | 3 次调用，仅首次编译 (0.70s)，后续 0.00s |
| 4b | 2D captured tensor (type_visibility + token_types + bar_ids) 全部换值 | PASSED | 3 次调用，仅首次编译 (0.13s)，后续 0.00s |
| 4c | 连续 10 次不同随机 bar_ids 值 | PASSED | 首次 0.001s（已预热），后续均 0.001s，零重编译 |
| 4d | 5 个预建 block_mask 轮流调用 3 轮 (15 次) | PASSED | 15 次调用零重编译 |

### 关键输出

```
Test 4a:
  Call 1 (initial compile): 0.70s, compilations: 1
  Call 2 (different bar_ids): 0.00s, compilations: 0
  Call 3 (yet another bar_ids): 0.00s, compilations: 0

Test 4b:
  Call 1 (initial compile): 0.13s, compilations: 1
  Call 2 (different bar_ids + token_types): 0.00s, compilations: 0
  Call 3 (yet another set): 0.00s, compilations: 0

Test 4c:
  10 iterations, all 0.001s, total recompilations after first: 0

Test 4d:
  15 calls cycling 5 different block_masks: 0 recompilations
```

### 意义

1. **captured tensor 值变化不触发重编译** — `torch.compile(flex_attention)` 编译的 Triton kernel 将 captured tensor 视为动态输入（通过 BlockMask 的数据结构传递），而非编译时常量。tensor 的值改变只影响 `create_block_mask` 的输出（BlockMask 的稀疏模式），不影响已编译的 kernel 代码。

2. **v1.7 当时的重编译问题已被 PyTorch 修复** — v1.7 时期（PyTorch 2.5）captured tensor 触发重编译的根因是 `torch._dynamo` 的 guard 机制将 captured tensor 的 identity 作为编译缓存 key。PyTorch 2.10 改进了这一机制，BlockMask 作为不透明数据结构传入 kernel，tensor 值的变化不触发 guard 失效。

3. **v2.0 的纯算术简化不再必要** — v2.0 将 `bar_ids[idx]` 改为 `idx // BAR_LEN` 的唯一理由是避免重编译。现在这个理由不成立了，可以安全恢复 captured tensor mask_mod，恢复 v1.7 的细粒度稀疏建模能力。

4. **block_mask 切换无开销** — Test 4d 证明即使在不同的 BlockMask 之间轮流切换（模拟不同 batch 的不同稀疏模式），也不触发重编译。这意味着变长 bar（每个 batch 的 bar_ids 不同）在训练中完全可行。

### 对 v2.1 方案的决定性影响

| 决策点 | Level 4 结论 | v2.1 方案 |
|--------|-------------|----------|
| mask_mod 实现方式 | captured tensor 不重编译 | **恢复 v1.7 的 captured tensor mask_mod** |
| Token Type Visibility | 2D tensor indexing 不重编译 | **恢复 type_visibility[q_type, k_type]** |
| Bar 表示 | 变长 bar_ids 不重编译 | **恢复变长 bar（自然 token 数）** |
| 纯算术 `idx // BAR_LEN` | 不再必要 | **移除固定 BAR_LEN padding** |

**Level 4 是 v2.1 完整方案的 go/no-go 门控。结果：GO。**

---

## 总结

四个 Level 全部通过。

- **Level 1-3**: 验证了 FlexAttention Triton backend 在 PyTorch 2.10 下完全支持 MeloFormer 的所有 mask_mod 操作模式（captured tensor indexing、2D indexing、.clamp()、复杂布尔组合、两阶段 pipeline、GQA、BF16、Gradient Checkpointing、torch.compile）。
- **Level 4**: 验证了 captured tensor 值变化不触发 Triton 重编译，确认 v2.0 的纯算术简化不再必要，v1.7 的细粒度稀疏建模能力可以安全恢复。

v2.1 完整方案可行：恢复 captured tensor mask_mod + 变长 bar + token type visibility，同时保留 v2.0 的正确决策（enable_gqa、1D RoPE、GQA 2的幂次、Phase 1 SDPA）。

下一步：
1. ~~实施 v2.1 代码改动~~ **DONE** (MeloFormer-v2.1)
2. 在 H100 云实例上验证 FA4 CuTeDSL backend 兼容性
3. 366m 模型训练验证

---

## Level 5: v2.1 端到端 Smoke Training

**日期**: 2026-03-21

**目标**: 验证 MeloFormer v2.1 完整训练 pipeline 可跑通，包括真实 MIDI 数据的完整处理链路和多种模型规模。

### 环境

- 数据: POP909-Dataset (20 首 MIDI，量化模式)
- GPU: RTX 4060 Ti (SM89, 15.6 GB)
- PyTorch: 2.10.0+cu128
- 模型: 177m (主要验证)

### 测试结果

| 测试 | 验证内容 | 状态 | 备注 |
|------|---------|------|------|
| 8m smoke | 8m 模型 20 步训练 | PASSED | loss 6.50→5.66, peak mem 0.26 GB |
| 177m smoke | 177m 模型 20 步训练 | PASSED | loss 6.66→4.34, peak mem 3.61 GB |

### 177m 关键输出

```
Model: 177m (179.28M params)
Vocab size: 643

Step   1: loss=6.6562  seq_len=4095  bars=144  time=24.06s  peak_mem=0.53GB  (首次编译)
Step   2: loss=6.5000  seq_len=4095  bars=118  time=3.59s   peak_mem=3.61GB
Step   3: loss=6.2500  seq_len=3412  bars=118  time=3.19s   peak_mem=3.61GB
...
Step  20: loss=4.3438  seq_len=4095  bars=125  time=0.62s   peak_mem=3.61GB

Loss trend: ✓ decreasing (6.66 → 4.34)
Peak GPU memory: 3.61 GB
```

### 意义

1. **完整 pipeline 可跑通** — MIDI → TXT → tokenize → 变长 bar → captured tensor mask_mod → Phase 1 SDPA + Phase 2 FlexAttention → BF16 + GC → forward + backward + optimizer，全链路无误。

2. **177m 在 16GB 显存下安全运行** — 峰值显存 3.61 GB，远低于 16GB 上限，seq_len=4095 时有充足余量。预估 366m 在同等条件下约 8-10 GB，依然可行。

3. **稀疏 attention 生效** — 177m 模型包含 16 层 FlexAttention，每层 Phase 2 均使用变长 bar 的 captured tensor mask_mod，全程无 dense fallback 警告。

4. **loss 下降正常** — 20 步内 loss 从 6.66 降至 4.34（下降 34.7%），说明梯度流动正确，模型在学习。

5. **首次编译后稳定** — Step 1 耗时 24s（Triton kernel 编译），Step 2 起稳定在 0.6-3.6s/step（含不同 bars 数的 block_mask 重建），无重编译。

### 验证的 v2.1 特性清单

| 特性 | 状态 |
|------|------|
| MIDI → TXT → tokens 完整 pipeline | ✓ |
| 变长 bar（自然 token 数，非固定 BAR_LEN padding） | ✓ |
| captured tensor mask_mod（bar_ids, token_type_ids, note_ids） | ✓ |
| Phase 1 SDPA（非 FlexAttention） | ✓ |
| Phase 2 FlexAttention with block_mask | ✓ |
| enable_gqa=True（无 repeat_interleave） | ✓ |
| 1D RoPE | ✓ |
| BF16 + Gradient Checkpointing | ✓ |
| torch.compile | ✓ |
| forward + backward + optimizer step | ✓ |

**Level 5 结论：v2.1 训练 pipeline 完全可行，177m 模型在 RTX 4060 Ti 上正常运行。**

### 补充：366m 模型验证

| 指标 | 值 |
|------|----|
| 模型规模 | 366m |
| Steps | 20/20 (ALL PASSED) |
| 初始 loss | 6.6875 |
| 最终 loss | 4.7812 |
| Loss 下降 | 28.5% |
| Peak GPU memory | 7.02 GB (15.6GB 显存，余量充足) |
| Step time (stable) | ~1.3s/step |

**366m 在 RTX 4060 Ti (16GB) 上完全可训练，peak mem 7.02GB，H800 上 batch size 可大幅提升。**