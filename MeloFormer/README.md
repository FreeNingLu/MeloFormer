# MeloFormer v1.7

**基于 Summary Token + FlexAttention 的符号音乐生成模型**

> 专为音乐结构设计的稀疏注意力机制，实现层次化的长序列建模

---

## 核心创新：Summary Token 四阶段注意力

MeloFormer 的核心创新在于 **Summary Token 机制**——一种专门为音乐结构设计的稀疏注意力模式。

### 为什么需要 Summary Token？

传统 Transformer 使用全局 Causal Attention，每个 token 都能看到之前的所有 token。这带来两个问题：

1. **O(N²) 复杂度**：16k 序列需要 256M 个注意力计算
2. **缺乏结构感知**：音乐有天然的 bar（小节）结构，但模型把它当普通序列处理

MeloFormer 的解决方案：

```
┌─────────────────────────────────────────────────────────────┐
│                    Summary Token 机制                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   每个 Bar 一个 Summary Token                                │
│   ┌───┐ ┌───┐ ┌───┐ ┌───┐                                   │
│   │ S₀│ │ S₁│ │ S₂│ │ S₃│  ← Summary Tokens (压缩表示)      │
│   └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘                                   │
│     │     │     │     │                                     │
│   ┌─┴──┐┌─┴──┐┌─┴──┐┌─┴──┐                                  │
│   │Bar0││Bar1││Bar2││Bar3│  ← Regular Tokens (实际音符)     │
│   │····││····││····││····│                                  │
│   └────┘└────┘└────┘└────┘                                  │
│                                                             │
│   远距离: Token 通过 Summary 获取上下文 (O(N×M), M << N)     │
│   近距离: 同 Bar 内直接 Causal Attention                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 四阶段注意力 (SS + SR + RS + RR)

MeloFormer 将注意力分为四个阶段，每个阶段有不同的职责：

| 阶段 | Query → Key | 作用 | 复杂度 |
|------|-------------|------|--------|
| **SS** | Summary → Summary | 粗粒度跨 Bar 因果注意力 | O(M²) |
| **SR** | Summary ← Regular | 从同 Bar 的 Regular tokens 收集信息，更新 Summary | O(M×K) |
| **RS** | Regular → Summary | Regular tokens 通过 Summary 获取远距离上下文 | O(N×M) |
| **RR** | Regular → Regular | 同 Bar 内的细粒度因果注意力 | O(N×K) |

其中：
- N = 序列长度（如 16384）
- M = Bar 数量（如 256）
- K = 每个 Bar 的平均 token 数（约 64）

**总复杂度**：O(N×M + N×K) << O(N²)

### 注意力掩码可视化

```
          ┌─────────────┬─────────────────────────────────────┐
          │  Summary    │           Regular                   │
          │  (256)      │           (16384)                   │
┌─────────┼─────────────┼─────────────────────────────────────┤
│         │             │                                     │
│ Summary │     SS      │              SR                     │
│  (256)  │  (Causal)   │     (Same-Bar Attention)            │
│         │     ▼       │              ◀                      │
├─────────┼─────────────┼─────────────────────────────────────┤
│         │             │                                     │
│ Regular │     RS      │              RR                     │
│ (16384) │ (All Past   │     (Same-Bar Causal)               │
│         │  Summaries) │              ▼                      │
│         │     ▶       │                                     │
└─────────┴─────────────┴─────────────────────────────────────┘

图例:
  ▼ = Causal (下三角)
  ◀ = 信息压缩 (Regular → Summary)
  ▶ = 远距离查询 (Summary → Regular)
```

### 为什么这种设计适合音乐？

1. **Bar 是音乐的自然单位**
   - 一个 Bar 通常包含一个完整的节奏模式
   - Summary Token 压缩这个模式的精华

2. **远距离依赖通过 Summary 传递**
   - 第 100 个 Bar 的音符不需要直接看第 1 个 Bar 的每个音符
   - 只需要看第 1 个 Bar 的 Summary（主旋律、和弦走向）

3. **近距离保持细粒度**
   - 同一个 Bar 内的音符需要精确对齐
   - 使用标准 Causal Attention 保证细节

---

## 技术架构 (v1.7)

### 模型配置

| Size | Params | Layers | Dim | Heads | KV Heads | FFN | GQA Ratio |
|------|--------|--------|-----|-------|----------|-----|-----------|
| small | 17M | 6 | 256 | 4 | 2 | 1408 | 2:1 |
| base | 85M | 12 | 512 | 8 | 4 | 2816 | 2:1 |
| **large** | **397M** | **16** | **768** | **12** | **4** | **4096** | **3:1** |
| xlarge | 450M | 24 | 1024 | 16 | 4 | 5632 | 4:1 |

### 核心组件

#### 1. GQA (Grouped Query Attention)
```python
# 传统 MHA: 12 heads × 12 KV heads = 144 对
# GQA:       12 heads × 4 KV heads  = 48 对 (节省 3x KV Cache)
```

#### 2. 2D RoPE (分层位置编码)
```python
# 标准 RoPE: position = absolute_position
# 2D RoPE:   position = (bar_id, token_in_bar_id)
#
# 好处: 不同 Bar 的相同位置共享位置信息
# 例如: Bar 1 的第 3 拍 和 Bar 50 的第 3 拍 有相似的位置编码
```

#### 3. RMSNorm (替代 LayerNorm)
```python
# LayerNorm: x = (x - mean) / std * gamma + beta
# RMSNorm:   x = x / rms(x) * gamma
#
# 好处: 无需计算 mean，BF16 下更稳定
```

#### 4. SwiGLU FFN
```python
# 标准 FFN:    relu(x @ W1) @ W2
# SwiGLU:      (silu(x @ W_gate) * (x @ W_up)) @ W_down
#
# 好处: 更好的梯度流动，现代 LLM 标配
```

### 亚层级 Gradient Checkpointing (v1.7.4)

```python
# 传统 GC: checkpoint(整个 layer)
# 亚层级 GC: checkpoint(attention) + checkpoint(ffn)
#
# 时序:
# Forward:  [Attn] ─── [FFN] ─── [Attn] ─── [FFN] ───▶
# Backward: ◀─── [FFN] [Attn] ─── [FFN] [Attn] ───
#                  ↑      ↑
#           此时 Attn 中间变量已释放，只重算 FFN
```

---

## HID 编码格式

MeloFormer 使用 **HID (Hierarchical Instrument-aware Duration)** 编码：

```
Token 结构: <Type> <Pitch> <Duration> <Velocity>

示例:
  C4 全音符 forte → [T:note] [P:60] [D:96] [V:100]

特殊 Token:
  - [BAR]     开始新小节
  - [INST:X]  切换乐器
  - [CHORD:X] 和弦标记
```

### 数据格式

```python
{
    'token_ids':      [4623],  # 音乐 token 序列
    'chord_ids':      [4623],  # 每个 token 属于哪个 bar (用于生成 Summary)
    'instrument_ids': [4623],  # 乐器 ID
    'token_type_ids': [4623],  # 0=T, 1=P, 2=D, 3=V
    'note_ids':       [4623],  # 音符组 ID
}
```

---

## 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.5+ (FlexAttention 支持)
- CUDA 12.1+
- H800/A100 80GB GPU

```bash
pip install torch>=2.5.0 datasets tqdm wandb
```

### 训练

#### 基础命令
```bash
python train.py \
    --model_size large \
    --data_dir ~/autodl-tmp/arrow_data \
    --use_arrow \
    --batch_size 1 \
    --max_seq_len 8192 \
    --max_bars 128 \
    --gradient_accumulation_steps 16 \
    --epochs 20 \
    --learning_rate 1e-4 \
    --warmup_steps 2000 \
    --bf16 \
    --gradient_checkpointing \
    --use_packing \
    --output_dir ./outputs_large
```

#### 推荐配置 (H800 80GB)

| 配置 | 序列长度 | 显存 | MIDI 覆盖率 | 说明 |
|-----|---------|------|------------|------|
| 保守 | 8k | ~55GB | 86.6% | 稳定，25GB 余量 |
| 激进 | 12k | ~78GB | 94.0% | 接近极限，风险高 |
| ❌ | 16k | >80GB | 96.9% | OOM |

### 监控

```bash
# 查看日志
tail -f train_meloformer.log

# GPU 状态
watch -n 1 nvidia-smi

# 查看进程
ps aux | grep train.py
```

---

## 性能分析

### FlexAttention 的限制

当前 MeloFormer 使用 PyTorch 2.5 的 FlexAttention，存在以下问题：

| 问题 | 原因 | 影响 |
|-----|------|------|
| **反向传播 OOM** | `sdpa_dense_backward` 回退到 O(N²) | 16k 需要额外 12GB |
| **动态重编译** | 每个序列长度重新编译 Triton kernel | 速度仅 ~800 tokens/s |
| **GQA repeat_interleave** | 物理复制 KV heads | 额外显存开销 |

### 静态编译模式

通过固定输入维度消除重编译：

```bash
python train.py \
    --static_compile \
    --static_max_seq_len 16384 \
    --static_max_bars 256
```

| 模式 | 速度 | 说明 |
|-----|------|------|
| 动态 | ~800 tokens/s | 每 batch 可能重编译 |
| 静态 | ~12,000 tokens/s | 仅首次编译 |

### 与 FlashAttention 对比

| | MeloFormer (FlexAttention) | 标准 Causal + FlashAttention |
|--|---------------------------|------------------------------|
| **显存** | O(N²) backward | O(N) 真正省显存 |
| **速度** | ~800-12k tokens/s | 15,000+ tokens/s |
| **结构感知** | ✅ Bar 级别层次结构 | ❌ 普通序列 |
| **最大序列** | 12k (80GB) | 32k+ (80GB) |

**权衡**：MeloFormer 的 Summary Token 机制为音乐结构提供了归纳偏置，但工程实现上受限于 FlexAttention 的成熟度。

---

## 代码结构

```
MeloFormer/
├── model/
│   ├── meloformer.py              # 主模型入口
│   ├── attention_flex.py          # FlexAttention 基础 + 2D RoPE
│   ├── attention_flex_summary.py  # Summary Token 四阶段注意力
│   └── rms_norm.py                # RMSNorm 实现
├── data/
│   ├── tokenizer_v2.py            # HID Tokenizer
│   ├── midi_to_txt.py             # MIDI → TXT
│   └── txt_to_midi.py             # TXT → MIDI
├── train.py                       # 训练脚本
├── generate.py                    # 生成脚本
├── preprocess_data.py             # MIDI 预处理
├── convert_to_arrow.py            # 转换为 Arrow 格式
└── run_meloformer.sh              # H800 一键启动
```

---

## 版本历史

### v1.7 (2024-12-05) - 当前版本
- **亚层级 Gradient Checkpointing** - Attention 和 FFN 分别 checkpoint
- **序列长度优化** - 确定 12k 为 80GB 显存极限
- **MIDI 覆盖率分析** - 8k=86.6%, 12k=94.0%, 16k=96.9%

### v1.4 (2024-12-04) - 深度优化版
- **GQA** - KV Cache 降低 2-4x
- **RMSNorm** - 替代 LayerNorm
- **2D RoPE** - 分层位置编码

### v1.3 (2024-12-04)
- **静态编译** - 消除 FlexAttention 重编译

### v1.0 (2024-12-01)
- 初始版本：FlexAttention + Summary Token

---

## 当前问题与暂停原因

### 核心问题：FlexAttention 工程成熟度不足

经过深入测试，发现 PyTorch 2.5 的 FlexAttention 在工程层面存在严重问题：

| 问题 | 描述 | 影响 |
|-----|------|------|
| **sdpa_dense_backward** | 反向传播回退到 O(N²) 密集计算 | 16k 序列额外需要 12GB 显存，导致 OOM |
| **动态重编译** | 每个不同的 `(seq_len, num_bars)` 组合触发 Triton kernel 重编译 | 速度仅 ~800 tokens/s（应为 10,000+）|
| **GQA repeat_interleave** | 物理复制 KV heads 而非逻辑广播 | 额外显存开销，抵消 GQA 优势 |

### 实测数据 (H800 80GB)

**训练配置**：
```
硬件: NVIDIA H800 PCIe 80GB
模型: MeloFormer Large (397M params)
数据: 529,036 MIDI 样本 (Arrow 格式)
精度: BF16 + TF32
优化: Gradient Checkpointing ON, 序列打包 ON
```

**显存占用**：
```
16k 序列: 73GB base + 12GB backward = 85GB → OOM ❌
12k 序列: ~78GB (接近极限，可运行但风险高) ⚠️
 8k 序列: ~55GB (稳定，25GB 余量) ✅
```

**训练速度**：
```
实测: 816 tokens/s (动态模式，每 batch 重编译)
预期: 10,000+ tokens/s (如果用 FlashAttention)
差距: 约 12x 慢
```

**MIDI 覆盖率分析** (449,041 文件)：
```
 8k tokens (128 bars): 86.6% 覆盖 (388,678 文件)
12k tokens (192 bars): 94.0% 覆盖 (422,178 文件)
16k tokens (256 bars): 96.9% 覆盖 (435,194 文件)
```

**实际训练日志**：
```
[Warning] Flash Attention not available, using PyTorch SDPA
Arrow 数据集: 529,036 个样本
训练集: 502,585 样本
验证集: 26,451 样本

模型参数量: 397.28M
[✓] Gradient Checkpointing enabled
[✓] 序列打包已启用 (target_length=12288)
[✓] BF16 混合精度 (H800 推荐)
[✓] TF32: matmul=True, cudnn=True

Step 50 | Loss: 6.3978 | LR: 2.50e-06 | Tokens/s: 816
```

### 决策：战略暂停，而非妥协

**这不是放弃，而是务实的产品化决策。**

我们面临一个清晰的权衡：

| 维度 | Summary Token (MeloFormer) | Causal + FlashAttention |
|-----|---------------------------|------------------------|
| **理论优雅性** | ✅ 专为音乐设计 | ❌ 通用序列 |
| **工程成熟度** | ❌ FlexAttention 不成熟 | ✅ 工业级稳定 |
| **开发周期** | 3-6 个月等待生态 | **1 个月出 Demo** |
| **用户价值** | 理论上更好 | **实际可用** |

**核心逻辑**：
1. Summary Token 的**设计理念是正确的**——音乐确实需要结构感知
2. 但 PyTorch FlexAttention 的**工程实现不成熟**——无法发挥设计优势
3. **等待 vs 行动**：等 6 个月让 PyTorch 修复，还是现在用成熟方案出产品？

**我们选择：先出产品，再迭代架构。**

```
短期目标: 1 个月内用 Causal + FlashAttention 出可用 Demo
中期目标: 验证产品市场后，等 PyTorch 2.6+ 回归 Summary Token
长期目标: 在推理层实现 Neuro-Symbolic 可控生成
```

### Summary Token 的价值保留

虽然训练阶段暂时不用 Summary Token，但其核心思想可以在**推理阶段**复活：

**方案：Logits Processor + 音乐规则引擎**

```python
import torch.nn.functional as F

# PyTorch 2.0+ 自动启用 FlashAttention-2
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

| 方案 | 显存 | 速度 | 最大序列 | 音乐结构 |
|-----|------|------|---------|---------|
| **MeloFormer (当前)** | 78GB @ 12k | 800 t/s | 12k | ✅ Summary Token |
| **Causal + FlashAttention** | ~40GB @ 32k | 15,000+ t/s | 32k+ | ❌ 普通序列 |

**可控性的替代方案**：

不在 Attention 层做结构感知，而是在 **推理阶段** 加入逻辑控制：

```python
# Logits Processor 示例
def apply_music_rules(logits, current_chord):
    """在采样前强制执行乐理规则"""
    # 如果当前和弦是 C 大调，禁止生成 F#
    if current_chord == "C_major":
        logits[F_SHARP_TOKEN] = -float('inf')
    return logits
```

这种方式：
1. **训练快**：用标准 Causal Attention + FlashAttention
2. **可控**：推理时插入自定义逻辑
3. **灵活**：可以随时调整规则，无需重新训练

---

## 未来方向

### 短期 (产品化)
1. **切换到 Causal + FlashAttention** - 放弃 Summary Token，换取速度和显存
2. **Logits Processor** - 在推理层实现乐理规则控制
3. **一个月内出 MVP** - 先跑通再优化

### 中期 (等待生态成熟)
1. **PyTorch 2.6+** - 等待 FlexAttention backward 优化
2. **混合方案** - FlashAttention Causal + Cross-Attention to Summary
3. **xformers/flash-attn** - 尝试其他稀疏注意力实现

### 长期 (手搓推理)
1. **纯 Python 推理** - 脱离 PyTorch，嵌入 VST 插件
2. **Neuro-Symbolic** - 在推理层深度整合乐理规则

---

## 许可证

MIT License

---

## 致谢

- [MuseFormer](https://arxiv.org/abs/2210.10349) - Summary Token 原始论文
- [FlexAttention](https://pytorch.org/blog/flexattention/) - PyTorch 官方稀疏注意力
- [flash-attention](https://github.com/Dao-AILab/flash-attention) - 高效注意力实现
