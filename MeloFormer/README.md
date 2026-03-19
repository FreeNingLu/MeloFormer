# MeloFormer

基于 Summary Token + FlexAttention 稀疏注意力的符号音乐生成模型。

## 核心创新

### WSA 风格稀疏注意力

借鉴 Windowed Sink Attention (WSA) 的思路，将音乐序列按 bar（小节）分段，通过 Summary Token 实现跨 bar 信息传递，同时保持 bar 内的细粒度注意力。

注意力分为四个子模式：

| 模式 | 方向 | 作用 |
|------|------|------|
| SS | Summary → Summary | 粗粒度跨 bar 交互（因果） |
| SR | Summary ← Regular | 信息压缩，S 聚合同 bar 的 R |
| RS | Regular → Summary | 获取远距离上下文，R 读取已完成 bar 的 S |
| RR | Regular → Regular | 同 bar 内因果注意力 |

信息流：Summarize 阶段（SS + SR → sum_x2）→ 二次投影（K2, V2）→ Updating 阶段（RS + RR → reg_output）

### 纯算术 mask_mod

v2.0 的关键改进：所有 mask_mod 函数只使用 Python int 常量和索引算术（`bar_id = idx // BAR_LEN`），不捕获任何 GPU 张量。编译一次，永久复用，彻底消除 Triton kernel 重编译问题。

### HID 编码

Hierarchical Instrument-aware Duration-free 编码，用和弦 token 替代传统 BAR token，同时携带小节边界和和声信息。

## 架构

```
Input Token IDs ──→ Token Embedding + Instrument Embedding
                         │
                         ▼
              Summary Token Embedding (可学习，每 bar 一个)
                         │
                    ┌────┴────┐
                    │         │
                 sum_x      reg_x
                    │         │
              ┌─────┴─────────┴─────┐
              │  FlexSummaryBlock ×N │  (Pre-Norm + 残差)
              │  ├─ Attention (SS+SR+RS+RR)
              │  │  ├─ GQA (enable_gqa=True)
              │  │  ├─ 1D RoPE (Summary/Regular 独立)
              │  │  └─ FlexAttention + BlockMask
              │  └─ SwiGLU FFN (Summary/Regular 独立)
              └─────┬─────────┬─────┘
                    │         │
                 sum_x      reg_x
                              │
                         RMSNorm → LM Head (权重共享)
                              │
                           logits
```

## 模型配置

| 名称 | embed | layers | heads | kv_heads | ffn | GQA ratio |
|------|-------|--------|-------|----------|-----|-----------|
| 8m | 256 | 6 | 4 | 2 | 1408 | 2:1 |
| 62m | 512 | 12 | 8 | 4 | 2816 | 2:1 |
| 177m | 768 | 16 | 12 | 6 | 4096 | 2:1 |
| 366m | 1024 | 24 | 16 | 4 | 4096 | 4:1 |
| 600m | 1280 | 28 | 20 | 10 | 5120 | 2:1 |
| 800m | 1408 | 30 | 22 | 11 | 5632 | 2:1 |
| 1.5b | 1536 | 32 | 24 | 12 | 5632 | 2:1 |

GQA ratio 必须是 2 的幂（FlexAttention Triton kernel 要求）。

## HID Token 词汇表（643 tokens）

| 范围 | 名称 | 数量 | 说明 |
|------|------|------|------|
| 0-5 | Special | 6 | PAD, BOS, EOS, MASK, SEP, UNK |
| 6-222 | Chord | 217 | N + 12 根音 × 18 和弦类型 |
| 223-238 | Position | 16 | T0-T15，小节内 16 分音符位置 |
| 239-366 | Pitch | 128 | P0-P127，MIDI 音高 |
| 367-495 | Instrument | 129 | #D（鼓）+ #P0-#P127（GM Program） |
| 496-527 | Tempo | 32 | BPM_0-BPM_31，40-250 BPM 对数量化 |
| 528-546 | TimeSig | 19 | TS_4/4 等 |
| 547-610 | Duration | 64 | L1-L64 |
| 611-642 | Velocity | 32 | V0-V31 |

和弦 token 替代传统 BAR token，同时标记小节边界和和声信息。和弦检测基于 music21 库，支持三和弦、七和弦、挂留和弦等 18 种类型。

## 项目结构

```
MeloFormer/
├── model/
│   ├── meloformer.py              # 主模型：MeloFormer + create_model()
│   ├── attention_flex_summary.py  # 核心：WSA 风格 FlexAttention + Summary Token
│   ├── attention_flex.py          # 基础模块：RoPE, 2D RoPE, FC-Attention mask
│   └── rms_norm.py                # RMSNorm
├── data/
│   ├── tokenizer_v2.py            # HIDTokenizerV2（643 tokens，和弦版）
│   ├── chord_detector_music21.py  # 和弦检测（music21）
│   ├── midi_to_txt.py             # MIDI → TXT 中间格式
│   ├── txt_to_midi.py             # TXT → MIDI
│   ├── filter_midi.py             # MIDI 质量过滤
│   └── build_vocab_minimal.py     # 词表构建
└── train.py                       # H800 训练脚本（DDP, GC, 序列打包, 三阶段优化）
```

## 关键技术细节

### FlexAttention 稀疏加速

- 后端：Triton JIT 编译（非 CUDA C++ kernel）
- 必须配合 `torch.compile` 使用，否则 backward 退化为 dense
- 实测 99.2% 稀疏度下 backward 加速 20x（seq=16384）
- 首次编译约 160s，之后无额外开销

### 训练优化

- 三阶段 DataLoader：Phase 1（编译期 batch=1, workers=0）→ Phase 2（batch=1, workers=8）→ Phase 3（目标 batch, workers=8）
- 序列打包（Sequence Packing）：贪心算法拼接短曲，有效 token 利用率 ~100%
- 亚层级 Gradient Checkpointing：Attention 和 FFN 分别 checkpoint
- 序列长度分桶：512 token 粒度，减少重编译
- Fused AdamW + BF16 + TF32

### 1.5B 模型 Benchmark（单卡 H800 80GB）

| 指标 | 数值 |
|------|------|
| 序列长度 | 16,384 |
| 吞吐量 | 7,165 tokens/sec |
| 单步耗时 | 2,287 ms |
| 峰值显存 | 70.1 GB / 79.2 GB |

## 依赖

```
torch>=2.5.0       # FlexAttention 必需
music21             # 和弦检测
mido                # MIDI 读写
tqdm
wandb               # 可选，训练日志
datasets            # 可选，Arrow 格式数据集
```

## 训练

```bash
# 单卡
python train.py \
    --data_dir /path/to/preprocessed \
    --model_size 62m \
    --bf16 \
    --compile \
    --gradient_checkpointing \
    --batch_size 8 \
    --max_seq_len 16384

# 多卡 DDP
torchrun --nproc_per_node=4 train.py \
    --data_dir /path/to/preprocessed \
    --model_size 366m \
    --bf16 \
    --compile \
    --gradient_checkpointing \
    --batch_size 4 \
    --use_packing
```

## 版本历史

- v2.0: WSA 风格重写，纯算术 mask_mod，消除重编译，enable_gqa=True
- v1.7: 恢复 repeat_interleave（v1.6 的 enable_gqa 触发 dense 退化，v2.0 已修复）
- v1.4: GQA + RMSNorm + 2D RoPE + Gradient Checkpointing
- v1.3: 静态编译模式 + TF32 优化
- v1.1: 序列打包（Sequence Packing）
- v1.0: Summary Token + FlexAttention 初版
