# MeloFormer v2.1

基于 Summary Token + FlexAttention 稀疏注意力的符号音乐生成模型。

## v2.1 相比 v2.0 的核心改进

| 维度 | v2.0 | v2.1 | 决策依据 |
|------|------|------|----------|
| mask_mod | 纯算术 `idx // BAR_LEN` | **captured tensor** (chord_ids, token_type_ids, note_ids) | PoC Level 4 证明 tensor 值变化不触发重编译 |
| Chord 表示 | 固定 BAR_LEN=128 padding | **变长 chord**（自然 token 数） | 消除 37-85% 的 padding 浪费 |
| RR 稀疏 | 同 chord 内全可见 | **Token Type Visibility** (T/P/D/V 可见性矩阵) | 恢复音乐结构归纳偏置 |
| Phase 1 | FlexAttention (极端不对称) | **F.scaled_dot_product_attention** | Q_LEN=num_bars 极小，SDPA 更高效 |
| GQA | enable_gqa=True | enable_gqa=True（保留） | PyTorch 2.10 backward 不再退化 |
| RoPE | 1D RoPE | 1D RoPE（保留） | 2D RoPE 已证伪 |
| 遗留代码 | 含 2D RoPE、dead code | **已清理** | 移除 HierarchicalRoPE、_checkpoint_forward 等 |

## 核心架构

### Summary Token 稀疏注意力

将音乐序列按 chord（和弦段）分段，通过 Summary Token 实现跨 chord 信息传递。注意力分四个子模式：

| 模式 | 方向 | 作用 | 实现 |
|------|------|------|------|
| SS | Summary → Summary | 粗粒度跨 chord 交互（因果） | Phase 1 SDPA |
| SR | Summary ← Regular | 信息压缩，S 聚合同 chord 的 R | Phase 1 SDPA |
| RS | Regular → Summary | 获取远距离上下文 | Phase 2 FlexAttention |
| RR | Regular → Regular | 同 chord 内因果 + Token Type Visibility | Phase 2 FlexAttention |

信息流：
```
Phase 1 (SDPA):       SS + SR → sum_attn_out
中间投影:             sum_attn_out → K2, V2
Phase 2 (FlexAttn):   RS + RR → reg_output
```

### Token Type Visibility (v2.1 恢复)

每个音符由 T(时间)/P(音高)/D(时值)/V(力度) 四元组构成，v2.1 恢复了基于 token 类型的可见性矩阵：

```python
TOKEN_TYPE_VISIBILITY = torch.tensor([
    # T  P  D  V
    [1, 1, 1, 1],  # T 可看所有
    [1, 1, 0, 0],  # P 只看 T/P
    [1, 1, 1, 0],  # D 只看 T/P/D
    [1, 1, 1, 1],  # V 可看所有
], dtype=torch.bool)
```

### 变长 Chord（v2.1 恢复）

v2.1 使用真实的 chord 边界（通过 `chord_ids` captured tensor 传入 mask_mod），不做 BAR_LEN 对齐 padding。每个 batch 的 `chord_ids` 不同，PoC Level 4 证明这不会触发 Triton 重编译。

## 模型结构

```
Input Token IDs ──→ Token Embedding + Instrument Embedding
                         │
                         ▼
              Summary Token Embedding (可学习，每 chord 一个)
                         │
                    ┌────┴────┐
                    │         │
                 sum_x      reg_x
                    │         │
              ┌─────┴─────────┴─────┐
              │  FlexSummaryBlock ×N │  (Pre-Norm + 残差)
              │  ├─ Phase 1: SDPA (SS+SR)
              │  │  ├─ GQA (enable_gqa=True)
              │  │  ├─ 1D RoPE
              │  │  └─ 显式 bool mask
              │  ├─ K2/V2 二次投影
              │  ├─ Phase 2: FlexAttention (RS+RR)
              │  │  ├─ GQA (enable_gqa=True)
              │  │  ├─ 1D RoPE
              │  │  └─ captured tensor block_mask
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

| 名称 | embed | layers | heads | kv_heads | ffn | GQA ratio | 参数量 |
|------|-------|--------|-------|----------|-----|-----------|--------|
| 8m   | 256   | 6      | 4     | 2        | 1408 | 2:1      | ~16M   |
| 62m  | 512   | 12     | 8     | 4        | 2816 | 2:1      | ~62M   |
| 177m | 768   | 16     | 12    | 6        | 4096 | 2:1      | ~179M  |
| 366m | 1024  | 24     | 16    | 4        | 4096 | 4:1      | ~366M  |
| 600m | 1280  | 28     | 20    | 10       | 5120 | 2:1      | ~600M  |
| 800m | 1408  | 30     | 22    | 11       | 5632 | 2:1      | ~800M  |
| 1.5b | 1536  | 32     | 24    | 12       | 5632 | 2:1      | ~1.5B  |

> GQA ratio 必须是 2 的幂次（FlexAttention `enable_gqa=True` 约束）。
> 366m 使用 4:1，其余使用 2:1。

## Smoke Training 验证结果

在 RTX 4060 Ti (16GB, SM89) + PyTorch 2.10 上，使用 POP909 数据集 20 首歌训练 20 步：

| 模型 | Loss (初→末) | Peak Mem | Step Time | 状态 |
|------|-------------|----------|-----------|------|
| 8m   | 6.50 → 5.66 | 0.26 GB  | ~0.07s    | PASSED |
| 177m | 6.66 → 4.34 | 3.61 GB  | ~0.62s    | PASSED |
| 366m | 6.69 → 4.78 | 7.02 GB  | ~1.30s    | PASSED |

**366m 在 16GB 显存下 peak mem 7.02 GB，余量充足。** H800 上 batch_size 可大幅提升。

## 显存估算（H800 80GB, BF16, batch=1）

| 模型 | 参数 | 优化器 | 梯度 | 激活(GC) | 总计 |
|------|------|--------|------|---------|------|
| 177m | 0.36 GB | 1.4 GB | 0.36 GB | ~5 GB  | ~7 GB  |
| 366m | 0.73 GB | 2.9 GB | 0.73 GB | ~10 GB | ~14 GB |
| 1.5b | 3.0 GB  | 12 GB  | 3.0 GB  | ~40 GB | ~58 GB |

## 稀疏注意力效率

以 seq_len=16384, num_chords=128 为例：

| 计算 | Dense | v2.1 稀疏 | 节省 |
|------|-------|-----------|------|
| Phase 2 有效计算量 | N² = 268M | ~4.2M (RR+RS) | ~64x 理论 |
| 实际 wall-clock 加速 | — | 8-15x | (含 kernel overhead) |
| Backward 显存 | O(N²) | O(nnz) | ~8-15x |

## 数据格式

### HID 编码

Hierarchical Instrument-aware Duration-free 编码：

```
[BOS] [BPM_x] [TS_4/4]
[Chord_Cmaj]  # chord 0
[#P0] [SEP] T0 P60 L4 V8  T4 P62 ...   # 乐器 0 的音符
[#P25] [SEP] T0 P48 ...               # 乐器 1 的音符
[Chord_Amin]  # chord 1
...
[EOS]
```

词汇表大小：643 tokens（含 217 个和弦 token）

### 训练数据格式

每个样本包含：
- `token_ids`: (L,) 主 token 序列
- `chord_ids`: (L,) 每个 token 的 chord 编号（变长）
- `chord_ids`: (L,) 和弦索引（每个 token 归属的和弦段 ID）
- `token_type_ids`: (L,) T/P/D/V/-1
- `note_ids`: (L,) 音符组 ID
- `instrument_ids`: (L,) 乐器 ID

## 依赖

```
torch>=2.10.0      # FlexAttention + enable_gqa=True 修复
music21            # 和弦检测
mido               # MIDI 读写
tqdm
wandb              # 可选，训练日志
datasets           # 可选，Arrow 格式数据集
```

## 训练

### 重要约束

v2.1 要求 `batch_size=1`（FlexAttention captured tensor mask_mod 是 batch 共享的），使用 `--gradient_accumulation_steps` 控制有效批大小。

### 快速验证

```bash
# smoke test（无需预处理数据，直接从 POP909 MIDI 跑）
cd MeloFormer-v2.1
python tests/smoke_train.py
```

### 使用 train.sh 启动（H800 推荐）

`train.sh` 自动在 tmux session 中启动训练，支持 nohup 独立运行，断开 SSH 不影响训练进程。
通过 `--mode` 选择训练脚本：`causal`（默认，预训练）或 `fim`（FIM 四模式微调）。

```bash
# 配置路径（编辑 train.sh 中的 MIDILM_DIR）
vim train.sh

# Causal LM 预训练（默认，366m，20 epochs）
./train.sh

# FIM 微调（从预训练权重启动）
./train.sh --mode fim --pretrained runs/xxx/checkpoints/best.pt

# FIM 微调（调整 FIM 比例和 mode 权重）
./train.sh --mode fim --pretrained runs/xxx/checkpoints/best.pt \
    --fim-ratio 0.8 --mode-weights 0.2,0.3,0.3,0.2

# 断点续训（causal 或 fim 均可）
./train.sh --resume runs/xxx/checkpoints/step_10000.pt
./train.sh --mode fim --resume runs/xxx/checkpoints/step_5000.pt

# 自定义参数
./train.sh --lr 3e-5 --epochs 10
./train.sh --model 177m --run-name my_177m_run

# 监控训练
tmux attach -t meloformer_train        # 切回 causal 训练窗口
tmux attach -t meloformer_fim          # 切回 fim 训练窗口
tail -f runs/<run_name>/logs/stdout.log  # 查看标准输出
tail -f runs/<run_name>/logs/train*.log  # 查看结构化日志
pkill -f train.py                      # 停止 causal 训练
pkill -f train_fim.py                  # 停止 FIM 训练
```

**train.sh 默认超参：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | causal | `causal` = 预训练；`fim` = FIM 微调 |
| model_size | 366m | — |
| epochs | 20 | — |
| learning_rate | 1e-4 | — |
| warmup_steps | 2000 | — |
| gradient_accumulation_steps | 8 | — |
| max_seq_len | 24576 | — |
| max_chords | 2048 | — |
| num_workers | 16 | — |
| fim_ratio | 0.7 | FIM 模式下 FIM 样本比例 |
| mode_weights | 0.3,0.25,0.25,0.2 | FIM 模式下 CAUSAL/TRACK/SKELETON/TIME 权重 |

### 手动启动 — Causal LM 预训练

```bash
# 单卡
python train.py \
    --data_dir /path/to/arrow_dataset \
    --use_arrow \
    --model_size 366m \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --bf16 --compile --gradient_checkpointing \
    --max_seq_len 24576 --max_chords 2048 \
    --learning_rate 1e-4 --warmup_steps 2000 \
    --output_dir ./runs/my_run

# 多卡 DDP（H800 × 8）
torchrun --nproc_per_node=8 train.py \
    --data_dir /path/to/arrow_dataset \
    --use_arrow \
    --model_size 366m \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --bf16 --compile --gradient_checkpointing \
    --max_seq_len 24576 --num_workers 16
```

### 手动启动 — FIM 微调

`train_fim.py` 同时支持 4 种训练模式（单脚本混合训练，无需分开跑）：

| 模式 | 说明 | 默认权重 |
|------|------|----------|
| CAUSAL | 标准因果语言模型，防止灾难性遗忘 | 30% |
| TRACK_MASK | 遮蔽整个乐器轨道，用于自动编曲/声部生成 | 25% |
| SKELETON_FIM | 给定乐器+和弦骨架，生成音符 | 25% |
| TIME_FIM | 遮蔽单轨时间段，用于片段填充 | 20% |

```bash
# 从预训练权重启动（推荐）
python train_fim.py \
    --data_dir /path/to/arrow_dataset \
    --use_arrow \
    --model_size 366m \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --bf16 --compile --gradient_checkpointing \
    --max_seq_len 24576 --max_chords 2048 \
    --learning_rate 5e-5 --warmup_steps 500 \
    --fim_ratio 0.7 \
    --mode_weights 0.3,0.25,0.25,0.2 \
    --pretrained ./runs/causal_run/checkpoints/best.pt \
    --output_dir ./runs/fim_run

# 多卡 DDP（H800 × 8）
torchrun --nproc_per_node=8 train_fim.py \
    --data_dir /path/to/arrow_dataset \
    --use_arrow \
    --model_size 366m \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --bf16 --compile --gradient_checkpointing \
    --pretrained ./runs/causal_run/checkpoints/best.pt \
    --output_dir ./runs/fim_ddp_run

# 断点续训
python train_fim.py \
    --data_dir /path/to/data --use_arrow \
    --model_size 366m --batch_size 1 \
    --bf16 --compile --gradient_checkpointing \
    --resume ./runs/fim_run/checkpoints/step_5000.pt
```

### 训练日志

每次训练会在 `runs/<run_name>/logs/` 下生成：

| 文件 | 内容 |
|------|------|
| `stdout.log` | tee 标准输出（tqdm 进度条 + 关键消息） |
| `train_<ts>.log` | 结构化日志（每步指标，带时间戳） |
| `train_fim_<ts>.log` | FIM 专用日志（含 per-mode 损失） |
| `summary.txt` | 训练结束自动生成的汇总表格 |

训练结束后汇总表示例：
```
========================================================================
  训练汇总 - Per-Step 统计
========================================================================
   Step  Epoch      Loss        LR   Tok/s  ms/step  Mem(G)
------------------------------------------------------------------------
    100      1    2.3451  9.98e-05   18420      214    12.34
    200      1    2.1203  1.00e-04   18890      210    12.38
...
    avg              ...            18600      212    12.36
```

### 关键参数说明

#### 通用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch_size` | 1 | v2.1 固定为 1 |
| `--gradient_accumulation_steps` | 4 | 有效批大小 = batch_size × steps × world_size |
| `--max_seq_len` | 24576 | H800 推荐 24576；16GB 显卡推荐 4096-8192 |
| `--max_chords` | 2048 | 覆盖 99.8% 样本 |
| `--use_arrow` | — | 使用 HuggingFace Arrow 格式，推荐云训练使用 |
| `--use_packing` | — | 序列打包，提升短序列利用率 |
| `--num_workers` | 16 | H800 推荐 16；单卡调试可用 0 |

#### FIM 专用参数（train_fim.py）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pretrained` | — | 预训练 checkpoint 路径（强烈推荐） |
| `--fim_ratio` | 0.7 | FIM 样本占比（其余为 CAUSAL） |
| `--mode_weights` | 0.3,0.25,0.25,0.2 | CAUSAL/TRACK/SKELETON/TIME 损失权重 |
| `--curriculum_learning` | — | 启用 curriculum（逐渐增加 FIM 比例） |
| `--scheduler` | wsd | `wsd`（推荐）或 `cosine` |

## 版本历史

- **v2.1**: 恢复 captured tensor mask_mod + 变长 chord + Token Type Visibility；Phase 1 改用 SDPA；清理遗留代码；修复 batch_size>1 约束、torch.compile 缓存、GC autocast_dtype 传递、max_seq_len 硬编码等关键 bug
- v2.0: WSA 风格重写，纯算术 mask_mod，enable_gqa=True，1D RoPE
- v1.7: repeat_interleave workaround（v1.6 的 enable_gqa 触发 dense 退化）
- v1.4: GQA + RMSNorm + 2D RoPE + Gradient Checkpointing
- v1.3: 静态编译模式 + TF32 优化
- v1.1: 序列打包（Sequence Packing）
- v1.0: Summary Token + FlexAttention 初版