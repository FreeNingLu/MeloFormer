# HID-MuseFormer v1.2.1

基于 HID 编码 + MuseFormer Summary Token 机制的符号音乐生成模型。

## v1.2 核心更新

- ✅ **全量 Gradient Checkpointing** - 每层仅 46MB 开销，压缩 78%
- ✅ **HuggingFace Arrow 数据格式** - 零拷贝内存映射，GPU 利用率 95%+
- ✅ **Large 模型支持** - 400M 参数，H800 单卡可训练

## 环境要求

- Python 3.10+
- PyTorch 2.5+ (FlexAttention)
- CUDA 12.1+

```bash
pip install torch>=2.5.0 datasets tqdm
```

## 模型规格

| Size | Params | Layers | Dim | Heads | FFN | H800 显存 |
|------|--------|--------|-----|-------|-----|----------|
| small | 17M | 6 | 256 | 4 | 1408 | ~6 GB |
| base | 85M | 12 | 512 | 8 | 2816 | ~10 GB |
| **large** | **400M** | **16** | **768** | **12** | **4096** | **~60 GB** |
| xlarge | 450M | 24 | 1024 | 16 | 5632 | ~80 GB |

## 快速开始

### 1. 数据准备

#### 方案 A: 使用现有 .pt 分片（直接训练）

```bash
# 数据目录结构
~/data/processed_data/
├── shard_0000.pt
├── shard_0001.pt
├── ...
└── meta.json
```

#### 方案 B: 转换为 Arrow 格式（推荐，GPU 利用率更高）

```bash
pip install datasets

# 设置缓存目录（避免根目录空间不足）
export HF_HOME=~/autodl-tmp/.hf_cache
export HF_DATASETS_CACHE=~/autodl-tmp/.hf_cache/datasets

python convert_to_arrow.py \
    --input ~/data/processed_data \
    --output ~/data/arrow_data
```

### 2. 训练

#### 使用 Arrow 数据（推荐）

```bash
# 后台运行 + 日志记录
nohup python -u train.py \
    --model_size large \
    --batch_size 4 \
    --max_seq_len 8192 \
    --gradient_accumulation_steps 12 \
    --num_workers 16 \
    --epochs 10 \
    --learning_rate 1e-4 \
    --data_dir ~/autodl-tmp/arrow_data \
    --use_arrow \
    --output_dir ~/autodl-tmp/checkpoints_large \
    > train.log 2>&1 &

# 实时查看日志
tail -f train.log

# 查看进程状态
ps aux | grep train.py

# 查看 GPU 状态
watch -n 1 nvidia-smi
```

#### 使用 .pt 分片数据

```bash
nohup python -u train.py \
    --model_size large \
    --batch_size 4 \
    --max_seq_len 8192 \
    --gradient_accumulation_steps 12 \
    --num_workers 8 \
    --epochs 10 \
    --data_dir ~/data/processed_data \
    --output_dir ~/checkpoints \
    > train.log 2>&1 &
```

#### 使用 screen（推荐，终端断开也不丢失）

```bash
# 安装 screen
apt install screen -y

# 新建 session
screen -S train

# 在 screen 里运行训练
python -u train.py \
    --model_size large \
    --batch_size 4 \
    --max_seq_len 8192 \
    --gradient_accumulation_steps 12 \
    --num_workers 16 \
    --epochs 10 \
    --learning_rate 1e-4 \
    --data_dir ~/autodl-tmp/arrow_data \
    --use_arrow \
    --output_dir ~/autodl-tmp/checkpoints_large

# 断开 session: Ctrl+A, D
# 恢复 session: screen -r train
# 列出所有 session: screen -ls
```

### 3. 生成

```bash
python generate.py \
    --checkpoint ~/checkpoints/best.pt \
    --output generated.mid \
    --temperature 0.9 \
    --max_length 4096
```

### 4. 验证数据预处理

```bash
# 测试 processed_data -> MIDI 解码
python test_decode_midi.py \
    --data_dir ~/autodl-tmp/processed_data \
    --sample_idx 0 \
    --output test_output.mid
```

## 数据格式对比

| 格式 | 加载方式 | GPU 利用率 | 推荐场景 |
|------|---------|-----------|---------|
| `.pt` 分片 | Pickle 反序列化 | 60-80% | 小数据集、快速测试 |
| **Arrow** | 零拷贝内存映射 | **90-95%** | **大规模训练** |

## Gradient Checkpointing 效果

### 显存优化

| 层数 | 无 GC | 有 GC | 节省 |
|------|-------|-------|------|
| 6 | 4943 MB | 4115 MB | 17% |
| 12 | 6197 MB | 4391 MB | 29% |
| 24 | 8724 MB | 4939 MB | **43%** |

### 关键指标

- **每层增量**：46 MB（vs 无 GC 的 210 MB）
- **压缩率**：78%
- **时间开销**：+30-35%
- **核心价值**：batch 可提升 50%+，总吞吐量提升

## H800 推荐配置

| 模型 | Batch | Seq | GA | 显存 | 有效 Batch |
|------|-------|-----|-----|------|-----------|
| large | 4 | 8192 | 12 | ~60 GB | 48 |
| large | 6 | 8192 | 8 | ~70 GB | 48 |
| xlarge | 2 | 8192 | 24 | ~80 GB | 48 |

## 目录结构

```
hid_museformer_v1.0/
├── model/
│   ├── attention_flex.py           # FlexAttention 基础实现
│   ├── attention_flex_summary.py   # Summary Token 核心
│   └── hid_museformer.py           # 主模型 + 全量 GC
├── data/
│   ├── tokenizer_v2.py             # HID Tokenizer
│   ├── midi_to_txt.py              # MIDI → TXT
│   └── txt_to_midi.py              # TXT → MIDI
├── train.py                        # 训练脚本 (支持 --use_arrow)
├── generate.py                     # 生成脚本
├── preprocess_data.py              # MIDI 预处理
├── convert_to_arrow.py             # 转换为 Arrow 格式
├── test_decode_midi.py             # 验证数据预处理
├── test_gc.py                      # GC 效果测试
└── test_gc_layers.py               # GC 层数验证
```

## 架构

**Summary Token + FlexAttention 四阶段注意力**:

| 阶段 | 描述 | 作用 |
|------|------|------|
| SS | Summary → Summary | 粗粒度跨 bar 因果注意力 |
| SR | Summary ← Regular | 信息压缩，每个 Summary 收集对应 bar 的 Regular tokens |
| RS | Regular → Summary | 远距离上下文，Regular tokens 看历史 Summary |
| RR | Regular → Regular | 细粒度近距离，同 bar 内全连接 |

## 常用命令速查

```bash
# 查看训练日志
tail -f train.log

# 查看进程
ps aux | grep train.py

# 查看 GPU
nvidia-smi
watch -n 1 nvidia-smi

# 查看 checkpoint
ls -la ~/autodl-tmp/checkpoints_large/checkpoints/

# 杀掉训练进程
pkill -f train.py

# screen 操作
screen -S train     # 新建
screen -r train     # 恢复
screen -ls          # 列出
# Ctrl+A, D         # 断开
```

## 版本历史

### v1.2.1 (2024-12-04)
- ✅ **修复 CacheLimitExceeded 崩溃** - dynamo cache 从 2048 增加到 16384
- ✅ 添加 `suppress_errors=True` - 超限时回退到 eager 模式而不是崩溃

### v1.2 (2024-12-04)
- ✅ 全量 Gradient Checkpointing（每层 46MB，压缩 78%）
- ✅ HuggingFace Arrow 数据格式支持
- ✅ `--use_arrow` 参数
- ✅ GC 验证测试脚本
- ✅ 数据预处理验证脚本 `test_decode_midi.py`
- ✅ 常用命令速查

### v1.0.1 (2024-12-02)
- 三阶段动态优化
- 序列长度分桶

### v1.0 (2024-12-01)
- 初始版本
- FlexAttention + Summary Token

## 故障排查

### OOM 问题

```bash
# 方案 1: 降低 batch_size
--batch_size 2

# 方案 2: 增加梯度累积
--gradient_accumulation_steps 24

# 方案 3: 降低序列长度
--max_seq_len 4096
```

### GPU 利用率低

1. 使用 Arrow 格式：`--use_arrow`
2. 增加 num_workers：`--num_workers 16`
3. 检查数据是否按长度排序

### 验证 GC 是否生效

```bash
python test_gc.py          # 基础测试
python test_gc_layers.py   # 层数验证（推荐）
```

### CacheLimitExceeded 崩溃

```
torch._dynamo.exc.CacheLimitExceeded: cache_size_limit reached
```

原因：FlexAttention 的 `create_block_mask` 为每个不同的序列长度重编译。

已修复（v1.2.1）：
- `cache_size_limit` 从 2048 增加到 16384
- 添加 `suppress_errors=True` 作为保底

如果仍然崩溃，可以手动增加：
```python
import torch._dynamo
torch._dynamo.config.cache_size_limit = 32768
```

### Arrow 转换空间不足

```bash
# 设置缓存目录到大容量磁盘
export HF_HOME=~/autodl-tmp/.hf_cache
export HF_DATASETS_CACHE=~/autodl-tmp/.hf_cache/datasets

# 清理失败的输出
rm -rf ~/autodl-tmp/arrow_data
rm -rf ~/.cache/huggingface
```

## 许可证

MIT License
