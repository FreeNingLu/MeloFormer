# MeloFormer v0.8

基于 HID 编码的符号音乐生成模型，使用 Summary Token + FlexAttention 实现高效稀疏注意力。

## 环境要求

- Python 3.10+
- PyTorch 2.5+ (FlexAttention)
- CUDA 12.0+ (训练)

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 解压部署

```bash
# 解压到服务器
cd ~/autodl-tmp && tar -xzvf hid_museformer_v0.8.tar.gz
```

### 2. 训练

#### 快速测试 (验证流程)

```bash
cd ~/autodl-tmp/hid_museformer_v0.8 && python train.py \
    --data_dir ~/autodl-tmp/processed_data_mini \
    --model_size small \
    --max_seq_len 2048 \
    --max_bars 128 \
    --max_samples 50 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-4 \
    --epochs 1
```

#### 单 GPU 训练

```bash
cd ~/autodl-tmp/hid_museformer_v0.8 && python train.py \
    --data_dir ~/autodl-tmp/processed_data \
    --model_size base \
    --max_seq_len 24576 \
    --max_bars 2048 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-4 \
    --epochs 100
```

#### 多 GPU (DDP) 训练

```bash
cd ~/autodl-tmp/hid_museformer_v0.8 && torchrun --nproc_per_node=8 train.py \
    --data_dir ~/autodl-tmp/processed_data \
    --model_size large \
    --max_seq_len 24576 \
    --max_bars 2048 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-4 \
    --epochs 100
```

### 3. 数据预处理

```bash
# 生成 MIDI 文件列表
find MIDI/ -name "*.mid" -o -name "*.midi" > midi_files.txt

# 预处理
python preprocess_data.py \
    --input midi_files.txt \
    --output processed_data/ \
    --workers 10 \
    --shard-size 10000
```

## 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_size` | base | 模型大小: small/base/large/xlarge |
| `--max_seq_len` | 24576 | 最大序列长度 |
| `--max_bars` | 2048 | 最大 bar 数量 (超过的样本自动跳过) |
| `--batch_size` | 4 | 每 GPU 批大小 |
| `--gradient_accumulation_steps` | 8 | 梯度累积步数 |
| `--learning_rate` | 3e-4 | 学习率 |
| `--epochs` | 100 | 训练轮数 |

## 模型规格

| Size | Params | Layers | Dim | Heads |
|------|--------|--------|-----|-------|
| small | 17M | 6 | 256 | 4 |
| base | 85M | 12 | 512 | 8 |
| large | 200M | 16 | 768 | 12 |
| xlarge | 450M | 24 | 1024 | 16 |

## 架构

**Summary Token + FlexAttention**:
- SS: Summary -> Summary (粗粒度跨 bar)
- SR: Summary <- Regular (信息压缩)
- RS: Regular -> Summary (远距离上下文)
- RR: Regular -> Regular (细粒度近距离)

**稀疏策略**:
- Bar 级: 同乐器全连接，跨乐器选择性
- Token 类型级: T-T, T-P, P-P, V-V 可见
