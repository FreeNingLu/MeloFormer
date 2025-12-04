# HID-MuseFormer

基于层次化乐器解耦 (Hierarchical Instrument Disentanglement) 的音乐生成 Transformer 模型。

## 目录

1. [环境部署](#环境部署)
2. [数据准备](#数据准备)
3. [数据预处理](#数据预处理)
4. [模型训练](#模型训练)
5. [模型推理](#模型推理)
6. [模型架构](#模型架构)
7. [常见问题](#常见问题)

---

## 环境部署

### 1. 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| GPU | RTX 3090 (24GB) | H800 (80GB) |
| CPU | 8 核 | 64 核 |
| 内存 | 32GB | 128GB |
| 存储 | 100GB SSD | 500GB NVMe |
| CUDA | 11.8+ | 12.1+ |
| Python | 3.9+ | 3.10 |

### 2. 创建 Conda 环境

```bash
# 创建新环境
conda create -n museformer python=3.10 -y
conda activate museformer

# 安装 PyTorch (根据 CUDA 版本选择)
# CUDA 11.8
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. 安装依赖

```bash
# 进入项目目录
cd hid_museformer

# 安装依赖
pip install -r requirements.txt
```

**requirements.txt 内容：**
```
torch>=2.0.0
numpy>=1.21.0
tqdm>=4.60.0
mido>=1.2.10
pretty_midi>=0.2.10
tensorboard>=2.10.0
wandb>=0.13.0  # 可选
```

### 4. 验证安装

```bash
# 验证 PyTorch 和 CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# 验证 Flash Attention (PyTorch SDPA)
python -c "import torch.nn.functional as F; print(f'Flash Attention: {hasattr(F, \"scaled_dot_product_attention\")}')"

# 验证模型
python -c "from model.hid_museformer import create_model; m = create_model(415, 'small'); print(f'Model params: {sum(p.numel() for p in m.parameters())/1e6:.2f}M')"
```

---

## 数据准备

### 1. 需要上传的文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `hid_museformer_h800.tar.gz` | ~100KB | 代码包 |
| `midi_valid_dataset.tar.gz` | ~1.5GB | 45万首有效 MIDI 文件 |

### 2. 上传到服务器

```bash
# 使用 scp
scp hid_museformer_h800.tar.gz midi_valid_dataset.tar.gz user@server:/data/

# 或使用 rsync (支持断点续传，推荐)
rsync -avz --progress hid_museformer_h800.tar.gz midi_valid_dataset.tar.gz user@server:/data/
```

### 3. 服务器上解压

```bash
cd /data

# 解压代码
tar -xzvf hid_museformer_h800.tar.gz

# 解压 MIDI 数据
mkdir -p MIDI
tar -xzvf midi_valid_dataset.tar.gz -C MIDI/
```

### 4. 数据目录结构

```
/data/
├── hid_museformer/          # 代码目录
│   ├── model/
│   ├── data/
│   ├── train_h800.py
│   ├── preprocess_data.py
│   └── README.md
├── MIDI/                    # 原始 MIDI 文件 (解压后)
│   ├── song1.mid
│   ├── song2.mid
│   └── ... (450,453 个文件)
└── processed_data/          # 预处理后的数据 (云端生成)
    ├── song1_xxxx.pt
    ├── song2_xxxx.pt
    ├── meta.json
    └── train_files.txt
```

---

## 数据预处理

预处理将 MIDI 文件转换为 `.pt` 格式，加速训练 10-100x。

**重要**: 推荐在云端服务器上进行预处理（CPU 核心多，速度快）。

### 1. 云端预处理 (推荐)

```bash
cd /data/hid_museformer

# H800 服务器 (64 workers, 约 2 小时处理 45 万文件)
python preprocess_data.py \
    --input /data/MIDI/ \
    --output /data/processed_data/ \
    --workers 64 \
    --max-seq-len 24576
```

### 2. 本地预处理 (可选，较慢)

```bash
# MacBook/普通 PC (10 workers, 约 57 小时)
python preprocess_data.py \
    --input ./MIDI/ \
    --output ./processed_data/ \
    --workers 10 \
    --max-seq-len 24576
```

### 3. 预处理时间估算

| 环境 | Workers | 45万文件预计时间 |
|------|---------|-----------------|
| H800 服务器 | 64 | ~2 小时 |
| RTX 3090 PC | 16 | ~8 小时 |
| MacBook Pro | 10 | ~57 小时 |

### 4. 预处理参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | MIDI 文件目录或文件列表 | 必填 |
| `--output` | 输出目录 | 必填 |
| `--workers` | 并行进程数 | 8 |
| `--max-seq-len` | 最大序列长度 | 24576 |
| `--limit` | 限制处理文件数 (测试用) | None |

### 3. 预处理输出

```
processed_data/
├── song1_a1b2c3d4.pt    # 预处理后的 tensor 文件
├── song2_e5f6g7h8.pt
├── ...
├── meta.json            # 元数据 (文件数、token 总数、vocab 大小)
├── train_files.txt      # 训练文件列表
└── failed.txt           # 失败文件列表 (如有)
```

### 4. 跳过预处理 (运行时转换)

如果不想预处理，可以直接使用 MIDI 文件训练：

```bash
python train_h800.py \
    --data-path /data/MIDI/ \
    --data-mode midi \
    --batch-size 2
```

**注意**: 运行时转换会使训练速度降低 5-10x。

---

## 模型训练

### 1. 快速开始

```bash
# 使用预处理数据训练 Small 模型
python train_h800.py \
    --data-path /data/processed_data/ \
    --model-size small \
    --batch-size 4 \
    --max-seq-len 24576 \
    --epochs 100
```

### 2. 完整训练命令

```bash
python train_h800.py \
    --data-path /data/processed_data/ \
    --model-size base \
    --batch-size 4 \
    --max-seq-len 24576 \
    --epochs 100 \
    --lr 1e-4 \
    --warmup-steps 1000 \
    --gradient-accumulation 4 \
    --gradient-checkpointing \
    --output-dir ./checkpoints \
    --save-every 5 \
    --eval-every 1 \
    --wandb-project hid-museformer
```

### 3. 训练参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data-path` | 数据目录 | 必填 |
| `--data-mode` | 数据模式: token/midi/txt | token |
| `--model-size` | 模型大小: small/base/large/xlarge | small |
| `--batch-size` | 批次大小 | 4 |
| `--max-seq-len` | 最大序列长度 | 24576 |
| `--epochs` | 训练轮数 | 100 |
| `--lr` | 学习率 | 1e-4 |
| `--warmup-steps` | 预热步数 | 1000 |
| `--gradient-accumulation` | 梯度累积步数 | 1 |
| `--gradient-checkpointing` | 启用梯度检查点 | False |
| `--output-dir` | 输出目录 | ./output |
| `--save-every` | 每 N 个 epoch 保存 | 5 |
| `--eval-every` | 每 N 个 epoch 评估 | 1 |
| `--wandb-project` | WandB 项目名 | None |
| `--resume` | 从检查点恢复 | None |

### 4. 模型大小与资源需求

| 模型 | 参数量 | H800 显存 | RTX 3090 显存 | 训练时间/epoch |
|------|--------|-----------|---------------|----------------|
| Small | 14.7M | ~5GB | ~4GB | ~15 分钟 |
| Base | 50M | ~12GB | ~10GB | ~30 分钟 |
| Large | 120M | ~25GB | 不推荐 | ~1 小时 |
| XLarge | 300M | ~50GB | 不支持 | ~2 小时 |

*显存占用基于 batch_size=4, max_seq_len=24576, Flash Attention 启用*

### 5. 多 GPU 训练

```bash
# 使用 torchrun (推荐)
torchrun --nproc_per_node=8 train_h800.py \
    --data-path /data/processed_data/ \
    --model-size large \
    --batch-size 4 \
    --distributed

# 使用 FSDP (超大模型)
torchrun --nproc_per_node=8 train_h800.py \
    --data-path /data/processed_data/ \
    --model-size xlarge \
    --batch-size 2 \
    --fsdp
```

### 6. 监控训练

```bash
# TensorBoard
tensorboard --logdir ./output/logs --port 6006

# 查看 GPU 状态
watch -n 1 nvidia-smi
```

---

## 模型推理

### 1. 生成音乐

```python
from model.hid_museformer import create_model
from data.tokenizer_v2 import HIDTokenizerV2
import torch

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model(vocab_size=415, model_size='small')
model.load_state_dict(torch.load('checkpoints/best_model.pt'))
model.to(device)
model.eval()

# 创建 tokenizer
tokenizer = HIDTokenizerV2()

# 生成
with torch.no_grad():
    generated = model.generate(
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        max_length=4096,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        device=device,
    )

# 解码为 MIDI
output_text = tokenizer.decode(generated[0].tolist())
# 使用 txt_to_midi 转换为 MIDI 文件
```

### 2. 续写音乐

```python
# 读取已有的 MIDI
from data.midi_to_txt import midi_to_txt
midi_to_txt('input.mid', 'input.txt', quantize=True, detect_chords=True)

# 编码为 tokens
with open('input.txt', 'r') as f:
    prompt_text = f.read()
prompt_ids, _ = tokenizer.encode_with_info(prompt_text, add_special=True)

# 续写
prompt_tensor = torch.tensor([prompt_ids], device=device)
with torch.no_grad():
    generated = model.generate(
        prompt=prompt_tensor,
        eos_id=tokenizer.eos_id,
        max_length=8192,
        temperature=0.85,
    )
```

### 3. 生成参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `temperature` | 采样温度 (越高越随机) | 0.8-1.0 |
| `top_k` | Top-K 采样 | 50 |
| `top_p` | Top-P (nucleus) 采样 | 0.9-0.95 |
| `max_length` | 最大生成长度 | 4096-8192 |
| `repetition_penalty` | 重复惩罚 | 1.1 |

---

## 模型架构

### 核心创新

1. **层次化乐器解耦 (HID)**
   - Summary Token: 每个小节每个乐器一个汇总 token
   - 分离乐器信息，提升多乐器协调能力

2. **Fine-Coarse Attention (FC)**
   - 近距离 token 使用完整 attention
   - 远距离 token 通过 Summary Token 交互
   - 显著降低计算复杂度

3. **Flash Attention (SDPA)**
   - 使用 PyTorch 2.0+ 的 scaled_dot_product_attention
   - 自动选择最优 kernel (Flash/Memory-Efficient)
   - 显存占用降低 5-10x

### 模型配置

```python
# Small (14.7M)
{
    'embed_dim': 256,
    'num_heads': 4,
    'num_layers': 6,
    'ff_dim': 1024,
}

# Base (50M)
{
    'embed_dim': 512,
    'num_heads': 8,
    'num_layers': 8,
    'ff_dim': 2048,
}

# Large (120M)
{
    'embed_dim': 768,
    'num_heads': 12,
    'num_layers': 12,
    'ff_dim': 3072,
}

# XLarge (300M)
{
    'embed_dim': 1024,
    'num_heads': 16,
    'num_layers': 16,
    'ff_dim': 4096,
}
```

---

## 常见问题

### Q1: CUDA out of memory

```
解决方案:
1. 减小 batch_size
2. 减小 max_seq_len
3. 启用 gradient_checkpointing
4. 使用更小的模型

示例:
python train_h800.py --batch-size 2 --gradient-checkpointing
```

### Q2: 训练 loss 不下降

```
检查:
1. 学习率是否合适 (尝试 1e-4 到 5e-4)
2. 数据是否正确加载 (检查 token 分布)
3. 是否有 NaN/Inf (检查梯度)

调试:
python train_h800.py --debug --limit-batches 10
```

### Q3: 预处理太慢

```
优化:
1. 增加 workers 数量
2. 使用 SSD 存储
3. 在云端服务器上预处理

H800 推荐: --workers 64
```

### Q4: 如何恢复训练

```bash
python train_h800.py \
    --data-path /data/processed_data/ \
    --resume ./checkpoints/checkpoint_epoch_50.pt
```

### Q5: 生成的音乐质量差

```
调整:
1. 训练更多 epoch
2. 使用更大的模型
3. 调整生成参数 (降低 temperature)
4. 使用更好的数据集
```

---

## 文件结构

```
hid_museformer/
├── model/
│   ├── attention.py         # 注意力机制 (FC Attention, Summary Attention)
│   ├── hid_museformer.py    # 主模型
│   └── embeddings.py        # 嵌入层
├── data/
│   ├── tokenizer_v2.py      # Tokenizer
│   ├── dataset.py           # 数据集类
│   └── midi_to_txt.py       # MIDI 转换
├── train_h800.py            # H800 优化训练脚本
├── preprocess_data.py       # 数据预处理脚本
├── requirements.txt         # 依赖
└── README.md                # 本文档
```

---

## 联系方式

如有问题，请提交 Issue 或联系作者。
