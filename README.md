# MeloFormer - HID-MuseFormer v1.2.1

基于 HID (Hierarchical Instrument-aware Duration-free) 编码的音乐生成模型，使用 FlexAttention + Summary Token 机制。

## 🎵 特性

- **HID 编码**: 层次化乐器感知无时值编码，更适合多声部音乐
- **FlexAttention**: PyTorch 2.5+ 可编程注意力框架，支持自定义稀疏模式
- **Summary Token**: 双层注意力机制（Fine-grained + Coarse-grained）
- **Arrow 数据格式**: HuggingFace Arrow 零拷贝内存映射，GPU 利用率 95%+
- **Gradient Checkpointing**: 每层仅 46MB 开销，压缩 78%
- **长序列支持**: 最大支持 seq_len=8192+

## 📊 模型规格

| 模型 | 参数量 | embed_dim | num_layers | num_heads | H800 显存 |
|------|--------|-----------|------------|-----------|---------|
| small | 17M | 256 | 6 | 4 | ~6 GB |
| base | 85M | 512 | 12 | 8 | ~10 GB |
| large | 400M | 768 | 16 | 12 | ~60 GB |
| xlarge | 450M | 1024 | 24 | 16 | ~80 GB |

## 🚀 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.5+
- CUDA 12.1+

### 安装

```bash
pip install torch>=2.5.0 datasets tqdm
```

### 数据准备

#### 方案 A: 使用 .pt 分片（快速测试）

```bash
# 数据目录结构
~/data/processed_data/
├── shard_0000.pt
├── shard_0001.pt
└── meta.json
```

#### 方案 B: 转换为 Arrow 格式（推荐）

```bash
pip install datasets

# 设置缓存目录
export HF_HOME=~/autodl-tmp/.hf_cache
export HF_DATASETS_CACHE=~/autodl-tmp/.hf_cache/datasets

python convert_to_arrow.py \
    --input ~/data/processed_data \
    --output ~/data/arrow_data
```

### 训练

```bash
cd hid_museformer_v1.0

# 使用 Arrow 数据（推荐）
python train.py \
    --model_size large \
    --batch_size 4 \
    --max_seq_len 8192 \
    --gradient_accumulation_steps 12 \
    --num_workers 16 \
    --epochs 10 \
    --data_dir ~/autodl-tmp/arrow_data \
    --use_arrow \
    --output_dir ~/autodl-tmp/checkpoints_large
```

## 📝 版本历史

### v1.2.1 (2024-12-04)
- 修复 CacheLimitExceeded 崩溃 - dynamo cache 从 2048 增加到 16384
- 添加 `suppress_errors=True` - 超限时回退到 eager 模式

### v1.2 (2024-12-04)
- 全量 Gradient Checkpointing（每层 46MB，压缩 78%）
- HuggingFace Arrow 数据格式支持
- `--use_arrow` 参数
- GC 验证测试脚本

### v1.0.1 (2024-12-02)
- 三阶段动态优化
- GPU 利用率修复

## 📄 License

MIT License

## 🙏 致谢

基于 MuseFormer 架构改进，感谢原作者的开创性工作。
