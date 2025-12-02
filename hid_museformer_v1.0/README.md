# MeloFormer v1.0

基于 HID 编码的符号音乐生成模型，使用 Summary Token + FlexAttention 实现高效稀疏注意力。

**v1.0.1 特性**:
- ✅ 三阶段动态优化 (batch_size + num_workers)
- ✅ 智能显存管理 (编译 10GB → 训练 52GB)
- ✅ GPU 利用率优化 (稳定 80-95%)

## 环境要求

- Python 3.10+
- PyTorch 2.5+ (FlexAttention)
- CUDA 12.1+ (训练)

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 训练 (推荐配置)

#### RTX 6000 Ada (96GB) - 推荐 ⭐⭐⭐⭐⭐

```bash
python train.py \
    --model_size small \
    --data_dir ~/autodl-tmp/processed_data \
    --output_dir ~/autodl-tmp/checkpoints \
    --max_seq_len 8192 \
    --batch_size 8 \
    --gradient_accumulation_steps 6 \
    --learning_rate 3e-4 \
    --num_workers 8 \
    --epochs 3
```

**预期效果**:
- 训练时间: ~50-52 小时 (3 epochs)
- 成本: ~¥260
- GPU 利用率: 80-95% (Phase 2 后)
- Tokens/s: ~10000 (Phase 3)

#### H800 80GB

```bash
python train.py \
    --model_size small \
    --data_dir ~/autodl-tmp/processed_data \
    --output_dir ~/autodl-tmp/checkpoints \
    --max_seq_len 8192 \
    --batch_size 6 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-4 \
    --num_workers 8 \
    --epochs 3
```

**预期效果**:
- 训练时间: ~40 小时 (3 epochs)
- 成本: ~¥355
- GPU 利用率: 95-100%
- Tokens/s: ~10000

### 2. 三阶段训练过程

训练过程自动分为三个阶段：

```
Phase 1 (Step 1-10): 编译阶段
├─ batch_size=1, num_workers=0
├─ GPU 显存: ~10GB
└─ 持续时间: ~10 分钟

Phase 2 (Step 11-100): 数据加载优化
├─ batch_size=1, num_workers=8
├─ GPU 显存: ~10GB
├─ GPU 利用率: 80-95% ⬆️
└─ 持续时间: ~45-60 分钟

Phase 3 (Step 101+): 全速训练
├─ batch_size=6-8, num_workers=8
├─ GPU 显存: 52-70GB
├─ GPU 利用率: 95-100%
└─ Tokens/s: 10000+
```

**日志示例**:

```
Step 10:
============================================================
🔄 Phase 2: 切换数据加载
👷 num_workers: 0 → 8
📊 batch_size: 1 (保持)
⚡ 预期: GPU 利用率提升，编译继续...
============================================================

Step 100:
============================================================
🚀 切换到 Phase 3 (高速训练模式)
📊 batch_size: 1 → 6
👷 num_workers: 8 → 8
⚡ 预期加速: 6.0x
============================================================
```

### 3. 监控训练

```bash
# 实时查看日志
tail -f train.log

# 监控 GPU
watch -n 1 nvidia-smi

# 检查训练进度
grep "Phase\|Step.*Loss" train.log | tail -20
```

## 主要参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--model_size` | small | 模型大小: small/base/large/xlarge |
| `--max_seq_len` | 8192 | 最大序列长度 (推荐 8192) |
| `--batch_size` | 6-8 | 目标 batch (Phase 3 使用) |
| `--gradient_accumulation_steps` | 6-8 | 梯度累积步数 |
| `--learning_rate` | 3e-4 | 学习率 |
| `--num_workers` | 8 | 目标 worker 数 (Phase 2/3 使用) |
| `--epochs` | 3-5 | 训练轮数 |

## 模型规格

| Size | Params | Layers | Dim | Heads | 推荐显存 |
|------|--------|--------|-----|-------|---------|
| small | 17M | 6 | 256 | 4 | 80GB |
| base | 85M | 12 | 512 | 8 | 80GB |
| large | 200M | 16 | 768 | 12 | 80GB+ |
| xlarge | 450M | 24 | 1024 | 16 | 80GB+ |

## 架构

**Summary Token + FlexAttention**:
- SS: Summary → Summary (粗粒度跨 bar)
- SR: Summary ← Regular (信息压缩)
- RS: Regular → Summary (远距离上下文)
- RR: Regular → Regular (细粒度近距离)

**稀疏策略**:
- Bar 级: 同乐器全连接，跨乐器选择性
- Token 类型级: T-T, T-P, P-P, V-V 可见

## 故障排查

### OOM 问题

如果遇到 OOM:

```bash
# 方案 1: 降低 batch_size
--batch_size 4  # 或更小

# 方案 2: 增加梯度累积
--gradient_accumulation_steps 16

# 方案 3: 降低序列长度
--max_seq_len 4096
```

### GPU 利用率低

- 确认已经到达 Phase 2 (Step 11+)
- 检查日志中是否有切换信息
- 确认 `num_workers` 已切换到 8

### 训练速度慢

- Phase 1-2 慢是正常的（编译 + 稳定期）
- Phase 3 应达到 8000-10000 tokens/s
- 如果 Phase 3 仍慢，检查磁盘 I/O

## 版本信息

**当前版本**: v1.0.1
**发布日期**: 2024-12-02

查看完整更新日志: [CHANGELOG.md](../CHANGELOG.md)

## 性能基准

### H800 80GB (实测)

```
配置:
- small 模型 (17M)
- batch_size=6, seq_len=8192
- 529K 样本, 3 epochs

结果:
- 编译显存: 10.1 GB
- 训练显存: 52.0 GB
- Phase 3 速度: ~10000 tokens/s
- 总时间: ~40 小时
- 成本: ~¥355
```

### RTX 6000 Ada (预估)

```
配置:
- small 模型 (17M)
- batch_size=8, seq_len=8192
- 529K 样本, 3 epochs

预估:
- 编译显存: ~10 GB
- 训练显存: ~65-70 GB
- Phase 3 速度: ~9000-10000 tokens/s
- 总时间: ~50-52 小时
- 成本: ~¥260
```

## 数据预处理

如果需要从头预处理数据：

```bash
# 1. 生成 MIDI 文件列表
find /path/to/midi/ -name "*.mid" -o -name "*.midi" > midi_files.txt

# 2. 预处理
python preprocess_data.py \
    --input midi_files.txt \
    --output processed_data/ \
    --workers 10 \
    --shard-size 10000
```

## 许可证

MIT License

---

更多信息请查看项目主页: [../README.md](../README.md)
