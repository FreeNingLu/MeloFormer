# MeloFormer v1.0.1 发布说明

发布日期: 2024-12-02

## 🎉 主要特性

### 三阶段动态优化

自动调整训练配置以平衡显存使用和训练速度：

| 阶段 | Steps | batch_size | num_workers | GPU 显存 | 目的 |
|------|-------|-----------|-------------|---------|------|
| Phase 1 | 1-10 | 1 | 0 | ~10GB | 编译初期，避免冲突 |
| Phase 2 | 11-100 | 1 | 8 | ~10GB | 加速数据加载，继续编译 |
| Phase 3 | 101+ | 6 | 8 | ~52GB | 全速训练 |

### 性能提升

相比 v0.9.2:
- **编译显存**: 40GB → 10GB (-75%)
- **训练速度**: +50% (batch_size 4→6)
- **GPU 利用率**: Phase 2 后立即提升到 80-95%

## 🔧 核心改进

1. **修复 GPU 利用率低问题**
   - v1.0.0 问题: Phase 2 仍用 num_workers=0，GPU 70% 时间空闲
   - v1.0.1 解决: Step 10 立即切换到 num_workers=8

2. **更保守的切换策略**
   - batch_size 切换延迟到 Step 100
   - 给 FlexAttention 编译更多时间

3. **基于实测数据优化**
   - H800 80GB 实测: batch_size=6 显存峰值 52GB
   - 留 28GB 余量 (35%)

## 📦 文件结构

```
MeloFormer/
├── VERSION                    # 版本信息
├── CHANGELOG.md              # 完整更新日志
├── RELEASE_NOTES.md          # 本文件
├── README.md                 # 使用文档
├── hid_museformer_v1.0/      # 核心代码
├── test_memory_profile.py    # 显存测试工具
├── test_max_seqlen.py        # 序列长度测试工具
├── text2music/               # Diffusion Bridge (开发中)
└── archived/                 # 历史版本归档
    ├── releases/             # v0.8-v1.0.0 归档
    └── test_results/         # 测试结果归档
```

## 🚀 快速开始

### 最新版本文件

- **代码包**: `MeloFormer.tar.gz` (v1.0.1)
- **部署文档**: `服务器部署命令.txt`

### 推荐硬件

| GPU | 推荐度 | 理由 |
|-----|--------|------|
| RTX 6000 Ada (96GB) | ⭐⭐⭐⭐⭐ | 最佳性价比 |
| A100 80GB | ⭐⭐⭐⭐ | 稳定可靠 |
| H800 80GB | ⭐⭐⭐ | 最快但贵 |

### 训练命令

```bash
python train.py \
    --model_size small \
    --batch_size 6 \
    --gradient_accumulation_steps 8 \
    --max_seq_len 8192 \
    --num_workers 8 \
    --epochs 3
```

## 📊 预期效果

### RTX 6000 Ada (推荐)

- batch_size: 8-10 (比 H800 还大！)
- 训练时间: ~50 小时 (3 epochs)
- 总成本: ~¥250
- GPU 利用率: 80-95%

### H800 80GB

- batch_size: 6
- 训练时间: ~40 小时 (3 epochs)
- 总成本: ~¥355
- GPU 利用率: 95-100%

## 🔄 从旧版本升级

如果正在使用 v0.9.x 或 v1.0.0:

1. 停止当前训练: `pkill -f train.py`
2. 上传新版本: `MeloFormer.tar.gz`
3. 解压: `tar -xzf MeloFormer.tar.gz`
4. 重新开始训练

## ⚠️ 已知问题

1. **FlexAttention BF16 限制**
   - PyTorch 2.5-2.7 需要 FP32 workaround
   - 导致 50% 显存开销
   - 等待 PyTorch 官方修复

2. **首次编译时间长**
   - 5-10 分钟正常
   - 动态序列长度需要多次编译
   - cache_size_limit=2048 已优化

## 🐛 报告问题

如遇到问题，请提供：
- GPU 型号和显存
- 训练日志 (train.log)
- nvidia-smi 输出
- 配置参数

## 📝 下一步计划

- [ ] Diffusion Bridge 集成 (text → summary token → MIDI)
- [ ] 多卡训练优化
- [ ] 更多模型规格测试 (base, large)
- [ ] PyTorch 2.6+ FlexAttention BF16 支持验证

---

**完整更新历史**: 见 [CHANGELOG.md](CHANGELOG.md)
