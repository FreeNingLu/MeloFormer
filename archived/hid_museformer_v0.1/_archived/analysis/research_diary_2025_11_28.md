# 研究日记 - 2025年11月28日

## 主题：小节层面相关性统计分析 & 掩码策略设计

---

## 一、同乐器不同小节相关性分析

### 1.1 PitchClass 相似度统计

| Offset | PitchClass 相似度 | 观察 |
|--------|------------------|------|
| 1 | 0.487 | 高（相邻小节） |
| 2 | 0.492 | 高 |
| 3 | 0.444 | 下降 |
| **4** | **0.528** ⭐ | **最高！验证了 4 小节循环** |
| 5 | 0.423 | 下降 |
| 6 | 0.454 | - |
| 7 | 0.421 | - |
| **8** | **0.517** ⭐ | 高（8 小节循环） |
| 12 | 0.464 | 高 |
| 16 | 0.471 | 高 |
| 24 | 0.466 | 高 |
| 32 | 0.459 | 高 |

### 1.2 关键发现

**验证了 MuseFormer 的发现：**
- offset = 4, 8, 12, 16, 24, 32 的相似度明显高于其他位置
- 尤其是 **4 小节**和 **8 小节**周期最显著
- 这符合音乐的常见结构（4小节乐句、8小节段落）

---

## 二、跨乐器相关性分析

### 2.1 跨乐器不同小节

| Offset | PitchClass 相似度 |
|--------|------------------|
| 1 | 0.251 |
| 2 | 0.252 |
| 3 | 0.237 |
| 4 | 0.263 |
| 5 | 0.232 |
| 6 | 0.241 |
| 7 | 0.231 |
| 8 | 0.257 |
| 12 | 0.242 |
| 16 | 0.245 |
| 24 | 0.239 |
| 32 | 0.232 |

### 2.2 特点分析

- **曲线几乎是平的** - 无明显周期性
- 标准差约 0.13
- 均值差异 < 0.03
- **与同乐器形成鲜明对比**

### 2.3 跨乐器同小节

- 很多乐器在同一小节内相似度高达 **1.0**
- **结论：跨乐器同小节必须全连接（和声协调非常重要）**

---

## 三、同乐器 vs 跨乐器对比

| 类型 | 相似度 | 周期性 |
|------|--------|--------|
| 同乐器 | ~0.50 | 强（4, 8, 16...） |
| 跨乐器 | ~0.25 | 无 |

**差距约 2 倍！**

这说明：
1. 同乐器的历史信息非常重要（旋律连续性）
2. 跨乐器主要是同小节的和声协调

---

## 四、小节层面掩码策略 (已实现)

### 4.1 掩码策略

| 连接类型 | 策略 | 理由 |
|----------|------|------|
| **同乐器** | **全连接** | 即使 offset=29 相似度 (0.389) 仍高于跨乐器最高 (0.263) |
| **跨乐器** | offset ≤ 2 全连接 + offset = 4 | 近距离和声协调 + 乐段边界参考 |

### 4.2 配置参数

```python
# attention.py 中的实际配置
same_inst_full_connection = True        # 同乐器全连接！
cross_inst_full_range = 2               # 跨乐器 offset <= 2 全连接
CROSS_INST_FAR_OFFSETS = (-4,)          # 跨乐器远距离只看 4 小节前
```

### 4.3 为什么同乐器全连接？

**关键数据对比：**
- 同乐器最远 offset=29 相似度: **0.389**
- 跨乐器最高相似度 (offset=4): **0.263**

**结论：同乐器即使最远的小节，相关性仍然比跨乐器最近的小节还高！**

---

## 五、可学习注意力掩码的相关研究

### 5.1 SeerAttention (2024) ⭐ 最相关

**论文：** SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs

**核心思想：**
- 不依赖预定义的稀疏模式
- 用可学习的 gate 选择重要的 attention block
- 类似 MoE 的门控机制
- 128k 序列上达到 7.3x 加速，90% 稀疏度

### 5.2 其他相关工作

- **Trainable Dynamic Mask Sparse Attention (DMA)**: 动态稀疏掩码，根据内容特征决定
- **Learnable Attention Mask (LAM)**: 全局调节 attention map
- **SparseK Attention**: 可微分的 top-k 掩码操作

### 5.3 音乐领域的特殊性

| 方法 | 学习什么 | 适用场景 |
|------|----------|----------|
| SeerAttention | 块级稀疏模式 | 长序列 LLM |
| DMA | 内容感知的动态掩码 | 通用 |
| **我的需求** | **offset 级别的连接权重** | **音乐生成** |

**创新点：基于音乐结构（小节偏移、乐器关系）的稀疏模式**

---

## 六、创新方向：音乐结构感知的可学习掩码

```python
class MusicStructureAwareLearnableMask(nn.Module):
    """
    创新点：
    1. 不是 token 级别，而是 offset 级别
    2. 区分同乐器/跨乐器
    3. 可以用相似度统计初始化
    """
    def __init__(self, max_offset=32):
        # 同乐器：初始化为统计分析的相似度
        self.same_inst_prior = [0.487, 0.492, 0.444, 0.528, ...]
        self.same_inst_logits = nn.Parameter(torch.logit(self.same_inst_prior))

        # 跨乐器
        self.cross_inst_prior = [0.251, 0.252, 0.237, 0.263, ...]
        self.cross_inst_logits = nn.Parameter(torch.logit(self.cross_inst_prior))
```

---

## 七、今日关键收获

1. **同乐器周期性验证**：offset = 4, 8, 12, 16... 相似度显著高于其他
2. **跨乐器无周期性**：相似度稳定在 ~0.25，只需短期依赖
3. **掩码策略设计**：同乐器全连接，跨乐器 offset ≤ 2 + offset = 4
4. **可学习掩码创新方向**：offset 级别的可学习稀疏性

---

## 八、参考文献

- SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs
- Trainable Dynamic Mask Sparse Attention
- Multi-layer Learnable Attention Mask for Multimodal Tasks
- Sparser is Faster and Less is More
