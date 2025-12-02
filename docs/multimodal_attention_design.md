# 多模态音乐生成模型:文本+MIDI Decoder-only架构

## 一、核心设计理念

### 关键洞察
1. **Summary Token是天然的跨模态桥梁**
   - 文本无需attend到所有Regular Token(会有几万个)
   - 文本只需attend到Summary Token(每个bar一个)
   - Summary Token已经压缩了该bar的音乐信息

2. **复用现有的FlexSummary机制**
   - 音乐部分的注意力机制完全不变
   - 只需扩展mask逻辑,允许文本参与交互

3. **最小化计算开销**
   - 文本长度: ~50-200 tokens
   - Summary数量: ~100-200 tokens (对应100-200个bar)
   - Regular数量: ~20K tokens
   - 文本避免了对20K Regular Token的注意力计算!

---

## 二、序列组织结构

### 完整序列格式

```
位置索引:  0    1    2    3   ...  n   n+1  n+2      n+3  n+4  n+5      ...
Token类型: [文本Token₁ ... 文本Tokenₙ] [SEP] [Sum₀] [Reg₀,₁...Reg₀,ₘ] [Sum₁] [Reg₁,₁...] ...

标记:
├─────── Text Region ─────────┤      ├──────────── Music Region ─────────────────→
                                      ├─ Bar 0 ─┤ ├─ Bar 1 ─┤
```

### Token属性表

| Token | 位置 | note_id | inst_id | chord_id | type_id |
|-------|------|---------|---------|----------|---------|
| 文本Token₁ | 0 | -1 | 255 | -1 | -1 |
| 文本Token₂ | 1 | -1 | 255 | -1 | -1 |
| ... | ... | -1 | 255 | -1 | -1 |
| SEP | n | -1 | 255 | -1 | -1 |
| Sum₀ | n+1 | -1 | 255 | 0 | -1 |
| T0 (Bar0) | n+2 | 0 | 0 | 0 | 0 |
| P60 | n+3 | 0 | 0 | 0 | 1 |
| L8 | n+4 | 0 | 0 | 0 | 2 |
| V24 | n+5 | 0 | 0 | 0 | 3 |
| Sum₁ | ... | -1 | 255 | 1 | -1 |
| ... | ... | ... | ... | ... | ... |

**约定**:
- `inst_id = 255`: 标识文本区域token
- `chord_id = -1`: 文本token无和弦概念
- `type_id = -1`: Summary Token和文本token无类型

---

## 三、注意力机制详解

### 3.1 整体注意力矩阵结构

```
注意力矩阵 (Query 行 × Key 列):

          │  Text区  │SEP│ Sum₀│ Reg₀ │ Sum₁│ Reg₁ │ Sum₂│ Reg₂ │
══════════╪══════════╪═══╪═════╪══════╪═════╪══════╪═════╪══════╪
  Text    │ 双向全连  │ ✓ │  ✓  │  ✗   │  ✓  │  ✗   │  ✓  │  ✗   │
  Query   │ (标准attn)│   │(桥梁)│      │(桥梁)│      │(桥梁)│      │
══════════╪══════════╪═══╪═════╪══════╪═════╪══════╪═════╪══════╪
  SEP     │    ✓     │ ✓ │  ✓  │  ✗   │  ✗  │  ✗   │  ✗  │  ✗   │
══════════╪══════════╪═══╪═════╪══════╪═════╪══════╪═════╪══════╪
  Sum₀    │    ✓     │ ✓ │ SS  │  SR  │  ✗  │  ✗   │  ✗  │  ✗   │
  Query   │ (全文本) │   │因果 │同bar │     │      │     │      │
──────────┼──────────┼───┼─────┼──────┼─────┼──────┼─────┼──────┤
  Reg₀    │    ✓     │ ✓ │ RS  │  RR  │  ✗  │  ✗   │  ✗  │  ✗   │
  Query   │ (全文本) │   │前bar│稀疏  │     │      │     │      │
══════════╪══════════╪═══╪═════╪══════╪═════╪══════╪═════╪══════╪
  Sum₁    │    ✓     │ ✓ │ SS  │  ✗   │ SS  │  SR  │  ✗  │  ✗   │
  Query   │          │   │因果 │      │自己 │同bar │     │      │
──────────┼──────────┼───┼─────┼──────┼─────┼──────┼─────┼──────┤
  Reg₁    │    ✓     │ ✓ │ RS  │  ✗   │ RS  │  RR  │  ✗  │  ✗   │
  Query   │          │   │前bar│      │当前 │稀疏  │     │      │
══════════╪══════════╪═══╪═════╪══════╪═════╪══════╪═════╪══════╪

图例:
✓  = 全连接/可见
✗  = 不可见
SS = Summary-Summary (因果)
SR = Summary-Regular (同bar)
RS = Regular-Summary (看前面bar的Sum)
RR = Regular-Regular (FlexAttention稀疏)
```

### 3.2 注意力规则详细说明

#### 规则1: 文本Query
```python
if q_is_text:
    # 1.1 看所有文本(双向)
    if k_is_text:
        return True

    # 1.2 看所有Summary Token(关键!)
    if k_is_summary:
        return True  # 不管因果,文本作为"条件"可以看所有Summary

    # 1.3 不看Regular Token
    if k_is_regular:
        return False
```

**设计理由**:
- 文本是"条件输入",需要双向理解语义
- Summary Token压缩了bar级音乐信息,是高效的跨模态接口
- 避免文本attend到海量Regular Token(计算浪费)

#### 规则2: Summary Token Query (SS + SR)
```python
if q_is_summary:
    # 2.1 看所有文本(文本是条件)
    if k_is_text:
        return True

    # 2.2 看前面的Summary(SS, 因果)
    if k_is_summary:
        return k_chord <= q_chord  # 因果mask

    # 2.3 看同bar的Regular(SR)
    if k_is_regular:
        return k_chord == q_chord

    return False
```

**设计理由**:
- Summary需要聚合文本条件信息
- SS保持因果性(自回归生成)
- SR聚合同bar的Regular信息

#### 规则3: Regular Token Query (RS + RR)
```python
if q_is_regular:
    # 3.1 看所有文本(文本是条件)
    if k_is_text:
        return True

    # 3.2 看已完成bar的Summary(RS)
    if k_is_summary:
        return k_chord < q_chord  # 不看当前bar的Summary(还在构建中)

    # 3.3 Regular-Regular: 你的FlexAttention逻辑!
    if k_is_regular:
        # 因果性
        causal = q_pos >= k_pos

        # 同乐器全连接
        same_inst = (q_inst == k_inst) & (q_inst < 129)

        # 跨乐器bar级稀疏
        chord_diff = q_chord - k_chord
        cross_near = (q_inst != k_inst) & (chord_diff >= 0) & (chord_diff <= 2)
        cross_far = (q_inst != k_inst) & (chord_diff == 4)
        bar_mask = same_inst | cross_near | cross_far

        # Token Type稀疏
        same_note = (q_note == k_note) & (q_note >= 0)
        type_visible = TOKEN_TYPE_VISIBILITY[q_type, k_type]
        token_mask = same_note | type_visible | (k_type == -1)

        return causal & bar_mask & token_mask

    return False
```

**设计理由**:
- Regular也需要文本条件信息
- RS获取远距离bar的压缩信息
- RR保持你的原有FlexAttention逻辑

---

## 四、代码实现

### 4.1 统一的mask_mod函数

```python
def create_multimodal_mask_mod(text_len, music_bar_positions, TOKEN_TYPE_VISIBILITY):
    """
    创建多模态mask函数

    Args:
        text_len: 文本token数量
        music_bar_positions: shape [total_music_len], 每个token对应的bar索引
        TOKEN_TYPE_VISIBILITY: [4, 4] bool矩阵

    Returns:
        mask_mod函数,用于FlexAttention
    """
    music_start = text_len + 1  # +1是SEP

    def mask_mod(b, h, q_idx, kv_idx):
        # ===== 获取token属性 =====
        q_is_text = q_idx < text_len
        k_is_text = kv_idx < text_len

        q_is_sep = q_idx == text_len
        k_is_sep = kv_idx == text_len

        q_in_music = q_idx >= music_start
        k_in_music = kv_idx >= music_start

        # 从buffer中读取音乐token属性
        # (需要预先传入: inst_id, chord_id, note_id, type_id等)
        if q_in_music:
            q_music_idx = q_idx - music_start
            q_inst = inst_buffer[b, q_music_idx]
            q_chord = chord_buffer[b, q_music_idx]
            q_note = note_buffer[b, q_music_idx]
            q_type = type_buffer[b, q_music_idx]
            q_is_summary = (q_type == -1) and (q_chord >= 0)

        if k_in_music:
            k_music_idx = kv_idx - music_start
            k_inst = inst_buffer[b, k_music_idx]
            k_chord = chord_buffer[b, k_music_idx]
            k_note = note_buffer[b, k_music_idx]
            k_type = type_buffer[b, k_music_idx]
            k_is_summary = (k_type == -1) and (k_chord >= 0)

        # ===== 规则1: 文本Query =====
        if q_is_text:
            if k_is_text or k_is_sep:
                return True  # 文本双向
            if k_in_music and k_is_summary:
                return True  # 文本看所有Summary
            return False

        # ===== 规则2: SEP Query =====
        if q_is_sep:
            return kv_idx <= text_len  # SEP只看文本+自己

        # ===== 规则3: 音乐区域Query =====
        if q_in_music:
            # 3.1 所有音乐token都能看文本(条件)
            if k_is_text or k_is_sep:
                return True

            # 3.2 Summary Query (SS + SR)
            if q_is_summary:
                if k_is_summary:
                    # SS: 因果
                    return k_chord <= q_chord
                else:
                    # SR: 同bar
                    return k_chord == q_chord

            # 3.3 Regular Query (RS + RR)
            else:
                # RS: 看已完成bar的Summary
                if k_is_summary:
                    return k_chord < q_chord

                # RR: FlexAttention逻辑
                else:
                    # 因果性
                    causal = q_idx >= kv_idx
                    if not causal:
                        return False

                    # 同乐器全连接
                    same_inst = (q_inst == k_inst) and (q_inst < 129)
                    if same_inst:
                        return True

                    # 跨乐器bar级稀疏
                    chord_diff = q_chord - k_chord
                    if chord_diff < 0:
                        return False

                    cross_near = (chord_diff <= 2)
                    cross_far = (chord_diff == 4)
                    bar_visible = cross_near or cross_far

                    if not bar_visible:
                        return False

                    # Token Type稀疏
                    same_note = (q_note == k_note) and (q_note >= 0)
                    if same_note:
                        return True

                    is_global = (k_type == -1)
                    if is_global:
                        return True

                    type_visible = TOKEN_TYPE_VISIBILITY[q_type, k_type]
                    return type_visible

        return False

    return mask_mod
```

### 4.2 模型前向传播

```python
class MultiModalMusicDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 嵌入层
        self.text_embed = nn.Embedding(config.text_vocab_size, config.d_model)
        self.music_embed = nn.Embedding(config.music_vocab_size, config.d_model)
        self.summary_embed = nn.Embedding(config.max_bars, config.d_model)

        # Token type embedding(区分文本/音乐)
        self.token_type_embed = nn.Embedding(2, config.d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            FlexSummaryAttentionLayer(config) for _ in range(config.n_layers)
        ])

        # 输出头
        self.music_lm_head = nn.Linear(config.d_model, config.music_vocab_size)

    def forward(self, text_ids, music_ids, music_metadata):
        """
        Args:
            text_ids: [batch, text_len]
            music_ids: [batch, music_len]
            music_metadata: dict with keys:
                - inst_ids: [batch, music_len]
                - chord_ids: [batch, music_len]
                - note_ids: [batch, music_len]
                - type_ids: [batch, music_len]
                - summary_positions: [batch, n_bars] (Summary Token插入位置)
        """
        batch_size = text_ids.size(0)
        text_len = text_ids.size(1)
        music_len = music_ids.size(1)

        # === 嵌入 ===
        text_x = self.text_embed(text_ids)  # [B, text_len, D]
        music_x = self.music_embed(music_ids)  # [B, music_len, D]

        # 在Summary位置插入可学习token
        summary_positions = music_metadata['summary_positions']
        n_bars = summary_positions.size(1)
        summary_x = self.summary_embed(torch.arange(n_bars, device=text_ids.device))
        summary_x = summary_x.unsqueeze(0).expand(batch_size, -1, -1)  # [B, n_bars, D]

        # 合并序列: [Text] [SEP] [Music with Summary]
        sep_x = self.music_embed(torch.tensor([SEP_TOKEN_ID], device=text_ids.device))
        sep_x = sep_x.unsqueeze(0).expand(batch_size, 1, -1)

        # 将Summary Token插入到music_x中对应位置
        music_with_summary = self._insert_summary_tokens(music_x, summary_x, summary_positions)

        # 拼接完整序列
        x = torch.cat([text_x, sep_x, music_with_summary], dim=1)  # [B, total_len, D]

        # Token type embedding
        token_types = torch.cat([
            torch.zeros(batch_size, text_len + 1, dtype=torch.long, device=text_ids.device),  # 文本+SEP
            torch.ones(batch_size, music_with_summary.size(1), dtype=torch.long, device=text_ids.device)  # 音乐
        ], dim=1)
        x = x + self.token_type_embed(token_types)

        # === 创建mask ===
        mask_mod = create_multimodal_mask_mod(
            text_len=text_len,
            music_bar_positions=music_metadata['chord_ids'],
            TOKEN_TYPE_VISIBILITY=self.TOKEN_TYPE_VISIBILITY
        )

        block_mask = create_block_mask(
            mask_mod,
            B=batch_size,
            H=None,
            Q_LEN=x.size(1),
            KV_LEN=x.size(1),
            device=x.device,
            _compile=True
        )

        # === Transformer layers ===
        for layer in self.layers:
            x = layer(x, block_mask=block_mask)

        # === 输出 ===
        logits = self.music_lm_head(x)  # [B, total_len, music_vocab]

        return logits

    def _insert_summary_tokens(self, music_x, summary_x, summary_positions):
        """
        在指定位置插入Summary Token
        """
        # 实现略(需要根据summary_positions动态插入)
        # 可以预先在数据处理时就插入占位符,这里直接替换
        pass
```

### 4.3 训练Loss

```python
def compute_loss(logits, labels, text_len):
    """
    只对音乐部分计算loss

    Args:
        logits: [B, total_len, vocab]
        labels: [B, total_len] (文本部分为-100)
        text_len: 文本长度
    """
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # 只计算音乐部分的loss
    # (文本部分labels设为-100会被ignore)
    loss = F.cross_entropy(
        shift_logits.view(-1, logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100
    )

    return loss
```

---

## 五、训练策略

### 5.1 两阶段训练

#### 阶段1: 纯MIDI预训练 (已完成)
```python
# 你现有的训练
input_ids = [BOS, BPM_120, TS_4/4, Cmaj, #P0, T0, P60, ...]
labels = input_ids[1:]  # Next token prediction
```

#### 阶段2: 文本条件微调
```python
# 新增文本条件
text_ids = tokenize("明快的、充满希望的旋律")
music_ids = [BOS, BPM_120, TS_4/4, Cmaj, #P0, T0, P60, ...]

input_ids = torch.cat([text_ids, [SEP], music_ids])
labels = torch.cat([
    torch.full_like(text_ids, -100),  # 文本不计算loss
    [-100],  # SEP不计算loss
    music_ids[1:]  # 只对音乐计算loss
])
```

### 5.2 渐进式微调

```python
# Step 1: 冻结音乐编码器
for param in model.music_embed.parameters():
    param.requires_grad = False
for layer in model.layers:
    # 只解冻与文本相关的参数
    pass

# Step 2: 小学习率全局微调
optimizer = AdamW(model.parameters(), lr=1e-5)

# Step 3: 完全解冻
optimizer = AdamW(model.parameters(), lr=5e-5)
```

### 5.3 数据配比

```python
# 混合训练数据
# 80% 纯MIDI(保持音乐生成能力)
# 20% 文本+MIDI(学习条件生成)

dataloader = MixedDataLoader([
    (midi_only_dataset, 0.8),
    (text_music_dataset, 0.2)
])
```

---

## 六、推理与生成

### 6.1 条件生成

```python
# 用户输入文本描述
text = "一段明快的、充满希望的旋律,适合清晨"
text_ids = tokenizer.encode(text)

# 自回归生成
generated = model.generate(
    text_ids=text_ids,
    max_length=4096,
    temperature=0.9,
    top_p=0.95
)

# 解码为MIDI
midi = decode_to_midi(generated)
```

### 6.2 无条件生成(兼容模式)

```python
# 不提供文本,直接生成
generated = model.generate(
    text_ids=None,  # 空文本
    max_length=4096
)
```

### 6.3 部分条件生成

```python
# 只提供情绪标签
text = "情绪:欢快 速度:快 调性:大调"
text_ids = tokenizer.encode(text)

generated = model.generate(text_ids=text_ids, max_length=4096)
```

---

## 七、计算复杂度分析

### 7.1 注意力计算量对比

假设:
- 文本长度: T = 100
- Summary数量: S = 100 (100个bar)
- Regular数量: R = 20,000

#### 方案A: 文本attend所有音乐token(Naive)
```
文本侧计算: T × (T + S + R) = 100 × 20,200 = 2,020,000
```

#### 方案B: 文本只attend Summary Token(本方案)
```
文本侧计算: T × (T + S) = 100 × 200 = 20,000
节省: 99%!
```

### 7.2 总体复杂度

| 组件 | 复杂度 | 说明 |
|------|--------|------|
| 文本自注意力 | O(T²) | T很小,可忽略 |
| 文本→Summary | O(T × S) | 跨模态桥梁 |
| Summary机制 | O(S² + S×R) | 你的现有机制 |
| Regular稀疏 | O(R × W) | W是稀疏窗口 |
| **总计** | **O(T×S + S² + R×W)** | 线性增长! |

---

## 八、优势总结

### 8.1 架构优势
✅ **统一Decoder-only**: 不需要Encoder-Decoder
✅ **最小改动**: 复用FlexSummaryAttention
✅ **自然桥梁**: Summary Token天然适合跨模态
✅ **高效计算**: 文本避免attend海量Regular Token

### 8.2 功能优势
✅ **灵活条件**: 可以加或不加文本条件
✅ **渐进训练**: 先MIDI预训练,再文本微调
✅ **多样控制**: 情绪、风格、结构等多维度控制
✅ **向后兼容**: 无文本时退化为纯MIDI模型

### 8.3 工程优势
✅ **代码简洁**: mask_mod函数统一处理
✅ **调试友好**: 可以单独测试文本/音乐部分
✅ **扩展性强**: 未来可以加入更多模态(图像、音频等)

---

## 九、潜在问题与解决方案

### 问题1: Summary Token数量动态变化
**现象**: 不同样本的bar数量不同,Summary数量也不同

**解决**:
```python
# 方案A: Padding到最大bar数
max_bars = 200
summary_x = self.summary_embed(torch.arange(max_bars))
# 用mask标记有效的Summary

# 方案B: 动态插入(推荐)
# 在数据预处理时就插入Summary Token占位符
```

### 问题2: 文本长度不一致
**现象**: 不同样本的文本长度不同

**解决**:
```python
# Batch内padding + attention_mask
text_padded = pad_sequence(text_list, padding_value=PAD_TOKEN_ID)
```

### 问题3: 文本看"所有Summary"是否违反因果性?
**现象**: 文本作为条件可以看未来的Summary Token

**解决**:
```python
# 方案A: 训练时文本看所有Summary(条件建模)
if q_is_text and k_is_summary:
    return True  # 训练时双向

# 方案B: 推理时分阶段
# Stage 1: 先用文本生成第一个bar
# Stage 2: 自回归生成后续bar,文本mask改为因果
```

**推荐**: 方案A,因为文本是"已知条件",不参与生成

### 问题4: 模态不平衡
**现象**: 音乐token远多于文本,梯度可能被音乐主导

**解决**:
```python
# 分别计算loss
loss_music = compute_loss(logits, music_labels)
loss_text_alignment = contrastive_loss(text_repr, music_repr)

# 加权
total_loss = loss_music + λ * loss_text_alignment
```

---

## 十、实验验证计划

### 消融实验
1. **Baseline**: 纯MIDI模型(无文本)
2. **+Text(Naive)**: 文本attend所有音乐token
3. **+Text(Summary)**: 文本只attend Summary Token(本方案)
4. **+Text(Summary+Alignment)**: 额外加入对齐损失

### 评估指标
- **音乐质量**: PCH, GC, SS, IC(你的现有指标)
- **条件遵循度**: 生成的MIDI是否符合文本描述
- **计算效率**: 训练速度、显存占用
- **生成多样性**: 同一文本多次生成的差异度

---

## 十一、未来扩展

### 多模态扩展
```
文本 ──┐
       ├──→ Summary Token ──→ 音乐生成
图像 ──┘
```

### 双向生成
```
文本 ──→ 音乐
音乐 ──→ 文本(故事生成)
```

### 交互式编辑
```
用户: "把第二小节改得更激昂"
模型: 只重新生成第二小节,保持其他部分不变
```

---

*设计文档版本: v1.0*
*日期: 2025-12-02*
*项目: HID-MuseFormer 多模态扩展*
