# HID Tokenizer V2 Minimal (极简版)

## 概述

HID 极简版 tokenizer - 完全移除 Velocity 和 Duration tokens。

- **词汇表大小**: 331 tokens
- **Tokens/Note**: 1.91
- **相比完整版**: 序列长度减少 29%
- **相比 REMI**: 序列长度减少 56%

## 词汇表组成

| 类别 | Tokens | 数量 |
|------|--------|------|
| 特殊 | PAD, BOS, EOS, MASK, SEP, UNK | 6 |
| 小节 | BAR (相对编码) | 1 |
| 位置 | T0-T15 | 16 |
| 音高 | P0-P127 | 128 |
| ~~时值~~ | ~~L1-L64~~ | **0** |
| ~~力度~~ | ~~V0-V31~~ | **0** |
| 鼓轨 | #D | 1 |
| 音色 | #P0-#P127 | 128 |
| 速度 | BPM_0-BPM_31 | 32 |
| 拍号 | TS_* | 19 |
| **总计** | | **331** |

## 与完整版对比

| 指标 | 完整版 | 极简版 | 差异 |
|------|--------|--------|------|
| 词汇表 | 427 | 331 | -22.5% |
| Tokens/Note | 2.70 | 1.91 | -29.3% |
| V/L 精度 | 无损 | 推断 | 有损 |

## 解码时的 V/L 推断

根据乐器类型使用默认值：

| 乐器类型 | Program | Velocity | Duration |
|----------|---------|----------|----------|
| Piano | 0-7 | V20 | L4 |
| Bass | 32-39 | V24 | L8 |
| Strings | 40-47 | V18 | L16 |
| Brass | 56-63 | V24 | L4 |
| Synth Pad | 88-95 | V16 | L16 |
| Drums | #D | V24 | L1 |

## Trade-off

**优点**:
- 序列更短 → 训练更快
- 词汇表更小 → 模型更轻
- 更简单的学习任务

**缺点**:
- 丢失原始 Velocity (力度表现力)
- 丢失原始 Duration (节奏精度)
- 解码依赖固定的乐器默认值

## 适用场景

- 快速原型验证
- 对 V/L 精度要求不高的场景
- 音乐结构/旋律生成（后续人工调整 V/L）

## 文件说明

- `build_vocab_minimal.py` - Tokenizer 实现
- `midi_to_txt.py` - MIDI → TXT 转换
- `txt_to_midi.py` - TXT → MIDI 转换
- `vocab_minimal.pkl` - 预构建的词汇表

## 使用示例

```python
from build_vocab_minimal import MinimalTokenizer

tokenizer = MinimalTokenizer()

# 编码 (忽略 V/L)
tokens = tokenizer.encode_file("music.txt")

# 解码 (根据乐器推断 V/L)
txt_content = tokenizer.decode(tokens)
```
