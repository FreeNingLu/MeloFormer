"""
MeloFormer v2.0: Summary Token + FlexAttention 符号音乐生成模型

核心特点:
1. WSA 风格稀疏注意力 (bar-level block-sparse)
2. HID 编码 + 和弦 token
3. GQA + RMSNorm + SwiGLU

模块:
- data: 数据处理 (tokenizer, chord_detector)
- model: 模型结构 (FlexAttention, Summary Token)
- train: H800 训练脚本
"""

from .data import (
    HIDTokenizerV2,
    TokenInfo,
    Music21ChordDetector,
    get_chord_vocab,
)

from .model import (
    MeloFormer,
    create_model,
    FlexSummaryAttentionBlock,
    FlexSummaryAttentionMask,
)

# 向后兼容
HIDTokenizer = HIDTokenizerV2
ChordDetector = Music21ChordDetector

__version__ = '2.0.0'
__all__ = [
    'HIDTokenizerV2',
    'HIDTokenizer',
    'TokenInfo',
    'Music21ChordDetector',
    'ChordDetector',
    'get_chord_vocab',
    'MeloFormer',
    'create_model',
    'FlexSummaryAttentionBlock',
    'FlexSummaryAttentionMask',
]
