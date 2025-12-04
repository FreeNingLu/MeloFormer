"""
MeloFormer: 基于 HID 编码的符号音乐生成模型

主要特点:
1. HID (Hierarchical Instrument-aware Duration-free) 编码
2. 和弦 token 替代 BAR token，携带和声信息
3. FC-Attention 适配乐器分组结构

模块:
- data: 数据处理 (tokenizer, chord_detector, dataset)
- model: 模型结构 (attention, meloformer)
- train: 训练脚本
- generate: 生成脚本
"""

from .data import (
    HIDTokenizer,
    HIDTokenizerV2,
    ChordDetector,
    Music21ChordDetector,
    HIDMusicDataset,
    collate_fn,
    create_dataloaders,
)

from .model import (
    MeloFormer,
    create_model,
    FCAttentionBlock,
    FCAttentionMask,
)

__version__ = '1.0.2'
__all__ = [
    'HIDTokenizer',
    'HIDTokenizerV2',
    'ChordDetector',
    'Music21ChordDetector',
    'HIDMusicDataset',
    'collate_fn',
    'create_dataloaders',
    'MeloFormer',
    'create_model',
    'FCAttentionBlock',
    'FCAttentionMask',
]
