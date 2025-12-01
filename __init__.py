"""
HID-MuseFormer: 基于 HID 编码的 MuseFormer 变体

主要特点:
1. HID (Hierarchical Instrument-aware Duration-free) 编码
2. 和弦 token 替代 BAR token，携带和声信息
3. FC-Attention 适配乐器分组结构

模块:
- data: 数据处理 (tokenizer, chord_detector, dataset)
- model: 模型结构 (attention, hid_museformer)
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
    HIDMuseFormer,
    create_model,
    FCAttentionBlock,
    FCAttentionMask,
)

__version__ = '0.8.0'
__all__ = [
    'HIDTokenizer',
    'HIDTokenizerV2',
    'ChordDetector',
    'Music21ChordDetector',
    'HIDMusicDataset',
    'collate_fn',
    'create_dataloaders',
    'HIDMuseFormer',
    'create_model',
    'FCAttentionBlock',
    'FCAttentionMask',
]
