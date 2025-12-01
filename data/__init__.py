# HID-MuseFormer Data Module

from .tokenizer_v2 import HIDTokenizerV2, TokenInfo
from .chord_detector_music21 import Music21ChordDetector, get_chord_vocab

# 向后兼容的别名
HIDTokenizer = HIDTokenizerV2
ChordDetector = Music21ChordDetector
