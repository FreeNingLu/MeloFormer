"""
MeloFormer v2.1 Inference Module

Usage:
    from inference.generator import MusicGenerator, load_model
    from inference.staged_generate import staged_generate
    from inference.tda import TDAGuidedGenerator, TopologyComputer, MusicFeatureExtractor
"""

from .generator import MusicGenerator, load_model
