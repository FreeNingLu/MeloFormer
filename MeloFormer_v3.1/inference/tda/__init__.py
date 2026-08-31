"""
MeloFormer v2.1 TDA (Topological Data Analysis) Module

Provides topology-guided music generation using persistent homology.

Usage:
    from inference.tda import TopologyComputer, MusicFeatureExtractor, TDAGuidedGenerator
"""

from .topology_computer import TopologyComputer, PersistenceDiagram
from .feature_extractor import MusicFeatureExtractor, BarFeature
from .tda_guided_generate import TDAGuidedGenerator, TDAConfig
