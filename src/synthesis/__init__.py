"""
Synthesis Module
Q/A generation and feature extraction for vision-language data
"""

from .qa_generator import QAGenerator
from .feature_extractor import FeatureExtractor

__all__ = ['QAGenerator', 'FeatureExtractor']
