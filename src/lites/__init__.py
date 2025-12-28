"""
LITES: Hierarchical Time Series Event Labeling System

A comprehensive system for detecting, classifying, and organizing time series events
into hierarchical structures suitable for language model training.
"""

__version__ = "0.1.0"

from .detection.detector import EventDetector
from .classification.classifier import EventClassifier
from .hierarchy.builder import HierarchyBuilder
from .formatting.llm_formatter import LLMFormatter

__all__ = [
    "EventDetector",
    "EventClassifier",
    "HierarchyBuilder",
    "LLMFormatter",
]
