"""
Layer Selection Integration Pipeline

This package provides integration tools for extracting layer significance scores
from AlphaPruning and formatting them for CVXPY optimization.
"""

__version__ = '1.0.0'

from .extract_layer_scores import extract_layer_scores

__all__ = ['extract_layer_scores']

