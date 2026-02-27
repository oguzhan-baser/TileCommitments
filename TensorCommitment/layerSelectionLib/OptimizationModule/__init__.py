"""
OptimizationModule: Multi-Verifier Budgeted Contiguous Selection (MV-BCS)

This module implements an optimal O(ML) dynamic programming algorithm to solve
the problem of selecting M pairwise-disjoint contiguous intervals from L layers,
where each verifier k has budget B_k and maximizes total benefit.
"""

from .layer_selection_optimizer import LayerSelectionOptimizer
from .scores_loader import load_scores_from_json

__all__ = ['LayerSelectionOptimizer', 'load_scores_from_json']

