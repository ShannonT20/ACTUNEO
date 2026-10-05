"""
Loss Reserving Module

Claims development triangles and chain-ladder reserving.

This module provides tools for:
- Building claims triangles from wide or long data
- Cumulative/incremental conversion, latest diagonal and link ratios
- Deterministic chain-ladder projection
- Mack's distribution-free standard error of the reserve

Planned: Bornhuetter-Ferguson, bootstrap and other stochastic models.
"""

from .triangle import Triangle
from .chain_ladder import ChainLadder, MackChainLadder
from .datasets import load_raa, load_genins

__all__ = [
    'Triangle',
    'ChainLadder',
    'MackChainLadder',
    'load_raa',
    'load_genins',
]
