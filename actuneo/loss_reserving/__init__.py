"""
Loss Reserving Module

Claims development (run-off) triangles and reserving methods for general
insurance.

This module provides tools for:

- Building claims triangles from wide or long data
- Cumulative/incremental conversion, latest diagonal and link ratios
- Basic chain-ladder projection, with model checks and tail factors
- Inflation-adjusted chain-ladder
- Average cost per claim method and grossing-up factors
- Bornhuetter-Ferguson method
- Mack's distribution-free standard error of the reserve
- Bootstrap of the chain-ladder for the full reserve distribution
"""

from .triangle import Triangle
from .chain_ladder import ChainLadder, MackChainLadder, estimate_tail_factor
from .inflation import InflationAdjustedChainLadder
from .average_cost import AverageCostPerClaim, grossing_up
from .bornhuetter_ferguson import BornhuetterFerguson
from .datasets import load_raa, load_genins

__all__ = [
    'Triangle',
    'ChainLadder',
    'InflationAdjustedChainLadder',
    'AverageCostPerClaim',
    'grossing_up',
    'BornhuetterFerguson',
    'MackChainLadder',
    'estimate_tail_factor',
    'load_raa',
    'load_genins',
]
