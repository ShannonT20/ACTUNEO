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
- Bornhuetter-Ferguson and Cape Cod methods
- Mack's distribution-free standard error of the reserve
- Bootstrap of the chain-ladder for the full reserve distribution
- Future cash flows, discounted reserves, plots and Excel output
- Triangles built directly from a listing of dated claim transactions
- Diagnostic tests of the chain-ladder assumptions and back-testing
- Over-dispersed Poisson GLM reserving, with calendar year effects
- A machine learning chain-ladder for experiments (needs scikit-learn)
"""

from .triangle import Triangle
from .chain_ladder import ChainLadder, MackChainLadder, estimate_tail_factor
from .inflation import InflationAdjustedChainLadder
from .average_cost import AverageCostPerClaim, grossing_up
from .bornhuetter_ferguson import BornhuetterFerguson
from .cape_cod import CapeCod
from .bootstrap import BootChainLadder
from .munich import MunichChainLadder
from .datasets import load_raa, load_genins, load_mw2008, load_mcl
from .report import compare_methods, export_reserving_report
from .diagnostics import (
    calendar_year_effect_test,
    development_factor_correlation_test,
    intercept_test,
    link_ratio_trend_test,
    link_ratio_outliers,
    backtest,
    diagnose,
)
from .glm import GLMReserving
from .ml import MLChainLadder

__all__ = [
    'Triangle',
    'ChainLadder',
    'InflationAdjustedChainLadder',
    'AverageCostPerClaim',
    'grossing_up',
    'BornhuetterFerguson',
    'CapeCod',
    'MackChainLadder',
    'BootChainLadder',
    'estimate_tail_factor',
    'compare_methods',
    'export_reserving_report',
    'MunichChainLadder',
    'GLMReserving',
    'MLChainLadder',
    'calendar_year_effect_test',
    'development_factor_correlation_test',
    'intercept_test',
    'link_ratio_trend_test',
    'link_ratio_outliers',
    'backtest',
    'diagnose',
    'load_raa',
    'load_mw2008',
    'load_mcl',
    'load_genins',
]
