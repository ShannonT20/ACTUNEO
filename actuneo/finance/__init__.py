"""
Finance Module

Interest theory, yield curve construction, duration, and convexity measures.

This module provides tools for:
- Interest rate calculations
- Yield curve construction and analysis
- Duration and convexity measures
- Present value calculations
- Bond pricing and analysis
- Restatement helpers for hyperinflation (IAS 29)
"""

from .interest import InterestTheory
from .yield_curve import YieldCurve
from .duration_convexity import DurationConvexity
from .hyperinflation import (
    cumulative_inflation,
    exceeds_hyperinflation_indicator,
    restate,
    net_monetary_gain_or_loss,
)

__all__ = [
    'InterestTheory',
    'YieldCurve',
    'DurationConvexity',
    'cumulative_inflation',
    'exceeds_hyperinflation_indicator',
    'restate',
    'net_monetary_gain_or_loss',
]
