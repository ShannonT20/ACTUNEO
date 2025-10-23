"""
Finance Module

Interest theory, yield curve construction, duration, and convexity measures.

This module provides tools for:
- Interest rate calculations
- Yield curve construction and analysis
- Duration and convexity measures
- Present value calculations
- Bond pricing and analysis
"""

from .interest import InterestTheory
from .yield_curve import YieldCurve
from .duration_convexity import DurationConvexity

__all__ = [
    'InterestTheory',
    'YieldCurve',
    'DurationConvexity',
]
