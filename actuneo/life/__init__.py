"""
Life Module

Life assurance, annuities, reserves, and premium calculations.

This module provides tools for:
- Life assurance calculations
- Annuity pricing and valuation
- Reserve calculations
- Premium determination
- Policy value calculations
"""

from .life_assurance import LifeAssurance
from .annuities import Annuities
from .reserves import Reserves

__all__ = [
    'LifeAssurance',
    'Annuities',
    'Reserves',
]
