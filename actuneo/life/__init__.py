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
from .policy import LifePolicy, Expenses, simple_bonus_benefits, compound_bonus_benefits
from .decrements import (
    MultipleDecrementTable,
    MultiStateModel,
    dependent_from_independent,
    independent_from_dependent,
)
from .profit_testing import (
    ProfitTest,
    ProfitSignature,
    UnitLinkedPolicy,
    zeroise_negative_cashflows,
)

__all__ = [
    'LifeAssurance',
    'Annuities',
    'Reserves',
    'LifePolicy',
    'Expenses',
    'simple_bonus_benefits',
    'compound_bonus_benefits',
    'MultipleDecrementTable',
    'MultiStateModel',
    'dependent_from_independent',
    'independent_from_dependent',
    'ProfitTest',
    'ProfitSignature',
    'UnitLinkedPolicy',
    'zeroise_negative_cashflows',
]
