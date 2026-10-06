"""
Mortality Module

Mortality tables, survival functions, graduation, and mortality improvement models.

This module provides tools for:
- Loading and manipulating mortality tables
- Survival function calculations
- Mortality graduation techniques
- Mortality improvement models
- African market-specific mortality assumptions
"""

from .mortality_table import MortalityTable
from .survival_functions import SurvivalFunctions
from .commutation import CommutationFunctions
from .select import SelectMortalityTable, load_am92
# from .graduation import Graduation  # TODO: Implement later
# from .improvement import MortalityImprovement  # TODO: Implement later

__all__ = [
    'MortalityTable',
    'SurvivalFunctions',
    'CommutationFunctions',
    'SelectMortalityTable',
    'load_am92',
]
