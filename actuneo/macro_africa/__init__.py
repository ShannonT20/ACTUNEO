"""
Macro Africa Module

Economic indicators for African countries from the World Bank's open data.

The data is downloaded when a function is called, so an internet connection
is needed; nothing is stored in the package. World Bank data is licensed
under Creative Commons Attribution 4.0 (CC BY 4.0): when you use it, credit
"The World Bank: World Development Indicators".

Annual figures for recent years are often missing, and for countries with
very high inflation or several exchange rates the official series may differ
from the rates actually experienced. Check the figures before using them in
a valuation.
"""

from .world_bank import (
    INDICATORS,
    COUNTRIES,
    world_bank_indicator,
    macro_data,
)

__all__ = [
    'INDICATORS',
    'COUNTRIES',
    'world_bank_indicator',
    'macro_data',
]
