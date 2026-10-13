"""
Pensions Module

Actuarial valuation of pension schemes.

This module provides:

- :class:`DBValuation`: valuation of a defined benefit scheme, with the
  liabilities for active members, deferred pensioners and pensioners, the
  funding level, the contribution rate, sensitivities and a choice of
  funding method
- :class:`DefinedBenefitScheme` and :class:`ValuationBasis`: the scheme's
  rules and the valuation assumptions
- :class:`ServiceTable` and :class:`SalaryScale`: decrements of active
  members and projected salaries
- :func:`pension_annuity`: annuity factors with increases, a guarantee and a
  spouse's pension
- :class:`IAS19Period`: defined benefit accounting under IAS 19
- :class:`DCProjection` and :class:`DCFundValuation`: defined contribution
  accounts and funds
- :func:`analysis_of_surplus`: why the surplus changed between valuations
- :mod:`actuneo.pensions.zimbabwe`: calculations from Zimbabwe's Pensions
  and Provident Funds Act [Chapter 24:32]

Read the Status and Limitations page of the documentation before relying on
any result. A valuation depends on judgement about the assumptions that this
module does not supply.
"""

from . import zimbabwe
from .analysis import analysis_of_surplus
from .annuity import pension_annuity, survival_probabilities
from .dc import DCFundValuation, DCProjection
from .ias19 import IAS19Period
from .scheme import DefinedBenefitScheme, ValuationBasis
from .service_table import SalaryScale, ServiceTable
from .valuation import METHODS, DBValuation

__all__ = [
    'DBValuation',
    'DefinedBenefitScheme',
    'ValuationBasis',
    'ServiceTable',
    'SalaryScale',
    'pension_annuity',
    'survival_probabilities',
    'IAS19Period',
    'DCProjection',
    'DCFundValuation',
    'analysis_of_surplus',
    'METHODS',
    'zimbabwe',
]
