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
from .rates import (
    InterestRate,
    ForceOfInterest,
    simple_accumulation,
    simple_discount_value,
    real_rate,
    money_rate,
    inflation_from_index,
)
from .annuities import (
    annuity,
    accumulated_annuity,
    perpetuity,
    increasing_annuity,
    continuously_increasing_annuity,
    decreasing_annuity,
    accumulated_increasing_annuity,
    geometric_annuity,
    stepped_annuity,
)
from .cashflows import (
    Cashflows, present_value, accumulated_value, internal_rate_of_return, crossover_rate,
)
from . import instruments
from .loans import Loan, loan_schedule, annual_percentage_rate, flat_rate
from .securities import (
    Bond,
    bond_price_with_optional_redemption,
    equity_price,
    equity_yield,
    present_value_of_dividends,
    real_yield,
    index_linked_cashflows,
    property_value,
)
from .term_structure import (
    forward_rate,
    spot_rates_from_forwards,
    forwards_from_spot_rates,
    par_yield,
    spot_rates_from_bonds,
    discounted_mean_term,
    volatility,
    convexity,
    immunisation_check,
    immunising_amounts,
    continuous_spot_rate,
    continuous_forward_rate,
    continuous_forward_from_spots,
    instantaneous_forward_rate,
    effective_to_continuous,
    continuous_to_effective,
    estimated_value_change,
    effective_duration,
)
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
    'InterestRate',
    'ForceOfInterest',
    'simple_accumulation',
    'simple_discount_value',
    'real_rate',
    'money_rate',
    'inflation_from_index',
    'annuity',
    'accumulated_annuity',
    'perpetuity',
    'increasing_annuity',
    'continuously_increasing_annuity',
    'decreasing_annuity',
    'accumulated_increasing_annuity',
    'geometric_annuity',
    'stepped_annuity',
    'Cashflows',
    'present_value',
    'accumulated_value',
    'internal_rate_of_return',
    'crossover_rate',
    'instruments',
    'Loan',
    'loan_schedule',
    'annual_percentage_rate',
    'flat_rate',
    'Bond',
    'bond_price_with_optional_redemption',
    'equity_price',
    'equity_yield',
    'present_value_of_dividends',
    'real_yield',
    'index_linked_cashflows',
    'property_value',
    'forward_rate',
    'spot_rates_from_forwards',
    'forwards_from_spot_rates',
    'par_yield',
    'spot_rates_from_bonds',
    'discounted_mean_term',
    'volatility',
    'convexity',
    'immunisation_check',
    'immunising_amounts',
    'continuous_spot_rate',
    'continuous_forward_rate',
    'continuous_forward_from_spots',
    'instantaneous_forward_rate',
    'effective_to_continuous',
    'continuous_to_effective',
    'estimated_value_change',
    'effective_duration',
    'cumulative_inflation',
    'exceeds_hyperinflation_indicator',
    'restate',
    'net_monetary_gain_or_loss',
]
