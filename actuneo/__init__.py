"""
ACTUNEO: African Actuarial Python Library

An open-source, community-driven actuarial Python library that empowers African and
Zimbabwean actuaries to perform core actuarial, financial, and statistical computations.

Modules:
- mortality: Mortality tables, survival functions, graduation, and mortality improvement models
- life: Life assurance, annuities, reserves, and premium calculations
- pensions: Contribution schedules, benefit projections, and actuarial valuations for pension schemes
- ifrs17: Measurement models (GMM, VFA, PAA), CSM, risk adjustment, discounting
- loss_reserving: Chain-ladder, Bornhuetter-Ferguson, and stochastic reserving models
- finance: Interest theory, yield curve construction, duration, and convexity measures
- macro_africa: Country-specific economic data connectors (inflation, GDP, currency exchange)
- simulation: Monte Carlo simulations for stochastic actuarial models
- utils: Excel/CSV input-output functions, validation, and reporting

Author: Shannon Tafadzwa Sikadi
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Shannon Tafadzwa Sikadi"
__description__ = "African Actuarial Python Library for insurance, pensions, and investment analytics"

# Import main modules for easy access
from . import mortality
from . import finance
from . import life
# Other modules will be imported as they are developed

__all__ = [
    'mortality',
    'finance',
    'life',
    # Add other modules as they are implemented
]
