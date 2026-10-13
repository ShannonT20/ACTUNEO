"""
ACTUNEO: African Actuarial Python Library

An open-source, community-driven actuarial Python library that empowers African and
Zimbabwean actuaries to perform core actuarial, financial, and statistical computations.

Modules:
- mortality: Mortality tables, survival functions, commutation functions
- life: Life assurance, annuities, reserves, and premium calculations
- finance: Interest theory, yield curve construction, duration, and convexity measures
- loss_reserving: Claims triangles, chain-ladder, inflation-adjusted chain-ladder, average cost
  per claim, Bornhuetter-Ferguson, Mack's standard error and bootstrap
- ifrs17: Premium allocation approach, liability for incurred claims, risk adjustment,
  reinsurance held and the IFRS 17 financial statements
- pensions: Defined benefit valuations with a choice of funding method, IAS 19 accounting,
  defined contribution projections, and calculations from Zimbabwe's Pensions and Provident
  Funds Act
- ai: A local language model chosen for the machine, or a provider with the user's own key
- zimbabwe: Zimbabwe 2023 mortality table catalogue and currency reference data
- simulation: Aggregate claims and a claims simulator for testing reserving methods
- macro_africa: World Bank economic indicators for African countries

Planned modules (not yet implemented):
- ifrs17 (further): General measurement model and variable fee approach, with the CSM
- macro_africa (further): data beyond the World Bank indicators now read
- simulation (further): stochastic models beyond aggregate claims and claim payments
- utils: Excel/CSV input-output functions, validation, and reporting

Author: Shannon Tafadzwa Sikadi
Version: 0.4.1
"""

__version__ = "0.4.1"
__author__ = "Shannon Tafadzwa Sikadi"
__description__ = "African Actuarial Python Library for insurance, pensions, and investment analytics"

# Import main modules for easy access
from . import mortality
from . import finance
from . import life
from . import loss_reserving
from . import ifrs17
from . import simulation
from . import macro_africa
from . import pensions
from . import ai
from . import zimbabwe
from . import templates
# Other modules will be imported as they are developed

__all__ = [
    'mortality',
    'finance',
    'life',
    'loss_reserving',
    'ifrs17',
    'simulation',
    'macro_africa',
    'pensions',
    'ai',
    'zimbabwe',
    'templates',
    # Add other modules as they are implemented
]
