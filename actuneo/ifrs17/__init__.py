"""
IFRS 17 Module

Measurement and presentation of insurance contracts under IFRS 17.

This module provides tools for:

- The premium allocation approach (PAA): liability for remaining coverage,
  insurance revenue, acquisition cash flows and onerous contracts
- The liability for incurred claims: discounting and risk adjustment, taken
  directly from the reserving models in :mod:`actuneo.loss_reserving`
- Reinsurance contracts held under the PAA
- The IFRS 17 reconciliations and the primary financial statements
- Earning premiums from a policy listing
- Grouping contracts by portfolio, cohort and profitability
- Disclosure tables: claims development, maturity analysis and sensitivities

Planned: the general measurement model (GMM) and the variable fee approach
(VFA), with the contractual service margin.
"""

from .premiums import earned_fraction, unearned_premium, earned_premium_by_period
from .lic import (
    LiabilityForIncurredClaims,
    lic_analysis_of_change,
    risk_adjustment_confidence_level,
    risk_adjustment_cost_of_capital,
    implied_confidence_level,
)
from .paa import PAAGroup, PAAReinsuranceHeld, paa_eligibility
from .statements import IFRS17Statements
from .aggregation import group_contracts
from .disclosures import claims_development_table, maturity_analysis, lic_sensitivity

__all__ = [
    'PAAGroup',
    'PAAReinsuranceHeld',
    'paa_eligibility',
    'LiabilityForIncurredClaims',
    'lic_analysis_of_change',
    'risk_adjustment_confidence_level',
    'risk_adjustment_cost_of_capital',
    'implied_confidence_level',
    'IFRS17Statements',
    'group_contracts',
    'claims_development_table',
    'maturity_analysis',
    'lic_sensitivity',
    'earned_fraction',
    'unearned_premium',
    'earned_premium_by_period',
]
