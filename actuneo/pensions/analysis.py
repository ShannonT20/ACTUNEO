"""
Analysis of surplus

Explains how the surplus of a pension scheme moved between two valuations.
"""

from typing import Optional

import pandas as pd


def analysis_of_surplus(opening_assets: float,
                        opening_liabilities: float,
                        closing_assets: float,
                        closing_liabilities: float,
                        discount_rate: float,
                        normal_cost: float,
                        contributions: float = 0.0,
                        benefits_paid: float = 0.0,
                        expenses: float = 0.0,
                        closing_liabilities_on_opening_basis: Optional[float] = None,
                        years: float = 1.0) -> pd.Series:
    """
    Split the change in surplus between two valuations into its sources.

    Cash flows are taken to fall half way through the period and the normal
    cost to be measured at its start.

    Args:
        opening_assets: Assets at the first valuation
        opening_liabilities: Liabilities at the first valuation
        closing_assets: Assets at the second valuation
        closing_liabilities: Liabilities at the second valuation
        discount_rate: Annual discount rate of the first valuation
        normal_cost: Normal cost for the period, valued at its start
        contributions: Contributions paid in the period, by members and
            employer
        benefits_paid: Benefits paid in the period
        expenses: Expenses paid from the fund in the period
        closing_liabilities_on_opening_basis: Closing liabilities valued on
            the assumptions of the first valuation. When given, the effect
            of changing the assumptions is shown separately.
        years: Length of the period in years

    Returns:
        The opening surplus, the items of change and the closing surplus.
        The items add up exactly to the change in surplus.

        - *Interest on the surplus*: the opening surplus growing at the
          discount rate
        - *Investment return*: assets earned more (or less) than the
          discount rate
        - *Contributions*: contributions differed from the normal cost and
          expenses
        - *Liability experience*: salaries, deaths, withdrawals and
          retirements differed from the assumptions
        - *Change of assumptions*
    """
    if discount_rate <= -1:
        raise ValueError("discount_rate must be greater than -1")
    if years <= 0:
        raise ValueError("years must be positive")
    full = (1 + discount_rate) ** years
    half = (1 + discount_rate) ** (years / 2)
    opening_surplus = opening_assets - opening_liabilities
    closing_surplus = closing_assets - closing_liabilities
    old_basis = (closing_liabilities if closing_liabilities_on_opening_basis is None
                 else closing_liabilities_on_opening_basis)

    expected_assets = opening_assets * full + (contributions - benefits_paid - expenses) * half
    expected_liabilities = (opening_liabilities + normal_cost) * full - benefits_paid * half
    return pd.Series({
        "Surplus at start": opening_surplus,
        "Interest on the surplus": opening_surplus * (full - 1),
        "Investment return above the discount rate": closing_assets - expected_assets,
        "Contributions less normal cost and expenses":
            (contributions - expenses) * half - normal_cost * full,
        "Liability experience": expected_liabilities - old_basis,
        "Change of assumptions": old_basis - closing_liabilities,
        "Surplus at end": closing_surplus,
    })
