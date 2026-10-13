"""
Zimbabwe: Pensions and Provident Funds Act [Chapter 24:32]

Calculations set out in the Pensions and Provident Funds Act (No. 3 of 2022)
and reference figures from the Insurance and Pensions Commission (IPEC).

What is here was taken from the text of the Act. The minimum funding level
comes from IPEC's own publications, not from a statutory instrument. The
*minimum individual reserve* and the *minimum pension increase* are amounts
"determined in accordance with the formula approved by the Commission"
(section 2): those formulas are not in the Act and are not implemented, so
the reserves must be supplied.

This module does the arithmetic. It does not give legal advice: read the Act
and the regulations made under it, and check for amendments.
"""

from typing import Sequence, Union

import numpy as np
import pandas as pd

#: Minimum funding level of a defined benefit fund, as stated in IPEC's
#: trustee training material ("IPEC regulations require that funds should be
#: at least 75% funded") and referred to in IPEC's quarterly pensions reports
#: as the regulatory minimum. The instrument that sets it was not located.
MINIMUM_FUNDING_LEVEL = 0.75

#: Longest interval in years between pension increases of at least the
#: minimum pension increase, after the first (section 18(1)(d) of the Act).
PENSION_INCREASE_INTERVAL_YEARS = {"defined_benefit": 3, "defined_contribution": 1}

#: Purposes for which the board may use an actuarial surplus, with the
#: Commission's approval (section 20(2) of the Act).
SURPLUS_USES = (
    "Improve benefits for existing members and, additionally or alternatively, pensioners",
    "Supplement benefits previously paid to former members, or transfer values previously "
    "transferred in respect of former members",
    "Reduce current contributions due from participating employers of a defined benefit fund",
    "Transfer, wholly or partly, to a contingency reserve account",
)


def funding_level_check(funding_level: float) -> dict:
    """
    Compare a funding level with full funding and with IPEC's minimum.

    Args:
        funding_level: Assets as a proportion of liabilities

    Returns:
        Whether the fund is fully funded, whether it meets the minimum of
        :data:`MINIMUM_FUNDING_LEVEL`, and a description
    """
    if funding_level < 0:
        raise ValueError("funding_level must be non-negative")
    fully_funded = funding_level >= 1.0
    above_minimum = funding_level >= MINIMUM_FUNDING_LEVEL
    if fully_funded:
        status = "Fully funded: the assets cover the liabilities."
    elif above_minimum:
        status = ("In deficit but above the minimum funding level of "
                  f"{MINIMUM_FUNDING_LEVEL:.0%}: the employer should fund the difference.")
    else:
        status = f"Below the minimum funding level of {MINIMUM_FUNDING_LEVEL:.0%}."
    return {"funding_level": float(funding_level), "fully_funded": fully_funded,
            "meets_minimum": above_minimum, "status": status}


def dissolution_reserves(assets: float,
                         minimum_individual_reserves: Union[Sequence[float], pd.Series],
                         benefits_previously_paid: Union[float, Sequence[float]] = 0.0,
                         annuity_cost: float = 0.0) -> pd.DataFrame:
    """
    Minimum benefits on the dissolution of a fund (section 18(1)(b)).

    Each member receives at least the minimum individual reserve. Where the
    fair value of the assets is lower than the reserves of the members
    sharing in the distribution, after adjustment for benefits previously
    paid to them, plus the cost of annuity policies to provide equivalent
    pensions to pensioners and deferred pensioners, the reserves may be
    reduced in the ratio of the assets to that total.

    Args:
        assets: Fair value of the fund's assets
        minimum_individual_reserves: Minimum individual reserve of each
            member, from the formula approved by the Commission
        benefits_previously_paid: Benefits already paid to each member
        annuity_cost: Cost of annuity policies for pensioners and deferred
            pensioners

    Returns:
        One row per member: the adjusted reserve and the amount payable
        after any reduction. The ratio applied is in ``attrs["ratio"]``.
    """
    index = (minimum_individual_reserves.index
             if isinstance(minimum_individual_reserves, pd.Series) else None)
    reserves = np.asarray(minimum_individual_reserves, dtype=float)
    paid = np.broadcast_to(np.asarray(benefits_previously_paid, dtype=float), reserves.shape)
    if assets < 0 or annuity_cost < 0 or np.any(reserves < 0) or np.any(paid < 0):
        raise ValueError("amounts must be non-negative")
    adjusted = np.maximum(reserves - paid, 0.0)
    required = adjusted.sum() + annuity_cost
    ratio = 1.0 if required <= 0 else min(1.0, assets / required)
    result = pd.DataFrame({
        "minimum_individual_reserve": reserves,
        "adjusted_reserve": adjusted,
        "payable": adjusted * ratio,
    }, index=index)
    result.attrs["ratio"] = ratio
    result.attrs["annuity_cost_covered"] = annuity_cost * ratio
    return result


def surplus_share_on_exit(leaver_past_service_liability: float,
                          total_past_service_liability: float,
                          actuarial_surplus: float = 0.0,
                          investment_reserve: float = 0.0,
                          contingency_reserves: float = 0.0) -> float:
    """
    Share of surplus and reserves due to members who leave a fund
    (section 18(3)).

    Leavers receive, as part of their transfer values or benefits, a share
    of any actuarial surplus, of the investment reserve account and of such
    contingency reserve accounts as the board deems appropriate. The share
    is in the ratio of the fund's past service liability for the leavers to
    its past service liability for all members at the date of leaving.

    Args:
        leaver_past_service_liability: Past service liability for those
            leaving
        total_past_service_liability: Past service liability for all members
        actuarial_surplus: Actuarial surplus in the fund. A deficit is not
            shared: pass zero.
        investment_reserve: Investment reserve account
        contingency_reserves: Contingency reserves the board has decided to
            share

    Returns:
        The amount to add to the leavers' benefits
    """
    if total_past_service_liability <= 0:
        raise ValueError("total_past_service_liability must be positive")
    if not 0 <= leaver_past_service_liability <= total_past_service_liability:
        raise ValueError("the leavers' liability must lie between zero and the total")
    if min(actuarial_surplus, investment_reserve, contingency_reserves) < 0:
        raise ValueError("surplus and reserves must be non-negative")
    pool = actuarial_surplus + investment_reserve + contingency_reserves
    return pool * leaver_past_service_liability / total_past_service_liability
