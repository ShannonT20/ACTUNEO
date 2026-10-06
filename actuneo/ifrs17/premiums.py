"""
Earning of Premiums

Under the premium allocation approach, insurance revenue for a period is the
part of the expected premium receipts that relates to the cover provided in
that period, normally allocated by the passage of time (IFRS 17.B126).

These functions earn premiums by days of cover from a policy listing, the
"365ths" method used for unearned premium calculations.
"""

import numpy as np
import pandas as pd
from typing import Sequence


def _days_of_cover(start: pd.Series, end: pd.Series) -> pd.Series:
    days = (end - start).dt.days + 1
    if (days <= 0).any():
        raise ValueError("a policy ends before it starts")
    return days


def earned_fraction(start_dates, end_dates, valuation_date) -> np.ndarray:
    """
    Proportion of each policy's cover that has expired at the valuation date.

    Cover runs from the start date to the end date, both included, and the
    valuation is at the close of the valuation date.

    Args:
        start_dates: First day of cover of each policy
        end_dates: Last day of cover of each policy
        valuation_date: Date of the valuation

    Returns:
        Array of proportions between 0 and 1
    """
    start = pd.Series(pd.to_datetime(start_dates)).reset_index(drop=True)
    end = pd.Series(pd.to_datetime(end_dates)).reset_index(drop=True)
    valuation = pd.Timestamp(valuation_date)
    total = _days_of_cover(start, end)
    expired = ((valuation - start).dt.days + 1).clip(lower=0)
    return np.minimum(expired, total).to_numpy() / total.to_numpy()


def unearned_premium(premiums, start_dates, end_dates, valuation_date) -> np.ndarray:
    """
    Unearned premium of each policy at the valuation date, by days of cover.

    Args:
        premiums: Premium of each policy for its full period of cover
        start_dates: First day of cover of each policy
        end_dates: Last day of cover of each policy
        valuation_date: Date of the valuation

    Returns:
        Array of unearned premiums
    """
    premiums = np.asarray(premiums, dtype=float)
    return premiums * (1 - earned_fraction(start_dates, end_dates, valuation_date))


def earned_premium_by_period(policies: pd.DataFrame,
                             period_ends: Sequence,
                             premium: str = "premium",
                             start: str = "start_date",
                             end: str = "end_date") -> pd.Series:
    """
    Premium earned in each reporting period from a listing of policies.

    Args:
        policies: DataFrame with one row per policy
        period_ends: Reporting dates, in order. The first period runs from
            the earliest start of cover to the first reporting date.
        premium: Column with the premium for the full period of cover
        start: Column with the first day of cover
        end: Column with the last day of cover

    Returns:
        Series of earned premium indexed by reporting date. Premium for
        cover after the last reporting date is not included; it is the
        unearned premium at that date.
    """
    dates = [pd.Timestamp(d) for d in period_ends]
    if any(later <= earlier for earlier, later in zip(dates, dates[1:])):
        raise ValueError("period_ends must be in increasing order")

    premiums = policies[premium].to_numpy(dtype=float)
    earned_to_date = np.array([
        float(np.sum(premiums * earned_fraction(policies[start], policies[end], d)))
        for d in dates
    ])
    earned = np.diff(np.concatenate(([0.0], earned_to_date)))
    return pd.Series(earned, index=pd.Index(dates, name="period_end"), name="earned_premium")
