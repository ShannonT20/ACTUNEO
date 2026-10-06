"""
IFRS 17 Disclosures

Tables for the notes to the financial statements that relate to incurred
claims:

- the claims development table (IFRS 17.130);
- the maturity analysis of future cash flows (IFRS 17.132(b));
- sensitivity of the liability to changes in assumptions (IFRS 17.128).
"""

import numpy as np
import pandas as pd
from typing import Optional, Sequence
from ._common import discount_factors

_YEARS_LATER = ["At end of accident year", "One year later", "Two years later",
                "Three years later", "Four years later", "Five years later",
                "Six years later", "Seven years later", "Eight years later",
                "Nine years later"]


def _years_later(k: int) -> str:
    return _YEARS_LATER[k] if k < len(_YEARS_LATER) else f"{k} years later"


def claims_development_table(triangle,
                             estimates: Optional[pd.DataFrame] = None,
                             tail: float = 1.0,
                             factors: Optional[Sequence[float]] = None,
                             earlier_years_liability: float = 0.0,
                             effect_of_discounting: Optional[float] = None,
                             discount_rate=None,
                             risk_adjustment: float = 0.0) -> pd.DataFrame:
    """
    Claims development table: how the estimate of ultimate claims for each
    accident year has changed over time, reconciled to the liability for
    incurred claims (IFRS 17.130).

    The standard asks for actual claims compared with previous estimates of
    their undiscounted amount, going back to the earliest year with material
    uncertainty, but not more than ten years.

    Args:
        triangle: Triangle of cumulative paid claims with yearly origin and
            development periods, observed to the same year end for every
            origin year
        estimates: The estimates of ultimate claims actually made at each
            past year end, as a DataFrame with origin years in rows and the
            number of years later (0, 1, 2, ...) in columns. If omitted they
            are rebuilt by applying today's chain-ladder development factors
            to the claims paid at each past year end, which is a substitute
            for entities that did not keep their past estimates.
        tail: Tail factor for the rebuilt estimates
        factors: Selected development factors for the rebuilt estimates
        earlier_years_liability: Undiscounted liability for accident years
            before those in the table
        effect_of_discounting: Reduction in the liability from discounting.
            Calculated from ``discount_rate`` if that is given instead.
        discount_rate: Annual effective rate, or a yield curve object
        risk_adjustment: Risk adjustment for non-financial risk

    Returns:
        DataFrame with one column per accident year and a "Total" column
    """
    from ..loss_reserving import ChainLadder

    model = ChainLadder(triangle, tail=tail, factors=factors)
    paid = model.triangle.values
    n_origin, n_dev = paid.shape
    latest_idx = model._latest_idx
    origins = list(model.triangle.origin)

    if estimates is None:
        cdf = model.cdf.to_numpy()
        history = paid * cdf[None, :]
    else:
        history = estimates.reindex(index=origins).to_numpy(dtype=float)
        if history.shape[1] < latest_idx.max() + 1:
            raise ValueError("estimates needs a column for every year of development observed")
        history = history[:, :n_dev]
        for i, j0 in enumerate(latest_idx):
            history[i, j0 + 1:] = np.nan

    columns = origins + ["Total"]
    table = pd.DataFrame(columns=columns, dtype=float)
    for k in range(int(latest_idx.max()) + 1):
        table.loc[_years_later(k)] = list(history[:, k]) + [np.nan]

    current = np.array([history[i, j0] for i, j0 in enumerate(latest_idx)])
    cumulative_paid = model.latest.to_numpy()
    liability = current - cumulative_paid
    table.loc["Current estimate of cumulative claims"] = list(current) + [current.sum()]
    table.loc["Cumulative claims paid"] = list(-cumulative_paid) + [-cumulative_paid.sum()]
    table.loc["Liabilities for the accident years shown"] = list(liability) + [liability.sum()]

    undiscounted = liability.sum() + earlier_years_liability
    if effect_of_discounting is None:
        if discount_rate is None or estimates is not None:
            effect_of_discounting = 0.0
        else:
            effect_of_discounting = model.reserve() - model.discounted_reserve(discount_rate)

    blank = [np.nan] * n_origin
    table.loc["Liabilities for earlier accident years"] = blank + [earlier_years_liability]
    table.loc["Effect of discounting"] = blank + [-effect_of_discounting]
    table.loc["Risk adjustment for non-financial risk"] = blank + [risk_adjustment]
    table.loc["Liabilities for incurred claims"] = blank + [
        undiscounted - effect_of_discounting + risk_adjustment
    ]
    return table


def maturity_analysis(cash_flows: Sequence[float],
                      discount_rate=0.0,
                      timing: float = 0.5,
                      periods_per_year: int = 1,
                      years: int = 5) -> pd.Series:
    """
    Maturity analysis: the present value of the future cash flows by the
    year in which they are expected to be paid (IFRS 17.132(b)).

    Liabilities for remaining coverage measured under the premium allocation
    approach need not be included in this analysis.

    Args:
        cash_flows: Expected payments in each future period, nearest first
        discount_rate: Annual effective rate, or a yield curve object. Use 0
            for an analysis of undiscounted cash flows, which the standard
            also permits.
        timing: When payments fall within each period
        periods_per_year: Number of periods in a year
        years: Number of single-year bands before the final "more than" band

    Returns:
        Series of amounts by time band, with a total
    """
    flows = np.atleast_1d(np.asarray(cash_flows, dtype=float))
    periods = np.arange(len(flows))
    present_values = flows * discount_factors(discount_rate, (periods + timing) / periods_per_year)
    band = periods // periods_per_year

    labels = ["1 year or less"] + [f"{k}-{k + 1} years" for k in range(1, years)]
    result = {label: float(present_values[band == k].sum()) for k, label in enumerate(labels)}
    result[f"More than {years} years"] = float(present_values[band >= years].sum())
    result["Total"] = float(present_values.sum())
    return pd.Series(result, name="present_value")


def lic_sensitivity(cash_flows: Sequence[float],
                    discount_rate=0.0,
                    risk_adjustment: float = 0.0,
                    claims_shock: float = 0.05,
                    rate_shock: float = 0.01,
                    tax_rate: float = 0.0,
                    timing: float = 0.5,
                    periods_per_year: int = 1) -> pd.DataFrame:
    """
    Sensitivity of the liability for incurred claims, and so of profit and
    equity, to reasonably possible changes in assumptions (IFRS 17.128).

    Args:
        cash_flows: Expected payments in each future period, nearest first
        discount_rate: Annual effective discount rate. Rate sensitivities
            are given only when this is a single rate.
        risk_adjustment: Risk adjustment; it moves in proportion to claims
        claims_shock: Proportional change in ultimate claims to test
        rate_shock: Change in the discount rate to test
        tax_rate: Rate of tax, to show the effect on equity after tax
        timing: When payments fall within each period
        periods_per_year: Number of periods in a year

    Returns:
        DataFrame with the liability and the effect on profit before tax and
        on equity under each scenario. A higher liability reduces profit.
    """
    flows = np.atleast_1d(np.asarray(cash_flows, dtype=float))
    years = (np.arange(len(flows)) + timing) / periods_per_year

    def liability(scale: float, rate) -> float:
        return float(np.sum(scale * flows * discount_factors(rate, years))
                     + scale * risk_adjustment)

    base = liability(1.0, discount_rate)
    scenarios = {
        "Base": base,
        f"Claims +{claims_shock:.0%}": liability(1 + claims_shock, discount_rate),
        f"Claims -{claims_shock:.0%}": liability(1 - claims_shock, discount_rate),
    }
    if not hasattr(discount_rate, "get_discount_factor"):
        scenarios[f"Discount rate +{rate_shock:.1%}"] = liability(1.0, discount_rate + rate_shock)
        scenarios[f"Discount rate -{rate_shock:.1%}"] = liability(1.0, discount_rate - rate_shock)

    table = pd.DataFrame({"liability": pd.Series(scenarios)})
    table["profit_before_tax"] = base - table["liability"]
    table["equity"] = table["profit_before_tax"] * (1 - tax_rate)
    return table + 0.0
