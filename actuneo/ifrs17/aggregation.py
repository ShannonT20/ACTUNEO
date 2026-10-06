"""
Level of Aggregation

IFRS 17 measures insurance contracts in groups (IFRS 17.14-24). A group is
formed in three steps:

1. **Portfolio**: contracts subject to similar risks and managed together,
   for example motor or fire.
2. **Cohort**: contracts issued no more than one year apart.
3. **Profitability** at initial recognition:

   - onerous;
   - no significant possibility of becoming onerous subsequently;
   - the remaining contracts.

Under the premium allocation approach contracts are assumed not to be
onerous at initial recognition unless facts and circumstances indicate
otherwise (IFRS 17.18), so the profitability split relies on the expected
result of each contract or class of contracts.
"""

import pandas as pd

ONEROUS = "onerous"
NO_SIGNIFICANT_POSSIBILITY = "no significant possibility of becoming onerous"
REMAINING = "remaining"


def group_contracts(policies: pd.DataFrame,
                    portfolio: str = "portfolio",
                    inception: str = "start_date",
                    expected_combined_ratio: str = "expected_combined_ratio",
                    onerous_above: float = 1.0,
                    no_significant_possibility_below: float = 0.8,
                    cohort: str = "Y") -> pd.DataFrame:
    """
    Assign each contract to an IFRS 17 group.

    Profitability is judged from the expected combined ratio of each
    contract: expected claims, directly attributable expenses and risk
    adjustment for the cover, divided by the premium.

    Args:
        policies: DataFrame with one row per contract
        portfolio: Column naming the portfolio of each contract
        inception: Column with the date each contract is issued
        expected_combined_ratio: Column with the expected combined ratio
        onerous_above: Contracts with an expected combined ratio above this
            are onerous at initial recognition
        no_significant_possibility_below: Contracts with an expected combined
            ratio below this are treated as having no significant possibility
            of becoming onerous. Where the line is drawn is a judgement for
            the entity.
        cohort: Length of a cohort: "Y" for annual (the longest IFRS 17
            allows), "Q" for quarterly or "M" for monthly

    Returns:
        Copy of ``policies`` with ``cohort``, ``profitability`` and ``group``
        columns added
    """
    if cohort not in ("Y", "Q", "M"):
        raise ValueError("cohort must be 'Y', 'Q' or 'M'")
    if no_significant_possibility_below > onerous_above:
        raise ValueError("no_significant_possibility_below must not exceed onerous_above")

    result = policies.copy()
    dates = pd.to_datetime(result[inception])
    if cohort == "Y":
        result["cohort"] = dates.dt.year.astype(str)
    elif cohort == "Q":
        result["cohort"] = dates.dt.year.astype(str) + "Q" + dates.dt.quarter.astype(str)
    else:
        result["cohort"] = dates.dt.strftime("%Y-%m")

    ratio = result[expected_combined_ratio].astype(float)
    if ratio.isna().any():
        raise ValueError("expected combined ratios must not be missing")
    result["profitability"] = REMAINING
    result.loc[ratio > onerous_above, "profitability"] = ONEROUS
    result.loc[ratio < no_significant_possibility_below, "profitability"] = \
        NO_SIGNIFICANT_POSSIBILITY

    result["group"] = (result[portfolio].astype(str) + " | " + result["cohort"]
                       + " | " + result["profitability"])
    return result
