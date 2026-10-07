"""
Zimbabwe Module

Zimbabwe-specific actuarial reference data.

This module provides:

- A catalogue of the Zimbabwe 2023 mortality tables shipped with ACTUNEO
- Currency reference data for converting Zimbabwe dollar (ZWL) amounts
  to Zimbabwe Gold (ZiG)
- Minimum capital requirements for insurers as reported for Statutory
  Instrument 67 of 2025, for reference only

Background
----------
The Insurance and Pensions Commission (IPEC) launched Zimbabwe's first
country-specific mortality tables in July 2023. Before that, insurers and
pension funds relied on tables from other countries, mainly South Africa and
the United Kingdom.

Zimbabwe Gold (ZiG, ISO 4217 code ZWG) replaced the Zimbabwe dollar (ZWL) in
April 2024 under Statutory Instrument 60 of 2024. Zimbabwe dollar balances
were converted at 2,498.7242 ZWL for one ZiG.
"""

import pandas as pd
from ..mortality import MortalityTable

# name: (class of business, lives covered, sex)
_TABLE_INFO = {
    "male_assured_lives": ("Individual life assurance", "Assured lives", "male"),
    "group_life_male": ("Group life assurance", "Members", "male"),
    "group_life_female": ("Group life assurance", "Members", "female"),
    "funeral_principal_members": ("Funeral assurance", "Principal members", "combined"),
    "funeral_spouses": ("Funeral assurance", "Spouses", "combined"),
    "funeral_adult_dependents": ("Funeral assurance", "Adult dependents", "combined"),
    "pre_retirement_pensions_male": ("Pensions, pre-retirement", "Active members", "male"),
    "pre_retirement_pensions_female": ("Pensions, pre-retirement", "Active members", "female"),
    "post_retirement_pensions_male": ("Pensions, post-retirement", "Pensioners", "male"),
    "post_retirement_pensions_female": ("Pensions, post-retirement", "Pensioners", "female"),
}

#: ISO 4217 code of Zimbabwe Gold (ZiG)
CURRENCY_CODE = "ZWG"

#: Zimbabwe dollars (ZWL) exchanged for one ZiG at the April 2024 conversion
ZWL_PER_ZWG = 2498.7242

#: Date from which ZiG replaced the Zimbabwe dollar
ZWG_EFFECTIVE_DATE = "2024-04-08"

#: Minimum capital in US dollars by class of insurer, as reported in published
#: summaries of Statutory Instrument 67 of 2025. Taken from secondary sources,
#: not from the gazetted instrument: check the instrument before relying on them.
MINIMUM_CAPITAL_USD = {
    "life": 2_000_000,
    "funeral": 500_000,
    "non-life": 1_500_000,
    "reinsurance": 2_000_000,
    "micro-insurance": 100_000,
    "broker": 100_000,
}


def mortality_tables() -> pd.DataFrame:
    """
    Catalogue of the Zimbabwe 2023 mortality tables.

    Returns:
        DataFrame indexed by table name with the class of business, lives
        covered, sex and the age range tabulated
    """
    rows = []
    for name in MortalityTable.zimbabwe_2023_tables():
        table = MortalityTable.from_zimbabwe_2023(name)
        business, lives, sex = _TABLE_INFO[name]
        rows.append({
            "table": name,
            "class_of_business": business,
            "lives": lives,
            "sex": sex,
            "min_age": table.min_age,
            "max_age": table.max_age,
        })
    return pd.DataFrame(rows).set_index("table")


def load_mortality_table(name: str) -> MortalityTable:
    """
    Load a Zimbabwe 2023 mortality table by name.

    Args:
        name: Table name, see :func:`mortality_tables`

    Returns:
        MortalityTable instance
    """
    return MortalityTable.from_zimbabwe_2023(name)


def minimum_capital(insurer_class: str) -> int:
    """
    Minimum capital in US dollars for a class of insurer, as reported for
    Statutory Instrument 67 of 2025.

    The figures come from published summaries of the instrument, not from
    the gazetted text, and regulations change. They are a reference point,
    not a statement of what an insurer must hold: the risk-based capital
    required under ZICARP is a separate calculation that this library does
    not perform.

    Args:
        insurer_class: One of "life", "funeral", "non-life", "reinsurance",
            "micro-insurance" or "broker"

    Returns:
        Minimum capital in US dollars
    """
    key = str(insurer_class).strip().lower().replace("_", "-").replace(" ", "-")
    if key not in MINIMUM_CAPITAL_USD:
        raise ValueError(
            f"Unknown class {insurer_class!r}. Choose from {sorted(MINIMUM_CAPITAL_USD)}")
    return MINIMUM_CAPITAL_USD[key]


def zwl_to_zwg(amount_zwl: float) -> float:
    """
    Convert a Zimbabwe dollar (ZWL) amount to ZiG at the official April 2024
    conversion rate of 2,498.7242 ZWL for one ZiG.

    Args:
        amount_zwl: Amount in Zimbabwe dollars

    Returns:
        Amount in ZiG
    """
    return amount_zwl / ZWL_PER_ZWG


__all__ = [
    'mortality_tables',
    'load_mortality_table',
    'zwl_to_zwg',
    'minimum_capital',
    'MINIMUM_CAPITAL_USD',
    'CURRENCY_CODE',
    'ZWL_PER_ZWG',
    'ZWG_EFFECTIVE_DATE',
]
