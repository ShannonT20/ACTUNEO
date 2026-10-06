"""
World Bank indicators.

Source: The World Bank, World Development Indicators, through the World
Bank Indicators API (api.worldbank.org). Licence: CC BY 4.0.
"""

import json
from typing import Callable, Optional, Sequence
from urllib.request import urlopen

import pandas as pd

#: Short names for commonly used World Bank indicator codes
INDICATORS = {
    "inflation": "FP.CPI.TOTL.ZG",              # consumer prices, annual %
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",          # real GDP growth, annual %
    "gdp_per_capita": "NY.GDP.PCAP.CD",         # current US$
    "exchange_rate": "PA.NUS.FCRF",             # local currency per US$, period average
    "lending_rate": "FR.INR.LEND",              # %
    "deposit_rate": "FR.INR.DPST",              # %
    "real_interest_rate": "FR.INR.RINR",        # %
    "population": "SP.POP.TOTL",
    "life_expectancy": "SP.DYN.LE00.IN",        # at birth, years
}

#: ISO 3166 three-letter codes of some African countries
COUNTRIES = {
    "Botswana": "BWA", "Egypt": "EGY", "Ethiopia": "ETH", "Ghana": "GHA", "Kenya": "KEN",
    "Malawi": "MWI", "Mauritius": "MUS", "Morocco": "MAR", "Mozambique": "MOZ",
    "Namibia": "NAM", "Nigeria": "NGA", "Rwanda": "RWA", "South Africa": "ZAF",
    "Tanzania": "TZA", "Uganda": "UGA", "Zambia": "ZMB", "Zimbabwe": "ZWE",
}

_URL = ("https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
        "?format=json&per_page=1000&date={start}:{end}")


def _download(url: str) -> str:
    with urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")


def world_bank_indicator(country: str,
                         indicator: str,
                         start: int = 2000,
                         end: int = 2030,
                         fetch: Optional[Callable[[str], str]] = None) -> pd.Series:
    """
    One World Bank indicator for one country, by year.

    Args:
        country: Country name as in :data:`COUNTRIES`, or an ISO three-letter code
        indicator: Short name as in :data:`INDICATORS`, or a World Bank
            indicator code
        start: First year
        end: Last year
        fetch: Function that takes a URL and returns the response text.
            Supply one to use a cache or a proxy; by default the data is
            downloaded from api.worldbank.org.

    Returns:
        Series indexed by year, in the units of the indicator (percentages
        are in per cent, not decimals). Years with no figure are left out.
    """
    code = COUNTRIES.get(country, country).upper()
    series = INDICATORS.get(indicator, indicator)
    url = _URL.format(country=code, indicator=series, start=int(start), end=int(end))
    try:
        payload = json.loads((fetch or _download)(url))
    except OSError as exc:
        raise ConnectionError(
            "Could not reach the World Bank API. An internet connection is needed."
        ) from exc
    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        message = payload[0].get("message") if isinstance(payload, list) and payload else payload
        raise ValueError(f"The World Bank API returned no data for {code} / {series}: {message}")

    values = {int(row["date"]): row["value"] for row in payload[1] if row["value"] is not None}
    result = pd.Series(values, name=indicator, dtype=float).sort_index()
    result.index.name = "year"
    return result


def macro_data(country: str,
               indicators: Sequence[str] = ("inflation", "gdp_growth", "exchange_rate",
                                            "lending_rate"),
               start: int = 2000,
               end: int = 2030,
               fetch: Optional[Callable[[str], str]] = None) -> pd.DataFrame:
    """
    Several World Bank indicators for one country, side by side.

    Args:
        country: Country name or ISO three-letter code
        indicators: Short names or World Bank indicator codes
        start: First year
        end: Last year
        fetch: Function that takes a URL and returns the response text

    Returns:
        DataFrame indexed by year with one column per indicator. An
        indicator the World Bank does not publish for the country is an
        empty column.
    """
    columns = {}
    for indicator in indicators:
        try:
            columns[indicator] = world_bank_indicator(country, indicator, start, end, fetch)
        except ValueError:
            columns[indicator] = pd.Series(dtype=float, name=indicator)
    frame = pd.DataFrame(columns)
    frame.index.name = "year"
    return frame.sort_index()
