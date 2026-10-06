"""
Hyperinflation Helpers

Basic calculations used when financial statements are restated for
hyperinflation under IAS 29 *Financial Reporting in Hyperinflationary
Economies*: restating amounts with a general price index and measuring the
gain or loss on the net monetary position.

These are building blocks, not a complete application of IAS 29. Choosing
the price index, classifying items as monetary or non-monetary, restating
comparatives and equity, and the interaction with IAS 21 all need judgement
and are left to the preparer.

Under IAS 29:

- monetary items (cash, receivables, payables and, by their nature as
  amounts to be paid in money, insurance contract liabilities) are already
  in the measuring unit current at the reporting date and are not restated;
- non-monetary items and the items of income and expense are restated from
  the date they arose by the change in a general price index;
- holding monetary assets loses purchasing power and holding monetary
  liabilities gains it. The net effect is reported in profit or loss.
"""

import numpy as np
from typing import Sequence


def cumulative_inflation(index_start: float, index_end: float) -> float:
    """
    Inflation between two dates from a general price index.

    Args:
        index_start: Index at the earlier date
        index_end: Index at the later date

    Returns:
        Proportional increase in prices (1.0 for 100%)
    """
    if index_start <= 0 or index_end <= 0:
        raise ValueError("price indices must be positive")
    return index_end / index_start - 1


def exceeds_hyperinflation_indicator(index_three_years_ago: float, index_now: float,
                                     threshold: float = 1.0) -> bool:
    """
    Whether cumulative inflation over three years approaches or exceeds 100%.

    This is one of the indicators of hyperinflation in IAS 29.3, not a test
    on its own: the standard also looks at how people hold wealth, quote
    prices and index wages, and the conclusion is a matter of judgement.

    Args:
        index_three_years_ago: General price index three years before
        index_now: General price index now
        threshold: Cumulative inflation regarded as hyperinflationary

    Returns:
        True if cumulative three-year inflation is at or above the threshold
    """
    return cumulative_inflation(index_three_years_ago, index_now) >= threshold


def restate(amounts, index_at_origin, index_at_reporting_date: float):
    """
    Restate amounts to the measuring unit current at the reporting date.

    Args:
        amounts: Amount, or amounts, in the money of the date they arose
        index_at_origin: General price index when each amount arose
        index_at_reporting_date: General price index at the reporting date

    Returns:
        Restated amount or array of amounts
    """
    amounts = np.asarray(amounts, dtype=float)
    origin = np.asarray(index_at_origin, dtype=float)
    if np.any(origin <= 0) or index_at_reporting_date <= 0:
        raise ValueError("price indices must be positive")
    restated = amounts * index_at_reporting_date / origin
    return float(restated) if restated.ndim == 0 else restated


def net_monetary_gain_or_loss(opening_net_monetary_position: float,
                              net_monetary_flows: Sequence[float],
                              indices: Sequence[float],
                              opening_index: float,
                              closing_index: float) -> float:
    """
    Gain or loss on the net monetary position for a period (IAS 29.27-28).

    The opening net monetary position and each movement in it are restated
    to closing prices. The gain or loss is the actual closing position less
    that restated amount: an entity with net monetary assets loses, and one
    with net monetary liabilities gains.

    Args:
        opening_net_monetary_position: Monetary assets less monetary
            liabilities at the start of the period
        net_monetary_flows: Net increase in monetary items in each
            sub-period (for example each month)
        indices: General price index for each sub-period
        opening_index: General price index at the start of the period
        closing_index: General price index at the end of the period

    Returns:
        Gain (positive) or loss (negative) on the net monetary position
    """
    flows = np.asarray(net_monetary_flows, dtype=float)
    index = np.asarray(indices, dtype=float)
    if flows.shape != index.shape:
        raise ValueError("net_monetary_flows and indices must have the same length")
    actual_closing = opening_net_monetary_position + flows.sum()
    restated = (restate(opening_net_monetary_position, opening_index, closing_index)
                + np.sum(restate(flows, index, closing_index)))
    return float(actual_closing - restated)
