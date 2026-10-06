"""
Term Structure, Duration and Immunisation

Spot rates, forward rates and par yields, and the measures of interest rate
sensitivity used to match assets to liabilities: discounted mean term,
volatility and convexity, and Redington's conditions for immunisation.

Rates are effective annual rates and times are in years.
"""

import numpy as np
from typing import Sequence


def forward_rate(spot_start: float, start: float, spot_end: float, end: float) -> float:
    """
    Forward rate implied by two spot rates: the effective annual rate agreed
    now for an investment made at ``start`` and repaid at ``end``.

    Args:
        spot_start: Spot rate for a term of ``start`` years
        start: Time the forward investment begins
        spot_end: Spot rate for a term of ``end`` years
        end: Time the forward investment is repaid
    """
    if end <= start or start < 0:
        raise ValueError("end must be later than start, and start not negative")
    growth = (1 + spot_end) ** end / (1 + spot_start) ** start
    return float(growth ** (1 / (end - start)) - 1)


def spot_rates_from_forwards(forward_rates: Sequence[float]) -> np.ndarray:
    """
    Spot rates for terms 1, 2, ... from one-year forward rates.

    Args:
        forward_rates: One-year forward rates for years starting at times
            0, 1, 2, ...
    """
    forwards = np.asarray(forward_rates, dtype=float)
    growth = np.cumprod(1 + forwards)
    return growth ** (1 / np.arange(1, len(forwards) + 1)) - 1


def forwards_from_spot_rates(spot_rates: Sequence[float]) -> np.ndarray:
    """One-year forward rates for years starting at 0, 1, ... from spot rates for terms 1, 2, ..."""
    spots = np.asarray(spot_rates, dtype=float)
    growth = (1 + spots) ** np.arange(1, len(spots) + 1)
    return growth / np.concatenate(([1.0], growth[:-1])) - 1


def par_yield(spot_rates: Sequence[float]) -> float:
    """
    Par yield for the term of the last spot rate: the annual coupon, paid in
    arrears, at which a bond redeemed at par is priced at par.

    Args:
        spot_rates: Spot rates for terms 1, 2, ..., n
    """
    spots = np.asarray(spot_rates, dtype=float)
    discount = (1 + spots) ** -np.arange(1, len(spots) + 1)
    return float((1 - discount[-1]) / discount.sum())


def spot_rates_from_bonds(cashflows: Sequence[Sequence[float]],
                          prices: Sequence[float]) -> np.ndarray:
    """
    Spot rates for terms 1, 2, ..., n implied by the prices of n bonds.

    Args:
        cashflows: One row per bond with its payments at the end of years
            1 to n (coupons, and the coupon plus redemption in its last year)
        prices: Price of each bond

    Returns:
        Spot rates that reproduce every price
    """
    payments = np.asarray(cashflows, dtype=float)
    prices = np.asarray(prices, dtype=float)
    if payments.ndim != 2 or payments.shape[0] != payments.shape[1] \
            or payments.shape[0] != len(prices):
        raise ValueError("n bonds with payments over n years are needed")
    discount = np.linalg.solve(payments, prices)
    if np.any(discount <= 0):
        raise ValueError("The prices imply a discount factor that is not positive")
    return discount ** (-1 / np.arange(1, len(prices) + 1)) - 1


def _flows(amounts, times):
    amounts = np.atleast_1d(np.asarray(amounts, dtype=float))
    times = np.atleast_1d(np.asarray(times, dtype=float))
    if amounts.shape != times.shape:
        raise ValueError("amounts and times must have the same length")
    return amounts, times


def discounted_mean_term(amounts: Sequence[float], times: Sequence[float], i: float) -> float:
    """
    Discounted mean term (Macaulay duration): the average time of the
    cashflows, weighted by their present values.
    """
    amounts, times = _flows(amounts, times)
    present_values = amounts * (1 + i) ** -times
    return float(np.sum(times * present_values) / np.sum(present_values))


def volatility(amounts: Sequence[float], times: Sequence[float], i: float) -> float:
    """
    Volatility (modified duration): the proportional fall in present value
    for a small rise in the rate of interest, ``-P'(i) / P(i)``.
    """
    return discounted_mean_term(amounts, times, i) / (1 + i)


def convexity(amounts: Sequence[float], times: Sequence[float], i: float) -> float:
    """Convexity: ``P''(i) / P(i)``, the sum of ``t (t + 1) C v**(t + 2)`` over the present value."""
    amounts, times = _flows(amounts, times)
    v = 1 / (1 + i)
    present_value = np.sum(amounts * v ** times)
    return float(np.sum(times * (times + 1) * amounts * v ** (times + 2)) / present_value)


def immunisation_check(asset_amounts: Sequence[float], asset_times: Sequence[float],
                       liability_amounts: Sequence[float], liability_times: Sequence[float],
                       i: float, tolerance: float = 1e-4) -> dict:
    """
    Test Redington's three conditions for immunisation against a small
    change in the rate of interest:

    1. the present values of assets and liabilities are equal;
    2. their discounted mean terms (or volatilities) are equal;
    3. the convexity of the assets is greater than that of the liabilities.

    Args:
        asset_amounts: Asset cashflows
        asset_times: Times of the asset cashflows
        liability_amounts: Liability cashflows
        liability_times: Times of the liability cashflows
        i: Current effective annual rate of interest
        tolerance: Relative difference treated as equal

    Returns:
        Dictionary with the present values, discounted mean terms and
        convexities of both sides, whether each condition holds and whether
        the fund is ``immunised``
    """
    a_amounts, a_times = _flows(asset_amounts, asset_times)
    l_amounts, l_times = _flows(liability_amounts, liability_times)
    v = 1 / (1 + i)
    pv_assets = float(np.sum(a_amounts * v ** a_times))
    pv_liabilities = float(np.sum(l_amounts * v ** l_times))
    result = {
        "pv_assets": pv_assets,
        "pv_liabilities": pv_liabilities,
        "dmt_assets": discounted_mean_term(a_amounts, a_times, i),
        "dmt_liabilities": discounted_mean_term(l_amounts, l_times, i),
        "convexity_assets": convexity(a_amounts, a_times, i),
        "convexity_liabilities": convexity(l_amounts, l_times, i),
    }
    result["present_values_equal"] = bool(
        abs(pv_assets - pv_liabilities) <= tolerance * abs(pv_liabilities))
    result["mean_terms_equal"] = bool(
        abs(result["dmt_assets"] - result["dmt_liabilities"])
        <= tolerance * abs(result["dmt_liabilities"]))
    result["assets_more_convex"] = bool(
        result["convexity_assets"] > result["convexity_liabilities"])
    result["immunised"] = (result["present_values_equal"] and result["mean_terms_equal"]
                           and result["assets_more_convex"])
    return result


def immunising_amounts(liability_amounts: Sequence[float], liability_times: Sequence[float],
                       asset_a_amounts: Sequence[float], asset_a_times: Sequence[float],
                       asset_b_amounts: Sequence[float], asset_b_times: Sequence[float],
                       i: float) -> dict:
    """
    Amounts to invest in two assets so that the present value and the
    discounted mean term of the assets equal those of the liabilities
    (Redington's first two conditions).

    Args:
        liability_amounts: Liability cashflows
        liability_times: Times of the liability cashflows
        asset_a_amounts: Cashflows of one unit of asset A
        asset_a_times: Times of asset A's cashflows
        asset_b_amounts: Cashflows of one unit of asset B
        asset_b_times: Times of asset B's cashflows
        i: Effective annual rate of interest

    Returns:
        Dictionary with the money to invest in each asset (``invest_a``,
        ``invest_b``) and whether the third condition, on convexity, holds.
        A negative amount means the two assets cannot match the liabilities
        without selling one short.
    """
    l_amounts, l_times = _flows(liability_amounts, liability_times)
    pv = float(np.sum(l_amounts * (1 + i) ** -l_times))
    target = discounted_mean_term(l_amounts, l_times, i)
    dmt_a = discounted_mean_term(asset_a_amounts, asset_a_times, i)
    dmt_b = discounted_mean_term(asset_b_amounts, asset_b_times, i)
    if abs(dmt_a - dmt_b) < 1e-12:
        raise ValueError("The two assets have the same discounted mean term")
    invest_a = pv * (target - dmt_b) / (dmt_a - dmt_b)
    invest_b = pv - invest_a
    convexity_assets = (invest_a * convexity(asset_a_amounts, asset_a_times, i)
                        + invest_b * convexity(asset_b_amounts, asset_b_times, i)) / pv
    convexity_liabilities = convexity(l_amounts, l_times, i)
    return {
        "invest_a": float(invest_a),
        "invest_b": float(invest_b),
        "present_value": pv,
        "discounted_mean_term": target,
        "convexity_assets": float(convexity_assets),
        "convexity_liabilities": convexity_liabilities,
        "assets_more_convex": bool(convexity_assets > convexity_liabilities),
    }


def continuous_spot_rate(zero_coupon_price: float, term: float) -> float:
    """
    Continuous-time spot rate (the spot force of interest) from the price of
    a zero-coupon bond paying 1 at the end of the term: ``-ln(P) / t``.
    """
    if zero_coupon_price <= 0 or term <= 0:
        raise ValueError("price and term must be positive")
    return float(-np.log(zero_coupon_price) / term)


def continuous_forward_rate(price_start: float, price_end: float, period: float) -> float:
    """
    Continuous-time forward rate over a period, from the prices of
    zero-coupon bonds maturing at its start and at its end:
    ``ln(P_start / P_end) / period``.

    Args:
        price_start: Price of a unit zero-coupon bond maturing at the start
        price_end: Price of a unit zero-coupon bond maturing at the end
        period: Length of the forward period in years
    """
    if price_start <= 0 or price_end <= 0 or period <= 0:
        raise ValueError("prices and period must be positive")
    return float(np.log(price_start / price_end) / period)


def continuous_forward_from_spots(spot_start: float, start: float,
                                  spot_end: float, end: float) -> float:
    """
    Continuous-time forward rate between two times from the continuous-time
    spot rates for those terms: ``(T * Y_T - t * Y_t) / (T - t)``.
    """
    if end <= start:
        raise ValueError("end must be later than start")
    return float((end * spot_end - start * spot_start) / (end - start))


def instantaneous_forward_rate(spot_function, t: float, step: float = 1e-5) -> float:
    """
    Instantaneous forward rate at time t from a function giving the
    continuous-time spot rate for any term: the derivative of ``t * Y(t)``.

    Args:
        spot_function: Function of the term returning the continuous-time spot rate
        t: Time at which the forward rate applies
        step: Step for the numerical derivative
    """
    lower = max(t - step, 0.0)
    upper = t + step
    return float((upper * spot_function(upper) - lower * spot_function(lower))
                 / (upper - lower))


def effective_to_continuous(rate: float) -> float:
    """Continuous-time rate equivalent to an effective annual rate: ``ln(1 + rate)``."""
    return float(np.log1p(rate))


def continuous_to_effective(rate: float) -> float:
    """Effective annual rate equivalent to a continuous-time rate: ``exp(rate) - 1``."""
    return float(np.expm1(rate))


def estimated_value_change(present_value: float, volatility_: float, convexity_: float,
                           shift: float) -> float:
    """
    Approximate change in a present value for a shift in the rate of
    interest, from its volatility and convexity:
    ``PV * (-volatility * shift + convexity * shift**2 / 2)``.

    Args:
        present_value: Present value at the current rate
        volatility_: Volatility (effective duration) at the current rate
        convexity_: Convexity at the current rate
        shift: Change in the effective annual rate (0.01 for one point)
    """
    return float(present_value * (-volatility_ * shift + 0.5 * convexity_ * shift ** 2))


#: Effective duration is another name for volatility
effective_duration = volatility
