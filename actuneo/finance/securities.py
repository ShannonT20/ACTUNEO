"""
Fixed-Interest Securities, Equities and Property

Prices and yields of bonds (with income tax and capital gains tax, and with
optional redemption dates), of shares by the discounted dividend model, and
real yields on index-linked or inflation-affected investments.

Yields are effective annual rates.
"""

import numpy as np
from typing import Optional, Sequence
from .cashflows import Cashflows


class Bond:
    """
    A fixed-interest security with coupons payable in arrears.

    Amounts are per ``nominal`` of the stock (100 by default).
    """

    def __init__(self,
                 coupon_rate: float,
                 term: float,
                 redemption: float = 100.0,
                 frequency: int = 2,
                 nominal: float = 100.0):
        """
        Args:
            coupon_rate: Annual coupon as a proportion of nominal (0.06 for 6%)
            term: Years to redemption (a whole number of coupon periods)
            redemption: Redemption payment per 100 nominal (100 for par)
            frequency: Number of coupons a year
            nominal: Nominal amount held
        """
        periods = term * frequency
        if term <= 0 or abs(periods - round(periods)) > 1e-9:
            raise ValueError("term must be a positive whole number of coupon periods")
        self.coupon_rate = float(coupon_rate)
        self.term = float(term)
        self.frequency = int(frequency)
        self.nominal = float(nominal)
        self.redemption = float(redemption) * nominal / 100.0
        self.annual_coupon = coupon_rate * nominal
        self.n_coupons = int(round(periods))

    def has_capital_gain(self, yield_: float, income_tax: float = 0.0) -> bool:
        """
        Whether a buyer at issue at this yield makes a capital gain on
        redemption: true when the yield, convertible at the coupon frequency,
        exceeds the net coupon as a proportion of the redemption amount.
        """
        nominal_yield = self.frequency * ((1 + yield_) ** (1 / self.frequency) - 1)
        return nominal_yield > (1 - income_tax) * self.annual_coupon / self.redemption

    def _tax_delays(self, income_tax_delay) -> np.ndarray:
        """Delay between each coupon and the income tax on it, cycling through the year."""
        delays = np.atleast_1d(np.asarray(income_tax_delay, dtype=float))
        if np.any(delays < 0):
            raise ValueError("tax delays must not be negative")
        return np.resize(delays, self.n_coupons)

    def _remaining(self, elapsed: float):
        """Coupon numbers still to be paid, and their times from the purchase date."""
        if not 0 <= elapsed < self.term:
            raise ValueError("elapsed must be from 0 to less than the term")
        numbers = np.arange(1, self.n_coupons + 1)
        times = numbers / self.frequency - elapsed
        keep = times > 1e-12
        return numbers[keep], times[keep]

    def price(self, yield_: float, income_tax: float = 0.0,
              capital_gains_tax: float = 0.0, elapsed: float = 0.0,
              income_tax_delay=0.0, capital_gains_tax_delay: float = 0.0) -> float:
        """
        Price to give the required yield.

        Args:
            yield_: Required effective annual yield (net of the taxes given)
            income_tax: Rate of income tax on coupons
            capital_gains_tax: Rate of tax on any capital gain at redemption
            elapsed: Years since issue, for a purchase between coupon dates.
                The price includes the interest accrued since the last coupon
                (the dirty price).
            income_tax_delay: Years after a coupon that the income tax on it
                is paid. A list gives the delay for each coupon of the year
                in turn, for tax collected on a fixed date.
            capital_gains_tax_delay: Years after redemption that capital
                gains tax is paid

        Returns:
            Price for the nominal amount held
        """
        if yield_ <= -1:
            raise ValueError("the yield must be greater than -1")
        numbers, times = self._remaining(elapsed)
        delays = self._tax_delays(income_tax_delay)[numbers - 1]
        growth = 1 + yield_
        coupon = self.annual_coupon / self.frequency
        gross = coupon * np.sum(growth ** -times)
        tax = income_tax * coupon * np.sum(growth ** -(times + delays))
        to_redemption = self.term - elapsed
        v_redemption = growth ** -to_redemption
        v_tax = growth ** -(to_redemption + capital_gains_tax_delay)
        without_gain = gross - tax + self.redemption * v_redemption
        if not capital_gains_tax:
            return float(without_gain)
        # Tax of cgt * (R - P) on a gain: solve for P, and use it only if there is a gain
        with_gain = (without_gain - capital_gains_tax * self.redemption * v_tax) \
            / (1 - capital_gains_tax * v_tax)
        return float(with_gain if with_gain < self.redemption else without_gain)

    def accrued_interest(self, elapsed: float) -> float:
        """Coupon accrued since the last coupon date, in proportion to time."""
        since_last = elapsed - np.floor(elapsed * self.frequency + 1e-12) / self.frequency
        return float(self.annual_coupon * since_last)

    def clean_price(self, yield_: float, elapsed: float = 0.0, **taxes) -> float:
        """Price excluding accrued interest: the dirty price less the accrued coupon."""
        return self.price(yield_, elapsed=elapsed, **taxes) - self.accrued_interest(elapsed)

    def capital_gains_tax_payable(self, price: float, capital_gains_tax: float) -> float:
        """Capital gains tax due at redemption on a stock bought at the given price."""
        return capital_gains_tax * max(0.0, self.redemption - price)

    def cashflows(self, price: float, income_tax: float = 0.0,
                  capital_gains_tax: float = 0.0, elapsed: float = 0.0,
                  income_tax_delay=0.0, capital_gains_tax_delay: float = 0.0) -> Cashflows:
        """Cashflows to a buyer at the given price, after tax, from the purchase date."""
        numbers, times = self._remaining(elapsed)
        delays = self._tax_delays(income_tax_delay)[numbers - 1]
        coupon = self.annual_coupon / self.frequency
        flows = Cashflows([0.0], [-price])
        for t, delay in zip(times, delays):
            flows.add(t, coupon)
            if income_tax:
                flows.add(t + delay, -income_tax * coupon)
        to_redemption = self.term - elapsed
        flows.add(to_redemption, self.redemption)
        gains_tax = self.capital_gains_tax_payable(price, capital_gains_tax)
        if gains_tax:
            flows.add(to_redemption + capital_gains_tax_delay, -gains_tax)
        return flows

    def redemption_yield(self, price: float, income_tax: float = 0.0,
                         capital_gains_tax: float = 0.0, elapsed: float = 0.0,
                         income_tax_delay=0.0, capital_gains_tax_delay: float = 0.0) -> float:
        """
        Redemption yield for a given price: gross if no taxes are given, net
        of tax otherwise.
        """
        return self.cashflows(price, income_tax, capital_gains_tax, elapsed, income_tax_delay,
                              capital_gains_tax_delay).internal_rate_of_return()

    def running_yield(self, price: float, income_tax: float = 0.0) -> float:
        """Annual coupon, net of income tax, as a proportion of the price."""
        return (1 - income_tax) * self.annual_coupon / price

    def __repr__(self) -> str:
        return (f"Bond(coupon={self.coupon_rate:.2%}, term={self.term:g} years, "
                f"redemption={self.redemption:g}, {self.frequency} coupons a year)")


def bond_price_with_optional_redemption(coupon_rate: float,
                                        earliest: float,
                                        latest: float,
                                        yield_: float,
                                        redemption: float = 100.0,
                                        frequency: int = 2,
                                        income_tax: float = 0.0,
                                        capital_gains_tax: float = 0.0) -> dict:
    """
    Maximum price to secure at least a given yield on a bond that the
    borrower may redeem at any coupon date between two dates.

    The buyer assumes the redemption date that is worst for them: the latest
    date if there is a capital gain (the gain comes as late as possible),
    the earliest date if there is a capital loss.

    Args:
        coupon_rate: Annual coupon as a proportion of nominal
        earliest: Years to the earliest redemption date
        latest: Years to the latest redemption date
        yield_: Minimum effective annual yield required
        redemption: Redemption payment per 100 nominal
        frequency: Number of coupons a year
        income_tax: Rate of income tax on coupons
        capital_gains_tax: Rate of tax on a capital gain

    Returns:
        Dictionary with the ``price`` per 100 nominal that secures the yield
        whatever date is chosen, the ``assumed_term`` it is based on, and the
        ``lower`` and ``upper`` bounds of the value: the prices if redemption
        were certain at the less and the more favourable of the two dates
    """
    if latest < earliest:
        raise ValueError("latest must not be before earliest")
    at_earliest = Bond(coupon_rate, earliest, redemption, frequency).price(
        yield_, income_tax, capital_gains_tax)
    at_latest = Bond(coupon_rate, latest, redemption, frequency).price(
        yield_, income_tax, capital_gains_tax)
    probe = Bond(coupon_rate, latest, redemption, frequency)
    term = latest if probe.has_capital_gain(yield_, income_tax) else earliest
    return {"price": at_latest if term == latest else at_earliest, "assumed_term": term,
            "lower": min(at_earliest, at_latest), "upper": max(at_earliest, at_latest)}


def equity_price(next_dividend: float, yield_: float, growth: float = 0.0,
                 time_to_next: float = 1.0, dividends_per_year: int = 1) -> float:
    """
    Price of a share as the present value of its expected dividends, growing
    at a constant rate for ever.

    Args:
        next_dividend: Amount of the next dividend
        yield_: Required effective annual rate of return
        growth: Rate of growth of dividends from one dividend to the next
        time_to_next: Years until the next dividend
        dividends_per_year: Number of dividends a year

    Returns:
        Price. With annual dividends starting in a year this is the familiar
        ``D / (i - g)``.
    """
    v_period = (1 + yield_) ** (-1 / dividends_per_year)
    ratio = (1 + growth) * v_period
    if ratio >= 1:
        raise ValueError("The yield must exceed the rate of dividend growth")
    return next_dividend * (1 + yield_) ** -time_to_next / (1 - ratio)


def equity_yield(price: float, next_dividend: float, growth: float = 0.0,
                 time_to_next: float = 1.0, dividends_per_year: int = 1) -> float:
    """
    Expected effective annual return on a share bought at a given price,
    with dividends growing at a constant rate for ever.
    """
    from scipy import optimize
    floor = (1 + growth) ** dividends_per_year - 1

    def difference(i):
        return equity_price(next_dividend, i, growth, time_to_next, dividends_per_year) - price

    return float(optimize.brentq(difference, floor + 1e-10, floor + 10.0, xtol=1e-12))


def present_value_of_dividends(dividends: Sequence[float], yield_: float,
                               terminal_growth: Optional[float] = None) -> float:
    """
    Present value of annual dividends paid at the end of each year, with an
    optional constant rate of growth after the last one given.

    Args:
        dividends: Expected dividends at the end of years 1, 2, ...
        yield_: Required effective annual rate of return
        terminal_growth: Growth rate of dividends after the last one listed,
            continuing for ever (None if dividends stop)
    """
    dividends = np.atleast_1d(np.asarray(dividends, dtype=float))
    years = np.arange(1, len(dividends) + 1)
    value = float(np.sum(dividends * (1 + yield_) ** -years))
    if terminal_growth is not None:
        if yield_ <= terminal_growth:
            raise ValueError("The yield must exceed the terminal rate of growth")
        following = dividends[-1] * (1 + terminal_growth)
        value += following / (yield_ - terminal_growth) * (1 + yield_) ** -len(dividends)
    return value


def real_yield(amounts: Sequence[float], times: Sequence[float],
               index_values: Sequence[float]) -> float:
    """
    Real rate of return on a set of cashflows, given the value of a price
    index at the time of each.

    Each cashflow is restated in the money of the first by dividing by the
    growth in the index, and the yield is found on the restated amounts.

    Args:
        amounts: Cashflows (paid negative, received positive)
        times: Time of each cashflow in years
        index_values: Price index at the time of each cashflow
    """
    amounts = np.asarray(amounts, dtype=float)
    index = np.asarray(index_values, dtype=float)
    if amounts.shape != index.shape or np.any(index <= 0):
        raise ValueError("one positive index value is needed for each cashflow")
    return Cashflows(times, amounts * index[0] / index).internal_rate_of_return()


def index_linked_cashflows(coupon_rate: float, term: float, base_index: float,
                           index_values: Sequence[float], frequency: int = 2,
                           redemption: float = 100.0, nominal: float = 100.0) -> tuple:
    """
    Money payments of an index-linked bond.

    Each coupon and the redemption payment are the nominal amounts scaled by
    the ratio of an index value to the base index. Where the bond uses a
    time lag, supply the lagged index value that applies to each payment.

    Args:
        coupon_rate: Nominal annual coupon as a proportion of nominal
        term: Years to redemption
        base_index: Index value the payments are measured against
        index_values: Index value applying to each coupon date, in order
        frequency: Number of coupons a year
        redemption: Nominal redemption payment per 100 nominal
        nominal: Nominal amount held

    Returns:
        ``(times, amounts)``: the time of each payment in years and its
        amount in money, with the last amount including the redemption
    """
    index = np.atleast_1d(np.asarray(index_values, dtype=float))
    n_coupons = int(round(term * frequency))
    if len(index) != n_coupons:
        raise ValueError(f"one index value is needed for each of the {n_coupons} coupons")
    if base_index <= 0 or np.any(index <= 0):
        raise ValueError("index values must be positive")
    times = np.arange(1, n_coupons + 1) / frequency
    amounts = coupon_rate * nominal / frequency * index / base_index
    amounts[-1] += redemption * nominal / 100.0 * index[-1] / base_index
    return times, amounts


def property_value(initial_rent: float, yield_: float, term: float, rent_growth: float = 0.0,
                   review_period: float = 1.0, p: int = 4, timing: str = "advance",
                   sale_proceeds: float = 0.0) -> float:
    """
    Value of a property as the present value of its rent, with rent reviewed
    at fixed intervals, and any proceeds of sale at the end.

    Args:
        initial_rent: Annual rent at the start
        yield_: Required effective annual rate of return
        term: Years for which rent is received (a whole number of review periods)
        rent_growth: Annual rate of growth in rents, applied at each review
        review_period: Years between rent reviews
        p: Number of rent payments a year
        timing: "advance" (the usual case for rent), "arrears" or "continuous"
        sale_proceeds: Amount received when the property is sold at the end
    """
    from .annuities import stepped_annuity
    steps = term / review_period
    if abs(steps - round(steps)) > 1e-9:
        raise ValueError("term must be a whole number of review periods")
    rent = initial_rent * stepped_annuity(review_period, int(round(steps)), yield_,
                                          rent_growth, p, timing)
    return float(rent + sale_proceeds * (1 + yield_) ** -term)
