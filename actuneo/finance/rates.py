"""
Interest Rates

Rates of interest and discount and the conversions between them: effective
and nominal rates, simple interest and discount, the force of interest
(constant or varying with time), and real and money rates.

Notation
--------
- ``i``      effective rate of interest per year
- ``d``      effective rate of discount per year, ``d = i / (1 + i)``
- ``v``      discount factor for one year, ``v = 1 / (1 + i)``
- ``delta``  force of interest, ``delta = ln(1 + i)``
- ``i(p)``   nominal rate of interest convertible p times a year
- ``d(p)``   nominal rate of discount convertible p times a year
"""

import numpy as np
import pandas as pd
from typing import Callable, Sequence, Union


class InterestRate:
    """
    A rate of interest, held as its effective annual equivalent, that can be
    expressed in any of the usual forms.

    Create it from whichever form is given, then read off the others::

        rate = InterestRate.from_nominal(0.12, 12)   # 12% pa convertible monthly
        rate.effective        # 0.126825...
        rate.nominal_discount(4)
    """

    def __init__(self, effective: float):
        """
        Args:
            effective: Effective annual rate of interest
        """
        if effective <= -1:
            raise ValueError("The effective rate of interest must be greater than -1")
        self.effective = float(effective)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_effective(cls, i: float) -> 'InterestRate':
        """From an effective annual rate of interest."""
        return cls(i)

    @classmethod
    def from_nominal(cls, rate: float, p: int) -> 'InterestRate':
        """From a nominal rate of interest convertible p times a year, i(p)."""
        return cls((1 + rate / p) ** p - 1)

    @classmethod
    def from_discount(cls, d: float) -> 'InterestRate':
        """From an effective annual rate of discount."""
        if d >= 1:
            raise ValueError("A rate of discount must be less than 1")
        return cls(d / (1 - d))

    @classmethod
    def from_nominal_discount(cls, rate: float, p: int) -> 'InterestRate':
        """From a nominal rate of discount convertible p times a year, d(p)."""
        if rate / p >= 1:
            raise ValueError("The rate of discount per period must be less than 1")
        return cls((1 - rate / p) ** -p - 1)

    @classmethod
    def from_force(cls, delta: float) -> 'InterestRate':
        """From a constant force of interest."""
        return cls(float(np.expm1(delta)))

    @classmethod
    def from_period(cls, rate: float, years: float) -> 'InterestRate':
        """
        From an effective rate over a period of any length.

        Args:
            rate: Effective rate of interest over the period
            years: Length of the period in years (1/12 for a month, 2 for
                two years)
        """
        return cls((1 + rate) ** (1 / years) - 1)

    @classmethod
    def from_simple(cls, rate: float, term: float) -> 'InterestRate':
        """
        The effective annual rate that gives the same accumulation as simple
        interest at ``rate`` per year over ``term`` years.
        """
        return cls((1 + rate * term) ** (1 / term) - 1)

    @classmethod
    def from_simple_discount(cls, rate: float, term: float) -> 'InterestRate':
        """
        The effective annual rate that gives the same present value as
        simple (commercial) discount at ``rate`` per year over ``term`` years.
        """
        if rate * term >= 1:
            raise ValueError("Simple discount over the term must be less than 1")
        return cls((1 - rate * term) ** (-1 / term) - 1)

    # ------------------------------------------------------------------
    # Equivalent rates
    # ------------------------------------------------------------------
    @property
    def i(self) -> float:
        """Effective annual rate of interest."""
        return self.effective

    @property
    def v(self) -> float:
        """Discount factor for one year."""
        return 1 / (1 + self.effective)

    @property
    def d(self) -> float:
        """Effective annual rate of discount."""
        return self.effective / (1 + self.effective)

    @property
    def delta(self) -> float:
        """Force of interest."""
        return float(np.log1p(self.effective))

    def nominal(self, p: int) -> float:
        """Nominal rate of interest convertible p times a year, i(p)."""
        return p * ((1 + self.effective) ** (1 / p) - 1)

    def nominal_discount(self, p: int) -> float:
        """Nominal rate of discount convertible p times a year, d(p)."""
        return p * (1 - (1 + self.effective) ** (-1 / p))

    def per_period(self, years: float) -> float:
        """Effective rate of interest over a period of the given length in years."""
        return (1 + self.effective) ** years - 1

    def simple_equivalent(self, term: float) -> float:
        """Simple rate of interest giving the same accumulation over the term."""
        return ((1 + self.effective) ** term - 1) / term

    def simple_discount_equivalent(self, term: float) -> float:
        """Simple rate of discount giving the same present value over the term."""
        return (1 - (1 + self.effective) ** -term) / term

    # ------------------------------------------------------------------
    # Single payments
    # ------------------------------------------------------------------
    def accumulate(self, amount: float = 1.0, t: float = 1.0) -> float:
        """Accumulated value of an amount after t years."""
        return amount * (1 + self.effective) ** t

    def discount(self, amount: float = 1.0, t: float = 1.0) -> float:
        """Present value of an amount due in t years."""
        return amount * (1 + self.effective) ** -t

    def summary(self, frequencies: Sequence[int] = (2, 4, 12)) -> pd.Series:
        """The rate in its equivalent forms."""
        values = {"i": self.i, "d": self.d, "v": self.v, "delta": self.delta}
        for p in frequencies:
            values[f"i({p})"] = self.nominal(p)
            values[f"d({p})"] = self.nominal_discount(p)
        return pd.Series(values, name="rate")

    def __repr__(self) -> str:
        return f"InterestRate(effective={self.effective:.6f})"


def simple_accumulation(amount: float, rate: float, t: float) -> float:
    """Accumulated value under simple interest: ``amount * (1 + rate * t)``."""
    return amount * (1 + rate * t)


def simple_discount_value(amount: float, rate: float, t: float) -> float:
    """
    Present value under simple (commercial) discount:
    ``amount * (1 - rate * t)``. Used for treasury bills and bills of exchange.
    """
    return amount * (1 - rate * t)


def real_rate(money_rate: float, inflation_rate: float) -> float:
    """Real rate of interest from a money rate and a rate of inflation."""
    return (1 + money_rate) / (1 + inflation_rate) - 1


def money_rate(real: float, inflation_rate: float) -> float:
    """Money rate of interest from a real rate and a rate of inflation."""
    return (1 + real) * (1 + inflation_rate) - 1


def inflation_from_index(index_start: float, index_end: float, years: float) -> float:
    """Average annual rate of inflation between two values of a price index."""
    if index_start <= 0 or index_end <= 0 or years <= 0:
        raise ValueError("index values and years must be positive")
    return (index_end / index_start) ** (1 / years) - 1


class ForceOfInterest:
    """
    A force of interest that varies with time, defined piecewise by
    polynomials in t.

    Each piece is ``(upper_limit, coefficients)`` where the coefficients are
    in increasing powers of t. For example::

        delta(t) = 0.04              for 0 <= t < 6
        delta(t) = 0.2 - 0.02 t      for t >= 6

    is ``ForceOfInterest([(6, [0.04]), (None, [0.2, -0.02])])``.
    A single number gives a constant force of interest.
    """

    def __init__(self, pieces: Union[float, Sequence]):
        """
        Args:
            pieces: A constant force of interest, or a list of
                ``(upper_limit, coefficients)`` in increasing order of time.
                Use None for the upper limit of the last piece.
        """
        if isinstance(pieces, (int, float)):
            pieces = [(None, [float(pieces)])]
        self._limits = []
        self._coefficients = []
        previous = 0.0
        for k, (limit, coefficients) in enumerate(pieces):
            upper = np.inf if limit is None else float(limit)
            if upper <= previous:
                raise ValueError("upper limits must be increasing and positive")
            if np.isinf(upper) and k != len(pieces) - 1:
                raise ValueError("only the last piece may be unlimited")
            self._limits.append(upper)
            self._coefficients.append(np.atleast_1d(np.asarray(coefficients, dtype=float)))
            previous = upper
        if not np.isinf(self._limits[-1]):
            raise ValueError("the last piece must have no upper limit (use None)")

    def delta(self, t: float) -> float:
        """Force of interest at time t."""
        if t < 0:
            raise ValueError("t must not be negative")
        for limit, coefficients in zip(self._limits, self._coefficients):
            if t < limit:
                return float(np.polyval(coefficients[::-1], t))
        return float(np.polyval(self._coefficients[-1][::-1], t))

    def integral(self, t1: float, t2: float) -> float:
        """Integral of the force of interest from t1 to t2."""
        if t1 < 0 or t2 < 0:
            raise ValueError("times must not be negative")
        if t2 < t1:
            return -self.integral(t2, t1)
        total, lower = 0.0, 0.0
        for limit, coefficients in zip(self._limits, self._coefficients):
            a, b = max(t1, lower), min(t2, limit)
            if b > a:
                antiderivative = np.polyint(coefficients[::-1])
                total += float(np.polyval(antiderivative, b) - np.polyval(antiderivative, a))
            lower = limit
        return total

    def accumulation_factor(self, t1: float, t2: float) -> float:
        """A(t1, t2): accumulated value at t2 of 1 invested at t1."""
        return float(np.exp(self.integral(t1, t2)))

    def discount_factor(self, t1: float, t2: float) -> float:
        """Value at t1 of 1 due at t2."""
        return float(np.exp(-self.integral(t1, t2)))

    def accumulate(self, amount: float, t1: float, t2: float) -> float:
        """Accumulated value at t2 of an amount paid at t1."""
        return amount * self.accumulation_factor(t1, t2)

    def present_value(self, amount: float, due: float, at: float = 0.0) -> float:
        """Value at time ``at`` of an amount due at time ``due``."""
        return amount * self.discount_factor(at, due)

    def equivalent_effective_rate(self, t1: float, t2: float) -> float:
        """Constant effective annual rate giving the same accumulation from t1 to t2."""
        if t2 <= t1:
            raise ValueError("t2 must be later than t1")
        return self.accumulation_factor(t1, t2) ** (1 / (t2 - t1)) - 1

    def value_of_stream(self,
                        rate: Union[float, Callable[[float], float]],
                        start: float,
                        end: float,
                        at: float = 0.0) -> float:
        """
        Value at time ``at`` of a continuous payment stream between two times.

        Use an ``at`` before the stream for a present value and after it for
        an accumulated value.

        Args:
            rate: Rate of payment per year, a constant or a function of time
            start: Time the stream starts
            end: Time the stream ends
            at: Time at which the stream is valued
        """
        from scipy import integrate
        payment = rate if callable(rate) else (lambda t: rate)
        breaks = [x for x in self._limits if start < x < end]
        value, _ = integrate.quad(lambda t: payment(t) * np.exp(-self.integral(at, t)),
                                  start, end, points=breaks or None, limit=200)
        return float(value)
