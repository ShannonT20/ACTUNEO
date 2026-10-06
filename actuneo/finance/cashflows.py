"""
Cashflows, Equations of Value and Project Appraisal

A :class:`Cashflows` object holds a set of payments, at single times or as
continuous streams, and values them: present and accumulated values, the
yield that solves the equation of value (internal rate of return), payback
periods and accumulated profit.

Money received is positive and money paid out is negative. Times are in
years.
"""

import numpy as np
import pandas as pd
from typing import Optional, Sequence


class Cashflows:
    """A series of cashflows: payments at single times and level continuous streams."""

    def __init__(self, times: Sequence[float] = (), amounts: Sequence[float] = (),
                 probabilities: Optional[Sequence[float]] = None):
        """
        Args:
            times: Times of the payments, in years
            amounts: Amounts of the payments (received positive, paid negative)
            probabilities: Probability that each payment is made, for
                payments that are uncertain. Values are then expected values.
        """
        times = np.atleast_1d(np.asarray(times, dtype=float))
        amounts = np.atleast_1d(np.asarray(amounts, dtype=float))
        if times.shape != amounts.shape:
            raise ValueError("times and amounts must have the same length")
        if probabilities is not None:
            chance = np.atleast_1d(np.asarray(probabilities, dtype=float))
            if chance.shape != amounts.shape or np.any((chance < 0) | (chance > 1)):
                raise ValueError("one probability between 0 and 1 is needed for each payment")
            amounts = amounts * chance
        self._times = list(times)
        self._amounts = list(amounts)
        self._streams = []  # (start, end, rate per year)

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------
    def add(self, time: float, amount: float, probability: float = 1.0) -> 'Cashflows':
        """
        Add a single payment.

        Args:
            time: Time of the payment in years
            amount: Amount (received positive, paid negative)
            probability: Probability that the payment is made; the expected
                amount is stored
        """
        if not 0 <= probability <= 1:
            raise ValueError("probability must be between 0 and 1")
        self._times.append(float(time))
        self._amounts.append(float(amount) * probability)
        return self

    def __neg__(self) -> 'Cashflows':
        result = Cashflows(self._times, [-a for a in self._amounts])
        result._streams = [(s, e, -r) for s, e, r in self._streams]
        return result

    def __add__(self, other: 'Cashflows') -> 'Cashflows':
        result = Cashflows(self._times + other._times, self._amounts + other._amounts)
        result._streams = self._streams + other._streams
        return result

    def __sub__(self, other: 'Cashflows') -> 'Cashflows':
        return self + (-other)

    def has_unique_yield(self) -> bool:
        """
        Whether the equation of value is certain to have exactly one yield
        above -100%: true when all the outgo comes before all the income, or
        all the income before all the outgo.
        """
        events = [(t, a) for t, a in zip(self._times, self._amounts) if a != 0]
        events += [(s, r) for s, _, r in self._streams if r != 0]
        signs = [np.sign(a) for _, a in sorted(events, key=lambda x: x[0])]
        if not signs:
            return False
        changes = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
        # Streams that overlap payments of the other sign are not covered by the rule
        for start, end, rate in self._streams:
            for t, a in zip(self._times, self._amounts):
                if start < t < end and np.sign(a) != np.sign(rate) and a != 0:
                    return False
        return changes == 1

    def add_level(self, start: float, end: float, amount_per_year: float,
                  p: int = 1, timing: str = "arrears") -> 'Cashflows':
        """
        Add a level series of payments between two times.

        Args:
            start: Start of the first period
            end: End of the last period
            amount_per_year: Total paid each year
            p: Number of payments a year
            timing: "arrears" (end of each period), "advance" (start of each
                period) or "continuous"
        """
        if end <= start:
            raise ValueError("end must be later than start")
        if timing == "continuous":
            return self.add_continuous(start, end, amount_per_year)
        if timing not in ("arrears", "advance"):
            raise ValueError("timing must be 'arrears', 'advance' or 'continuous'")
        count = int(round((end - start) * p))
        if abs(count - (end - start) * p) > 1e-9:
            raise ValueError("the period must be a whole number of payment intervals")
        offset = 1 if timing == "arrears" else 0
        for k in range(count):
            self.add(start + (k + offset) / p, amount_per_year / p)
        return self

    def add_continuous(self, start: float, end: float, rate_per_year: float) -> 'Cashflows':
        """Add a level continuous payment stream between two times."""
        if end <= start:
            raise ValueError("end must be later than start")
        self._streams.append((float(start), float(end), float(rate_per_year)))
        return self

    # ------------------------------------------------------------------
    # Valuation
    # ------------------------------------------------------------------
    def to_frame(self) -> pd.DataFrame:
        """The single payments in time order."""
        frame = pd.DataFrame({"time": self._times, "amount": self._amounts})
        return frame.sort_values("time", kind="stable").reset_index(drop=True)

    def _value(self, i: float, at: float, until: float = np.inf) -> float:
        """Value at time ``at`` of the cashflows up to and including time ``until``."""
        if i <= -1:
            raise ValueError("i must be greater than -1")
        growth = 1 + i
        total = 0.0
        for t, amount in zip(self._times, self._amounts):
            if t <= until + 1e-12:
                total += amount * growth ** (at - t)
        delta = np.log1p(i)
        for start, end, rate in self._streams:
            end = min(end, until)
            if end <= start:
                continue
            if abs(delta) < 1e-14:
                total += rate * (end - start)
            else:
                total += rate * (growth ** (at - start) - growth ** (at - end)) / delta
        return float(total)

    def present_value(self, i: float, at: float = 0.0) -> float:
        """Value of all the cashflows at a point in time (default time 0)."""
        return self._value(i, at)

    def net_present_value(self, i: float) -> float:
        """Net present value at time 0."""
        return self._value(i, 0.0)

    def accumulated_value(self, i: float, at: Optional[float] = None) -> float:
        """Accumulated value at a point in time (default: the last cashflow)."""
        return self._value(i, self.end if at is None else at)

    @property
    def end(self) -> float:
        """Time of the last cashflow."""
        times = list(self._times) + [e for _, e, _ in self._streams]
        if not times:
            raise ValueError("there are no cashflows")
        return float(max(times))

    def internal_rate_of_return(self, low: float = -0.99, high: float = 10.0,
                                guess: float = 0.05) -> float:
        """
        The yield: the effective annual rate at which the net present value
        is zero.

        A set of cashflows that changes sign more than once can have more
        than one yield, or none. When there are several in the range
        searched, the one nearest to ``guess`` is returned.

        Args:
            low: Lowest rate to search
            high: Highest rate to search
            guess: Rate near which the yield is expected
        """
        from scipy import optimize
        grid = np.unique(np.concatenate((
            np.linspace(low, min(0.0, high), 60), np.linspace(max(0.0, low), min(1.0, high), 401),
            np.linspace(min(1.0, high), high, 60))))
        values = np.array([self.net_present_value(r) for r in grid])
        roots = [float(r) for r, value in zip(grid, values) if value == 0]
        for k in np.flatnonzero(np.sign(values[:-1]) * np.sign(values[1:]) < 0):
            roots.append(float(optimize.brentq(self.net_present_value, grid[k], grid[k + 1],
                                               xtol=1e-13)))
        if not roots:
            raise ValueError(
                "No yield between the search limits: the net present value does not "
                "change sign"
            )
        return min(roots, key=lambda r: abs(r - guess))

    yield_ = internal_rate_of_return

    def discounted_payback_period(self, i: float) -> Optional[float]:
        """
        The earliest time at which the cashflows to date, with interest at
        rate i, are no longer negative.

        Args:
            i: Effective annual rate at which the project is financed

        Returns:
            Time in years, or None if the project never pays back
        """
        events = sorted(set(self._times) | {s for s, _, _ in self._streams}
                        | {e for _, e, _ in self._streams})
        if not events:
            return None
        from scipy import optimize
        previous = None
        for t in events:
            # Within a continuous stream the balance can cross zero between events
            if previous is not None and self._value(i, 0.0, previous) < 0:
                just_before = self._value_before(i, t, t)
                if just_before >= 0:
                    return float(optimize.brentq(
                        lambda u: self._value_before(i, u, t), previous, t, xtol=1e-10))
            if self._value(i, 0.0, t) >= -1e-9:
                return float(t)
            previous = t
        return None

    def _value_before(self, i: float, t: float, next_event: float) -> float:
        """Value at time 0 of single payments before the next event, plus streams up to t."""
        growth = 1 + i
        total = sum(a * growth ** -s for s, a in zip(self._times, self._amounts)
                    if s < next_event - 1e-12)
        delta = np.log1p(i)
        for start, end, rate in self._streams:
            end = min(end, t)
            if end > start:
                total += (rate * (end - start) if abs(delta) < 1e-14
                          else rate * (growth ** -start - growth ** -end) / delta)
        return float(total)

    def discounted_mean_term(self, i: float) -> float:
        """Discounted mean term (duration) of the cashflows at rate i."""
        v = 1 / (1 + i)
        weighted = sum(t * a * v ** t for t, a in zip(self._times, self._amounts))
        delta = np.log1p(i)
        for start, end, rate in self._streams:
            if abs(delta) < 1e-14:
                weighted += rate * (end ** 2 - start ** 2) / 2
            else:
                weighted += rate * ((start * v ** start - end * v ** end) / delta
                                    + (v ** start - v ** end) / delta ** 2)
        return float(weighted / self._value(i, 0.0))

    def volatility(self, i: float) -> float:
        """Volatility (modified or effective duration) of the cashflows at rate i."""
        return self.discounted_mean_term(i) / (1 + i)

    def payback_period(self) -> Optional[float]:
        """The earliest time at which total income covers total outgo, ignoring interest."""
        return self.discounted_payback_period(0.0)

    def accumulated_profit(self, borrowing_rate: float, investing_rate: float,
                           at: Optional[float] = None) -> float:
        """
        Accumulated profit when the project is financed by borrowing until it
        has paid back, and surplus money is invested afterwards.

        Args:
            borrowing_rate: Effective annual rate paid on borrowings
            investing_rate: Effective annual rate earned on surplus funds
            at: Time at which the profit is measured (default: the last cashflow)

        Returns:
            Accumulated profit. If the project never pays back, the whole
            period is at the borrowing rate.
        """
        at = self.end if at is None else at
        payback = self.discounted_payback_period(borrowing_rate)
        if payback is None or payback >= at:
            return self._value(borrowing_rate, at)
        balance = self._value(borrowing_rate, payback, until=payback)
        later = Cashflows()
        for t, amount in zip(self._times, self._amounts):
            if t > payback + 1e-12:
                later.add(t, amount)
        for start, end, rate in self._streams:
            if end > payback:
                later.add_continuous(max(start, payback), end, rate)
        after = later._value(investing_rate, at) if (later._times or later._streams) else 0.0
        return float(balance * (1 + investing_rate) ** (at - payback) + after)

    def __repr__(self) -> str:
        return (f"Cashflows({len(self._times)} payment(s), "
                f"{len(self._streams)} continuous stream(s))")


def present_value(amounts: Sequence[float], times: Sequence[float], i: float,
                  at: float = 0.0) -> float:
    """Value at a point in time of payments made at the given times."""
    return Cashflows(times, amounts).present_value(i, at)


def accumulated_value(amounts: Sequence[float], times: Sequence[float], i: float,
                      at: Optional[float] = None) -> float:
    """Accumulated value of payments made at the given times."""
    return Cashflows(times, amounts).accumulated_value(i, at)


def internal_rate_of_return(amounts: Sequence[float], times: Sequence[float],
                            low: float = -0.99, high: float = 10.0,
                            guess: float = 0.05) -> float:
    """The yield on payments made at the given times."""
    return Cashflows(times, amounts).internal_rate_of_return(low, high, guess)


def crossover_rate(project_a: Cashflows, project_b: Cashflows, guess: float = 0.05) -> float:
    """
    Rate of interest at which two projects have the same net present value.

    Below and above this rate the projects rank differently, which is why
    the internal rate of return alone cannot choose between them.

    Args:
        project_a: Cashflows of the first project
        project_b: Cashflows of the second project
        guess: Rate near which the crossover is expected
    """
    return (project_a - project_b).internal_rate_of_return(guess=guess)
