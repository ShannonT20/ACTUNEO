"""
Annuities Certain

Present and accumulated values of annuities whose payments are certain to be
made: level, payable several times a year or continuously, deferred,
perpetual, and increasing or decreasing.

All functions value an annuity of **1 per year**. Multiply by the annual
amount. ``i`` is the effective annual rate of interest.

Timing of payments:

- ``"arrears"``     at the end of each period (an immediate annuity)
- ``"advance"``     at the start of each period (an annuity-due)
- ``"continuous"``  continuously through the year
"""

import numpy as np

_TIMINGS = ("arrears", "advance", "continuous")


def _rates(i: float):
    if i <= -1:
        raise ValueError("i must be greater than -1")
    return 1 / (1 + i), float(np.log1p(i))


def _denominator(i: float, p: int, timing: str) -> float:
    """i(p), d(p) or delta: the rate that converts 1 - v^n into an annuity value."""
    if timing not in _TIMINGS:
        raise ValueError(f"timing must be one of {_TIMINGS}")
    if p < 1 or p != int(p):
        raise ValueError("p must be a positive whole number")
    if timing == "continuous":
        return float(np.log1p(i))
    if timing == "arrears":
        return p * ((1 + i) ** (1 / p) - 1)
    return p * (1 - (1 + i) ** (-1 / p))


def annuity(n: float, i: float, p: int = 1, timing: str = "arrears",
            deferred: float = 0.0) -> float:
    """
    Present value of a level annuity of 1 per year for n years.

    Covers a(n), ä(n), a(p)(n), ä(p)(n) and the continuous annuity, with an
    optional deferred period.

    Args:
        n: Term in years
        i: Effective annual rate of interest
        p: Number of payments a year (each of 1/p)
        timing: "arrears", "advance" or "continuous"
        deferred: Years before the annuity starts

    Returns:
        Present value
    """
    if n < 0 or deferred < 0:
        raise ValueError("n and deferred must not be negative")
    v, _ = _rates(i)
    if i == 0:
        return float(n)
    return v ** deferred * (1 - v ** n) / _denominator(i, p, timing)


def accumulated_annuity(n: float, i: float, p: int = 1, timing: str = "arrears") -> float:
    """
    Accumulated value at the end of n years of a level annuity of 1 per year:
    s(n), and its advance, p-thly and continuous forms.

    Args:
        n: Term in years
        i: Effective annual rate of interest
        p: Number of payments a year
        timing: "arrears", "advance" or "continuous"
    """
    return annuity(n, i, p, timing) * (1 + i) ** n


def perpetuity(i: float, p: int = 1, timing: str = "arrears", deferred: float = 0.0) -> float:
    """
    Present value of a level payment of 1 per year for ever.

    Args:
        i: Effective annual rate of interest (must be positive)
        p: Number of payments a year
        timing: "arrears", "advance" or "continuous"
        deferred: Years before the payments start
    """
    if i <= 0:
        raise ValueError("i must be positive for a perpetuity")
    return (1 + i) ** -deferred / _denominator(i, p, timing)


def increasing_annuity(n: int, i: float, timing: str = "arrears",
                       deferred: float = 0.0) -> float:
    """
    Present value of an annuity increasing by 1 each year: payments of
    1, 2, ..., n. Covers (Ia)(n), (Iä)(n) and, for "continuous", payments
    made continuously at a rate of k per year during year k.

    Args:
        n: Term in whole years
        i: Effective annual rate of interest
        timing: "arrears", "advance" or "continuous"
        deferred: Years before the annuity starts
    """
    if n != int(n) or n < 0:
        raise ValueError("n must be a non-negative whole number")
    v, _ = _rates(i)
    if i == 0:
        return n * (n + 1) / 2
    value = (annuity(n, i, timing="advance") - n * v ** n) / _denominator(i, 1, timing)
    return v ** deferred * value


def continuously_increasing_annuity(n: float, i: float) -> float:
    """
    Present value of a continuous payment stream whose rate of payment at
    time t is t, for n years.
    """
    v, delta = _rates(i)
    if i == 0:
        return n ** 2 / 2
    return (annuity(n, i, timing="continuous") - n * v ** n) / delta


def decreasing_annuity(n: int, i: float, timing: str = "arrears",
                       deferred: float = 0.0) -> float:
    """
    Present value of an annuity decreasing by 1 each year: payments of
    n, n-1, ..., 1.

    Args:
        n: Term in whole years
        i: Effective annual rate of interest
        timing: "arrears", "advance" or "continuous"
    """
    if n != int(n) or n < 0:
        raise ValueError("n must be a non-negative whole number")
    if i == 0:
        return n * (n + 1) / 2
    return (1 + i) ** -deferred * (n - annuity(n, i)) / _denominator(i, 1, timing)


def accumulated_increasing_annuity(n: int, i: float, timing: str = "arrears") -> float:
    """Accumulated value at the end of n years of an increasing annuity."""
    return increasing_annuity(n, i, timing) * (1 + i) ** n


def geometric_annuity(n: int, i: float, growth: float, timing: str = "arrears") -> float:
    """
    Present value of annual payments that grow at a compound rate: 1, then
    (1 + growth), (1 + growth)**2, and so on, for n payments.

    Args:
        n: Number of annual payments
        i: Effective annual rate of interest
        growth: Compound rate of increase per year (negative for a decrease)
        timing: "arrears" or "advance"
    """
    if timing not in ("arrears", "advance"):
        raise ValueError("timing must be 'arrears' or 'advance'")
    if growth <= -1:
        raise ValueError("growth must be greater than -1")
    net = (1 + i) / (1 + growth) - 1
    in_advance = annuity(n, net, timing="advance")
    return in_advance if timing == "advance" else in_advance / (1 + i)


def stepped_annuity(years_per_step: float, n_steps: int, i: float, growth: float,
                    p: int = 1, timing: str = "arrears") -> float:
    """
    Present value of an annuity that is level for a fixed number of years and
    then steps up, such as rent that is reviewed every five years.

    The annuity starts at 1 per year. At each review it becomes
    ``(1 + growth) ** years_per_step`` times its previous level, so ``growth``
    is the annual rate of increase that the reviews catch up with.

    Args:
        years_per_step: Years between reviews
        n_steps: Number of level periods
        i: Effective annual rate of interest
        growth: Annual rate of growth reflected at each review
        p: Number of payments a year
        timing: "arrears", "advance" or "continuous"
    """
    if n_steps < 1 or n_steps != int(n_steps):
        raise ValueError("n_steps must be a positive whole number")
    one_step = annuity(years_per_step, i, p, timing)
    ratio = ((1 + growth) / (1 + i)) ** years_per_step
    if abs(ratio - 1) < 1e-14:
        return one_step * n_steps
    return one_step * (1 - ratio ** n_steps) / (1 - ratio)
