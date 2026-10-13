"""
Pension annuity factors

The value of a pension of 1 a year, allowing for the features that pension
scheme rules usually carry: payment monthly (or at another frequency),
increases in payment, a guarantee period, and a pension to the spouse after
the member's death.

Survival to a fractional age is found by linear interpolation of ``lx``
(deaths uniform over each year of age), so the factor can be taken at any
age, not only at whole ages.
"""

import math
import warnings
from typing import Optional

import numpy as np

from ..mortality import MortalityTable

#: Proportion of lives surviving beyond the end of a table above which closing
#: the table is reported with a warning
CUT_OFF_WARNING = 0.001


def survival_probabilities(table: MortalityTable, age: float, times) -> np.ndarray:
    """
    Probability that a life aged ``age`` survives each of the given periods.

    Args:
        table: Mortality table
        age: Present age, which may be fractional
        times: Periods in years, which may be fractional

    Returns:
        Array of survival probabilities. A table that stops while ``qx`` is
        still below 1 is closed by assuming that nobody survives the last
        year of age. A warning is given when that removes more than 1 in
        1,000 of the lives.
    """
    times = np.atleast_1d(np.asarray(times, dtype=float))
    if np.any(times < 0):
        raise ValueError("times must be non-negative")
    if age < table.min_age or age >= table.max_age + 1:
        raise ValueError(
            f"age {age} is outside the ages of '{table.name}' "
            f"({table.min_age} to {table.max_age})"
        )
    grid = np.append(table.ages.astype(float), table.max_age + 1.0)
    lives = np.append(table.lx_values, 0.0)
    start = np.interp(age, grid, lives)
    if start <= 0:
        return np.zeros_like(times)
    cut_off = table.lx_values[-1] * table.px_values[-1] / start
    if cut_off > CUT_OFF_WARNING:
        warnings.warn(
            f"Mortality table '{table.name}' ends at age {table.max_age} with qx below 1. "
            f"It has been closed by assuming that nobody survives that year of age, which "
            f"removes {cut_off:.1%} of the lives now aged {age:g}.",
            stacklevel=3,
        )
    return np.interp(age + times, grid, lives, right=0.0) / start


def pension_annuity(table: MortalityTable,
                    age: float,
                    interest: float,
                    *,
                    increase: float = 0.0,
                    frequency: int = 12,
                    timing: str = "advance",
                    guarantee: float = 0.0,
                    spouse_fraction: float = 0.0,
                    spouse_age: Optional[float] = None,
                    spouse_table: Optional[MortalityTable] = None,
                    proportion_married: float = 1.0) -> float:
    """
    Value of a pension of 1 a year payable for life.

    Args:
        table: Mortality table of the pensioner
        age: Age of the pensioner, which may be fractional
        interest: Effective annual rate of interest
        increase: Rate at which the pension increases on each anniversary of
            its start
        frequency: Number of payments a year
        timing: "advance" (first payment now) or "arrears" (first payment at
            the end of the first period)
        guarantee: Years for which the pension is paid whether or not the
            pensioner is alive
        spouse_fraction: Pension paid to the spouse after the pensioner's
            death, as a fraction of the pensioner's pension. During the
            guarantee period the full pension continues and the spouse's
            pension starts when the guarantee ends.
        spouse_age: Age of the spouse (required when ``spouse_fraction`` is
            not zero)
        spouse_table: Mortality table of the spouse (the pensioner's table if
            omitted)
        proportion_married: Probability that there is a spouse. The lives
            are assumed independent.

    Returns:
        The annuity factor: multiply by the annual pension to value it
    """
    if interest <= -1:
        raise ValueError("interest must be greater than -1")
    if increase <= -1:
        raise ValueError("increase must be greater than -1")
    if isinstance(frequency, bool) or frequency != int(frequency) or frequency < 1:
        raise ValueError("frequency must be a positive whole number")
    if timing not in ("advance", "arrears"):
        raise ValueError("timing must be 'advance' or 'arrears'")
    if guarantee < 0:
        raise ValueError("guarantee must be non-negative")
    if not 0 <= spouse_fraction:
        raise ValueError("spouse_fraction must be non-negative")
    if not 0 <= proportion_married <= 1:
        raise ValueError("proportion_married must be between 0 and 1")
    with_spouse = spouse_fraction > 0 and proportion_married > 0
    if with_spouse and spouse_age is None:
        raise ValueError("spouse_age is needed to value a spouse's pension")

    m = int(frequency)
    spouse_table = spouse_table or table
    horizon = max(table.max_age + 1 - age, guarantee)
    if with_spouse:
        horizon = max(horizon, spouse_table.max_age + 1 - spouse_age)
    payments = np.arange(int(math.ceil(horizon * m - 1e-9)))
    times = payments / m if timing == "advance" else (payments + 1) / m

    amount = (1.0 + increase) ** (payments // m) / m
    discount = (1.0 + interest) ** -times
    alive = survival_probabilities(table, age, times)
    if timing == "advance":
        certain = times < guarantee - 1e-9
    else:
        certain = times <= guarantee + 1e-9
    paid = np.where(certain, 1.0, alive)
    if with_spouse:
        spouse_alive = survival_probabilities(spouse_table, spouse_age, times)
        paid = paid + np.where(
            certain, 0.0, spouse_fraction * proportion_married * spouse_alive * (1.0 - alive)
        )
    return float(np.sum(amount * discount * paid))
