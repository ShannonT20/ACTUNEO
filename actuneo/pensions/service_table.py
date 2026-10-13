"""
Service table and salary scale

The service table follows the active members of a pension scheme from year to
year. A member can leave by withdrawal (leaving the employer), death, ill-health
retirement or age retirement; whoever is still in service at the normal
retirement age retires then.

The salary scale projects a member's salary to the date of leaving.
"""

from typing import Callable, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..life.decrements import MultipleDecrementTable
from ..mortality import MortalityTable

CAUSES = ("withdrawal", "death", "ill_health", "retirement")

Rates = Union[None, float, Sequence[float], Mapping[int, float], Callable[[int], float]]


def _rates_by_age(rates: Rates, ages: np.ndarray, name: str) -> np.ndarray:
    """One rate per age from a number, a sequence, a mapping or a function."""
    if rates is None:
        return np.zeros(len(ages))
    if callable(rates):
        values = np.array([float(rates(int(age))) for age in ages])
    elif isinstance(rates, Mapping):
        values = np.array([float(rates.get(int(age), 0.0)) for age in ages])
    elif np.ndim(rates) == 0:
        values = np.full(len(ages), float(rates))
    else:
        values = np.asarray(rates, dtype=float)
    if values.shape != ages.shape:
        raise ValueError(f"'{name}' needs one rate for each of the {len(ages)} ages")
    if np.any(np.isnan(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError(f"'{name}' rates must be between 0 and 1")
    return values


class ServiceTable:
    """
    A multiple decrement table for the active members of a pension scheme.

    The table holds, for each age below the normal retirement age, the
    dependent probability of leaving service during the year by each cause.
    Members still in service at the normal retirement age retire on that
    birthday.
    """

    def __init__(self,
                 ages: Sequence[int],
                 retirement_age: Optional[int] = None,
                 *,
                 withdrawal: Rates = None,
                 death: Rates = None,
                 ill_health: Rates = None,
                 retirement: Rates = None):
        """
        Args:
            ages: Consecutive integer ages, ending one year below the normal
                retirement age
            retirement_age: Normal retirement age (the age after the last
                one given, if omitted)
            withdrawal: Dependent probability of leaving the employer at each
                age: a single number, one per age, a mapping from age, or a
                function of age
            death: Dependent probability of dying in service
            ill_health: Dependent probability of retiring through ill health
            retirement: Dependent probability of retiring early, before the
                normal retirement age
        """
        self.ages = np.asarray(ages, dtype=int)
        if self.ages.ndim != 1 or len(self.ages) == 0:
            raise ValueError("ages must be a non-empty sequence")
        if np.any(np.diff(self.ages) != 1):
            raise ValueError("ages must be consecutive integers")
        last = int(self.ages[-1]) + 1
        self.retirement_age = last if retirement_age is None else int(retirement_age)
        if self.retirement_age != last:
            raise ValueError(
                "ages must end one year below the normal retirement age "
                f"(last age {last - 1}, retirement age {self.retirement_age})"
            )
        supplied = {"withdrawal": withdrawal, "death": death,
                    "ill_health": ill_health, "retirement": retirement}
        rates = {cause: _rates_by_age(supplied[cause], self.ages, cause) for cause in CAUSES}
        self.table = MultipleDecrementTable(self.ages, rates, radix=1.0)

    @classmethod
    def from_independent_rates(cls,
                               ages: Sequence[int],
                               retirement_age: Optional[int] = None,
                               *,
                               withdrawal: Rates = None,
                               death: Rates = None,
                               ill_health: Rates = None,
                               retirement: Rates = None) -> 'ServiceTable':
        """
        Build the table from independent rates (each cause as if it acted
        alone), assuming constant forces of decrement over each year of age.
        """
        ages_arr = np.asarray(ages, dtype=int)
        supplied = {"withdrawal": withdrawal, "death": death,
                    "ill_health": ill_health, "retirement": retirement}
        independent = {c: _rates_by_age(supplied[c], ages_arr, c) for c in CAUSES}
        dependent = MultipleDecrementTable.from_independent_rates(ages_arr, independent, 1.0)
        return cls(ages_arr, retirement_age, **dependent.rates)

    @classmethod
    def from_mortality(cls,
                       table: MortalityTable,
                       retirement_age: int,
                       start_age: Optional[int] = None,
                       *,
                       withdrawal: Rates = None,
                       ill_health: Rates = None,
                       retirement: Rates = None) -> 'ServiceTable':
        """
        Build the table from a mortality table for deaths in service, with
        the other causes given as independent rates.

        Args:
            table: Mortality of active members
            retirement_age: Normal retirement age
            start_age: Youngest age (the youngest in the mortality table if
                omitted)
            withdrawal: Independent rates of withdrawal
            ill_health: Independent rates of ill-health retirement
            retirement: Independent rates of early retirement
        """
        first = table.min_age if start_age is None else int(start_age)
        ages = np.arange(first, int(retirement_age))
        if len(ages) == 0:
            raise ValueError("start_age must be below retirement_age")
        if first < table.min_age or retirement_age - 1 > table.max_age:
            raise ValueError(
                f"'{table.name}' covers ages {table.min_age} to {table.max_age}, "
                f"not {first} to {retirement_age - 1}"
            )
        return cls.from_independent_rates(
            ages, retirement_age, withdrawal=withdrawal, death=table.qx(ages),
            ill_health=ill_health, retirement=retirement,
        )

    @classmethod
    def no_decrements(cls, start_age: int, retirement_age: int) -> 'ServiceTable':
        """A table in which every member stays to the normal retirement age."""
        return cls(np.arange(int(start_age), int(retirement_age)), retirement_age)

    @property
    def min_age(self) -> int:
        """Youngest age in the table."""
        return int(self.ages[0])

    def rate(self, age: int, cause: str) -> float:
        """Dependent probability of leaving by a cause during the year of age."""
        if cause not in CAUSES:
            raise ValueError(f"cause must be one of {CAUSES}")
        return self.table.dependent_rate(age, cause)

    def total_rate(self, age: int) -> float:
        """Dependent probability of leaving by any cause during the year of age."""
        return float(self.table.total_rate[self.table._index(age)])

    def survival(self, age: int, years: int) -> float:
        """Probability that a member aged ``age`` is still in service after ``years``."""
        if years == 0:
            return 1.0
        return self.table.survival_probability(age, years)

    def to_dataframe(self) -> pd.DataFrame:
        """Members in service at each age and the numbers leaving by each cause."""
        frame = self.table.to_dataframe()
        return frame.rename(columns={"al": "in_service",
                                     **{f"ad_{c}": c for c in CAUSES}})

    def __repr__(self) -> str:
        return (f"ServiceTable(ages={self.min_age}-{self.retirement_age - 1}, "
                f"retirement_age={self.retirement_age})")


class SalaryScale:
    """
    Projected salaries.

    Salaries are reviewed once a year, at the valuation date and on each
    anniversary. The salary rate in the year that starts ``t`` years from now
    is the present rate times ``(1 + general_increase) ** t`` and the ratio of
    the promotional scale at the two ages.
    """

    def __init__(self, general_increase: float = 0.0,
                 promotional: Optional[Mapping[int, float]] = None):
        """
        Args:
            general_increase: Annual rate of general salary increases
            promotional: Promotional (merit) scale: an index by age. Ages
                beyond either end of the scale take the nearest value.
        """
        if general_increase <= -1:
            raise ValueError("general_increase must be greater than -1")
        self.general_increase = float(general_increase)
        self.promotional = None
        if promotional is not None:
            series = pd.Series(promotional, dtype=float).sort_index()
            if len(series) == 0 or np.any(series.to_numpy() <= 0):
                raise ValueError("the promotional scale must hold positive values")
            self.promotional = series

    def _promotion(self, age: float) -> float:
        if self.promotional is None:
            return 1.0
        return float(np.interp(age, self.promotional.index.to_numpy(dtype=float),
                               self.promotional.to_numpy()))

    def index(self, age: int, year: int) -> float:
        """
        Salary rate in the year starting ``year`` years from now, relative to
        the present rate, for a member now aged ``age``. ``year`` may be
        negative, for past years.
        """
        growth = (1.0 + self.general_increase) ** year
        return growth * self._promotion(age + year) / self._promotion(age)

    def projected(self, salary: float, age: int, year: int) -> float:
        """Salary rate in the year starting ``year`` years from now."""
        return salary * self.index(age, year)

    def final_average(self, salary: float, age: int, exit_time: float,
                      years: float = 1.0) -> float:
        """
        Average salary over the ``years`` before leaving at ``exit_time``.

        Args:
            salary: Present salary rate
            age: Present age
            exit_time: Years from now to the date of leaving
            years: Averaging period. Zero gives the salary rate just before
                leaving.

        Returns:
            The exact average of the salary rate over the period
        """
        if years < 0:
            raise ValueError("years must be non-negative")
        if years == 0:
            return self.projected(salary, age, int(np.ceil(exit_time - 1e-12)) - 1)
        start = exit_time - years
        total = 0.0
        for year in range(int(np.floor(start + 1e-12)), int(np.ceil(exit_time - 1e-12))):
            overlap = min(exit_time, year + 1) - max(start, year)
            if overlap > 0:
                total += overlap * self.index(age, year)
        return salary * total / years

    def shifted(self, change: float) -> 'SalaryScale':
        """A scale with the general increase changed by an amount."""
        return SalaryScale(self.general_increase + change, self.promotional)

    def __repr__(self) -> str:
        promo = "" if self.promotional is None else ", with promotional scale"
        return f"SalaryScale(general_increase={self.general_increase:.4f}{promo})"
