"""
Select Mortality

A select table gives mortality that depends on the time since a life was
selected (for example, accepted for insurance after medical underwriting)
as well as on age. For the first few years, the *select period*, a recently
selected life has lighter mortality than the general body of lives of the
same age. After the select period the *ultimate* rates apply.

Notation: ``q[x]+r`` is the mortality rate at age x+r of a life selected at
age x, r years ago.
"""

from importlib import resources
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .mortality_table import MortalityTable


class SelectMortalityTable:
    """
    A select and ultimate mortality table.

    Select rates are held by **attained age** for each duration: the rate for
    duration r at attained age y is ``q[y-r]+r``.

    Use :meth:`table_for` to get an ordinary :class:`MortalityTable` for a
    life selected at a given age. Every function that takes a mortality
    table then works with select mortality.
    """

    def __init__(self,
                 ages: Sequence[int],
                 q_ultimate: Sequence[float],
                 q_select: Sequence[Sequence[float]],
                 name: str = "Select table",
                 metadata: Optional[Dict] = None):
        """
        Args:
            ages: Consecutive integer ages
            q_ultimate: Ultimate mortality rates at those ages
            q_select: One sequence per duration 0, 1, ..., each giving the
                select rate at every attained age (NaN where not tabulated).
                The number of sequences is the select period.
            name: Name of the table
            metadata: Additional information about the table. ``radix`` sets
                the number of lives at the first age of the ultimate table.
        """
        self.name = name
        self.metadata = dict(metadata) if metadata else {}
        self.ultimate = MortalityTable(ages, q_ultimate, name=f"{name} (ultimate)",
                                       metadata=self.metadata)
        self.ages = self.ultimate.ages
        self.select_period = len(q_select)
        if self.select_period < 1:
            raise ValueError("at least one duration of select rates is needed")
        self._select = []
        for duration, rates in enumerate(q_select):
            arr = np.asarray(rates, dtype=float)
            if arr.shape != self.ages.shape:
                raise ValueError(f"select rates for duration {duration} must have one value "
                                 "per age")
            known = arr[~np.isnan(arr)]
            if np.any((known < 0) | (known > 1)):
                raise ValueError("select rates must be between 0 and 1")
            self._select.append(arr)

    def q(self, selection_age: int, duration: int = 0) -> float:
        """
        Mortality rate ``q[x]+r`` of a life selected at age x, r years later.

        Args:
            selection_age: Age at selection, x
            duration: Complete years since selection, r

        Returns:
            The select rate within the select period, the ultimate rate
            after it. Where a select rate is not tabulated the ultimate rate
            is used.
        """
        if duration < 0 or duration != int(duration):
            raise ValueError("duration must be a non-negative whole number")
        attained = int(selection_age) + int(duration)
        idx = self.ultimate._index(attained)
        if duration < self.select_period:
            rate = self._select[int(duration)][idx]
            if not np.isnan(rate):
                return float(rate)
        return float(self.ultimate.qx_values[idx])

    def table_for(self, selection_age: int, duration: int = 0) -> MortalityTable:
        """
        The mortality table that applies to one life from now on.

        Args:
            selection_age: Age at which the life was selected
            duration: Years since selection (0 for a newly selected life).
                The table starts at the life's present age.

        Returns:
            MortalityTable starting at age ``selection_age + duration``, with
            select rates for what remains of the select period and ultimate
            rates afterwards
        """
        present_age = int(selection_age) + int(duration)
        start = self.ultimate._index(present_age)
        ages = self.ages[start:]
        rates = self.ultimate.qx_values[start:].copy()
        for k in range(len(ages)):
            r = duration + k
            if r >= self.select_period:
                break
            rates[k] = self.q(selection_age, r)
        label = f"[{selection_age}]" + (f"+{duration}" if duration else "")
        return MortalityTable(ages, rates, name=f"{self.name} {label}",
                              metadata={**self.metadata, "radix": self.lx(selection_age, duration)})

    def lx(self, selection_age: int, duration: int = 0) -> float:
        """
        ``l[x]+r``: the number of lives at age x+r, selected at age x, on
        the scale of the ultimate table, so that select and ultimate
        survivors agree once the select period is over.
        """
        end_age = int(selection_age) + self.select_period
        if duration >= self.select_period:
            return float(self.ultimate.lx(int(selection_age) + int(duration)))
        survivors = float(self.ultimate.lx(end_age))
        for r in range(self.select_period - 1, int(duration) - 1, -1):
            survivors /= (1 - self.q(selection_age, r))
        return survivors

    def to_dataframe(self) -> pd.DataFrame:
        """Select and ultimate rates by attained age."""
        data = {"age": self.ages}
        for duration, rates in enumerate(self._select):
            data[f"q_select_{duration}"] = rates
        data["q_ultimate"] = self.ultimate.qx_values
        return pd.DataFrame(data)

    def __repr__(self) -> str:
        return (f"SelectMortalityTable(name='{self.name}', select_period={self.select_period}, "
                f"ages=({self.ages[0]}-{self.ages[-1]}))")


def load_am92() -> SelectMortalityTable:
    """
    AM92: UK assured lives, males, 1991-94, with a two-year select period.

    The table is the standard basis of examples in UK actuarial education.
    Use ``load_am92().ultimate`` for the ultimate table and
    ``load_am92().table_for(40)`` for a life selected at 40.

    Source: Continuous Mortality Investigation, CMI Report 17 (1999). The
    rates shipped here were taken from the Society of Actuaries mortality
    table database and checked against the published graduation formula and
    against values quoted in textbooks. Rates at ages 17 and 18 of the
    ultimate table come from the graduation formula. The table belongs to
    the Continuous Mortality Investigation; it is included for education
    and testing.
    """
    path = resources.files("actuneo.mortality") / "data" / "am92" / "am92.csv"
    with resources.as_file(path) as file:
        data = pd.read_csv(file)
    return SelectMortalityTable(
        data["age"].to_numpy(),
        data["q_ultimate"].to_numpy(),
        [data["q_select_0"].to_numpy(), data["q_select_1"].to_numpy()],
        name="AM92",
        metadata={"radix": 10000.0, "country": "United Kingdom", "source": "CMI Report 17"},
    )
