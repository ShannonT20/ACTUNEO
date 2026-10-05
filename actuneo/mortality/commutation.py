"""
Commutation Functions

Classical commutation columns (Dx, Nx, Sx, Cx, Mx, Rx) built from a mortality
table and a rate of interest. They reproduce the standard textbook identities,
for example::

    äx      = Nx / Dx
    Ax      = Mx / Dx
    nEx     = D(x+n) / Dx
    äx:n    = (Nx - N(x+n)) / Dx
    A1x:n   = (Mx - M(x+n)) / Dx

The table is closed at its last tabulated age (all survivors die in that
year), so that ``Mx = Dx - d * Nx`` holds at every age.
"""

import numpy as np
import pandas as pd
from typing import Optional
from .mortality_table import MortalityTable


class CommutationFunctions:
    """Commutation columns for a mortality table at a given interest rate."""

    def __init__(self, mortality_table: MortalityTable, interest_rate: float):
        """
        Args:
            mortality_table: MortalityTable instance
            interest_rate: Annual interest rate (e.g. 0.05 for 5%)
        """
        if interest_rate <= -1:
            raise ValueError("interest_rate must be greater than -1.0.")

        self.mt = mortality_table
        self.i = float(interest_rate)
        self.v = 1 / (1 + self.i)

        ages = mortality_table.ages
        lx = mortality_table.lx_values
        dx = mortality_table.dx_values.copy()
        dx[-1] = lx[-1]  # close the table at the last age

        self.ages = ages
        self.Dx = self.v ** ages * lx
        self.Cx = self.v ** (ages + 1) * dx
        self.Nx = np.cumsum(self.Dx[::-1])[::-1]
        self.Sx = np.cumsum(self.Nx[::-1])[::-1]
        self.Mx = np.cumsum(self.Cx[::-1])[::-1]
        self.Rx = np.cumsum(self.Mx[::-1])[::-1]

    def _value(self, column: np.ndarray, age: int) -> float:
        """Column value at an age; zero at the first age beyond the table."""
        if age == self.mt.omega:
            return 0.0
        return float(column[self.mt._index(age)])

    def to_dataframe(self) -> pd.DataFrame:
        """Commutation columns as a DataFrame."""
        return pd.DataFrame(
            {
                "age": self.ages,
                "Dx": self.Dx,
                "Nx": self.Nx,
                "Sx": self.Sx,
                "Cx": self.Cx,
                "Mx": self.Mx,
                "Rx": self.Rx,
            }
        )

    def pure_endowment(self, x: int, n: int) -> float:
        """nEx = D(x+n) / Dx"""
        return self._value(self.Dx, x + n) / self._value(self.Dx, x)

    def annuity_due(self, x: int, n: Optional[int] = None) -> float:
        """äx = Nx / Dx, or äx:n = (Nx - N(x+n)) / Dx"""
        upper = 0.0 if n is None else self._value(self.Nx, x + n)
        return (self._value(self.Nx, x) - upper) / self._value(self.Dx, x)

    def assurance(self, x: int, n: Optional[int] = None) -> float:
        """Ax = Mx / Dx, or A1x:n = (Mx - M(x+n)) / Dx"""
        upper = 0.0 if n is None else self._value(self.Mx, x + n)
        return (self._value(self.Mx, x) - upper) / self._value(self.Dx, x)

    def endowment_assurance(self, x: int, n: int) -> float:
        """Ax:n = (Mx - M(x+n) + D(x+n)) / Dx"""
        return self.assurance(x, n) + self.pure_endowment(x, n)

    def increasing_assurance(self, x: int) -> float:
        """(IA)x = Rx / Dx"""
        return self._value(self.Rx, x) / self._value(self.Dx, x)

    def increasing_annuity_due(self, x: int) -> float:
        """(Iä)x = Sx / Dx"""
        return self._value(self.Sx, x) / self._value(self.Dx, x)
