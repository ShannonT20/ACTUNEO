"""
Average Cost per Claim Method

Projects claim numbers and average claim amounts separately and multiplies
the two ultimates together. Looking at frequency and severity apart can show
patterns that the total claims triangle hides.

The method is not uniquely defined. Two common forms are provided: projection
with grossing-up factors (simple averages) and with chain-ladder development
factors.
"""

import numpy as np
import pandas as pd
from typing import Optional
from .triangle import Triangle
from .chain_ladder import ChainLadder


def grossing_up(triangle: Triangle) -> pd.DataFrame:
    """
    Project a triangle to ultimate with grossing-up factors.

    A grossing-up factor is the proportion of the ultimate reached at a
    development period. The first origin period is taken as fully run off.
    Each later origin period is grossed up by the simple average of the
    factors of the earlier origin periods at its latest development period,
    which in turn fixes its own factors.

    Args:
        triangle: Triangle of cumulative amounts, numbers or averages. The
            values are used as they stand.

    Returns:
        DataFrame of grossing-up factors by origin and development period,
        with the projected ultimate in a final "ultimate" column
    """
    values = triangle.values
    n_origin, _ = values.shape
    latest_idx = triangle._latest_index()
    if latest_idx[0] != values.shape[1] - 1:
        raise ValueError("The first origin period must be observed to the last development period")

    factors = np.full(values.shape, np.nan)
    ultimate = np.full(n_origin, np.nan)
    for i in range(n_origin):
        j0 = latest_idx[i]
        if i == 0:
            ultimate[i] = values[i, j0]
        else:
            earlier = factors[:i, j0]
            earlier = earlier[~np.isnan(earlier)]
            if len(earlier) == 0:
                raise ValueError(
                    f"No earlier origin period has reached the development period of "
                    f"origin period {triangle.origin[i]}"
                )
            ultimate[i] = values[i, j0] / earlier.mean()
        factors[i] = values[i] / ultimate[i]

    table = pd.DataFrame(factors, index=pd.Index(triangle.origin, name="origin"),
                         columns=list(triangle.development))
    table["ultimate"] = ultimate
    return table


class AverageCostPerClaim:
    """
    Average cost per claim projection.

    Attributes:
        average_cost: Triangle of cumulative claims divided by cumulative numbers
        ultimate_numbers: Projected ultimate number of claims by origin period
        ultimate_average_cost: Projected ultimate average cost by origin period
        ultimate: Projected ultimate claims by origin period
        latest: Latest observed cumulative claims by origin period
        ibnr: Ultimate less latest claims by origin period
    """

    def __init__(self,
                 claims: Triangle,
                 numbers: Triangle,
                 method: str = "grossing_up"):
        """
        Fit the average cost per claim method.

        The two triangles must be on the same basis: paid claims with numbers
        settled, or incurred claims with numbers reported.

        Args:
            claims: Triangle of claim amounts (cumulative or incremental)
            numbers: Triangle of claim numbers (cumulative or incremental)
            method: "grossing_up" to project numbers and average costs with
                simple-average grossing-up factors, "chain_ladder" to project
                both with volume-weighted development factors
        """
        if not isinstance(claims, Triangle) or not isinstance(numbers, Triangle):
            raise TypeError("claims and numbers must be Triangles")
        if method not in ("grossing_up", "chain_ladder"):
            raise ValueError("method must be 'grossing_up' or 'chain_ladder'")

        claims = claims.to_cumulative()
        numbers = numbers.to_cumulative()
        if claims.shape != numbers.shape or not np.array_equal(
            np.isnan(claims.values), np.isnan(numbers.values)
        ):
            raise ValueError("claims and numbers must cover the same cells")
        if np.any(numbers.values[~np.isnan(numbers.values)] <= 0):
            raise ValueError("claim numbers must be positive in every observed cell")

        self.claims = claims
        self.numbers = numbers
        self.method = method
        self.average_cost = Triangle(claims.values / numbers.values, claims.origin,
                                     claims.development, cumulative=True,
                                     name=f"{claims.name} average cost")

        origin_index = pd.Index(claims.origin, name="origin")
        if method == "grossing_up":
            ultimate_numbers = grossing_up(numbers)["ultimate"].to_numpy()
            ultimate_average = grossing_up(self.average_cost)["ultimate"].to_numpy()
        else:
            ultimate_numbers = ChainLadder(numbers).ultimate.to_numpy()
            # The average cost table is projected as it stands, without accumulating
            ultimate_average = ChainLadder(self.average_cost).ultimate.to_numpy()

        self.ultimate_numbers = pd.Series(ultimate_numbers, index=origin_index,
                                          name="ultimate_numbers")
        self.ultimate_average_cost = pd.Series(ultimate_average, index=origin_index,
                                               name="ultimate_average_cost")
        self.ultimate = (self.ultimate_numbers * self.ultimate_average_cost).rename("ultimate")
        self.latest = claims.latest_diagonal()
        self.ibnr = (self.ultimate - self.latest).rename("ibnr")

    @property
    def total_ibnr(self) -> float:
        """Total of ultimate less latest claims over all origin periods."""
        return float(self.ibnr.sum())

    def reserve(self, paid_to_date: Optional[float] = None) -> float:
        """
        Total reserve.

        Args:
            paid_to_date: Total claims paid so far, needed when the claims
                triangle holds incurred claims. If omitted, the reserve is
                the ultimate less the latest diagonal of the claims triangle.
        """
        if paid_to_date is None:
            return self.total_ibnr
        return float(self.ultimate.sum() - paid_to_date)

    def summary(self, total: bool = True) -> pd.DataFrame:
        """
        Results by origin period.

        Args:
            total: Add a "Total" row
        """
        table = pd.DataFrame({
            "latest": self.latest,
            "ultimate_numbers": self.ultimate_numbers,
            "ultimate_average_cost": self.ultimate_average_cost,
            "ultimate": self.ultimate,
            "ibnr": self.ibnr,
        })
        if total:
            table.loc["Total"] = [
                self.latest.sum(), self.ultimate_numbers.sum(), np.nan,
                self.ultimate.sum(), self.ibnr.sum(),
            ]
        return table

    def __repr__(self) -> str:
        return (f"AverageCostPerClaim(claims='{self.claims.name}', method='{self.method}')\n"
                f"{self.summary().to_string()}")
