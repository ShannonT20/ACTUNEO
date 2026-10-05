"""
Bornhuetter-Ferguson Method

Combines an initial estimate of ultimate claims, usually premium times an
expected loss ratio, with the development pattern of the triangle. Claims
already reported are taken as they are; only the part still to emerge is
based on the initial estimate.

Reference
---------
Bornhuetter, R.L. and Ferguson, R.E. (1972). The actuary and IBNR.
Proceedings of the Casualty Actuarial Society 59, 181-195.
"""

import numpy as np
import pandas as pd
from typing import Optional, Sequence, Union
from .triangle import Triangle
from .chain_ladder import ChainLadder


class BornhuetterFerguson(ChainLadder):
    """
    Bornhuetter-Ferguson projection of a claims triangle.

    For each origin period, with ``f`` the chain-ladder factor from its
    latest development period to ultimate::

        emerging claims = initial ultimate * (1 - 1/f)
        ultimate claims = latest claims + emerging claims

    Attributes:
        initial_ultimate: Initial estimate of ultimate claims by origin period
        emerging: Claims still expected to emerge by origin period (the IBNR)
        chain_ladder_ultimate: Ultimate claims from the chain-ladder alone
    """

    def __init__(self,
                 triangle: Triangle,
                 premium: Optional[Sequence[float]] = None,
                 loss_ratio: Optional[Union[float, Sequence[float]]] = None,
                 initial_ultimate: Optional[Sequence[float]] = None,
                 average: str = "volume",
                 n_periods: Optional[int] = None,
                 tail: Union[float, bool] = 1.0,
                 factors: Optional[Sequence[float]] = None):
        """
        Fit the Bornhuetter-Ferguson method.

        Give either ``premium`` and ``loss_ratio``, or ``initial_ultimate``.

        Args:
            triangle: Claims triangle (cumulative or incremental), paid or incurred
            premium: Earned premium for each origin period
            loss_ratio: Expected ultimate loss ratio, one figure for all
                origin periods or one for each
            initial_ultimate: Initial estimate of ultimate claims for each
                origin period, from any source
            average: "volume" or "simple" averaging of the link ratios
            n_periods: Average only the latest n origin periods (None for all)
            tail: Tail factor, or True to estimate it
            factors: Selected development factors to use instead of those
                estimated from the triangle
        """
        super().__init__(triangle, average=average, n_periods=n_periods, tail=tail,
                         factors=factors)
        n_origin = self.triangle.n_origin

        if initial_ultimate is not None:
            if premium is not None or loss_ratio is not None:
                raise ValueError("Give either initial_ultimate, or premium and loss_ratio")
            initial = np.asarray(initial_ultimate, dtype=float)
        else:
            if premium is None or loss_ratio is None:
                raise ValueError("Give either initial_ultimate, or premium and loss_ratio")
            initial = np.asarray(premium, dtype=float) * np.asarray(loss_ratio, dtype=float)
        if initial.shape != (n_origin,):
            raise ValueError(f"One value is needed for each of the {n_origin} origin periods")

        origin_index = self.latest.index
        cdf = self.cdf.to_numpy()
        proportion_developed = 1 / cdf

        self.chain_ladder_ultimate = self.ultimate.rename("chain_ladder_ultimate")
        self.initial_ultimate = pd.Series(initial, index=origin_index, name="initial_ultimate")

        # Expected cumulative development of the initial estimate beyond the latest diagonal
        full = self.triangle.values.copy()
        for i, j0 in enumerate(self._latest_idx):
            future = np.arange(j0 + 1, full.shape[1])
            full[i, future] = full[i, j0] + initial[i] * (
                proportion_developed[future] - proportion_developed[j0]
            )
        self._full = full
        self.full_triangle = pd.DataFrame(full, index=origin_index,
                                          columns=self.full_triangle.columns)

        emerging = initial * (1 - proportion_developed[self._latest_idx])
        self.emerging = pd.Series(emerging, index=origin_index, name="emerging")
        self.ultimate = (self.latest + self.emerging).rename("ultimate")
        self.ibnr = self.emerging.rename("ibnr")

    def summary(self, total: bool = True) -> pd.DataFrame:
        """
        Results by origin period.

        Args:
            total: Add a "Total" row

        Returns:
            DataFrame with latest claims, the initial estimate, the factor to
            ultimate, emerging claims (IBNR) and ultimate claims
        """
        table = pd.DataFrame({
            "latest": self.latest,
            "initial_ultimate": self.initial_ultimate,
            "cdf": self.cdf.to_numpy()[self._latest_idx],
            "ibnr": self.ibnr,
            "ultimate": self.ultimate,
        })
        if total:
            table.loc["Total"] = [
                self.latest.sum(), self.initial_ultimate.sum(), np.nan,
                self.ibnr.sum(), self.ultimate.sum(),
            ]
        return table
