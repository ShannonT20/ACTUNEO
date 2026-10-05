"""
Inflation-Adjusted Chain-Ladder

The basic chain-ladder assumes that past claims inflation continues unchanged.
The inflation-adjusted method makes inflation explicit: past payments are
restated in the money of the latest calendar period, the chain-ladder is
applied in real terms, and the projected payments are inflated again at an
assumed future rate.
"""

import numpy as np
import pandas as pd
from typing import Optional, Sequence, Union
from .triangle import Triangle
from .chain_ladder import ChainLadder


class InflationAdjustedChainLadder(ChainLadder):
    """
    Chain-ladder with explicit allowance for past and future claims inflation.

    Calendar periods run along the diagonals of the triangle, so origin and
    development periods must have the same length (for example accident
    years and development years). Payments are treated as made mid-period.

    Attributes:
        index: Inflation index for each past calendar period, 1 at the latest
        real_triangle: Cumulative triangle in the money of the latest period
        real_full_triangle: Projection in the money of the latest period
        full_triangle: Projection in money of the period of payment
        factors: Development factors of the real-terms triangle
    """

    def __init__(self,
                 triangle: Triangle,
                 past_inflation: Union[float, Sequence[float]],
                 future_inflation: Union[float, Sequence[float]] = 0.0,
                 average: str = "volume",
                 n_periods: Optional[int] = None):
        """
        Fit the inflation-adjusted chain-ladder.

        Args:
            triangle: Claims triangle (cumulative or incremental)
            past_inflation: Claims inflation from each calendar period to the
                next, oldest first: one rate per calendar period after the
                first (four rates for a triangle covering five years). A
                single number applies the same rate throughout.
            future_inflation: Assumed inflation for each future calendar
                period, nearest first, or a single rate for all of them
            average: "volume" or "simple" averaging of the link ratios
            n_periods: Average only the latest n origin periods (None for all)
        """
        if not isinstance(triangle, Triangle):
            raise TypeError("triangle must be a Triangle")

        nominal = triangle.to_cumulative()
        incremental = nominal.to_incremental().values
        n_origin, n_dev = incremental.shape
        calendar = np.add.outer(np.arange(n_origin), np.arange(n_dev))
        observed = ~np.isnan(incremental)
        latest_calendar = int(calendar[observed].max())

        latest_idx = nominal._latest_index()
        if np.any(np.arange(n_origin) + latest_idx != latest_calendar):
            raise ValueError(
                "The triangle must be observed up to the same calendar period for every "
                "origin period"
            )

        past = self._rates(past_inflation, latest_calendar, "past_inflation")
        n_future = int(calendar.max()) - latest_calendar
        future = self._rates(future_inflation, n_future, "future_inflation")

        # Index relative to the latest calendar period
        growth = np.concatenate(([1.0], np.cumprod(1 + past)))
        index = growth / growth[-1]
        future_index = np.concatenate(([1.0], np.cumprod(1 + future)))

        real_incremental = incremental.copy()
        real_incremental[observed] = incremental[observed] / index[calendar[observed]]
        real = Triangle(np.cumsum(real_incremental, axis=1), nominal.origin,
                        nominal.development, cumulative=True, name=nominal.name)

        super().__init__(real, average=average, n_periods=n_periods)

        self.real_triangle = real
        self.real_full_triangle = self.full_triangle
        self.index = pd.Series(index, index=pd.RangeIndex(latest_calendar + 1, name="calendar"),
                               name="index")
        self.past_inflation = past
        self.future_inflation = future

        # Inflate the projected real payments to the period in which they fall
        real_full_incremental = self._full.copy()
        real_full_incremental[:, 1:] = np.diff(self._full, axis=1)
        full_incremental = incremental.copy()
        ahead = np.clip(calendar - latest_calendar, 0, None)
        full_incremental[~observed] = (real_full_incremental[~observed]
                                       * future_index[ahead[~observed]])
        full = np.cumsum(full_incremental, axis=1)

        origin_index = pd.Index(nominal.origin, name="origin")
        self.triangle = nominal
        self._full = full
        self.full_triangle = pd.DataFrame(
            full, index=origin_index, columns=pd.Index(nominal.development, name="development")
        )
        self.latest = nominal.latest_diagonal()
        self.ultimate = pd.Series(full[:, -1], index=origin_index, name="ultimate")
        self.ibnr = (self.ultimate - self.latest).rename("ibnr")

    @staticmethod
    def _rates(rates, length: int, name: str) -> np.ndarray:
        arr = np.atleast_1d(np.asarray(rates, dtype=float))
        if arr.size == 1:
            arr = np.full(length, arr[0])
        if arr.size != length:
            raise ValueError(f"{name} needs {length} rates (or a single rate), got {arr.size}")
        if np.any(arr <= -1):
            raise ValueError(f"{name} rates must be greater than -1")
        return arr

    def fitted_triangle(self) -> pd.DataFrame:
        raise NotImplementedError(
            "Model checks apply to the real-terms triangle: use ChainLadder(model.real_triangle)"
        )

    def fit_errors(self) -> pd.DataFrame:
        raise NotImplementedError(
            "Model checks apply to the real-terms triangle: use ChainLadder(model.real_triangle)"
        )
