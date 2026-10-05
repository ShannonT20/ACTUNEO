"""
Claims Development Triangles

A :class:`Triangle` holds claims (paid, incurred or counts) by origin period
and development period, and provides the basic operations every reserving
method starts from:

- conversion between cumulative and incremental form
- conversion between wide (matrix) and long (one row per cell) layouts
- the latest diagonal
- individual link ratios (age-to-age factors) and their averages

The functionality mirrors the triangle helpers of the R ``ChainLadder``
package (``as.triangle``, ``incr2cum``, ``cum2incr``, ``ata``,
``getLatestCumulative``), see Gesmann et al., *ChainLadder: Statistical
Methods and Models for Claims Reserving in General Insurance*.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Sequence, Union


class Triangle:
    """
    A claims development triangle.

    Rows are origin periods (accident, underwriting or reporting periods) and
    columns are development periods. Cells that have not yet been observed
    are ``NaN``.
    """

    def __init__(self,
                 values: Union[np.ndarray, pd.DataFrame, Sequence[Sequence[float]]],
                 origin: Optional[Sequence] = None,
                 development: Optional[Sequence] = None,
                 cumulative: bool = True,
                 name: str = "Triangle"):
        """
        Initialize a triangle.

        Args:
            values: 2-dimensional data, origin periods in rows and development
                periods in columns, with NaN for cells not yet observed. A
                DataFrame supplies its index and columns as labels.
            origin: Labels of the origin periods (default 1, 2, ...)
            development: Labels of the development periods (default 1, 2, ...)
            cumulative: True if the values are cumulative, False if incremental
            name: Name/description of the triangle
        """
        if isinstance(values, pd.DataFrame):
            if origin is None:
                origin = values.index.tolist()
            if development is None:
                development = values.columns.tolist()
            values = values.to_numpy()

        arr = np.array(values, dtype=float)
        if arr.ndim != 2:
            raise ValueError("values must be 2-dimensional (origin x development)")
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError("values must not be empty")
        if np.all(np.isnan(arr[:, 0])):
            raise ValueError("the first development period has no observations")

        n_origin, n_dev = arr.shape
        self.origin = list(origin) if origin is not None else list(range(1, n_origin + 1))
        self.development = (
            list(development) if development is not None else list(range(1, n_dev + 1))
        )
        if len(self.origin) != n_origin:
            raise ValueError("origin must have one label per row of values")
        if len(self.development) != n_dev:
            raise ValueError("development must have one label per column of values")

        # Each row must be observed up to some development period and unknown after
        observed = ~np.isnan(arr)
        for i in range(n_origin):
            if not observed[i].any():
                raise ValueError(f"origin period {self.origin[i]} has no observations")
            last = np.flatnonzero(observed[i])[-1]
            if not observed[i, :last + 1].all():
                raise ValueError(
                    f"origin period {self.origin[i]} has missing values before its "
                    "latest observation"
                )

        self.values = arr
        self.is_cumulative = bool(cumulative)
        self.name = name

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_long(cls,
                  df: pd.DataFrame,
                  origin: str = "origin",
                  development: str = "development",
                  value: str = "value",
                  cumulative: bool = True,
                  name: str = "Triangle") -> 'Triangle':
        """
        Create a Triangle from long format data, one row per cell.

        Several rows for the same cell (for example individual claim payments)
        are added together. For incremental data, an empty cell before the
        latest cell with data in its origin period is treated as zero. An
        origin period with no movement in its most recent development periods
        therefore needs explicit zero rows for those periods.

        Args:
            df: DataFrame with origin, development and value columns
            origin: Column name of the origin period
            development: Column name of the development period
            value: Column name of the claims amount
            cumulative: True if the values are cumulative, False if incremental
            name: Name for the triangle

        Returns:
            Triangle instance
        """
        wide = df.groupby([origin, development])[value].sum(min_count=1).unstack()
        wide = wide.sort_index(axis=0).sort_index(axis=1)
        if not cumulative:
            # An incremental cell with no transactions inside the observed
            # part of the triangle is a zero, not an unknown
            arr = wide.to_numpy(dtype=float, copy=True)
            observed = ~np.isnan(arr)
            for i in range(arr.shape[0]):
                if observed[i].any():
                    last = np.flatnonzero(observed[i])[-1]
                    arr[i, :last + 1] = np.nan_to_num(arr[i, :last + 1])
            wide = pd.DataFrame(arr, index=wide.index, columns=wide.columns)
        return cls(wide, cumulative=cumulative, name=name)

    @classmethod
    def from_dataframe(cls,
                       df: pd.DataFrame,
                       cumulative: bool = True,
                       name: str = "Triangle") -> 'Triangle':
        """
        Create a Triangle from a wide DataFrame (origin index, development columns).
        """
        return cls(df, cumulative=cumulative, name=name)

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------
    @property
    def shape(self) -> tuple:
        """(number of origin periods, number of development periods)"""
        return self.values.shape

    @property
    def n_origin(self) -> int:
        return self.values.shape[0]

    @property
    def n_development(self) -> int:
        return self.values.shape[1]

    def _with_values(self, values: np.ndarray, cumulative: bool) -> 'Triangle':
        return Triangle(values, self.origin, self.development, cumulative, self.name)

    def _latest_index(self) -> np.ndarray:
        """Column position of the latest observation of each origin period."""
        observed = ~np.isnan(self.values)
        return observed.shape[1] - 1 - np.argmax(observed[:, ::-1], axis=1)

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------
    def to_cumulative(self) -> 'Triangle':
        """Cumulative triangle (R: ``incr2cum``)."""
        if self.is_cumulative:
            return self
        return self._with_values(np.cumsum(self.values, axis=1), True)

    def to_incremental(self) -> 'Triangle':
        """Incremental triangle (R: ``cum2incr``)."""
        if not self.is_cumulative:
            return self
        incremental = self.values.copy()
        incremental[:, 1:] = np.diff(self.values, axis=1)
        return self._with_values(incremental, False)

    def to_frame(self) -> pd.DataFrame:
        """Wide DataFrame with origin periods as index and development periods as columns."""
        return pd.DataFrame(
            self.values,
            index=pd.Index(self.origin, name="origin"),
            columns=pd.Index(self.development, name="development"),
        )

    def to_long(self, dropna: bool = True) -> pd.DataFrame:
        """
        Long format DataFrame, one row per cell (R: ``as.LongTriangle``).

        Args:
            dropna: Leave out cells that have not been observed
        """
        long = pd.DataFrame(
            {
                "origin": np.repeat(np.array(self.origin, dtype=object), self.n_development),
                "development": np.tile(np.array(self.development, dtype=object), self.n_origin),
                "value": self.values.ravel(),
            }
        ).infer_objects()
        if dropna:
            long = long.dropna(subset=["value"]).reset_index(drop=True)
        return long

    # ------------------------------------------------------------------
    # Diagonals
    # ------------------------------------------------------------------
    def latest_diagonal(self) -> pd.Series:
        """
        Latest observed value of each origin period (R: ``getLatestCumulative``).

        For an incremental triangle this is the latest incremental amount.
        """
        idx = self._latest_index()
        latest = self.values[np.arange(self.n_origin), idx]
        return pd.Series(latest, index=pd.Index(self.origin, name="origin"), name="latest")

    def latest_development(self) -> pd.Series:
        """Development period reached by each origin period."""
        idx = self._latest_index()
        return pd.Series([self.development[j] for j in idx],
                         index=pd.Index(self.origin, name="origin"), name="development")

    # ------------------------------------------------------------------
    # Link ratios
    # ------------------------------------------------------------------
    def _link_labels(self) -> List[str]:
        dev = self.development
        return [f"{dev[k]}-{dev[k + 1]}" for k in range(len(dev) - 1)]

    def link_ratios(self) -> pd.DataFrame:
        """
        Individual link ratios (age-to-age factors) C[i, k+1] / C[i, k].

        Returns:
            DataFrame of link ratios, origin periods in rows
        """
        cum = self.to_cumulative().values
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = cum[:, 1:] / cum[:, :-1]
        ratios[~np.isfinite(ratios)] = np.nan
        return pd.DataFrame(ratios,
                            index=pd.Index(self.origin, name="origin"),
                            columns=self._link_labels())

    def development_factors(self,
                            average: str = "volume",
                            n_periods: Optional[int] = None) -> pd.Series:
        """
        Average link ratio for each development step.

        Args:
            average: "volume" for the volume-weighted average (the chain-ladder
                factor, sum of C[i, k+1] over sum of C[i, k]), "simple" for
                the arithmetic mean of the individual link ratios
            n_periods: Use only the latest n origin periods available for
                each development step (None for all)

        Returns:
            Series of development factors
        """
        if average not in ("volume", "simple"):
            raise ValueError("average must be 'volume' or 'simple'")
        if n_periods is not None and n_periods < 1:
            raise ValueError("n_periods must be at least 1")

        cum = self.to_cumulative().values
        n_steps = cum.shape[1] - 1
        factors = np.full(n_steps, np.nan)
        for k in range(n_steps):
            rows = np.flatnonzero(~np.isnan(cum[:, k]) & ~np.isnan(cum[:, k + 1]))
            if n_periods is not None:
                rows = rows[-n_periods:]
            if len(rows) == 0:
                continue
            start, end = cum[rows, k], cum[rows, k + 1]
            if average == "volume":
                if start.sum() != 0:
                    factors[k] = end.sum() / start.sum()
            else:
                usable = start != 0
                if usable.any():
                    factors[k] = np.mean(end[usable] / start[usable])
        return pd.Series(factors, index=self._link_labels(), name=f"{average}")

    def age_to_age(self) -> pd.DataFrame:
        """
        Link ratio table with simple and volume-weighted averages (R: ``ata``).

        Returns:
            DataFrame of individual link ratios with two summary rows,
            "simple" and "volume"
        """
        table = self.link_ratios()
        table.loc["simple"] = self.development_factors("simple")
        table.loc["volume"] = self.development_factors("volume")
        return table

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        kind = "cumulative" if self.is_cumulative else "incremental"
        return (f"Triangle(name='{self.name}', {kind}, "
                f"{self.n_origin} origin x {self.n_development} development)\n"
                f"{self.to_frame().to_string(na_rep='')}")
