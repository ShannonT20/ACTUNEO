"""
Mortality Table Implementation

Provides classes and methods for working with mortality tables,
including loading, manipulation, and basic calculations.

Notation follows the International Actuarial Notation:

- ``qx``  probability that a life aged x dies before age x+1
- ``px``  probability that a life aged x survives to age x+1
- ``lx``  expected number of survivors at exact age x out of the radix
- ``dx``  expected number of deaths between ages x and x+1
- ``Lx``  expected years lived between ages x and x+1
- ``Tx``  expected years lived after age x (within the tabulated range)
- ``ex``  expectation of life at age x
"""

import numpy as np
import pandas as pd
import os
from importlib import resources
from typing import Union, Dict, List, Optional

_PUBLISHED_COLUMNS = ["mx", "px", "lx", "dx", "Lx", "Tx", "ex"]

_ZIMBABWE_2023_TABLES = {
    "male_assured_lives": "a_male_assured_lives_a_life_table.csv",
    "group_life_male": "c_group_life_assurance_male_and_female_a_male_life_table.csv",
    "group_life_female": "c_group_life_assurance_male_and_female_b_female_life_table.csv",
    "funeral_principal_members": "d_funeral_principal_members_a_life_table.csv",
    "funeral_spouses": "e_funeral_spouses_a_life_table.csv",
    "funeral_adult_dependents": "f_funeral_adult_dependents_a_life_table.csv",
    "pre_retirement_pensions_male": "g_pre_retirement_pensions_males_females_a_male_life_table.csv",
    "pre_retirement_pensions_female": "g_pre_retirement_pensions_males_females_b_female_life_table.csv",
    "post_retirement_pensions_male": "h_post_retirement_pensions_male_life_tables_a_life_table.csv",
    "post_retirement_pensions_female": "i_post_retirement_pensions_female_life_tables_i_life_table.csv",
}


class MortalityTable:
    """
    A class for handling mortality tables and related calculations.

    The table is defined by one-year mortality rates ``qx`` at consecutive
    integer ages. All other life table columns (``lx``, ``dx``, ``Lx``,
    ``Tx``, ``ex``) are derived from ``qx`` so that they are always internally
    consistent. Columns supplied with the source data are kept unchanged in
    :attr:`published` for reference.

    Many published tables stop at an age where ``qx`` is still below 1 (for
    example at age 100, or at retirement age for pre-retirement tables).
    ``Tx`` and ``ex`` then only count years lived within the tabulated ages,
    unless the source publishes a ``Tx`` column that includes later years, in
    which case those years are carried in :attr:`tail_Tx`.
    """

    def __init__(self,
                 ages: Union[List[int], np.ndarray],
                 qx: Union[List[float], np.ndarray],
                 name: str = "Unnamed Table",
                 metadata: Optional[Dict] = None,
                 table_columns: Optional[Dict[str, Union[List[float], np.ndarray]]] = None):
        """
        Initialize a mortality table.

        Args:
            ages: Array of consecutive integer ages
            qx: Array of mortality rates (probability of death between age x and x+1)
            name: Name/description of the mortality table
            metadata: Additional metadata about the table. ``metadata["radix"]``
                sets the starting number of lives.
            table_columns: Published actuarial columns aligned to ages
                (e.g. mx, lx, dx, Lx, Tx, ex). They are stored in
                :attr:`published`; a published ``lx`` also sets the radix.
        """
        ages_raw = np.asarray(ages, dtype=float)
        qx_arr = np.asarray(qx, dtype=float)

        if ages_raw.ndim != 1 or qx_arr.ndim != 1:
            raise ValueError("ages and qx must be 1-dimensional")

        if len(ages_raw) != len(qx_arr):
            raise ValueError("ages and qx must have the same length")

        if len(ages_raw) == 0:
            raise ValueError("ages and qx must not be empty")

        if np.any(np.isnan(ages_raw)) or np.any(ages_raw != np.round(ages_raw)):
            raise ValueError("ages must be integers")

        ages_arr = ages_raw.astype(int)
        sort_idx = np.argsort(ages_arr)
        ages_arr = ages_arr[sort_idx]
        qx_arr = qx_arr[sort_idx]

        unique_ages, counts = np.unique(ages_arr, return_counts=True)
        if np.any(counts > 1):
            duplicate_ages = unique_ages[counts > 1].tolist()
            raise ValueError(f"ages must be unique. Duplicate ages found: {duplicate_ages}")

        if np.any(np.diff(ages_arr) != 1):
            raise ValueError(
                "ages must be consecutive integers (one-year age steps). "
                "Abridged tables must be interpolated to single ages first."
            )

        if np.any(np.isnan(qx_arr)) or not np.all((qx_arr >= 0) & (qx_arr <= 1)):
            raise ValueError("All qx values must be between 0 and 1")

        self.ages = ages_arr
        self.qx_values = qx_arr
        self.name = name
        self.metadata = dict(metadata) if metadata else {}

        self._table_columns = {}
        if table_columns:
            for key, values in table_columns.items():
                arr = np.asarray(values, dtype=float)
                if arr.ndim != 1:
                    raise ValueError(f"table column '{key}' must be 1-dimensional")
                if len(arr) != len(self.ages):
                    raise ValueError(f"table column '{key}' must have the same length as ages")
                self._table_columns[key] = arr[sort_idx]

        self._calculate_life_table()

    def _calculate_life_table(self):
        """Derive px, lx, dx, Lx, Tx and ex from qx."""
        self.px_values = 1.0 - self.qx_values

        radix = self.metadata.get("radix")
        published_lx = self._table_columns.get("lx")
        if published_lx is not None and not np.isnan(published_lx[0]) and published_lx[0] > 0:
            radix = float(published_lx[0])
        if radix is None:
            radix = 1.0
        if radix <= 0:
            raise ValueError("radix must be positive")
        self.radix = float(radix)

        # lx[k] = radix * p_{x0} * ... * p_{x0+k-1}
        self.lx_values = self.radix * np.concatenate(([1.0], np.cumprod(self.px_values[:-1])))
        self.dx_values = self.lx_values * self.qx_values
        # Deaths assumed uniformly distributed over each year of age
        self.Lx_values = self.lx_values - 0.5 * self.dx_values

        # Years lived beyond the last tabulated age. This is zero unless the
        # source table was cut off below its limiting age and its published Tx
        # still counts the later years (as in the Zimbabwe 2023 tables that
        # stop at age 70).
        self.tail_Tx = 0.0
        published_Tx = self._table_columns.get("Tx")
        if published_Tx is not None and not np.isnan(published_Tx[-1]):
            published_Lx = self._table_columns.get("Lx")
            last_Lx = self.Lx_values[-1]
            if published_Lx is not None and not np.isnan(published_Lx[-1]):
                last_Lx = published_Lx[-1]
            self.tail_Tx = max(0.0, float(published_Tx[-1] - last_Lx))

        self.Tx_values = np.cumsum(self.Lx_values[::-1])[::-1] + self.tail_Tx

        survivors_end = self.lx_values[-1] * self.px_values[-1]
        curtate_tail = max(0.0, self.tail_Tx - 0.5 * survivors_end)

        with np.errstate(divide="ignore", invalid="ignore"):
            self.ex_values = np.where(self.lx_values > 0, self.Tx_values / self.lx_values, 0.0)
            # Curtate expectation: sum of k-year survival probabilities, k >= 1
            survivors_after = np.cumsum((self.lx_values * self.px_values)[::-1])[::-1]
            self.curtate_ex_values = np.where(
                self.lx_values > 0, (survivors_after + curtate_tail) / self.lx_values, 0.0
            )

        self.mx_values = self._table_columns.get("mx")

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------
    @property
    def min_age(self) -> int:
        """Youngest age in the table."""
        return int(self.ages[0])

    @property
    def max_age(self) -> int:
        """Oldest age in the table."""
        return int(self.ages[-1])

    @property
    def omega(self) -> int:
        """First age beyond the table (max_age + 1)."""
        return int(self.ages[-1]) + 1

    @property
    def is_closed(self) -> bool:
        """True if the table ends with certain death (qx = 1 at the last age)."""
        return bool(self.qx_values[-1] >= 1.0)

    @property
    def published(self) -> pd.DataFrame:
        """Columns supplied with the source data, exactly as published."""
        data = {"age": self.ages}
        data.update(self._table_columns)
        return pd.DataFrame(data)

    def _index(self, age: int) -> int:
        """Position of an age in the table, raising if it is not tabulated."""
        if isinstance(age, bool) or age != int(age):
            raise ValueError(f"age must be an integer, got {age!r}")
        age = int(age)
        if age < self.min_age or age > self.max_age:
            raise ValueError(
                f"Age {age} not found in mortality table '{self.name}' "
                f"(ages {self.min_age}-{self.max_age})"
            )
        return age - self.min_age

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_dataframe(cls,
                      df: pd.DataFrame,
                      age_col: str = 'age',
                      qx_col: str = 'qx',
                      name: str = "DataFrame Table",
                      metadata: Optional[Dict] = None) -> 'MortalityTable':
        """
        Create a MortalityTable from a pandas DataFrame.

        Args:
            df: DataFrame containing age and mortality data
            age_col: Column name for ages
            qx_col: Column name for mortality rates
            name: Name for the table
            metadata: Additional metadata about the table

        Returns:
            MortalityTable instance
        """
        ages = df[age_col].values
        qx = df[qx_col].values
        extra_cols = {}
        for key in _PUBLISHED_COLUMNS:
            if key in df.columns and key not in (age_col, qx_col):
                extra_cols[key] = df[key].values
        return cls(ages, qx, name, metadata=metadata, table_columns=extra_cols or None)

    @classmethod
    def from_csv(cls,
                filepath: str,
                age_col: str = 'age',
                qx_col: str = 'qx',
                name: Optional[str] = None,
                metadata: Optional[Dict] = None) -> 'MortalityTable':
        """
        Create a MortalityTable from a CSV file.

        Args:
            filepath: Path to CSV file
            age_col: Column name for ages
            qx_col: Column name for mortality rates
            name: Name for the table (defaults to filename)
            metadata: Additional metadata about the table

        Returns:
            MortalityTable instance
        """
        df = pd.read_csv(filepath)
        table_name = name or os.path.splitext(os.path.basename(filepath))[0]
        return cls.from_dataframe(df, age_col, qx_col, table_name, metadata=metadata)

    @staticmethod
    def zimbabwe_2023_tables() -> List[str]:
        """Names accepted by :meth:`from_zimbabwe_2023`."""
        return sorted(_ZIMBABWE_2023_TABLES)

    @classmethod
    def from_zimbabwe_2023(cls, table: str) -> 'MortalityTable':
        """
        Load one of the Zimbabwe 2023 mortality tables shipped with ACTUNEO.

        Args:
            table: Table name, see :meth:`zimbabwe_2023_tables`

        Returns:
            MortalityTable instance
        """
        table = table.strip().lower()
        if table not in _ZIMBABWE_2023_TABLES:
            raise ValueError(
                f"Unknown Zimbabwe 2023 table '{table}'. Available: {sorted(_ZIMBABWE_2023_TABLES)}"
            )
        csv_rel = f"data/zimbabwe_2023/{_ZIMBABWE_2023_TABLES[table]}"
        csv_path = resources.files("actuneo.mortality") / csv_rel
        with resources.as_file(csv_path) as p:
            return cls.from_csv(
                str(p),
                age_col="age",
                qx_col="qx",
                name=f"Zimbabwe 2023 - {table}",
                metadata={"country": "Zimbabwe", "year": 2023, "table": table},
            )

    def to_dataframe(self) -> pd.DataFrame:
        """Life table derived from qx as a DataFrame."""
        df = pd.DataFrame(
            {
                "age": self.ages,
                "qx": self.qx_values,
                "px": self.px_values,
                "lx": self.lx_values,
                "dx": self.dx_values,
                "Lx": self.Lx_values,
                "Tx": self.Tx_values,
                "ex": self.ex_values,
            }
        )
        if self.mx_values is not None:
            df.insert(1, "mx", self.mx_values)
        return df

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------
    def _lookup(self, values: np.ndarray, age) -> Union[float, np.ndarray]:
        """Look up a column by age; ages outside the table give NaN."""
        ages_array = np.atleast_1d(np.asarray(age, dtype=float))
        result = np.full(ages_array.shape, np.nan)
        idx = ages_array - self.min_age
        valid = (idx >= 0) & (idx < len(self.ages)) & (ages_array == np.round(ages_array))
        result[valid] = values[idx[valid].astype(int)]
        if np.ndim(age) == 0:
            return float(result[0])
        return result

    def qx(self, age: Union[int, List[int]]) -> Union[float, np.ndarray]:
        """
        Get mortality rate(s) for given age(s).

        Args:
            age: Age(s) to get mortality rate for

        Returns:
            Mortality rate(s); NaN for ages outside the table
        """
        return self._lookup(self.qx_values, age)

    def px(self, age: Union[int, List[int]]) -> Union[float, np.ndarray]:
        """
        Get survival probability(ies) for given age(s).

        Args:
            age: Age(s) to get survival probability for

        Returns:
            Survival probability(ies); NaN for ages outside the table
        """
        return self._lookup(self.px_values, age)

    def lx(self, age: Union[int, List[int]]) -> Union[float, np.ndarray]:
        """Number of survivors at given age(s); NaN for ages outside the table."""
        return self._lookup(self.lx_values, age)

    def dx(self, age: Union[int, List[int]]) -> Union[float, np.ndarray]:
        """Number of deaths at given age(s); NaN for ages outside the table."""
        return self._lookup(self.dx_values, age)

    def npx(self, x: int, n: int) -> float:
        """
        Probability that a life aged x survives n years.

        Survival beyond the end of the table is taken as zero.

        Args:
            x: Age (must be in the table)
            n: Number of years (non-negative integer)

        Returns:
            n-year survival probability
        """
        if isinstance(n, bool) or n != int(n) or n < 0:
            raise ValueError("n must be a non-negative integer")
        n = int(n)
        start = self._index(x)
        if start + n > len(self.ages):
            return 0.0
        return float(np.prod(self.px_values[start:start + n]))

    def nqx(self, x: int, n: int) -> float:
        """Probability that a life aged x dies within n years."""
        return 1.0 - self.npx(x, n)

    def deferred_qx(self, x: int, n: int, m: int = 1) -> float:
        """
        Probability that a life aged x survives n years and then dies
        within the following m years (n|m qx).
        """
        if isinstance(m, bool) or m != int(m) or m < 0:
            raise ValueError("m must be a non-negative integer")
        return self.npx(x, n) - self.npx(x, n + int(m))

    def ex(self, age: int, kind: str = "complete") -> float:
        """
        Expectation of life at a given age.

        Args:
            age: Age (must be in the table)
            kind: "complete" (Tx / lx, deaths uniform over the year of age)
                or "curtate" (whole future years only)

        Returns:
            Expected future lifetime in years. Years beyond the last
            tabulated age are counted only if the source table publishes them.
        """
        idx = self._index(age)
        if kind == "complete":
            return float(self.ex_values[idx])
        if kind == "curtate":
            return float(self.curtate_ex_values[idx])
        raise ValueError("kind must be 'complete' or 'curtate'")

    def get_qx(self, age: Union[int, List[int]]) -> Union[float, np.ndarray]:
        return self.qx(age)

    def get_px(self, age: Union[int, List[int]]) -> Union[float, np.ndarray]:
        return self.px(age)

    def life_expectancy(self, age: int, kind: str = "complete") -> float:
        """
        Calculate life expectancy at a given age.

        Args:
            age: Age to calculate life expectancy for
            kind: "complete" or "curtate"

        Returns:
            Life expectancy in years
        """
        return self.ex(age, kind)

    def __len__(self) -> int:
        return len(self.ages)

    def __repr__(self) -> str:
        return f"MortalityTable(name='{self.name}', ages={len(self.ages)}, range=({self.ages[0]}-{self.ages[-1]}))"
