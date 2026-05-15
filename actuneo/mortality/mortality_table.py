"""
Mortality Table Implementation

Provides classes and methods for working with mortality tables,
including loading, manipulation, and basic calculations.
"""

import numpy as np
import pandas as pd
import os
from importlib import resources
from typing import Union, Dict, List, Optional


class MortalityTable:
    """
    A class for handling mortality tables and related calculations.

    Supports various formats of mortality data and provides methods
    for survival probabilities, life expectancy, and other actuarial functions.
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
            ages: Array of ages
            qx: Array of mortality rates (probability of death between age x and x+1)
            name: Name/description of the mortality table
            metadata: Additional metadata about the table
            table_columns: Additional actuarial columns aligned to ages (e.g. mx, lx, dx, Lx, Tx, ex)
        """
        ages_arr = np.array(ages, dtype=int)
        qx_arr = np.array(qx, dtype=float)

        if ages_arr.ndim != 1 or qx_arr.ndim != 1:
            raise ValueError("ages and qx must be 1-dimensional")

        if len(ages_arr) != len(qx_arr):
            raise ValueError("ages and qx must have the same length")

        sort_idx = np.argsort(ages_arr)
        ages_arr = ages_arr[sort_idx]
        qx_arr = qx_arr[sort_idx]

        unique_ages, counts = np.unique(ages_arr, return_counts=True)
        if np.any(counts > 1):
            duplicate_ages = unique_ages[counts > 1].tolist()
            raise ValueError(f"ages must be unique. Duplicate ages found: {duplicate_ages}")

        self.ages = ages_arr
        self.qx_values = qx_arr
        self.name = name
        self.metadata = metadata or {}
        self._table_columns = {}
        if table_columns:
            for key, values in table_columns.items():
                arr = np.array(values, dtype=float)
                if arr.ndim != 1:
                    raise ValueError(f"table column '{key}' must be 1-dimensional")
                if len(arr) != len(self.ages):
                    raise ValueError(f"table column '{key}' must have the same length as ages")
                self._table_columns[key] = arr
            for key in list(self._table_columns.keys()):
                self._table_columns[key] = self._table_columns[key][sort_idx]

        # Validate inputs
        if not np.all((self.qx_values >= 0) & (self.qx_values <= 1)):
            raise ValueError("All qx values must be between 0 and 1")

        # Calculate derived quantities
        self._calculate_survival_probabilities()

    def _calculate_survival_probabilities(self):
        """Calculate survival probabilities (px) and cumulative survival (lx, dx)."""
        self.px_values = 1.0 - self.qx_values

        radix = self.metadata.get("radix")
        if "lx" in self._table_columns and len(self._table_columns["lx"]) > 0 and not np.isnan(self._table_columns["lx"][0]):
            radix = float(self._table_columns["lx"][0])
        if radix is None:
            radix = 1.0

        if "lx" in self._table_columns:
            self.lx_values = self._table_columns["lx"].astype(float)
        else:
            lx = np.zeros_like(self.qx_values, dtype=float)
            lx[0] = float(radix)
            for i in range(1, len(lx)):
                gap = int(self.ages[i] - self.ages[i - 1])
                if gap <= 0:
                    raise ValueError("ages must be strictly increasing")
                lx[i] = lx[i - 1] * (self.px_values[i - 1] ** gap)
            self.lx_values = lx

        if "dx" in self._table_columns:
            self.dx_values = self._table_columns["dx"].astype(float)
        else:
            self.dx_values = self.lx_values * self.qx_values

        if "Lx" in self._table_columns:
            self.Lx_values = self._table_columns["Lx"].astype(float)
        else:
            self.Lx_values = self.lx_values - 0.5 * self.dx_values

        if "Tx" in self._table_columns:
            self.Tx_values = self._table_columns["Tx"].astype(float)
        else:
            self.Tx_values = np.cumsum(self.Lx_values[::-1])[::-1]

        if "ex" in self._table_columns:
            self.ex_values = self._table_columns["ex"].astype(float)
        else:
            self.ex_values = np.where(self.lx_values > 0, self.Tx_values / self.lx_values, 0.0)

        self.mx_values = self._table_columns.get("mx")

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

        Returns:
            MortalityTable instance
        """
        ages = df[age_col].values
        qx = df[qx_col].values
        extra_cols = {}
        for key in ["mx", "lx", "dx", "Lx", "Tx", "ex"]:
            if key in df.columns:
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

        Returns:
            MortalityTable instance
        """
        df = pd.read_csv(filepath)
        table_name = name or os.path.splitext(os.path.basename(filepath))[0]
        return cls.from_dataframe(df, age_col, qx_col, table_name, metadata=metadata)

    @classmethod
    def from_zimbabwe_2023(cls, table: str) -> 'MortalityTable':
        table = table.strip().lower()
        mapping = {
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
        if table not in mapping:
            raise ValueError(f"Unknown Zimbabwe 2023 table '{table}'. Available: {sorted(mapping.keys())}")
        csv_rel = f\"data/zimbabwe_2023/{mapping[table]}\"
        csv_path = resources.files("actuneo.mortality") / csv_rel
        with resources.as_file(csv_path) as p:
            return cls.from_csv(str(p), age_col="age", qx_col="qx", name=f\"Zimbabwe 2023 - {table}\", metadata={\"country\": \"Zimbabwe\", \"year\": 2023})

    def to_dataframe(self) -> pd.DataFrame:
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

    def qx(self, age: Union[int, List[int]]) -> Union[float, np.ndarray]:
        """
        Get mortality rate(s) for given age(s).

        Args:
            age: Age(s) to get mortality rate for

        Returns:
            Mortality rate(s)
        """
        ages_array = np.atleast_1d(age).astype(int)
        result = np.zeros_like(ages_array, dtype=float)

        for i, a in enumerate(ages_array):
            if a in self.ages:
                result[i] = self.qx_values[np.where(self.ages == a)[0][0]]
            else:
                result[i] = np.nan

        if np.isscalar(age) or (hasattr(age, '__len__') and len(age) == 1):
            return float(result.item()) if result.ndim == 0 else float(result[0])
        else:
            return result

    def px(self, age: Union[int, List[int]]) -> Union[float, np.ndarray]:
        """
        Get survival probability(ies) for given age(s).

        Args:
            age: Age(s) to get survival probability for

        Returns:
            Survival probability(ies)
        """
        qx_vals = self.qx(age)
        px_vals = 1 - qx_vals

        if np.isscalar(px_vals):
            return px_vals if not np.isnan(px_vals) else np.nan
        else:
            px_vals = np.where(np.isnan(qx_vals), np.nan, px_vals)
            return px_vals

    def npx(self, x: int, n: int) -> float:
        if n < 0:
            return 0.0
        if n == 0:
            return 1.0
        if x not in self.ages:
            return 0.0

        start_idx = int(np.where(self.ages == x)[0][0])
        end_age = x + n
        if end_age in self.ages:
            end_idx = int(np.where(self.ages == end_age)[0][0])
            if end_idx <= start_idx:
                return 0.0
            return float(np.prod(self.px_values[start_idx:end_idx]))

        if end_age > self.ages[-1]:
            px = float(np.prod(self.px_values[start_idx:]))
            last_px = float(self.px_values[-1])
            remaining_years = end_age - int(self.ages[-1])
            return float(px * (last_px ** max(0, remaining_years)))

        return 0.0

    def ex(self, age: int) -> float:
        if age not in self.ages:
            raise ValueError(f"Age {age} not found in mortality table")
        idx = int(np.where(self.ages == age)[0][0])
        return float(self.ex_values[idx])

    def get_qx(self, age: Union[int, List[int]]) -> Union[float, np.ndarray]:
        return self.qx(age)

    def get_px(self, age: Union[int, List[int]]) -> Union[float, np.ndarray]:
        return self.px(age)

    def life_expectancy(self, age: int) -> float:
        """
        Calculate life expectancy at a given age.

        Args:
            age: Age to calculate life expectancy for

        Returns:
            Life expectancy in years
        """
        return self.ex(age)

    def __repr__(self) -> str:
        return f"MortalityTable(name='{self.name}', ages={len(self.ages)}, range=({self.ages[0]}-{self.ages[-1]}))"
