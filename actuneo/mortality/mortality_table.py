"""
Mortality Table Implementation

Provides classes and methods for working with mortality tables,
including loading, manipulation, and basic calculations.
"""

import numpy as np
import pandas as pd
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
                 metadata: Optional[Dict] = None):
        """
        Initialize a mortality table.

        Args:
            ages: Array of ages
            qx: Array of mortality rates (probability of death between age x and x+1)
            name: Name/description of the mortality table
            metadata: Additional metadata about the table
        """
        self.ages = np.array(ages, dtype=int)
        self.qx = np.array(qx, dtype=float)
        self.name = name
        self.metadata = metadata or {}

        # Validate inputs
        if len(self.ages) != len(self.qx):
            raise ValueError("ages and qx must have the same length")

        if not np.all((self.qx >= 0) & (self.qx <= 1)):
            raise ValueError("All qx values must be between 0 and 1")

        # Calculate derived quantities
        self._calculate_survival_probabilities()

    def _calculate_survival_probabilities(self):
        """Calculate survival probabilities (px) and cumulative survival (lx, dx)."""
        self.px = 1 - self.qx  # Survival probability
        self.lx = np.cumprod(self.px[::-1])[::-1]  # Number alive at age x (assuming lx[0] = 1)
        self.lx = self.lx / self.lx[0]  # Normalize so lx[0] = 1
        self.dx = self.lx * self.qx  # Number dying between age x and x+1

    @classmethod
    def from_dataframe(cls,
                      df: pd.DataFrame,
                      age_col: str = 'age',
                      qx_col: str = 'qx',
                      name: str = "DataFrame Table") -> 'MortalityTable':
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
        return cls(ages, qx, name)

    @classmethod
    def from_csv(cls,
                filepath: str,
                age_col: str = 'age',
                qx_col: str = 'qx',
                name: Optional[str] = None) -> 'MortalityTable':
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
        table_name = name or filepath.split('/')[-1].split('.')[0]
        return cls.from_dataframe(df, age_col, qx_col, table_name)

    def get_qx(self, age: Union[int, List[int]]) -> Union[float, np.ndarray]:
        """
        Get mortality rate(s) for given age(s).

        Args:
            age: Age(s) to get mortality rate for

        Returns:
            Mortality rate(s)
        """
        # Convert to numpy array to handle both scalars and lists uniformly
        ages_array = np.atleast_1d(age).astype(int)
        result = np.zeros_like(ages_array, dtype=float)

        for i, a in enumerate(ages_array):
            if a in self.ages:
                result[i] = self.qx[np.where(self.ages == a)[0][0]]
            else:
                result[i] = np.nan

        # Return scalar if input was scalar
        if np.isscalar(age) or (hasattr(age, '__len__') and len(age) == 1):
            return float(result.item()) if result.ndim == 0 else float(result[0])
        else:
            return result

    def get_px(self, age: Union[int, List[int]]) -> Union[float, np.ndarray]:
        """
        Get survival probability(ies) for given age(s).

        Args:
            age: Age(s) to get survival probability for

        Returns:
            Survival probability(ies)
        """
        qx_vals = self.get_qx(age)
        px_vals = 1 - qx_vals

        # Handle NaN values
        if np.isscalar(px_vals):
            return px_vals if not np.isnan(px_vals) else np.nan
        else:
            px_vals = np.where(np.isnan(qx_vals), np.nan, px_vals)
            return px_vals

    def life_expectancy(self, age: int) -> float:
        """
        Calculate life expectancy at a given age.

        Args:
            age: Age to calculate life expectancy for

        Returns:
            Life expectancy in years
        """
        if age not in self.ages:
            raise ValueError(f"Age {age} not found in mortality table")

        idx = np.where(self.ages == age)[0][0]
        remaining_lx = self.lx[idx:]

        if len(remaining_lx) <= 1:
            return 0.0

        # Calculate life expectancy as sum of survival probabilities
        life_exp = 0.5  # Start with 0.5 for the first year
        for i in range(1, len(remaining_lx)):
            life_exp += remaining_lx[i]

        return life_exp

    def __repr__(self) -> str:
        return f"MortalityTable(name='{self.name}', ages={len(self.ages)}, range=({self.ages[0]}-{self.ages[-1]}))"
