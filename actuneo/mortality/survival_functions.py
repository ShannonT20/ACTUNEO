"""
Survival Functions

Provides actuarial survival functions and calculations including
temporary and permanent life functions, actuarial present values, etc.
"""

import numpy as np
from typing import Union, Optional
from .mortality_table import MortalityTable


class SurvivalFunctions:
    """
    A class for calculating various actuarial survival functions
    and life contingencies based on mortality tables.
    """

    def __init__(self, mortality_table: MortalityTable, interest_rate: float = 0.05):
        """
        Initialize SurvivalFunctions with a mortality table and interest rate.

        Args:
            mortality_table: MortalityTable instance
            interest_rate: Annual interest rate for discounting (default 5%)
        """
        self.mt = mortality_table
        self.i = interest_rate
        self.v = 1 / (1 + interest_rate)  # Discount factor

    def npx(self, x: int, n: int) -> float:
        """
        Calculate n-year survival probability: npx

        Args:
            x: Age
            n: Number of years

        Returns:
            Probability that (x) survives n years
        """
        if n < 0:
            return 0.0
        if n == 0:
            return 1.0

        if x + n > self.mt.ages[-1]:
            # Extrapolate using last available survival probability
            last_age_idx = np.where(self.mt.ages <= x)[0]
            if len(last_age_idx) == 0:
                return 0.0
            last_idx = last_age_idx[-1]
            remaining_years = x + n - self.mt.ages[last_idx]
            px_remaining = self.mt.px[last_idx] ** remaining_years
            return self.mt.lx[last_idx] * px_remaining / self.mt.lx[np.where(self.mt.ages == x)[0][0]]

        # Find indices for ages x to x+n
        start_idx = np.where(self.mt.ages == x)[0]
        end_idx = np.where(self.mt.ages == x + n)[0]

        if len(start_idx) == 0 or len(end_idx) == 0:
            return 0.0

        start_idx = start_idx[0]
        end_idx = end_idx[0]

        # Calculate cumulative survival probability
        survival_prob = np.prod(self.mt.px[start_idx:end_idx])
        return float(survival_prob)

    def nqx(self, x: int, n: int) -> float:
        """
        Calculate n-year mortality probability: nqx

        Args:
            x: Age
            n: Number of years

        Returns:
            Probability that (x) dies within n years
        """
        return 1 - self.npx(x, n)

    def tpx(self, x: int, t: float) -> float:
        """
        Calculate t-year survival probability using linear interpolation.

        Args:
            x: Age
            t: Fractional years

        Returns:
            Probability that (x) survives t years
        """
        if t == 0:
            return 1.0
        elif t < 0:
            return 0.0

        # Integer and fractional parts
        n = int(t)
        frac = t - n

        # Base survival probability for n years
        n_year_survival = self.npx(x, n)

        if n + x >= self.mt.ages[-1]:
            # Extrapolate using constant force of mortality from last age
            last_age = self.mt.ages[-1]
            if x + n >= last_age:
                remaining_years = t - (last_age - x)
                q_last = self.mt.get_qx(last_age)
                mu = -np.log(1 - q_last)  # Force of mortality approximation
                return n_year_survival * np.exp(-mu * remaining_years)
            else:
                return n_year_survival

        # Linear interpolation for fractional year
        q_next = self.mt.get_qx(x + n)
        survival_frac = 1 - (frac * q_next)

        return n_year_survival * survival_frac

    def annuity_due(self, x: int, n: Optional[int] = None) -> float:
        """
        Calculate the actuarial present value of an annuity-due: äx or äx:n

        Args:
            x: Age
            n: Term of annuity (None for whole life)

        Returns:
            Actuarial present value of annuity-due
        """
        if n is None:
            # Whole life annuity
            annuity = 0.0
            v_t = 1.0
            for t in range(len(self.mt.ages) - np.where(self.mt.ages == x)[0][0]):
                annuity += v_t * self.tpx(x, t)
                v_t *= self.v
            return annuity
        else:
            # Term annuity
            annuity = 0.0
            v_t = 1.0
            for t in range(n):
                annuity += v_t * self.tpx(x, t)
                v_t *= self.v
            return annuity

    def annuity_immediate(self, x: int, n: Optional[int] = None) -> float:
        """
        Calculate the actuarial present value of an immediate annuity: a¨x or a¨x:n

        Args:
            x: Age
            n: Term of annuity (None for whole life)

        Returns:
            Actuarial present value of immediate annuity
        """
        return self.annuity_due(x, n) - 1.0

    def assurance(self, x: int, n: Optional[int] = None) -> float:
        """
        Calculate the actuarial present value of a whole life assurance: Ax or Ax:n

        Args:
            x: Age
            n: Term of assurance (None for whole life)

        Returns:
            Actuarial present value of assurance
        """
        if n is None:
            # Whole life assurance
            assurance = 0.0
            v_t = self.v
            max_t = len(self.mt.ages) - np.where(self.mt.ages == x)[0][0] - 1
            for t in range(1, max_t + 1):
                # Simplified: probability of death in year t
                q_xt = 1 - self.npx(x, t) / self.npx(x, t-1) if t > 1 else 1 - self.npx(x, 1)
                assurance += v_t * min(q_xt, 1.0)  # Cap at 1.0
                v_t *= self.v
            return float(assurance)
        else:
            # Term assurance
            assurance = 0.0
            v_t = self.v
            for t in range(1, n + 1):
                # Simplified: probability of death in year t
                q_xt = 1 - self.npx(x, t) / self.npx(x, t-1) if t > 1 else 1 - self.npx(x, 1)
                assurance += v_t * min(q_xt, 1.0)  # Cap at 1.0
                v_t *= self.v
            return float(assurance)

    def net_single_premium(self, x: int, n: Optional[int] = None) -> float:
        """
        Calculate net single premium for whole life or term assurance.

        Args:
            x: Age
            n: Term (None for whole life)

        Returns:
            Net single premium
        """
        return self.assurance(x, n)
