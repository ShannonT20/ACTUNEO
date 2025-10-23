"""
Life Assurance Calculations

Provides calculations for various life assurance products including
whole life, term life, endowment assurance, and related products.
"""

import numpy as np
from typing import Optional, Union
from ..mortality import MortalityTable, SurvivalFunctions


class LifeAssurance:
    """
    A class for calculating life assurance premiums, reserves, and values.
    """

    def __init__(self,
                 mortality_table: MortalityTable,
                 interest_rate: float = 0.05,
                 expense_loading: float = 0.0):
        """
        Initialize LifeAssurance calculator.

        Args:
            mortality_table: MortalityTable instance
            interest_rate: Annual interest rate for discounting
            expense_loading: Expense loading as percentage of premium
        """
        self.mt = mortality_table
        self.sf = SurvivalFunctions(mortality_table, interest_rate)
        self.i = interest_rate
        self.v = 1 / (1 + interest_rate)
        self.expense_loading = expense_loading

    def whole_life_assurance(self, x: int, discrete: bool = True) -> float:
        """
        Calculate net single premium for whole life assurance.

        Args:
            x: Age at entry
            discrete: True for discrete model, False for continuous

        Returns:
            Net single premium for whole life assurance
        """
        return self.sf.assurance(x)

    def term_assurance(self, x: int, n: int, discrete: bool = True) -> float:
        """
        Calculate net single premium for term assurance.

        Args:
            x: Age at entry
            n: Term of assurance in years
            discrete: True for discrete model, False for continuous

        Returns:
            Net single premium for n-year term assurance
        """
        return self.sf.assurance(x, n)

    def endowment_assurance(self, x: int, n: int) -> float:
        """
        Calculate net single premium for endowment assurance.

        Args:
            x: Age at entry
            n: Term of assurance in years

        Returns:
            Net single premium for endowment assurance
        """
        # Endowment assurance = Term assurance + Pure endowment
        term_assurance = self.term_assurance(x, n)
        pure_endowment = self.pure_endowment(x, n)

        return term_assurance + pure_endowment

    def pure_endowment(self, x: int, n: int) -> float:
        """
        Calculate net single premium for pure endowment.

        Args:
            x: Age at entry
            n: Term in years

        Returns:
            Net single premium for pure endowment
        """
        survival_prob = self.sf.npx(x, n)
        return survival_prob * self.v ** n

    def deferred_assurance(self, x: int, u: int, n: int) -> float:
        """
        Calculate net single premium for deferred assurance.

        Args:
            x: Age at entry
            u: Deferment period in years
            n: Assurance term in years

        Returns:
            Net single premium for deferred assurance
        """
        # Deferred assurance pays if death occurs between age x+u and x+u+n
        survival_to_deferment = self.sf.npx(x, u)

        if survival_to_deferment == 0:
            return 0.0

        # Calculate assurance from age x+u for n years
        deferred_assurance = self.term_assurance(x + u, n)

        return survival_to_deferment * self.v ** u * deferred_assurance

    def temporary_life_annuity(self, x: int, n: int) -> float:
        """
        Calculate net single premium for temporary life annuity.

        Args:
            x: Age at entry
            n: Term in years

        Returns:
            Net single premium for temporary life annuity
        """
        return self.sf.annuity_immediate(x, n)

    def whole_life_annuity(self, x: int) -> float:
        """
        Calculate net single premium for whole life annuity.

        Args:
            x: Age at entry

        Returns:
            Net single premium for whole life annuity
        """
        return self.sf.annuity_immediate(x)

    def contingent_assurance(self, x: int, y: int, n: int) -> float:
        """
        Calculate net single premium for contingent assurance (last survivor).

        Args:
            x: Age of first life
            y: Age of second life
            n: Term in years

        Returns:
            Net single premium for contingent assurance
        """
        # This is a simplified calculation assuming independent lives
        # In practice, this would require joint life mortality tables

        assurance = 0.0
        v_t = self.v

        for t in range(1, n + 1):
            # Probability that (x) dies in year t and (y) survives to t
            qx_t = 1 - self.sf.npx(x, t) + self.sf.npx(x, t-1)  # Approximate
            py_t = self.sf.npx(y, t)

            assurance += v_t * qx_t * py_t
            v_t *= self.v

        return assurance

    def joint_life_assurance(self, x: int, y: int) -> float:
        """
        Calculate net single premium for joint life assurance.

        Args:
            x: Age of first life
            y: Age of second life

        Returns:
            Net single premium for joint life assurance
        """
        # Simplified calculation - in practice needs joint mortality
        min_age = min(x, y)
        assurance = self.whole_life_assurance(min_age) * 0.8  # Rough approximation
        return assurance

    def gross_premium(self, net_premium: float, initial_expenses: float = 0.0) -> float:
        """
        Calculate gross premium including loadings.

        Args:
            net_premium: Net single premium
            initial_expenses: Initial expense loading

        Returns:
            Gross premium
        """
        expense_loading = net_premium * self.expense_loading
        return net_premium + expense_loading + initial_expenses

    def annual_premium(self,
                      net_single_premium: float,
                      annuity_factor: float,
                      gross_margin: float = 0.0) -> float:
        """
        Calculate annual premium using annuity factor.

        Args:
            net_single_premium: Net single premium
            annuity_factor: Annuity factor for premium payments
            gross_margin: Additional margin for expenses and profit

        Returns:
            Annual premium
        """
        if annuity_factor == 0:
            return 0.0

        net_annual = net_single_premium / annuity_factor
        return net_annual * (1 + gross_margin)

    def reserve_whole_life(self, x: int, duration: int) -> float:
        """
        Calculate prospective reserve for whole life assurance.

        Args:
            x: Original age at entry
            duration: Number of years policy has been in force

        Returns:
            Prospective reserve
        """
        current_age = x + duration

        # Reserve = A_{x+t} / ä_{x+t}
        # Where A is whole life assurance, ä is life annuity

        remaining_assurance = self.whole_life_assurance(current_age)
        remaining_annuity = self.whole_life_annuity(current_age)

        if remaining_annuity == 0:
            return remaining_assurance

        return remaining_assurance / remaining_annuity

    def reserve_term(self, x: int, n: int, duration: int) -> float:
        """
        Calculate prospective reserve for term assurance.

        Args:
            x: Original age at entry
            n: Original term
            duration: Number of years policy has been in force

        Returns:
            Prospective reserve
        """
        current_age = x + duration
        remaining_term = n - duration

        if remaining_term <= 0:
            return 0.0

        remaining_assurance = self.term_assurance(current_age, remaining_term)
        remaining_annuity = self.temporary_life_annuity(current_age, remaining_term)

        if remaining_annuity == 0:
            return remaining_assurance

        return remaining_assurance / remaining_annuity

    def reserve_endowment(self, x: int, n: int, duration: int) -> float:
        """
        Calculate prospective reserve for endowment assurance.

        Args:
            x: Original age at entry
            n: Original term
            duration: Number of years policy has been in force

        Returns:
            Prospective reserve
        """
        current_age = x + duration
        remaining_term = n - duration

        if remaining_term <= 0:
            return 0.0  # Policy matured

        remaining_endowment = self.endowment_assurance(current_age, remaining_term)
        remaining_annuity = self.temporary_life_annuity(current_age, remaining_term)

        if remaining_annuity == 0:
            return remaining_endowment

        return remaining_endowment / remaining_annuity
