"""
Annuity Calculations

Provides comprehensive annuity calculations including immediate annuities,
annuities-due, life annuities, and various annuity forms.
"""

import numpy as np
from typing import Optional, Union, List
from ..mortality import MortalityTable, SurvivalFunctions


class Annuities:
    """
    A class for calculating various types of annuities and annuity values.
    """

    def __init__(self,
                 mortality_table: Optional[MortalityTable] = None,
                 interest_rate: float = 0.05):
        """
        Initialize Annuities calculator.

        Args:
            mortality_table: MortalityTable instance (None for deterministic annuities)
            interest_rate: Annual interest rate for discounting
        """
        self.mt = mortality_table
        if mortality_table:
            self.sf = SurvivalFunctions(mortality_table, interest_rate)
        self.i = interest_rate
        self.v = 1 / (1 + interest_rate)

    def immediate_annuity(self,
                         periods: int,
                         payment: float = 1.0) -> float:
        """
        Calculate present value of immediate annuity.

        Args:
            periods: Number of periods
            payment: Periodic payment amount

        Returns:
            Present value of immediate annuity
        """
        if self.i == 0:
            return payment * periods

        return payment * ((1 - self.v ** periods) / self.i)

    def annuity_due(self,
                   periods: int,
                   payment: float = 1.0) -> float:
        """
        Calculate present value of annuity-due.

        Args:
            periods: Number of periods
            payment: Periodic payment amount

        Returns:
            Present value of annuity-due
        """
        if self.i == 0:
            return payment * periods

        return payment * ((1 - self.v ** periods) / self.i) * (1 + self.i)

    def life_annuity_immediate(self,
                              x: int,
                              payment: float = 1.0) -> float:
        """
        Calculate present value of immediate life annuity.

        Args:
            x: Age
            payment: Annual payment amount

        Returns:
            Present value of immediate life annuity
        """
        if not self.mt:
            raise ValueError("Mortality table required for life annuities")

        return payment * self.sf.annuity_immediate(x)

    def life_annuity_due(self,
                        x: int,
                        payment: float = 1.0) -> float:
        """
        Calculate present value of life annuity-due.

        Args:
            x: Age
            payment: Annual payment amount

        Returns:
            Present value of life annuity-due
        """
        if not self.mt:
            raise ValueError("Mortality table required for life annuities")

        return payment * self.sf.annuity_due(x)

    def temporary_life_annuity_immediate(self,
                                        x: int,
                                        n: int,
                                        payment: float = 1.0) -> float:
        """
        Calculate present value of temporary immediate life annuity.

        Args:
            x: Age
            n: Number of years
            payment: Annual payment amount

        Returns:
            Present value of temporary immediate life annuity
        """
        if not self.mt:
            raise ValueError("Mortality table required for life annuities")

        return payment * self.sf.annuity_immediate(x, n)

    def temporary_life_annuity_due(self,
                                  x: int,
                                  n: int,
                                  payment: float = 1.0) -> float:
        """
        Calculate present value of temporary life annuity-due.

        Args:
            x: Age
            n: Number of years
            payment: Annual payment amount

        Returns:
            Present value of temporary life annuity-due
        """
        if not self.mt:
            raise ValueError("Mortality table required for life annuities")

        return payment * self.sf.annuity_due(x, n)

    def deferred_life_annuity(self,
                             x: int,
                             u: int,
                             payment: float = 1.0) -> float:
        """
        Calculate present value of deferred life annuity.

        Args:
            x: Age
            u: Deferment period
            payment: Annual payment amount

        Returns:
            Present value of deferred life annuity
        """
        if not self.mt:
            raise ValueError("Mortality table required for life annuities")

        # Deferred annuity = v^u * ä_{x+u}
        survival_prob = self.sf.npx(x, u)
        annuity_value = self.life_annuity_immediate(x + u, payment)

        return survival_prob * self.v ** u * annuity_value

    def guaranteed_annuity(self,
                          x: int,
                          n: int,
                          payment: float = 1.0) -> float:
        """
        Calculate present value of guaranteed annuity.

        Args:
            x: Age
            n: Guarantee period
            payment: Annual payment amount

        Returns:
            Present value of guaranteed annuity
        """
        if not self.mt:
            raise ValueError("Mortality table required for life annuities")

        # Guaranteed annuity = ä_x:n¨ + v^n * ä_{x+n}
        temp_annuity = self.temporary_life_annuity_immediate(x, n, payment)
        deferred_annuity = self.deferred_life_annuity(x, n, payment)

        return temp_annuity + deferred_annuity

    def joint_life_annuity(self,
                          x: int,
                          y: int,
                          payment: float = 1.0) -> float:
        """
        Calculate present value of joint life annuity (last survivor).

        Args:
            x: Age of first life
            y: Age of second life
            payment: Annual payment amount

        Returns:
            Present value of joint life annuity
        """
        if not self.mt:
            raise ValueError("Mortality table required for life annuities")

        # Simplified calculation - in practice needs joint mortality tables
        annuity_x = self.life_annuity_immediate(x, payment)
        annuity_y = self.life_annuity_immediate(y, payment)

        # Rough approximation for last survivor annuity
        return annuity_x + annuity_y - min(annuity_x, annuity_y) * 0.7

    def contingent_annuity(self,
                          x: int,
                          y: int,
                          payment: float = 1.0) -> float:
        """
        Calculate present value of contingent annuity.

        Args:
            x: Age of annuitant
            y: Age of contingent beneficiary
            payment: Annual payment amount

        Returns:
            Present value of contingent annuity
        """
        if not self.mt:
            raise ValueError("Mortality table required for life annuities")

        # Contingent annuity pays while (x) is alive and (y) has died
        # This is a simplified calculation

        annuity = 0.0
        v_t = 1.0

        # Approximate calculation for ages up to 100
        for t in range(1, 81):  # Reasonable maximum
            # Probability (x) survives to t and (y) dies before t
            px = self.sf.npx(x, t)
            qy_t = 1 - self.sf.npx(y, t-1)  # Approximate

            annuity += v_t * px * qy_t * payment
            v_t *= self.v

        return annuity

    def increasing_annuity(self,
                          periods: int,
                          payment: float = 1.0,
                          increase_rate: float = 1.0) -> float:
        """
        Calculate present value of increasing annuity.

        Args:
            periods: Number of periods
            payment: Initial payment amount
            increase_rate: Rate of increase per period

        Returns:
            Present value of increasing annuity
        """
        if self.i == increase_rate:
            # Special case when interest rate equals increase rate
            return payment * periods * self.v

        total_pv = 0.0
        for t in range(1, periods + 1):
            payment_t = payment * ((1 + increase_rate) ** (t - 1))
            total_pv += payment_t * self.v ** t

        return total_pv

    def decreasing_annuity(self,
                          periods: int,
                          payment: float = 1.0,
                          decrease_rate: float = 1.0) -> float:
        """
        Calculate present value of decreasing annuity.

        Args:
            periods: Number of periods
            payment: Initial payment amount
            decrease_rate: Rate of decrease per period

        Returns:
            Present value of decreasing annuity
        """
        total_pv = 0.0
        for t in range(1, periods + 1):
            payment_t = payment * (decrease_rate ** (t - 1))
            total_pv += payment_t * self.v ** t

        return total_pv

    def annuity_with_withdrawal(self,
                               principal: float,
                               withdrawal_rate: float,
                               periods: Optional[int] = None) -> dict:
        """
        Calculate annuity payments from principal with systematic withdrawals.

        Args:
            principal: Initial principal amount
            withdrawal_rate: Annual withdrawal rate (decimal)
            periods: Number of periods (None for perpetual)

        Returns:
            Dictionary with payment amount, remaining principal, etc.
        """
        if periods is None:
            # Perpetual annuity
            payment = principal * withdrawal_rate
            return {
                'annual_payment': payment,
                'remaining_principal': principal,
                'periods': 'perpetual'
            }
        else:
            # Finite periods
            payment = principal * (withdrawal_rate / (1 - (1 + withdrawal_rate - self.i) ** (-periods)))
            remaining = principal

            schedule = []
            for t in range(1, periods + 1):
                interest = remaining * self.i
                withdrawal = payment
                remaining = remaining + interest - withdrawal
                schedule.append({
                    'period': t,
                    'starting_balance': remaining + withdrawal - interest,
                    'interest': interest,
                    'withdrawal': withdrawal,
                    'ending_balance': remaining
                })

            return {
                'annual_payment': payment,
                'remaining_principal': remaining,
                'periods': periods,
                'schedule': schedule
            }

    def annuity_certain_with_life_contingency(self,
                                            x: int,
                                            n: int,
                                            payment: float = 1.0) -> float:
        """
        Calculate present value of annuity certain with life contingency.

        Args:
            x: Age
            n: Certain period
            payment: Annual payment amount

        Returns:
            Present value of annuity certain with life contingency
        """
        if not self.mt:
            raise ValueError("Mortality table required for life annuities")

        # Pays for certain n years or until death, whichever is shorter
        return self.temporary_life_annuity_immediate(x, n, payment)
