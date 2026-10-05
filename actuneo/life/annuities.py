"""
Annuity Calculations

Provides comprehensive annuity calculations including immediate annuities,
annuities-due, life annuities, and various annuity forms.

"Immediate" annuities are payable in arrears (at the end of each period) and
annuities-due are payable in advance (at the start of each period).
"""

import numpy as np
from typing import Optional
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
            mortality_table: MortalityTable instance (None for annuities certain)
            interest_rate: Annual interest rate for discounting
        """
        if interest_rate <= -1:
            raise ValueError("interest_rate must be greater than -1.0.")
        self.mt = mortality_table
        self.sf = SurvivalFunctions(mortality_table, interest_rate) if mortality_table else None
        self.i = interest_rate
        self.v = 1 / (1 + interest_rate)

    def _require_table(self):
        if self.sf is None:
            raise ValueError("Mortality table required for life annuities")

    # ------------------------------------------------------------------
    # Annuities certain
    # ------------------------------------------------------------------
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

    def increasing_annuity(self,
                          periods: int,
                          payment: float = 1.0,
                          increase_rate: float = 0.0) -> float:
        """
        Calculate present value of an annuity in arrears whose payments grow
        at a compound rate.

        Args:
            periods: Number of periods
            payment: First payment amount
            increase_rate: Compound rate of increase per period (0.05 for 5%)

        Returns:
            Present value of increasing annuity
        """
        t = np.arange(1, periods + 1)
        payments = payment * (1 + increase_rate) ** (t - 1)
        return float(np.sum(payments * self.v ** t))

    def decreasing_annuity(self,
                          periods: int,
                          payment: float = 1.0,
                          decrease_rate: float = 0.0) -> float:
        """
        Calculate present value of an annuity in arrears whose payments fall
        at a compound rate.

        Args:
            periods: Number of periods
            payment: First payment amount
            decrease_rate: Compound rate of decrease per period (0.05 for 5%)

        Returns:
            Present value of decreasing annuity
        """
        if not 0 <= decrease_rate <= 1:
            raise ValueError("decrease_rate must be between 0 and 1")
        return self.increasing_annuity(periods, payment, -decrease_rate)

    def arithmetic_increasing_annuity(self, periods: int, payment: float = 1.0) -> float:
        """
        Present value of (Ia)n: payments of 1, 2, ..., n times ``payment`` in arrears.
        """
        t = np.arange(1, periods + 1)
        return float(payment * np.sum(t * self.v ** t))

    def arithmetic_decreasing_annuity(self, periods: int, payment: float = 1.0) -> float:
        """
        Present value of (Da)n: payments of n, n-1, ..., 1 times ``payment`` in arrears.
        """
        t = np.arange(1, periods + 1)
        return float(payment * np.sum((periods + 1 - t) * self.v ** t))

    # ------------------------------------------------------------------
    # Single life annuities
    # ------------------------------------------------------------------
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
        self._require_table()
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
        self._require_table()
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
        self._require_table()
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
        self._require_table()
        return payment * self.sf.annuity_due(x, n)

    def deferred_life_annuity(self,
                             x: int,
                             u: int,
                             payment: float = 1.0,
                             due: bool = False) -> float:
        """
        Calculate present value of deferred life annuity.

        Args:
            x: Age
            u: Deferment period
            payment: Annual payment amount
            due: True if the first payment is at the end of the deferred
                period, False if it is one year later

        Returns:
            Present value of deferred life annuity
        """
        self._require_table()
        endowment = self.sf.pure_endowment(x, u)
        if endowment == 0:
            return 0.0
        annuity = self.sf.annuity_due(x + u) if due else self.sf.annuity_immediate(x + u)
        return payment * endowment * annuity

    def guaranteed_annuity(self,
                          x: int,
                          n: int,
                          payment: float = 1.0) -> float:
        """
        Calculate present value of a life annuity in arrears with a guaranteed
        period: payments are certain for n years and continue for life after.

        Args:
            x: Age
            n: Guarantee period
            payment: Annual payment amount

        Returns:
            Present value of guaranteed annuity
        """
        self._require_table()
        return self.immediate_annuity(n, payment) + self.deferred_life_annuity(x, n, payment)

    def monthly_life_annuity(self,
                             x: int,
                             annual_payment: float = 1.0,
                             n: Optional[int] = None,
                             due: bool = True,
                             frequency: int = 12) -> float:
        """
        Present value of a life annuity paid in instalments during the year
        (monthly by default), as is usual for pensions.

        Args:
            x: Age
            annual_payment: Total amount paid per year
            n: Term in years (None for whole life)
            due: True for payments in advance, False for payments in arrears
            frequency: Number of payments per year

        Returns:
            Present value using Woolhouse's two-term approximation
        """
        self._require_table()
        return annual_payment * self.sf.annuity_mthly(x, frequency, n, due)

    def annuity_certain_with_life_contingency(self,
                                            x: int,
                                            n: int,
                                            payment: float = 1.0) -> float:
        """
        Calculate present value of an annuity paid for at most n years and
        only while (x) is alive.

        Args:
            x: Age
            n: Maximum number of years
            payment: Annual payment amount

        Returns:
            Present value of the temporary life annuity in arrears
        """
        return self.temporary_life_annuity_immediate(x, n, payment)

    # ------------------------------------------------------------------
    # Two lives (independent)
    # ------------------------------------------------------------------
    def _joint_immediate(self, x: int, y: int, table_y: Optional[MortalityTable]) -> float:
        self._require_table()
        return self.sf.joint_annuity_due(x, y, None, table_y) - 1.0

    def _single_immediate(self, age: int, table: Optional[MortalityTable]) -> float:
        self._require_table()
        sf = self.sf if table is None else SurvivalFunctions(table, self.i)
        return sf.annuity_immediate(age)

    def joint_life_annuity(self,
                          x: int,
                          y: int,
                          payment: float = 1.0,
                          table_y: Optional[MortalityTable] = None) -> float:
        """
        Calculate present value of a joint life annuity in arrears, paid
        while both lives are alive.

        Args:
            x: Age of first life
            y: Age of second life
            payment: Annual payment amount
            table_y: Mortality table for the second life (defaults to the
                table of the first life)

        Returns:
            Present value of joint life annuity
        """
        return payment * self._joint_immediate(x, y, table_y)

    def last_survivor_annuity(self,
                              x: int,
                              y: int,
                              payment: float = 1.0,
                              table_y: Optional[MortalityTable] = None) -> float:
        """
        Calculate present value of a last survivor annuity in arrears, paid
        while at least one of the two lives is alive.

        Args:
            x: Age of first life
            y: Age of second life
            payment: Annual payment amount
            table_y: Mortality table for the second life

        Returns:
            ax + ay - axy
        """
        value = (self._single_immediate(x, None) + self._single_immediate(y, table_y)
                 - self._joint_immediate(x, y, table_y))
        return payment * value

    def reversionary_annuity(self,
                             x: int,
                             y: int,
                             payment: float = 1.0,
                             table_y: Optional[MortalityTable] = None) -> float:
        """
        Calculate present value of a reversionary annuity in arrears, paid to
        (y) after the death of (x), for example a spouse's pension.

        Args:
            x: Age of the life whose death starts the annuity
            y: Age of the annuitant
            payment: Annual payment amount
            table_y: Mortality table for the annuitant

        Returns:
            ay - axy
        """
        value = self._single_immediate(y, table_y) - self._joint_immediate(x, y, table_y)
        return payment * value

    def contingent_annuity(self,
                          x: int,
                          y: int,
                          payment: float = 1.0,
                          table_y: Optional[MortalityTable] = None) -> float:
        """
        Calculate present value of contingent annuity in arrears, paid
        while (x) is alive and (y) has died.

        Args:
            x: Age of annuitant
            y: Age of the life whose death starts the annuity
            payment: Annual payment amount
            table_y: Mortality table for (y)

        Returns:
            ax - axy
        """
        value = self._single_immediate(x, None) - self._joint_immediate(x, y, table_y)
        return payment * value

    # ------------------------------------------------------------------
    # Drawdown
    # ------------------------------------------------------------------
    def annuity_with_withdrawal(self,
                               principal: float,
                               withdrawal_rate: float,
                               periods: Optional[int] = None) -> dict:
        """
        Project a fund from which a level amount is withdrawn at the end of
        each year (income drawdown / living annuity).

        The annual withdrawal is ``principal * withdrawal_rate`` and the fund
        earns the calculator's interest rate. The last withdrawal is limited
        to the money left in the fund.

        Args:
            principal: Initial principal amount
            withdrawal_rate: Annual withdrawal as a proportion of the initial principal
            periods: Number of years to project (None to project until the
                fund runs out, up to 100 years)

        Returns:
            Dictionary with the annual payment, remaining principal, the year
            in which the fund runs out (None if it does not) and the schedule
        """
        if principal < 0 or withdrawal_rate < 0:
            raise ValueError("principal and withdrawal_rate must be non-negative")

        payment = principal * withdrawal_rate
        horizon = periods if periods is not None else 100
        remaining = principal
        exhausted_in = None

        schedule = []
        for t in range(1, horizon + 1):
            starting = remaining
            interest = starting * self.i
            withdrawal = min(payment, starting + interest)
            remaining = starting + interest - withdrawal
            schedule.append({
                'period': t,
                'starting_balance': starting,
                'interest': interest,
                'withdrawal': withdrawal,
                'ending_balance': remaining
            })
            if payment > 0 and remaining <= 1e-9 * max(principal, 1.0):
                remaining = 0.0
                exhausted_in = t
                break

        return {
            'annual_payment': payment,
            'remaining_principal': remaining,
            'periods': periods if periods is not None else (exhausted_in or 'perpetual'),
            'exhausted_in': exhausted_in,
            'schedule': schedule
        }
