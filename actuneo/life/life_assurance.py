"""
Life Assurance Calculations

Provides calculations for various life assurance products including
whole life, term life, endowment assurance, and related products.

All values are for a unit sum assured. Death benefits are payable at the end
of the year of death (``discrete=True``) or at the moment of death under a
uniform distribution of deaths (``discrete=False``). Premiums are payable
annually in advance.
"""

from typing import Optional
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

    @staticmethod
    def _timing(discrete: bool) -> str:
        return "end_of_year" if discrete else "immediate"

    def whole_life_assurance(self, x: int, discrete: bool = True) -> float:
        """
        Calculate net single premium for whole life assurance.

        Args:
            x: Age at entry
            discrete: True for benefit at end of year of death, False for
                benefit at the moment of death

        Returns:
            Net single premium for whole life assurance
        """
        return self.sf.assurance(x, timing=self._timing(discrete))

    def term_assurance(self, x: int, n: int, discrete: bool = True) -> float:
        """
        Calculate net single premium for term assurance.

        Args:
            x: Age at entry
            n: Term of assurance in years
            discrete: True for benefit at end of year of death, False for
                benefit at the moment of death

        Returns:
            Net single premium for n-year term assurance
        """
        return self.sf.assurance(x, n, timing=self._timing(discrete))

    def endowment_assurance(self, x: int, n: int, discrete: bool = True) -> float:
        """
        Calculate net single premium for endowment assurance.

        Args:
            x: Age at entry
            n: Term of assurance in years
            discrete: Timing of the death benefit, see :meth:`term_assurance`

        Returns:
            Net single premium for endowment assurance
        """
        # Endowment assurance = Term assurance + Pure endowment
        return self.term_assurance(x, n, discrete) + self.pure_endowment(x, n)

    def pure_endowment(self, x: int, n: int) -> float:
        """
        Calculate net single premium for pure endowment.

        Args:
            x: Age at entry
            n: Term in years

        Returns:
            Net single premium for pure endowment
        """
        return self.sf.pure_endowment(x, n)

    def deferred_assurance(self, x: int, u: int, n: Optional[int] = None) -> float:
        """
        Calculate net single premium for deferred assurance.

        Args:
            x: Age at entry
            u: Deferment period in years
            n: Assurance term in years after deferment (None for whole life)

        Returns:
            Net single premium for deferred assurance
        """
        # Pays if death occurs between age x+u and x+u+n
        endowment = self.sf.pure_endowment(x, u)
        if endowment == 0:
            return 0.0
        return endowment * self.sf.assurance(x + u, n)

    def temporary_life_annuity(self, x: int, n: int, due: bool = False) -> float:
        """
        Calculate present value of a temporary life annuity.

        Args:
            x: Age at entry
            n: Term in years
            due: True for payments in advance, False for payments in arrears

        Returns:
            Present value of temporary life annuity
        """
        return self.sf.annuity_due(x, n) if due else self.sf.annuity_immediate(x, n)

    def whole_life_annuity(self, x: int, due: bool = False) -> float:
        """
        Calculate present value of a whole life annuity.

        Args:
            x: Age at entry
            due: True for payments in advance, False for payments in arrears

        Returns:
            Present value of whole life annuity
        """
        return self.sf.annuity_due(x) if due else self.sf.annuity_immediate(x)

    def joint_life_assurance(self,
                             x: int,
                             y: int,
                             n: Optional[int] = None,
                             table_y: Optional[MortalityTable] = None) -> float:
        """
        Calculate net single premium for joint life assurance, payable on the
        first death of two independent lives.

        Args:
            x: Age of first life
            y: Age of second life
            n: Term in years (None for whole life)
            table_y: Mortality table for the second life (defaults to the
                table of the first life)

        Returns:
            Net single premium for joint life assurance
        """
        return self.sf.joint_assurance(x, y, n, table_y)

    def last_survivor_assurance(self,
                                x: int,
                                y: int,
                                table_y: Optional[MortalityTable] = None) -> float:
        """
        Calculate net single premium for a whole life last survivor assurance,
        payable on the second death of two independent lives.

        Args:
            x: Age of first life
            y: Age of second life
            table_y: Mortality table for the second life

        Returns:
            Ax + Ay - Axy
        """
        sf_y = self.sf if table_y is None else SurvivalFunctions(table_y, self.i)
        return self.sf.assurance(x) + sf_y.assurance(y) - self.sf.joint_assurance(x, y, None, table_y)

    def contingent_assurance(self,
                             x: int,
                             y: int,
                             n: Optional[int] = None,
                             table_y: Optional[MortalityTable] = None) -> float:
        """
        Calculate net single premium for contingent assurance, payable on the
        death of (x) only if (y) is then still alive.

        Args:
            x: Age of the life assured
            y: Age of the counter life
            n: Term in years (None for whole life)
            table_y: Mortality table for the counter life

        Returns:
            Net single premium for contingent assurance
        """
        return self.sf.contingent_assurance(x, y, n, table_y)

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
            annuity_factor: Annuity-due factor for premium payments
            gross_margin: Additional margin for expenses and profit

        Returns:
            Annual premium
        """
        if annuity_factor <= 0:
            raise ValueError("annuity_factor must be positive")

        net_annual = net_single_premium / annuity_factor
        return net_annual * (1 + gross_margin)

    def net_annual_premium(self,
                           x: int,
                           n: Optional[int] = None,
                           product: str = "whole_life",
                           premium_term: Optional[int] = None) -> float:
        """
        Net level annual premium, payable in advance, for a unit sum assured.

        Args:
            x: Age at entry
            n: Policy term (not needed for whole life)
            product: "whole_life", "term" or "endowment"
            premium_term: Premium paying term (defaults to the policy term)

        Returns:
            Net annual premium
        """
        return self.sf.net_annual_premium(x, n, product, premium_term)

    def reserve_whole_life(self, x: int, duration: int) -> float:
        """
        Calculate net premium reserve for whole life assurance.

        Args:
            x: Original age at entry
            duration: Number of years policy has been in force

        Returns:
            Reserve per unit sum assured at the given duration
        """
        # tVx = A_{x+t} - P_x * ä_{x+t}
        premium = self.sf.net_annual_premium(x)
        current_age = x + duration
        return self.sf.assurance(current_age) - premium * self.sf.annuity_due(current_age)

    def reserve_term(self, x: int, n: int, duration: int) -> float:
        """
        Calculate net premium reserve for term assurance.

        Args:
            x: Original age at entry
            n: Original term
            duration: Number of years policy has been in force

        Returns:
            Reserve per unit sum assured at the given duration
        """
        remaining_term = n - duration
        if remaining_term <= 0:
            return 0.0

        premium = self.sf.net_annual_premium(x, n, "term")
        current_age = x + duration
        return (self.sf.assurance(current_age, remaining_term)
                - premium * self.sf.annuity_due(current_age, remaining_term))

    def reserve_endowment(self, x: int, n: int, duration: int) -> float:
        """
        Calculate net premium reserve for endowment assurance.

        Args:
            x: Original age at entry
            n: Original term
            duration: Number of years policy has been in force

        Returns:
            Reserve per unit sum assured at the given duration (the maturity
            value of 1 at duration n, zero afterwards)
        """
        remaining_term = n - duration
        if remaining_term < 0:
            return 0.0  # Policy matured

        premium = self.sf.net_annual_premium(x, n, "endowment")
        current_age = x + duration
        return (self.sf.endowment_assurance(current_age, remaining_term)
                - premium * self.sf.annuity_due(current_age, remaining_term))
