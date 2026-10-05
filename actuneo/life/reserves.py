"""
Reserve Calculations

Provides calculations for policy reserves, prospective and retrospective reserves,
and related actuarial valuations.

Reserves are per policy in force at the valuation date. Death benefits are
payable at the end of the year of death and premiums annually in advance.
"""

import numpy as np
from typing import Optional, List, Dict
from ..mortality import MortalityTable, SurvivalFunctions


class Reserves:
    """
    A class for calculating policy reserves and actuarial valuations.
    """

    def __init__(self,
                 mortality_table: MortalityTable,
                 interest_rate: float = 0.05,
                 expense_rate: float = 0.0,
                 profit_margin: float = 0.0):
        """
        Initialize Reserves calculator.

        Args:
            mortality_table: MortalityTable instance
            interest_rate: Annual interest rate for discounting
            expense_rate: Annual expense rate as percentage of premium
            profit_margin: Required profit margin
        """
        self.mt = mortality_table
        self.sf = SurvivalFunctions(mortality_table, interest_rate)
        self.i = interest_rate
        self.v = 1 / (1 + interest_rate)
        self.expense_rate = expense_rate
        self.profit_margin = profit_margin

    @staticmethod
    def _floor(reserve: float, floor_at_zero: bool) -> float:
        return max(0.0, reserve) if floor_at_zero else reserve

    def prospective_reserve_whole_life(self,
                                     x: int,
                                     duration: int,
                                     annual_premium: float,
                                     sum_assured: float = 1000.0,
                                     floor_at_zero: bool = False) -> float:
        """
        Calculate prospective reserve for whole life assurance.

        Args:
            x: Original age at entry
            duration: Number of years in force
            annual_premium: Annual premium amount
            sum_assured: Sum assured amount
            floor_at_zero: Replace a negative reserve by zero

        Returns:
            Prospective reserve
        """
        current_age = x + duration

        # Reserve = SA * A_{x+t} - P * ä_{x+t}
        benefits = self.sf.assurance(current_age) * sum_assured
        premiums = annual_premium * self.sf.annuity_due(current_age)

        return self._floor(benefits - premiums, floor_at_zero)

    def prospective_reserve_term(self,
                               x: int,
                               n: int,
                               duration: int,
                               annual_premium: float,
                               sum_assured: float = 1000.0,
                               floor_at_zero: bool = False) -> float:
        """
        Calculate prospective reserve for term assurance.

        Args:
            x: Original age at entry
            n: Original term
            duration: Number of years in force
            annual_premium: Annual premium amount
            sum_assured: Sum assured amount
            floor_at_zero: Replace a negative reserve by zero

        Returns:
            Prospective reserve
        """
        current_age = x + duration
        remaining_term = n - duration

        if remaining_term <= 0:
            return 0.0  # Policy expired

        benefits = self.sf.assurance(current_age, remaining_term) * sum_assured
        premiums = annual_premium * self.sf.annuity_due(current_age, remaining_term)

        return self._floor(benefits - premiums, floor_at_zero)

    def prospective_reserve_endowment(self,
                                    x: int,
                                    n: int,
                                    duration: int,
                                    annual_premium: float,
                                    sum_assured: float = 1000.0,
                                    floor_at_zero: bool = False) -> float:
        """
        Calculate prospective reserve for endowment assurance.

        Args:
            x: Original age at entry
            n: Original term
            duration: Number of years in force
            annual_premium: Annual premium amount
            sum_assured: Sum assured amount
            floor_at_zero: Replace a negative reserve by zero

        Returns:
            Prospective reserve (the sum assured at maturity, zero afterwards)
        """
        current_age = x + duration
        remaining_term = n - duration

        if remaining_term < 0:
            return 0.0  # Policy matured

        # Reserve = SA * A_{x+t:n-t} - P * ä_{x+t:n-t}
        benefits = self.sf.endowment_assurance(current_age, remaining_term) * sum_assured
        premiums = annual_premium * self.sf.annuity_due(current_age, remaining_term)

        return self._floor(benefits - premiums, floor_at_zero)

    def retrospective_reserve_whole_life(self,
                                       x: int,
                                       duration: int,
                                       annual_premium: float,
                                       sum_assured: float = 1000.0,
                                       floor_at_zero: bool = False) -> float:
        """
        Calculate retrospective reserve for whole life assurance.

        Premiums received less death claims paid, accumulated with interest
        and survivorship to the valuation date. It equals the prospective
        reserve when the premium is the net premium on the same basis.

        Args:
            x: Original age at entry
            duration: Number of years in force
            annual_premium: Annual premium amount
            sum_assured: Sum assured amount
            floor_at_zero: Replace a negative reserve by zero

        Returns:
            Retrospective reserve per surviving policy
        """
        if duration == 0:
            return 0.0

        endowment = self.sf.pure_endowment(x, duration)
        if endowment == 0:
            raise ValueError(f"No survivors from age {x} to duration {duration}")

        premiums = annual_premium * self.sf.annuity_due(x, duration)
        claims = sum_assured * self.sf.assurance(x, duration)

        return self._floor((premiums - claims) / endowment, floor_at_zero)

    def net_level_premium_reserve(self,
                                x: int,
                                duration: int,
                                net_premium: Optional[float] = None,
                                sum_assured: float = 1000.0,
                                floor_at_zero: bool = False) -> float:
        """
        Calculate net level premium reserve for whole life assurance.

        Args:
            x: Original age at entry
            duration: Number of years in force
            net_premium: Net level annual premium (calculated on the
                reserving basis if not given)
            sum_assured: Sum assured amount
            floor_at_zero: Replace a negative reserve by zero

        Returns:
            Net level premium reserve
        """
        if net_premium is None:
            net_premium = sum_assured * self.sf.net_annual_premium(x)
        return self.prospective_reserve_whole_life(
            x, duration, net_premium, sum_assured, floor_at_zero
        )

    def zillmer_reserve(self,
                        x: int,
                        n: int,
                        duration: int,
                        zillmer_rate: float,
                        sum_assured: float = 1000.0,
                        floor_at_zero: bool = False) -> float:
        """
        Calculate the Zillmerised net premium reserve for endowment assurance.

        Initial expenses of ``zillmer_rate`` per unit sum assured are spread
        over the premium term, which reduces the net premium reserve by
        ``zillmer_rate * ä_{x+t:n-t} / ä_{x:n}``.

        Args:
            x: Original age at entry
            n: Original term
            duration: Number of years in force
            zillmer_rate: Initial expense allowance per unit sum assured
            sum_assured: Sum assured amount
            floor_at_zero: Replace a negative reserve by zero

        Returns:
            Zillmerised reserve
        """
        remaining_term = n - duration
        if remaining_term < 0:
            return 0.0

        current_age = x + duration
        annuity_now = self.sf.annuity_due(current_age, remaining_term)
        annuity_start = self.sf.annuity_due(x, n)

        net_premium = self.sf.endowment_assurance(x, n) / annuity_start
        net_reserve = self.sf.endowment_assurance(current_age, remaining_term) - net_premium * annuity_now
        reserve = sum_assured * (net_reserve - zillmer_rate * annuity_now / annuity_start)

        return self._floor(reserve, floor_at_zero)

    def gross_reserve(self,
                     net_reserve: float,
                     duration: int,
                     expense_reserve: Optional[float] = None) -> float:
        """
        Calculate gross reserve including expenses and contingencies.

        Args:
            net_reserve: Net mathematical reserve
            duration: Duration in years
            expense_reserve: Additional expense reserve

        Returns:
            Gross reserve
        """
        if expense_reserve is None:
            # Estimate expense reserve as percentage of net reserve
            expense_reserve = net_reserve * self.expense_rate

        # Add profit margin
        profit_reserve = net_reserve * self.profit_margin

        gross_reserve = net_reserve + expense_reserve + profit_reserve

        return gross_reserve

    def reserve_release(self,
                       initial_reserve: float,
                       final_reserve: float,
                       duration: int) -> float:
        """
        Calculate reserve release over a period.

        Args:
            initial_reserve: Reserve at start of period
            final_reserve: Reserve at end of period
            duration: Duration in years

        Returns:
            Reserve release amount
        """
        return initial_reserve - final_reserve

    def terminal_reserve(self,
                        x: int,
                        n: int,
                        sum_assured: float = 1000.0) -> float:
        """
        Calculate terminal reserve (reserve at maturity).

        Args:
            x: Original age at entry
            n: Term of policy
            sum_assured: Sum assured amount

        Returns:
            Terminal reserve (should be sum assured for endowment)
        """
        # For endowment assurance, terminal reserve equals sum assured
        return sum_assured

    def reserve_distribution(self,
                           reserves: List[float],
                           total_portfolio_value: float) -> Dict[str, float]:
        """
        Analyze reserve distribution across portfolio.

        Args:
            reserves: List of individual policy reserves
            total_portfolio_value: Total portfolio value

        Returns:
            Dictionary with distribution statistics
        """
        reserves_array = np.array(reserves)

        distribution = {
            'mean_reserve': np.mean(reserves_array),
            'median_reserve': np.median(reserves_array),
            'min_reserve': np.min(reserves_array),
            'max_reserve': np.max(reserves_array),
            'total_reserves': np.sum(reserves_array),
            'reserve_to_portfolio_ratio': np.sum(reserves_array) / total_portfolio_value if total_portfolio_value > 0 else 0,
            'percentiles': {
                '25th': np.percentile(reserves_array, 25),
                '75th': np.percentile(reserves_array, 75),
                '90th': np.percentile(reserves_array, 90),
                '95th': np.percentile(reserves_array, 95)
            }
        }

        return distribution

    def zillmerized_reserve(self,
                           net_reserve: float,
                           initial_expenses: float,
                           duration: int,
                           amortization_period: int = 10) -> float:
        """
        Reduce a reserve by acquisition costs not yet recovered, writing the
        costs off in equal instalments over the amortization period.

        This is a straight-line approximation; see :meth:`zillmer_reserve`
        for the actuarial Zillmer adjustment.

        Args:
            net_reserve: Net mathematical reserve
            initial_expenses: Initial acquisition expenses
            duration: Duration in years
            amortization_period: Period over which to amortize expenses

        Returns:
            Reserve net of unamortized acquisition costs, not less than zero
        """
        if duration >= amortization_period:
            # Fully amortized
            return net_reserve

        # Amortize initial expenses over the period
        amortized_expenses = initial_expenses * (amortization_period - duration) / amortization_period

        zillmer_reserve = net_reserve - amortized_expenses

        return max(0, zillmer_reserve)

    def contingency_reserve(self,
                          base_reserve: float,
                          risk_factor: float = 0.05) -> float:
        """
        Calculate contingency reserve for adverse deviations.

        Args:
            base_reserve: Base mathematical reserve
            risk_factor: Risk factor for contingency

        Returns:
            Contingency reserve
        """
        return base_reserve * risk_factor
