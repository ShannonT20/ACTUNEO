"""
Reserve Calculations

Provides calculations for policy reserves, prospective and retrospective reserves,
and related actuarial valuations.
"""

import numpy as np
from typing import Optional, Union, List, Dict
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

    def prospective_reserve_whole_life(self,
                                     x: int,
                                     duration: int,
                                     annual_premium: float,
                                     sum_assured: float = 1000.0) -> float:
        """
        Calculate prospective reserve for whole life assurance.

        Args:
            x: Original age at entry
            duration: Number of years in force
            annual_premium: Annual premium amount
            sum_assured: Sum assured amount

        Returns:
            Prospective reserve
        """
        current_age = x + duration

        # Reserve = (A_{x+t} * sum_assured - P * ä_{x+t}) / ä_{x+t}
        # Where A is whole life assurance, ä is life annuity, P is annual premium

        remaining_assurance = self.sf.assurance(current_age) * sum_assured
        remaining_annuity = self.sf.annuity_immediate(current_age)

        if remaining_annuity == 0:
            return remaining_assurance - annual_premium

        reserve = (remaining_assurance - annual_premium * remaining_annuity) / remaining_annuity

        return max(0, reserve)  # Reserve cannot be negative

    def prospective_reserve_term(self,
                               x: int,
                               n: int,
                               duration: int,
                               annual_premium: float,
                               sum_assured: float = 1000.0) -> float:
        """
        Calculate prospective reserve for term assurance.

        Args:
            x: Original age at entry
            n: Original term
            duration: Number of years in force
            annual_premium: Annual premium amount
            sum_assured: Sum assured amount

        Returns:
            Prospective reserve
        """
        current_age = x + duration
        remaining_term = n - duration

        if remaining_term <= 0:
            return 0.0  # Policy expired

        remaining_assurance = self.sf.assurance(current_age, remaining_term) * sum_assured
        remaining_annuity = self.sf.annuity_immediate(current_age, remaining_term)

        if remaining_annuity == 0:
            return remaining_assurance - annual_premium

        reserve = (remaining_assurance - annual_premium * remaining_annuity) / remaining_annuity

        return max(0, reserve)

    def prospective_reserve_endowment(self,
                                    x: int,
                                    n: int,
                                    duration: int,
                                    annual_premium: float,
                                    sum_assured: float = 1000.0) -> float:
        """
        Calculate prospective reserve for endowment assurance.

        Args:
            x: Original age at entry
            n: Original term
            duration: Number of years in force
            annual_premium: Annual premium amount
            sum_assured: Sum assured amount

        Returns:
            Prospective reserve
        """
        current_age = x + duration
        remaining_term = n - duration

        if remaining_term <= 0:
            return 0.0  # Policy matured

        # Endowment reserve = [A_{x+t}:n¨ * SA + v^{n-t} * SA - P * ä_{x+t}:n¨] / ä_{x+t}:n¨

        # Calculate remaining endowment assurance
        term_assurance = self.sf.assurance(current_age, remaining_term) * sum_assured
        pure_endowment = self.sf.npx(current_age, remaining_term) * self.v ** remaining_term * sum_assured
        remaining_assurance = term_assurance + pure_endowment

        remaining_annuity = self.sf.annuity_immediate(current_age, remaining_term)

        if remaining_annuity == 0:
            return remaining_assurance - annual_premium

        reserve = (remaining_assurance - annual_premium * remaining_annuity) / remaining_annuity

        return max(0, reserve)

    def retrospective_reserve_whole_life(self,
                                       x: int,
                                       duration: int,
                                       annual_premium: float,
                                       sum_assured: float = 1000.0) -> float:
        """
        Calculate retrospective reserve for whole life assurance.

        Args:
            x: Original age at entry
            duration: Number of years in force
            annual_premium: Annual premium amount
            sum_assured: Sum assured amount

        Returns:
            Retrospective reserve
        """
        # Retrospective reserve = Accumulated premiums + interest - claims paid

        accumulated_premiums = 0.0
        interest_factor = 1.0

        for t in range(duration):
            age_at_premium = x + t
            survival_prob = self.sf.npx(x, t)  # Probability of surviving to pay premium
            accumulated_premiums += annual_premium * survival_prob * interest_factor
            interest_factor *= (1 + self.i)

        # Claims paid (simplified - only death claims, ignoring surrenders, etc.)
        claims_paid = 0.0
        interest_factor = 1.0

        for t in range(1, duration + 1):
            # Probability of dying in year t
            q_t = 1 - self.sf.npx(x, t) + self.sf.npx(x, t-1)
            claims_paid += q_t * sum_assured * interest_factor
            interest_factor *= (1 + self.i)

        reserve = accumulated_premiums - claims_paid

        return max(0, reserve)

    def net_level_premium_reserve(self,
                                x: int,
                                duration: int,
                                net_premium: float,
                                sum_assured: float = 1000.0) -> float:
        """
        Calculate net level premium reserve.

        Args:
            x: Original age at entry
            duration: Number of years in force
            net_premium: Net level annual premium
            sum_assured: Sum assured amount

        Returns:
            Net level premium reserve
        """
        # Reserve = v * [A_{x+t} * SA - P * ä_{x+t}]
        current_age = x + duration

        remaining_assurance = self.sf.assurance(current_age) * sum_assured
        remaining_annuity = self.sf.annuity_immediate(current_age)

        reserve = remaining_assurance - net_premium * remaining_annuity

        return max(0, reserve)

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
        Calculate Zillmerized reserve (reserve net of unamortized acquisition costs).

        Args:
            net_reserve: Net mathematical reserve
            initial_expenses: Initial acquisition expenses
            duration: Duration in years
            amortization_period: Period over which to amortize expenses

        Returns:
            Zillmerized reserve
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
