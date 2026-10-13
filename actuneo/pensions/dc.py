"""
Defined contribution schemes

:class:`DCProjection` projects one member's account to retirement and turns
it into a pension. :class:`DCFundValuation` sets the assets of a fund against
its members' accounts and reserves.
"""

from typing import Optional

import numpy as np
import pandas as pd

from ..mortality import MortalityTable
from .annuity import pension_annuity


class DCProjection:
    """
    Projection of a defined contribution account to retirement.

    Salary is reviewed once a year. Contributions are paid at the end of each
    payment period (monthly by default). The projection is deterministic: it
    shows what follows from the assumptions, not the range of outcomes.
    """

    def __init__(self,
                 age: int,
                 retirement_age: int,
                 salary: float,
                 fund: float = 0.0,
                 member_rate: float = 0.0,
                 employer_rate: float = 0.0,
                 salary_increase: float = 0.0,
                 investment_return: float = 0.0,
                 inflation: float = 0.0,
                 contribution_charge: float = 0.0,
                 asset_charge: float = 0.0,
                 contributions_per_year: int = 12):
        """
        Args:
            age: Present age
            retirement_age: Age at retirement
            salary: Present annual salary
            fund: Present value of the account
            member_rate: Member's contribution as a fraction of salary
            employer_rate: Employer's contribution as a fraction of salary
            salary_increase: Annual rate of salary increase
            investment_return: Annual investment return before charges
            inflation: Annual rate of price inflation, used to express
                amounts in today's money
            contribution_charge: Charge taken from each contribution, as a
                fraction of it
            asset_charge: Annual charge as a fraction of the fund
            contributions_per_year: Number of contributions paid each year
        """
        if retirement_age != int(retirement_age) or age != int(age):
            raise ValueError("age and retirement_age must be whole numbers")
        if retirement_age <= age:
            raise ValueError("retirement_age must be above the present age")
        if salary < 0 or fund < 0:
            raise ValueError("salary and fund must be non-negative")
        if member_rate < 0 or employer_rate < 0:
            raise ValueError("contribution rates must be non-negative")
        if not 0 <= contribution_charge < 1 or not 0 <= asset_charge < 1:
            raise ValueError("charges must be between 0 and 1")
        for name, rate in (("salary_increase", salary_increase),
                           ("investment_return", investment_return),
                           ("inflation", inflation)):
            if rate <= -1:
                raise ValueError(f"{name} must be greater than -1")
        if contributions_per_year != int(contributions_per_year) or contributions_per_year < 1:
            raise ValueError("contributions_per_year must be a positive whole number")
        self.age = int(age)
        self.retirement_age = int(retirement_age)
        self.salary = float(salary)
        self.fund = float(fund)
        self.member_rate = float(member_rate)
        self.employer_rate = float(employer_rate)
        self.salary_increase = float(salary_increase)
        self.investment_return = float(investment_return)
        self.inflation = float(inflation)
        self.contribution_charge = float(contribution_charge)
        self.asset_charge = float(asset_charge)
        self.contributions_per_year = int(contributions_per_year)

    @property
    def net_return(self) -> float:
        """Annual investment return after the charge on assets."""
        return (1 + self.investment_return) * (1 - self.asset_charge) - 1

    @property
    def years(self) -> int:
        """Years to retirement."""
        return self.retirement_age - self.age

    def _accumulation_factor(self) -> float:
        """Value at the end of a year of contributions of 1 spread over it."""
        p = self.contributions_per_year
        growth = (1 + self.net_return) ** (1 / p)
        if abs(growth - 1) < 1e-12:
            return 1.0
        return (growth ** p - 1) / (p * (growth - 1))

    def _run(self, total_rate: float, fund: float) -> pd.DataFrame:
        rows = []
        factor = self._accumulation_factor()
        for year in range(self.years):
            salary = self.salary * (1 + self.salary_increase) ** year
            contributions = total_rate * salary
            invested = contributions * (1 - self.contribution_charge)
            closing = fund * (1 + self.net_return) + invested * factor
            rows.append({
                "age": self.age + year,
                "salary": salary,
                "contributions": contributions,
                "fund_start": fund,
                "growth": closing - fund - invested,
                "fund_end": closing,
                "fund_end_real": closing / (1 + self.inflation) ** (year + 1),
            })
            fund = closing
        return pd.DataFrame(rows)

    def projection(self) -> pd.DataFrame:
        """
        The account year by year: salary, contributions paid, the fund at
        the start and end, growth after charges, and the closing fund in
        today's money.
        """
        return self._run(self.member_rate + self.employer_rate, self.fund)

    @property
    def fund_at_retirement(self) -> float:
        """Projected account at retirement."""
        return float(self.projection()["fund_end"].iloc[-1])

    @property
    def real_fund_at_retirement(self) -> float:
        """Projected account at retirement in today's money."""
        return float(self.projection()["fund_end_real"].iloc[-1])

    @property
    def final_salary(self) -> float:
        """Salary in the last year before retirement."""
        return self.salary * (1 + self.salary_increase) ** (self.years - 1)

    def annuity_factor(self, table: MortalityTable, interest: float, **options) -> float:
        """
        Cost at retirement of a pension of 1 a year, from a mortality table.
        ``options`` are those of :func:`~actuneo.pensions.pension_annuity`.
        """
        return pension_annuity(table, self.retirement_age, interest, **options)

    def pension(self, annuity_factor: float) -> float:
        """Annual pension bought with the account at a given annuity factor."""
        if annuity_factor <= 0:
            raise ValueError("annuity_factor must be positive")
        return self.fund_at_retirement / annuity_factor

    def replacement_ratio(self, annuity_factor: float) -> float:
        """Pension as a proportion of salary in the last year before retirement."""
        return self.pension(annuity_factor) / self.final_salary

    def required_contribution_rate(self, target_replacement_ratio: float,
                                   annuity_factor: float) -> float:
        """
        Total contribution rate (member and employer together) needed from
        now on to retire on a target proportion of final salary. It can be
        negative if the present fund is already more than enough.
        """
        if annuity_factor <= 0:
            raise ValueError("annuity_factor must be positive")
        needed = target_replacement_ratio * self.final_salary * annuity_factor
        from_fund = self.fund * (1 + self.net_return) ** self.years
        per_unit_rate = float(self._run(1.0, 0.0)["fund_end"].iloc[-1])
        return (needed - from_fund) / per_unit_rate

    def __repr__(self) -> str:
        return (f"DCProjection(age={self.age}, retirement_age={self.retirement_age}, "
                f"fund_at_retirement={self.fund_at_retirement:,.0f})")


class DCFundValuation:
    """
    Financial position of a defined contribution fund.

    The liabilities are the members' accounts, any pensions the fund pays
    itself, the investment reserve that smooths returns, contingency
    reserves and other liabilities. This follows the definition of actuarial
    surplus for a fund exempt from actuarial valuation in section 2 of
    Zimbabwe's Pensions and Provident Funds Act [Chapter 24:32].
    """

    def __init__(self,
                 assets: float,
                 member_accounts,
                 pensioner_liability: float = 0.0,
                 investment_reserve: float = 0.0,
                 contingency_reserve: float = 0.0,
                 other_liabilities: float = 0.0):
        """
        Args:
            assets: Fair value of the fund's assets
            member_accounts: Total of the members' accounts, or the
                individual accounts
            pensioner_liability: Value of pensions paid from the fund
            investment_reserve: Investment reserve account
            contingency_reserve: Contingency reserve accounts
            other_liabilities: Any other liabilities
        """
        self.assets = float(assets)
        self.member_accounts = float(np.sum(np.asarray(member_accounts, dtype=float)))
        self.pensioner_liability = float(pensioner_liability)
        self.investment_reserve = float(investment_reserve)
        self.contingency_reserve = float(contingency_reserve)
        self.other_liabilities = float(other_liabilities)
        if min(self.member_accounts, self.pensioner_liability, self.investment_reserve,
               self.contingency_reserve, self.other_liabilities) < 0:
            raise ValueError("accounts, reserves and liabilities must be non-negative")

    @property
    def reserves(self) -> float:
        """Investment and contingency reserves together."""
        return self.investment_reserve + self.contingency_reserve

    @property
    def total_liabilities(self) -> float:
        """Members' accounts, pensions, reserves and other liabilities."""
        return (self.member_accounts + self.pensioner_liability + self.reserves
                + self.other_liabilities)

    @property
    def surplus(self) -> float:
        """Assets less total liabilities (negative for a deficit)."""
        return self.assets - self.total_liabilities

    @property
    def funding_level(self) -> float:
        """Assets as a proportion of total liabilities."""
        total = self.total_liabilities
        return self.assets / total if total > 0 else float("nan")

    def summary(self) -> pd.Series:
        """The financial position of the fund, as set out in a valuation report."""
        return pd.Series({
            "Net assets": self.assets,
            "Active members": self.member_accounts,
            "Pensioners": self.pensioner_liability,
            "Reserves": self.reserves,
            "Other liabilities": self.other_liabilities,
            "Total actuarial liabilities": self.total_liabilities,
            "Surplus / (deficit)": self.surplus,
            "Funding level": self.funding_level,
        })

    def bonus_rate(self, target_funding_level: float = 1.0,
                   share_to_reserves: float = 0.0) -> float:
        """
        Bonus that can be added to members' accounts out of a surplus.

        Args:
            target_funding_level: Funding level to be left after the bonus
            share_to_reserves: Fraction of the distributable surplus moved
                to reserves instead of being paid as bonus

        Returns:
            The bonus as a proportion of members' accounts. It is negative
            when the assets fall short of the target, which is the reduction
            in accounts that would restore it.
        """
        if target_funding_level <= 0:
            raise ValueError("target_funding_level must be positive")
        if not 0 <= share_to_reserves <= 1:
            raise ValueError("share_to_reserves must be between 0 and 1")
        if self.member_accounts <= 0:
            raise ValueError("there are no member accounts to credit a bonus to")
        distributable = self.assets / target_funding_level - self.total_liabilities
        if distributable > 0:
            distributable *= 1 - share_to_reserves
        return distributable / self.member_accounts

    def __repr__(self) -> str:
        return (f"DCFundValuation(assets={self.assets:,.0f}, "
                f"liabilities={self.total_liabilities:,.0f}, "
                f"funding_level={self.funding_level:.1%})")
