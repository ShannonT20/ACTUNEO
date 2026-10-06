"""
Profit Testing

Projects the cashflows of a policy year by year and measures the profit that
emerges, for conventional and unit-linked business.

- The **profit vector** is the profit expected at the end of each year per
  policy in force at the start of that year.
- The **profit signature** is the profit vector multiplied by the
  probability of being in force at the start of each year: profit per policy
  sold.
- Discounting the signature at the **risk discount rate** gives the net
  present value; the rate that makes it zero is the internal rate of return.
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..finance.cashflows import Cashflows


def _per_year(values, n: int, name: str) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(0.0 if values is None else values, dtype=float))
    if arr.size == 1 and n != 1:
        arr = np.full(n, arr[0])
    if arr.shape != (n,):
        raise ValueError(f"{name} needs one value for each of the {n} years")
    return arr


def _decrements(death, surrender, basis: str):
    """Probabilities of death, surrender and staying in force during each year."""
    if basis == "dependent":
        return death, surrender, 1 - death - surrender
    if basis == "year_end":
        # Surrenders at the end of the year among those who survive it
        return death, (1 - death) * surrender, (1 - death) * (1 - surrender)
    raise ValueError("surrender_basis must be 'dependent' or 'year_end'")


class ProfitSignature:
    """Results of a profit test: profit vector, signature and summary measures."""

    def __init__(self, profit_vector: Sequence[float], stay_probabilities: Sequence[float],
                 premiums: Optional[Sequence[float]] = None):
        """
        Args:
            profit_vector: Profit at the end of each year per policy in force
                at the start of the year
            stay_probabilities: Probability of staying in force through each year
            premiums: Premium at the start of each year, for the profit margin
        """
        self.profit_vector = np.asarray(profit_vector, dtype=float)
        n = len(self.profit_vector)
        self.stay_probabilities = _per_year(stay_probabilities, n, "stay_probabilities")
        self.in_force = np.concatenate(([1.0], np.cumprod(self.stay_probabilities[:-1])))
        self.profit_signature = self.profit_vector * self.in_force
        self.premiums = None if premiums is None else _per_year(premiums, n, "premiums")

    def net_present_value(self, risk_discount_rate: float) -> float:
        """Present value of the profit signature at the risk discount rate."""
        years = np.arange(1, len(self.profit_signature) + 1)
        return float(np.sum(self.profit_signature * (1 + risk_discount_rate) ** -years))

    def internal_rate_of_return(self) -> float:
        """Rate of discount at which the net present value is zero."""
        years = np.arange(1, len(self.profit_signature) + 1)
        return Cashflows(years, self.profit_signature).internal_rate_of_return(guess=0.1)

    def present_value_of_premiums(self, risk_discount_rate: float) -> float:
        """Expected present value of the premiums at the risk discount rate."""
        if self.premiums is None:
            raise ValueError("premiums were not supplied")
        years = np.arange(len(self.premiums))
        return float(np.sum(self.premiums * self.in_force
                            * (1 + risk_discount_rate) ** -years))

    def profit_margin(self, risk_discount_rate: float) -> float:
        """Net present value as a proportion of the present value of premiums."""
        return (self.net_present_value(risk_discount_rate)
                / self.present_value_of_premiums(risk_discount_rate))

    def discounted_payback_period(self, risk_discount_rate: float) -> Optional[int]:
        """First year by which the accumulated present value of profit is not negative."""
        years = np.arange(1, len(self.profit_signature) + 1)
        running = np.cumsum(self.profit_signature * (1 + risk_discount_rate) ** -years)
        reached = np.flatnonzero(running >= -1e-9)
        return int(reached[0] + 1) if len(reached) else None

    def table(self) -> pd.DataFrame:
        """Profit vector, probability in force and profit signature by year."""
        return pd.DataFrame({
            "profit_vector": self.profit_vector,
            "in_force": self.in_force,
            "profit_signature": self.profit_signature,
        }, index=pd.RangeIndex(1, len(self.profit_vector) + 1, name="year"))


class ProfitTest(ProfitSignature):
    """
    Profit test of a conventional (non-linked) policy.

    Each year, per policy in force at the start::

        profit = (reserve at start + premium - expenses) * (1 + interest)
                 - expected death cost - expected surrender cost
                 - expected maturity cost - expected reserve at end
    """

    def __init__(self,
                 premiums,
                 expenses,
                 interest,
                 death_probability,
                 death_benefit,
                 reserves: Optional[Sequence[float]] = None,
                 surrender_probability=0.0,
                 surrender_value=0.0,
                 maturity_benefit: float = 0.0,
                 claim_expense: float = 0.0,
                 surrender_expense: Optional[float] = None,
                 surrender_basis: str = "dependent"):
        """
        Args:
            premiums: Premium at the start of each year
            expenses: Expenses at the start of each year
            interest: Rate of interest earned in each year
            death_probability: Probability of death in each year
            death_benefit: Benefit paid at the end of the year of death
            reserves: Reserve per policy in force at the start of year 1 and
                at the end of every year (n + 1 values). Zero if omitted.
            surrender_probability: Probability of surrender in each year
            surrender_value: Amount paid on surrender at the end of each year
            maturity_benefit: Amount paid to policies in force at the end of
                the last year
            claim_expense: Expense of paying a death claim
            surrender_expense: Expense of paying a surrender (defaults to
                ``claim_expense``)
            surrender_basis: "dependent" if the death and surrender
                probabilities both apply to policies in force at the start
                of the year; "year_end" if surrenders happen at the end of
                the year among those who have survived it
        """
        premiums = np.atleast_1d(np.asarray(premiums, dtype=float))
        n = len(premiums)
        expenses = _per_year(expenses, n, "expenses")
        interest = _per_year(interest, n, "interest")
        q = _per_year(death_probability, n, "death_probability")
        w = _per_year(surrender_probability, n, "surrender_probability")
        benefit = _per_year(death_benefit, n, "death_benefit")
        surrender = _per_year(surrender_value, n, "surrender_value")
        if reserves is None:
            reserves = np.zeros(n + 1)
        reserves = np.asarray(reserves, dtype=float)
        if reserves.shape != (n + 1,):
            raise ValueError(f"reserves needs {n + 1} values: the start and each year end")
        if surrender_expense is None:
            surrender_expense = claim_expense

        die, leave, stay = _decrements(q, w, surrender_basis)
        maturity = np.zeros(n)
        maturity[-1] = maturity_benefit

        self.columns = pd.DataFrame({
            "premium": premiums,
            "expenses": -expenses,
            "interest": (reserves[:-1] + premiums - expenses) * interest,
            "death_cost": -die * (benefit + claim_expense),
            "surrender_cost": -leave * (surrender + surrender_expense * (surrender != 0)),
            "maturity_cost": -stay * maturity,
            "change_in_reserve": reserves[:-1] - stay * reserves[1:],
        }, index=pd.RangeIndex(1, n + 1, name="year"))
        super().__init__(self.columns.sum(axis=1).to_numpy(), stay, premiums)

    def table(self) -> pd.DataFrame:
        """The cashflows of each year with the profit vector and signature."""
        return self.columns.join(super().table())


class UnitLinkedPolicy:
    """
    A unit-linked policy: premiums buy units, charges are taken from the
    unit fund, and the insurer's profit arises in the non-unit fund.
    """

    def __init__(self,
                 premiums,
                 allocation=1.0,
                 bid_offer_spread: float = 0.0,
                 unit_growth=0.0,
                 management_charge=0.0):
        """
        Args:
            premiums: Premium at the start of each year (zero after a single
                premium)
            allocation: Proportion of each premium used to buy units
            bid_offer_spread: Difference between the offer price at which
                units are bought and the bid price at which they are valued,
                as a proportion of the offer price
            unit_growth: Rate of growth of unit prices in each year
            management_charge: Charge at the end of each year as a proportion
                of the unit fund
        """
        self.premiums = np.atleast_1d(np.asarray(premiums, dtype=float))
        n = len(self.premiums)
        self.n = n
        self.allocation = _per_year(allocation, n, "allocation")
        self.bid_offer_spread = float(bid_offer_spread)
        growth = _per_year(unit_growth, n, "unit_growth")
        charge_rate = _per_year(management_charge, n, "management_charge")

        self.invested = self.premiums * self.allocation * (1 - self.bid_offer_spread)
        start = np.zeros(n)
        before_charge = np.zeros(n)
        charge = np.zeros(n)
        end = np.zeros(n)
        fund = 0.0
        for t in range(n):
            start[t] = fund + self.invested[t]
            before_charge[t] = start[t] * (1 + growth[t])
            charge[t] = before_charge[t] * charge_rate[t]
            end[t] = before_charge[t] - charge[t]
            fund = end[t]
        self.fund_start, self.fund_before_charge = start, before_charge
        self.management_charges, self.fund_end = charge, end

    def unit_fund(self) -> pd.DataFrame:
        """Projection of the unit fund year by year."""
        return pd.DataFrame({
            "premium": self.premiums,
            "invested": self.invested,
            "fund_at_start": self.fund_start,
            "fund_before_charge": self.fund_before_charge,
            "management_charge": self.management_charges,
            "fund_at_end": self.fund_end,
        }, index=pd.RangeIndex(1, self.n + 1, name="year"))

    def profit_test(self,
                    expenses,
                    interest,
                    death_probability,
                    surrender_probability=0.0,
                    minimum_death_benefit: float = 0.0,
                    surrender_penalty=0.0,
                    non_unit_reserves: Optional[Sequence[float]] = None,
                    surrender_basis: str = "dependent") -> ProfitSignature:
        """
        Profit test of the non-unit fund.

        Each year, per policy in force at the start, the non-unit fund
        receives the part of the premium not invested in units and the
        management charge, pays the expenses, earns interest, and bears the
        cost of any death benefit in excess of the unit fund.

        Args:
            expenses: Expenses and commission at the start of each year
            interest: Rate of interest earned on the non-unit fund
            death_probability: Probability of death in each year
            surrender_probability: Probability of surrender in each year
            minimum_death_benefit: Guaranteed minimum paid on death; the
                excess over the unit fund is paid from the non-unit fund
            surrender_penalty: Amount kept by the insurer on each surrender
            non_unit_reserves: Non-unit reserve per policy at the start of
                year 1 and at the end of every year (n + 1 values)
            surrender_basis: "dependent" or "year_end", as in
                :class:`ProfitTest`

        Returns:
            ProfitSignature, with the yearly cashflows in ``.columns``
        """
        n = self.n
        expenses = _per_year(expenses, n, "expenses")
        interest = _per_year(interest, n, "interest")
        q = _per_year(death_probability, n, "death_probability")
        w = _per_year(surrender_probability, n, "surrender_probability")
        penalty = _per_year(surrender_penalty, n, "surrender_penalty")
        reserves = np.zeros(n + 1) if non_unit_reserves is None \
            else np.asarray(non_unit_reserves, dtype=float)
        if reserves.shape != (n + 1,):
            raise ValueError(f"non_unit_reserves needs {n + 1} values")
        die, leave, stay = _decrements(q, w, surrender_basis)

        margin = self.premiums - self.invested
        columns = pd.DataFrame({
            "unallocated_premium": margin,
            "expenses": -expenses,
            "interest": (reserves[:-1] + margin - expenses) * interest,
            "management_charge": self.management_charges,
            "extra_death_cost": -die * np.maximum(minimum_death_benefit - self.fund_end, 0.0),
            "surrender_profit": leave * penalty,
            "change_in_reserve": reserves[:-1] - stay * reserves[1:],
        }, index=pd.RangeIndex(1, n + 1, name="year"))
        result = ProfitSignature(columns.sum(axis=1).to_numpy(), stay, self.premiums)
        result.columns = columns
        return result


def zeroise_negative_cashflows(cashflows: Sequence[float],
                               stay_probabilities: Sequence[float],
                               interest: float) -> pd.DataFrame:
    """
    Reserves that remove expected negative cashflows after the first year.

    An insurer should not expect to have to put more money into a policy
    after it is sold. Working back from the last year, a reserve is set up
    at the start of any year whose cashflow would otherwise be negative, big
    enough to cover it. Setting up the reserves reduces earlier profit.

    Args:
        cashflows: Expected cashflow at the end of each year per policy in
            force at the start of that year, before any reserves
        stay_probabilities: Probability of staying in force through each year
        interest: Rate of interest earned on the reserves

    Returns:
        DataFrame by year with the original cashflow, the reserve at the
        start and end of the year, and the profit vector after reserving
    """
    cashflows = np.asarray(cashflows, dtype=float)
    n = len(cashflows)
    stay = _per_year(stay_probabilities, n, "stay_probabilities")
    reserve = np.zeros(n + 1)  # reserve[t] is held at the end of year t
    for t in range(n, 1, -1):
        needed = (stay[t - 1] * reserve[t] - cashflows[t - 1]) / (1 + interest)
        reserve[t - 1] = max(0.0, needed)
    profit = cashflows + reserve[:-1] * (1 + interest) - stay * reserve[1:]
    return pd.DataFrame({
        "cashflow": cashflows,
        "reserve_at_start": reserve[:-1],
        "reserve_at_end": reserve[1:],
        "profit_vector": profit,
    }, index=pd.RangeIndex(1, n + 1, name="year"))
