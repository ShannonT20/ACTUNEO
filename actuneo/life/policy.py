"""
Life Policies with Expenses

A :class:`LifePolicy` values one life assurance policy from first principles:
benefits that may vary from year to year, premiums payable annually in
advance, and expenses. From it come the gross and net premiums, prospective
and retrospective reserves, the distribution of the insurer's loss, and
mortality profit.

Conventions
-----------
- Policy years are numbered from 1. A death benefit for year k is paid at the
  end of that year, or at the moment of death if ``timing="immediate"``,
  which is valued with the usual approximation of half a year's interest,
  ``(1 + i) ** 0.5``.
- Premiums are paid at the start of each year while the life is alive.
- Renewal expenses are paid at the start of each policy year from the second.
- For a select life, pass ``select_table.table_for(age)`` as the table.
"""

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..mortality import MortalityTable


class Expenses:
    """Expense assumptions for a life policy."""

    def __init__(self,
                 initial: float = 0.0,
                 initial_premium: float = 0.0,
                 renewal: float = 0.0,
                 renewal_premium: float = 0.0,
                 renewal_inflation: float = 0.0,
                 claim: float = 0.0,
                 claim_benefit: float = 0.0,
                 maturity_claim: Optional[float] = None):
        """
        Args:
            initial: Fixed expense at the start of the policy
            initial_premium: Initial expense or commission as a proportion of
                the first premium
            renewal: Fixed expense at the start of each policy year from the
                second
            renewal_premium: Renewal expense or commission as a proportion of
                each premium after the first
            renewal_inflation: Annual rate at which the fixed renewal expense
                increases after the second year
            claim: Fixed expense when a death claim is paid
            claim_benefit: Claim expense as a proportion of the benefit paid
            maturity_claim: Fixed expense when a survival benefit is paid
                (defaults to ``claim``)
        """
        self.initial = float(initial)
        self.initial_premium = float(initial_premium)
        self.renewal = float(renewal)
        self.renewal_premium = float(renewal_premium)
        self.renewal_inflation = float(renewal_inflation)
        self.claim = float(claim)
        self.claim_benefit = float(claim_benefit)
        self.maturity_claim = self.claim if maturity_claim is None else float(maturity_claim)


def simple_bonus_benefits(sum_assured: float, bonus_rate: float, term: int,
                          past_bonus: float = 0.0) -> tuple:
    """
    Benefits of a with-profits policy with simple reversionary bonuses that
    vest at the end of each policy year.

    Args:
        sum_assured: Basic sum assured
        bonus_rate: Annual bonus as a proportion of the basic sum assured
        term: Number of policy years to project
        past_bonus: Bonuses already attaching at the start

    Returns:
        ``(death_benefits, maturity_benefit)``: the benefit on death in each
        year (which has no bonus for the year of death) and at the end
    """
    years = np.arange(term)
    start = sum_assured + past_bonus
    return start + sum_assured * bonus_rate * years, start + sum_assured * bonus_rate * term


def compound_bonus_benefits(sum_assured: float, bonus_rate: float, term: int,
                            past_bonus: float = 0.0) -> tuple:
    """
    Benefits of a with-profits policy with compound reversionary bonuses
    that vest at the end of each policy year.

    Args:
        sum_assured: Basic sum assured
        bonus_rate: Annual bonus as a proportion of the sum assured and
            attaching bonuses
        term: Number of policy years to project
        past_bonus: Bonuses already attaching at the start

    Returns:
        ``(death_benefits, maturity_benefit)``
    """
    start = sum_assured + past_bonus
    return start * (1 + bonus_rate) ** np.arange(term), start * (1 + bonus_rate) ** term


class LifePolicy:
    """One life assurance policy valued with expenses."""

    def __init__(self,
                 table: MortalityTable,
                 age: int,
                 interest: float,
                 term: Optional[int] = None,
                 death_benefit: Union[float, Sequence[float]] = 0.0,
                 survival_benefit: float = 0.0,
                 premium_term: Optional[int] = None,
                 timing: str = "end_of_year",
                 expenses: Optional[Expenses] = None,
                 in_force: bool = False):
        """
        Args:
            table: Mortality table (for a select life, the table for that life)
            age: Age at the start of the policy, or the present age of a
                policy already in force
            interest: Effective annual rate of interest
            term: Policy term in years (None for whole life)
            death_benefit: Benefit on death: one amount, or one for each
                policy year
            survival_benefit: Benefit paid on survival to the end of the term
            premium_term: Years for which premiums are payable (defaults to
                the policy term)
            timing: "end_of_year" or "immediate" for the death benefit
            expenses: Expense assumptions (none if omitted)
            in_force: True to value the remainder of a policy that is already
                in force: ``age`` and ``term`` are then the present age and
                the outstanding term, there are no initial expenses, and
                every future premium and policy year bears renewal expenses
        """
        if timing not in ("end_of_year", "immediate"):
            raise ValueError("timing must be 'end_of_year' or 'immediate'")
        if interest <= -1:
            raise ValueError("interest must be greater than -1")
        start = table._index(age)
        remaining = len(table.ages) - start
        self.whole_life = term is None
        self.n = remaining if term is None else int(term)
        if self.n < 1 or self.n > remaining:
            raise ValueError("the term must be at least one year and within the mortality table")
        if survival_benefit and self.whole_life:
            raise ValueError("a survival benefit needs a fixed term")

        self.table = table
        self.age = int(age)
        self.i = float(interest)
        self.v = 1 / (1 + self.i)
        self.premium_term = self.n if premium_term is None else int(premium_term)
        if not 1 <= self.premium_term <= self.n:
            raise ValueError("premium_term must be between 1 and the policy term")
        self.timing = timing
        self.expenses = expenses or Expenses()
        self.survival_benefit = float(survival_benefit)

        benefit = np.atleast_1d(np.asarray(death_benefit, dtype=float))
        if benefit.size == 1:
            benefit = np.full(self.n, benefit[0])
        if benefit.shape != (self.n,):
            raise ValueError(f"death_benefit needs one value for each of the {self.n} years")
        self.death_benefit = benefit

        q = table.qx_values[start:start + self.n].copy()
        if self.whole_life:
            q[-1] = 1.0  # close the table
        self._q = q
        self._kpx = np.concatenate(([1.0], np.cumprod(1 - q)))  # k = 0 .. n
        self._acceleration = (1 + self.i) ** 0.5 if timing == "immediate" else 1.0

        e = self.expenses
        self.in_force = bool(in_force)
        renewal = np.zeros(self.n)
        if self.in_force:
            renewal[:] = e.renewal * (1 + e.renewal_inflation) ** np.arange(self.n)
        elif self.n > 1:
            renewal[1:] = e.renewal * (1 + e.renewal_inflation) ** np.arange(self.n - 1)
        self._renewal = renewal

    # ------------------------------------------------------------------
    # Building blocks, valued at integer duration t for a life then alive
    # ------------------------------------------------------------------
    def _check_duration(self, t) -> int:
        if t != int(t) or not 0 <= t <= self.n:
            raise ValueError(f"duration must be a whole number from 0 to {self.n}")
        return int(t)

    def _survival(self, t: int) -> np.ndarray:
        """Probability of being alive at the start of each year k >= t, given alive at t."""
        return self._kpx[t:self.n] / self._kpx[t]

    def _discount(self, t: int) -> np.ndarray:
        return self.v ** np.arange(self.n - t)

    def epv_death_benefit(self, t: int = 0, with_expenses: bool = False) -> float:
        """Expected present value of the death benefits, at duration t."""
        t = self._check_duration(t)
        if t == self.n:
            return 0.0
        amount = self.death_benefit[t:]
        if with_expenses:
            pays = amount != 0
            amount = amount * (1 + self.expenses.claim_benefit) + self.expenses.claim * pays
        deaths = self._survival(t) * self._q[t:]
        return float(self._acceleration * np.sum(self.v * self._discount(t) * deaths * amount))

    def epv_survival_benefit(self, t: int = 0, with_expenses: bool = False) -> float:
        """Expected present value of the survival benefit, at duration t."""
        t = self._check_duration(t)
        if not self.survival_benefit:
            return 0.0
        amount = self.survival_benefit
        if with_expenses:
            amount = amount * (1 + self.expenses.claim_benefit) + self.expenses.maturity_claim
        return float(amount * self.v ** (self.n - t) * self._kpx[self.n] / self._kpx[t])

    def epv_benefits(self, t: int = 0, with_expenses: bool = False) -> float:
        """Expected present value of all benefits, at duration t."""
        return self.epv_death_benefit(t, with_expenses) + self.epv_survival_benefit(t, with_expenses)

    def premium_annuity(self, t: int = 0) -> float:
        """Expected present value of 1 at the start of each remaining premium year."""
        t = self._check_duration(t)
        if t >= self.premium_term:
            return 0.0
        years = self.premium_term - t
        return float(np.sum(self._discount(t)[:years] * self._survival(t)[:years]))

    def epv_fixed_expenses(self, t: int = 0) -> float:
        """Expected present value of the fixed initial and renewal expenses, at duration t."""
        t = self._check_duration(t)
        if t == self.n:
            return 0.0
        value = float(np.sum(self._discount(t) * self._survival(t) * self._renewal[t:]))
        return value + (self.expenses.initial if t == 0 and not self.in_force else 0.0)

    def _premium_loading(self, t: int) -> float:
        """Expected present value of premium-related expenses per unit of premium."""
        e = self.expenses
        annuity = self.premium_annuity(t)
        if t == 0 and not self.in_force:
            return e.initial_premium + e.renewal_premium * (annuity - 1)
        return e.renewal_premium * annuity

    # ------------------------------------------------------------------
    # Premiums
    # ------------------------------------------------------------------
    def net_premium(self) -> float:
        """Level annual premium that covers the benefits only."""
        return self.epv_benefits(0) / self.premium_annuity(0)

    def gross_premium(self, extra: float = 0.0) -> float:
        """
        Level annual premium by the equivalence principle: the expected
        present value of premiums equals that of benefits and expenses.

        Args:
            extra: Additional amount to be covered at outset, for example a
                margin for risk expressed as a present value
        """
        outgo = self.epv_benefits(0, with_expenses=True) + self.epv_fixed_expenses(0) + extra
        income_per_unit = self.premium_annuity(0) - self._premium_loading(0)
        if income_per_unit <= 0:
            raise ValueError("premium-related expenses use up the whole premium")
        return outgo / income_per_unit

    # ------------------------------------------------------------------
    # Reserves
    # ------------------------------------------------------------------
    def gross_premium_reserve(self, t: int, premium: Optional[float] = None) -> float:
        """
        Prospective gross premium reserve at duration t, just before the
        premium then due: future benefits and expenses less future premiums.

        Args:
            t: Complete years the policy has been in force
            premium: Gross annual premium (calculated on this basis if omitted)
        """
        t = self._check_duration(t)
        premium = self.gross_premium() if premium is None else premium
        return (self.epv_benefits(t, with_expenses=True) + self.epv_fixed_expenses(t)
                - premium * (self.premium_annuity(t) - self._premium_loading(t)))

    def net_premium_reserve(self, t: int, premium: Optional[float] = None) -> float:
        """Prospective net premium reserve at duration t: benefits less net premiums."""
        t = self._check_duration(t)
        premium = self.net_premium() if premium is None else premium
        return self.epv_benefits(t) - premium * self.premium_annuity(t)

    def retrospective_reserve(self, t: int, premium: Optional[float] = None,
                              with_expenses: bool = True) -> float:
        """
        Retrospective reserve at duration t: premiums received less benefits
        and expenses paid, accumulated with interest and survivorship.

        It equals the prospective reserve when the premium was calculated on
        the same basis.
        """
        t = self._check_duration(t)
        if premium is None:
            premium = self.gross_premium() if with_expenses else self.net_premium()
        if t == 0:
            return 0.0
        k = np.arange(t)
        alive = self._kpx[:t]
        discount = self.v ** k
        years = min(t, self.premium_term)
        premiums = premium * np.sum(discount[:years] * alive[:years])
        benefit = self.death_benefit[:t]
        expenses = 0.0
        if with_expenses:
            e = self.expenses
            benefit = benefit * (1 + e.claim_benefit) + e.claim * (benefit != 0)
            expenses = (e.initial + np.sum(discount * alive * self._renewal[:t])
                        + premium * (e.initial_premium
                                     + e.renewal_premium * np.sum(discount[1:years]
                                                                  * alive[1:years])))
        claims = self._acceleration * np.sum(self.v * discount * alive * self._q[:t] * benefit)
        return float((premiums - claims - expenses) / (self.v ** t * self._kpx[t]))

    # ------------------------------------------------------------------
    # The insurer's loss as a random variable
    # ------------------------------------------------------------------
    def loss_distribution(self, premium: float, with_expenses: bool = True) -> pd.DataFrame:
        """
        Distribution of the present value of the insurer's loss at outset:
        benefits and expenses paid less premiums received.

        Args:
            premium: Annual premium charged
            with_expenses: Include expenses in the loss

        Returns:
            DataFrame with one row for death in each policy year and one for
            survival to the end, giving the probability and the loss
        """
        e = self.expenses if with_expenses else Expenses()
        k = np.arange(self.n)
        discount = self.v ** k
        premium_years = k < self.premium_term
        per_premium = np.ones(self.n)
        per_premium[1:] -= e.renewal_premium
        per_premium[0] -= e.initial_premium
        net_income = np.cumsum(discount * premium_years * premium * per_premium)
        renewal = self._renewal if with_expenses else np.zeros(self.n)
        fixed = e.initial + np.cumsum(discount * renewal)

        benefit = self.death_benefit * (1 + e.claim_benefit) + e.claim * (self.death_benefit != 0)
        loss_on_death = self._acceleration * benefit * self.v ** (k + 1) + fixed - net_income
        rows = {
            "outcome": [f"death in year {j + 1}" for j in k],
            "probability": list(self._kpx[:self.n] * self._q),
            "loss": list(loss_on_death),
        }
        survive = self._kpx[self.n]
        if survive > 0:
            maturity = self.survival_benefit
            if maturity:
                maturity = maturity * (1 + e.claim_benefit) + e.maturity_claim
            rows["outcome"].append("survival")
            rows["probability"].append(survive)
            rows["loss"].append(maturity * self.v ** self.n + fixed[-1] - net_income[-1])
        return pd.DataFrame(rows)

    def expected_loss(self, premium: float, with_expenses: bool = True) -> float:
        """Expected present value of the loss at outset."""
        table = self.loss_distribution(premium, with_expenses)
        return float(np.sum(table["probability"] * table["loss"]))

    def loss_standard_deviation(self, premium: float, with_expenses: bool = True) -> float:
        """Standard deviation of the present value of the loss at outset."""
        table = self.loss_distribution(premium, with_expenses)
        mean = np.sum(table["probability"] * table["loss"])
        return float(np.sqrt(np.sum(table["probability"] * (table["loss"] - mean) ** 2)))

    def benefit_standard_deviation(self) -> float:
        """Standard deviation of the present value of the benefits alone."""
        return self.loss_standard_deviation(0.0, with_expenses=False)

    def probability_of_loss(self, premium: float, with_expenses: bool = True) -> float:
        """Probability that the policy makes a loss at the given premium."""
        table = self.loss_distribution(premium, with_expenses)
        return float(table.loc[table["loss"] > 1e-9, "probability"].sum())

    def premium_for_loss_probability(self, probability: float,
                                     with_expenses: bool = True) -> float:
        """
        Smallest annual premium at which the probability of a loss on the
        policy is no more than the given level.
        """
        if not 0 <= probability < 1:
            raise ValueError("probability must be from 0 to less than 1")
        from scipy import optimize
        zero = self.loss_distribution(0.0, with_expenses)
        one = self.loss_distribution(1.0, with_expenses)
        slope = (zero["loss"] - one["loss"]).to_numpy()  # loss falls by this per unit of premium
        with np.errstate(divide="ignore", invalid="ignore"):
            break_even = np.where(slope > 0, zero["loss"].to_numpy() / slope, np.inf)
        for candidate in np.sort(np.unique(break_even[np.isfinite(break_even)])):
            if self.probability_of_loss(candidate, with_expenses) <= probability + 1e-12:
                return float(candidate)
        high = float(np.max(break_even[np.isfinite(break_even)])) * 2 + 1
        return float(optimize.brentq(
            lambda g: self.probability_of_loss(g, with_expenses) - probability - 1e-12, 0, high))

    # ------------------------------------------------------------------
    # Mortality profit
    # ------------------------------------------------------------------
    def death_strain_at_risk(self, t: int, premium: Optional[float] = None,
                             basis: str = "net") -> float:
        """
        Death strain at risk for the policy year starting at duration t: the
        extra amount the insurer must find if the life dies in the year, the
        death benefit less the reserve that would otherwise be held at the
        end of the year.

        Args:
            t: Duration at the start of the year
            premium: Premium the reserves are based on
            basis: "net" for net premium reserves, "gross" for gross premium
                reserves
        """
        t = self._check_duration(t)
        if t >= self.n:
            raise ValueError("there is no policy year starting at that duration")
        if basis == "net":
            end_reserve = self.net_premium_reserve(t + 1, premium)
            benefit = self.death_benefit[t]
        elif basis == "gross":
            end_reserve = self.gross_premium_reserve(t + 1, premium)
            benefit = (self.death_benefit[t] * (1 + self.expenses.claim_benefit)
                       + self.expenses.claim * (self.death_benefit[t] != 0))
        else:
            raise ValueError("basis must be 'net' or 'gross'")
        return float(benefit * self._acceleration - end_reserve)

    def mortality_profit(self, lives: float, deaths: float, t: int,
                         premium: Optional[float] = None, basis: str = "net") -> dict:
        """
        Mortality profit for one policy year on a group of identical policies.

        Args:
            lives: Policies in force at the start of the year
            deaths: Deaths during the year
            t: Duration at the start of the year
            premium: Premium the reserves are based on
            basis: "net" or "gross" reserves

        Returns:
            Dictionary with the death strain at risk per policy, the expected
            and actual death strain, and the mortality profit (expected less
            actual)
        """
        strain = self.death_strain_at_risk(t, premium, basis)
        expected = lives * self._q[t] * strain
        actual = deaths * strain
        return {
            "death_strain_at_risk": strain,
            "expected_death_strain": float(expected),
            "actual_death_strain": float(actual),
            "mortality_profit": float(expected - actual),
        }
