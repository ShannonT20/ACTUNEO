"""
Survival Functions

Provides actuarial survival functions and life contingencies: survival and
death probabilities, pure endowments, life annuities, assurances and net
premiums, in International Actuarial Notation.

Conventions
-----------
- Benefits and annuity payments are for a unit amount.
- Assurances are payable at the end of the year of death unless
  ``timing="immediate"`` is requested, in which case the benefit is payable
  at the moment of death under a uniform distribution of deaths (UDD).
- Whole life functions need a table that ends with certain death. If the
  table stops at an age where qx < 1 it is closed at the last tabulated age
  (all survivors are assumed to die in that year) and a warning is issued.
"""

import warnings
import numpy as np
from numbers import Real
from typing import Optional
from .mortality_table import MortalityTable


class SurvivalFunctions:
    """
    A class for calculating various actuarial survival functions
    and life contingencies based on mortality tables.
    """

    def __init__(self, mortality_table: MortalityTable, interest_rate: float):
        """
        Initialize SurvivalFunctions with a mortality table and interest rate.

        Args:
            mortality_table: MortalityTable instance
            interest_rate: Annual interest rate for discounting (e.g. 0.10 for 10%)
        """
        if isinstance(interest_rate, bool) or not isinstance(interest_rate, Real):
            raise TypeError("interest_rate must be a real number (e.g. 0.10 for 10%).")
        if interest_rate <= -1:
            raise ValueError("interest_rate must be greater than -1.0.")

        self.mt = mortality_table
        self.i = float(interest_rate)
        self.v = 1 / (1 + self.i)  # Discount factor
        self.d = self.i * self.v  # Rate of discount
        self.delta = float(np.log1p(self.i))  # Force of interest

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _check_term(n, name: str = "n") -> int:
        if isinstance(n, bool) or n != int(n) or n < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        return int(n)

    def _horizon(self, x: int, n: Optional[int]) -> int:
        """Number of years from age x covered by the calculation."""
        start = self.mt._index(x)
        remaining = len(self.mt.ages) - start
        if n is None:
            return remaining
        n = self._check_term(n)
        if n > remaining:
            raise ValueError(
                f"Term of {n} years from age {x} runs past the end of mortality table "
                f"'{self.mt.name}' (last age {self.mt.max_age})"
            )
        return n

    def _kpx(self, x: int, years: int) -> np.ndarray:
        """Array of k-year survival probabilities for k = 0, ..., years."""
        start = self.mt._index(x)
        px = self.mt.px_values[start:start + years]
        return np.concatenate(([1.0], np.cumprod(px)))

    def _qx(self, x: int, years: int, whole_life: bool) -> np.ndarray:
        """Mortality rates at ages x, ..., x+years-1, closing the table for whole life."""
        start = self.mt._index(x)
        q = self.mt.qx_values[start:start + years].copy()
        if whole_life and not self.mt.is_closed:
            warnings.warn(
                f"Mortality table '{self.mt.name}' ends at age {self.mt.max_age} with "
                f"qx = {self.mt.qx_values[-1]:.4f} < 1. Whole life values assume all "
                f"survivors die between ages {self.mt.max_age} and {self.mt.omega}.",
                UserWarning,
                stacklevel=3,
            )
            q[-1] = 1.0
        return q

    def _immediate_factor(self) -> float:
        """UDD adjustment i/delta from end-of-year to immediate payment of claims."""
        if self.i == 0:
            return 1.0
        return self.i / self.delta

    # ------------------------------------------------------------------
    # Probabilities
    # ------------------------------------------------------------------
    def npx(self, x: int, n: int) -> float:
        """
        Calculate n-year survival probability: npx

        Args:
            x: Age
            n: Number of years

        Returns:
            Probability that (x) survives n years
        """
        return self.mt.npx(x, n)

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

    def deferred_qx(self, x: int, n: int, m: int = 1) -> float:
        """
        Calculate deferred mortality probability: n|m qx

        Args:
            x: Age
            n: Deferred period in years
            m: Number of years after deferment

        Returns:
            Probability that (x) survives n years and dies in the next m years
        """
        return self.mt.deferred_qx(x, n, m)

    def tpx(self, x: int, t: float, assumption: str = "udd") -> float:
        """
        Calculate t-year survival probability for fractional t.

        Args:
            x: Age
            t: Years (may be fractional)
            assumption: Fractional age assumption, "udd" (uniform distribution
                of deaths) or "cfm" (constant force of mortality)

        Returns:
            Probability that (x) survives t years
        """
        if t < 0:
            raise ValueError("t must be non-negative")
        if assumption not in ("udd", "cfm"):
            raise ValueError("assumption must be 'udd' or 'cfm'")

        n = int(np.floor(t))
        frac = t - n
        survival = self.npx(x, n)
        if frac == 0 or survival == 0:
            return survival
        if x + n > self.mt.max_age:
            return 0.0

        q_next = self.mt.qx_values[self.mt._index(x + n)]
        if assumption == "udd":
            return survival * (1 - frac * q_next)
        return survival * (1 - q_next) ** frac

    # ------------------------------------------------------------------
    # Pure endowment and annuities
    # ------------------------------------------------------------------
    def pure_endowment(self, x: int, n: int) -> float:
        """
        Calculate the actuarial present value of a pure endowment: nEx

        Args:
            x: Age
            n: Term in years

        Returns:
            v^n * npx
        """
        n = self._check_term(n)
        return self.v ** n * self.npx(x, n)

    def annuity_due(self, x: int, n: Optional[int] = None) -> float:
        """
        Calculate the actuarial present value of an annuity-due: äx or äx:n

        Payments of 1 are made at the start of each year while (x) is alive.

        Args:
            x: Age
            n: Term of annuity (None for whole life)

        Returns:
            Actuarial present value of annuity-due
        """
        years = self._horizon(x, n)
        if years == 0:
            return 0.0
        kpx = self._kpx(x, years)[:years]
        return float(np.sum(self.v ** np.arange(years) * kpx))

    def annuity_immediate(self, x: int, n: Optional[int] = None) -> float:
        """
        Calculate the actuarial present value of an immediate annuity: ax or ax:n

        Payments of 1 are made at the end of each year while (x) is alive.

        Args:
            x: Age
            n: Term of annuity (None for whole life)

        Returns:
            Actuarial present value of immediate annuity
        """
        years = self._horizon(x, n)
        if years == 0:
            return 0.0
        kpx = self._kpx(x, years)[1:]
        return float(np.sum(self.v ** np.arange(1, years + 1) * kpx))

    def deferred_annuity_due(self, x: int, u: int, n: Optional[int] = None) -> float:
        """
        Calculate the actuarial present value of a deferred annuity-due: u|äx

        Args:
            x: Age
            u: Deferred period in years
            n: Payment term after deferment (None for whole life)

        Returns:
            Actuarial present value of deferred annuity-due
        """
        endowment = self.pure_endowment(x, u)
        if endowment == 0:
            return 0.0
        return endowment * self.annuity_due(x + u, n)

    def annuity_mthly(self, x: int, m: int, n: Optional[int] = None, due: bool = True) -> float:
        """
        Annuity of 1 per year payable m times a year (Woolhouse's two-term formula).

        Args:
            x: Age
            m: Number of payments per year
            n: Term of annuity (None for whole life)
            due: True for payments in advance, False for payments in arrears

        Returns:
            Approximate actuarial present value
        """
        if isinstance(m, bool) or m != int(m) or m < 1:
            raise ValueError("m must be a positive integer")
        years = self._horizon(x, n)
        endowment = 0.0 if n is None else self.pure_endowment(x, years)
        adjustment = (m - 1) / (2 * m) * (1 - endowment)
        value_due = self.annuity_due(x, n) - adjustment
        if due:
            return value_due
        return value_due - (1 - endowment) / m

    # ------------------------------------------------------------------
    # Assurances
    # ------------------------------------------------------------------
    def assurance(self, x: int, n: Optional[int] = None, timing: str = "end_of_year") -> float:
        """
        Calculate the actuarial present value of an assurance: Ax or A1x:n

        Args:
            x: Age
            n: Term of assurance (None for whole life)
            timing: "end_of_year" for payment at the end of the year of death,
                "immediate" for payment at the moment of death (UDD)

        Returns:
            Actuarial present value of assurance
        """
        if timing not in ("end_of_year", "immediate"):
            raise ValueError("timing must be 'end_of_year' or 'immediate'")
        years = self._horizon(x, n)
        if years == 0:
            return 0.0
        kpx = self._kpx(x, years)[:years]
        q = self._qx(x, years, whole_life=n is None)
        value = float(np.sum(self.v ** np.arange(1, years + 1) * kpx * q))
        if timing == "immediate":
            value *= self._immediate_factor()
        return value

    def endowment_assurance(self, x: int, n: int, timing: str = "end_of_year") -> float:
        """
        Calculate the actuarial present value of an endowment assurance: Ax:n

        Args:
            x: Age
            n: Term in years
            timing: Timing of the death benefit, see :meth:`assurance`

        Returns:
            Term assurance plus pure endowment
        """
        return self.assurance(x, n, timing) + self.pure_endowment(x, n)

    def increasing_assurance(self, x: int, n: Optional[int] = None) -> float:
        """
        Increasing assurance (IA)x: pays k+1 at the end of year of death k+1.

        Args:
            x: Age
            n: Term of assurance (None for whole life)
        """
        years = self._horizon(x, n)
        if years == 0:
            return 0.0
        kpx = self._kpx(x, years)[:years]
        q = self._qx(x, years, whole_life=n is None)
        k = np.arange(1, years + 1)
        return float(np.sum(k * self.v ** k * kpx * q))

    def increasing_annuity_due(self, x: int, n: Optional[int] = None) -> float:
        """
        Increasing annuity-due (Iä)x: pays 1, 2, 3, ... at the start of each year.

        Args:
            x: Age
            n: Term of annuity (None for whole life)
        """
        years = self._horizon(x, n)
        if years == 0:
            return 0.0
        kpx = self._kpx(x, years)[:years]
        k = np.arange(years)
        return float(np.sum((k + 1) * self.v ** k * kpx))

    # ------------------------------------------------------------------
    # Two lives (independent)
    # ------------------------------------------------------------------
    def _joint_kpxy(self, x: int, y: int, n: Optional[int], table_y: Optional[MortalityTable]):
        """Joint survival probabilities k p_xy for k = 0, ..., years, lives independent."""
        mt_y = table_y if table_y is not None else self.mt
        start_x = self.mt._index(x)
        start_y = mt_y._index(y)
        years = min(len(self.mt.ages) - start_x, len(mt_y.ages) - start_y)
        if n is not None:
            n = self._check_term(n)
            if n > years:
                raise ValueError(
                    f"Term of {n} years runs past the end of the mortality table "
                    f"for lives aged {x} and {y}"
                )
            years = n
        px = self.mt.px_values[start_x:start_x + years]
        py = mt_y.px_values[start_y:start_y + years]
        kpxy = np.concatenate(([1.0], np.cumprod(px * py)))
        return kpxy, years, mt_y

    def joint_annuity_due(self,
                          x: int,
                          y: int,
                          n: Optional[int] = None,
                          table_y: Optional[MortalityTable] = None) -> float:
        """
        Joint life annuity-due äxy: paid while both (x) and (y) are alive.

        Args:
            x: Age of first life
            y: Age of second life
            n: Term of annuity (None for whole life)
            table_y: Mortality table for (y) (defaults to the table for (x))
        """
        kpxy, years, _ = self._joint_kpxy(x, y, n, table_y)
        return float(np.sum(self.v ** np.arange(years) * kpxy[:years]))

    def joint_assurance(self,
                        x: int,
                        y: int,
                        n: Optional[int] = None,
                        table_y: Optional[MortalityTable] = None) -> float:
        """
        Joint life assurance Axy: paid at the end of the year of the first death.

        For whole life the tables are closed at the earlier of their last ages.

        Args:
            x: Age of first life
            y: Age of second life
            n: Term of assurance (None for whole life)
            table_y: Mortality table for (y) (defaults to the table for (x))
        """
        kpxy, years, _ = self._joint_kpxy(x, y, n, table_y)
        survive_next = kpxy[1:].copy()
        if n is None:
            survive_next[-1] = 0.0
        return float(np.sum(self.v ** np.arange(1, years + 1) * (kpxy[:years] - survive_next)))

    def contingent_assurance(self,
                             x: int,
                             y: int,
                             n: Optional[int] = None,
                             table_y: Optional[MortalityTable] = None) -> float:
        """
        Contingent assurance A1xy: paid at the end of the year of death of (x),
        provided (x) dies before (y). Deaths are uniform over each year of age.

        Args:
            x: Age of the life assured
            y: Age of the counter life
            n: Term of assurance (None for whole life)
            table_y: Mortality table for (y) (defaults to the table for (x))
        """
        kpxy, years, mt_y = self._joint_kpxy(x, y, n, table_y)
        start_x = self.mt._index(x)
        start_y = mt_y._index(y)
        qx = self.mt.qx_values[start_x:start_x + years].copy()
        qy = mt_y.qx_values[start_y:start_y + years]
        if n is None and start_x + years == len(self.mt.ages):
            qx[-1] = 1.0
        deaths = kpxy[:years] * qx * (1 - 0.5 * qy)
        return float(np.sum(self.v ** np.arange(1, years + 1) * deaths))

    # ------------------------------------------------------------------
    # Premiums
    # ------------------------------------------------------------------
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
            Net annual premium by the equivalence principle
        """
        if product == "whole_life":
            benefit = self.assurance(x)
            term = None
        elif product in ("term", "endowment"):
            if n is None:
                raise ValueError(f"n is required for a {product} assurance")
            benefit = self.assurance(x, n) if product == "term" else self.endowment_assurance(x, n)
            term = n
        else:
            raise ValueError("product must be 'whole_life', 'term' or 'endowment'")

        annuity = self.annuity_due(x, premium_term if premium_term is not None else term)
        if annuity == 0:
            raise ValueError("premium paying term must be at least one year")
        return benefit / annuity
