"""
Claims Simulator

Generates individual claims with payment delays, so that a claims triangle
can be built from them and compared with what the claims finally cost.

A real triangle cannot tell you whether a reserving method got the right
answer until years later. A simulated one can: the generator knows every
future payment. It can also inject the changes that break the chain-ladder
assumptions (a jump in inflation, faster or slower settlement, a year of
unusually large claims), which makes it a test bed for reserving methods.
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..loss_reserving import Triangle


class ClaimsSimulator:
    """
    Simulator of individual claim payments by accident year.

    Each accident year has a Poisson number of claims with lognormal sizes.
    Every claim is paid in instalments over the development years according
    to a payment pattern, and payments are inflated to the calendar year in
    which they are made.
    """

    def __init__(self,
                 n_years: int = 10,
                 first_year: int = 2015,
                 claims_per_year: float = 500,
                 mean_claim: float = 10_000.0,
                 claim_cv: float = 1.5,
                 payment_pattern: Sequence[float] = (0.35, 0.25, 0.15, 0.10, 0.07, 0.05, 0.03),
                 inflation: float = 0.0,
                 seed: Optional[int] = None,
                 pattern_concentration: Optional[float] = None):
        """
        Args:
            n_years: Number of accident years
            first_year: First accident year
            claims_per_year: Expected number of claims in each accident year
            mean_claim: Mean claim size at the prices of the first year
            claim_cv: Coefficient of variation of claim sizes
            payment_pattern: Proportion of each claim paid in each
                development year (it is rescaled to add up to 1)
            inflation: Annual claims inflation, applied by calendar year of
                payment
            seed: Seed of the random number generator
            pattern_concentration: If given, each claim is paid in its own
                proportions, drawn from a Dirichlet distribution centred on
                the payment pattern. Smaller values make claims differ more
                from each other; None pays every claim in exactly the
                pattern's proportions, which gives triangles with no
                randomness in their development.
        """
        pattern = np.asarray(payment_pattern, dtype=float)
        if np.any(pattern < 0) or pattern.sum() <= 0:
            raise ValueError("payment_pattern must be non-negative with a positive total")
        if n_years < 2 or claims_per_year <= 0 or mean_claim <= 0 or claim_cv <= 0:
            raise ValueError("n_years must be at least 2 and the claim parameters positive")
        if pattern_concentration is not None and pattern_concentration <= 0:
            raise ValueError("pattern_concentration must be positive")
        self.pattern_concentration = pattern_concentration
        self.n_years = int(n_years)
        self.first_year = int(first_year)
        self.claims_per_year = float(claims_per_year)
        self.mean_claim = float(mean_claim)
        self.claim_cv = float(claim_cv)
        self.payment_pattern = pattern / pattern.sum()
        self.base_inflation = float(inflation)
        self.seed = seed
        self._inflation_shock = None
        self._speed_change = None
        self._large_loss_year = None

    # ------------------------------------------------------------------
    # Distortions
    # ------------------------------------------------------------------
    def with_inflation_shock(self, from_year: int, inflation: float) -> 'ClaimsSimulator':
        """Change the rate of claims inflation for payments from a calendar year onwards."""
        self._inflation_shock = (int(from_year), float(inflation))
        return self

    def with_settlement_change(self, from_year: int,
                               payment_pattern: Sequence[float]) -> 'ClaimsSimulator':
        """Use a different payment pattern for claims arising from an accident year onwards."""
        pattern = np.asarray(payment_pattern, dtype=float)
        self._speed_change = (int(from_year), pattern / pattern.sum())
        return self

    def with_large_loss_year(self, year: int, severity_multiple: float) -> 'ClaimsSimulator':
        """Multiply the claim sizes of one accident year."""
        self._large_loss_year = (int(year), float(severity_multiple))
        return self

    # ------------------------------------------------------------------
    def _price_index(self, last_year: int) -> dict:
        index, level = {}, 1.0
        for year in range(self.first_year, last_year + 1):
            if year > self.first_year:
                rate = self.base_inflation
                if self._inflation_shock and year >= self._inflation_shock[0]:
                    rate = self._inflation_shock[1]
                level *= 1 + rate
            index[year] = level
        return index

    def simulate(self) -> pd.DataFrame:
        """
        Generate the payments on every claim, to final settlement.

        Returns:
            DataFrame with one row per payment: ``claim_id``,
            ``accident_year``, ``development_year`` (1 for the accident
            year), ``payment_year``, ``loss_date``, ``payment_date`` and
            ``amount``
        """
        rng = np.random.default_rng(self.seed)
        sigma2 = np.log1p(self.claim_cv ** 2)
        mu = np.log(self.mean_claim) - sigma2 / 2
        longest = len(self.payment_pattern)
        if self._speed_change:
            longest = max(longest, len(self._speed_change[1]))
        index = self._price_index(self.first_year + self.n_years - 1 + longest)

        rows = []
        claim_id = 0
        for year in range(self.first_year, self.first_year + self.n_years):
            count = rng.poisson(self.claims_per_year)
            sizes = rng.lognormal(mu, np.sqrt(sigma2), count)
            if self._large_loss_year and year == self._large_loss_year[0]:
                sizes = sizes * self._large_loss_year[1]
            pattern = self.payment_pattern
            if self._speed_change and year >= self._speed_change[0]:
                pattern = self._speed_change[1]
            paid_from = np.flatnonzero(np.asarray(pattern) > 0)
            if self.pattern_concentration is not None:
                drawn = rng.dirichlet(np.asarray(pattern)[paid_from] * self.pattern_concentration
                                      / np.sum(pattern), count)
            for number, size in enumerate(sizes):
                claim_id += 1
                shares = pattern
                if self.pattern_concentration is not None:
                    shares = np.zeros(len(pattern))
                    shares[paid_from] = drawn[number]
                for dev, share in enumerate(shares):
                    if share == 0:
                        continue
                    paid_in = year + dev
                    rows.append((claim_id, year, dev + 1, paid_in, size * share * index[paid_in]))
        frame = pd.DataFrame(rows, columns=["claim_id", "accident_year", "development_year",
                                            "payment_year", "amount"])
        frame["loss_date"] = pd.to_datetime(frame["accident_year"].astype(str) + "-07-01")
        frame["payment_date"] = pd.to_datetime(frame["payment_year"].astype(str) + "-12-31")
        return frame

    def triangle(self, payments: Optional[pd.DataFrame] = None,
                 valuation_year: Optional[int] = None) -> Triangle:
        """
        Cumulative paid triangle as it would be seen at the end of a year.

        Args:
            payments: Output of :meth:`simulate` (simulated afresh if omitted)
            valuation_year: Last calendar year observed (defaults to the
                last accident year)
        """
        payments = self.simulate() if payments is None else payments
        if valuation_year is None:
            valuation_year = self.first_year + self.n_years - 1
        seen = payments[payments["payment_year"] <= valuation_year]
        return Triangle.from_long(seen, origin="accident_year", development="development_year",
                                  value="amount", cumulative=False,
                                  name="Simulated paid").to_cumulative()

    def true_ultimate(self, payments: pd.DataFrame) -> pd.Series:
        """The amount finally paid on each accident year, known only to the simulator."""
        return payments.groupby("accident_year")["amount"].sum().rename("true_ultimate")
