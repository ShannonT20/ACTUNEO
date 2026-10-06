"""
Aggregate Claims

The collective risk model: total claims in a period are the sum of a random
number of claims, each of a random size, with numbers and sizes independent.
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd

_FREQUENCIES = ("poisson", "negative_binomial", "binomial")
_SEVERITIES = ("exponential", "gamma", "lognormal", "pareto", "fixed")


def _claim_counts(rng, frequency: str, parameters: Sequence[float], size: int) -> np.ndarray:
    if frequency == "poisson":
        (mean,) = parameters
        return rng.poisson(mean, size)
    if frequency == "negative_binomial":
        mean, variance = parameters
        if variance <= mean:
            raise ValueError("a negative binomial needs a variance greater than its mean")
        p = mean / variance
        return rng.negative_binomial(mean * p / (1 - p), p, size)
    if frequency == "binomial":
        n, p = parameters
        return rng.binomial(int(n), p, size)
    raise ValueError(f"frequency must be one of {_FREQUENCIES}")


def _claim_sizes(rng, severity: str, parameters: Sequence[float], size: int) -> np.ndarray:
    if severity == "exponential":
        (mean,) = parameters
        return rng.exponential(mean, size)
    if severity == "gamma":
        shape, scale = parameters
        return rng.gamma(shape, scale, size)
    if severity == "lognormal":
        mu, sigma = parameters
        return rng.lognormal(mu, sigma, size)
    if severity == "pareto":
        # Pareto with density alpha * lam**alpha / (lam + x)**(alpha + 1)
        alpha, lam = parameters
        return lam * (rng.uniform(size=size) ** (-1 / alpha) - 1)
    if severity == "fixed":
        (amount,) = parameters
        return np.full(size, float(amount))
    raise ValueError(f"severity must be one of {_SEVERITIES}")


class AggregateClaims:
    """Simulated aggregate claims with summary statistics and risk measures."""

    def __init__(self, totals: np.ndarray, counts: np.ndarray):
        self.totals = np.asarray(totals, dtype=float)
        self.counts = np.asarray(counts)

    @property
    def mean(self) -> float:
        return float(self.totals.mean())

    @property
    def standard_deviation(self) -> float:
        return float(self.totals.std(ddof=1))

    def value_at_risk(self, level: float = 0.995) -> float:
        """The amount that total claims exceed with probability ``1 - level``."""
        return float(np.quantile(self.totals, level))

    def tail_value_at_risk(self, level: float = 0.995) -> float:
        """Average of total claims in the worst ``1 - level`` of outcomes."""
        threshold = self.value_at_risk(level)
        return float(self.totals[self.totals >= threshold].mean())

    def probability_of_exceeding(self, amount: float) -> float:
        """Proportion of simulations in which total claims exceed an amount."""
        return float(np.mean(self.totals > amount))

    def summary(self, levels: Sequence[float] = (0.75, 0.95, 0.995)) -> pd.Series:
        """Mean, standard deviation and quantiles of total claims."""
        values = {"mean": self.mean, "standard_deviation": self.standard_deviation,
                  "mean_number_of_claims": float(self.counts.mean())}
        for level in levels:
            values[f"quantile_{100 * level:g}%"] = self.value_at_risk(level)
        return pd.Series(values, name="aggregate_claims")


def simulate_aggregate_claims(frequency: str,
                              frequency_parameters: Sequence[float],
                              severity: str,
                              severity_parameters: Sequence[float],
                              n_simulations: int = 10_000,
                              retention: Optional[float] = None,
                              seed: Optional[int] = None) -> AggregateClaims:
    """
    Simulate total claims for one period under the collective risk model.

    Args:
        frequency: Distribution of the number of claims: "poisson" (mean),
            "negative_binomial" (mean, variance) or "binomial" (n, p)
        frequency_parameters: Parameters of that distribution, as listed
        severity: Distribution of each claim: "exponential" (mean), "gamma"
            (shape, scale), "lognormal" (mu, sigma), "pareto" (alpha, lambda)
            or "fixed" (amount)
        severity_parameters: Parameters of that distribution, as listed
        n_simulations: Number of simulated periods
        retention: Limit on each claim for the insurer under an individual
            excess of loss reinsurance (claims are cut off at this amount)
        seed: Seed of the random number generator

    Returns:
        AggregateClaims holding the simulated totals
    """
    if n_simulations < 1:
        raise ValueError("n_simulations must be at least 1")
    rng = np.random.default_rng(seed)
    counts = _claim_counts(rng, frequency, frequency_parameters, n_simulations)
    sizes = _claim_sizes(rng, severity, severity_parameters, int(counts.sum()))
    if retention is not None:
        sizes = np.minimum(sizes, retention)
    period = np.repeat(np.arange(n_simulations), counts)
    totals = np.bincount(period, weights=sizes, minlength=n_simulations)
    return AggregateClaims(totals, counts)
