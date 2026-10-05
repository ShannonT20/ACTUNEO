"""
Bootstrap Chain-Ladder

Simulates the full predictive distribution of the reserve by bootstrapping
the over-dispersed Poisson model that underlies the chain-ladder, following
England and Verrall and the ``BootChainLadder`` function of the R
ChainLadder package.

References
----------
- England, P.D. and Verrall, R.J. (1999). Analytic and bootstrap estimates
  of prediction errors in claims reserving. Insurance: Mathematics and
  Economics 25(3), 281-293.
- England, P.D. (2002). Addendum to "Analytic and bootstrap estimates of
  prediction errors in claims reserving". Insurance: Mathematics and
  Economics 31(3), 461-466.
- England, P.D. and Verrall, R.J. (2002). Stochastic claims reserving in
  general insurance. British Actuarial Journal 8(3), 443-518.
"""

import numpy as np
import pandas as pd
from typing import Optional, Sequence
from .triangle import Triangle
from .chain_ladder import ChainLadder


class BootChainLadder:
    """
    Bootstrap of the chain-ladder with process error.

    Steps:

    1. Fit the chain-ladder and back-fit the incremental claims it implies
       for the observed cells.
    2. Form Pearson residuals ``(actual - fitted) / sqrt(fitted)``, adjusted
       for the number of parameters, and the scale parameter ``phi``.
    3. For each simulation, resample the residuals to build a pseudo
       triangle, refit the chain-ladder and project the future incremental
       claims (estimation error).
    4. Draw each future incremental claim from a distribution with the
       projected mean and variance ``phi * mean`` (process error).

    Attributes:
        scale: Pearson scale parameter phi
        residuals: Adjusted Pearson residuals of the observed cells
        ibnr_simulations: Simulated reserve by origin period, one row per simulation
        total_ibnr_simulations: Simulated total reserve
        latest: Latest observed cumulative claims by origin period
        chain_ladder: The chain-ladder fitted to the actual triangle
    """

    def __init__(self,
                 triangle: Triangle,
                 n_simulations: int = 999,
                 process_distribution: str = "gamma",
                 seed: Optional[int] = None):
        """
        Run the bootstrap.

        Args:
            triangle: Claims triangle (cumulative or incremental)
            n_simulations: Number of bootstrap simulations
            process_distribution: "gamma" or "od_poisson" (over-dispersed
                Poisson) for the process error
            seed: Seed of the random number generator, for reproducible results
        """
        if not isinstance(triangle, Triangle):
            raise TypeError("triangle must be a Triangle")
        if process_distribution not in ("gamma", "od_poisson"):
            raise ValueError("process_distribution must be 'gamma' or 'od_poisson'")
        if n_simulations < 1:
            raise ValueError("n_simulations must be at least 1")

        self.triangle = triangle.to_cumulative()
        self.n_simulations = int(n_simulations)
        self.process_distribution = process_distribution
        self.chain_ladder = ChainLadder(self.triangle)
        self.latest = self.chain_ladder.latest

        rng = np.random.default_rng(seed)
        cum = self.triangle.values
        observed = ~np.isnan(cum)
        n_origin, n_dev = cum.shape
        factors = self.chain_ladder.factors.to_numpy()
        latest_idx = self.triangle._latest_index()

        # Back-fit: start from the latest diagonal and divide by the factors
        fitted_cum = np.full(cum.shape, np.nan)
        for i, j0 in enumerate(latest_idx):
            fitted_cum[i, j0] = cum[i, j0]
            for k in range(j0 - 1, -1, -1):
                fitted_cum[i, k] = fitted_cum[i, k + 1] / factors[k]
        fitted = self._incremental(fitted_cum)
        actual = self._incremental(cum)

        if np.any(fitted[observed] == 0):
            raise ValueError("The fitted incremental claims contain zeros; cannot form residuals")
        root = np.sqrt(np.abs(fitted))
        unscaled = (actual - fitted) / root

        n_obs = int(observed.sum())
        n_parameters = n_origin + n_dev - 1
        if n_obs <= n_parameters:
            raise ValueError("The triangle has too few observations to bootstrap")
        self.scale = float(np.sum(unscaled[observed] ** 2) / (n_obs - n_parameters))
        adjusted = unscaled * np.sqrt(n_obs / (n_obs - n_parameters))
        self.residuals = pd.DataFrame(adjusted, index=self.chain_ladder.full_triangle.index,
                                      columns=self.chain_ladder.full_triangle.columns)

        # Pseudo triangles: resampled residuals applied to the fitted increments
        pool = adjusted[observed]
        sims = self.n_simulations
        draws = rng.choice(pool, size=(sims, n_obs), replace=True)
        pseudo_incremental = np.zeros((sims, n_origin, n_dev))
        pseudo_incremental[:, observed] = draws * root[observed] + fitted[observed]
        pseudo_cum = np.cumsum(pseudo_incremental, axis=2)

        # Refit the development factors to every pseudo triangle
        pair = observed[:, :-1] & observed[:, 1:]
        numerator = np.sum(pseudo_cum[:, :, 1:] * pair, axis=1)
        denominator = np.sum(pseudo_cum[:, :, :-1] * pair, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            pseudo_factors = numerator / denominator
        pseudo_factors[~np.isfinite(pseudo_factors)] = 1.0

        # Project each pseudo triangle and keep the future incremental means
        projected = pseudo_cum.copy()
        for k in range(n_dev - 1):
            future = ~observed[:, k + 1]
            projected[:, future, k + 1] = (projected[:, future, k]
                                           * pseudo_factors[:, k][:, None])
        future_mean = np.diff(projected, axis=2)[:, ~observed[:, 1:]]

        # Process error around each future incremental mean
        size = np.abs(future_mean) / self.scale
        sign = np.sign(future_mean)
        if process_distribution == "gamma":
            simulated = np.where(size > 0, rng.gamma(np.where(size > 0, size, 1.0), self.scale), 0.0)
        else:
            simulated = rng.poisson(size) * self.scale
        future_claims = np.zeros((sims, n_origin, n_dev - 1))
        future_claims[:, ~observed[:, 1:]] = sign * simulated

        origin_index = pd.Index(self.triangle.origin, name="origin")
        self.ibnr_simulations = pd.DataFrame(future_claims.sum(axis=2), columns=origin_index)
        self.total_ibnr_simulations = self.ibnr_simulations.sum(axis=1).rename("total_ibnr")

    @staticmethod
    def _incremental(cumulative: np.ndarray) -> np.ndarray:
        incremental = cumulative.copy()
        incremental[:, 1:] = np.diff(cumulative, axis=1)
        return incremental

    @property
    def mean_ibnr(self) -> float:
        """Mean of the simulated total reserve."""
        return float(self.total_ibnr_simulations.mean())

    @property
    def sd_ibnr(self) -> float:
        """Standard deviation of the simulated total reserve (the prediction error)."""
        return float(self.total_ibnr_simulations.std(ddof=1))

    def quantile(self, q) -> pd.DataFrame:
        """
        Quantiles of the simulated reserve.

        Args:
            q: Probability or list of probabilities, for example [0.75, 0.995]

        Returns:
            DataFrame of reserve quantiles by origin period with a "Total" row
        """
        probabilities = np.atleast_1d(q)
        by_origin = self.ibnr_simulations.quantile(probabilities).T
        by_origin.loc["Total"] = self.total_ibnr_simulations.quantile(probabilities)
        by_origin.columns = [f"ibnr_{100 * p:g}%" for p in probabilities]
        return by_origin

    def summary(self, quantiles: Sequence[float] = (0.75, 0.95)) -> pd.DataFrame:
        """
        Results by origin period with a "Total" row.

        Args:
            quantiles: Reserve quantiles to report

        Returns:
            DataFrame with latest claims, mean ultimate, mean and standard
            deviation of the reserve, and the requested quantiles
        """
        mean = self.ibnr_simulations.mean()
        table = pd.DataFrame({
            "latest": self.latest,
            "mean_ultimate": self.latest + mean,
            "mean_ibnr": mean,
            "sd_ibnr": self.ibnr_simulations.std(ddof=1),
        })
        table.loc["Total"] = [
            self.latest.sum(), self.latest.sum() + self.mean_ibnr, self.mean_ibnr, self.sd_ibnr,
        ]
        return table.join(self.quantile(list(quantiles)))

    def __repr__(self) -> str:
        return (f"BootChainLadder(triangle='{self.triangle.name}', "
                f"n_simulations={self.n_simulations}, "
                f"process_distribution='{self.process_distribution}')\n"
                f"{self.summary().to_string()}")
