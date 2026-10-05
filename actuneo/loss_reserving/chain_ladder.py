"""
Chain-Ladder Reserving

:class:`ChainLadder` is the deterministic chain-ladder method: development
factors are averaged from the triangle and applied to the latest diagonal to
project every origin period to ultimate.

:class:`MackChainLadder` adds the distribution-free standard error of Mack
(1993), following the formulation of the R ``ChainLadder`` package.

References
----------
- Mack, T. (1993). Distribution-free calculation of the standard error of
  chain ladder reserve estimates. ASTIN Bulletin 23(2), 213-225.
- Mack, T. (1999). The standard error of chain ladder reserve estimates:
  recursive calculation and inclusion of a tail factor. ASTIN Bulletin
  29(2), 361-366.
- England, P.D. and Verrall, R.J. (2002). Stochastic claims reserving in
  general insurance. British Actuarial Journal 8(3), 443-518.
- Gesmann, M., Murphy, D., Zhang, Y., Carrato, A., Wuthrich, M., Concina, F.
  and Dal Moro, E. ChainLadder: Statistical Methods and Models for Claims
  Reserving in General Insurance. R package.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Optional
from .triangle import Triangle


class ChainLadder:
    """
    Deterministic chain-ladder projection of a claims triangle.

    Attributes:
        triangle: The cumulative triangle the method was fitted to
        factors: Development factor for each development step
        tail: Tail factor from the last development period to ultimate
        cdf: Cumulative development factor to ultimate by development period
        full_triangle: Triangle with the unobserved cells projected
        latest: Latest observed cumulative claims by origin period
        ultimate: Projected ultimate claims by origin period
        ibnr: Reserve (ultimate less latest) by origin period
    """

    def __init__(self,
                 triangle: Triangle,
                 average: str = "volume",
                 n_periods: Optional[int] = None,
                 tail: float = 1.0):
        """
        Fit the chain-ladder method.

        Args:
            triangle: Claims triangle (cumulative or incremental)
            average: "volume" for volume-weighted development factors,
                "simple" for the arithmetic mean of the link ratios
            n_periods: Average only the latest n origin periods (None for all)
            tail: Tail factor applied after the last development period
        """
        if not isinstance(triangle, Triangle):
            raise TypeError("triangle must be a Triangle")
        if tail <= 0:
            raise ValueError("tail must be positive")

        self.triangle = triangle.to_cumulative()
        self.average = average
        self.n_periods = n_periods
        self.tail = float(tail)
        self._fit(self._development_factors())

    def _development_factors(self) -> np.ndarray:
        return self.triangle.development_factors(self.average, self.n_periods).to_numpy()

    def _fit(self, factors: np.ndarray):
        tri = self.triangle
        if np.any(np.isnan(factors)):
            missing = [tri._link_labels()[k] for k in np.flatnonzero(np.isnan(factors))]
            raise ValueError(
                f"No development factor can be estimated for development step(s) {missing}"
            )

        self._f = factors
        self._latest_idx = tri._latest_index()
        origin_index = pd.Index(tri.origin, name="origin")

        # Project each origin period forward from its latest observation
        full = tri.values.copy()
        for i, j0 in enumerate(self._latest_idx):
            for k in range(j0, tri.n_development - 1):
                full[i, k + 1] = full[i, k] * factors[k]
        self._full = full

        self.factors = pd.Series(factors, index=tri._link_labels(), name="factor")
        to_ultimate = np.append(np.cumprod(factors[::-1])[::-1], 1.0) * self.tail
        self.cdf = pd.Series(to_ultimate,
                             index=pd.Index(tri.development, name="development"), name="cdf")
        self.full_triangle = pd.DataFrame(
            full, index=origin_index, columns=pd.Index(tri.development, name="development")
        )
        self.latest = tri.latest_diagonal()
        self.ultimate = pd.Series(full[:, -1] * self.tail, index=origin_index, name="ultimate")
        self.ibnr = (self.ultimate - self.latest).rename("ibnr")

    @property
    def total_ibnr(self) -> float:
        """Total reserve over all origin periods."""
        return float(self.ibnr.sum())

    def summary(self, total: bool = True) -> pd.DataFrame:
        """
        Results by origin period.

        Args:
            total: Add a "Total" row

        Returns:
            DataFrame with latest claims, proportion developed to date,
            ultimate claims and IBNR
        """
        table = pd.DataFrame({
            "latest": self.latest,
            "dev_to_date": self.latest / self.ultimate,
            "ultimate": self.ultimate,
            "ibnr": self.ibnr,
        })
        if total:
            table.loc["Total"] = [
                self.latest.sum(),
                self.latest.sum() / self.ultimate.sum(),
                self.ultimate.sum(),
                self.ibnr.sum(),
            ]
        return table

    def __repr__(self) -> str:
        return f"{type(self).__name__}(triangle='{self.triangle.name}')\n{self.summary().to_string()}"


class MackChainLadder(ChainLadder):
    """
    Mack's distribution-free chain-ladder model.

    The model assumes, for cumulative claims ``C[i, k]``:

    - ``E[C[i, k+1] | C[i, 1..k]] = f[k] * C[i, k]``
    - ``Var(C[i, k+1] | C[i, 1..k]) = sigma[k]**2 * C[i, k]**(2 - alpha)``
    - origin periods are independent

    Attributes:
        sigma: Estimated sigma[k] for each development step
        f_se: Standard error of each development factor
        process_risk: Process standard deviation of the reserve by origin period
        parameter_risk: Estimation standard error of the reserve by origin period
        mack_se: Root mean squared error of the reserve by origin period
        total_process_risk: Process standard deviation of the total reserve
        total_parameter_risk: Estimation standard error of the total reserve,
            allowing for the development factors shared by all origin periods
        total_mack_se: Root mean squared error of the total reserve
    """

    def __init__(self,
                 triangle: Triangle,
                 alpha: float = 1.0,
                 est_sigma: str = "log-linear"):
        """
        Fit the Mack chain-ladder model.

        Args:
            triangle: Claims triangle (cumulative or incremental)
            alpha: Variance exponent. 1 gives the volume-weighted chain-ladder
                factors, 0 the simple average of the link ratios, 2 the
                least-squares regression through the origin.
            est_sigma: How to estimate sigma for a development step with a
                single link ratio. "log-linear" extrapolates a regression of
                log(sigma) on the development step, "mack" uses Mack's
                approximation ``min(s2**2 / s1, min(s1, s2))`` from the two
                preceding variances. "log-linear" falls back to "mack" when
                the regression is not significant at 5%.
        """
        if est_sigma not in ("log-linear", "mack"):
            raise ValueError("est_sigma must be 'log-linear' or 'mack'")
        if not isinstance(triangle, Triangle):
            raise TypeError("triangle must be a Triangle")

        self.triangle = triangle.to_cumulative()
        self.alpha = float(alpha)
        self.est_sigma = est_sigma
        self.average = {1.0: "volume", 0.0: "simple"}.get(self.alpha, f"alpha={self.alpha}")
        self.n_periods = None
        self.tail = 1.0

        factors, sigma, f_se = self._estimate_parameters()
        self._fit(factors)
        self._standard_errors(sigma, f_se)

    def _estimate_parameters(self):
        cum = self.triangle.values
        if np.any(cum[~np.isnan(cum)] <= 0):
            raise ValueError("Mack's model needs positive cumulative claims in every cell")

        n_steps = cum.shape[1] - 1
        factors = np.full(n_steps, np.nan)
        sigma2 = np.full(n_steps, np.nan)
        weight_sum = np.full(n_steps, np.nan)

        for k in range(n_steps):
            rows = ~np.isnan(cum[:, k]) & ~np.isnan(cum[:, k + 1])
            if not rows.any():
                continue
            start, end = cum[rows, k], cum[rows, k + 1]
            weights = start ** self.alpha
            ratios = end / start
            factors[k] = np.sum(weights * ratios) / np.sum(weights)
            weight_sum[k] = np.sum(weights)
            if rows.sum() > 1:
                sigma2[k] = np.sum(weights * (ratios - factors[k]) ** 2) / (rows.sum() - 1)

        sigma = np.sqrt(self._fill_sigma2(sigma2))
        f_se = sigma / np.sqrt(weight_sum)
        return factors, sigma, f_se

    def _fill_sigma2(self, sigma2: np.ndarray) -> np.ndarray:
        """Estimate sigma**2 for development steps with a single link ratio."""
        missing = np.flatnonzero(np.isnan(sigma2))
        if len(missing) == 0:
            return sigma2

        sigma2 = sigma2.copy()
        known = np.flatnonzero(~np.isnan(sigma2) & (sigma2 > 0))
        method = self.est_sigma

        if method == "log-linear":
            if len(known) < 3:
                method = "mack"
            else:
                from scipy import stats
                fit = stats.linregress(known + 1, 0.5 * np.log(sigma2[known]))
                if fit.pvalue > 0.05:
                    warnings.warn(
                        "The log-linear model for sigma is not significant at 5%; "
                        "Mack's approximation is used instead.",
                        UserWarning,
                        stacklevel=4,
                    )
                    method = "mack"
                else:
                    sigma2[missing] = np.exp(2 * (fit.intercept + fit.slope * (missing + 1)))
                    return sigma2

        for k in missing:
            if k < 2 or np.isnan(sigma2[k - 1]) or np.isnan(sigma2[k - 2]):
                raise ValueError(
                    "Not enough development periods to estimate sigma for the last "
                    "development step"
                )
            s1, s2 = sigma2[k - 2], sigma2[k - 1]
            sigma2[k] = min(s2 ** 2 / s1, min(s1, s2)) if s1 > 0 else 0.0
        return sigma2

    def _standard_errors(self, sigma: np.ndarray, f_se: np.ndarray):
        tri = self.triangle
        full, f = self._full, self._f
        n_origin, n_dev = full.shape
        labels = tri._link_labels()
        origin_index = pd.Index(tri.origin, name="origin")

        # Recursion of Mack (1999): squared risks roll forward one development
        # step at a time from the latest diagonal
        process2 = np.zeros(n_origin)
        parameter2 = np.zeros(n_origin)
        for i, j0 in enumerate(self._latest_idx):
            for k in range(j0, n_dev - 1):
                c = full[i, k]
                process2[i] = process2[i] * f[k] ** 2 + sigma[k] ** 2 * c ** (2 - self.alpha)
                parameter2[i] = parameter2[i] * f[k] ** 2 + c ** 2 * f_se[k] ** 2

        # Total: process risk adds across independent origin periods, while the
        # estimation error of each factor is common to all origins it projects
        total_parameter2 = 0.0
        for k in range(n_dev - 1):
            projected = self._latest_idx <= k
            total_parameter2 = (total_parameter2 * f[k] ** 2
                                + full[projected, k].sum() ** 2 * f_se[k] ** 2)

        self.sigma = pd.Series(sigma, index=labels, name="sigma")
        self.f_se = pd.Series(f_se, index=labels, name="f_se")
        self.process_risk = pd.Series(np.sqrt(process2), index=origin_index, name="process_risk")
        self.parameter_risk = pd.Series(np.sqrt(parameter2), index=origin_index,
                                        name="parameter_risk")
        self.mack_se = pd.Series(np.sqrt(process2 + parameter2), index=origin_index,
                                 name="mack_se")
        self.total_process_risk = float(np.sqrt(process2.sum()))
        self.total_parameter_risk = float(np.sqrt(total_parameter2))
        self.total_mack_se = float(np.sqrt(process2.sum() + total_parameter2))

    def summary(self, total: bool = True) -> pd.DataFrame:
        """
        Results by origin period.

        Args:
            total: Add a "Total" row

        Returns:
            DataFrame with latest claims, proportion developed to date,
            ultimate claims, IBNR, Mack standard error and coefficient of
            variation of the IBNR
        """
        table = super().summary(total=False)
        table["mack_se"] = self.mack_se
        if total:
            table.loc["Total"] = [
                self.latest.sum(),
                self.latest.sum() / self.ultimate.sum(),
                self.ultimate.sum(),
                self.ibnr.sum(),
                self.total_mack_se,
            ]
        with np.errstate(divide="ignore", invalid="ignore"):
            table["cv_ibnr"] = table["mack_se"] / table["ibnr"]
        return table
