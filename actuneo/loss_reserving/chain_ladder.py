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
from typing import Optional, Tuple, Union
from .triangle import Triangle


def _loglinear_fit(values: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Least-squares line through log(values) against development step 1, 2, ...

    Steps where the value is not positive are left out. Returns the intercept
    and slope, or None when fewer than two steps can be used.
    """
    steps = np.arange(1, len(values) + 1, dtype=float)
    usable = np.isfinite(values) & (values > 0)
    if usable.sum() < 2:
        return None
    slope, intercept = np.polyfit(steps[usable], np.log(values[usable]), 1)
    return float(intercept), float(slope)


def estimate_tail_factor(factors, n_periods: int = 100) -> float:
    """
    Tail factor from an exponential decay of the development factors.

    A straight line is fitted to ``log(f - 1)`` against the development step,
    using the factors above 1, and extended for ``n_periods`` further steps.
    The tail factor is the product of the extrapolated factors (the approach
    of ``tail=TRUE`` in the R ChainLadder package).

    Args:
        factors: Development factors in development order
        n_periods: Number of future development steps to extrapolate

    Returns:
        Tail factor from the last development period to ultimate
    """
    factors = np.asarray(factors, dtype=float)
    fit = _loglinear_fit(factors - 1)
    if fit is None:
        raise ValueError("At least two development factors above 1 are needed to estimate a tail")
    intercept, slope = fit
    if slope >= 0:
        raise ValueError(
            "The development factors do not decay towards 1, so no tail factor can be "
            "extrapolated. Supply a tail factor instead."
        )
    steps = np.arange(len(factors) + 1, len(factors) + 1 + n_periods)
    tail = float(np.prod(1 + np.exp(intercept + slope * steps)))
    if tail > 2:
        warnings.warn(
            f"The estimated tail factor is {tail:.3f}. A tail this large is usually "
            "unreliable; consider supplying one.",
            UserWarning,
            stacklevel=3,
        )
    return tail


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
                 tail: Union[float, bool] = 1.0):
        """
        Fit the chain-ladder method.

        Args:
            triangle: Claims triangle (cumulative or incremental)
            average: "volume" for volume-weighted development factors,
                "simple" for the arithmetic mean of the link ratios
            n_periods: Average only the latest n origin periods (None for all)
            tail: Tail factor applied after the last development period, or
                True to estimate it with :func:`estimate_tail_factor`
        """
        if not isinstance(triangle, Triangle):
            raise TypeError("triangle must be a Triangle")

        self.triangle = triangle.to_cumulative()
        self.average = average
        self.n_periods = n_periods
        factors = self._development_factors()
        self.tail = self._resolve_tail(tail, factors)
        self._fit(factors)

    @staticmethod
    def _resolve_tail(tail: Union[float, bool], factors: np.ndarray) -> float:
        if tail is True:
            return estimate_tail_factor(factors)
        if tail is False:
            return 1.0
        if tail <= 0:
            raise ValueError("tail must be positive")
        return float(tail)

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

    def reserve(self, paid_to_date: Optional[float] = None) -> float:
        """
        Total reserve.

        Args:
            paid_to_date: Total claims paid so far. Needed when the triangle
                holds incurred claims, where the reserve is the projected
                ultimate less the amount paid. If omitted, the reserve is the
                ultimate less the latest diagonal of the triangle.

        Returns:
            Total outstanding claims reserve
        """
        if paid_to_date is None:
            return self.total_ibnr
        return float(self.ultimate.sum() - paid_to_date)

    def fitted_triangle(self) -> pd.DataFrame:
        """
        Cumulative claims the development factors would have produced.

        Each origin period starts from its actual first development period
        and is rolled forward with the fitted development factors, over the
        cells that have been observed.
        """
        tri = self.triangle
        fitted = np.full(tri.shape, np.nan)
        fitted[:, 0] = tri.values[:, 0]
        for k in range(tri.n_development - 1):
            fitted[:, k + 1] = fitted[:, k] * self._f[k]
        fitted[np.isnan(tri.values)] = np.nan
        return pd.DataFrame(fitted, index=self.full_triangle.index,
                            columns=self.full_triangle.columns)

    def fit_errors(self) -> pd.DataFrame:
        """
        Actual less fitted incremental claims, for checking the model.

        Large errors, or errors of one sign along a diagonal or down a
        column, suggest that the development pattern is not stable.
        """
        actual = self.triangle.to_incremental().values
        fitted = self.fitted_triangle().to_numpy()
        fitted_incremental = fitted.copy()
        fitted_incremental[:, 1:] = np.diff(fitted, axis=1)
        return pd.DataFrame(actual - fitted_incremental, index=self.full_triangle.index,
                            columns=self.full_triangle.columns)

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
        tail_se: Standard error of the tail factor (0 without a tail)
        tail_sigma: Sigma of the tail development step (0 without a tail)
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
                 est_sigma: str = "log-linear",
                 tail: Union[float, bool] = 1.0,
                 tail_se: Optional[float] = None,
                 tail_sigma: Optional[float] = None):
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
            tail: Tail factor applied after the last development period, or
                True to estimate it with :func:`estimate_tail_factor`
            tail_se: Standard error of the tail factor
            tail_sigma: Sigma of the tail development step

        Following Mack (1999), when ``tail_se`` or ``tail_sigma`` is not given
        it is read off a log-linear trend of the standard errors (or sigmas)
        of the development factors, at the development step where a factor
        equal to the tail would sit on the log-linear trend of ``f - 1``.
        These are judgemental quantities and should be reviewed.
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

        factors, sigma, f_se = self._estimate_parameters()
        self.tail = self._resolve_tail(tail, factors)
        self.tail_se, self.tail_sigma = self._tail_uncertainty(
            factors, sigma, f_se, tail_se, tail_sigma
        )
        self._fit(factors)
        self._standard_errors(sigma, f_se)

    def _tail_uncertainty(self, factors, sigma, f_se, tail_se, tail_sigma):
        """Standard error and sigma of the tail step, estimating those not supplied."""
        if self.tail == 1.0:
            return float(tail_se or 0.0), float(tail_sigma or 0.0)
        if tail_se is not None and tail_sigma is not None:
            return float(tail_se), float(tail_sigma)

        factor_fit = _loglinear_fit(factors - 1)
        if self.tail <= 1 or factor_fit is None or factor_fit[1] >= 0:
            raise ValueError(
                "tail_se and tail_sigma cannot be estimated for this tail factor; supply both"
            )
        # Development step at which a factor equal to the tail would sit
        position = (np.log(self.tail - 1) - factor_fit[0]) / factor_fit[1]

        def extrapolate(values, name):
            fit = _loglinear_fit(values)
            if fit is None:
                raise ValueError(f"{name} cannot be estimated; supply it")
            return float(np.exp(fit[0] + fit[1] * position))

        if tail_se is None:
            tail_se = extrapolate(f_se, "tail_se")
        if tail_sigma is None:
            tail_sigma = extrapolate(sigma, "tail_sigma")
        return float(tail_se), float(tail_sigma)

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

        # The tail is one more development step, applied to every origin period
        if self.tail != 1.0:
            last = full[:, -1]
            process2 = process2 * self.tail ** 2 + self.tail_sigma ** 2 * last ** (2 - self.alpha)
            parameter2 = parameter2 * self.tail ** 2 + last ** 2 * self.tail_se ** 2
            total_parameter2 = (total_parameter2 * self.tail ** 2
                                + last.sum() ** 2 * self.tail_se ** 2)

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
