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
- Buchwalder, M., Buhlmann, H., Merz, M. and Wuthrich, M.V. (2006). The mean
  square error of prediction in the chain ladder reserving method (Mack and
  Murphy revisited). ASTIN Bulletin 36(2), 521-542.
- Gesmann, M., Murphy, D., Zhang, Y., Carrato, A., Wuthrich, M., Concina, F.
  and Dal Moro, E. ChainLadder: Statistical Methods and Models for Claims
  Reserving in General Insurance. R package.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Optional, Sequence, Tuple, Union
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
                 tail: Union[float, bool] = 1.0,
                 factors: Optional[Sequence[float]] = None):
        """
        Fit the chain-ladder method.

        Args:
            triangle: Claims triangle (cumulative or incremental)
            average: "volume" for volume-weighted development factors,
                "simple" for the arithmetic mean of the link ratios
            n_periods: Average only the latest n origin periods (None for all)
            tail: Tail factor applied after the last development period, or
                True to estimate it with :func:`estimate_tail_factor`
            factors: Selected development factors to use instead of those
                estimated from the triangle, one per development step. Use
                None (or NaN) for a step to keep the estimated factor.
        """
        if not isinstance(triangle, Triangle):
            raise TypeError("triangle must be a Triangle")

        self.triangle = triangle.to_cumulative()
        self.average = average
        self.n_periods = n_periods
        self.selected_factors = factors
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
        estimated = self.triangle.development_factors(self.average, self.n_periods).to_numpy()
        if self.selected_factors is None:
            return estimated
        selected = np.array([np.nan if f is None else f for f in self.selected_factors],
                            dtype=float)
        if selected.shape != estimated.shape:
            raise ValueError(
                f"factors needs one value for each of the {len(estimated)} development steps"
            )
        if np.any(selected[~np.isnan(selected)] <= 0):
            raise ValueError("factors must be positive")
        return np.where(np.isnan(selected), estimated, selected)

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

    # ------------------------------------------------------------------
    # Future payments
    # ------------------------------------------------------------------
    def future_incremental(self) -> pd.DataFrame:
        """
        Projected incremental claims for the cells not yet observed.

        Returns:
            DataFrame by origin and development period, NaN for observed
            cells. When the projection includes a tail, the amount expected
            after the last development period is in a final "tail" column.
        """
        full = self._full
        incremental = full.copy()
        incremental[:, 1:] = np.diff(full, axis=1)
        incremental[~np.isnan(self.triangle.values)] = np.nan
        table = pd.DataFrame(incremental, index=self.full_triangle.index,
                             columns=list(self.full_triangle.columns))
        tail_amount = self.ultimate.to_numpy() - full[:, -1]
        if np.any(np.abs(tail_amount) > 1e-9 * np.maximum(1.0, np.abs(full[:, -1]))):
            table["tail"] = tail_amount
        return table

    def cash_flows(self) -> pd.Series:
        """
        Expected future claims by period of payment.

        Period 1 is the development period following the valuation date,
        period 2 the one after, and so on. A tail amount is placed in the
        period after the origin period reaches the last development period.

        Returns:
            Series of expected payments indexed by future period
        """
        future = self.future_incremental().to_numpy()
        periods_ahead = np.arange(future.shape[1])[None, :] - self._latest_idx[:, None]
        valid = ~np.isnan(future) & (periods_ahead > 0)
        n_periods = int(periods_ahead[valid].max()) if valid.any() else 0
        flows = np.zeros(n_periods)
        np.add.at(flows, periods_ahead[valid] - 1, future[valid])
        return pd.Series(flows, index=pd.RangeIndex(1, n_periods + 1, name="period"),
                         name="cash_flow")

    def discounted_reserve(self,
                           discount_rate,
                           timing: float = 0.5,
                           periods_per_year: int = 1) -> float:
        """
        Present value of the expected future claims.

        Args:
            discount_rate: Annual effective rate of interest, or a yield
                curve object with a ``get_discount_factor(years)`` method
                such as :class:`actuneo.finance.YieldCurve`
            timing: When payments fall within each period: 0.5 for mid-period
                (the default), 0 for the start, 1 for the end
            periods_per_year: Development periods in a year (1 for annual
                triangles, 4 for quarterly, 12 for monthly)

        Returns:
            Discounted reserve. Compare with :meth:`reserve` for the
            undiscounted figure.
        """
        if not 0 <= timing <= 1:
            raise ValueError("timing must be between 0 and 1")
        flows = self.cash_flows()
        years = (flows.index.to_numpy() - 1 + timing) / periods_per_year
        if hasattr(discount_rate, "get_discount_factor"):
            factors = np.array([discount_rate.get_discount_factor(t) for t in years])
        else:
            if discount_rate <= -1:
                raise ValueError("discount_rate must be greater than -1")
            factors = (1 + discount_rate) ** -years
        return float(np.sum(flows.to_numpy() * factors))

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def to_excel(self, path: str) -> None:
        """
        Write the results to an Excel workbook (needs openpyxl).

        Sheets: Summary, Triangle, Projection, Factors and Cash flows.

        Args:
            path: File name of the workbook, ending in .xlsx
        """
        try:
            import openpyxl  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Writing Excel files requires openpyxl. Install it with: "
                "pip install actuneo[excel]"
            ) from exc

        factors = pd.DataFrame({"factor": self.factors})
        factors["cdf"] = self.cdf.to_numpy()[:-1]
        for name in ("sigma", "f_se"):
            if hasattr(self, name):
                factors[name] = getattr(self, name)
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            self.summary().to_excel(writer, sheet_name="Summary")
            self.triangle.to_frame().to_excel(writer, sheet_name="Triangle")
            self.full_triangle.to_excel(writer, sheet_name="Projection")
            factors.to_excel(writer, sheet_name="Factors")
            self.cash_flows().to_frame().to_excel(writer, sheet_name="Cash flows")

    def plot(self, ax=None):
        """
        Bar chart of latest claims and reserve by origin period (needs
        matplotlib). Mack's standard error is shown as error bars when the
        model provides it.

        Args:
            ax: Matplotlib axes to draw on (a new figure if omitted)

        Returns:
            The matplotlib axes
        """
        from ._plotting import plot_reserves
        return plot_reserves(self.latest, self.ibnr, getattr(self, "mack_se", None),
                             f"{self.triangle.name}: latest claims and reserve", ax)

    # ------------------------------------------------------------------
    # Model checks
    # ------------------------------------------------------------------
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
                 tail_sigma: Optional[float] = None,
                 mse_method: str = "mack"):
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
            mse_method: "mack" for Mack's formula for the estimation error,
                "independence" to include the cross-product term of
                Murphy (1994) and Buchwalder, Buhlmann, Merz and Wuthrich
                (2006), which gives a slightly larger parameter risk

        Following Mack (1999), when ``tail_se`` or ``tail_sigma`` is not given
        it is read off a log-linear trend of the standard errors (or sigmas)
        of the development factors, at the development step where a factor
        equal to the tail would sit on the log-linear trend of ``f - 1``.
        These are judgemental quantities and should be reviewed.
        """
        if est_sigma not in ("log-linear", "mack"):
            raise ValueError("est_sigma must be 'log-linear' or 'mack'")
        if mse_method not in ("mack", "independence"):
            raise ValueError("mse_method must be 'mack' or 'independence'")
        self.mse_method = mse_method
        if not isinstance(triangle, Triangle):
            raise TypeError("triangle must be a Triangle")

        self.triangle = triangle.to_cumulative()
        self.alpha = float(alpha)
        self.est_sigma = est_sigma
        self.selected_factors = None
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
        if np.any(cum[~np.isnan(cum)] < 0):
            raise ValueError("Mack's model cannot be fitted to negative cumulative claims")

        n_steps = cum.shape[1] - 1
        factors = np.full(n_steps, np.nan)
        sigma2 = np.full(n_steps, np.nan)
        weight_sum = np.full(n_steps, np.nan)

        for k in range(n_steps):
            # A link ratio needs positive claims at the start of the step
            with np.errstate(invalid="ignore"):
                rows = ~np.isnan(cum[:, k + 1]) & (cum[:, k] > 0)
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
        cross = 1.0 if self.mse_method == "independence" else 0.0
        process2 = np.zeros(n_origin)
        parameter2 = np.zeros(n_origin)
        for i, j0 in enumerate(self._latest_idx):
            for k in range(j0, n_dev - 1):
                c = full[i, k]
                process2[i] = process2[i] * f[k] ** 2 + sigma[k] ** 2 * c ** (2 - self.alpha)
                parameter2[i] = (parameter2[i] * (f[k] ** 2 + cross * f_se[k] ** 2)
                                 + c ** 2 * f_se[k] ** 2)

        # Total: process risk adds across independent origin periods, while the
        # estimation error of each factor is common to all origins it projects
        total_parameter2 = 0.0
        for k in range(n_dev - 1):
            projected = self._latest_idx <= k
            total_parameter2 = (total_parameter2 * (f[k] ** 2 + cross * f_se[k] ** 2)
                                + full[projected, k].sum() ** 2 * f_se[k] ** 2)

        # The tail is one more development step, applied to every origin period
        if self.tail != 1.0:
            last = full[:, -1]
            process2 = process2 * self.tail ** 2 + self.tail_sigma ** 2 * last ** (2 - self.alpha)
            tail_growth = self.tail ** 2 + cross * self.tail_se ** 2
            parameter2 = parameter2 * tail_growth + last ** 2 * self.tail_se ** 2
            total_parameter2 = (total_parameter2 * tail_growth
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

    def residuals(self) -> pd.DataFrame:
        """
        Standardised residuals of the individual link ratios,
        ``(F[i, k] - f[k]) * sqrt(C[i, k]**alpha) / sigma[k]``.

        Under the model they have mean 0 and a similar spread in every
        development step, origin period and calendar period. Trends point to
        a breach of the chain-ladder assumptions.
        """
        cum = self.triangle.values
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = cum[:, 1:] / cum[:, :-1]
            standardised = ((ratios - self._f) * np.sqrt(cum[:, :-1] ** self.alpha)
                            / self.sigma.to_numpy())
        standardised[~np.isfinite(standardised)] = np.nan
        return pd.DataFrame(standardised, index=self.full_triangle.index,
                            columns=self.triangle._link_labels())

    def reserve_quantile(self, q, distribution: str = "lognormal") -> pd.DataFrame:
        """
        Quantiles of the reserve from a distribution fitted to the estimated
        reserve and its Mack standard error.

        Mack's method gives a mean and a standard error but no distribution,
        so a shape has to be assumed. The lognormal is the usual choice
        because reserves are positive and skewed.

        Args:
            q: Probability or list of probabilities, for example [0.75, 0.995]
            distribution: "lognormal" or "normal"

        Returns:
            DataFrame of reserve quantiles by origin period with a "Total"
            row. A lognormal quantile is NaN where the reserve is not positive.
        """
        from scipy import stats
        if distribution not in ("lognormal", "normal"):
            raise ValueError("distribution must be 'lognormal' or 'normal'")
        probabilities = np.atleast_1d(np.asarray(q, dtype=float))
        if np.any((probabilities <= 0) | (probabilities >= 1)):
            raise ValueError("probabilities must be between 0 and 1")

        mean = np.append(self.ibnr.to_numpy(), self.total_ibnr)
        se = np.append(self.mack_se.to_numpy(), self.total_mack_se)
        z = stats.norm.ppf(probabilities)[None, :]
        if distribution == "normal":
            values = mean[:, None] + z * se[:, None]
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                log_var = np.log1p((se / mean) ** 2)
                values = np.exp((np.log(mean) - log_var / 2)[:, None]
                                + z * np.sqrt(log_var)[:, None])
            values[mean <= 0] = np.nan
        index = pd.Index(list(self.ibnr.index) + ["Total"], name="origin")
        return pd.DataFrame(values, index=index,
                            columns=[f"ibnr_{100 * p:g}%" for p in probabilities])

    def plot_residuals(self, ax=None):
        """
        Plot the standardised residuals against development step (needs
        matplotlib).

        Args:
            ax: Matplotlib axes to draw on (a new figure if omitted)

        Returns:
            The matplotlib axes
        """
        from ._plotting import plot_residuals
        return plot_residuals(self.residuals(), f"{self.triangle.name}: Mack residuals", ax)

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
