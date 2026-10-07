"""
Generalised Linear Model Reserving

The over-dispersed Poisson model of incremental claims, fitted as a
generalised linear model with a log link. With an effect for each origin
period and each development period it gives exactly the chain-ladder
reserve (Renshaw and Verrall), with a prediction error calculated from the
model in place of a bootstrap.

Its value is that the structure can be changed. Replacing the origin effects
by calendar year effects gives a model of the kind behind the separation
method, in which claims inflation is estimated from the triangle and has to
be assumed for the future; that is the structure the chain-ladder lacks when
inflation changes.

References
----------
- Renshaw, A.E. and Verrall, R.J. (1998). A stochastic model underlying the
  chain-ladder technique. British Actuarial Journal 4(4), 903-923.
- England, P.D. and Verrall, R.J. (2002). Stochastic claims reserving in
  general insurance. British Actuarial Journal 8(3), 443-518.
- Taylor, G.C. (1977). Separation of inflation and other effects from the
  distribution of non-life insurance claim delays. ASTIN Bulletin 9, 217-230.
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .triangle import Triangle


class GLMReserving:
    """
    Over-dispersed Poisson reserving model with a log link.

    ``structure="origin"`` fits an effect for each origin period and each
    development period, which reproduces the chain-ladder.

    ``structure="calendar"`` fits an effect for each development period and
    each calendar period, with one level of claims for all origin periods
    (scaled by ``exposure`` if given). It suits a portfolio of steady size
    whose claims are driven by inflation along the calendar. Future calendar
    effects are not in the data and are set by ``future_inflation``.

    Attributes:
        dispersion: Pearson scale parameter
        fitted: Fitted incremental claims for the whole rectangle
        latest, ultimate, ibnr: By origin period
        prediction_error: Root mean squared error of prediction of the
            reserve, by origin period
        total_ibnr, total_prediction_error: For all origin periods together
        calendar_inflation: Estimated rate of inflation between successive
            calendar periods (``structure="calendar"`` only)
    """

    def __init__(self,
                 triangle: Triangle,
                 structure: str = "origin",
                 exposure: Optional[Sequence[float]] = None,
                 future_inflation: Optional[float] = None,
                 trend_periods: Optional[int] = None):
        """
        Args:
            triangle: Claims triangle (cumulative or incremental), without
                missing cells
            structure: "origin" (chain-ladder) or "calendar"
            exposure: Measure of the volume of business of each origin
                period, used by the "calendar" structure
            future_inflation: Rate of inflation assumed for each future
                calendar period under the "calendar" structure. If None, the
                average of the estimated rates over the last
                ``trend_periods`` calendar periods is used.
            trend_periods: Number of latest calendar periods averaged when
                ``future_inflation`` is None (None for all of them)
        """
        if not isinstance(triangle, Triangle):
            raise TypeError("triangle must be a Triangle")
        if structure not in ("origin", "calendar"):
            raise ValueError('structure must be "origin" or "calendar"')
        self.triangle = triangle.to_cumulative()
        self.triangle._require_complete("GLM reserving")
        self.structure = structure
        n_origin, n_dev = self.triangle.n_origin, self.triangle.n_development
        if exposure is None:
            exposure = np.ones(n_origin)
        exposure = np.asarray(exposure, dtype=float)
        if exposure.shape != (n_origin,) or np.any(exposure <= 0):
            raise ValueError("exposure must hold one positive value for each origin period")
        if structure == "origin" and not np.allclose(exposure, exposure[0]):
            raise ValueError('exposure has no effect with structure="origin"; '
                             'the origin effects absorb it')
        self.exposure = exposure

        incremental = self.triangle.to_incremental().values
        observed = ~np.isnan(incremental)
        rows, cols = np.nonzero(np.ones((n_origin, n_dev), dtype=bool))
        calendar = rows + cols
        seen = observed[rows, cols]
        last_calendar = int(calendar[seen].max())

        design = self._design(rows, cols, calendar, n_origin, n_dev, last_calendar)
        offset = np.log(exposure)[rows]
        y = incremental[rows, cols][seen]
        if np.any(np.bincount(cols[seen], weights=y, minlength=n_dev) <= 0):
            raise ValueError("Every development period must have positive total incremental "
                             "claims for the over-dispersed Poisson model")

        # A column of the design with no observed cell cannot be estimated
        used = np.abs(design[seen]).sum(axis=0) > 0
        beta = np.zeros(design.shape[1])
        beta[used], covariance, mu_seen = self._irls(design[seen][:, used], y, offset[seen])
        n_parameters = int(used.sum())
        if len(y) <= n_parameters:
            raise ValueError("The triangle has too few cells for this model")
        self.dispersion = float(np.sum((y - mu_seen) ** 2 / mu_seen) / (len(y) - n_parameters))
        self.n_parameters = n_parameters

        future_design = design[~seen].copy()
        future_offset = offset[~seen].copy()
        if structure == "calendar":
            self._set_future_calendar(beta, n_dev, last_calendar, future_inflation, trend_periods)
            future_offset = future_offset + self._future_level[calendar[~seen] - last_calendar - 1]
            future_design[:, n_dev:] = 0.0
            future_design[:, n_dev - 1 + last_calendar] = 1.0 if last_calendar > 0 else 0.0
        mu_future = np.exp(future_design @ beta + future_offset)

        fitted = np.empty((n_origin, n_dev))
        fitted[rows[seen], cols[seen]] = mu_seen
        fitted[rows[~seen], cols[~seen]] = mu_future
        origin_index = pd.Index(self.triangle.origin, name="origin")
        self.fitted = pd.DataFrame(fitted, index=origin_index,
                                   columns=pd.Index(self.triangle.development,
                                                    name="development"))

        future_rows = rows[~seen]
        reserve = np.bincount(future_rows, weights=mu_future, minlength=n_origin)
        # Delta method: the reserve is a sum of exp(linear predictor)
        gradient = np.zeros((n_origin, n_parameters))
        np.add.at(gradient, future_rows, future_design[:, used] * mu_future[:, None])
        cov = self.dispersion * covariance
        estimation = np.einsum("ip,pq,iq->i", gradient, cov, gradient)
        total_gradient = gradient.sum(axis=0)
        self.latest = self.triangle.latest_diagonal()
        self.ibnr = pd.Series(reserve, index=origin_index, name="ibnr")
        self.ultimate = (self.latest + self.ibnr).rename("ultimate")
        self.prediction_error = pd.Series(np.sqrt(self.dispersion * reserve + estimation),
                                          index=origin_index, name="prediction_error")
        self.total_ibnr = float(reserve.sum())
        self.total_prediction_error = float(np.sqrt(
            self.dispersion * reserve.sum() + total_gradient @ cov @ total_gradient))
        self._future = (future_rows, cols[~seen], mu_future)

    # ------------------------------------------------------------------
    def _design(self, rows, cols, calendar, n_origin, n_dev, last_calendar):
        """Intercept, development effects, then origin or calendar effects."""
        blocks = [np.ones((len(rows), 1)),
                  (cols[:, None] == np.arange(1, n_dev)[None, :]).astype(float)]
        if self.structure == "origin":
            blocks.append((rows[:, None] == np.arange(1, n_origin)[None, :]).astype(float))
        else:
            blocks.append((calendar[:, None] == np.arange(1, last_calendar + 1)[None, :])
                          .astype(float))
        return np.hstack(blocks)

    @staticmethod
    def _irls(design, y, offset, iterations: int = 100, tolerance: float = 1e-10):
        """Iteratively reweighted least squares for a Poisson log-link model."""
        mu = np.maximum(y, 0.0) + max(float(np.mean(y)), 1e-8) * 0.1
        eta = np.log(mu)
        beta = np.zeros(design.shape[1])
        for _ in range(iterations):
            working = eta - offset + (y - mu) / mu
            weighted = design * mu[:, None]
            information = design.T @ weighted
            new_beta = np.linalg.solve(information, weighted.T @ working)
            eta = design @ new_beta + offset
            mu = np.exp(eta)
            if np.max(np.abs(new_beta - beta)) < tolerance:
                beta = new_beta
                break
            beta = new_beta
        else:
            raise RuntimeError("The model did not converge")
        covariance = np.linalg.inv(design.T @ (design * mu[:, None]))
        return beta, covariance, mu

    def _set_future_calendar(self, beta, n_dev, last_calendar, future_inflation, trend_periods):
        effects = np.concatenate([[0.0], beta[n_dev:n_dev + last_calendar]])
        changes = np.diff(effects)
        labels = [f"{t}-{t + 1}" for t in range(1, last_calendar + 1)]
        self.calendar_inflation = pd.Series(np.exp(changes) - 1,
                                            index=pd.Index(labels, name="calendar periods"),
                                            name="inflation")
        if future_inflation is None:
            if len(changes) == 0:
                raise ValueError("future_inflation is needed: the triangle has one diagonal")
            recent = changes if trend_periods is None else changes[-int(trend_periods):]
            log_rate = float(np.mean(recent))
        else:
            if future_inflation <= -1:
                raise ValueError("future_inflation must be above -100%")
            log_rate = float(np.log1p(future_inflation))
        self.future_inflation = float(np.expm1(log_rate))
        horizon = self.triangle.n_origin + self.triangle.n_development
        self._future_level = log_rate * np.arange(1, horizon + 1)

    # ------------------------------------------------------------------
    def future_incremental(self) -> pd.DataFrame:
        """Expected incremental claims in the cells not yet observed."""
        rows, cols, mu = self._future
        table = np.full(self.fitted.shape, np.nan)
        table[rows, cols] = mu
        return pd.DataFrame(table, index=self.fitted.index, columns=self.fitted.columns)

    def residuals(self) -> pd.DataFrame:
        """Pearson residuals of the observed cells, scaled by the dispersion."""
        incremental = self.triangle.to_incremental().values
        fitted = self.fitted.to_numpy()
        scaled = (incremental - fitted) / np.sqrt(self.dispersion * fitted)
        return pd.DataFrame(scaled, index=self.fitted.index, columns=self.fitted.columns)

    def summary(self, total: bool = True) -> pd.DataFrame:
        """Latest, ultimate, reserve and prediction error by origin period."""
        table = pd.DataFrame({
            "latest": self.latest, "ultimate": self.ultimate, "ibnr": self.ibnr,
            "prediction_error": self.prediction_error,
        })
        if total:
            table.loc["Total"] = [self.latest.sum(), self.ultimate.sum(), self.total_ibnr,
                                  self.total_prediction_error]
        with np.errstate(divide="ignore", invalid="ignore"):
            table["cv"] = table["prediction_error"] / table["ibnr"]
        return table

    def __repr__(self) -> str:
        return (f"GLMReserving(structure='{self.structure}', "
                f"total_ibnr={self.total_ibnr:,.0f}, "
                f"prediction_error={self.total_prediction_error:,.0f})")
