"""
Munich Chain-Ladder

The chain-ladder applied separately to paid and to incurred claims usually
gives two different ultimates for the same origin period. The Munich
chain-ladder of Quarg and Mack projects the two triangles together: an
origin period whose paid claims are low relative to its incurred claims is
expected to pay faster than average in future, and one whose incurred
claims are high relative to paid is expected to develop more slowly. The
two projections are pulled towards each other.

Reference: Quarg, G. and Mack, T. (2004). Munich chain ladder. Blatter
DGVFM 26(4), 597-630.
"""

import numpy as np
import pandas as pd

from .triangle import Triangle


def _extrapolate(values: np.ndarray) -> np.ndarray:
    """
    Fill a squared parameter that cannot be estimated (a step with a single
    link ratio) from a log-linear trend of the others, as the R ChainLadder
    package does by default.
    """
    values = values.copy()
    missing = np.flatnonzero(np.isnan(values))
    if len(missing) == 0:
        return values
    steps = np.arange(1, len(values) + 1)
    known = ~np.isnan(values) & (values > 0)
    if known.sum() < 2:
        raise ValueError("Not enough development periods for the Munich chain-ladder")
    slope, intercept = np.polyfit(steps[known], 0.5 * np.log(values[known]), 1)
    values[missing] = np.exp(2 * (intercept + slope * (missing + 1)))
    return values


def _mack_parameters(cum: np.ndarray):
    """Volume-weighted factors, sigma and standardised residuals of the link ratios."""
    n_steps = cum.shape[1] - 1
    f = np.full(n_steps, np.nan)
    sigma2 = np.full(n_steps, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = cum[:, 1:] / cum[:, :-1]
    for k in range(n_steps):
        rows = ~np.isnan(cum[:, k]) & ~np.isnan(cum[:, k + 1])
        f[k] = cum[rows, k + 1].sum() / cum[rows, k].sum()
        if rows.sum() > 1:
            sigma2[k] = np.sum(cum[rows, k] * (ratios[rows, k] - f[k]) ** 2) / (rows.sum() - 1)
    single = np.isnan(sigma2)
    sigma = np.sqrt(_extrapolate(sigma2))
    with np.errstate(divide="ignore", invalid="ignore"):
        residuals = (ratios - f) * np.sqrt(cum[:, :-1]) / sigma
    # A step with one link ratio has a residual of zero by construction and
    # carries no information about the correlation
    residuals[:, single] = np.nan
    return f, sigma, residuals


def _ratio_parameters(numerator: np.ndarray, denominator: np.ndarray):
    """Weighted average ratio by development period, its spread, and residuals."""
    n_dev = numerator.shape[1]
    q = np.full(n_dev, np.nan)
    rho2 = np.full(n_dev, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = numerator / denominator
    for k in range(n_dev):
        rows = ~np.isnan(numerator[:, k]) & ~np.isnan(denominator[:, k])
        q[k] = numerator[rows, k].sum() / denominator[rows, k].sum()
        if rows.sum() > 1:
            rho2[k] = np.sum(denominator[rows, k] * (ratios[rows, k] - q[k]) ** 2) \
                / (rows.sum() - 1)
    rho = np.sqrt(rho2)
    with np.errstate(divide="ignore", invalid="ignore"):
        residuals = (ratios - q) * np.sqrt(denominator) / rho
    return q, rho, residuals


class MunichChainLadder:
    """
    Munich chain-ladder projection of a paid and an incurred triangle.

    Attributes:
        lambda_paid: Slope linking paid development to the incurred/paid ratio
        lambda_incurred: Slope linking incurred development to the
            paid/incurred ratio
        paid_full, incurred_full: The two triangles projected to ultimate
        ultimate_paid, ultimate_incurred: Ultimate claims from each projection
    """

    def __init__(self, paid: Triangle, incurred: Triangle):
        """
        Args:
            paid: Triangle of paid claims
            incurred: Triangle of incurred claims, covering the same cells
        """
        if not isinstance(paid, Triangle) or not isinstance(incurred, Triangle):
            raise TypeError("paid and incurred must be Triangles")
        paid, incurred = paid.to_cumulative(), incurred.to_cumulative()
        paid._require_complete("The Munich chain-ladder")
        incurred._require_complete("The Munich chain-ladder")
        p, c = paid.values, incurred.values
        if p.shape != c.shape or not np.array_equal(np.isnan(p), np.isnan(c)):
            raise ValueError("paid and incurred must cover the same cells")
        if np.any(p[~np.isnan(p)] <= 0) or np.any(c[~np.isnan(c)] <= 0):
            raise ValueError("The Munich chain-ladder needs positive paid and incurred claims")

        self.paid, self.incurred = paid, incurred
        f_p, sigma_p, res_fp = _mack_parameters(p)
        f_i, sigma_i, res_fi = _mack_parameters(c)
        q, rho_i, res_q = _ratio_parameters(p, c)          # paid / incurred, weighted by incurred
        q_inv, rho_p, res_q_inv = _ratio_parameters(c, p)  # incurred / paid, weighted by paid

        def slope(x, y):
            x, y = x[:, :-1], y
            keep = np.isfinite(x) & np.isfinite(y)
            spread = np.sum(x[keep] ** 2)
            # No variation in the paid/incurred ratio: nothing to adjust for
            return float(np.sum(x[keep] * y[keep]) / spread) if spread > 1e-12 else 0.0

        self.lambda_paid = slope(res_q_inv, res_fp)
        self.lambda_incurred = slope(res_q, res_fi)

        n_origin, n_dev = p.shape
        paid_full, incurred_full = p.copy(), c.copy()
        latest = paid._latest_index()
        for i, j0 in enumerate(latest):
            for k in range(j0, n_dev - 1):
                pk, ck = paid_full[i, k], incurred_full[i, k]
                paid_factor = f_p[k]
                incurred_factor = f_i[k]
                if rho_p[k] > 0 and np.isfinite(rho_p[k]):
                    paid_factor += self.lambda_paid * sigma_p[k] / rho_p[k] * (ck / pk - q_inv[k])
                if rho_i[k] > 0 and np.isfinite(rho_i[k]):
                    incurred_factor += (self.lambda_incurred * sigma_i[k] / rho_i[k]
                                        * (pk / ck - q[k]))
                paid_full[i, k + 1] = pk * paid_factor
                incurred_full[i, k + 1] = ck * incurred_factor

        index = pd.Index(paid.origin, name="origin")
        columns = pd.Index(paid.development, name="development")
        self.paid_full = pd.DataFrame(paid_full, index=index, columns=columns)
        self.incurred_full = pd.DataFrame(incurred_full, index=index, columns=columns)
        self.latest_paid = paid.latest_diagonal()
        self.latest_incurred = incurred.latest_diagonal()
        self.ultimate_paid = pd.Series(paid_full[:, -1], index=index, name="ultimate_paid")
        self.ultimate_incurred = pd.Series(incurred_full[:, -1], index=index,
                                           name="ultimate_incurred")
        self.factors_paid = pd.Series(f_p, index=paid._link_labels(), name="factor_paid")
        self.factors_incurred = pd.Series(f_i, index=paid._link_labels(), name="factor_incurred")

    def summary(self, total: bool = True) -> pd.DataFrame:
        """
        Latest and ultimate paid and incurred claims with their ratios.

        Args:
            total: Add a "Total" row
        """
        table = pd.DataFrame({
            "latest_paid": self.latest_paid,
            "latest_incurred": self.latest_incurred,
            "latest_ratio": self.latest_paid / self.latest_incurred,
            "ultimate_paid": self.ultimate_paid,
            "ultimate_incurred": self.ultimate_incurred,
            "ultimate_ratio": self.ultimate_paid / self.ultimate_incurred,
        })
        if total:
            sums = table[["latest_paid", "latest_incurred", "ultimate_paid",
                          "ultimate_incurred"]].sum()
            table.loc["Total"] = [sums["latest_paid"], sums["latest_incurred"],
                                  sums["latest_paid"] / sums["latest_incurred"],
                                  sums["ultimate_paid"], sums["ultimate_incurred"],
                                  sums["ultimate_paid"] / sums["ultimate_incurred"]]
        return table

    def __repr__(self) -> str:
        return f"MunichChainLadder\n{self.summary().to_string()}"
