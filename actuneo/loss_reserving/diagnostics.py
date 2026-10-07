"""
Reserving Diagnostics

Tests of whether a triangle satisfies the assumptions behind the
chain-ladder, and checks of how well a method would have predicted the past.

The chain-ladder assumes that every origin period develops in the same way,
that one period's development says nothing about the next step beyond its
current level, and that origin periods are independent. In practice these
fail when something changes along the calendar: claims inflation jumps,
claims are settled faster or slower, case reserving is strengthened, or the
mix of business shifts. The functions here look for the footprints of such
changes.

References
----------
- Mack, T. (1994). Measuring the variability of chain ladder reserve
  estimates. Casualty Actuarial Society Forum, Spring 1994. (Appendices G
  and H give the correlation and calendar year tests.)
- Venter, G.G. (1998). Testing the assumptions of age-to-age factors.
  Proceedings of the Casualty Actuarial Society 85, 807-847.
"""

from math import comb
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .triangle import Triangle
from .chain_ladder import ChainLadder


def _link_ratios(triangle: Triangle) -> np.ndarray:
    cum = triangle.to_cumulative().values
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = cum[:, 1:] / cum[:, :-1]
    ratios[~np.isfinite(ratios)] = np.nan
    return ratios


def calendar_year_effect_test(triangle: Triangle, confidence: float = 0.95) -> dict:
    """
    Mack's test for calendar year effects.

    Within each development step the link ratios are marked as small (below
    the median of the step) or large (above it). If nothing changes along
    the calendar, each diagonal should hold a similar number of small and
    large ratios. A diagonal dominated by one kind points to an effect in
    that calendar year, such as a jump in inflation or a change in claims
    handling.

    Args:
        triangle: Claims triangle
        confidence: Confidence level of the acceptance range

    Returns:
        Dictionary with the statistic ``z``, its expected value and variance
        if there is no effect, the acceptance ``lower`` and ``upper``
        bounds, ``effect`` (True if the statistic falls outside them), and
        the counts by diagonal
    """
    from scipy import stats
    ratios = _link_ratios(triangle)
    n_origin, n_steps = ratios.shape
    marks = np.zeros(ratios.shape)  # +1 large, -1 small, 0 excluded
    for k in range(n_steps):
        column = ratios[:, k]
        known = ~np.isnan(column)
        if known.sum() == 0:
            continue
        median = np.median(column[known])
        marks[known & (column > median), k] = 1
        marks[known & (column < median), k] = -1

    rows = []
    z = expected = variance = 0.0
    calendar = np.add.outer(np.arange(n_origin), np.arange(n_steps))
    for diagonal in range(calendar.max() + 1):
        on_diagonal = (calendar == diagonal) & ~np.isnan(ratios)
        large = int(np.sum(marks[on_diagonal] == 1))
        small = int(np.sum(marks[on_diagonal] == -1))
        n = large + small
        if n < 2:
            continue
        m = (n - 1) // 2
        mean = n / 2 - comb(n - 1, m) * n / 2 ** n
        var = n * (n - 1) / 4 - comb(n - 1, m) * n * (n - 1) / 2 ** n + mean - mean ** 2
        z += min(small, large)
        expected += mean
        variance += var
        rows.append({"diagonal": diagonal + 1, "small": small, "large": large,
                     "z": min(small, large), "expected": mean, "variance": var})

    width = stats.norm.ppf(0.5 + confidence / 2) * np.sqrt(variance)
    lower, upper = expected - width, expected + width
    return {
        "z": float(z), "expected": float(expected), "variance": float(variance),
        "lower": float(lower), "upper": float(upper),
        "effect": bool(z < lower or z > upper),
        "by_diagonal": pd.DataFrame(rows),
    }


def development_factor_correlation_test(triangle: Triangle, confidence: float = 0.5) -> dict:
    """
    Mack's test for correlation between successive development factors.

    The chain-ladder assumes that a high link ratio in one step is not
    followed, on average, by a high or a low one in the next. For each pair
    of adjacent steps Spearman's rank correlation is calculated, and the
    correlations are combined into one weighted statistic.

    Args:
        triangle: Claims triangle
        confidence: Confidence level of the acceptance range. Mack suggests
            50%, a deliberately strict test, because the statistic combines
            many weak correlations.

    Returns:
        Dictionary with the statistic ``t``, its ``variance`` if there is no
        correlation, the acceptance ``lower`` and ``upper`` bounds,
        ``correlated`` (True if the statistic falls outside them) and the
        correlation of each pair of steps
    """
    from scipy import stats
    ratios = _link_ratios(triangle)
    n_steps = ratios.shape[1]
    rows = []
    weighted, weights = 0.0, 0.0
    for k in range(1, n_steps):
        both = ~np.isnan(ratios[:, k - 1]) & ~np.isnan(ratios[:, k])
        pairs = int(both.sum())
        if pairs < 2:
            continue
        r = stats.rankdata(ratios[both, k - 1])
        s = stats.rankdata(ratios[both, k])
        correlation = 1 - 6 * np.sum((r - s) ** 2) / (pairs ** 3 - pairs)
        weighted += (pairs - 1) * correlation
        weights += pairs - 1
        rows.append({"steps": f"{k}/{k + 1}", "pairs": pairs, "correlation": correlation})
    if weights == 0:
        raise ValueError("The triangle is too small to test for correlation")
    t = weighted / weights
    variance = 1 / weights
    width = stats.norm.ppf(0.5 + confidence / 2) * np.sqrt(variance)
    return {
        "t": float(t), "variance": float(variance),
        "lower": float(-width), "upper": float(width),
        "correlated": bool(abs(t) > width),
        "by_step": pd.DataFrame(rows),
    }


def intercept_test(triangle: Triangle, significance: float = 0.05) -> pd.DataFrame:
    """
    Test whether development is proportional to claims to date.

    The chain-ladder predicts next period's claims as a multiple of this
    period's, a line through the origin. For each development step, next
    period's cumulative claims are regressed on this period's with an
    intercept. A significant intercept means part of the development does
    not depend on claims to date, as a Bornhuetter-Ferguson or additive
    method assumes, and that the chain-ladder is a poor fit for that step.

    Args:
        triangle: Claims triangle
        significance: Level at which an intercept is flagged

    Returns:
        DataFrame by development step with the number of points, the
        intercept and slope, the p-value of the intercept and whether it is
        ``significant``. Steps with fewer than four points are not tested.
    """
    from scipy import stats
    cum = triangle.to_cumulative().values
    rows = []
    labels = triangle._link_labels()
    for k in range(cum.shape[1] - 1):
        both = ~np.isnan(cum[:, k]) & ~np.isnan(cum[:, k + 1])
        n = int(both.sum())
        row = {"step": labels[k], "points": n, "intercept": np.nan, "slope": np.nan,
               "p_value": np.nan, "significant": False}
        if n >= 4 and np.ptp(cum[both, k]) > 0:
            fit = stats.linregress(cum[both, k], cum[both, k + 1])
            x = cum[both, k]
            residual_variance = np.sum((cum[both, k + 1] - fit.intercept - fit.slope * x) ** 2) \
                / (n - 2)
            se = np.sqrt(residual_variance * (1 / n + x.mean() ** 2 / np.sum((x - x.mean()) ** 2)))
            p_value = 2 * stats.t.sf(abs(fit.intercept / se), n - 2) if se > 0 else 0.0
            row.update(intercept=fit.intercept, slope=fit.slope, p_value=float(p_value),
                       significant=bool(p_value < significance))
        rows.append(row)
    return pd.DataFrame(rows).set_index("step")


def link_ratio_trend_test(triangle: Triangle, significance: float = 0.05) -> pd.DataFrame:
    """
    Test for a trend in link ratios across origin periods.

    If claims are being settled faster, or reserved more strongly, the link
    ratios of a development step drift up or down from older to newer origin
    periods. For each step, the ranks of the link ratios are correlated with
    the order of the origin periods (Spearman), which is robust to outliers.

    Args:
        triangle: Claims triangle
        significance: Level at which a trend is flagged

    Returns:
        DataFrame by development step with the number of points, the rank
        correlation with origin order, its p-value and whether there is a
        significant ``trend``. Steps with fewer than five points are not
        tested.
    """
    from scipy import stats
    ratios = _link_ratios(triangle)
    rows = []
    labels = triangle._link_labels()
    for k in range(ratios.shape[1]):
        known = ~np.isnan(ratios[:, k])
        n = int(known.sum())
        row = {"step": labels[k], "points": n, "correlation": np.nan, "p_value": np.nan,
               "trend": False}
        if n >= 5 and np.ptp(ratios[known, k]) > 0:
            result = stats.spearmanr(np.flatnonzero(known), ratios[known, k])
            row.update(correlation=float(result.statistic), p_value=float(result.pvalue),
                       trend=bool(result.pvalue < significance))
        rows.append(row)
    return pd.DataFrame(rows).set_index("step")


def link_ratio_outliers(triangle: Triangle, threshold: float = 3.5,
                        min_points: int = 4) -> pd.DataFrame:
    """
    Link ratios that stand apart from the others in their development step.

    Uses a robust score: the distance from the median of the step in units
    of the median absolute deviation, scaled to be comparable with a
    standard deviation. Robust measures are used because one extreme ratio
    would otherwise hide itself by inflating the ordinary standard deviation.

    Args:
        triangle: Claims triangle
        threshold: Score beyond which a ratio is reported
        min_points: Fewest link ratios a step must have to be examined

    Returns:
        DataFrame with one row per outlying link ratio: origin period,
        development step, the ratio, the median of its step and the score.
        With only a handful of ratios in a step the score is itself very
        variable, so some ratios will be reported in triangles with nothing
        wrong; treat the list as cells to look at, not as errors.
    """
    ratios = _link_ratios(triangle)
    labels = triangle._link_labels()
    rows = []
    for k in range(ratios.shape[1]):
        known = ~np.isnan(ratios[:, k])
        if known.sum() < min_points:
            continue
        median = np.median(ratios[known, k])
        spread = np.median(np.abs(ratios[known, k] - median))
        if spread == 0:
            continue
        scores = 0.6745 * (ratios[:, k] - median) / spread
        for i in np.flatnonzero(known & (np.abs(scores) > threshold)):
            rows.append({"origin": triangle.origin[i], "step": labels[k],
                         "link_ratio": ratios[i, k], "median": median, "score": scores[i]})
    return pd.DataFrame(rows, columns=["origin", "step", "link_ratio", "median", "score"])


def backtest(triangle: Triangle,
             method: Optional[Callable[[Triangle], object]] = None,
             diagonals: int = 1) -> pd.DataFrame:
    """
    Actual against expected: how well a method would have predicted the
    latest diagonals from the triangle as it stood before them.

    The most recent diagonals are removed, the method is fitted to what is
    left, and its projection of the removed cells is compared with what
    actually happened.

    Args:
        triangle: Claims triangle
        method: Function taking a Triangle and returning a fitted model with
            a ``full_triangle`` (defaults to the basic chain-ladder)
        diagonals: Number of latest diagonals to hold out

    Returns:
        DataFrame by origin period with the actual and expected increase in
        cumulative claims over the held-out period, the difference and the
        ratio of actual to expected, with a "Total" row. The oldest origin
        periods are left out, because the earlier triangle holds nothing
        from which to project their latest development. A total far from
        zero, or differences of one sign, shows a bias the method would
        have had.
    """
    if diagonals < 1:
        raise ValueError("diagonals must be at least 1")
    triangle = triangle.to_cumulative()
    triangle._require_complete("Back-testing")
    values = triangle.values
    latest = triangle._latest_index()
    earlier = values.copy()
    keep = []
    for i, j0 in enumerate(latest):
        cut = j0 - diagonals
        if cut < 0:
            continue
        earlier[i, cut + 1:] = np.nan
        keep.append(i)
    if len(keep) < 2:
        raise ValueError("The triangle is too small to hold out that many diagonals")
    past = Triangle(earlier[keep], [triangle.origin[i] for i in keep], triangle.development,
                    cumulative=True, name=triangle.name)
    # Development periods that no remaining origin period has reached cannot be projected
    width = int(past._latest_index().max()) + 1
    past = Triangle(past.values[:, :width], past.origin, triangle.development[:width],
                    cumulative=True, name=triangle.name)
    model = (method or ChainLadder)(past)
    projected = model.full_triangle.to_numpy()

    rows = []
    for row, i in enumerate(keep):
        j0 = latest[i]
        start = j0 - diagonals
        if j0 >= projected.shape[1]:
            continue
        actual = values[i, j0] - values[i, start]
        expected = projected[row, j0] - values[i, start]
        rows.append({"origin": triangle.origin[i], "actual": actual, "expected": expected})
    table = pd.DataFrame(rows).set_index("origin")
    table.loc["Total"] = table.sum()
    table["difference"] = table["actual"] - table["expected"]
    with np.errstate(divide="ignore", invalid="ignore"):
        table["actual_to_expected"] = table["actual"] / table["expected"]
    return table


def diagnose(triangle: Triangle, significance: float = 0.05) -> pd.DataFrame:
    """
    Run the diagnostic tests and summarise what they found.

    The summary is stricter than the individual tests, so that a triangle
    with nothing wrong is seldom flagged. The tests made step by step (trend
    and intercept) divide the significance level by the number of steps
    tested, since otherwise one of several steps would often be flagged by
    chance. The correlation test uses a range at the same level in place of
    the 50% range Mack suggests, which by construction rejects half of all
    well-behaved triangles.

    Args:
        triangle: Claims triangle
        significance: Level at which each check raises its flag

    Returns:
        DataFrame with one row per check: whether it raised a ``flag``, the
        ``finding`` in words and what it ``suggests``. A triangle that
        passes every check is not proved suitable for the chain-ladder; the
        tests have little power on small triangles.
    """
    rows = []

    calendar = calendar_year_effect_test(triangle, confidence=1 - significance)
    rows.append({
        "check": "Calendar year effect",
        "flag": calendar["effect"],
        "finding": f"Z = {calendar['z']:.0f}, expected {calendar['expected']:.1f}, "
                   f"acceptable {calendar['lower']:.1f} to {calendar['upper']:.1f}",
        "suggests": "A change along the calendar, such as inflation or claims handling. "
                    "Consider the inflation-adjusted method or excluding the affected years."
                    if calendar["effect"] else "No evidence of a calendar year effect.",
    })

    correlation = development_factor_correlation_test(triangle, confidence=1 - significance)
    rows.append({
        "check": "Correlation between development factors",
        "flag": correlation["correlated"],
        "finding": f"T = {correlation['t']:.3f}, acceptable "
                   f"{correlation['lower']:.3f} to {correlation['upper']:.3f}",
        "suggests": "Successive development steps are related, which the chain-ladder "
                    "ignores. Its standard error will be unreliable."
                    if correlation["correlated"] else "No evidence of correlation.",
    })

    trend = link_ratio_trend_test(triangle)
    tested = max(int(trend["p_value"].notna().sum()), 1)
    trending = trend.index[trend["p_value"] < significance / tested].tolist()
    rows.append({
        "check": "Trend in link ratios by origin period",
        "flag": bool(trending),
        "finding": f"Trend in steps {trending}" if trending else "No significant trend",
        "suggests": "Development is changing over time, for example faster settlement. "
                    "Consider averaging recent origin periods only."
                    if trending else "Development looks stable across origin periods.",
    })

    intercept = intercept_test(triangle)
    tested = max(int(intercept["p_value"].notna().sum()), 1)
    with_intercept = intercept.index[intercept["p_value"] < significance / tested].tolist()
    rows.append({
        "check": "Development proportional to claims to date",
        "flag": bool(with_intercept),
        "finding": f"Significant intercept in steps {with_intercept}" if with_intercept
                   else "No significant intercept",
        "suggests": "Part of the development does not depend on claims to date. Consider "
                    "Bornhuetter-Ferguson or Cape Cod for immature periods."
                    if with_intercept else "Proportional development is not contradicted.",
    })

    outliers = link_ratio_outliers(triangle, threshold=6.0, min_points=6)
    rows.append({
        "check": "Outlying link ratios",
        "flag": len(outliers) > 0,
        "finding": f"{len(outliers)} outlying link ratio(s)" if len(outliers)
                   else "No outlying link ratios",
        "suggests": "Check the data for errors or large claims; consider excluding them "
                    "from the averages." if len(outliers) else "No single ratio dominates.",
    })

    try:
        check = backtest(triangle)
        ratio = check.loc["Total", "actual_to_expected"]
        off = bool(abs(ratio - 1) > 0.10)
        rows.append({
            "check": "Back-test of the latest diagonal",
            "flag": off,
            "finding": f"Actual was {ratio:.0%} of what the chain-ladder expected",
            "suggests": "The chain-ladder would have been materially wrong a period ago."
                        if off else "The chain-ladder would have predicted the latest "
                                    "diagonal within 10%.",
        })
    except ValueError:
        pass
    return pd.DataFrame(rows).set_index("check")
