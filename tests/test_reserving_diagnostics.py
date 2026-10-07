"""
Tests of the reserving diagnostics, the GLM and machine learning reserving
models, and the benchmark against simulated claims.

Published targets:

- Mack's calendar year and correlation tests on the RAA triangle, as given
  in the documentation of ``cyEffTest`` and ``dfCorTest`` in the R
  ChainLadder package.
- The over-dispersed Poisson model on the Taylor and Ashe triangle, from
  England and Verrall (2002), Table 1 and section 7.
"""

import numpy as np
import pytest

from actuneo.loss_reserving import (
    Triangle, ChainLadder, GLMReserving, load_raa, load_genins,
    calendar_year_effect_test, development_factor_correlation_test, intercept_test,
    link_ratio_trend_test, link_ratio_outliers, backtest, diagnose,
)
from actuneo.simulation import ClaimsSimulator, benchmark_reserving


def simulator(seed=0, **kwargs):
    return ClaimsSimulator(n_years=10, claims_per_year=400, claim_cv=1.0, inflation=0.03,
                           pattern_concentration=3, seed=seed, **kwargs)


class TestMackTests:
    def test_calendar_year_effect_matches_r(self):
        result = calendar_year_effect_test(load_raa())
        assert result["z"] == 14
        assert result["expected"] == pytest.approx(12.875)
        assert result["variance"] == pytest.approx(3.978516, abs=1e-6)
        assert result["lower"] == pytest.approx(8.965613, abs=1e-6)
        assert result["upper"] == pytest.approx(16.784387, abs=1e-6)
        assert result["effect"] is False

    def test_correlation_matches_r(self):
        result = development_factor_correlation_test(load_raa())
        assert result["t"] == pytest.approx(0.06955782, abs=1e-8)
        assert result["variance"] == pytest.approx(0.03571429, abs=1e-8)
        assert result["upper"] == pytest.approx(0.1274666, abs=1e-7)
        assert result["lower"] == pytest.approx(-0.1274666, abs=1e-7)
        assert result["correlated"] is False

    def test_inflation_shock_shows_as_calendar_effect(self):
        shocked = simulator().with_inflation_shock(2022, 0.40).triangle()
        assert calendar_year_effect_test(shocked)["effect"] is True

    def test_too_small_for_correlation(self):
        with pytest.raises(ValueError, match="too small"):
            development_factor_correlation_test(Triangle([[100, 150], [110, np.nan]]))


class TestStepTests:
    def test_exactly_proportional_triangle(self):
        factors = np.array([2.0, 1.5, 1.2, 1.1, 1.05])
        start = np.array([100.0, 120.0, 90.0, 150.0, 110.0, 130.0])
        values = np.full((6, 6), np.nan)
        for i in range(6):
            row = start[i] * np.concatenate([[1.0], np.cumprod(factors)])
            values[i, :6 - i] = row[:6 - i]
        triangle = Triangle(values)
        assert not link_ratio_trend_test(triangle)["trend"].any()
        assert len(link_ratio_outliers(triangle)) == 0
        assert backtest(triangle).loc["Total", "actual_to_expected"] == pytest.approx(1.0)
        fit = intercept_test(triangle)
        assert fit.loc[fit["points"] >= 4, "intercept"].abs().max() < 1e-6

    def test_additive_development_has_an_intercept(self):
        # Every origin period gains the same amount whatever its claims to date
        rng = np.random.default_rng(1)
        start = rng.uniform(500, 3000, 12)
        values = np.full((12, 12), np.nan)
        for i in range(12):
            row = start[i] + 1000.0 * np.arange(12) + rng.normal(0, 5, 12)
            values[i, :12 - i] = row[:12 - i]
        fit = intercept_test(Triangle(values))
        assert fit.loc["1-2", "significant"]
        assert fit.loc["1-2", "intercept"] == pytest.approx(1000, rel=0.05)
        assert fit.loc["1-2", "slope"] == pytest.approx(1.0, abs=0.02)

    def test_settlement_change_shows_as_trend_and_backtest(self):
        triangle = simulator().with_settlement_change(2021, (0.6, 0.25, 0.1, 0.05)).triangle()
        report = diagnose(triangle)
        assert report.loc["Back-test of the latest diagonal", "flag"]
        assert report.loc["Development proportional to claims to date", "flag"]

    def test_raa_outlier(self):
        outliers = link_ratio_outliers(load_raa())
        first = outliers.iloc[0]
        assert first["origin"] == 1982 and first["step"] == "1-2"
        assert first["link_ratio"] == pytest.approx(4285 / 106)

    def test_backtest_by_hand(self):
        triangle = Triangle([[100, 150, 165], [110, 170, np.nan], [120, np.nan, np.nan]])
        # A period earlier only origin 1 had developed: factor 150/100
        table = backtest(triangle)
        assert table.loc[2, "expected"] == pytest.approx(110 * 1.5 - 110)
        assert table.loc[2, "actual"] == pytest.approx(60)
        assert list(table.index) == [2, "Total"]
        with pytest.raises(ValueError):
            backtest(triangle, diagonals=0)
        with pytest.raises(ValueError, match="too small"):
            backtest(triangle, diagonals=2)

    def test_backtest_with_another_method(self):
        table = backtest(load_genins(), method=lambda t: ChainLadder(t, average="simple"),
                         diagonals=2)
        assert table.index[-1] == "Total" and len(table) == 7

    def test_diagnose_false_alarms_are_limited(self):
        # Each check is meant to flag few well-behaved triangles
        flags = np.array([diagnose(simulator(seed).triangle())["flag"].to_numpy()
                          for seed in range(40)])
        assert flags.mean(axis=0).max() <= 0.30
        shocked = np.array([
            diagnose(simulator(seed).with_inflation_shock(2022, 0.40).triangle())["flag"]
            .to_numpy() for seed in range(10)])
        assert shocked.any(axis=1).all()


class TestGLMReserving:
    def test_reproduces_chain_ladder(self):
        for triangle in (load_genins(), load_raa()):
            assert np.allclose(GLMReserving(triangle).ibnr, ChainLadder(triangle).ibnr)

    def test_taylor_ashe_published_figures(self):
        model = GLMReserving(load_genins())
        assert model.dispersion == pytest.approx(52601, abs=1)
        assert model.total_ibnr == pytest.approx(18_680_856, abs=1)
        published = [110_100, 216_043, 260_871, 303_549, 375_013, 495_377, 789_960,
                     1_046_512, 1_980_101]
        assert np.allclose(model.prediction_error.to_numpy()[1:], published, rtol=2e-5)
        assert model.total_prediction_error == pytest.approx(2_945_659, rel=2e-5)
        summary = model.summary()
        assert summary.loc["Total", "ibnr"] == pytest.approx(model.total_ibnr)

    def test_fitted_and_residuals(self):
        triangle = load_genins()
        model = GLMReserving(triangle)
        incremental = triangle.to_incremental().values
        seen = ~np.isnan(incremental)
        fitted = model.fitted.to_numpy()
        # Poisson fits reproduce the row and column totals of the observed cells
        assert np.allclose(np.where(seen, fitted, 0).sum(axis=1), np.nansum(incremental, axis=1))
        assert np.allclose(np.where(seen, fitted, 0).sum(axis=0), np.nansum(incremental, axis=0))
        assert np.nansum(model.future_incremental().to_numpy()) == pytest.approx(model.total_ibnr)
        residuals = model.residuals().to_numpy()
        assert np.isnan(residuals[~seen]).all()
        assert np.nansum(residuals ** 2) == pytest.approx(seen.sum() - model.n_parameters)

    def test_calendar_structure_recovers_inflation(self):
        base = ClaimsSimulator(n_years=10, claims_per_year=2000, claim_cv=0.5, inflation=0.03,
                               pattern_concentration=20, seed=3)
        payments = base.with_inflation_shock(2022, 0.40).simulate()
        triangle = base.triangle(payments)
        model = GLMReserving(triangle, structure="calendar", trend_periods=2)
        assert model.calendar_inflation.iloc[:6].mean() == pytest.approx(0.03, abs=0.03)
        assert model.calendar_inflation.iloc[-2:].mean() == pytest.approx(0.40, abs=0.05)
        truth = base.true_ultimate(payments).sum() - triangle.latest_diagonal().sum()
        assert model.total_ibnr == pytest.approx(truth, rel=0.10)
        assert ChainLadder(triangle).total_ibnr < 0.75 * truth

    def test_future_inflation_assumption(self):
        triangle = simulator(2).triangle()
        low = GLMReserving(triangle, structure="calendar", future_inflation=0.0)
        high = GLMReserving(triangle, structure="calendar", future_inflation=0.20)
        assert high.total_ibnr > low.total_ibnr
        assert high.future_inflation == pytest.approx(0.20)

    def test_exposure(self):
        triangle = simulator(2).triangle()
        doubled = GLMReserving(triangle, structure="calendar", exposure=np.full(10, 2.0),
                               future_inflation=0.03)
        plain = GLMReserving(triangle, structure="calendar", future_inflation=0.03)
        assert doubled.total_ibnr == pytest.approx(plain.total_ibnr)
        growing = GLMReserving(triangle, structure="calendar", exposure=np.linspace(1, 2, 10),
                               future_inflation=0.03)
        assert growing.total_ibnr > plain.total_ibnr

    def test_validation(self):
        triangle = load_genins()
        with pytest.raises(ValueError, match="structure"):
            GLMReserving(triangle, structure="other")
        with pytest.raises(ValueError, match="exposure"):
            GLMReserving(triangle, exposure=[1.0, 2.0])
        with pytest.raises(ValueError, match="no effect"):
            GLMReserving(triangle, exposure=np.linspace(1, 2, 10))
        with pytest.raises(ValueError, match="future_inflation"):
            GLMReserving(triangle, structure="calendar", future_inflation=-1.5)
        with pytest.raises(TypeError):
            GLMReserving([[1, 2], [3, None]])
        negative = Triangle([[100, 90, 95], [110, 100, np.nan], [120, np.nan, np.nan]])
        with pytest.raises(ValueError, match="positive total"):
            GLMReserving(negative)


class TestMLChainLadder:
    def test_zero_adjustment_is_chain_ladder(self):
        pytest.importorskip("sklearn")
        from actuneo.loss_reserving import MLChainLadder

        class NoAdjustment:
            def fit(self, X, y, sample_weight=None):
                return self

            def predict(self, X):
                return np.zeros(len(X))

        triangle = load_genins()
        model = MLChainLadder(triangle, model=NoAdjustment())
        assert np.allclose(model.ibnr, ChainLadder(triangle).ibnr)
        assert len(model.training) == 45

    def test_default_model(self):
        pytest.importorskip("sklearn")
        from actuneo.loss_reserving import MLChainLadder
        triangle = load_genins()
        model = MLChainLadder(triangle)
        again = MLChainLadder(triangle)
        assert model.total_ibnr == pytest.approx(again.total_ibnr)
        assert 0.5 < model.total_ibnr / ChainLadder(triangle).total_ibnr < 2.0
        assert model.summary().loc["Total", "ibnr"] == pytest.approx(model.total_ibnr)
        assert np.isnan(model.predicted_factors.iloc[0]).all()

    def test_follows_faster_settlement(self):
        pytest.importorskip("sklearn")
        from actuneo.loss_reserving import MLChainLadder
        table = benchmark_reserving(
            simulator().with_settlement_change(2021, (0.6, 0.25, 0.1, 0.05)),
            {"cl": ChainLadder, "ml": MLChainLadder}, n_simulations=5)
        assert table.loc["ml", "rmse"] < table.loc["cl", "rmse"]


class TestBenchmark:
    def test_chain_ladder_unbiased_on_clean_data(self):
        sim = simulator(seed=99)
        table = benchmark_reserving(sim, {"cl": ChainLadder,
                                          "half": lambda t: 0.5 * ChainLadder(t).total_ibnr},
                                    n_simulations=20)
        assert abs(table.loc["cl", "mean_error"]) < 0.05
        assert table.loc["half", "mean_error"] == pytest.approx(
            0.5 * (1 + table.loc["cl", "mean_error"]) - 1)
        assert table.loc["cl", "worst_under"] <= table.loc["cl", "worst_over"]
        assert sim.seed == 99

    def test_chain_ladder_fails_under_inflation_shock(self):
        table = benchmark_reserving(simulator().with_inflation_shock(2022, 0.40),
                                    {"cl": ChainLadder}, n_simulations=5)
        assert table.loc["cl", "worst_over"] < -0.25

    def test_failures_are_counted(self):
        def broken(triangle):
            raise ValueError("no")

        table = benchmark_reserving(simulator(), {"broken": broken}, n_simulations=2)
        assert table.loc["broken", "failed"] == 2 and np.isnan(table.loc["broken", "rmse"])
        with pytest.raises(ValueError):
            benchmark_reserving(simulator(), {"cl": ChainLadder}, n_simulations=0)


class TestSimulatorPatternVariation:
    def test_same_totals_with_varying_patterns(self):
        fixed = ClaimsSimulator(n_years=4, claims_per_year=50, seed=5)
        assert fixed.pattern_concentration is None
        varied = ClaimsSimulator(n_years=4, claims_per_year=50, seed=5, pattern_concentration=2)
        shares = varied.simulate()
        by_claim = shares.groupby("claim_id")["amount"].apply(lambda a: a.iloc[0] / a.sum())
        assert by_claim.std() > 0.05
        # On average claims still follow the payment pattern
        large = ClaimsSimulator(n_years=3, claims_per_year=3000, claim_cv=0.1, seed=5,
                                pattern_concentration=2).simulate()
        paid = large.groupby("development_year")["amount"].sum()
        assert paid.iloc[0] / paid.sum() == pytest.approx(0.35, abs=0.02)
        with pytest.raises(ValueError):
            ClaimsSimulator(pattern_concentration=0)
