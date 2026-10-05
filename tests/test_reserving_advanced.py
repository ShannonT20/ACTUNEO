"""
Tests for the practical reserving tools: triangles from claim listings,
irregular and awkward triangles, cash flows and discounting, Cape Cod,
Mack diagnostics and quantiles, Excel output and plots.
"""

import numpy as np
import pandas as pd
import pytest
from actuneo.finance import YieldCurve
from actuneo.loss_reserving import (
    Triangle, ChainLadder, MackChainLadder, BornhuetterFerguson, CapeCod,
    InflationAdjustedChainLadder, BootChainLadder, load_raa, load_genins,
)

N = np.nan


@pytest.fixture
def small():
    return Triangle(
        [[100, 150, 165],
         [110, 176, N],
         [120, N, N]],
        origin=[2021, 2022, 2023], development=[1, 2, 3],
    )


class TestFromTransactions:
    """Triangles built from a listing of dated transactions."""

    @pytest.fixture
    def payments(self):
        return pd.DataFrame({
            "loss_date": ["2021-03-10", "2021-03-10", "2021-11-02", "2022-06-30",
                          "2022-06-30", "2023-01-15"],
            "paid_date": ["2021-05-01", "2022-02-01", "2023-08-20", "2022-09-09",
                          "2023-03-03", "2023-12-31"],
            "amount": [60.0, 50.0, 15.0, 110.0, 66.0, 120.0],
        })

    def test_annual(self, payments):
        tri = Triangle.from_transactions(payments, "loss_date", "paid_date", "amount",
                                         cumulative=False)
        assert tri.origin == ["2021", "2022", "2023"]
        assert tri.development == [1, 2, 3]
        np.testing.assert_array_equal(
            tri.values, [[60, 50, 15], [110, 66, N], [120, N, N]]
        )
        cumulative = Triangle.from_transactions(payments, "loss_date", "paid_date", "amount")
        assert cumulative.is_cumulative
        assert cumulative.latest_diagonal().tolist() == [125, 176, 120]

    def test_quiet_periods_are_zero(self):
        """No payments in a period up to the valuation date means zero, not unknown."""
        df = pd.DataFrame({"loss": ["2020-02-01", "2022-05-05"],
                           "paid": ["2020-03-01", "2022-06-06"], "amount": [10.0, 30.0]})
        tri = Triangle.from_transactions(df, "loss", "paid", "amount", cumulative=False,
                                         valuation_date="2023-12-31")
        assert tri.origin == ["2020", "2021", "2022", "2023"]
        np.testing.assert_array_equal(
            tri.values,
            [[10, 0, 0, 0], [0, 0, 0, N], [30, 0, N, N], [0, N, N, N]],
        )

    def test_annual_origin_quarterly_development(self, payments):
        tri = Triangle.from_transactions(payments, "loss_date", "paid_date", "amount",
                                         development_grain="Q", cumulative=False)
        assert tri.shape == (3, 12)
        # 2021 payments: May 2021 is quarter 2, February 2022 quarter 5, August 2023 quarter 11
        assert tri.values[0, 1] == 60 and tri.values[0, 4] == 50 and tri.values[0, 10] == 15
        # 2023 has four quarters of development, 2022 eight
        assert np.isnan(tri.values[2, 4]) and not np.isnan(tri.values[2, 3])
        assert np.isnan(tri.values[1, 8]) and not np.isnan(tri.values[1, 7])
        assert np.nansum(tri.values) == payments["amount"].sum()

    def test_quarterly_origin(self, payments):
        tri = Triangle.from_transactions(payments, "loss_date", "paid_date", "amount",
                                         origin_grain="Q", development_grain="Q")
        assert tri.origin[0] == "2021Q1" and tri.origin[-1] == "2023Q4"
        assert tri.n_origin == 12
        assert tri.latest_diagonal().sum() == payments["amount"].sum()

    def test_monthly_labels(self):
        df = pd.DataFrame({"loss": ["2023-01-15", "2023-03-01"],
                           "paid": ["2023-02-01", "2023-03-20"], "amount": [5.0, 7.0]})
        tri = Triangle.from_transactions(df, "loss", "paid", "amount", "M", "M")
        assert tri.origin == ["2023-01", "2023-02", "2023-03"]

    def test_claim_counts(self):
        """Counting reported claims: one row per claim with a value of 1."""
        df = pd.DataFrame({"loss": ["2022-01-01", "2022-07-01", "2022-12-01", "2023-04-01"],
                           "reported": ["2022-02-01", "2022-08-01", "2023-01-15", "2023-05-01"]})
        df["count"] = 1
        tri = Triangle.from_transactions(df, "loss", "reported", "count")
        np.testing.assert_array_equal(tri.values, [[2, 3], [1, N]])

    def test_invalid(self, payments):
        with pytest.raises(ValueError, match="before its origin date"):
            Triangle.from_transactions(
                pd.DataFrame({"a": ["2022-05-01"], "b": ["2022-01-01"], "v": [1.0]}),
                "a", "b", "v",
            )
        with pytest.raises(ValueError, match="after the valuation date"):
            Triangle.from_transactions(payments, "loss_date", "paid_date", "amount",
                                       valuation_date="2022-12-31")
        with pytest.raises(ValueError, match="longer than origin"):
            Triangle.from_transactions(payments, "loss_date", "paid_date", "amount", "Q", "Y")
        with pytest.raises(ValueError, match="grains"):
            Triangle.from_transactions(payments, "loss_date", "paid_date", "amount", "W", "W")


class TestIrregularTriangles:
    """Triangles that are not square, very small, or contain awkward values."""

    def test_more_origin_than_development_periods(self):
        """Older origin years fully observed over a shorter development window."""
        raa = load_raa()
        values = raa.values[:, :6].copy()
        tri = Triangle(values, origin=raa.origin, development=raa.development[:6])
        assert tri.shape == (10, 6)

        cl = ChainLadder(tri)
        np.testing.assert_allclose(cl.factors, ChainLadder(raa).factors[:5])
        assert (cl.ibnr.iloc[:5] == 0).all()

        mack = MackChainLadder(tri, est_sigma="mack")
        assert (mack.mack_se.iloc[:5] == 0).all()
        assert mack.total_mack_se > 0
        # Every development step has several link ratios, so no sigma is extrapolated
        np.testing.assert_allclose(mack.sigma, MackChainLadder(raa, est_sigma="mack").sigma[:5])

        boot = BootChainLadder(tri, 300, seed=1)
        assert boot.mean_ibnr == pytest.approx(cl.total_ibnr, rel=0.15)

    def test_fewer_origin_than_development_periods(self):
        """Only the first origin periods of a long triangle."""
        raa = load_raa()
        tri = Triangle(raa.values[:6], origin=raa.origin[:6], development=raa.development)
        cl = ChainLadder(tri)
        assert cl.ultimate.iloc[0] == 18834
        assert len(cl.factors) == 9
        mack = MackChainLadder(tri, est_sigma="mack")
        assert np.isfinite(mack.total_mack_se) and mack.total_mack_se > 0

    def test_two_by_two(self):
        tri = Triangle([[100, 150], [120, N]])
        cl = ChainLadder(tri)
        assert cl.ultimate.tolist() == [150, 180]
        assert cl.cash_flows().tolist() == [60]
        with pytest.raises(ValueError, match="Not enough development periods"):
            MackChainLadder(tri)
        with pytest.raises(ValueError, match="too few observations"):
            BootChainLadder(tri, 10)

    def test_negative_incremental_claims(self):
        """Recoveries make incremental claims negative; the methods still run."""
        tri = Triangle([[1000, 1400, 1350, 1360],
                        [1100, 1500, 1480, N],
                        [1200, 1700, N, N],
                        [1300, N, N, N]])
        assert (tri.to_incremental().values[:2, 2] < 0).all()
        cl = ChainLadder(tri)
        assert cl.factors.iloc[1] < 1
        mack = MackChainLadder(tri, est_sigma="mack")
        assert np.isfinite(mack.total_mack_se)
        boot = BootChainLadder(tri, 500, seed=2)
        assert np.isfinite(boot.sd_ibnr)
        assert boot.mean_ibnr == pytest.approx(cl.total_ibnr, rel=0.25)

    def test_zero_first_development_period(self):
        tri = Triangle([[0, 150, 165, 170], [110, 176, 190, N],
                        [120, 180, N, N], [0, N, N, N]])
        cl = ChainLadder(tri)
        # Nothing reported yet for the last origin year: the chain-ladder projects nothing
        assert cl.ultimate.iloc[-1] == 0
        # Bornhuetter-Ferguson still provides for it
        bf = BornhuetterFerguson(tri, premium=[300, 300, 300, 300], loss_ratio=0.6)
        assert bf.ibnr.iloc[-1] > 0

    def test_exact_fit_cannot_be_bootstrapped(self):
        tri = Triangle([[100, 150, 165], [200, 300, N], [300, N, N]])
        with pytest.raises(ValueError, match="nothing to bootstrap"):
            BootChainLadder(tri, 10)


class TestCashFlowsAndDiscounting:
    """Future payments by period and their present value."""

    def test_future_incremental_by_hand(self, small):
        cl = ChainLadder(small)
        f1, f2 = 326 / 210, 1.1
        future = cl.future_incremental()
        assert np.isnan(future.loc[2021]).all()
        assert future.loc[2022, 3] == pytest.approx(17.6)
        assert future.loc[2023, 2] == pytest.approx(120 * (f1 - 1))
        assert future.loc[2023, 3] == pytest.approx(120 * f1 * (f2 - 1))
        assert "tail" not in future.columns

    def test_cash_flows_by_hand(self, small):
        cl = ChainLadder(small)
        f1, f2 = 326 / 210, 1.1
        flows = cl.cash_flows()
        assert flows.index.tolist() == [1, 2]
        assert flows.loc[1] == pytest.approx(17.6 + 120 * (f1 - 1))
        assert flows.loc[2] == pytest.approx(120 * f1 * (f2 - 1))
        assert flows.sum() == pytest.approx(cl.total_ibnr)

    def test_tail_cash_flow(self, small):
        cl = ChainLadder(small, tail=1.05)
        future = cl.future_incremental()
        np.testing.assert_allclose(future["tail"], cl.full_triangle[3] * 0.05)
        flows = cl.cash_flows()
        # The oldest year's tail is paid next period, the youngest year's in period 3
        assert flows.index.tolist() == [1, 2, 3]
        assert flows.sum() == pytest.approx(cl.total_ibnr)

    def test_cash_flows_total_for_every_method(self):
        raa = load_raa()
        premium = np.full(10, 25000.0)
        models = [
            ChainLadder(raa), ChainLadder(raa, tail=1.03),
            MackChainLadder(raa, est_sigma="mack", tail=True),
            BornhuetterFerguson(raa, premium=premium, loss_ratio=0.9),
            BornhuetterFerguson(raa, premium=premium, loss_ratio=0.9, tail=1.03),
            CapeCod(raa, premium),
            InflationAdjustedChainLadder(raa, 0.05, 0.08),
        ]
        for model in models:
            assert model.cash_flows().sum() == pytest.approx(model.total_ibnr)

    def test_discounted_reserve(self, small):
        cl = ChainLadder(small)
        flows = cl.cash_flows().to_numpy()
        assert cl.discounted_reserve(0.0) == pytest.approx(cl.reserve())
        expected = flows[0] * 1.1 ** -0.5 + flows[1] * 1.1 ** -1.5
        assert cl.discounted_reserve(0.10) == pytest.approx(expected)
        end = flows[0] * 1.1 ** -1 + flows[1] * 1.1 ** -2
        assert cl.discounted_reserve(0.10, timing=1) == pytest.approx(end)
        assert cl.discounted_reserve(0.10) < cl.reserve()

    def test_periods_per_year(self, small):
        """Quarterly development periods are a quarter of a year apart."""
        cl = ChainLadder(small)
        flows = cl.cash_flows().to_numpy()
        expected = flows[0] * 1.1 ** -0.125 + flows[1] * 1.1 ** -0.375
        assert cl.discounted_reserve(0.10, periods_per_year=4) == pytest.approx(expected)

    def test_yield_curve(self, small):
        cl = ChainLadder(small)
        flat = YieldCurve([1, 5], [0.10, 0.10])
        assert cl.discounted_reserve(flat) == pytest.approx(cl.discounted_reserve(0.10))
        rising = YieldCurve([1, 5], [0.10, 0.20])
        assert cl.discounted_reserve(rising) < cl.discounted_reserve(flat)

    def test_invalid(self, small):
        with pytest.raises(ValueError):
            ChainLadder(small).discounted_reserve(0.1, timing=2)
        with pytest.raises(ValueError):
            ChainLadder(small).discounted_reserve(-1.5)


class TestCapeCod:
    """Cape Cod method."""

    def test_by_hand(self, small):
        premium = np.array([200.0, 220.0, 240.0])
        cc = CapeCod(small, premium)
        f1, f2 = 326 / 210, 1.1
        to_ultimate = np.array([1.0, f2, f1 * f2])
        used_up = premium / to_ultimate
        loss_ratio = (165 + 176 + 120) / used_up.sum()

        assert cc.loss_ratio == pytest.approx(loss_ratio)
        np.testing.assert_allclose(cc.used_up_premium, used_up)
        np.testing.assert_allclose(cc.ibnr, loss_ratio * premium * (1 - 1 / to_ultimate))
        np.testing.assert_allclose(cc.ultimate, cc.latest + cc.ibnr)

    def test_same_as_bornhuetter_ferguson_with_its_loss_ratio(self):
        raa = load_raa()
        premium = np.linspace(20000, 30000, 10)
        cc = CapeCod(raa, premium)
        bf = BornhuetterFerguson(raa, premium=premium, loss_ratio=cc.loss_ratio)
        np.testing.assert_allclose(cc.ultimate, bf.ultimate)

    def test_premium_proportional_to_chain_ladder_ultimate(self):
        """If every year has the same chain-ladder loss ratio, Cape Cod agrees with it."""
        raa = load_raa()
        cl = ChainLadder(raa)
        cc = CapeCod(raa, cl.ultimate.to_numpy() / 0.7)
        assert cc.loss_ratio == pytest.approx(0.7)
        np.testing.assert_allclose(cc.ultimate, cl.ultimate)

    def test_total_reserve_matches_chain_ladder_total(self):
        """The Cape Cod loss ratio balances total claims to date."""
        raa = load_raa()
        premium = np.linspace(20000, 30000, 10)
        cc = CapeCod(raa, premium)
        assert (cc.loss_ratio * cc.used_up_premium).sum() == pytest.approx(cc.latest.sum())

    def test_invalid(self, small):
        with pytest.raises(ValueError, match="One premium"):
            CapeCod(small, [1, 2])


class TestMackDiagnostics:
    """Residuals, alternative estimation error and quantiles."""

    def test_residuals(self):
        raa = load_raa()
        mack = MackChainLadder(raa, est_sigma="mack")
        residuals = mack.residuals()
        assert residuals.shape == (10, 9)
        # By construction the squared residuals of a step sum to its degrees of freedom
        for k in range(8):
            column = residuals.iloc[:, k].dropna()
            assert (column ** 2).sum() == pytest.approx(len(column) - 1)
        # Weighted residuals of a step sum to zero for the volume-weighted factor
        weights = np.sqrt(raa.values[:9, 0])
        assert (residuals.iloc[:9, 0] * weights).sum() == pytest.approx(0, abs=1e-8)

    def test_independence_estimation_error(self):
        raa = load_raa()
        mack = MackChainLadder(raa, est_sigma="mack")
        bbmw = MackChainLadder(raa, est_sigma="mack", mse_method="independence")
        np.testing.assert_allclose(bbmw.process_risk, mack.process_risk)
        assert (bbmw.parameter_risk.iloc[2:] > mack.parameter_risk.iloc[2:]).all()
        # One step of development: both formulas coincide
        assert bbmw.parameter_risk.iloc[1] == pytest.approx(mack.parameter_risk.iloc[1])
        assert mack.total_mack_se < bbmw.total_mack_se < 1.02 * mack.total_mack_se
        with pytest.raises(ValueError):
            MackChainLadder(raa, mse_method="other")

    def test_quantiles(self):
        mack = MackChainLadder(load_genins(), est_sigma="mack")
        normal = mack.reserve_quantile([0.5, 0.975], distribution="normal")
        assert normal.loc["Total", "ibnr_50%"] == pytest.approx(mack.total_ibnr)
        assert normal.loc["Total", "ibnr_97.5%"] == pytest.approx(
            mack.total_ibnr + 1.959964 * mack.total_mack_se, rel=1e-6
        )

        lognormal = mack.reserve_quantile([0.5, 0.75, 0.995])
        assert list(lognormal.columns) == ["ibnr_50%", "ibnr_75%", "ibnr_99.5%"]
        total = lognormal.loc["Total"]
        # Lognormal median lies below the mean; the distribution is skewed to the right
        cv2 = (mack.total_mack_se / mack.total_ibnr) ** 2
        assert total["ibnr_50%"] == pytest.approx(mack.total_ibnr / np.sqrt(1 + cv2))
        assert total["ibnr_99.5%"] > normal.loc["Total", "ibnr_97.5%"]
        # The fully developed first year has no reserve to take a quantile of
        assert lognormal.iloc[0].isna().all()

        with pytest.raises(ValueError):
            mack.reserve_quantile(1.5)
        with pytest.raises(ValueError):
            mack.reserve_quantile(0.5, distribution="gamma")

    def test_lognormal_agrees_broadly_with_bootstrap(self):
        genins = load_genins()
        mack = MackChainLadder(genins, est_sigma="mack").reserve_quantile(0.95)
        boot = BootChainLadder(genins, 5000, seed=11).quantile(0.95)
        assert mack.loc["Total", "ibnr_95%"] == pytest.approx(
            boot.loc["Total", "ibnr_95%"], rel=0.10
        )


class TestInflationWithShorterDevelopmentPeriods:
    """Accident years developed half-yearly."""

    @pytest.fixture
    def half_yearly(self):
        # Three accident years, two development periods a year, valued at the end of year 3
        return Triangle(
            [[50, 100, 130, 150, 160, 165],
             [55, 115, 150, 172, N, N],
             [60, 120, N, N, N, N]],
            origin=[2021, 2022, 2023],
        )

    def test_no_inflation_is_the_basic_chain_ladder(self, half_yearly):
        model = InflationAdjustedChainLadder(half_yearly, 0.0, 0.0, development_per_origin=2)
        np.testing.assert_allclose(model.ultimate, ChainLadder(half_yearly).ultimate)

    def test_rates_are_per_development_period(self, half_yearly):
        # Six half-years observed means five past rates; four half-years remain to run off
        model = InflationAdjustedChainLadder(
            half_yearly, [0.02] * 5, [0.03] * 4, development_per_origin=2
        )
        assert len(model.index) == 6
        higher = InflationAdjustedChainLadder(half_yearly, 0.02, 0.06, development_per_origin=2)
        assert higher.reserve() > model.reserve()
        with pytest.raises(ValueError, match="past_inflation needs 5 rates"):
            InflationAdjustedChainLadder(half_yearly, [0.02] * 3, 0.0, development_per_origin=2)

    def test_wrong_grain_is_rejected(self, half_yearly):
        with pytest.raises(ValueError, match="same calendar period"):
            InflationAdjustedChainLadder(half_yearly, 0.02, 0.03)


class TestExcelOutput:
    """Results written to a workbook."""

    def test_round_trip(self, tmp_path):
        pytest.importorskip("openpyxl")
        mack = MackChainLadder(load_raa(), est_sigma="mack")
        path = tmp_path / "raa.xlsx"
        mack.to_excel(str(path))

        sheets = pd.read_excel(path, sheet_name=None, index_col=0)
        assert list(sheets) == ["Summary", "Triangle", "Projection", "Factors", "Cash flows"]
        assert sheets["Summary"].loc["Total", "ibnr"] == pytest.approx(mack.total_ibnr)
        assert sheets["Summary"].loc["Total", "mack_se"] == pytest.approx(mack.total_mack_se)
        assert sheets["Projection"].shape == (10, 10)
        assert list(sheets["Factors"].columns) == ["factor", "cdf", "sigma", "f_se"]
        assert sheets["Cash flows"]["cash_flow"].sum() == pytest.approx(mack.total_ibnr)

    def test_other_methods(self, tmp_path, small):
        pytest.importorskip("openpyxl")
        bf = BornhuetterFerguson(small, premium=[200, 220, 240], loss_ratio=0.8)
        path = tmp_path / "bf.xlsx"
        bf.to_excel(str(path))
        summary = pd.read_excel(path, sheet_name="Summary", index_col=0)
        assert summary.loc["Total", "ultimate"] == pytest.approx(bf.ultimate.sum())


class TestPlots:
    """Plots draw without error and return the axes."""

    @pytest.fixture(autouse=True)
    def backend(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        yield
        plt.close("all")

    def test_triangle_plot(self):
        raa = load_raa()
        ax = raa.plot()
        assert len(ax.lines) == 10
        assert "cumulative" in ax.get_title()
        assert "incremental" in raa.to_incremental().plot().get_title()

    def test_reserve_plots(self):
        raa = load_raa()
        assert len(ChainLadder(raa).plot().patches) == 20
        mack = MackChainLadder(raa, est_sigma="mack")
        assert mack.plot() is not None
        ax = mack.plot_residuals()
        assert ax.get_ylabel() == "Standardised residual"

    def test_bootstrap_plot_on_given_axes(self):
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
        returned = BootChainLadder(load_raa(), 200, seed=1).plot(ax=ax, bins=20)
        assert returned is ax
        assert len(ax.patches) == 20


class TestSelectedFactors:
    """Development factors chosen by judgement instead of estimated."""

    def test_selected_factors_replace_estimates(self, small):
        cl = ChainLadder(small, factors=[2.0, 1.2])
        np.testing.assert_allclose(cl.factors, [2.0, 1.2])
        np.testing.assert_allclose(cl.ultimate, [165, 176 * 1.2, 120 * 2.0 * 1.2])

    def test_partial_selection(self, small):
        cl = ChainLadder(small, factors=[None, 1.2])
        np.testing.assert_allclose(cl.factors, [326 / 210, 1.2])

    def test_selected_factors_need_no_history(self):
        """With every factor supplied, a step no origin period has reached can be projected."""
        tri = Triangle([[100, 150, N], [110, N, N]])
        cl = ChainLadder(tri, factors=[1.5, 1.1])
        np.testing.assert_allclose(cl.ultimate, [165, 181.5])

    def test_bornhuetter_ferguson(self, small):
        bf = BornhuetterFerguson(small, premium=[200, 220, 240], loss_ratio=0.8,
                                 factors=[2.0, 1.25])
        np.testing.assert_allclose(bf.ibnr, [0, 176 * (1 - 1 / 1.25), 192 * (1 - 1 / 2.5)])

    def test_invalid(self, small):
        with pytest.raises(ValueError, match="one value for each"):
            ChainLadder(small, factors=[2.0])
        with pytest.raises(ValueError, match="positive"):
            ChainLadder(small, factors=[2.0, -1.0])

    def test_inflation_shock_with_selected_factors_and_tail(self):
        """
        Monthly triangle with a 10% a month inflation step-up from the second
        calendar month, stable real-terms factors of 2.20 and 1.40 and a tail
        of 1.15 on the money-terms cumulative claims.
        """
        tri = Triangle([[400, 920, 1288], [450, 1080, N], [500, N, N]],
                       origin=["Oct 2026", "Nov 2026", "Dec 2026"], development=[0, 1, 2])
        model = InflationAdjustedChainLadder(
            tri, past_inflation=[0.10, 0.10], future_inflation=0.10,
            factors=[2.20, 1.40], tail=1.15,
        )
        # November: 1,125 in December prices grows by 450, paid in January at 1.10
        assert model.full_triangle.loc["Nov 2026", 2] == pytest.approx(1080 + 450 * 1.10)
        # December: 600 paid in January at 1.10, then 440 paid in February at 1.21
        assert model.full_triangle.loc["Dec 2026", 1] == pytest.approx(500 + 600 * 1.10)
        assert model.full_triangle.loc["Dec 2026", 2] == pytest.approx(1160 + 440 * 1.21)
        np.testing.assert_allclose(model.ultimate, [1481.2, 1811.25, 1946.26])
        assert model.reserve() == pytest.approx(2370.71)
        assert model.cash_flows().sum() == pytest.approx(model.reserve())
