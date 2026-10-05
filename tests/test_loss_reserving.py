"""
Tests for the loss_reserving module.

Reference figures are the results published in Mack (1993) and printed by the
R ChainLadder package for the RAA and Taylor/Ashe (GenIns) triangles.
"""

import numpy as np
import pandas as pd
import pytest
from actuneo.loss_reserving import (
    Triangle, ChainLadder, MackChainLadder, load_raa, load_genins
)


@pytest.fixture
def raa():
    return load_raa()


@pytest.fixture
def small():
    """A 3 x 3 cumulative triangle that is easy to work by hand."""
    return Triangle(
        [[100, 150, 165],
         [110, 176, np.nan],
         [120, np.nan, np.nan]],
        origin=[2021, 2022, 2023],
        development=[12, 24, 36],
    )


class TestTriangle:
    """Test cases for Triangle class."""

    def test_basic_properties(self, raa):
        assert raa.shape == (10, 10)
        assert raa.n_origin == 10 and raa.n_development == 10
        assert raa.is_cumulative
        assert raa.origin[0] == 1981 and raa.origin[-1] == 1990
        assert "RAA" in repr(raa)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError, match="2-dimensional"):
            Triangle([1, 2, 3])
        with pytest.raises(ValueError, match="origin"):
            Triangle([[1, 2], [3, np.nan]], origin=[2020])
        with pytest.raises(ValueError, match="development"):
            Triangle([[1, 2], [3, np.nan]], development=[1, 2, 3])
        with pytest.raises(ValueError, match="missing values"):
            Triangle([[1, np.nan, 3], [1, 2, np.nan]])
        with pytest.raises(ValueError, match="no observations"):
            Triangle([[1, 2], [np.nan, np.nan]])

    def test_latest_diagonal(self, raa, small):
        latest = raa.latest_diagonal()
        assert latest.loc[1981] == 18834
        assert latest.loc[1990] == 2063
        assert latest.sum() == 160987
        assert small.latest_diagonal().tolist() == [165, 176, 120]
        assert small.latest_development().tolist() == [36, 24, 12]

    def test_cumulative_incremental_round_trip(self, raa, small):
        incremental = small.to_incremental()
        assert not incremental.is_cumulative
        np.testing.assert_array_equal(
            incremental.values,
            [[100, 50, 15], [110, 66, np.nan], [120, np.nan, np.nan]],
        )
        np.testing.assert_array_equal(incremental.to_cumulative().values, small.values)
        np.testing.assert_allclose(raa.to_incremental().to_cumulative().values, raa.values)
        assert raa.to_cumulative() is raa
        assert incremental.to_incremental() is incremental

    def test_incremental_input(self, small):
        incremental = Triangle(small.to_incremental().values, small.origin,
                               small.development, cumulative=False)
        assert ChainLadder(incremental).total_ibnr == pytest.approx(ChainLadder(small).total_ibnr)

    def test_long_format_round_trip(self, raa):
        long = raa.to_long()
        assert list(long.columns) == ["origin", "development", "value"]
        assert len(long) == 55
        assert len(raa.to_long(dropna=False)) == 100
        rebuilt = Triangle.from_long(long.sample(frac=1, random_state=0))
        np.testing.assert_array_equal(rebuilt.values, raa.values)
        assert rebuilt.origin == raa.origin

    def test_from_long_sums_transactions(self):
        """Individual payments are added up; quiet periods count as zero."""
        payments = pd.DataFrame({
            "accident_year": [2021, 2021, 2021, 2022, 2022, 2023],
            "dev": [1, 1, 3, 1, 2, 1],
            "paid": [60.0, 40.0, 15.0, 110.0, 66.0, 120.0],
        })
        tri = Triangle.from_long(payments, "accident_year", "dev", "paid", cumulative=False)
        np.testing.assert_array_equal(
            tri.values,
            [[100, 0, 15], [110, 66, np.nan], [120, np.nan, np.nan]],
        )
        assert tri.to_cumulative().latest_diagonal().tolist() == [115, 176, 120]

    def test_from_dataframe(self, raa):
        tri = Triangle.from_dataframe(raa.to_frame(), name="copy")
        assert tri.origin == raa.origin and tri.development == raa.development
        np.testing.assert_array_equal(tri.values, raa.values)

    def test_link_ratios(self, small):
        ratios = small.link_ratios()
        assert list(ratios.columns) == ["12-24", "24-36"]
        assert ratios.loc[2021, "12-24"] == pytest.approx(1.5)
        assert ratios.loc[2022, "12-24"] == pytest.approx(1.6)
        assert ratios.loc[2021, "24-36"] == pytest.approx(1.1)
        assert np.isnan(ratios.loc[2023, "12-24"])

    def test_development_factors(self, small):
        volume = small.development_factors()
        simple = small.development_factors("simple")
        assert volume["12-24"] == pytest.approx(326 / 210)
        assert simple["12-24"] == pytest.approx(1.55)
        assert small.development_factors(n_periods=1)["12-24"] == pytest.approx(1.6)
        with pytest.raises(ValueError):
            small.development_factors("median")

    def test_raa_age_to_age(self, raa):
        """Published RAA chain-ladder factors (Mack 1993, R ata(RAA))."""
        ata = raa.age_to_age()
        expected_volume = [2.999, 1.624, 1.271, 1.172, 1.113, 1.042, 1.033, 1.017, 1.009]
        expected_simple = [8.206, 1.696, 1.315, 1.183, 1.127, 1.043, 1.034, 1.018, 1.009]
        np.testing.assert_allclose(ata.loc["volume"], expected_volume, atol=5e-4)
        np.testing.assert_allclose(ata.loc["simple"], expected_simple, atol=5e-4)
        assert ata.loc[1982, "1-2"] == pytest.approx(4285 / 106)


class TestChainLadder:
    """Test cases for the deterministic chain-ladder."""

    def test_small_triangle_by_hand(self, small):
        cl = ChainLadder(small)
        f1, f2 = 326 / 210, 1.1
        np.testing.assert_allclose(cl.factors, [f1, f2])
        np.testing.assert_allclose(cl.cdf, [f1 * f2, f2, 1.0])
        np.testing.assert_allclose(cl.ultimate, [165, 176 * f2, 120 * f1 * f2])
        np.testing.assert_allclose(cl.ibnr, [0, 17.6, 120 * f1 * f2 - 120])
        assert cl.full_triangle.loc[2023, 24] == pytest.approx(120 * f1)
        # The observed part of the triangle is unchanged
        assert cl.full_triangle.loc[2022, 24] == 176

    def test_raa(self, raa):
        cl = ChainLadder(raa)
        expected_ibnr = [0, 154, 617, 1636, 2747, 3649, 5435, 10907, 10650, 16339]
        np.testing.assert_allclose(cl.ibnr, expected_ibnr, atol=0.5)
        assert cl.total_ibnr == pytest.approx(52135, abs=0.5)

        summary = cl.summary()
        assert list(summary.columns) == ["latest", "dev_to_date", "ultimate", "ibnr"]
        assert summary.loc["Total", "latest"] == 160987
        assert summary.loc["Total", "ultimate"] == pytest.approx(213122, abs=0.5)
        assert summary.loc[1990, "dev_to_date"] == pytest.approx(0.112, abs=5e-4)
        assert len(cl.summary(total=False)) == 10

    def test_tail_factor(self, raa):
        base = ChainLadder(raa)
        tailed = ChainLadder(raa, tail=1.05)
        np.testing.assert_allclose(tailed.ultimate, base.ultimate * 1.05)
        assert tailed.ibnr.iloc[0] == pytest.approx(18834 * 0.05)
        assert tailed.cdf.iloc[-1] == pytest.approx(1.05)
        with pytest.raises(ValueError):
            ChainLadder(raa, tail=0)

    def test_simple_average_and_recent_periods(self, raa):
        assert ChainLadder(raa, average="simple").total_ibnr > ChainLadder(raa).total_ibnr
        recent = ChainLadder(raa, n_periods=3)
        assert recent.factors["1-2"] == pytest.approx((6947 + 4020 + 5395) / (1351 + 557 + 3133))

    def test_rejects_non_triangle(self):
        with pytest.raises(TypeError):
            ChainLadder([[1, 2], [3, np.nan]])

    def test_missing_factor(self):
        """A development step nobody has reached cannot be projected."""
        tri = Triangle([[100, 150, np.nan], [110, np.nan, np.nan]])
        with pytest.raises(ValueError, match="No development factor"):
            ChainLadder(tri)


class TestMackChainLadder:
    """Test cases for Mack's model against published results."""

    def test_raa_published_results(self, raa):
        """Mack (1993), table for the RAA data; R MackChainLadder(RAA, est.sigma='Mack')."""
        mack = MackChainLadder(raa, est_sigma="mack")

        expected_se = [0, 206, 623, 747, 1469, 2002, 2209, 5358, 6333, 24566]
        np.testing.assert_allclose(mack.mack_se, expected_se, atol=0.5)
        assert mack.total_mack_se == pytest.approx(26909, abs=0.5)
        assert mack.total_ibnr == pytest.approx(52135, abs=0.5)

        summary = mack.summary()
        assert summary.loc["Total", "cv_ibnr"] == pytest.approx(0.52, abs=5e-3)
        assert summary.loc[1990, "cv_ibnr"] == pytest.approx(1.503, abs=5e-4)
        assert summary.loc[1982, "cv_ibnr"] == pytest.approx(1.339, abs=5e-4)

    def test_genins_published_results(self):
        """Taylor/Ashe data: Mack (1993); R MackChainLadder(GenIns, est.sigma='Mack')."""
        mack = MackChainLadder(load_genins(), est_sigma="mack")
        assert mack.total_ibnr == pytest.approx(18680856, abs=1)
        assert mack.total_mack_se == pytest.approx(2447095, abs=1)
        expected_se = [0, 75535, 121699, 133549, 261406, 411010, 558317, 875328, 971258, 1363155]
        np.testing.assert_allclose(mack.mack_se, expected_se, atol=1)

    def test_point_estimates_match_chain_ladder(self, raa):
        mack = MackChainLadder(raa)
        cl = ChainLadder(raa)
        np.testing.assert_allclose(mack.factors, cl.factors)
        np.testing.assert_allclose(mack.ultimate, cl.ultimate)

    def test_risk_decomposition(self, raa):
        mack = MackChainLadder(raa, est_sigma="mack")
        np.testing.assert_allclose(
            mack.mack_se ** 2, mack.process_risk ** 2 + mack.parameter_risk ** 2
        )
        assert mack.total_mack_se ** 2 == pytest.approx(
            mack.total_process_risk ** 2 + mack.total_parameter_risk ** 2
        )
        # Process risk adds across independent origin years
        assert mack.total_process_risk ** 2 == pytest.approx((mack.process_risk ** 2).sum())
        # Shared factors make the total parameter risk exceed the independent sum
        assert mack.total_parameter_risk ** 2 > (mack.parameter_risk ** 2).sum()
        assert mack.mack_se.iloc[0] == 0

    def test_sigma_estimation_methods(self, raa):
        by_mack = MackChainLadder(raa, est_sigma="mack")
        log_linear = MackChainLadder(raa, est_sigma="log-linear")

        # Only the last development step, with a single link ratio, differs
        np.testing.assert_allclose(by_mack.sigma[:-1], log_linear.sigma[:-1])
        s = by_mack.sigma.to_numpy() ** 2
        assert s[-1] == pytest.approx(min(s[-2] ** 2 / s[-3], min(s[-3], s[-2])))
        assert log_linear.sigma.iloc[-1] > 0
        assert log_linear.total_mack_se == pytest.approx(by_mack.total_mack_se, rel=0.01)

        with pytest.raises(ValueError):
            MackChainLadder(raa, est_sigma="other")

    def test_alpha(self, raa):
        """alpha = 0 gives the simple average link ratios."""
        mack = MackChainLadder(raa, alpha=0, est_sigma="mack")
        np.testing.assert_allclose(mack.factors, raa.development_factors("simple"))
        np.testing.assert_allclose(mack.ultimate, ChainLadder(raa, average="simple").ultimate)

        # alpha = 2 is least squares through the origin
        ols = MackChainLadder(raa, alpha=2, est_sigma="mack")
        c = raa.values
        assert ols.factors["1-2"] == pytest.approx(
            np.sum(c[:9, 0] * c[:9, 1]) / np.sum(c[:9, 0] ** 2)
        )

    def test_zero_and_negative_claims(self):
        """A zero cell gives no link ratio and is skipped; negatives are rejected."""
        zero = Triangle([[0, 150, 165, 170], [110, 176, 190, np.nan],
                         [120, 180, np.nan, np.nan], [130, np.nan, np.nan, np.nan]])
        mack = MackChainLadder(zero, est_sigma="mack")
        assert mack.factors.iloc[0] == pytest.approx((176 + 180) / (110 + 120))
        assert np.isfinite(mack.total_mack_se)

        negative = Triangle([[-5, 150, 165], [110, 176, np.nan], [120, np.nan, np.nan]])
        with pytest.raises(ValueError, match="negative"):
            MackChainLadder(negative)
