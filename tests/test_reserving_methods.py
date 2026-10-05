"""
Tests for the run-off triangle methods: model checks, tail factors,
inflation-adjusted chain-ladder, average cost per claim, Bornhuetter-Ferguson
and the bootstrap.

The small triangles are standard worked examples. Published answers are
quoted where they exist; they were produced with rounded intermediate
figures, so they are matched to within that rounding.
"""

import numpy as np
import pytest
from actuneo.loss_reserving import (
    Triangle, ChainLadder, MackChainLadder, InflationAdjustedChainLadder,
    AverageCostPerClaim, BornhuetterFerguson, grossing_up, estimate_tail_factor, load_raa, load_genins,
)

N = np.nan


@pytest.fixture
def paid():
    """Cumulative claim payments, accident years 2008-2012."""
    return Triangle(
        [[786, 1410, 2216, 2440, 2519],
         [904, 1575, 2515, 2796, N],
         [995, 1814, 2880, N, N],
         [1220, 2142, N, N, N],
         [1182, N, N, N, N]],
        origin=range(2008, 2013), development=range(5), name="paid",
    )


@pytest.fixture
def incurred():
    """Cumulative incurred claims for six accident years."""
    return Triangle(
        [[2777, 3264, 3452, 3594, 3719, 3717],
         [3252, 3804, 3973, 4231, 4319, N],
         [3725, 4404, 4779, 4946, N, N],
         [4521, 5422, 5676, N, N, N],
         [5369, 6142, N, N, N, N],
         [5818, N, N, N, N, N]],
        name="incurred",
    )


@pytest.fixture
def reported_numbers():
    """Cumulative numbers of reported claims matching the incurred triangle."""
    return Triangle(
        [[414, 460, 482, 488, 492, 494],
         [453, 506, 526, 536, 539, N],
         [494, 548, 572, 582, N, N],
         [530, 588, 615, N, N, N],
         [545, 605, N, N, N, N],
         [557, N, N, N, N, N]],
        name="numbers",
    )


class TestBasicChainLadderExample:
    """Basic chain-ladder on the paid triangle."""

    def test_development_factors_and_reserve(self, paid):
        cl = ChainLadder(paid)
        np.testing.assert_allclose(cl.factors, [1.777, 1.586, 1.107, 1.032], atol=5e-4)
        # Published projections: 2,885, 3,290, 3,881, 3,806 and a reserve of 4,862
        np.testing.assert_allclose(cl.ultimate, [2519, 2885, 3290, 3881, 3806], atol=4)
        assert cl.reserve() == pytest.approx(4862, abs=5)
        assert cl.full_triangle.loc[2012, 1] == pytest.approx(2100, abs=1)

    def test_claim_numbers_from_incremental_data(self):
        reported = Triangle(
            [[17500, 5000, 2250, 750],
             [21000, 6200, 2750, N],
             [18800, 5500, N, N],
             [21300, N, N, N]],
            origin=range(2009, 2013), development=range(4), cumulative=False,
        )
        cl = ChainLadder(reported)
        np.testing.assert_allclose(cl.factors, [1.2914, 1.1006, 1.0303], atol=5e-5)
        assert cl.ultimate.sum() == pytest.approx(115106, abs=2)

    def test_loss_ratio_triangle(self):
        """The chain-ladder applies equally to a triangle of loss ratios."""
        ratios = Triangle(
            [[0.47, 0.63, 0.70, 0.74],
             [0.48, 0.62, 0.71, N],
             [0.49, 0.60, N, N],
             [0.50, N, N, N]],
            origin=range(2009, 2013),
        )
        cl = ChainLadder(ratios)
        np.testing.assert_allclose(cl.ultimate, [0.74, 0.7506, 0.7155, 0.7660], atol=5e-5)
        premium = np.array([1.42, 1.64, 1.73, 1.82])
        assert float((cl.ibnr * premium).sum()) == pytest.approx(0.7505, abs=5e-4)

    def test_model_check(self, paid):
        cl = ChainLadder(paid)
        fitted = cl.fitted_triangle()
        # The first development period is reproduced exactly
        np.testing.assert_array_equal(fitted[0], paid.values[:, 0])
        assert fitted.loc[2008, 1] == pytest.approx(1397, abs=1)
        assert np.isnan(fitted.loc[2012, 1])

        errors = cl.fit_errors()
        expected = [[0, 13, -12, -13, 0],
                    [0, -31, -2, 9, N],
                    [0, 46, 30, N, N],
                    [0, -26, N, N, N],
                    [0, N, N, N, N]]
        np.testing.assert_allclose(errors.to_numpy(), expected, atol=1.5)
        # The volume-weighted factor makes the first-step errors sum to zero
        assert errors[1].sum() == pytest.approx(0, abs=1e-9)

    def test_reserve_with_paid_to_date(self, incurred):
        cl = ChainLadder(incurred)
        assert cl.reserve(20334) == pytest.approx(cl.ultimate.sum() - 20334)
        assert cl.reserve() == pytest.approx(cl.total_ibnr)


class TestTailFactor:
    """Tail factor estimation."""

    def test_exponential_decay_is_recovered(self):
        """Factors of exactly 1 + a*b**k extrapolate to the product of the rest."""
        steps = np.arange(1, 8)
        factors = 1 + 0.5 * 0.5 ** steps
        expected = np.prod(1 + 0.5 * 0.5 ** np.arange(8, 108))
        assert estimate_tail_factor(factors) == pytest.approx(expected)

    def test_chain_ladder_tail_true(self):
        raa = load_raa()
        base = ChainLadder(raa)
        tailed = ChainLadder(raa, tail=True)
        assert 1.0 < tailed.tail < 1.05
        assert tailed.tail == pytest.approx(estimate_tail_factor(base.factors))
        np.testing.assert_allclose(tailed.ultimate, base.ultimate * tailed.tail)

    def test_no_decay(self):
        with pytest.raises(ValueError, match="do not decay"):
            estimate_tail_factor([1.1, 1.2, 1.3])
        with pytest.raises(ValueError, match="At least two"):
            estimate_tail_factor([1.1, 1.0, 0.99])

    def test_large_tail_warns(self):
        with pytest.warns(UserWarning, match="tail factor"):
            estimate_tail_factor([1.9, 1.85, 1.8])


class TestMackTail:
    """Tail factor in Mack's model."""

    def test_tail_of_one_changes_nothing(self):
        raa = load_raa()
        base = MackChainLadder(raa, est_sigma="mack")
        same = MackChainLadder(raa, est_sigma="mack", tail=1.0)
        assert same.total_mack_se == base.total_mack_se
        assert same.tail_se == 0.0 and same.tail_sigma == 0.0

    def test_certain_tail_scales_everything(self):
        """A tail with no uncertainty of its own scales ultimates and errors."""
        raa = load_raa()
        base = MackChainLadder(raa, est_sigma="mack")
        tailed = MackChainLadder(raa, est_sigma="mack", tail=1.05, tail_se=0.0, tail_sigma=0.0)
        np.testing.assert_allclose(tailed.ultimate, base.ultimate * 1.05)
        np.testing.assert_allclose(tailed.mack_se, base.mack_se * 1.05)
        assert tailed.total_mack_se == pytest.approx(base.total_mack_se * 1.05)

    def test_tail_uncertainty_by_hand(self):
        """The tail adds one more step of Mack's recursion."""
        raa = load_raa()
        base = MackChainLadder(raa, est_sigma="mack")
        tail, tail_se, tail_sigma = 1.05, 0.02, 2.0
        tailed = MackChainLadder(raa, est_sigma="mack", tail=tail, tail_se=tail_se,
                                 tail_sigma=tail_sigma)

        last = base.ultimate.to_numpy()
        process2 = base.process_risk.to_numpy() ** 2 * tail ** 2 + tail_sigma ** 2 * last
        parameter2 = base.parameter_risk.to_numpy() ** 2 * tail ** 2 + last ** 2 * tail_se ** 2
        np.testing.assert_allclose(tailed.mack_se, np.sqrt(process2 + parameter2))

        total2 = (process2.sum() + base.total_parameter_risk ** 2 * tail ** 2
                  + last.sum() ** 2 * tail_se ** 2)
        assert tailed.total_mack_se == pytest.approx(np.sqrt(total2))
        # The fully developed first year now carries tail risk
        assert tailed.mack_se.iloc[0] > 0

    def test_estimated_tail(self):
        raa = load_raa()
        base = MackChainLadder(raa, est_sigma="mack")
        tailed = MackChainLadder(raa, est_sigma="mack", tail=True)
        assert tailed.tail == pytest.approx(ChainLadder(raa, tail=True).tail)
        assert tailed.tail_se > 0 and tailed.tail_sigma > 0
        # The tail step sits beyond the triangle, so it is less uncertain
        # than the first development steps
        assert tailed.tail_se < base.f_se.iloc[0]
        assert tailed.total_ibnr > base.total_ibnr
        assert tailed.total_mack_se > base.total_mack_se

    def test_tail_below_one_needs_inputs(self):
        with pytest.raises(ValueError, match="supply both"):
            MackChainLadder(load_raa(), tail=0.99)


class TestInflationAdjustedChainLadder:
    """Inflation-adjusted chain-ladder on the paid triangle."""

    PAST = [0.051, 0.064, 0.073, 0.054]

    def test_published_example(self, paid):
        model = InflationAdjustedChainLadder(paid, self.PAST, future_inflation=0.10)

        # Payments restated at the prices of the latest year
        real = model.real_triangle.to_frame()
        assert real.loc[2008, 0] == pytest.approx(994, abs=1)
        assert real.loc[2011, 0] == pytest.approx(1286, abs=1)
        assert real.loc[2012, 0] == 1182
        np.testing.assert_allclose(model.factors, [1.7334, 1.5321, 1.0941, 1.0273], atol=3e-4)

        # Published forecasts in money terms and a reserve of 5,136
        np.testing.assert_allclose(model.ultimate, [2519, 2890, 3306, 3954, 3986], atol=2)
        assert model.full_triangle.loc[2012, 1] == pytest.approx(2136, abs=1)
        assert model.reserve() == pytest.approx(5136, abs=3)

    def test_index(self, paid):
        model = InflationAdjustedChainLadder(paid, self.PAST)
        index = model.index.to_numpy()
        assert index[-1] == 1.0
        assert index[0] == pytest.approx(1 / (1.051 * 1.064 * 1.073 * 1.054))

    def test_no_inflation_is_the_basic_chain_ladder(self, paid):
        model = InflationAdjustedChainLadder(paid, 0.0, 0.0)
        np.testing.assert_allclose(model.ultimate, ChainLadder(paid).ultimate)

    def test_constant_inflation_matches_basic_chain_ladder(self, paid):
        """If future inflation repeats a constant past rate, nothing changes."""
        same = InflationAdjustedChainLadder(paid, 0.06, 0.06)
        higher = InflationAdjustedChainLadder(paid, 0.06, 0.10)
        basic = ChainLadder(paid)
        assert same.reserve() == pytest.approx(basic.reserve(), rel=0.01)
        assert higher.reserve() > same.reserve()

    def test_varying_future_inflation(self, paid):
        flat = InflationAdjustedChainLadder(paid, self.PAST, 0.10)
        varying = InflationAdjustedChainLadder(paid, self.PAST, [0.10, 0.10, 0.10, 0.10])
        np.testing.assert_allclose(varying.ultimate, flat.ultimate)
        with pytest.raises(ValueError, match="future_inflation needs 4 rates"):
            InflationAdjustedChainLadder(paid, self.PAST, [0.1, 0.1])
        with pytest.raises(ValueError, match="past_inflation needs 4 rates"):
            InflationAdjustedChainLadder(paid, [0.05, 0.05])


class TestAverageCostPerClaim:
    """Average cost per claim method."""

    def test_grossing_up_factors(self, reported_numbers):
        table = grossing_up(reported_numbers)
        # First year is fully run off: factors are proportions of its final figure
        assert table.loc[1, 1] == pytest.approx(414 / 494)
        assert table.loc[1, "ultimate"] == 494
        # Second year is grossed up by the first year's factor
        assert table.loc[2, "ultimate"] == pytest.approx(539 / (492 / 494))
        # Third year by the average of the first two
        expected = 582 / np.mean([488 / 494, 536 / table.loc[2, "ultimate"]])
        assert table.loc[3, "ultimate"] == pytest.approx(expected)
        np.testing.assert_allclose(table["ultimate"], [494, 541, 588, 632, 649, 664], atol=0.5)

    def test_grossing_up_example(self, incurred, reported_numbers):
        acpc = AverageCostPerClaim(incurred, reported_numbers)
        assert acpc.average_cost.values[0, 0] == pytest.approx(6.708, abs=5e-4)
        np.testing.assert_allclose(
            acpc.ultimate_average_cost, [7.524, 7.973, 8.632, 9.657, 10.766, 11.699], atol=5e-3
        )
        # Published: loss estimate 33,964 and reserve 13,630 using rounded figures
        assert acpc.ultimate.sum() == pytest.approx(33964, rel=1e-3)
        assert acpc.reserve(20334) == pytest.approx(13630, rel=2e-3)
        assert acpc.summary().loc["Total", "ultimate"] == pytest.approx(acpc.ultimate.sum())

    def test_development_factor_example(self):
        claims = Triangle([[632, 714, 788, 822], [729, 784, 803, N],
                           [800, 855, N, N], [824, N, N, N]], origin=range(2009, 2013))
        numbers = Triangle([[52, 60, 66, 70], [54, 63, 65, N],
                            [60, 70, N, N], [65, N, N, N]], origin=range(2009, 2013))
        acpc = AverageCostPerClaim(claims, numbers, method="chain_ladder")
        np.testing.assert_allclose(acpc.ultimate_numbers, [70, 68.939, 79.071, 85.366], atol=1e-3)
        np.testing.assert_allclose(
            acpc.ultimate_average_cost, [11.743, 12.151, 11.988, 11.667], atol=2e-3
        )
        assert acpc.ultimate.sum() == pytest.approx(3604, abs=1)
        assert acpc.reserve(1902) == pytest.approx(1702, abs=1)

    def test_invalid_inputs(self, incurred, reported_numbers):
        with pytest.raises(ValueError, match="same cells"):
            AverageCostPerClaim(incurred, load_raa())
        with pytest.raises(ValueError, match="method"):
            AverageCostPerClaim(incurred, reported_numbers, method="other")
        with pytest.raises(TypeError):
            AverageCostPerClaim(incurred, [[1]])


class TestBornhuetterFerguson:
    """Bornhuetter-Ferguson method."""

    @pytest.fixture
    def bf_triangle(self):
        return Triangle(
            [[2866, 3334, 3503, 3624, 3719, 3717],
             [3359, 3889, 4033, 4231, 4319, N],
             [3848, 4503, 4779, 4946, N, N],
             [4673, 5422, 5676, N, N, N],
             [5369, 6142, N, N, N, N],
             [5818, N, N, N, N, N]],
        )

    PREMIUM = [4486, 5024, 5680, 6590, 7482, 8502]

    def test_published_example(self, bf_triangle):
        bf = BornhuetterFerguson(bf_triangle, premium=self.PREMIUM, loss_ratio=0.83)

        np.testing.assert_allclose(bf.factors, [1.158, 1.049, 1.039, 1.023, 0.999], atol=6e-4)
        np.testing.assert_allclose(
            bf.cdf, [1.290, 1.114, 1.062, 1.022, 0.999, 1.000], atol=1.5e-3
        )
        np.testing.assert_allclose(
            bf.initial_ultimate, [3723, 4170, 4714, 5470, 6210, 7057], atol=0.5
        )
        # Published with factors rounded to 3 decimals: emerging liabilities
        # 0, -4, 104, 317, 633, 1,588 and a total ultimate of 33,256
        np.testing.assert_allclose(bf.emerging, [0, -4, 104, 317, 633, 1588], atol=10)
        assert bf.ultimate.sum() == pytest.approx(33256, rel=1e-3)
        assert bf.reserve(20334) == pytest.approx(12922, rel=2e-3)

    def test_formula(self, bf_triangle):
        bf = BornhuetterFerguson(bf_triangle, premium=self.PREMIUM, loss_ratio=0.83)
        f = bf.cdf.to_numpy()[::-1]  # factor to ultimate at each origin's latest period
        expected = 0.83 * np.array(self.PREMIUM) * (1 - 1 / f)
        np.testing.assert_allclose(bf.ibnr, expected)
        np.testing.assert_allclose(bf.ultimate, bf.latest + expected)
        np.testing.assert_allclose(bf.full_triangle.iloc[:, -1], bf.ultimate)

    def test_second_example(self):
        tri = Triangle([[473, 620, 690, 715], [512, 660, 750, N],
                        [611, 700, N, N], [647, N, N, N]], origin=range(2009, 2013))
        bf = BornhuetterFerguson(tri, premium=[860, 940, 980, 1020], loss_ratio=0.85)
        np.testing.assert_allclose(bf.factors, [1.2406, 1.1250, 1.0362], atol=5e-5)
        np.testing.assert_allclose(bf.ultimate, [715, 777.91, 818.42, 914.5], atol=0.05)
        assert bf.reserve(1942) == pytest.approx(1284, abs=0.5)

    def test_chain_ladder_ultimate_as_initial_estimate(self, bf_triangle):
        """Feeding the chain-ladder ultimates back in reproduces the chain-ladder."""
        cl = ChainLadder(bf_triangle)
        bf = BornhuetterFerguson(bf_triangle, initial_ultimate=cl.ultimate.to_numpy())
        np.testing.assert_allclose(bf.ultimate, cl.ultimate)
        np.testing.assert_allclose(bf.chain_ladder_ultimate, cl.ultimate)

    def test_loss_ratio_per_origin(self, bf_triangle):
        flat = BornhuetterFerguson(bf_triangle, premium=self.PREMIUM, loss_ratio=0.83)
        each = BornhuetterFerguson(bf_triangle, premium=self.PREMIUM, loss_ratio=[0.83] * 6)
        np.testing.assert_allclose(each.ultimate, flat.ultimate)

    def test_invalid_inputs(self, bf_triangle):
        with pytest.raises(ValueError, match="Give either"):
            BornhuetterFerguson(bf_triangle, premium=self.PREMIUM)
        with pytest.raises(ValueError, match="Give either"):
            BornhuetterFerguson(bf_triangle, premium=self.PREMIUM, loss_ratio=0.8,
                                initial_ultimate=[1] * 6)
        with pytest.raises(ValueError, match="One value"):
            BornhuetterFerguson(bf_triangle, premium=[1, 2, 3], loss_ratio=0.8)
