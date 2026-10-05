"""
Tests for the mortality module.
"""

import numpy as np
import pandas as pd
import pytest
from actuneo.mortality import MortalityTable, SurvivalFunctions, CommutationFunctions


class TestMortalityTable:
    """Test cases for MortalityTable class."""

    def test_initialization(self, sample_mortality_table):
        """Test basic initialization."""
        mt = sample_mortality_table
        assert mt.name == "Test Table"
        assert len(mt.ages) == len(mt.qx_values)
        assert np.all(mt.qx_values >= 0) and np.all(mt.qx_values <= 1)
        assert (mt.min_age, mt.max_age, mt.omega) == (20, 110, 111)
        assert mt.is_closed

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError):
            MortalityTable([20, 21, 22], [0.1, 0.2])  # Different length

        with pytest.raises(ValueError):
            MortalityTable([20, 21, 22], [0.1, 1.5, 0.2])  # Value > 1

        with pytest.raises(ValueError):
            MortalityTable([20, 21, 22], [0.1, np.nan, 0.2])

        with pytest.raises(ValueError, match="unique"):
            MortalityTable([20, 21, 21], [0.1, 0.1, 0.2])

        with pytest.raises(ValueError, match="consecutive"):
            MortalityTable([20, 30, 40], [0.1, 0.1, 0.2])

        with pytest.raises(ValueError, match="integers"):
            MortalityTable([20.5, 21.5], [0.1, 0.1])

    def test_unsorted_ages_are_sorted(self):
        mt = MortalityTable([22, 20, 21], [0.3, 0.1, 0.2])
        assert mt.ages.tolist() == [20, 21, 22]
        assert mt.qx_values.tolist() == [0.1, 0.2, 0.3]

    def test_life_table_columns(self):
        """Life table columns for a small table worked out by hand."""
        mt = MortalityTable([0, 1, 2], [0.1, 0.5, 1.0], metadata={"radix": 1000})
        np.testing.assert_allclose(mt.lx_values, [1000, 900, 450])
        np.testing.assert_allclose(mt.dx_values, [100, 450, 450])
        np.testing.assert_allclose(mt.Lx_values, [950, 675, 225])
        np.testing.assert_allclose(mt.Tx_values, [1850, 900, 225])
        np.testing.assert_allclose(mt.ex_values, [1.85, 1.0, 0.5])
        np.testing.assert_allclose(mt.curtate_ex_values, [1.35, 0.5, 0.0])
        assert mt.ex(0, kind="curtate") == pytest.approx(1.35)

    def test_get_qx(self, sample_mortality_table):
        """Test qx retrieval."""
        mt = sample_mortality_table

        qx_val = mt.get_qx(30)
        assert isinstance(qx_val, float)
        assert qx_val == mt.qx_values[10]

        qx_vals = mt.get_qx([25, 35, 45])
        np.testing.assert_array_equal(qx_vals, mt.qx_values[[5, 15, 25]])

        # A one-element list gives an array, not a scalar
        assert mt.get_qx([30]).shape == (1,)

        # Ages outside the table
        assert np.isnan(mt.get_qx(15))
        assert np.isnan(mt.get_qx([15, 30]))[0]

    def test_get_px(self, sample_mortality_table):
        """Test px retrieval."""
        mt = sample_mortality_table
        assert mt.get_px(30) + mt.get_qx(30) == pytest.approx(1)

    def test_npx(self, sample_mortality_table):
        mt = sample_mortality_table
        assert mt.npx(30, 0) == 1.0
        assert mt.npx(30, 1) == pytest.approx(mt.px(30))
        assert mt.npx(30, 10) == pytest.approx(mt.lx(40) / mt.lx(30))
        assert mt.npx(30, 5) * mt.npx(35, 5) == pytest.approx(mt.npx(30, 10))
        assert mt.npx(110, 1) == 0.0
        assert mt.npx(30, 200) == 0.0
        assert mt.deferred_qx(30, 5, 2) == pytest.approx(mt.npx(30, 5) - mt.npx(30, 7))

        with pytest.raises(ValueError):
            mt.npx(30, -1)
        with pytest.raises(ValueError):
            mt.npx(15, 1)

    def test_life_expectancy(self, sample_mortality_table):
        """Test life expectancy calculations."""
        mt = sample_mortality_table

        le_30 = mt.life_expectancy(30)
        le_50 = mt.life_expectancy(50)

        assert le_30 > le_50 > 0
        # Complete expectation exceeds curtate by half a year in a closed table
        assert le_30 - mt.life_expectancy(30, kind="curtate") == pytest.approx(0.5)

        with pytest.raises(ValueError):
            mt.life_expectancy(15)

    def test_from_dataframe(self, sample_mortality_table):
        """Test creation from DataFrame."""
        mt = sample_mortality_table
        df = pd.DataFrame({
            'age': mt.ages[:10],
            'qx': mt.qx_values[:10]
        })

        mt_from_df = MortalityTable.from_dataframe(df)
        assert len(mt_from_df.ages) == 10
        assert mt_from_df.name == "DataFrame Table"

    def test_csv_round_trip(self, sample_mortality_table, tmp_path):
        path = tmp_path / "table.csv"
        sample_mortality_table.to_dataframe().to_csv(path, index=False)
        loaded = MortalityTable.from_csv(str(path))
        assert loaded.name == "table"
        np.testing.assert_allclose(loaded.qx_values, sample_mortality_table.qx_values)
        np.testing.assert_allclose(loaded.ex_values, sample_mortality_table.ex_values)


class TestZimbabwe2023Tables:
    """The Zimbabwe 2023 tables shipped with the package."""

    @pytest.mark.parametrize("name", MortalityTable.zimbabwe_2023_tables())
    def test_tables_load_and_agree_with_published_columns(self, name):
        mt = MortalityTable.from_zimbabwe_2023(name)
        published = mt.published

        assert mt.metadata["country"] == "Zimbabwe"
        assert len(mt) == len(published)
        np.testing.assert_allclose(mt.px_values, published["px"], atol=1e-7)
        # Published lx and Tx are rounded to whole lives; rebuilding them
        # from the 7-decimal qx reproduces them closely
        np.testing.assert_allclose(mt.lx_values, published["lx"], rtol=2e-3)
        np.testing.assert_allclose(mt.ex_values, published["ex"], atol=0.6)

    def test_ten_tables(self):
        assert len(MortalityTable.zimbabwe_2023_tables()) == 10

    def test_unknown_table(self):
        with pytest.raises(ValueError, match="Unknown Zimbabwe 2023 table"):
            MortalityTable.from_zimbabwe_2023("no_such_table")


class TestSurvivalFunctions:
    """Test cases for SurvivalFunctions class."""

    @pytest.fixture
    def sf(self, sample_mortality_table):
        return SurvivalFunctions(sample_mortality_table, interest_rate=0.05)

    def test_initialization(self, sf):
        """Test SurvivalFunctions initialization."""
        assert sf.i == 0.05
        assert sf.v == 1 / 1.05
        assert sf.d == pytest.approx(0.05 / 1.05)

    def test_invalid_interest_rate(self, sample_mortality_table):
        with pytest.raises(TypeError):
            SurvivalFunctions(sample_mortality_table, interest_rate="0.05")

        with pytest.raises(TypeError):
            SurvivalFunctions(sample_mortality_table, interest_rate=True)

        with pytest.raises(ValueError):
            SurvivalFunctions(sample_mortality_table, interest_rate=-1.0)

    def test_npx_nqx(self, sf):
        assert 0 < sf.npx(30, 10) < sf.npx(30, 5) < 1
        assert sf.npx(30, 0) == 1.0
        assert sf.nqx(30, 5) + sf.npx(30, 5) == pytest.approx(1)

    def test_tpx(self, sf):
        """Test fractional year survival."""
        q30 = sf.mt.qx(30)
        assert sf.tpx(30, 0.5) == pytest.approx(1 - 0.5 * q30)
        assert sf.tpx(30, 0.5, assumption="cfm") == pytest.approx((1 - q30) ** 0.5)
        assert sf.tpx(30, 2.25) == pytest.approx(sf.npx(30, 2) * (1 - 0.25 * sf.mt.qx(32)))
        assert sf.tpx(30, 3) == pytest.approx(sf.npx(30, 3))
        assert sf.tpx(110, 1.5) == 0.0

    def test_values_against_direct_summation(self, sf):
        """Annuity and assurance values equal their defining sums."""
        v, x, n = sf.v, 40, 15
        annuity_due = sum(v ** k * sf.npx(x, k) for k in range(n))
        annuity_arrears = sum(v ** k * sf.npx(x, k) for k in range(1, n + 1))
        assurance = sum(v ** (k + 1) * sf.npx(x, k) * sf.mt.qx(x + k) for k in range(n))

        assert sf.annuity_due(x, n) == pytest.approx(annuity_due)
        assert sf.annuity_immediate(x, n) == pytest.approx(annuity_arrears)
        assert sf.assurance(x, n) == pytest.approx(assurance)

    def test_annuity_relationships(self, sf):
        x, n = 30, 10
        assert sf.annuity_immediate(x) == pytest.approx(sf.annuity_due(x) - 1)
        assert sf.annuity_immediate(x, n) == pytest.approx(
            sf.annuity_due(x, n) - 1 + sf.pure_endowment(x, n)
        )
        assert sf.annuity_due(x) == pytest.approx(
            sf.annuity_due(x, n) + sf.deferred_annuity_due(x, n)
        )
        assert sf.annuity_due(x, 0) == 0.0
        assert sf.annuity_mthly(x, 1) == pytest.approx(sf.annuity_due(x))
        assert sf.annuity_immediate(x) < sf.annuity_mthly(x, 12) < sf.annuity_due(x)
        assert sf.annuity_mthly(x, 12, due=False) == pytest.approx(
            sf.annuity_mthly(x, 12) - 1 / 12
        )

    def test_assurance_annuity_identities(self, sf):
        """A = 1 - d * ä for whole life and endowment assurances."""
        x, n = 35, 25
        assert sf.assurance(x) == pytest.approx(1 - sf.d * sf.annuity_due(x))
        assert sf.endowment_assurance(x, n) == pytest.approx(1 - sf.d * sf.annuity_due(x, n))
        assert sf.assurance(x, n) < sf.assurance(x)

    def test_recursion(self, sf):
        """A_x = v*q_x + v*p_x*A_{x+1}"""
        x = 50
        expected = sf.v * sf.mt.qx(x) + sf.v * sf.mt.px(x) * sf.assurance(x + 1)
        assert sf.assurance(x) == pytest.approx(expected)

    def test_zero_interest(self, sample_mortality_table):
        sf = SurvivalFunctions(sample_mortality_table, 0.0)
        assert sf.assurance(30) == pytest.approx(1.0)
        assert sf.annuity_immediate(30) == pytest.approx(sample_mortality_table.ex(30, "curtate"))
        assert sf.assurance(30, timing="immediate") == pytest.approx(1.0)

    def test_immediate_payment_of_claims(self, sf):
        factor = 0.05 / np.log(1.05)
        assert sf.assurance(30, 20, timing="immediate") == pytest.approx(
            factor * sf.assurance(30, 20)
        )
        # The survival benefit of an endowment is not accelerated
        assert sf.endowment_assurance(30, 20, timing="immediate") == pytest.approx(
            factor * sf.assurance(30, 20) + sf.pure_endowment(30, 20)
        )

    def test_term_beyond_table(self, sf):
        with pytest.raises(ValueError, match="past the end"):
            sf.annuity_due(100, 20)
        with pytest.raises(ValueError):
            sf.assurance(30, -1)

    def test_open_table_is_closed_with_warning(self, open_mortality_table):
        sf = SurvivalFunctions(open_mortality_table, 0.05)
        with pytest.warns(UserWarning, match="ends at age 100"):
            whole_life = sf.assurance(40)
        assert whole_life == pytest.approx(1 - sf.d * sf.annuity_due(40))

    def test_net_annual_premium(self, sf):
        x, n = 30, 20
        assert sf.net_annual_premium(x) == pytest.approx(sf.assurance(x) / sf.annuity_due(x))
        assert sf.net_annual_premium(x, n, "term") == pytest.approx(
            sf.assurance(x, n) / sf.annuity_due(x, n)
        )
        # Endowment premium: P = 1/ä - d
        assert sf.net_annual_premium(x, n, "endowment") == pytest.approx(
            1 / sf.annuity_due(x, n) - sf.d
        )
        limited = sf.net_annual_premium(x, product="whole_life", premium_term=10)
        assert limited > sf.net_annual_premium(x)
        with pytest.raises(ValueError):
            sf.net_annual_premium(x, product="term")

    def test_joint_life(self, sf):
        x, y = 40, 45
        joint_annuity = sf.joint_annuity_due(x, y)
        assert joint_annuity < min(sf.annuity_due(x), sf.annuity_due(y))
        assert sf.joint_assurance(x, y) == pytest.approx(1 - sf.d * joint_annuity)
        # One of the two lives dies first
        assert sf.contingent_assurance(x, y) + sf.contingent_assurance(y, x) == pytest.approx(
            sf.joint_assurance(x, y), rel=1e-3
        )


class TestCommutationFunctions:
    """Commutation functions agree with direct calculation."""

    def test_against_survival_functions(self, sample_mortality_table):
        cf = CommutationFunctions(sample_mortality_table, 0.05)
        sf = SurvivalFunctions(sample_mortality_table, 0.05)
        x, n = 40, 20

        assert cf.annuity_due(x) == pytest.approx(sf.annuity_due(x))
        assert cf.annuity_due(x, n) == pytest.approx(sf.annuity_due(x, n))
        assert cf.assurance(x) == pytest.approx(sf.assurance(x))
        assert cf.assurance(x, n) == pytest.approx(sf.assurance(x, n))
        assert cf.pure_endowment(x, n) == pytest.approx(sf.pure_endowment(x, n))
        assert cf.endowment_assurance(x, n) == pytest.approx(sf.endowment_assurance(x, n))
        assert cf.increasing_assurance(x) == pytest.approx(sf.increasing_assurance(x))
        assert cf.increasing_annuity_due(x) == pytest.approx(sf.increasing_annuity_due(x))

    def test_identity_and_frame(self, sample_mortality_table):
        cf = CommutationFunctions(sample_mortality_table, 0.04)
        d = 0.04 / 1.04
        np.testing.assert_allclose(cf.Mx, cf.Dx - d * cf.Nx)
        assert list(cf.to_dataframe().columns) == ["age", "Dx", "Nx", "Sx", "Cx", "Mx", "Rx"]
