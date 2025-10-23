"""
Tests for the mortality module.
"""

import numpy as np
import pytest
from actuneo.mortality import MortalityTable, SurvivalFunctions


class TestMortalityTable:
    """Test cases for MortalityTable class."""

    def test_initialization(self, sample_mortality_table):
        """Test basic initialization."""
        mt = sample_mortality_table
        assert mt.name == "Test Table"
        assert len(mt.ages) == len(mt.qx)
        assert np.all(mt.qx >= 0) and np.all(mt.qx <= 1)

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        ages = [20, 30, 40]
        qx = [0.1, 0.2]  # Different length

        with pytest.raises(ValueError):
            MortalityTable(ages, qx)

        # Invalid qx values
        qx_invalid = [0.1, 1.5, 0.2]  # Value > 1
        with pytest.raises(ValueError):
            MortalityTable(ages, qx_invalid)

    def test_get_qx(self, sample_mortality_table):
        """Test qx retrieval."""
        mt = sample_mortality_table

        # Test single age
        qx_val = mt.get_qx(30)
        assert isinstance(qx_val, (int, float))
        assert 0 <= qx_val <= 1

        # Test multiple ages
        qx_vals = mt.get_qx([25, 35, 45])
        assert len(qx_vals) == 3
        assert all(0 <= q for q in qx_vals if not np.isnan(q))

        # Test age not in table
        qx_missing = mt.get_qx(15)  # Age below range
        assert np.isnan(qx_missing)

    def test_get_px(self, sample_mortality_table):
        """Test px retrieval."""
        mt = sample_mortality_table

        px_val = mt.get_px(30)
        assert 0 <= px_val <= 1
        assert abs(px_val + mt.get_qx(30) - 1) < 1e-10  # px + qx = 1

    def test_life_expectancy(self, sample_mortality_table):
        """Test life expectancy calculations."""
        mt = sample_mortality_table

        le_30 = mt.life_expectancy(30)
        le_50 = mt.life_expectancy(50)

        assert le_30 > le_50  # Life expectancy should decrease with age
        assert le_30 > 0 and le_50 > 0

    def test_from_dataframe(self, sample_mortality_table):
        """Test creation from DataFrame."""
        import pandas as pd

        mt = sample_mortality_table
        df = pd.DataFrame({
            'age': mt.ages[:10],
            'qx': mt.qx[:10]
        })

        mt_from_df = MortalityTable.from_dataframe(df)
        assert len(mt_from_df.ages) == 10
        assert mt_from_df.name == "DataFrame Table"


class TestSurvivalFunctions:
    """Test cases for SurvivalFunctions class."""

    def test_initialization(self, sample_mortality_table):
        """Test SurvivalFunctions initialization."""
        sf = SurvivalFunctions(sample_mortality_table, interest_rate=0.05)
        assert sf.i == 0.05
        assert sf.v == 1 / 1.05

    def test_npx(self, sample_mortality_table):
        """Test n-year survival probability."""
        sf = SurvivalFunctions(sample_mortality_table)

        # Test basic survival
        p1 = sf.npx(30, 1)
        assert 0 < p1 < 1

        # Test multi-year survival
        p5 = sf.npx(30, 5)
        p10 = sf.npx(30, 10)
        assert p10 < p5  # Longer period should have lower survival

        # Test edge cases
        assert sf.npx(30, 0) == 1.0  # Survive 0 years
        assert sf.npx(30, -1) == 0.0  # Invalid negative period

    def test_nqx(self, sample_mortality_table):
        """Test n-year mortality probability."""
        sf = SurvivalFunctions(sample_mortality_table)

        q5 = sf.nqx(30, 5)
        p5 = sf.npx(30, 5)
        assert abs(q5 + p5 - 1) < 1e-10  # qx + px = 1

    def test_tpx(self, sample_mortality_table):
        """Test fractional year survival."""
        sf = SurvivalFunctions(sample_mortality_table)

        # Test fractional survival
        p_half = sf.tpx(30, 0.5)
        p1 = sf.npx(30, 1)

        # Should be close but not exactly equal
        assert 0 < p_half < 1
        assert p_half > p1  # Half year should have higher survival than full year

    def test_annuity_calculations(self, sample_mortality_table):
        """Test annuity calculations."""
        sf = SurvivalFunctions(sample_mortality_table)

        # Immediate annuity
        imm_ann = sf.annuity_immediate(30, n=10)
        assert imm_ann > 0

        # Annuity due
        due_ann = sf.annuity_due(30, n=10)
        assert due_ann > imm_ann  # Due should be larger

        # Whole life annuity
        whole_ann = sf.annuity_immediate(30)
        assert whole_ann > imm_ann  # Whole life should be larger than term

    def test_assurance_calculations(self, sample_mortality_table):
        """Test assurance calculations."""
        sf = SurvivalFunctions(sample_mortality_table)

        # Whole life assurance
        whole_ass = sf.assurance(30)
        assert whole_ass > 0

        # Term assurance
        term_ass = sf.assurance(30, n=20)
        assert term_ass < whole_ass  # Term should be less than whole life
        assert term_ass > 0
