"""
Tests for the life module.
"""

import pytest
from actuneo.life import LifeAssurance, Annuities, Reserves
from actuneo.mortality import MortalityTable, SurvivalFunctions
import numpy as np


class TestLifeAssurance:
    """Test cases for LifeAssurance class."""

    @pytest.fixture
    def life_assurance(self, sample_mortality_table):
        """Create LifeAssurance instance for testing."""
        return LifeAssurance(sample_mortality_table, interest_rate=0.05)

    def test_initialization(self, life_assurance):
        """Test LifeAssurance initialization."""
        assert life_assurance.i == 0.05
        assert life_assurance.expense_loading == 0.0

    def test_whole_life_assurance(self, life_assurance):
        """Test whole life assurance premiums."""
        premium = life_assurance.whole_life_assurance(30)
        assert premium > 0

        # Premium should increase with age (shorter remaining lifetime)
        premium_40 = life_assurance.whole_life_assurance(40)
        assert premium_40 > premium

    def test_term_assurance(self, life_assurance):
        """Test term assurance premiums."""
        premium_10y = life_assurance.term_assurance(30, 10)
        premium_20y = life_assurance.term_assurance(30, 20)

        assert premium_10y > 0 and premium_20y > 0
        assert premium_10y < premium_20y  # Longer term should cost more

    def test_endowment_assurance(self, life_assurance):
        """Test endowment assurance premiums."""
        endowment = life_assurance.endowment_assurance(30, 20)
        term = life_assurance.term_assurance(30, 20)
        pure_endowment = life_assurance.pure_endowment(30, 20)

        assert endowment > 0
        assert abs(endowment - (term + pure_endowment)) < 1e-10

    def test_pure_endowment(self, life_assurance):
        """Test pure endowment calculations."""
        pe = life_assurance.pure_endowment(30, 20)
        survival_prob = life_assurance.sf.npx(30, 20)
        expected = survival_prob * life_assurance.v ** 20
        assert abs(pe - expected) < 1e-10

    def test_gross_premium(self, life_assurance):
        """Test gross premium calculations."""
        net_premium = 100
        initial_expenses = 10

        gross = life_assurance.gross_premium(net_premium, initial_expenses)
        expected = net_premium * (1 + life_assurance.expense_loading) + initial_expenses
        assert abs(gross - expected) < 1e-10

    def test_reserve_calculations(self, life_assurance):
        """Test reserve calculations."""
        # Note: These methods calculate reserves but don't take annual_premium as parameter
        # The premium is implicit in the calculation

        # Whole life reserve
        reserve_wl = life_assurance.reserve_whole_life(30, 5)
        assert reserve_wl >= 0

        # Term assurance reserve
        reserve_term = life_assurance.reserve_term(30, 20, 5)
        assert reserve_term >= 0

        # Endowment reserve
        reserve_endowment = life_assurance.reserve_endowment(30, 20, 5)
        assert reserve_endowment >= 0


class TestAnnuities:
    """Test cases for Annuities class."""

    @pytest.fixture
    def annuities(self, sample_mortality_table):
        """Create Annuities instance for testing."""
        return Annuities(sample_mortality_table, interest_rate=0.05)

    @pytest.fixture
    def annuities_det(self):
        """Create deterministic Annuities instance."""
        return Annuities(interest_rate=0.05)

    def test_deterministic_annuities(self, annuities_det):
        """Test deterministic annuity calculations."""
        ann = annuities_det

        # Immediate annuity
        imm = ann.immediate_annuity(10, payment=100)
        expected_imm = 100 * ((1 - (1.05) ** (-10)) / 0.05)
        assert abs(imm - expected_imm) < 1e-10

        # Annuity due
        due = ann.annuity_due(10, payment=100)
        expected_due = 100 * ((1 - (1.05) ** (-10)) / 0.05) * 1.05
        assert abs(due - expected_due) < 1e-10

    def test_life_annuities(self, annuities):
        """Test life annuity calculations."""
        ann = annuities

        # Life annuity immediate
        life_imm = ann.life_annuity_immediate(30)
        assert life_imm > 0

        # Life annuity due
        life_due = ann.life_annuity_due(30)
        assert life_due > life_imm

        # Temporary life annuity
        temp = ann.temporary_life_annuity_immediate(30, 20)
        whole = ann.life_annuity_immediate(30)
        assert temp < whole

    def test_increasing_annuities(self, annuities_det):
        """Test increasing annuity calculations."""
        ann = annuities_det

        # Increasing annuity with high increase rate
        inc_ann = ann.increasing_annuity(5, payment=100, increase_rate=0.08)  # Higher than discount rate
        assert inc_ann > 0

        # Should be greater than level annuity when increase rate > discount rate
        level_ann = ann.immediate_annuity(5, payment=100)
        assert inc_ann > level_ann

    # Removed test for annuity_withdrawal as method doesn't exist


class TestReserves:
    """Test cases for Reserves class."""

    @pytest.fixture
    def reserves(self, sample_mortality_table):
        """Create Reserves instance for testing."""
        return Reserves(sample_mortality_table, interest_rate=0.05)

    def test_initialization(self, reserves):
        """Test Reserves initialization."""
        assert reserves.i == 0.05
        assert reserves.expense_rate == 0.0

    def test_prospective_reserves(self, reserves):
        """Test prospective reserve calculations."""
        annual_premium = 2000

        # Whole life reserve
        wl_reserve = reserves.prospective_reserve_whole_life(30, 10, annual_premium)
        assert wl_reserve >= 0

        # Term reserve
        term_reserve = reserves.prospective_reserve_term(30, 25, 10, annual_premium)
        assert term_reserve >= 0

        # Endowment reserve
        endowment_reserve = reserves.prospective_reserve_endowment(30, 25, 10, annual_premium)
        assert endowment_reserve >= 0

    def test_retrospective_reserve(self, reserves):
        """Test retrospective reserve calculations."""
        annual_premium = 1500

        retro_reserve = reserves.retrospective_reserve_whole_life(30, 8, annual_premium)
        assert retro_reserve >= 0

    def test_net_level_premium_reserve(self, reserves):
        """Test net level premium reserve."""
        net_premium = 1200

        nlp_reserve = reserves.net_level_premium_reserve(30, 7, net_premium)
        assert nlp_reserve >= 0

    def test_gross_reserve(self, reserves):
        """Test gross reserve calculations."""
        net_reserve = 50000
        duration = 5

        gross = reserves.gross_reserve(net_reserve, duration)
        assert gross >= net_reserve

    def test_zillmerized_reserve(self, reserves):
        """Test Zillmerized reserve calculations."""
        net_reserve = 30000
        initial_expenses = 5000

        # During amortization period, should be less than net reserve
        zillmer = reserves.zillmerized_reserve(net_reserve, initial_expenses, 3)
        assert zillmer < net_reserve
        assert zillmer >= 0

        # After amortization period, should equal net reserve (fully amortized)
        zillmer_full = reserves.zillmerized_reserve(net_reserve, initial_expenses, 15)
        assert zillmer_full == net_reserve
