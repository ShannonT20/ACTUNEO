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
        assert 0 < premium < 1

        # Premium should increase with age (shorter remaining lifetime)
        assert life_assurance.whole_life_assurance(40) > premium

        # A_x = 1 - d * ä_x
        d = 0.05 / 1.05
        assert premium == pytest.approx(1 - d * life_assurance.whole_life_annuity(30, due=True))

    def test_continuous_exceeds_discrete(self, life_assurance):
        discrete = life_assurance.whole_life_assurance(30)
        continuous = life_assurance.whole_life_assurance(30, discrete=False)
        assert continuous == pytest.approx(discrete * 0.05 / np.log(1.05))

    def test_term_assurance(self, life_assurance):
        """Test term assurance premiums."""
        premium_10y = life_assurance.term_assurance(30, 10)
        premium_20y = life_assurance.term_assurance(30, 20)

        assert 0 < premium_10y < premium_20y  # Longer term should cost more
        assert premium_20y < life_assurance.whole_life_assurance(30)

    def test_one_year_term(self, life_assurance):
        """A one-year term assurance is v * q_x."""
        expected = life_assurance.mt.qx(45) / 1.05
        assert life_assurance.term_assurance(45, 1) == pytest.approx(expected)

    def test_endowment_assurance(self, life_assurance):
        """Test endowment assurance premiums."""
        endowment = life_assurance.endowment_assurance(30, 20)
        term = life_assurance.term_assurance(30, 20)
        pure_endowment = life_assurance.pure_endowment(30, 20)

        assert endowment == pytest.approx(term + pure_endowment)
        # An endowment is worth more than a certain payment at maturity
        assert endowment > 1.05 ** -20

    def test_pure_endowment(self, life_assurance):
        """Test pure endowment calculations."""
        pe = life_assurance.pure_endowment(30, 20)
        expected = life_assurance.sf.npx(30, 20) * life_assurance.v ** 20
        assert pe == pytest.approx(expected)

    def test_deferred_assurance(self, life_assurance):
        """Whole life = term + deferred whole life."""
        la = life_assurance
        assert la.whole_life_assurance(30) == pytest.approx(
            la.term_assurance(30, 15) + la.deferred_assurance(30, 15)
        )
        assert la.term_assurance(30, 25) == pytest.approx(
            la.term_assurance(30, 15) + la.deferred_assurance(30, 15, 10)
        )

    def test_joint_and_last_survivor(self, life_assurance):
        la = life_assurance
        joint = la.joint_life_assurance(40, 45)
        last = la.last_survivor_assurance(40, 45)
        single_x, single_y = la.whole_life_assurance(40), la.whole_life_assurance(45)

        # First death comes sooner, second death later, than either single death
        assert joint > max(single_x, single_y)
        assert last < min(single_x, single_y)
        assert joint + last == pytest.approx(single_x + single_y)

    def test_contingent_assurance(self, life_assurance):
        la = life_assurance
        assert 0 < la.contingent_assurance(40, 45) < la.joint_life_assurance(40, 45)
        # Against a much younger counter life, nearly every death of (x) is paid
        assert la.contingent_assurance(80, 20, 10) == pytest.approx(
            la.term_assurance(80, 10), rel=0.01
        )

    def test_second_life_table(self, life_assurance, sample_mortality_table):
        """Lighter mortality for the second life delays the first death."""
        lighter = MortalityTable(
            sample_mortality_table.ages,
            np.append(sample_mortality_table.qx_values[:-1] * 0.5, 1.0),
        )
        same = life_assurance.joint_life_assurance(40, 45)
        mixed = life_assurance.joint_life_assurance(40, 45, table_y=lighter)
        assert mixed < same

    def test_gross_premium(self, sample_mortality_table):
        """Test gross premium calculations."""
        la = LifeAssurance(sample_mortality_table, 0.05, expense_loading=0.1)
        assert la.gross_premium(100, 10) == pytest.approx(120)

    def test_annual_premium(self, life_assurance):
        la = life_assurance
        nsp = la.term_assurance(30, 20)
        annuity = la.temporary_life_annuity(30, 20, due=True)
        assert la.annual_premium(nsp, annuity) == pytest.approx(la.net_annual_premium(30, 20, "term"))
        assert la.annual_premium(nsp, annuity, gross_margin=0.2) == pytest.approx(1.2 * nsp / annuity)
        with pytest.raises(ValueError):
            la.annual_premium(nsp, 0)

    def test_reserve_calculations(self, life_assurance):
        """Net premium reserves start at zero and build up."""
        la = life_assurance

        assert la.reserve_whole_life(30, 0) == pytest.approx(0, abs=1e-12)
        assert la.reserve_term(30, 20, 0) == pytest.approx(0, abs=1e-12)
        assert la.reserve_endowment(30, 20, 0) == pytest.approx(0, abs=1e-12)

        # tVx = 1 - ä_{x+t} / ä_x
        expected = 1 - la.whole_life_annuity(35, due=True) / la.whole_life_annuity(30, due=True)
        assert la.reserve_whole_life(30, 5) == pytest.approx(expected)

        # Endowment reserve grows to the sum assured at maturity
        reserves = [la.reserve_endowment(30, 20, t) for t in range(21)]
        assert all(b > a for a, b in zip(reserves, reserves[1:]))
        assert reserves[-1] == pytest.approx(1.0)

        assert la.reserve_term(30, 20, 20) == 0.0
        assert 0 < la.reserve_term(30, 20, 10) < la.reserve_endowment(30, 20, 10)


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

        imm = ann.immediate_annuity(10, payment=100)
        expected_imm = 100 * ((1 - (1.05) ** (-10)) / 0.05)
        assert imm == pytest.approx(expected_imm)

        due = ann.annuity_due(10, payment=100)
        assert due == pytest.approx(expected_imm * 1.05)

        assert Annuities(interest_rate=0.0).immediate_annuity(10, 100) == 1000

    def test_life_annuity_needs_table(self, annuities_det):
        with pytest.raises(ValueError, match="Mortality table required"):
            annuities_det.life_annuity_due(30)

    def test_life_annuities(self, annuities):
        """Test life annuity calculations."""
        ann = annuities

        life_imm = ann.life_annuity_immediate(30)
        life_due = ann.life_annuity_due(30)
        assert life_due == pytest.approx(life_imm + 1)
        assert ann.life_annuity_due(30, payment=1200) == pytest.approx(1200 * life_due)

        # A life annuity is worth less than a perpetuity
        assert life_imm < 1 / 0.05

        temp = ann.temporary_life_annuity_immediate(30, 20)
        assert temp < life_imm
        assert temp < Annuities(interest_rate=0.05).immediate_annuity(20)
        assert ann.annuity_certain_with_life_contingency(30, 20) == temp

    def test_deferred_and_guaranteed(self, annuities):
        ann = annuities
        whole = ann.life_annuity_immediate(60)
        temp = ann.temporary_life_annuity_immediate(60, 10)
        deferred = ann.deferred_life_annuity(60, 10)
        assert temp + deferred == pytest.approx(whole)

        deferred_due = ann.deferred_life_annuity(60, 10, due=True)
        assert deferred_due == pytest.approx(deferred + ann.sf.pure_endowment(60, 10))

        # The guarantee adds value, most of all at old ages
        guaranteed = ann.guaranteed_annuity(60, 10)
        assert guaranteed > whole
        assert guaranteed == pytest.approx(ann.immediate_annuity(10) + deferred)

    def test_monthly_annuity(self, annuities):
        ann = annuities
        monthly_advance = ann.monthly_life_annuity(65, annual_payment=12000)
        monthly_arrears = ann.monthly_life_annuity(65, annual_payment=12000, due=False)
        assert ann.life_annuity_immediate(65, 12000) < monthly_arrears < monthly_advance
        assert monthly_advance < ann.life_annuity_due(65, 12000)
        assert monthly_advance - monthly_arrears == pytest.approx(1000)

    def test_two_life_annuities(self, annuities):
        ann = annuities
        x, y = 65, 60
        a_x, a_y = ann.life_annuity_immediate(x), ann.life_annuity_immediate(y)
        joint = ann.joint_life_annuity(x, y)
        last = ann.last_survivor_annuity(x, y)

        assert joint < min(a_x, a_y)
        assert last > max(a_x, a_y)
        assert joint + last == pytest.approx(a_x + a_y)
        assert ann.reversionary_annuity(x, y) == pytest.approx(a_y - joint)
        assert ann.contingent_annuity(x, y) == pytest.approx(a_x - joint)
        assert ann.joint_life_annuity(x, y, payment=500) == pytest.approx(500 * joint)

    def test_increasing_annuities(self, annuities_det):
        """Test increasing annuity calculations."""
        ann = annuities_det

        level = ann.immediate_annuity(5, payment=100)
        assert ann.increasing_annuity(5, payment=100) == pytest.approx(level)
        assert ann.increasing_annuity(5, payment=100, increase_rate=0.08) > level
        # Increases at the rate of interest: every payment is worth payment * v
        assert ann.increasing_annuity(5, payment=100, increase_rate=0.05) == pytest.approx(
            5 * 100 / 1.05
        )
        assert ann.decreasing_annuity(5, payment=100, decrease_rate=0.1) < level

    def test_arithmetic_annuities(self, annuities_det):
        """(Ia)n + (Da)n = (n + 1) * a_n"""
        ann = annuities_det
        n = 10
        total = ann.arithmetic_increasing_annuity(n) + ann.arithmetic_decreasing_annuity(n)
        assert total == pytest.approx((n + 1) * ann.immediate_annuity(n))
        # (Ia)n = (ä_n - n v^n) / i
        expected = (ann.annuity_due(n) - n * 1.05 ** -n) / 0.05
        assert ann.arithmetic_increasing_annuity(n) == pytest.approx(expected)

    def test_drawdown(self, annuities_det):
        ann = annuities_det

        # Withdrawing exactly the interest keeps the fund level
        level = ann.annuity_with_withdrawal(1000, 0.05, periods=10)
        assert level['annual_payment'] == pytest.approx(50)
        assert level['remaining_principal'] == pytest.approx(1000)
        assert level['exhausted_in'] is None
        assert len(level['schedule']) == 10

        # Withdrawing the 10-year loan instalment empties the fund in year 10
        instalment_rate = 1 / ann.immediate_annuity(10)
        run_down = ann.annuity_with_withdrawal(1000, instalment_rate)
        assert run_down['exhausted_in'] == 10
        assert run_down['remaining_principal'] == 0.0


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
        """Reserves on the net premium are zero at outset and positive later."""
        sf = reserves.sf
        sa = 1000.0

        p_wl = sa * sf.net_annual_premium(30)
        assert reserves.prospective_reserve_whole_life(30, 0, p_wl) == pytest.approx(0, abs=1e-9)
        assert reserves.prospective_reserve_whole_life(30, 10, p_wl) > 0

        p_term = sa * sf.net_annual_premium(30, 25, "term")
        assert reserves.prospective_reserve_term(30, 25, 0, p_term) == pytest.approx(0, abs=1e-9)
        assert reserves.prospective_reserve_term(30, 25, 10, p_term) > 0
        assert reserves.prospective_reserve_term(30, 25, 25, p_term) == 0.0

        p_end = sa * sf.net_annual_premium(30, 25, "endowment")
        assert reserves.prospective_reserve_endowment(30, 25, 0, p_end) == pytest.approx(0, abs=1e-9)
        assert reserves.prospective_reserve_endowment(30, 25, 25, p_end) == pytest.approx(sa)

    def test_reserve_recursion(self, reserves):
        """(tV + P)(1 + i) = q * SA + p * (t+1)V"""
        sf, sa, x, t = reserves.sf, 1000.0, 30, 12
        premium = sa * sf.net_annual_premium(x)
        v_t = reserves.prospective_reserve_whole_life(x, t, premium)
        v_next = reserves.prospective_reserve_whole_life(x, t + 1, premium)
        q = sf.mt.qx(x + t)
        assert (v_t + premium) * 1.05 == pytest.approx(q * sa + (1 - q) * v_next)

    def test_negative_reserve_and_floor(self, reserves):
        """A premium above the net premium gives a negative reserve at outset."""
        premium = 2 * 1000 * reserves.sf.net_annual_premium(30)
        assert reserves.prospective_reserve_whole_life(30, 0, premium) < 0
        assert reserves.prospective_reserve_whole_life(30, 0, premium, floor_at_zero=True) == 0.0

    def test_retrospective_equals_prospective(self, reserves):
        """On the net premium the two reserves agree."""
        premium = 1000 * reserves.sf.net_annual_premium(30)
        retro = reserves.retrospective_reserve_whole_life(30, 8, premium)
        prospective = reserves.prospective_reserve_whole_life(30, 8, premium)
        assert retro == pytest.approx(prospective)
        assert reserves.retrospective_reserve_whole_life(30, 0, premium) == 0.0

    def test_net_level_premium_reserve(self, reserves):
        """Test net level premium reserve."""
        premium = 1000 * reserves.sf.net_annual_premium(30)
        with_premium = reserves.net_level_premium_reserve(30, 7, premium)
        assert with_premium == pytest.approx(reserves.net_level_premium_reserve(30, 7))
        assert with_premium > 0

    def test_zillmer_reserve(self, reserves):
        x, n, sa, rate = 30, 20, 1000.0, 0.03
        premium = sa * reserves.sf.net_annual_premium(x, n, "endowment")

        # Negative at outset by the full allowance, equal to the net reserve at maturity
        assert reserves.zillmer_reserve(x, n, 0, rate, sa) == pytest.approx(-rate * sa)
        assert reserves.zillmer_reserve(x, n, 0, rate, sa, floor_at_zero=True) == 0.0
        assert reserves.zillmer_reserve(x, n, n, rate, sa) == pytest.approx(sa)
        mid = reserves.zillmer_reserve(x, n, 10, rate, sa)
        assert mid < reserves.prospective_reserve_endowment(x, n, 10, premium)
        assert reserves.zillmer_reserve(x, n, 10, 0.0, sa) == pytest.approx(
            reserves.prospective_reserve_endowment(x, n, 10, premium)
        )

    def test_gross_reserve(self, sample_mortality_table):
        """Test gross reserve calculations."""
        reserves = Reserves(sample_mortality_table, 0.05, expense_rate=0.02, profit_margin=0.03)
        assert reserves.gross_reserve(50000, 5) == pytest.approx(52500)
        assert reserves.gross_reserve(50000, 5, expense_reserve=100) == pytest.approx(51600)

    def test_zillmerized_reserve(self, reserves):
        """Test straight-line amortization of acquisition costs."""
        net_reserve = 30000
        initial_expenses = 5000

        assert reserves.zillmerized_reserve(net_reserve, initial_expenses, 3) == pytest.approx(26500)
        assert reserves.zillmerized_reserve(net_reserve, initial_expenses, 15) == net_reserve
        assert reserves.zillmerized_reserve(100, initial_expenses, 0) == 0

    def test_reserve_distribution(self, reserves):
        dist = reserves.reserve_distribution([100, 200, 300], 1200)
        assert dist['total_reserves'] == 600
        assert dist['mean_reserve'] == 200
        assert dist['reserve_to_portfolio_ratio'] == 0.5


class TestZimbabweTablesEndToEnd:
    """Pricing on a shipped Zimbabwe 2023 table."""

    def test_term_assurance_premium(self):
        table = MortalityTable.from_zimbabwe_2023("male_assured_lives")
        la = LifeAssurance(table, interest_rate=0.08)
        premium = la.net_annual_premium(35, 20, "term")
        assert 0 < premium < la.net_annual_premium(35, 20, "endowment")
        assert la.reserve_endowment(35, 20, 20) == pytest.approx(1.0)

    def test_whole_life_warns_on_table_ending_at_70(self):
        table = MortalityTable.from_zimbabwe_2023("group_life_male")
        with pytest.warns(UserWarning, match="ends at age 70"):
            LifeAssurance(table, 0.08).whole_life_assurance(40)
