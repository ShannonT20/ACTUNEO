"""
Tests for the life modules against worked answers.

The expected figures are published answers to standard life contingencies
questions on the AM92 mortality table: gross premiums and reserves,
with-profits benefits, mortality profit, competing risks, unit-linked
policies and profit testing. Tolerances reflect the rounding of the
published answers, which are worked from tables printed to a few decimals.
"""

import numpy as np
import pandas as pd
import pytest
from actuneo.mortality import (
    MortalityTable, SelectMortalityTable, SurvivalFunctions, load_am92,
)
from actuneo.life import (
    LifePolicy, Expenses, simple_bonus_benefits, compound_bonus_benefits,
    MultipleDecrementTable, MultiStateModel, dependent_from_independent,
    independent_from_dependent, ProfitTest, ProfitSignature, UnitLinkedPolicy,
    zeroise_negative_cashflows,
)


@pytest.fixture(scope="module")
def am92():
    return load_am92()


class TestAM92:
    """The AM92 table reproduces its published values."""

    def test_ultimate_rates_and_functions(self, am92):
        table = am92.ultimate
        assert table.qx(40) == 0.000937
        assert table.qx(60) == 0.008022
        assert table.qx(65) == 0.014243
        assert table.lx(17) == 10000
        at_4 = SurvivalFunctions(table, 0.04)
        assert at_4.annuity_due(40) == pytest.approx(20.005, abs=5e-4)
        assert at_4.assurance(40) == pytest.approx(0.23056, abs=5e-6)
        assert at_4.annuity_due(60) == pytest.approx(14.134, abs=5e-4)
        assert at_4.assurance(60) == pytest.approx(0.45640, abs=5e-6)
        assert at_4.annuity_due(65) == pytest.approx(12.276, abs=5e-4)
        assert at_4.annuity_due(40, 20) == pytest.approx(13.927, abs=5e-4)
        assert at_4.endowment_assurance(40, 20) == pytest.approx(0.46433, abs=5e-6)
        at_6 = SurvivalFunctions(table, 0.06)
        assert at_6.annuity_due(40) == pytest.approx(15.491, abs=5e-4)
        assert at_6.assurance(40) == pytest.approx(0.12313, abs=1.5e-5)

    def test_select_rates(self, am92):
        assert am92.select_period == 2
        assert am92.q(62) == 0.007164          # q[62]
        assert am92.q(62, 1) == 0.010815       # q[62]+1
        assert am92.q(62, 2) == am92.ultimate.qx(64)
        assert am92.q(30, 1) == 0.000569
        assert am92.q(40) < am92.q(39, 1) < am92.ultimate.qx(40)

    def test_select_functions(self, am92):
        table = am92.table_for(40)
        at_4 = SurvivalFunctions(table, 0.04)
        assert at_4.annuity_due(40) == pytest.approx(20.009, abs=5e-4)
        assert at_4.assurance(40) == pytest.approx(0.23041, abs=5e-6)
        assert table.npx(40, 20) == pytest.approx(0.94245, abs=5e-6)
        # After the select period the select and ultimate tables agree
        assert am92.lx(40, 2) == pytest.approx(am92.ultimate.lx(42))
        assert am92.lx(40) < am92.ultimate.lx(40)
        assert am92.lx(40) == pytest.approx(9854.30, abs=0.02)

    def test_table_for_a_life_part_way_through_selection(self, am92):
        table = am92.table_for(60, duration=1)
        assert table.min_age == 61
        assert table.qx(61) == am92.q(60, 1)
        assert table.qx(62) == am92.ultimate.qx(62)

    def test_small_select_table(self):
        ages = [50, 51, 52, 53]
        ultimate = [0.010, 0.012, 0.014, 1.0]
        select = SelectMortalityTable(ages, ultimate, [[0.005, 0.006, 0.007, np.nan]],
                                      metadata={"radix": 1000})
        assert select.q(50) == 0.005 and select.q(50, 1) == 0.012
        assert select.q(53) == 1.0   # not tabulated: ultimate rate
        assert select.lx(50) == pytest.approx(990 / 0.995)
        frame = select.to_dataframe()
        assert list(frame.columns) == ["age", "q_select_0", "q_ultimate"]
        with pytest.raises(ValueError):
            SelectMortalityTable(ages, ultimate, [[0.005, 0.006]])


class TestGrossPremiums:
    def test_loss_on_a_term_assurance(self, am92):
        """5-year term on (60), sum assured 10,000, premium 200, at 5.5%."""
        policy = LifePolicy(am92.ultimate, 60, 0.055, term=5, death_benefit=10_000)
        assert policy.expected_loss(200) == pytest.approx(-462.06, abs=0.01)
        assert policy.loss_standard_deviation(200) == pytest.approx(1920, abs=1)
        distribution = policy.loss_distribution(200)
        assert distribution["probability"].sum() == pytest.approx(1)
        assert len(distribution) == 6

    def test_premium_with_a_risk_margin(self, am92):
        """2-year term, 50,000: premium covers the benefits plus 10% of their standard deviation."""
        policy = LifePolicy(am92.ultimate, 60, 0.04, term=2, death_benefit=50_000)
        assert policy.epv_benefits() == pytest.approx(798.88, abs=0.1)
        deviation = policy.benefit_standard_deviation()
        assert deviation == pytest.approx(6082.9, abs=0.1)
        premium = policy.gross_premium(extra=0.1 * deviation)
        assert premium == pytest.approx(720.21, abs=0.05)

    def test_with_profits_endowment(self, am92):
        """20-year endowment on [40], 10,000; initial expenses 150, claim expenses 300."""
        table = am92.table_for(40)
        expenses = Expenses(initial=150, claim=300)
        deaths, maturity = simple_bonus_benefits(10_000, 0.02, 20)
        assert deaths[0] == 10_000 and deaths[1] == 10_200 and maturity == 14_000
        simple = LifePolicy(table, 40, 0.04, term=20, death_benefit=deaths,
                            survival_benefit=maturity, expenses=expenses)
        assert simple.gross_premium() == pytest.approx(483.30, abs=0.01)

        deaths, maturity = compound_bonus_benefits(10_000, 0.04, 20)
        compound = LifePolicy(table, 40, 0.04, term=20, death_benefit=deaths,
                              survival_benefit=maturity, expenses=expenses)
        assert compound.gross_premium() == pytest.approx(737.05, abs=0.01)

    def test_whole_life_with_commission(self, am92):
        """Whole life 75,000 on [50] at 6%, with initial and renewal commission and expenses."""
        expenses = Expenses(initial=325, initial_premium=1.0, renewal=75, renewal_premium=0.025)
        policy = LifePolicy(am92.table_for(50), 50, 0.06, death_benefit=75_000,
                            expenses=expenses)
        premium = policy.gross_premium()
        assert premium == pytest.approx(1308.56, abs=0.05)
        # At that premium the expected loss is nil
        assert policy.expected_loss(premium) == pytest.approx(0, abs=1e-6)
        assert policy.gross_premium_reserve(0, premium) == pytest.approx(0, abs=1e-6)

    def test_premium_for_a_probability_of_loss(self, am92):
        policy = LifePolicy(am92.table_for(50), 50, 0.06, death_benefit=75_000,
                            expenses=Expenses(initial=325, initial_premium=1.0, renewal=75,
                                              renewal_premium=0.025))
        premium = policy.premium_for_loss_probability(0.10)
        assert policy.probability_of_loss(premium) <= 0.10
        assert policy.probability_of_loss(premium - 1) > 0.10
        assert premium > policy.gross_premium()

    def test_net_premium_matches_survival_functions(self, am92):
        table = am92.ultimate
        sf = SurvivalFunctions(table, 0.04)
        endowment = LifePolicy(table, 40, 0.04, term=20, death_benefit=1, survival_benefit=1)
        assert endowment.net_premium() == pytest.approx(sf.net_annual_premium(40, 20, "endowment"))
        whole = LifePolicy(table, 40, 0.04, death_benefit=1)
        assert whole.net_premium() == pytest.approx(sf.net_annual_premium(40))
        limited = LifePolicy(table, 40, 0.04, death_benefit=1, premium_term=10)
        assert limited.net_premium() == pytest.approx(
            sf.net_annual_premium(40, premium_term=10))

    def test_invalid(self, am92):
        with pytest.raises(ValueError, match="one value for each"):
            LifePolicy(am92.ultimate, 40, 0.04, term=10, death_benefit=[1, 2])
        with pytest.raises(ValueError, match="fixed term"):
            LifePolicy(am92.ultimate, 40, 0.04, survival_benefit=1)
        with pytest.raises(ValueError, match="premium_term"):
            LifePolicy(am92.ultimate, 40, 0.04, term=10, death_benefit=1, premium_term=12)


class TestReserves:
    def test_term_assurance_reserve_both_ways(self, am92):
        """10-year term 500,000 on (30) for a premium of 330.05: reserve after 5 years."""
        policy = LifePolicy(am92.ultimate, 30, 0.04, term=10, death_benefit=500_000)
        assert policy.net_premium() == pytest.approx(330.05, abs=0.15)
        prospective = policy.net_premium_reserve(5, 330.05)
        retrospective = policy.retrospective_reserve(5, 330.05, with_expenses=False)
        assert prospective == pytest.approx(182, abs=1)
        assert retrospective == pytest.approx(182, abs=1)

    def test_prospective_equals_retrospective_on_the_premium_basis(self, am92):
        expenses = Expenses(initial=200, initial_premium=0.5, renewal=30,
                            renewal_premium=0.03, renewal_inflation=0.02, claim=100)
        policy = LifePolicy(am92.table_for(35), 35, 0.05, term=25, death_benefit=100_000,
                            survival_benefit=100_000, expenses=expenses)
        premium = policy.gross_premium()
        for t in (1, 5, 12, 24):
            assert policy.gross_premium_reserve(t, premium) == pytest.approx(
                policy.retrospective_reserve(t, premium), rel=1e-9)
        assert policy.gross_premium_reserve(25, premium) == pytest.approx(100_100)

    def test_reserve_recursion(self, am92):
        """(V + premium - expenses)(1 + i) = q * benefit + p * next reserve."""
        expenses = Expenses(initial=300, renewal=43, claim=400)
        policy = LifePolicy(am92.ultimate, 50, 0.04, term=10, death_benefit=100_000,
                            expenses=expenses)
        premium = policy.gross_premium()
        for t in (0, 3, 7):
            q = am92.ultimate.qx(50 + t)
            expense = 300 if t == 0 else 43
            start = policy.gross_premium_reserve(t, premium)
            end = policy.gross_premium_reserve(t + 1, premium)
            assert (start + premium - expense) * 1.04 == pytest.approx(
                q * 100_400 + (1 - q) * end)

    def test_with_profits_reserve_for_a_policy_in_force(self, am92):
        """
        Endowment taken out at 35 for 25 years, now aged 40 with 3% compound bonuses
        declared for five years; future bonuses 1.92308%; 6% interest.
        """
        current = 50_000 * 1.03 ** 5
        deaths, maturity = compound_bonus_benefits(current, 0.0192308, 20)
        policy = LifePolicy(am92.ultimate, 40, 0.06, term=20, death_benefit=deaths,
                            survival_benefit=maturity, in_force=True,
                            expenses=Expenses(renewal_premium=0.05, claim=350))
        assert policy.gross_premium_reserve(0, 1500) == pytest.approx(9892, abs=1)

    def test_decreasing_term_paid_immediately_on_death(self, am92):
        benefits = 150_000 - 10_000 * np.arange(10)
        policy = LifePolicy(am92.ultimate, 50, 0.04, term=10, death_benefit=benefits,
                            timing="immediate",
                            expenses=Expenses(initial=300, renewal=43, claim=400))
        premium = policy.gross_premium()
        assert premium == pytest.approx(491.31, abs=0.1)
        # The reserve is negative: cover is falling while the premium stays level
        assert policy.gross_premium_reserve(4, premium) < 0


class TestMortalityProfit:
    def test_pure_endowment(self, am92):
        """Pure endowment of 40,000 on [30] for 10 years: second year with 50 lives, 1 death."""
        policy = LifePolicy(am92.table_for(30), 30, 0.04, term=10, survival_benefit=40_000)
        premium = policy.net_premium()
        assert premium == pytest.approx(3189.71, abs=0.15)
        result = policy.mortality_profit(lives=50, deaths=1, t=1, premium=premium)
        assert result["death_strain_at_risk"] == pytest.approx(-6773.71, abs=1)
        assert result["expected_death_strain"] == pytest.approx(-192.71, abs=0.05)
        assert result["mortality_profit"] == pytest.approx(6581, abs=1)

    def test_no_profit_when_experience_matches(self, am92):
        policy = LifePolicy(am92.ultimate, 45, 0.04, term=20, death_benefit=1000)
        expected_deaths = 500 * am92.ultimate.qx(54)
        result = policy.mortality_profit(500, expected_deaths, t=9)
        assert result["mortality_profit"] == pytest.approx(0, abs=1e-9)
        # More deaths than expected on a term assurance is a loss
        assert policy.mortality_profit(500, expected_deaths + 1, t=9)["mortality_profit"] < 0

    def test_gross_basis(self, am92):
        policy = LifePolicy(am92.ultimate, 45, 0.04, term=20, death_benefit=1000,
                            expenses=Expenses(claim=50, renewal=5))
        net = policy.death_strain_at_risk(9)
        gross = policy.death_strain_at_risk(9, basis="gross")
        assert gross > net
        with pytest.raises(ValueError):
            policy.death_strain_at_risk(9, basis="other")


class TestCompetingRisks:
    def test_dependent_and_independent_rates(self):
        independent = {"death": 0.01, "withdrawal": 0.08}
        dependent = dependent_from_independent(independent)
        # Each dependent rate is lower: the other cause removes lives first
        assert dependent["death"] < 0.01 and dependent["withdrawal"] < 0.08
        assert 1 - sum(dependent.values()) == pytest.approx(0.99 * 0.92)
        back = independent_from_dependent(dependent)
        assert back["death"] == pytest.approx(0.01)
        assert back["withdrawal"] == pytest.approx(0.08)

    def test_table_from_decrements(self):
        table = MultipleDecrementTable.from_decrements(
            [50, 51], [5000, 4848], {"alpha": [86, 80], "beta": [52, 56], "gamma": [14, 20]})
        assert table.dependent_rate(50, "alpha") == pytest.approx(86 / 5000)
        assert table.al[1] == pytest.approx(4848)
        # Leaves by beta between 51 and 52
        assert table.probability_of_decrement(50, "beta", deferred=1) == pytest.approx(56 / 5000)
        assert table.force(50, "alpha") == pytest.approx(0.017467, abs=5e-7)
        assert table.force(51, "alpha") == pytest.approx(0.016773, abs=5e-7)
        assert table.survival_probability(50, 2) == pytest.approx(4692 / 5000)

    def test_revised_table_after_a_change_in_withdrawal(self):
        """Force of withdrawal falls to 75% of its previous level; mortality unchanged."""
        table = MultipleDecrementTable.from_decrements(
            [40, 41], [10_000, 9_855], {"death": [25, 27], "withdrawal": [120, 144]})
        assert table.force(40, "death") == pytest.approx(0.002518, abs=5e-7)
        assert table.force(40, "withdrawal") == pytest.approx(0.012088, abs=5e-7)
        revised = table.with_scaled_force("withdrawal", 0.75)
        frame = revised.to_dataframe()
        np.testing.assert_allclose(frame["al"], [10_000, 9_884.8], atol=0.05)
        np.testing.assert_allclose(frame["ad_death"], [25.0, 27.1], atol=0.05)
        np.testing.assert_allclose(frame["ad_withdrawal"], [90.1, 108.5], atol=0.05)
        assert revised.force(40, "death") == pytest.approx(table.force(40, "death"))

    def test_pension_scheme_benefits(self):
        table = MultipleDecrementTable.from_decrements(
            [63, 64], [2200, 2055],
            {"retirement": [100, 200], "ill_health": [25, 30], "death": [20, 25]})
        # 20,000 on death after the 63rd birthday for a member counted from 3,111 at age 40
        epv = 20_000 * (20 * 1.04 ** -23.5 + 25 * 1.04 ** -24.5) / 3111
        assert epv == pytest.approx(112.64, abs=0.01)
        assert table.epv_of_decrement_benefit(63, "death", 0.04, 20_000) == pytest.approx(
            20_000 * (20 * 1.04 ** -0.5 + 25 * 1.04 ** -1.5) / 2200)

    def test_three_state_sickness_model(self):
        model = MultiStateModel(["able", "sick", "dead"], {
            ("able", "sick"): 0.05, ("able", "dead"): 0.02,
            ("sick", "able"): 0.04, ("sick", "dead"): 0.03,
        })
        assert model.probability_of_staying("able", 15) == pytest.approx(np.exp(-0.07 * 15))
        # No-claim bonus of half the premiums if able throughout 15 years, at 6% force
        bonus = 0.5 * 2500 * 15 * np.exp(-0.06 * 15) * model.probability_of_staying("able", 15)
        assert bonus == pytest.approx(18_750 * np.exp(-1.95))
        probabilities = model.transition_probabilities(10)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        assert probabilities.loc["dead", "dead"] == pytest.approx(1.0)
        # Being able after 10 years includes having been sick and recovered
        assert model.probability("able", "able", 10) > model.probability_of_staying("able", 10)

    def test_two_state_model_reduces_to_a_constant_force(self):
        model = MultiStateModel(["alive", "dead"], {("alive", "dead"): 0.02})
        assert model.probability("alive", "alive", 10) == pytest.approx(np.exp(-0.2))
        annuity = model.epv_annuity("alive", "alive", 20, 0.05)
        assert annuity == pytest.approx((1 - np.exp(-0.07 * 20)) / 0.07)
        assurance = model.epv_transition_benefit("alive", "alive", "dead", 20, 0.05)
        assert assurance == pytest.approx(0.02 * annuity)


class TestProfitTesting:
    def test_one_year_of_an_endowment(self):
        test = ProfitTest(premiums=[1500], expenses=[35], interest=0.08,
                          death_probability=0.02, death_benefit=25_000,
                          reserves=[8000, 9300], surrender_probability=0.05,
                          surrender_value=8 * 1500, claim_expense=75)
        assert test.profit_vector[0] == pytest.approx(467.95)

    @pytest.fixture
    def endowment(self):
        return ProfitTest(
            premiums=[3600] * 5, expenses=[750, 15, 15, 15, 15], interest=0.06,
            death_probability=0.01, death_benefit=20_000,
            reserves=[0, 3100, 6800, 10_900, 15_300, 0],
            surrender_probability=[0.05] * 4 + [0], surrender_value=[2800, 6250, 10_000, 14_500, 0],
            maturity_benefit=20_000, surrender_basis="year_end",
        )

    def test_profit_vector_and_signature(self, endowment):
        np.testing.assert_allclose(endowment.profit_vector,
                                   [-233.15, 181.33, 61.65, 46.70, 18.10], atol=0.01)
        np.testing.assert_allclose(endowment.in_force,
                                   [1, 0.9405, 0.8845, 0.8319, 0.7824], atol=5e-5)
        np.testing.assert_allclose(endowment.profit_signature,
                                   [-233.15, 170.54, 54.53, 38.85, 14.16], atol=0.01)
        table = endowment.table()
        assert table.loc[1, "surrender_cost"] == pytest.approx(-138.60)
        assert table.loc[5, "maturity_cost"] == pytest.approx(-19_800)

    def test_summary_measures(self, endowment):
        rate = endowment.internal_rate_of_return()
        assert endowment.net_present_value(rate) == pytest.approx(0, abs=1e-6)
        assert round(rate, 2) == 0.12
        assert endowment.net_present_value(0.08) > 0 > endowment.net_present_value(0.15)
        assert endowment.discounted_payback_period(0.08) == 4
        margin = endowment.profit_margin(0.08)
        assert margin == pytest.approx(
            endowment.net_present_value(0.08) / endowment.present_value_of_premiums(0.08))

    def test_unit_linked_single_premium(self):
        """10,000 single premium, 9% growth, 2% annual management charge."""
        policy = UnitLinkedPolicy([10_000, 0, 0, 0, 0], unit_growth=0.09,
                                  management_charge=0.02)
        fund = policy.unit_fund()
        np.testing.assert_allclose(fund["management_charge"],
                                   [218.00, 232.87, 248.75, 265.71, 283.84], atol=0.005)
        np.testing.assert_allclose(
            fund["fund_at_end"], [10_682.00, 11_410.51, 12_188.71, 13_019.98, 13_907.94],
            atol=0.01)

        expenses = [570] + [20 * 1.05 ** t for t in range(1, 5)]
        result = policy.profit_test(expenses=expenses, interest=0.09, death_probability=0.005,
                                    surrender_probability=0.05, surrender_basis="year_end")
        np.testing.assert_allclose(result.profit_vector,
                                   [-403.30, 209.98, 224.71, 240.48, 257.34], atol=0.01)
        np.testing.assert_allclose(result.profit_signature,
                                   [-403.30, 198.48, 200.78, 203.10, 205.44], atol=0.01)
        assert result.net_present_value(0.12) == pytest.approx(186.69, abs=0.02)

    def test_unit_linked_regular_premium(self):
        """Allocation of 95%, a 5% bid/offer spread and a 1% charge."""
        policy = UnitLinkedPolicy([1000] * 3, allocation=0.95, bid_offer_spread=0.05,
                                  unit_growth=0.075, management_charge=0.01)
        fund = policy.unit_fund()
        first = 1000 * 0.95 * 0.95 * 1.075 * 0.99
        assert fund.loc[1, "fund_at_end"] == pytest.approx(first)
        assert fund.loc[2, "fund_at_end"] == pytest.approx((first + 902.5) * 1.075 * 0.99)
        guaranteed = policy.profit_test(expenses=[100, 30, 30], interest=0.075,
                                        death_probability=0.01, minimum_death_benefit=5000)
        plain = policy.profit_test(expenses=[100, 30, 30], interest=0.075,
                                   death_probability=0.01)
        # A guaranteed minimum death benefit above the fund costs the non-unit fund money
        assert (guaranteed.profit_vector < plain.profit_vector).all()
        assert guaranteed.columns.loc[1, "extra_death_cost"] == pytest.approx(
            -0.01 * (5000 - first))

    def test_zeroising_negative_cashflows(self):
        result = zeroise_negative_cashflows([100, -50, -30, 200], [0.9, 0.9, 0.9, 0.9], 0.05)
        # Reserve at the start of year 3 covers that year's shortfall
        assert result.loc[3, "reserve_at_start"] == pytest.approx(30 / 1.05)
        assert result.loc[2, "reserve_at_start"] == pytest.approx(
            (0.9 * 30 / 1.05 + 50) / 1.05)
        np.testing.assert_allclose(result["profit_vector"].iloc[1:3], 0, atol=1e-9)
        assert result.loc[4, "profit_vector"] == 200
        assert result.loc[1, "profit_vector"] < 100
        # Nothing to do when no cashflow is negative
        clean = zeroise_negative_cashflows([-100, 50, 60], 0.95, 0.05)
        assert (clean["reserve_at_start"] == 0).all()

    def test_signature_from_a_vector(self):
        signature = ProfitSignature([-100, 60, 70], [0.9, 0.9, 0.9], premiums=[500, 500, 500])
        np.testing.assert_allclose(signature.in_force, [1, 0.9, 0.81])
        assert signature.present_value_of_premiums(0.0) == pytest.approx(500 * 2.71)
        assert signature.discounted_payback_period(0.0) == 3
