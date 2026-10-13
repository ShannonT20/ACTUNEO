"""
Tests of the pensions module.

The projected unit method is checked against the example illustrating
paragraph 68 of IAS 19, as reproduced at three discount rates in IFRS
Interpretations Committee agenda paper 9 of November 2012. Everything else is
checked against closed-form results, the existing life functions and
relationships that must hold.
"""

import numpy as np
import pandas as pd
import pytest

from actuneo.life import Annuities, MultipleDecrementTable
from actuneo.mortality import MortalityTable, SurvivalFunctions
from actuneo.pensions import (
    METHODS,
    DBValuation,
    DCFundValuation,
    DCProjection,
    DefinedBenefitScheme,
    IAS19Period,
    SalaryScale,
    ServiceTable,
    ValuationBasis,
    analysis_of_surplus,
    pension_annuity,
    survival_probabilities,
    zimbabwe,
)


@pytest.fixture
def table(sample_mortality_table):
    return sample_mortality_table


@pytest.fixture
def service_table(table):
    """Deaths from the mortality table, withdrawals falling with age, ill health rising."""
    return ServiceTable.from_mortality(
        table, 60, start_age=20,
        withdrawal=lambda age: max(0.0, 0.10 - 0.003 * (age - 20)),
        ill_health=lambda age: 0.0005 * 1.08 ** (age - 20),
    )


@pytest.fixture
def scheme():
    return DefinedBenefitScheme(
        accrual_rate=1 / 60, normal_retirement_age=60, final_average_years=3,
        spouse_fraction=0.5, guarantee_years=5, death_in_service_multiple=2.0,
        death_in_service_spouse_fraction=0.5, ill_health_service="prospective",
    )


@pytest.fixture
def basis(table, service_table):
    return ValuationBasis(
        discount_rate=0.09, mortality_post=table, service_table=service_table,
        salary_scale=SalaryScale(0.06, {20: 1.0, 40: 1.6, 60: 1.9}),
        pension_increase=0.03, deferred_revaluation=0.02,
        proportion_married=0.85, spouse_age_difference=3,
    )


@pytest.fixture
def members():
    actives = pd.DataFrame({
        "age": [25, 32, 41, 48, 55, 59],
        "service": [2.0, 6.5, 15.0, 20.0, 30.0, 12.0],
        "salary": [9_000, 14_000, 22_000, 30_000, 41_000, 26_000],
        "sex": ["M", "F", "M", "F", "M", "M"],
    })
    pensioners = pd.DataFrame({
        "age": [63, 71.5, 80],
        "pension": [8_000, 5_500, 3_000],
        "sex": ["M", "F", "M"],
        "reversion": [True, True, False],
    })
    deferreds = pd.DataFrame({"age": [38, 52.25], "pension": [1_200, 3_400]})
    return actives, pensioners, deferreds


@pytest.fixture
def valuation(scheme, basis, members):
    actives, pensioners, deferreds = members
    return DBValuation(scheme, basis, actives, pensioners, deferreds,
                       assets=260_000, reserves=5_000)


class TestPensionAnnuity:
    def test_annual_factors_match_life_functions(self, table):
        sf = SurvivalFunctions(table, 0.05)
        assert pension_annuity(table, 65, 0.05, frequency=1) == pytest.approx(sf.annuity_due(65))
        assert pension_annuity(table, 65, 0.05, frequency=1, timing="arrears") == \
            pytest.approx(sf.annuity_immediate(65))

    def test_increasing_pension_is_an_annuity_at_the_net_rate(self, table):
        net = 1.08 / 1.03 - 1
        assert pension_annuity(table, 60, 0.08, frequency=1, increase=0.03) == \
            pytest.approx(SurvivalFunctions(table, net).annuity_due(60))

    def test_monthly_factor_by_direct_summation(self, table):
        v = 1 / 1.06
        total = 0.0
        for month in range(12 * (111 - 70)):
            t = month / 12
            year = int(t)
            survival = table.npx(70, year) * (1 - (t - year) * table.qx(70 + year))
            total += survival * v ** t / 12
        assert pension_annuity(table, 70, 0.06, frequency=12) == pytest.approx(total)
        # A year of monthly payments in advance is worth 11/24 less than annual ones, roughly
        annual = pension_annuity(table, 70, 0.06, frequency=1)
        assert annual - total == pytest.approx(11 / 24, abs=0.02)

    def test_guarantee_and_spouse_match_life_functions(self, table):
        ann = Annuities(table, 0.05)
        single = pension_annuity(table, 65, 0.05, frequency=1, timing="arrears")
        guaranteed = pension_annuity(table, 65, 0.05, frequency=1, timing="arrears", guarantee=5)
        assert guaranteed == pytest.approx(ann.guaranteed_annuity(65, 5, 1.0))
        with_spouse = pension_annuity(table, 65, 0.05, frequency=1, timing="arrears",
                                      spouse_fraction=1.0, spouse_age=62)
        assert with_spouse - single == pytest.approx(ann.reversionary_annuity(65, 62))

    def test_spouse_pension_scales_with_fraction_and_proportion_married(self, table):
        options = dict(frequency=12, increase=0.02, spouse_age=60)
        single = pension_annuity(table, 64, 0.07, frequency=12, increase=0.02)
        full = pension_annuity(table, 64, 0.07, spouse_fraction=1.0, **options)
        part = pension_annuity(table, 64, 0.07, spouse_fraction=0.5,
                               proportion_married=0.8, **options)
        assert part - single == pytest.approx(0.4 * (full - single))

    def test_guarantee_delays_the_spouse_pension(self, table):
        # With the lives certain to be paid for 10 years, the spouse's pension
        # is worth less than without a guarantee
        options = dict(frequency=1, spouse_fraction=0.5, spouse_age=70)
        plain = pension_annuity(table, 70, 0.05, **options) \
            - pension_annuity(table, 70, 0.05, frequency=1)
        guaranteed = pension_annuity(table, 70, 0.05, guarantee=10, **options) \
            - pension_annuity(table, 70, 0.05, frequency=1, guarantee=10)
        assert 0 < guaranteed < plain

    def test_fractional_age_lies_between_whole_ages(self, table):
        low = pension_annuity(table, 66, 0.05)
        high = pension_annuity(table, 65, 0.05)
        assert low < pension_annuity(table, 65.5, 0.05) < high

    def test_survival_probabilities(self, table):
        probabilities = survival_probabilities(table, 50, [0, 1, 2.5, 100])
        assert probabilities[0] == 1.0
        assert probabilities[1] == pytest.approx(table.px(50))
        assert probabilities[2] == pytest.approx(table.npx(50, 2) * (1 - 0.5 * table.qx(52)))
        assert probabilities[3] == 0.0

    def test_open_table_is_closed_with_warning(self, open_mortality_table):
        with pytest.warns(UserWarning, match="ends at age 100"):
            pension_annuity(open_mortality_table, 70, 0.05)

    def test_no_warning_when_the_cut_off_is_immaterial(self):
        import warnings
        annuitants = MortalityTable.from_south_africa("saiml98")
        assert not annuitants.is_closed
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            factor = pension_annuity(annuitants, 60, 0.05)
        assert 10 < factor < 14
        # The same table warns for a life already close to its last age
        with pytest.warns(UserWarning, match="ends at age 110"):
            pension_annuity(annuitants, 105, 0.05)

    def test_invalid_arguments(self, table):
        with pytest.raises(ValueError, match="spouse_age"):
            pension_annuity(table, 65, 0.05, spouse_fraction=0.5)
        with pytest.raises(ValueError, match="timing"):
            pension_annuity(table, 65, 0.05, timing="middle")
        with pytest.raises(ValueError, match="frequency"):
            pension_annuity(table, 65, 0.05, frequency=0)
        with pytest.raises(ValueError, match="outside the ages"):
            pension_annuity(table, 15, 0.05)


class TestServiceTableAndSalaryScale:
    def test_agrees_with_multiple_decrement_table(self):
        ages = np.arange(50, 60)
        independent = {"withdrawal": np.full(10, 0.05), "death": np.linspace(0.004, 0.012, 10),
                       "ill_health": np.full(10, 0.002), "retirement": np.zeros(10)}
        table = ServiceTable.from_independent_rates(ages, 60, **independent)
        reference = MultipleDecrementTable.from_independent_rates(ages, independent)
        for cause in independent:
            assert table.rate(54, cause) == pytest.approx(reference.dependent_rate(54, cause))
        assert table.survival(50, 10) == pytest.approx(reference.survival_probability(50, 10))
        assert table.survival(55, 0) == 1.0
        assert table.total_rate(50) == pytest.approx(
            1 - (1 - 0.05) * (1 - 0.004) * (1 - 0.002))

    def test_rates_given_in_different_forms(self):
        ages = [57, 58, 59]
        table = ServiceTable(ages, withdrawal=0.1, death={58: 0.02},
                             ill_health=lambda age: 0.001 * (age - 56), retirement=[0, 0, 0.2])
        assert table.retirement_age == 60
        assert table.rate(58, "withdrawal") == 0.1
        assert table.rate(57, "death") == 0.0 and table.rate(58, "death") == 0.02
        assert table.rate(59, "ill_health") == pytest.approx(0.003)
        assert table.rate(59, "retirement") == 0.2
        frame = table.to_dataframe()
        assert list(frame.columns) == ["age", "in_service", "withdrawal", "death",
                                       "ill_health", "retirement"]
        assert frame["in_service"].iloc[1] == pytest.approx(1 - 0.1 - 0.001)

    def test_from_mortality_uses_the_table_for_deaths(self, table):
        service = ServiceTable.from_mortality(table, 60, start_age=30)
        assert service.min_age == 30
        assert service.rate(45, "death") == pytest.approx(table.qx(45))
        assert service.survival(30, 30) == pytest.approx(table.npx(30, 30))

    def test_invalid_tables(self, table):
        with pytest.raises(ValueError, match="end one year below"):
            ServiceTable([50, 51, 52], retirement_age=60)
        with pytest.raises(ValueError, match="consecutive"):
            ServiceTable([50, 52])
        with pytest.raises(ValueError, match="between 0 and 1"):
            ServiceTable([58, 59], withdrawal=1.5)
        with pytest.raises(ValueError, match="one rate for each"):
            ServiceTable([58, 59], withdrawal=[0.1])
        with pytest.raises(ValueError, match="covers ages"):
            ServiceTable.from_mortality(table, 60, start_age=10)
        with pytest.raises(ValueError, match="cause must be"):
            ServiceTable.no_decrements(50, 60).rate(55, "dismissal")

    def test_salary_projection(self):
        scale = SalaryScale(0.10, {30: 1.0, 40: 2.0})
        assert scale.projected(100, 30, 0) == 100
        assert scale.projected(100, 30, 2) == pytest.approx(100 * 1.21 * 1.2)
        assert scale.projected(100, 30, 20) == pytest.approx(100 * 1.1 ** 20 * 2.0)
        assert scale.projected(100, 35, -1) == pytest.approx(100 / 1.1 * 1.4 / 1.5)

    def test_final_average_salary(self):
        scale = SalaryScale(0.10)
        # Three years ending at time 2.5: half of last year, two whole years, half a year
        expected = 100 * (0.5 / 1.1 + 1 + 1.1 + 0.5 * 1.21) / 3
        assert scale.final_average(100, 40, 2.5, 3) == pytest.approx(expected)
        assert scale.final_average(100, 40, 5, 1) == pytest.approx(100 * 1.1 ** 4)
        assert scale.final_average(100, 40, 5, 0) == pytest.approx(100 * 1.1 ** 4)
        assert scale.final_average(100, 40, 2.5, 0) == pytest.approx(100 * 1.1 ** 2)
        assert scale.shifted(0.01).general_increase == pytest.approx(0.11)


class TestIAS19Example:
    """Example illustrating paragraph 68 of IAS 19, at three discount rates."""

    PUBLISHED = {
        0.10: {"closing": [90, 197, 325, 477, 655], "service": [90, 98, 108, 119, 131],
               "interest": [0, 9, 20, 32, 48]},
        0.04: {"closing": [112, 233, 364, 504, 655], "service": [112, 117, 121, 126, 131],
               "interest": [0, 4, 9, 15, 20]},
        0.07: {"closing": [100, 214, 343, 490, 655], "service": [100, 107, 114, 123, 131],
               "interest": [0, 7, 15, 24, 34]},
    }

    @pytest.mark.parametrize("rate", [0.10, 0.04, 0.07])
    def test_projected_unit_method_reproduces_the_example(self, table, rate):
        # A lump sum of 1% of final salary for each year of service, paid after
        # five years. Salary is 10,000 in year 1 and rises by 7% a year.
        scheme = DefinedBenefitScheme(accrual_rate=0.0, lump_sum_accrual=0.01,
                                      normal_retirement_age=60, final_average_years=1)
        basis = ValuationBasis(discount_rate=rate, mortality_post=table, salary_scale=0.07)

        def at_start_of_year(year):
            member = [{"age": 54 + year, "service": year - 1,
                       "salary": 10_000 * 1.07 ** (year - 1)}]
            return DBValuation(scheme, basis, actives=member)

        opening = 0.0
        closing, service, interest = [], [], []
        for year in range(1, 6):
            start, end = at_start_of_year(year), at_start_of_year(year + 1)
            period = IAS19Period(opening, 0.0, rate, start.normal_cost * (1 + rate),
                                 end.total_liability, 0.0)
            assert start.total_liability == pytest.approx(opening)
            assert period.actuarial_loss == pytest.approx(0.0, abs=1e-9)
            closing.append(round(end.total_liability))
            service.append(round(period.current_service_cost))
            interest.append(round(period.interest_cost))
            opening = end.total_liability

        published = self.PUBLISHED[rate]
        assert closing == published["closing"]
        assert service == published["service"]
        assert interest == published["interest"]


class TestActiveMembers:
    def test_closed_form_without_decrements(self, table):
        scheme = DefinedBenefitScheme(accrual_rate=1 / 50, normal_retirement_age=65,
                                      final_average_years=1, pension_frequency=1)
        basis = ValuationBasis(discount_rate=0.08, mortality_post=table, salary_scale=0.05)
        valuation = DBValuation(scheme, basis,
                                actives=[{"age": 45, "service": 10, "salary": 20_000}])
        annuity = SurvivalFunctions(table, 0.08).annuity_due(65)
        final_salary = 20_000 * 1.05 ** 19
        discount = 1.08 ** -20
        row = valuation.active_results.iloc[0]
        assert row["value_of_benefits"] == pytest.approx(30 / 50 * final_salary * annuity * discount)
        assert row["accrued_liability"] == pytest.approx(10 / 50 * final_salary * annuity * discount)
        assert row["normal_cost"] == pytest.approx(1 / 50 * final_salary * annuity * discount)
        salaries = sum(20_000 * 1.05 ** t * 1.08 ** -(t + 0.5) for t in range(20))
        assert row["value_of_salaries"] == pytest.approx(salaries)
        assert valuation.standard_contribution_rate == pytest.approx(
            row["normal_cost"] / (20_000 * 1.08 ** -0.5))

    def test_member_past_retirement_age_is_valued_as_retiring_now(self, table):
        scheme = DefinedBenefitScheme(accrual_rate=1 / 60, normal_retirement_age=60,
                                      final_average_years=0, pension_frequency=1)
        basis = ValuationBasis(discount_rate=0.08, mortality_post=table, salary_scale=0.05)
        valuation = DBValuation(scheme, basis,
                                actives=[{"age": 62, "service": 30, "salary": 21_000}])
        expected = 30 / 60 * 21_000 / 1.05 * SurvivalFunctions(table, 0.08).annuity_due(62)
        assert valuation.active_liability == pytest.approx(expected)
        assert valuation.normal_cost == 0.0

    def test_each_cause_of_leaving_in_the_last_year(self, table):
        """One year from retirement, every benefit can be written out by hand."""
        rate, growth = 0.08, 0.05
        service_table = ServiceTable([59], withdrawal=0.10, death=0.02, ill_health=0.03,
                                     retirement=0.05)
        scheme = DefinedBenefitScheme(
            accrual_rate=0.02, normal_retirement_age=60, final_average_years=0,
            pension_frequency=1, lump_sum_accrual=0.03, early_retirement_reduction=0.06,
            ill_health_service="prospective", death_in_service_multiple=3.0,
            death_in_service_spouse_fraction=0.5, spouse_fraction=0.5,
        )
        basis = ValuationBasis(
            discount_rate=rate, mortality_post=table, service_table=service_table,
            salary_scale=growth, pension_increase=0.02, deferred_revaluation=0.04,
            proportion_married=0.9, spouse_age_difference=4,
        )
        salary, service = 30_000.0, 20.0
        valuation = DBValuation(scheme, basis,
                                actives=[{"age": 59, "service": service, "salary": salary}])

        def annuity(age, spouse=True):
            options = dict(increase=0.02, frequency=1)
            if spouse:
                options.update(spouse_fraction=0.5, spouse_age=age - 4, proportion_married=0.9)
            return pension_annuity(table, age, rate, **options)

        v_half = 1.08 ** -0.5
        pay = salary                         # salary rate just before leaving in mid-year
        early = (0.02 * 20.5 * pay * (1 - 0.06 * 0.5) * annuity(59.5) + 0.03 * 20.5 * pay)
        ill = 0.02 * 21 * pay * annuity(59.5) + 0.03 * 20.5 * pay
        death = 3.0 * pay + 0.5 * 0.02 * 20.5 * pay * 0.9 * annuity(55.5, spouse=False)
        withdrawal = ((1.04 / 1.08) ** 0.5
                      * (0.02 * 20.5 * pay * annuity(60) + 0.03 * 20.5 * pay))
        normal = 0.02 * 21 * pay * annuity(60) + 0.03 * 21 * pay
        benefits = (v_half * (0.05 * early + 0.03 * ill + 0.02 * death + 0.10 * withdrawal)
                    + 0.80 * normal / 1.08)
        row = valuation.active_results.iloc[0]
        assert row["value_of_benefits"] == pytest.approx(benefits)

        mid_year = v_half * (0.05 * early + 0.03 * ill + 0.02 * death + 0.10 * withdrawal)
        end_year = 0.80 * normal / 1.08
        assert row["accrued_liability"] == pytest.approx(
            mid_year * 20 / 20.5 + end_year * 20 / 21)
        assert row["normal_cost"] == pytest.approx(mid_year * 0.5 / 20.5 + end_year * 1 / 21)
        assert row["value_of_salaries"] == pytest.approx(salary * (1 - 0.5 * 0.20) * v_half)

    def test_refund_of_contributions_on_withdrawal(self, table):
        service_table = ServiceTable([58, 59], withdrawal=[0.2, 0.0])
        scheme = DefinedBenefitScheme(accrual_rate=0.0, withdrawal_benefit="refund",
                                      member_contribution_rate=0.05)
        basis = ValuationBasis(discount_rate=0.10, mortality_post=table,
                               service_table=service_table, contribution_interest=0.04)
        valuation = DBValuation(scheme, basis, actives=[
            {"age": 58, "service": 4, "salary": 10_000, "contributions": 2_000}])
        refund = 2_000 * 1.04 ** 0.5 + 0.05 * 10_000 * 0.5 * 1.04 ** 0.25
        assert valuation.active_results["value_of_benefits"].iloc[0] == pytest.approx(
            0.2 * refund * 1.10 ** -0.5)

        nothing = DBValuation(
            DefinedBenefitScheme(accrual_rate=0.0, withdrawal_benefit="none"), basis,
            actives=[{"age": 58, "service": 4, "salary": 10_000}])
        assert nothing.active_results["value_of_benefits"].iloc[0] == 0.0

    def test_commutation_and_service_cap(self, table):
        basis = ValuationBasis(discount_rate=0.08, mortality_post=table, salary_scale=0.04)
        member = [{"age": 50, "service": 38, "salary": 50_000}]
        plain = DefinedBenefitScheme(accrual_rate=1 / 60, pension_frequency=1)
        capped = DefinedBenefitScheme(accrual_rate=1 / 60, pension_frequency=1, max_service=40)
        full = DBValuation(plain, basis, actives=member).active_results.iloc[0]
        limited = DBValuation(capped, basis, actives=member).active_results.iloc[0]
        assert limited["value_of_benefits"] == pytest.approx(full["value_of_benefits"] * 40 / 48)
        # Capped service: 38 of the 40 countable years are earned, and one more accrues
        assert limited["accrued_liability"] == pytest.approx(
            limited["value_of_benefits"] * 38 / 40)
        assert limited["normal_cost"] == pytest.approx(limited["value_of_benefits"] / 40)

        annuity = SurvivalFunctions(table, 0.08).annuity_due(60)
        cash = DefinedBenefitScheme(accrual_rate=1 / 60, pension_frequency=1,
                                    commutation_fraction=0.25, commutation_factor=9.0)
        commuted = DBValuation(cash, basis, actives=member)
        assert commuted.retirement_factor(60) == pytest.approx(0.75 * annuity + 0.25 * 9.0)
        neutral = DefinedBenefitScheme(accrual_rate=1 / 60, pension_frequency=1,
                                       commutation_fraction=0.25)
        assert DBValuation(neutral, basis, actives=member).retirement_factor(60) == \
            pytest.approx(annuity)

    @pytest.mark.parametrize("method", ["projected_unit", "entry_age"])
    def test_liability_moves_forward_as_expected(self, scheme, basis, method):
        """
        If experience follows the assumptions, the liability and normal cost
        pay for the benefits of those who leave in the year and for the
        liability of those who stay: both sides are the cost of a year.
        """
        discount = 1 / (1 + basis.discount_rate)
        for age, service, salary in [(30, 4, 12_000), (47, 20, 30_000), (58, 9, 18_000)]:
            now = DBValuation(scheme, basis, method=method, actives=[
                {"age": age, "service": service, "salary": salary}]).active_results.iloc[0]
            later = DBValuation(scheme, basis, method=method, actives=[
                {"age": age + 1, "service": service + 1,
                 "salary": basis.salary_scale.projected(salary, age, 1)},
            ]).active_results.iloc[0]
            staying = discount * basis.service_table.survival(age, 1)
            paid_to_leavers = now["value_of_benefits"] - staying * later["value_of_benefits"]
            assert (now["accrued_liability"] + now["normal_cost"]
                    - staying * later["accrued_liability"]) == pytest.approx(paid_to_leavers)

    def test_funding_methods(self, scheme, basis):
        entrant = [{"age": 30, "service": 0, "salary": 12_000}]
        member = [{"age": 45, "service": 15, "salary": 25_000}]
        results = {m: DBValuation(scheme, basis, method=m, actives=member).active_results.iloc[0]
                   for m in METHODS}
        # Attained age shares the projected unit liability
        assert results["attained_age"]["accrued_liability"] == pytest.approx(
            results["projected_unit"]["accrued_liability"])
        # Holding salaries level gives a lower liability
        assert results["current_unit"]["accrued_liability"] < \
            results["projected_unit"]["accrued_liability"]
        # Entry age funds faster than projected unit for a member in mid-career
        assert results["entry_age"]["accrued_liability"] > \
            results["projected_unit"]["accrued_liability"]
        for method in METHODS:
            assert results[method]["value_of_benefits"] == pytest.approx(
                results["projected_unit"]["value_of_benefits"])

        at_entry = {m: DBValuation(scheme, basis, method=m, actives=entrant) for m in METHODS}
        for method in METHODS:
            assert at_entry[method].active_liability == pytest.approx(0.0, abs=1e-9)
        level = at_entry["entry_age"].active_results.iloc[0]
        assert at_entry["entry_age"].standard_contribution_rate == pytest.approx(
            level["value_of_benefits"] / level["value_of_salaries"])
        assert at_entry["attained_age"].standard_contribution_rate == pytest.approx(
            at_entry["entry_age"].standard_contribution_rate)

    def test_aggregate_rate_equals_entry_age_rate_when_assets_match(self, scheme, basis):
        member = [{"age": 45, "service": 15, "salary": 25_000}]
        entry_age = DBValuation(scheme, basis, method="entry_age", actives=member)
        funded = entry_age.revalue(assets=entry_age.active_liability)
        assert funded.aggregate_contribution_rate() == pytest.approx(
            funded.standard_contribution_rate)

    def test_tables_by_sex(self, scheme, table, service_table):
        lighter = MortalityTable(table.ages, np.append(table.qx_values[:-1] * 0.7, 1.0))
        basis = ValuationBasis(discount_rate=0.09, salary_scale=0.06,
                               mortality_post={"M": table, "F": lighter},
                               service_table={"M": service_table, "F": service_table},
                               spouse_age_difference=3)
        assert basis.spouse_table("M") is lighter and basis.spouse_table("F") is table
        assert basis.spouse_age(60, "M") == 57 and basis.spouse_age(60, "F") == 63
        both = DBValuation(scheme, basis, actives=[
            {"age": 40, "service": 10, "salary": 20_000, "sex": "male"},
            {"age": 40, "service": 10, "salary": 20_000, "sex": "F"},
        ]).active_results
        assert list(both["sex"]) == ["M", "F"]
        male, female = both["accrued_liability"]
        assert male != pytest.approx(female)
        # The woman's own pension costs more; her widower's pension costs less
        own = pension_annuity(lighter, 60, 0.09) > pension_annuity(table, 60, 0.09)
        assert own

    def test_invalid_data(self, scheme, basis, table):
        with pytest.raises(ValueError, match="needs the columns"):
            DBValuation(scheme, basis, actives=[{"age": 40, "salary": 1}])
        with pytest.raises(ValueError, match="whole numbers"):
            DBValuation(scheme, basis, actives=[{"age": 40.5, "service": 1, "salary": 1}])
        with pytest.raises(ValueError, match="non-negative"):
            DBValuation(scheme, basis, actives=[{"age": 40, "service": -1, "salary": 1}])
        with pytest.raises(ValueError, match="sex must be"):
            DBValuation(scheme, basis, actives=[{"age": 40, "service": 1, "salary": 1,
                                                 "sex": "X"}])
        with pytest.raises(ValueError, match="method must be"):
            DBValuation(scheme, basis, method="unit")
        with pytest.raises(ValueError, match="normal retirement age"):
            DBValuation(DefinedBenefitScheme(normal_retirement_age=65), basis)
        with pytest.raises(ValueError, match="service table starts"):
            DBValuation(scheme, basis,
                        actives=[{"age": 18, "service": 0, "salary": 1}]).active_results
        with pytest.raises(ValueError, match="no table was supplied"):
            DBValuation(scheme, ValuationBasis(0.08, {"M": table}),
                        pensioners=[{"age": 70, "pension": 1, "sex": "F"}]).pensioner_results
        with pytest.raises(ValueError, match="withdrawal_benefit"):
            DefinedBenefitScheme(withdrawal_benefit="cash")
        with pytest.raises(ValueError, match="proportion_married"):
            ValuationBasis(0.08, table, proportion_married=1.2)


class TestValidation:
    def test_scheme_rules(self):
        for options in [{"accrual_rate": -0.01}, {"final_average_years": -1},
                        {"max_service": 0}, {"commutation_fraction": 1.5},
                        {"spouse_fraction": -0.5}, {"guarantee_years": -1},
                        {"pension_timing": "weekly"}, {"ill_health_service": "full"},
                        {"member_contribution_rate": 1.0}]:
            with pytest.raises(ValueError):
                DefinedBenefitScheme(**options)

    def test_basis(self, table):
        with pytest.raises(ValueError, match="discount_rate"):
            ValuationBasis(-1.0, table)
        with pytest.raises(ValueError, match="post_retirement_discount_rate"):
            ValuationBasis(0.05, table, post_retirement_discount_rate=-2)
        with pytest.raises(ValueError, match="general_increase"):
            SalaryScale(-1.5)
        with pytest.raises(ValueError, match="positive values"):
            SalaryScale(0.05, {30: 0.0})
        with pytest.raises(ValueError, match="years must be"):
            SalaryScale(0.05).final_average(100, 40, 5, -1)

    def test_annuity_and_projection_arguments(self, table):
        for options in [{"interest": -1}, {"increase": -1}, {"guarantee": -1},
                        {"spouse_fraction": -1}, {"proportion_married": 2}]:
            arguments = {"interest": 0.05, **options}
            with pytest.raises(ValueError):
                pension_annuity(table, 65, **arguments)
        for options in [{"salary": -1}, {"member_rate": -0.1}, {"inflation": -1},
                        {"contributions_per_year": 0}]:
            arguments = {"age": 30, "retirement_age": 60, "salary": 1, **options}
            with pytest.raises(ValueError):
                DCProjection(**arguments)
        with pytest.raises(ValueError, match="whole numbers"):
            DCProjection(30.5, 60, 1)
        fund = DCFundValuation(100, 0)
        with pytest.raises(ValueError, match="no member accounts"):
            fund.bonus_rate()
        with pytest.raises(ValueError, match="target_funding_level"):
            DCFundValuation(100, 90).bonus_rate(target_funding_level=0)
        with pytest.raises(ValueError, match="years must be positive"):
            analysis_of_surplus(1, 1, 1, 1, 0.05, 0, years=0)
        with pytest.raises(ValueError, match="funding_level"):
            zimbabwe.funding_level_check(-0.1)
        with pytest.raises(ValueError, match="must be positive"):
            zimbabwe.surplus_share_on_exit(0, 0)


class TestPensionersAndDeferreds:
    def test_pensioners(self, table):
        scheme = DefinedBenefitScheme(spouse_fraction=0.5, pension_frequency=1)
        basis = ValuationBasis(discount_rate=0.07, mortality_post=table, pension_increase=0.02,
                               proportion_married=0.8, spouse_age_difference=5)
        pensioners = pd.DataFrame({
            "age": [70, 70, 70, 70],
            "pension": [1_000.0, 1_000.0, 1_000.0, 1_000.0],
            "reversion": [False, True, True, True],
            "spouse_age": [np.nan, np.nan, 60, np.nan],
            "guarantee_remaining": [0, 0, 0, 3],
        })
        results = DBValuation(scheme, basis, pensioners=pensioners).pensioner_results
        options = dict(increase=0.02, frequency=1)
        assert results["annuity_factor"].iloc[0] == pytest.approx(
            pension_annuity(table, 70, 0.07, **options))
        assert results["annuity_factor"].iloc[1] == pytest.approx(pension_annuity(
            table, 70, 0.07, spouse_fraction=0.5, spouse_age=65, proportion_married=0.8,
            **options))
        # A known spouse is valued at the stated age, with certainty
        assert results["annuity_factor"].iloc[2] == pytest.approx(pension_annuity(
            table, 70, 0.07, spouse_fraction=0.5, spouse_age=60, **options))
        assert results["annuity_factor"].iloc[3] == pytest.approx(pension_annuity(
            table, 70, 0.07, spouse_fraction=0.5, spouse_age=65, proportion_married=0.8,
            guarantee=3, **options))
        assert results["liability"].sum() == pytest.approx(1_000 * results["annuity_factor"].sum())

    def test_ill_health_pensioners_use_their_own_table(self, table):
        heavier = MortalityTable(table.ages, np.minimum(table.qx_values * 2, 1.0))
        basis = ValuationBasis(discount_rate=0.07, mortality_post=table,
                               mortality_ill_health=heavier)
        results = DBValuation(DefinedBenefitScheme(), basis, pensioners=pd.DataFrame({
            "age": [55, 55], "pension": [100.0, 100.0], "ill_health": [False, True],
        })).pensioner_results
        assert results["liability"].iloc[1] < results["liability"].iloc[0]
        assert results["annuity_factor"].iloc[1] == pytest.approx(
            pension_annuity(heavier, 55, 0.07))

    def test_deferred_pensioners(self, table):
        scheme = DefinedBenefitScheme(normal_retirement_age=60, pension_frequency=1)
        basis = ValuationBasis(discount_rate=0.08, mortality_post=table,
                               deferred_revaluation=0.03)
        deferreds = [{"age": 40, "pension": 500}, {"age": 64, "pension": 500}]
        annuity = SurvivalFunctions(table, 0.08)
        results = DBValuation(scheme, basis, deferreds=deferreds).deferred_results
        assert results["liability"].iloc[0] == pytest.approx(
            500 * (1.03 / 1.08) ** 20 * annuity.annuity_due(60))
        assert results["liability"].iloc[1] == pytest.approx(500 * annuity.annuity_due(64))

        with_deaths = ValuationBasis(discount_rate=0.08, mortality_post=table,
                                     deferred_revaluation=0.03, mortality_deferment=table)
        lower = DBValuation(scheme, with_deaths, deferreds=deferreds).deferred_results
        assert lower["liability"].iloc[0] == pytest.approx(
            results["liability"].iloc[0] * table.npx(40, 20))


class TestValuationResults:
    def test_summary_adds_up(self, valuation):
        summary = valuation.summary()
        parts = summary[["Active members", "Deferred pensioners", "Pensioners", "Reserves"]]
        assert summary["Total actuarial liabilities"] == pytest.approx(parts.sum())
        assert summary["Surplus / (deficit)"] == pytest.approx(
            260_000 - summary["Total actuarial liabilities"])
        assert summary["Funding level"] == pytest.approx(
            260_000 / summary["Total actuarial liabilities"])
        assert summary["Reserves"] == 5_000
        assert valuation.payroll == 142_000
        assert all(valuation.active_results["accrued_liability"] > 0)
        assert "funding_level" in repr(valuation)

    def test_contribution_rate(self, valuation):
        assert valuation.surplus < 0
        plain = valuation.contribution_rate()
        assert plain["Deficit / (surplus) adjustment"] == 0.0
        assert plain["Total contribution rate"] == pytest.approx(
            valuation.standard_contribution_rate)

        rates = valuation.contribution_rate(recovery_period=10, expenses=0.01)
        ratio = 1.06 / 1.09
        annuity = sum(ratio ** (t + 0.5) for t in range(10))
        assert rates["Deficit / (surplus) adjustment"] == pytest.approx(
            -valuation.surplus / (valuation.payroll * annuity))
        assert rates["Total contribution rate"] == pytest.approx(
            rates["Standard contribution rate"] + rates["Deficit / (surplus) adjustment"] + 0.01)
        assert rates["Employer"] == pytest.approx(rates["Total contribution rate"])

        in_surplus = valuation.revalue(assets=2 * valuation.total_liability)
        assert in_surplus.contribution_rate(recovery_period=10)[
            "Deficit / (surplus) adjustment"] == 0.0
        assert in_surplus.contribution_rate(recovery_period=10, spread_surplus=True)[
            "Deficit / (surplus) adjustment"] < 0
        with pytest.raises(ValueError, match="recovery_period"):
            valuation.contribution_rate(recovery_period=2.5)

    def test_members_share_of_contributions(self, basis, members):
        scheme = DefinedBenefitScheme(member_contribution_rate=0.06)
        rates = DBValuation(scheme, basis, members[0]).contribution_rate()
        assert rates["Members"] == 0.06
        assert rates["Employer"] == pytest.approx(rates["Total contribution rate"] - 0.06)

    def test_aggregate_method(self, valuation):
        results = valuation.active_results
        expected = (results["value_of_benefits"].sum() + valuation.deferred_liability
                    + valuation.pensioner_liability + valuation.reserves
                    - valuation.assets) / results["value_of_salaries"].sum()
        assert valuation.aggregate_contribution_rate() == pytest.approx(expected)

    def test_method_comparison(self, valuation):
        comparison = valuation.method_comparison()
        assert list(comparison.index) == list(METHODS)
        assert comparison.loc["projected_unit", "Total liabilities"] == pytest.approx(
            valuation.total_liability)
        assert comparison.loc["current_unit", "Funding level"] > \
            comparison.loc["projected_unit", "Funding level"]

    def test_sensitivities_and_duration(self, valuation):
        table = valuation.sensitivities()
        base = table.loc["Valuation basis", "Total liabilities"]
        assert base == pytest.approx(valuation.total_liability)
        assert table.loc["Discount rate +1%", "Change"] < 0 < table.loc["Discount rate -1%", "Change"]
        assert table.loc["Salary increases +1%", "Change"] > 0
        assert table.loc["Pension increases +1%", "Change"] > 0
        assert table.loc["Pensioner mortality -10%", "Change"] > 0
        assert table.loc["Discount rate +1%", "Funding level"] > valuation.funding_level

        duration = valuation.duration()
        assert 5 < duration < 30
        # Duration predicts the effect of a 1% change to first order
        predicted = -duration * 0.01
        assert table.loc["Discount rate +1%", "Change %"] == pytest.approx(predicted, abs=0.02)

        custom = valuation.sensitivities({"Revaluation +1%": {"deferred_revaluation": 0.01}})
        assert custom.loc["Revaluation +1%", "Change"] > 0

    def test_shifted_basis(self, basis):
        shifted = basis.shifted(discount_rate=0.01, salary_increase=-0.01, pension_increase=0.005,
                                mortality_factor=0.9)
        assert shifted.discount_rate == pytest.approx(0.10)
        assert shifted.salary_scale.general_increase == pytest.approx(0.05)
        assert shifted.pension_increase == pytest.approx(0.035)
        assert shifted.mortality_post.qx(70) == pytest.approx(0.9 * basis.mortality_post.qx(70))
        assert shifted.mortality_post.is_closed
        split = ValuationBasis(0.09, basis.mortality_post, post_retirement_discount_rate=0.06)
        assert split.post_rate == 0.06
        assert split.shifted(discount_rate=0.01).post_rate == pytest.approx(0.07)

    def test_empty_scheme(self, scheme, basis):
        empty = DBValuation(scheme, basis, assets=100)
        assert empty.total_liability == 0.0
        assert np.isnan(empty.funding_level)
        assert empty.standard_contribution_rate == 0.0


class TestIAS19Period:
    @pytest.fixture
    def period(self):
        return IAS19Period(
            opening_obligation=1_000, opening_assets=800, discount_rate=0.10,
            current_service_cost=90, closing_obligation=1_200, closing_assets=930,
            benefits_paid=60, employer_contributions=100, member_contributions=20,
            past_service_cost=15, administration_expenses=5, cash_flow_timing=0.5,
            closing_obligation_before_assumption_changes=1_170,
        )

    def test_hand_calculation(self, period):
        assert period.interest_cost == pytest.approx(0.10 * (1_000 - 0.5 * 60))
        assert period.interest_income == pytest.approx(0.10 * (800 + 0.5 * (100 + 20 - 60 - 5)))
        assert period.net_interest == pytest.approx(97.0 - 82.75)
        expected_obligation = 1_000 + 97 + 90 + 15 + 20 - 60
        assert period.expected_closing_obligation == pytest.approx(expected_obligation)
        assert period.actuarial_loss == pytest.approx(1_200 - expected_obligation)
        assert period.experience_loss == pytest.approx(1_170 - expected_obligation)
        assert period.assumption_loss == pytest.approx(30)
        expected_assets = 800 + 82.75 + 100 + 20 - 60 - 5
        assert period.return_on_assets_excluding_interest == pytest.approx(930 - expected_assets)

        profit_or_loss = period.profit_or_loss()
        assert profit_or_loss["Defined benefit cost in profit or loss"] == pytest.approx(
            90 + 15 + 5 + 14.25)
        oci = period.other_comprehensive_income()
        assert oci.iloc[-1] == pytest.approx(38 - (930 - expected_assets))

    def test_reconciliations(self, period):
        obligation = period.obligation_reconciliation()
        assert obligation.iloc[:-1].sum() == pytest.approx(obligation["Closing obligation"])
        assets = period.asset_reconciliation()
        assert assets.iloc[:-1].sum() == pytest.approx(assets["Closing plan assets"])
        net = period.net_liability_reconciliation()
        assert net.iloc[:-1].sum() == pytest.approx(net["Closing net liability / (asset)"])
        assert net["Closing net liability / (asset)"] == pytest.approx(270)
        position = period.financial_position()
        assert position["Net defined benefit liability / (asset)"] == pytest.approx(270)
        assert "closing_net_liability" in repr(period)

    def test_cash_flows_at_the_end_carry_no_interest(self):
        period = IAS19Period(1_000, 800, 0.10, 0, 1_040, 880, benefits_paid=60,
                             cash_flow_timing=1.0)
        assert period.interest_cost == pytest.approx(100)
        assert period.interest_income == pytest.approx(80)
        assert period.experience_loss == period.actuarial_loss
        assert period.assumption_loss == 0.0

    def test_asset_ceiling(self):
        period = IAS19Period(
            opening_obligation=900, opening_assets=1_000, discount_rate=0.10,
            current_service_cost=50, closing_obligation=1_000, closing_assets=1_180,
            employer_contributions=40, cash_flow_timing=1.0,
            opening_asset_ceiling_effect=20, asset_ceiling=100,
        )
        assert period.opening_net_liability == pytest.approx(-80)
        assert period.closing_asset_ceiling_effect == pytest.approx(80)
        assert period.closing_net_liability == pytest.approx(-100)
        assert period.interest_on_asset_ceiling == pytest.approx(2)
        assert period.change_in_asset_ceiling == pytest.approx(80 - 20 - 2)
        assert period.net_interest == pytest.approx(90 - 100 + 2)
        net = period.net_liability_reconciliation()
        assert net.iloc[:-1].sum() == pytest.approx(-100)

        unlimited = IAS19Period(900, 1_000, 0.10, 50, 1_000, 1_180)
        assert unlimited.closing_asset_ceiling_effect == 0.0
        assert unlimited.closing_net_liability == pytest.approx(-180)

    def test_from_valuations(self, scheme, basis, members):
        actives = members[0]
        opening = DBValuation(scheme, basis, actives, assets=150_000)
        closing = DBValuation(scheme, basis.shifted(discount_rate=-0.005), actives,
                              assets=170_000)
        period = IAS19Period.from_valuations(opening, closing, employer_contributions=12_000,
                                             member_contributions=3_000)
        assert period.opening_obligation == pytest.approx(opening.total_liability)
        assert period.current_service_cost == pytest.approx(opening.normal_cost * 1.09 - 3_000)
        assert period.closing_net_liability == pytest.approx(closing.total_liability - 170_000)
        with pytest.raises(ValueError, match="projected unit"):
            IAS19Period.from_valuations(opening.revalue(method="entry_age"), closing)

    def test_invalid_arguments(self):
        with pytest.raises(ValueError, match="cash_flow_timing"):
            IAS19Period(1, 1, 0.1, 0, 1, 1, cash_flow_timing=2)
        with pytest.raises(ValueError, match="asset_ceiling"):
            IAS19Period(1, 1, 0.1, 0, 1, 1, asset_ceiling=-1)


class TestDefinedContribution:
    def test_fund_without_growth_is_the_sum_of_contributions(self):
        projection = DCProjection(age=40, retirement_age=45, salary=12_000, fund=5_000,
                                  member_rate=0.05, employer_rate=0.10)
        assert projection.fund_at_retirement == pytest.approx(5_000 + 5 * 0.15 * 12_000)
        assert projection.final_salary == 12_000

    def test_closed_form_accumulation(self):
        projection = DCProjection(age=30, retirement_age=60, salary=10_000, fund=20_000,
                                  member_rate=0.07, employer_rate=0.08, salary_increase=0.05,
                                  investment_return=0.09, inflation=0.04,
                                  contributions_per_year=12)
        monthly = 1.09 ** (1 / 12)
        within_year = (1.09 - 1) / (12 * (monthly - 1))
        expected = 20_000 * 1.09 ** 30 + sum(
            0.15 * 10_000 * 1.05 ** t * within_year * 1.09 ** (29 - t) for t in range(30))
        assert projection.fund_at_retirement == pytest.approx(expected)
        assert projection.real_fund_at_retirement == pytest.approx(expected / 1.04 ** 30)
        assert projection.final_salary == pytest.approx(10_000 * 1.05 ** 29)
        table = projection.projection()
        assert len(table) == 30 and table["age"].iloc[-1] == 59
        assert (table["fund_start"] + table["contributions"] + table["growth"]).iloc[7] == \
            pytest.approx(table["fund_end"].iloc[7])

    def test_charges(self):
        gross = DCProjection(30, 60, 10_000, fund=1_000, member_rate=0.1, investment_return=0.08)
        charged = DCProjection(30, 60, 10_000, fund=1_000, member_rate=0.1,
                               investment_return=0.08, contribution_charge=0.05,
                               asset_charge=0.01)
        assert charged.net_return == pytest.approx(1.08 * 0.99 - 1)
        assert charged.fund_at_retirement < gross.fund_at_retirement
        only_contribution_charge = DCProjection(30, 60, 10_000, member_rate=0.1,
                                                investment_return=0.08, contribution_charge=0.05)
        no_charge = DCProjection(30, 60, 10_000, member_rate=0.1, investment_return=0.08)
        assert only_contribution_charge.fund_at_retirement == pytest.approx(
            0.95 * no_charge.fund_at_retirement)

    def test_pension_and_required_contribution(self, table):
        projection = DCProjection(35, 60, 15_000, fund=8_000, member_rate=0.05,
                                  employer_rate=0.075, salary_increase=0.04,
                                  investment_return=0.07)
        factor = projection.annuity_factor(table, 0.05, increase=0.02)
        assert factor == pytest.approx(pension_annuity(table, 60, 0.05, increase=0.02))
        assert projection.pension(factor) == pytest.approx(projection.fund_at_retirement / factor)
        ratio = projection.replacement_ratio(factor)
        assert projection.required_contribution_rate(ratio, factor) == pytest.approx(0.125)
        assert projection.required_contribution_rate(0.6, factor) > 0.125

    def test_invalid_projection(self):
        with pytest.raises(ValueError, match="retirement_age"):
            DCProjection(60, 60, 1)
        with pytest.raises(ValueError, match="charges"):
            DCProjection(30, 60, 1, asset_charge=1.0)
        with pytest.raises(ValueError, match="annuity_factor"):
            DCProjection(30, 60, 1).pension(0)

    def test_fund_valuation(self):
        fund = DCFundValuation(assets=1_100, member_accounts=[400, 350, 150],
                               pensioner_liability=50, investment_reserve=40,
                               contingency_reserve=10)
        assert fund.total_liabilities == 1_000
        assert fund.surplus == 100
        assert fund.funding_level == pytest.approx(1.10)
        summary = fund.summary()
        assert summary["Reserves"] == 50 and summary["Active members"] == 900
        # All of the surplus as bonus, then half of it, then keeping a 5% margin
        assert fund.bonus_rate() == pytest.approx(100 / 900)
        assert fund.bonus_rate(share_to_reserves=0.5) == pytest.approx(50 / 900)
        margin = fund.bonus_rate(target_funding_level=1.05)
        assert 1_100 / (1_000 + margin * 900) == pytest.approx(1.05)

        short = DCFundValuation(assets=950, member_accounts=1_000)
        assert short.funding_level == pytest.approx(0.95)
        assert short.bonus_rate(share_to_reserves=0.5) == pytest.approx(-0.05)
        with pytest.raises(ValueError, match="non-negative"):
            DCFundValuation(100, -1)


class TestAnalysisOfSurplus:
    def test_items_add_up(self):
        analysis = analysis_of_surplus(
            opening_assets=900, opening_liabilities=1_000, closing_assets=1_080,
            closing_liabilities=1_150, discount_rate=0.08, normal_cost=70,
            contributions=110, benefits_paid=50, expenses=5,
            closing_liabilities_on_opening_basis=1_120,
        )
        assert analysis["Surplus at start"] == -100
        assert analysis["Surplus at end"] == -70
        assert analysis.iloc[1:-1].sum() == pytest.approx(30)
        assert analysis["Interest on the surplus"] == pytest.approx(-8)
        assert analysis["Change of assumptions"] == pytest.approx(-30)
        half = 1.08 ** 0.5
        assert analysis["Investment return above the discount rate"] == pytest.approx(
            1_080 - (900 * 1.08 + 55 * half))
        assert analysis["Liability experience"] == pytest.approx(
            1_070 * 1.08 - 50 * half - 1_120)

    def test_no_change_when_experience_follows_the_assumptions(self, scheme, table):
        basis = ValuationBasis(discount_rate=0.09, mortality_post=table, salary_scale=0.06,
                               pension_increase=0.03)
        opening = DBValuation(scheme, basis, actives=[
            {"age": 40, "service": 10, "salary": 20_000}], assets=30_000)
        cost = opening.normal_cost
        contributions = cost * 1.09 ** 0.5
        closing = DBValuation(scheme, basis, actives=[
            {"age": 41, "service": 11, "salary": 21_200}],
            assets=30_000 * 1.09 + contributions * 1.09 ** 0.5)
        analysis = analysis_of_surplus(
            opening.assets, opening.total_liability, closing.assets, closing.total_liability,
            0.09, cost, contributions=contributions)
        assert analysis["Investment return above the discount rate"] == pytest.approx(0, abs=1e-6)
        assert analysis["Contributions less normal cost and expenses"] == pytest.approx(0, abs=1e-6)
        assert analysis["Liability experience"] == pytest.approx(0, abs=1e-6)
        assert analysis["Surplus at end"] == pytest.approx(analysis["Surplus at start"] * 1.09)


class TestZimbabweAct:
    def test_funding_level_check(self):
        assert zimbabwe.MINIMUM_FUNDING_LEVEL == 0.75
        assert zimbabwe.funding_level_check(1.02)["fully_funded"]
        middle = zimbabwe.funding_level_check(0.80)
        assert not middle["fully_funded"] and middle["meets_minimum"]
        low = zimbabwe.funding_level_check(0.60)
        assert not low["meets_minimum"] and "Below" in low["status"]

    def test_dissolution_with_enough_assets(self):
        result = zimbabwe.dissolution_reserves(1_000, [300, 200], annuity_cost=400)
        assert result.attrs["ratio"] == 1.0
        assert list(result["payable"]) == [300, 200]

    def test_dissolution_with_a_shortfall(self):
        # Reserves of 500 after an earlier payment of 100, plus annuities of
        # 500, against assets of 800: everything is paid at 80%
        reserves = pd.Series([400, 200], index=["A", "B"])
        result = zimbabwe.dissolution_reserves(800, reserves,
                                               benefits_previously_paid=[100, 0],
                                               annuity_cost=500)
        assert result.attrs["ratio"] == pytest.approx(0.8)
        assert list(result.index) == ["A", "B"]
        assert list(result["payable"]) == pytest.approx([240, 160])
        assert result.attrs["annuity_cost_covered"] == pytest.approx(400)
        assert result["payable"].sum() + result.attrs["annuity_cost_covered"] == pytest.approx(800)

    def test_surplus_share_on_exit(self):
        share = zimbabwe.surplus_share_on_exit(
            leaver_past_service_liability=50, total_past_service_liability=1_000,
            actuarial_surplus=200, investment_reserve=60, contingency_reserves=40)
        assert share == pytest.approx(15)
        with pytest.raises(ValueError, match="between zero and the total"):
            zimbabwe.surplus_share_on_exit(2_000, 1_000)
        with pytest.raises(ValueError, match="non-negative"):
            zimbabwe.surplus_share_on_exit(50, 1_000, actuarial_surplus=-10)

    def test_reference_data(self):
        assert zimbabwe.PENSION_INCREASE_INTERVAL_YEARS == {
            "defined_benefit": 3, "defined_contribution": 1}
        assert len(zimbabwe.SURPLUS_USES) == 4
