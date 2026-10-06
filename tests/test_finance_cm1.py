"""
Tests for the finance module against worked answers.

The expected figures are the published answers to standard compound interest
questions (time value of money, interest rates, annuities, equations of
value, loan schedules, project appraisal, bonds and equities, and the term
structure of interest rates). Tolerances reflect the rounding of the
published answers.
"""

import numpy as np
import pytest
from actuneo.finance import (
    InterestRate, ForceOfInterest, simple_accumulation, simple_discount_value,
    real_rate, money_rate, inflation_from_index,
    annuity, accumulated_annuity, perpetuity, increasing_annuity,
    continuously_increasing_annuity, decreasing_annuity, accumulated_increasing_annuity,
    geometric_annuity, Cashflows, present_value, accumulated_value, internal_rate_of_return,
    Loan, loan_schedule, annual_percentage_rate, flat_rate,
    Bond, bond_price_with_optional_redemption, equity_price, equity_yield,
    present_value_of_dividends, real_yield,
    forward_rate, spot_rates_from_forwards, forwards_from_spot_rates, par_yield,
    spot_rates_from_bonds, discounted_mean_term, volatility, convexity,
    immunisation_check, immunising_amounts,
)


class TestTimeValueOfMoney:
    """Simple and compound interest and discount."""

    def test_simple_interest_and_discount(self):
        assert simple_accumulation(100, 0.04, 5) == pytest.approx(120)
        # Bill of 500,000 due in 6 months at 16% commercial discount
        assert simple_discount_value(500_000, 0.16, 0.5) == pytest.approx(460_000)

    def test_equivalent_rates_over_five_years(self):
        rate = InterestRate(0.10)
        assert rate.simple_equivalent(5) == pytest.approx(0.122, abs=5e-4)
        assert rate.per_period(1 / 12) == pytest.approx(0.00797, abs=5e-6)
        assert rate.per_period(2) == pytest.approx(0.21)
        assert rate.d == pytest.approx(0.0909, abs=5e-5)
        assert rate.simple_discount_equivalent(5) == pytest.approx(0.0758, abs=5e-5)

    def test_round_trips(self):
        assert InterestRate.from_simple(0.122102, 5).i == pytest.approx(0.10, abs=1e-6)
        assert InterestRate.from_period(0.21, 2).i == pytest.approx(0.10)
        assert InterestRate.from_discount(0.10 / 1.10).i == pytest.approx(0.10)
        assert InterestRate.from_simple_discount(0.16, 0.5).discount(500_000, 0.5) == \
            pytest.approx(460_000)

    def test_yield_on_single_and_double_investments(self):
        assert internal_rate_of_return([-200, 350], [0, 1.5]) == pytest.approx(0.4522, abs=5e-5)
        assert internal_rate_of_return([-100, -100, 350], [0, 1, 2]) == \
            pytest.approx(0.4365, abs=5e-5)

    def test_treasury_bill(self):
        """Bought at 96.50 earning 4% effective, sold at 98, redeemed at 100 after 182 days."""
        held = np.log(98 / 96.5) / np.log(1.04) * 365
        assert held == pytest.approx(143.5, abs=0.1)
        remaining = 38 / 365
        assert (100 / 98 - 1) / remaining == pytest.approx(0.1960, abs=5e-5)
        assert InterestRate.from_period(100 / 98 - 1, remaining).i == \
            pytest.approx(0.2142, abs=5e-5)

    def test_invalid(self):
        with pytest.raises(ValueError):
            InterestRate(-1.5)
        with pytest.raises(ValueError):
            InterestRate.from_discount(1.2)


class TestNominalRatesAndForce:
    """Nominal rates and the force of interest."""

    def test_nominal_rates_at_seven_per_cent(self):
        rate = InterestRate(0.07)
        assert rate.nominal(6) == pytest.approx(0.068042, abs=5e-7)
        assert rate.nominal_discount(6) == pytest.approx(0.067279, abs=5e-7)
        assert rate.nominal(4) == pytest.approx(0.068234, abs=5e-7)
        assert rate.nominal_discount(2) == pytest.approx(0.066527, abs=5e-7)

    def test_conversions(self):
        assert InterestRate.from_nominal(0.04, 12).d == pytest.approx(0.039147, abs=5e-7)
        assert InterestRate.from_nominal(0.11, 2).i == pytest.approx(0.113025, abs=5e-7)
        assert InterestRate.from_nominal(0.12, 12).i == pytest.approx(0.126825, abs=5e-7)
        assert InterestRate(0.142).nominal(12) == pytest.approx(0.133518, abs=5e-7)
        assert InterestRate.from_nominal(0.11, 3).nominal(12) == pytest.approx(0.108519, abs=5e-7)

    def test_ordering_of_equivalent_rates(self):
        """d < d(p) < delta < i(p) < i"""
        r = InterestRate(0.08)
        assert r.d < r.nominal_discount(12) < r.delta < r.nominal(12) < r.i
        summary = r.summary()
        assert summary["i(12)"] == pytest.approx(r.nominal(12))
        assert InterestRate.from_force(r.delta).i == pytest.approx(0.08)
        assert InterestRate.from_nominal_discount(r.nominal_discount(4), 4).i == \
            pytest.approx(0.08)

    def test_accumulations(self):
        assert InterestRate.from_nominal(0.15, 4).accumulate(300, 7) == pytest.approx(840.98, abs=0.01)
        first = InterestRate.from_nominal_discount(0.18, 12).accumulate(250, 0.25)
        assert first == pytest.approx(261.60, abs=0.01)
        assert InterestRate.from_nominal(0.20, 4).accumulate(first, 0.75) == \
            pytest.approx(302.83, abs=0.01)
        force = InterestRate.from_force(0.09)
        assert force.accumulate(6.34, 0.25) == pytest.approx(6.48, abs=0.005)
        assert force.accumulate(6.34, 3) == pytest.approx(8.31, abs=0.005)
        assert force.accumulate(6.34, 7 + 5 / 365) == pytest.approx(11.92, abs=0.005)

    def test_six_month_accumulation_three_ways(self):
        assert InterestRate.from_force(0.05).accumulate(100, 0.5) == pytest.approx(102.53, abs=0.005)
        assert InterestRate.from_nominal(0.05, 12).accumulate(100, 0.5) == \
            pytest.approx(102.53, abs=0.005)
        assert InterestRate(0.05).accumulate(100, 0.5) == pytest.approx(102.47, abs=0.005)

    def test_varying_force_of_interest(self):
        """delta = 0.04 to time 6, then 0.2 - 0.02t."""
        force = ForceOfInterest([(6, [0.04]), (None, [0.2, -0.02])])
        assert force.delta(3) == 0.04 and force.delta(8) == pytest.approx(0.04)
        assert force.accumulate(400, 3, 8) == pytest.approx(508.50, abs=0.01)

    def test_force_linear_in_time(self):
        """delta = 0.01t + 0.04: nominal discount rate convertible half-yearly from 1 to 2."""
        force = ForceOfInterest([(None, [0.04, 0.01])])
        assert force.integral(1, 2) == pytest.approx(0.055)
        effective = force.equivalent_effective_rate(1, 2)
        assert InterestRate(effective).nominal_discount(2) == pytest.approx(0.054251, abs=5e-7)

    def test_three_piece_force(self):
        force = ForceOfInterest([(3, [0.08, -0.001]), (5, [-0.04, 0.025]), (None, [0.03])])
        assert force.integral(2, 10) == pytest.approx(0.3475)
        assert force.present_value(1000, due=10, at=2) == pytest.approx(706.45, abs=0.01)
        assert force.equivalent_effective_rate(2, 10) == pytest.approx(0.0444, abs=5e-5)

    def test_constant_force_and_validation(self):
        constant = ForceOfInterest(0.05)
        assert constant.accumulation_factor(0, 2) == pytest.approx(np.exp(0.1))
        assert constant.discount_factor(0, 2) == pytest.approx(np.exp(-0.1))
        assert constant.integral(3, 1) == pytest.approx(-0.1)
        with pytest.raises(ValueError, match="no upper limit"):
            ForceOfInterest([(5, [0.04])])
        with pytest.raises(ValueError, match="increasing"):
            ForceOfInterest([(5, [0.04]), (3, [0.05]), (None, [0.06])])


class TestRealAndMoneyRates:
    def test_inflation_from_index(self):
        assert inflation_from_index(724, 913, 5) == pytest.approx(0.0475, abs=5e-5)

    def test_real_returns_with_twenty_per_cent_inflation(self):
        coins = InterestRate.from_period(20000 / 14000 - 1, 3).i
        assert real_rate(coins, 0.20) == pytest.approx(-0.0615, abs=5e-5)
        painting = InterestRate.from_period(3200 / 3000 - 1, 0.5).i
        assert real_rate(painting, 0.20) == pytest.approx(-0.0519, abs=5e-5)
        diamond = InterestRate.from_period(10000 / 13000 - 1, 2).i
        assert real_rate(diamond, 0.20) == pytest.approx(-0.2691, abs=5e-5)

    def test_money_and_real_are_inverse(self):
        assert money_rate(real_rate(0.12, 0.05), 0.05) == pytest.approx(0.12)
        # Expected inflation from a money return of 5% and a real return of 2%
        assert (1.05 / 1.02) - 1 == pytest.approx(0.0294, abs=5e-5)


class TestDiscountingAndAccumulating:
    def test_present_values(self):
        assert present_value([280, 360], [2, 2.5], 0.15) == pytest.approx(465.56, abs=0.01)
        rate = InterestRate.from_nominal_discount(0.12, 4).i
        assert present_value([100, 200], [7 / 12, 11 / 12], rate) == pytest.approx(272.00, abs=0.01)

    def test_mixed_simple_then_discount(self):
        value = simple_accumulation(1000, 0.08, 2)
        value = InterestRate.from_nominal_discount(0.06, 12).accumulate(value, 2)
        assert value == pytest.approx(1308.29, abs=0.01)

    def test_rates_changing_over_time(self):
        v3, v5, v8 = 1 / 1.03, 1 / 1.05, 1 / 1.08
        value = (1500 * v3 ** 3 + 4000 * v3 ** 4 * v5 ** 3
                 + 5500 * v3 ** 4 * v5 ** 4 * v8 ** 2)
        assert value == pytest.approx(7889.49, abs=0.01)
        assert value * 1.03 ** 4 * 1.05 == pytest.approx(9323.68, abs=0.01)

    def test_interest_on_a_deposit(self):
        """Present value of the interest = deposit less present value of its return."""
        assert 200 - InterestRate(0.06).discount(200, 5) == pytest.approx(50.55, abs=0.01)

    def test_continuous_stream_under_a_varying_force(self):
        force = ForceOfInterest([(6, [0.002, 0.01, 0.0004]), (10, [0.01, 0.003]),
                                 (None, [0.04])])
        assert force.value_of_stream(120, 10, 15, at=10) == pytest.approx(543.81, abs=0.01)
        assert force.value_of_stream(120, 10, 15, at=0) == pytest.approx(380.62, abs=0.01)

    def test_accumulated_value_of_an_increasing_stream(self):
        """Rate of payment 0.3 + 1.5t from 4 to 8, force 0.01 + 0.05t, valued at 10."""
        force = ForceOfInterest([(None, [0.01, 0.05])])
        value = force.value_of_stream(lambda t: 0.3 + 1.5 * t, 4, 8, at=10)
        assert value == pytest.approx(184.855, abs=0.005)

    def test_cashflow_helpers(self):
        assert accumulated_value([100, 100], [0, 1], 0.10) == pytest.approx(210)
        assert accumulated_value([100, 100], [0, 1], 0.10, at=2) == pytest.approx(231)


class TestLevelAnnuities:
    def test_standard_functions(self):
        assert annuity(3, 0.09, p=4) == pytest.approx(2.6152, abs=5e-5)
        assert annuity(4, 0.09, p=4, timing="advance") == pytest.approx(3.4200, abs=5e-5)
        assert annuity(10, 0.25, p=12, timing="advance") == pytest.approx(4.0375, abs=5e-5)
        assert annuity(6.5, 0.25, p=12) == pytest.approx(3.3989, abs=5e-5)
        assert annuity(20, 0.10) == pytest.approx(8.5136, abs=5e-5)

    def test_relationships(self):
        i, n = 0.06, 15
        assert annuity(n, i, timing="advance") == pytest.approx(annuity(n, i) * 1.06)
        assert accumulated_annuity(n, i) == pytest.approx(annuity(n, i) * 1.06 ** n)
        assert annuity(n, i, deferred=5) == pytest.approx(annuity(n + 5, i) - annuity(5, i))
        assert annuity(n, i) < annuity(n, i, p=12) < annuity(n, i, timing="continuous") \
            < annuity(n, i, p=12, timing="advance") < annuity(n, i, timing="advance")
        assert annuity(n, 0.0) == n
        assert perpetuity(i) == pytest.approx(1 / i)
        assert perpetuity(i, timing="advance") == pytest.approx(1 / (i / 1.06))
        assert annuity(500, i) == pytest.approx(perpetuity(i), rel=1e-9)

    def test_payments_every_two_years(self):
        """100 every two years from 1980 to 2018, accumulated to 2020 at 12%."""
        two_yearly = InterestRate(0.12).per_period(2)
        assert 100 * accumulated_annuity(20, two_yearly, timing="advance") == \
            pytest.approx(45_389, abs=1)

    def test_monthly_payments_with_quarterly_rate(self):
        monthly = InterestRate.from_nominal(0.08, 4).per_period(1 / 12)
        assert 1000 * annuity(6, monthly) == pytest.approx(5863, abs=1)
        # Perpetuity of 40 a month at 10% convertible quarterly
        monthly = InterestRate.from_nominal(0.10, 4).per_period(1 / 12)
        assert 40 * perpetuity(monthly) == pytest.approx(4839.78, abs=0.01)

    def test_deferred_monthly_annuity(self):
        """41 payments of 100 from 1 January 2019, valued at 1 June 2018."""
        i = InterestRate.from_nominal(0.10, 2).i
        value = 1200 * annuity(41 / 12, i, p=12, deferred=0.5)
        assert value == pytest.approx(3307, abs=1)

    def test_rates_changing_during_the_annuity(self):
        value = 1000 * (annuity(4, 0.034, p=4)
                        + 1.034 ** -4 * annuity(2, 0.042, p=4))
        assert value == pytest.approx(5399.40, abs=0.01)
        assert value * 1.034 ** 4 * 1.042 ** 3 == pytest.approx(6982.81, abs=0.01)

    def test_ratio_of_two_annuities(self):
        x = 2000 * annuity(8, InterestRate.from_nominal(0.08, 4).i)
        four_yearly = InterestRate.from_nominal(0.08, 2).per_period(4)
        y = 4000 * annuity(4, four_yearly)
        assert x == pytest.approx(11_388, abs=1)
        assert y == pytest.approx(7_759, abs=1)
        assert x / y == pytest.approx(1.468, abs=5e-4)

    def test_invalid(self):
        with pytest.raises(ValueError):
            annuity(10, 0.05, timing="weekly")
        with pytest.raises(ValueError):
            perpetuity(0.0)


class TestIncreasingAnnuities:
    def test_standard_functions(self):
        assert increasing_annuity(60, 0.09) == pytest.approx(130.02, abs=0.005)
        assert accumulated_increasing_annuity(10, 0.07, timing="advance") == \
            pytest.approx(73.12, abs=0.005)
        assert increasing_annuity(25, 0.06) == pytest.approx(128.7565, abs=5e-4)

    def test_relationships(self):
        n, i = 12, 0.05
        direct = sum(k * 1.05 ** -k for k in range(1, n + 1))
        assert increasing_annuity(n, i) == pytest.approx(direct)
        assert increasing_annuity(n, i, "advance") == pytest.approx(direct * 1.05)
        # (Ia) + (Da) = (n + 1) a
        assert increasing_annuity(n, i) + decreasing_annuity(n, i) == \
            pytest.approx((n + 1) * annuity(n, i))
        assert increasing_annuity(n, 0.0) == n * (n + 1) / 2
        assert continuously_increasing_annuity(n, i) < increasing_annuity(n, i, "continuous")

    def test_continuously_increasing_under_a_varying_force(self):
        """Payment rate t for 10 years with force of interest 0.01t."""
        force = ForceOfInterest([(None, [0.0, 0.01])])
        assert force.value_of_stream(lambda t: t, 0, 10) == pytest.approx(39.35, abs=0.005)

    def test_increasing_annuity_with_a_change_of_rate(self):
        """500 rising by 50 a year in advance for 20 years; 5% for 12 years then 7%."""
        first = 450 * annuity(12, 0.05, timing="advance") \
            + 50 * increasing_annuity(12, 0.05, "advance")
        second = 1050 * annuity(8, 0.07, timing="advance") \
            + 50 * increasing_annuity(8, 0.07, "advance")
        assert first + 1.05 ** -12 * second == pytest.approx(11_417, abs=1)

    def test_compound_increases(self):
        # 200 at time 5, five increases of 5.7692%, then four of 6.7961%, at 10%
        value = 200 * 1.1 ** -4 * (
            geometric_annuity(6, 0.10, 0.057692)
            + (1.057692 / 1.1) ** 5 * 1.067961 / 1.1 * geometric_annuity(4, 0.10, 0.067961)
        )
        assert value == pytest.approx(1056.44, abs=0.05)
        assert geometric_annuity(5, 0.05, 0.05) == pytest.approx(5 / 1.05)
        assert geometric_annuity(5, 0.05, 0.0) == pytest.approx(annuity(5, 0.05))

    def test_rent_reviewed_every_five_years(self):
        """20,000 a year quarterly in advance for 50 years, rising with 3% inflation every 5 years."""
        five_years = 20_000 * annuity(5, 0.12, p=4, timing="advance")
        factor = (1.03 / 1.12) ** 5
        value = five_years * (1 - factor ** 10) / (1 - factor)
        assert value == pytest.approx(222_830, abs=5)

    def test_arithmetic_increases(self):
        full = 2500 * annuity(25, 0.06) + 500 * increasing_annuity(25, 0.06)
        assert full == pytest.approx(96_337, abs=1)
        capped = (2500 * annuity(11, 0.06) + 500 * increasing_annuity(11, 0.06)
                  + 8000 * annuity(14, 0.06, deferred=11))
        assert capped == pytest.approx(80_268, abs=1)


class TestEquationsOfValue:
    def test_price_term_and_yield_of_a_security(self):
        assert 5 * annuity(20, 0.10) + 125 * 1.1 ** -20 == pytest.approx(61.15, abs=0.005)
        flows = Cashflows([0], [-75]).add_level(0, 10, 5).add(10, 120)
        assert flows.internal_rate_of_return() == pytest.approx(0.104, abs=5e-4)
        # Term: 83.73 = 4 a(n) + 101 v^n at 6%
        n = np.log((101 - 4 / 0.06) / (83.73 - 4 / 0.06)) / np.log(1.06)
        assert n == pytest.approx(12, abs=0.01)

    def test_rent_on_a_property(self):
        """80,000 buys rent for 99 years, doubling and tripling, and 1.5m at the end."""
        unit = annuity(33, 0.08) * (1 + 2 * 1.08 ** -33 + 3 * 1.08 ** -66)
        rent = (80_000 - 1_500_000 * 1.08 ** -99) / unit
        assert rent == pytest.approx(5852, abs=1)

    def test_level_withdrawals(self):
        withdrawal = (2000 - 400 * 1.08 ** -11) / annuity(11, 0.08)
        assert withdrawal == pytest.approx(256.12, abs=0.01)
        flows = Cashflows([0], [-2000]).add_level(0, 11, withdrawal).add(11, 400)
        assert flows.net_present_value(0.08) == pytest.approx(0, abs=1e-8)

    def test_no_yield(self):
        with pytest.raises(ValueError, match="No yield"):
            Cashflows([0, 1], [100, 100]).internal_rate_of_return()


class TestLoanSchedules:
    def test_interest_in_first_quarterly_payment(self):
        loan = Loan(120_000, 0.06, 25, payments_per_year=4)
        assert loan.interest_element(1) == pytest.approx(1760.86, abs=0.01)

    def test_capital_repaid_in_sixth_year(self):
        loan = Loan(1000, 0.10, 10, payments_per_year=12)
        assert loan.payment == pytest.approx(12.98, abs=0.005)
        assert loan.capital_repaid(61, 72) == pytest.approx(101.07, abs=0.02)

    def test_half_the_capital_repaid_at_the_end(self):
        loan = Loan(100_000, 0.08, 10, final_repayment=50_000)
        assert loan.payment == pytest.approx(11_451, abs=1)
        assert loan.outstanding(10) == pytest.approx(50_000)

    def test_consumer_credit(self):
        assert Loan(4000, 0.154, 5, payments_per_year=12).payment == pytest.approx(93.92, abs=0.005)
        apr = annual_percentage_rate(7500, 368.75, 24, payments_per_year=12)
        assert np.floor(apr * 1000) / 10 == 17.7
        assert flat_rate(7500, 24 * 368.75, 2) == pytest.approx(0.09)

    def test_ten_year_loan(self):
        loan = Loan(50_000, 0.08, 10, payments_per_year=12)
        assert loan.payment == pytest.approx(599.29, abs=0.005)
        assert loan.outstanding(12) == pytest.approx(46_548.71, abs=0.3)
        assert loan.interest_paid(1, 12) == pytest.approx(3_740.19, abs=0.3)
        # Two payments missed after year 7: the debt at 7 years 2 months over 34 payments
        j = loan.rate_per_period
        debt = loan.outstanding(84) * (1 + j) ** 2
        new_payment = debt * j / (1 - (1 + j) ** -34)
        assert new_payment - loan.payment == pytest.approx(39.49, abs=0.02)

    def test_schedule_with_increasing_payments(self):
        payments = 100 + 20 * np.arange(15)
        amount = 80 * annuity(15, 0.05) + 20 * increasing_annuity(15, 0.05)
        assert amount == pytest.approx(2303.73, abs=0.01)
        schedule = loan_schedule(amount, payments, 0.05)
        assert schedule.loc[6, "opening"] == pytest.approx(2177.38, abs=0.02)
        assert schedule.loc[6, "interest"] == pytest.approx(108.87, abs=0.01)
        assert schedule.loc[6, "capital"] == pytest.approx(91.13, abs=0.01)
        assert schedule.loc[6, "closing"] == pytest.approx(2086.25, abs=0.02)
        assert schedule.loc[7, "interest"] == pytest.approx(104.31, abs=0.01)
        assert schedule.loc[15, "closing"] == pytest.approx(0, abs=1e-6)

    def test_schedule_ties(self):
        loan = Loan(1000, 0.07, 3)
        schedule = loan.schedule()
        assert loan.payment == pytest.approx(381.05, abs=0.005)
        np.testing.assert_allclose(schedule["interest"], [70, 48.22, 24.93], atol=0.01)
        np.testing.assert_allclose(schedule["closing"], [688.95, 356.12, 0], atol=0.01)
        assert schedule["capital"].sum() == pytest.approx(1000)
        assert loan.total_interest == pytest.approx(schedule["interest"].sum())
        assert loan.capital_element(2) == pytest.approx(schedule.loc[2, "capital"])
        with pytest.raises(ValueError):
            loan.outstanding(4)

    def test_excel(self, tmp_path):
        openpyxl = pytest.importorskip("openpyxl")
        path = tmp_path / "loan.xlsx"
        Loan(50_000, 0.08, 10, payments_per_year=12).to_excel(str(path), currency="USD")
        book = openpyxl.load_workbook(path)
        assert book.sheetnames == ["Contents", "Schedule"]


class TestProjectAppraisal:
    @pytest.fixture
    def project(self):
        return (Cashflows([0], [-25_000]).add_continuous(0, 5, 10_000)
                .add_continuous(0, 5, -2_000).add(6, -5_000))

    def test_net_present_value_payback_and_yield(self, project):
        assert project.net_present_value(0.10) == pytest.approx(3996.16, abs=0.01)
        assert project.discounted_payback_period(0.10) == pytest.approx(3.71, abs=0.005)
        assert project.internal_rate_of_return() == pytest.approx(0.189, abs=1e-3)
        assert project.payback_period() == pytest.approx(25_000 / 8_000)

    def test_interest_only_loan_funding_a_project(self):
        interest = 1000 * InterestRate(0.06).per_period(1 / 12)
        assert interest == pytest.approx(4.87, abs=0.005)
        # Net income accumulated at 5%, less the 1,000 repaid at the end
        profit = 12 * (50 - interest) * accumulated_annuity(2, 0.05, p=12) - 1000
        assert profit == pytest.approx(135, abs=0.5)

    def test_payback_then_reinvestment(self):
        """Borrow 50m at 9%; 6m a year half-yearly for 20 years; surplus earns 7%."""
        project = Cashflows([0], [-50.0]).add_level(0, 20, 6.0, p=2)
        assert project.discounted_payback_period(0.09) == pytest.approx(15.5)
        assert project.accumulated_profit(0.09, 0.07) == pytest.approx(32.153, abs=0.005)

    def test_never_pays_back(self):
        project = Cashflows([0, 1], [-100, 50])
        assert project.discounted_payback_period(0.05) is None
        assert project.accumulated_profit(0.05, 0.03) == pytest.approx(-100 * 1.05 + 50)

    def test_building_cashflows(self):
        flows = Cashflows().add_level(0, 2, 1200, p=12, timing="advance")
        frame = flows.to_frame()
        assert len(frame) == 24 and frame["time"].iloc[0] == 0
        assert flows.present_value(0.05) == pytest.approx(
            1200 * annuity(2, 0.05, p=12, timing="advance"))
        assert flows.accumulated_value(0.05, at=2) == pytest.approx(
            1200 * accumulated_annuity(2, 0.05, p=12, timing="advance"))
        continuous = Cashflows().add_continuous(1, 4, 100)
        assert continuous.present_value(0.05) == pytest.approx(
            100 * annuity(3, 0.05, timing="continuous", deferred=1))
        assert continuous.present_value(0.0) == pytest.approx(300)


class TestBondsAndEquities:
    def test_price_net_of_income_tax(self):
        bond = Bond(0.06, 13, frequency=2)
        assert bond.price(0.05, income_tax=0.40) == pytest.approx(87.2666, abs=5e-4)
        assert Bond(0.06, 13, nominal=10_000).price(0.05, income_tax=0.40) == \
            pytest.approx(8726.66, abs=0.05)

    def test_capital_gains_tax_payable(self):
        bond = Bond(0.06, 15)
        price = bond.price(0.10)
        assert price == pytest.approx(70.69, abs=0.005)
        assert bond.capital_gains_tax_payable(price, 0.30) == pytest.approx(8.79, abs=0.005)

    def test_yield_from_price(self):
        bond = Bond(0.05, 9.5, redemption=105)
        assert bond.redemption_yield(85) == pytest.approx(0.078, abs=5e-4)
        # Pricing at that yield gives the price back
        assert bond.price(bond.redemption_yield(85)) == pytest.approx(85)

    def test_price_with_both_taxes_round_trips(self):
        bond = Bond(0.07, 10)
        price = bond.price(0.06, income_tax=0.40, capital_gains_tax=0.25)
        assert price < 100
        assert bond.redemption_yield(price, 0.40, 0.25) == pytest.approx(0.06)
        # No capital gain when the coupon is high relative to the yield
        rich = Bond(0.12, 10)
        assert not rich.has_capital_gain(0.06)
        assert rich.price(0.06, capital_gains_tax=0.25) == pytest.approx(rich.price(0.06))

    def test_optional_redemption_dates(self):
        """7% half-yearly, redeemable between 5 and 10 years; 40% income tax, 25% CGT, 6% net."""
        result = bond_price_with_optional_redemption(
            0.07, earliest=5, latest=10, yield_=0.06, income_tax=0.40, capital_gains_tax=0.25)
        assert result["assumed_term"] == 10
        assert result["price"] == pytest.approx(85.13, abs=0.005)
        assert Bond(0.07, 10).running_yield(result["price"], income_tax=0.40) == \
            pytest.approx(0.0493, abs=5e-5)
        # With a capital loss the earliest date is the worst case
        loss = bond_price_with_optional_redemption(0.12, 5, 10, 0.06)
        assert loss["assumed_term"] == 5

    def test_equity_with_half_yearly_dividends(self):
        """Price 3.60, next dividend 0.12 in three months, growth 2% per half-year."""
        assert equity_yield(3.60, 0.12, growth=0.02, time_to_next=0.25,
                            dividends_per_year=2) == pytest.approx(0.111, abs=5e-4)
        assert equity_price(0.12, 0.111, 0.02, 0.25, 2) == pytest.approx(3.60, abs=0.02)
        assert equity_price(5, 0.08, 0.03) == pytest.approx(5 / 0.05)
        with pytest.raises(ValueError, match="exceed"):
            equity_price(5, 0.03, 0.05)

    def test_dividends_with_changing_growth(self):
        """35p paid, growing 3%, then 5%, then 6% for ever; 100 shares at 8%."""
        dividends = [35 * 1.03, 35 * 1.03 * 1.05]
        value = present_value_of_dividends(dividends, 0.08, terminal_growth=0.06)
        assert value == pytest.approx(1785.81, abs=0.05)
        assert present_value_of_dividends([10, 10], 0.05) == pytest.approx(10 * annuity(2, 0.05))

    def test_index_linked_zero_coupon_bond(self):
        """10,000 nominal bought for 10,250, redeemed three years later with a lagged index."""
        redemption = 10_000 * 193 / 148
        assert redemption == pytest.approx(13_040.54, abs=0.01)
        assert internal_rate_of_return([-10_250, redemption], [0, 3]) == \
            pytest.approx(0.0836, abs=5e-5)
        assert real_yield([-10_250, redemption], [0, 3], [175, 201]) == \
            pytest.approx(0.0347, abs=5e-5)


class TestTermStructure:
    def test_forward_rate_from_a_spot_curve(self):
        def spot(n):
            return 0.09 - 0.03 * np.exp(-0.1 * n)
        assert forward_rate(spot(10), 10, spot(11), 11) == pytest.approx(0.0906, abs=5e-5)

    def test_par_yield_from_forward_rates(self):
        spots = spot_rates_from_forwards([0.06, 0.065, 0.07])
        assert par_yield(spots) == pytest.approx(0.06478, abs=5e-6)
        np.testing.assert_allclose(forwards_from_spot_rates(spots), [0.06, 0.065, 0.07])

    def test_spot_rates_from_two_bonds(self):
        """A two-year par yield of 5.65% and a 7% bond redeemed at 101 priced at 103.40."""
        spots = spot_rates_from_bonds([[5.65, 105.65], [7, 108]], [100, 103.40])
        np.testing.assert_allclose(spots, [0.0414, 0.0569], atol=5e-5)

    def test_spot_rates_from_three_bonds(self):
        """6% coupons, redeemed at 103, each priced at 97."""
        spots = spot_rates_from_bonds(
            [[109, 0, 0], [6, 109, 0], [6, 6, 109]], [97, 97, 97])
        np.testing.assert_allclose(spots[:2], [0.12371, 0.09049], atol=5e-6)
        flows = Cashflows([0, 1, 2, 3], [-97, 6, 6, 109])
        assert flows.internal_rate_of_return() == pytest.approx(0.0809, abs=2e-4)

    def test_zero_coupon_yields_from_expected_rates(self):
        spots = spot_rates_from_forwards([0.06, 0.05, 0.04, 0.03])
        np.testing.assert_allclose(spots, [0.06, 0.05499, 0.04997, 0.04494], atol=5e-6)
        discount = (1 + spots) ** -np.arange(1, 5)
        price = 4 * discount.sum() + 110 * discount[-1]
        gry = Cashflows([0, 1, 2, 3, 4], [-price, 4, 4, 4, 114]).internal_rate_of_return()
        assert gry == pytest.approx(0.0454, abs=2e-4)

    def test_duration_and_convexity_of_a_bond(self):
        amounts, times = [10, 10, 110], [1, 2, 3]
        assert discounted_mean_term(amounts, times, 0.08) == pytest.approx(2.74, abs=0.005)
        assert volatility(amounts, times, 0.08) == pytest.approx(2.54, abs=0.005)
        # Convexity against a numerical second derivative of the price
        def price(i):
            return sum(a * (1 + i) ** -t for a, t in zip(amounts, times))
        h = 1e-4
        numerical = (price(0.08 + h) - 2 * price(0.08) + price(0.08 - h)) / h ** 2 / price(0.08)
        assert convexity(amounts, times, 0.08) == pytest.approx(numerical, rel=1e-5)

    def test_discounted_mean_term_of_annuities(self):
        years = np.arange(1, 21)
        assert discounted_mean_term(np.full(20, 1000.0), years, 0.10) == \
            pytest.approx(7.51, abs=0.005)
        assert discounted_mean_term(1000 * 1.1 ** (years - 1), years, 0.10) == \
            pytest.approx(10.5)
        # Continuous annuity: (I-bar a-bar) / a-bar
        assert continuously_increasing_annuity(20, 0.10) / annuity(20, 0.10, timing="continuous") \
            == pytest.approx(7.00, abs=0.005)

    def test_immunisation_with_two_bonds(self):
        """Liabilities of 2,000(10 - t) at times 5 to 9, valued at 6%."""
        times = np.arange(5, 10)
        liabilities = 2000.0 * (10 - times)
        coupon_times = np.arange(1, 16)
        bond_a = np.full(15, 5.0)
        bond_a[-1] += 100
        result = immunising_amounts(liabilities, times, bond_a, coupon_times, [100], [5], 0.06)
        assert result["present_value"] == pytest.approx(20_796, abs=1)
        assert result["discounted_mean_term"] == pytest.approx(6.245, abs=5e-4)
        assert result["invest_a"] == pytest.approx(4_576, abs=5)
        assert result["invest_b"] == pytest.approx(16_220, abs=5)
        assert result["assets_more_convex"]

    def test_redington_conditions(self):
        """Liabilities of 10m and 20m at 10 and 15 years; zero-coupon assets at 2 and 25 years."""
        check = immunisation_check([7.404, 31.834], [2, 25], [10, 20], [10, 15], 0.07,
                                   tolerance=1e-3)
        assert check["pv_liabilities"] == pytest.approx(12.332, abs=5e-4)
        assert check["dmt_liabilities"] == pytest.approx(12.939, abs=5e-4)
        assert check["present_values_equal"] and check["mean_terms_equal"]
        assert check["assets_more_convex"] and check["immunised"]
        # A rise to 7.5% leaves a small profit
        profit = (present_value([7.404, 31.834], [2, 25], 0.075)
                  - present_value([10, 20], [10, 15], 0.075))
        assert profit == pytest.approx(0.0158, abs=5e-4)
        # A fall in rates also leaves a profit: that is what immunisation means
        fall = (present_value([7.404, 31.834], [2, 25], 0.065)
                - present_value([10, 20], [10, 15], 0.065))
        assert fall > 0


class TestSyllabusGaps:
    """Topics added after auditing the module against the syllabus objectives."""

    def test_bond_bought_between_coupon_dates(self):
        """8-year 4% annual bond redeemed at 105, bought after 6 months; 25% and 15% tax."""
        bond = Bond(0.04, 8, redemption=105, frequency=1)
        price = bond.price(0.05, income_tax=0.25, capital_gains_tax=0.15, elapsed=0.5)
        assert price == pytest.approx(91.26, abs=0.005)
        assert bond.redemption_yield(price, 0.25, 0.15, elapsed=0.5) == pytest.approx(0.05)
        assert bond.accrued_interest(0.5) == pytest.approx(2.0)
        assert bond.clean_price(0.05, elapsed=0.5) == pytest.approx(
            bond.price(0.05, elapsed=0.5) - 2.0)
        # With no time elapsed the price is unchanged
        assert bond.price(0.05, elapsed=0.0) == pytest.approx(
            4 * annuity(8, 0.05) + 105 * 1.05 ** -8)
        # Just after issue and just before the first coupon differ by a year's growth
        assert Bond(0.05, 3, frequency=1).accrued_interest(1.25) == pytest.approx(1.25)

    def test_tax_paid_on_a_fixed_date(self):
        """
        6% half-yearly coupons on 30 June and 31 December, redeemed at 105 after
        11 years; income tax of 23% and capital gains tax of 40% paid the following 1 April.
        """
        bond = Bond(0.06, 11, redemption=105)
        price = bond.price(0.05, income_tax=0.23, capital_gains_tax=0.40,
                           income_tax_delay=[0.75, 0.25], capital_gains_tax_delay=0.25)
        assert price == pytest.approx(99.18, abs=0.005)
        assert bond.redemption_yield(price, 0.23, 0.40, income_tax_delay=[0.75, 0.25],
                                     capital_gains_tax_delay=0.25) == pytest.approx(0.05)
        # Paying tax later is worth something to the investor
        assert price > bond.price(0.05, income_tax=0.23, capital_gains_tax=0.40)

    def test_bounds_for_optional_redemption(self):
        result = bond_price_with_optional_redemption(
            0.07, 5, 10, 0.06, income_tax=0.40, capital_gains_tax=0.25)
        assert result["lower"] == pytest.approx(result["price"])
        assert result["upper"] > result["lower"]
        assert result["upper"] == pytest.approx(
            Bond(0.07, 5).price(0.06, income_tax=0.40, capital_gains_tax=0.25))

    def test_index_linked_bond_payments(self):
        """3% index-linked stock, 10,000 nominal, base index 149.2."""
        from actuneo.finance import index_linked_cashflows
        times, amounts = index_linked_cashflows(
            0.03, 1, base_index=149.2, index_values=[171.4, 173.8], nominal=10_000)
        np.testing.assert_allclose(times, [0.5, 1.0])
        assert amounts[0] == pytest.approx(172.319, abs=5e-4)
        assert amounts[1] == pytest.approx(174.732 + 11_648.794, abs=1e-3)
        with pytest.raises(ValueError, match="one index value"):
            index_linked_cashflows(0.03, 2, 100, [101, 102])

    def test_property_with_rent_reviews(self):
        from actuneo.finance import property_value, stepped_annuity
        assert property_value(20_000, 0.12, 50, rent_growth=0.03, review_period=5) == \
            pytest.approx(222_830, abs=5)
        # Annual reviews are a compound increasing annuity
        assert stepped_annuity(1, 10, 0.08, 0.03) == pytest.approx(
            geometric_annuity(10, 0.08, 0.03))
        assert stepped_annuity(5, 4, 0.06, 0.0) == pytest.approx(annuity(20, 0.06))
        sold = property_value(20_000, 0.12, 50, 0.03, 5, sale_proceeds=1_000_000)
        assert sold == pytest.approx(222_829.76 + 1_000_000 * 1.12 ** -50, abs=0.05)

    def test_deferred_increasing_annuities(self):
        assert increasing_annuity(10, 0.06, deferred=5) == pytest.approx(
            1.06 ** -5 * increasing_annuity(10, 0.06))
        assert decreasing_annuity(10, 0.06, "advance", deferred=3) == pytest.approx(
            1.06 ** -3 * decreasing_annuity(10, 0.06, "advance"))

    def test_uncertain_payments(self):
        """Expected present value of lottery prizes, with daily interest of 0.016%."""
        i = 1.00016 ** 365 - 1
        day = 1 / 365
        prizes = Cashflows(
            [day, day, 7 * day, 14 * day, 28 * day],
            [20, 200, 2_000, 200_000, 2_000_000],
            probabilities=[1 / 50, 1 / 1_000, 1 / 50_000, 1 / 2_000_000, 1 / 14_000_000],
        )
        v = 1 / 1.00016
        expected = (20 / 50 * v + 200 / 1000 * v + 2000 / 50_000 * v ** 7
                    + 200_000 / 2e6 * v ** 14 + 2e6 / 14e6 * v ** 28)
        assert prizes.present_value(i) == pytest.approx(expected)
        assert Cashflows().add(1, 100, probability=0.9).present_value(0.0) == pytest.approx(90)
        with pytest.raises(ValueError):
            Cashflows([1], [100], probabilities=[1.5])

    def test_conditions_for_a_unique_yield(self):
        assert Cashflows([0, 1, 2], [-100, 60, 60]).has_unique_yield()
        assert Cashflows([0, 1, 2], [100, -60, -60]).has_unique_yield()
        assert not Cashflows([0, 1, 2], [-100, 230, -132]).has_unique_yield()
        assert Cashflows([0], [-100]).add_continuous(0, 5, 30).has_unique_yield()

    def test_project_with_stepped_continuous_income(self):
        """
        Outlay of 85,000 at times 0, 1 and 2 and 125,000 at time 10; income received
        continuously from year 3, rising to year 6 and then by 2% a year to year 20.
        """
        project = Cashflows([0, 1, 2, 10], [-85, -85, -85, -125])
        rates = [30, 32, 34, 36] + [36 * 1.02 ** k for k in range(1, 15)]
        for k, rate in enumerate(rates):
            project.add_continuous(2 + k, 3 + k, rate)
        assert project.net_present_value(0.07) == pytest.approx(45.484, abs=5e-4)
        assert project.discounted_payback_period(0.07) == pytest.approx(16.618, abs=5e-4)
        assert not project.has_unique_yield()

    def test_comparing_two_projects(self):
        from actuneo.finance import crossover_rate
        early = Cashflows([0, 1], [-100, 125])
        late = Cashflows([0, 5], [-100, 200])
        rate = crossover_rate(early, late)
        assert early.net_present_value(rate) == pytest.approx(late.net_present_value(rate))
        # The project with the higher yield is not the better one at low rates of interest
        assert early.internal_rate_of_return() > late.internal_rate_of_return()
        assert late.net_present_value(0.05) > early.net_present_value(0.05)
        combined = early + late
        assert combined.net_present_value(0.08) == pytest.approx(
            early.net_present_value(0.08) + late.net_present_value(0.08))

    def test_continuous_time_rates(self):
        from actuneo.finance import (
            continuous_spot_rate, continuous_forward_rate, continuous_forward_from_spots,
            instantaneous_forward_rate, effective_to_continuous, continuous_to_effective,
        )
        # Unit zero-coupon bond prices of 0.70, 0.47 and 0.30 for 5, 10 and 15 years
        assert continuous_spot_rate(0.47, 10) == pytest.approx(0.0755, abs=5e-5)
        assert continuous_forward_rate(0.70, 0.30, 10) == pytest.approx(0.0847, abs=5e-5)
        y5, y15 = continuous_spot_rate(0.70, 5), continuous_spot_rate(0.30, 15)
        assert continuous_forward_from_spots(y5, 5, y15, 15) == pytest.approx(
            continuous_forward_rate(0.70, 0.30, 10))
        assert continuous_to_effective(effective_to_continuous(0.08)) == pytest.approx(0.08)
        # A flat spot curve has the same instantaneous forward rate
        assert instantaneous_forward_rate(lambda t: 0.05, 3) == pytest.approx(0.05)
        # Y(t) = 0.03 + 0.002t gives a forward rate of 0.03 + 0.004t
        assert instantaneous_forward_rate(lambda t: 0.03 + 0.002 * t, 5) == \
            pytest.approx(0.05, abs=1e-8)

    def test_sensitivity_from_duration_and_convexity(self):
        from actuneo.finance import estimated_value_change, effective_duration
        amounts, times = [10, 10, 110], [1, 2, 3]
        price = present_value(amounts, times, 0.08)
        estimate = estimated_value_change(price, volatility(amounts, times, 0.08),
                                          convexity(amounts, times, 0.08), 0.01)
        actual = present_value(amounts, times, 0.09) - price
        assert estimate == pytest.approx(actual, abs=0.01)
        assert effective_duration(amounts, times, 0.08) == volatility(amounts, times, 0.08)
        flows = Cashflows(times, amounts)
        assert flows.discounted_mean_term(0.08) == pytest.approx(
            discounted_mean_term(amounts, times, 0.08))
        # A level continuous stream for 20 years at 10%
        stream = Cashflows().add_continuous(0, 20, 1000)
        assert stream.discounted_mean_term(0.10) == pytest.approx(7.00, abs=0.005)

    def test_instrument_cashflow_models(self):
        from actuneo.finance import instruments
        zero = instruments.zero_coupon_bond(54, 100, 15)
        assert zero.internal_rate_of_return() == pytest.approx(0.0419, abs=5e-5)

        bond = instruments.fixed_interest_security(85, 0.05, 9.5, redemption=105)
        assert bond.internal_rate_of_return() == pytest.approx(0.078, abs=5e-4)

        lender = instruments.repayment_loan(1000, 0.07, 3)
        assert lender.internal_rate_of_return() == pytest.approx(0.07)
        assert instruments.interest_only_loan(1000, 0.06, 2, 12).internal_rate_of_return() == \
            pytest.approx(0.06)
        assert instruments.annuity_certain(1000, 100, 15).has_unique_yield()

        share = instruments.equity(1720, [36.05, 37.8525, 40.1237], sale_proceeds=1800)
        assert share.net_present_value(0.0) == pytest.approx(36.05 + 37.8525 + 40.1237 + 80)

        linked = instruments.index_linked_security(100, [1, 2], [3, 103], 100, [104, 108])
        assert linked.to_frame()["amount"].tolist() == pytest.approx([-100, 3.12, 111.24])
