"""
Tests for the finance module.
"""

import numpy as np
import pytest
from actuneo.finance import InterestTheory, YieldCurve, DurationConvexity


class TestInterestTheory:
    """Test cases for InterestTheory class."""

    def test_initialization(self):
        """Test InterestTheory initialization."""
        it = InterestTheory(interest_rate=0.05, compounding_frequency=2)
        assert it.i == 0.05
        assert it.m == 2
        assert it.i_m == 0.025  # Periodic rate

    def test_future_value(self):
        """Test future value calculations."""
        it = InterestTheory(0.05)

        # Basic future value
        fv = it.future_value(1000, 5)
        expected = 1000 * (1.05 ** 5)
        assert abs(fv - expected) < 1e-10

        # With different rate
        fv2 = it.future_value(1000, 5, interest_rate=0.03)
        expected2 = 1000 * (1.03 ** 5)
        assert abs(fv2 - expected2) < 1e-10

    def test_present_value(self):
        """Test present value calculations."""
        it = InterestTheory(0.05)

        # Basic present value
        pv = it.present_value(1276.28, 5)  # Future value of $1000 at 5%
        expected = 1276.28 / (1.05 ** 5)
        assert abs(pv - expected) < 1e-10  # Check against calculated expected value

    def test_annuity_calculations(self):
        """Test annuity calculations."""
        it = InterestTheory(0.05)

        # Immediate annuity
        ann_imm = it.annuity_present_value(100, 10, immediate=True)
        assert ann_imm > 0

        # Annuity due
        ann_due = it.annuity_present_value(100, 10, immediate=False)
        assert ann_due > ann_imm

        # Future value of annuity
        fv_ann = it.annuity_future_value(100, 10)
        assert fv_ann > 0

    def test_loan_calculations(self):
        """Test loan payment and balance calculations."""
        it = InterestTheory(0.06)

        # Loan payment
        payment = it.loan_payment(100000, 30)  # 30-year loan
        assert payment > 0

        # Loan balance after payments
        balance = it.loan_balance(100000, 30, 5)  # After 5 payments
        assert balance < 100000  # Should be less than original

    def test_effective_rates(self):
        """Test effective rate conversions."""
        it = InterestTheory(0.05)

        # Effective annual rate
        ear = it.effective_annual_rate(0.05, 12)  # Monthly compounding
        assert ear > 0.05  # Should be higher than nominal

        # Real interest rate
        real_rate = it.real_interest_rate(0.08, 0.03)  # 8% nominal, 3% inflation
        expected_real = (1.08 / 1.03) - 1  # Proper Fisher equation
        assert abs(real_rate - expected_real) < 1e-10


class TestYieldCurve:
    """Test cases for YieldCurve class."""

    def test_initialization(self, sample_yield_curve):
        """Test YieldCurve initialization."""
        yc = sample_yield_curve
        assert len(yc.maturities) == len(yc.yields)
        assert yc.interpolation_method == 'linear'

    def test_yield_interpolation(self, sample_yield_curve):
        """Test yield interpolation."""
        yc = sample_yield_curve

        # Test interpolation within range
        y_4y = yc.get_yield(4)
        assert 0.04 < y_4y < 0.045  # Between 3y and 5y yields

        # Test extrapolation
        y_40y = yc.get_yield(40)
        assert y_40y > yc.yields[-1]  # Should extrapolate upward

    def test_spot_and_forward_rates(self, sample_yield_curve):
        """Test spot and forward rate calculations."""
        yc = sample_yield_curve

        # Spot rates (simplified)
        spot_5y = yc.get_spot_rate(5)
        assert spot_5y == yc.get_yield(5)

        # Forward rates
        fwd_1to2 = yc.get_forward_rate(1, 2)
        assert fwd_1to2 > 0

        fwd_5to10 = yc.get_forward_rate(5, 10)
        assert fwd_5to10 > 0

    def test_discount_factors(self, sample_yield_curve):
        """Test discount factor calculations."""
        yc = sample_yield_curve

        df_5y = yc.get_discount_factor(5)
        assert 0 < df_5y < 1

        df_10y = yc.get_discount_factor(10)
        assert df_10y < df_5y  # Longer maturity should have lower discount factor

    def test_bootstrap_spot_rates(self, sample_yield_curve):
        """Test spot rate bootstrapping."""
        yc = sample_yield_curve
        spot_rates = yc.bootstrap_spot_rates()

        assert len(spot_rates) == len(yc.maturities)
        assert all(rate > 0 for rate in spot_rates.values())


class TestDurationConvexity:
    """Test cases for DurationConvexity class."""

    def test_macaulay_duration(self):
        """Test Macaulay duration calculations."""
        dc = DurationConvexity()

        # Simple bond: $1000 face, 5% coupon, 5% yield, 5 years
        cash_flows = [50, 50, 50, 50, 1050]
        times = [1, 2, 3, 4, 5]

        duration = dc.macaulay_duration(cash_flows, times, 0.05)
        assert duration > 0 and duration <= 5

    def test_modified_duration(self):
        """Test modified duration."""
        dc = DurationConvexity()

        cash_flows = [50, 50, 50, 50, 1050]
        times = [1, 2, 3, 4, 5]

        macaulay_d = dc.macaulay_duration(cash_flows, times, 0.05)
        modified_d = dc.modified_duration(cash_flows, times, 0.05)

        expected_modified = macaulay_d / 1.05
        assert abs(modified_d - expected_modified) < 1e-10

    def test_convexity(self):
        """Test convexity calculations."""
        dc = DurationConvexity()

        cash_flows = [50, 50, 50, 50, 1050]
        times = [1, 2, 3, 4, 5]

        convexity = dc.convexity(cash_flows, times, 0.05)
        assert convexity > 0

    def test_bond_duration_convexity(self):
        """Test bond-specific duration and convexity."""
        dc = DurationConvexity()

        duration = dc.bond_duration(1000, 0.05, 5, 0.05)
        convexity = dc.bond_convexity(1000, 0.05, 5, 0.05)

        assert duration > 0 and duration <= 5
        assert convexity > 0

    def test_price_change_approximation(self):
        """Test duration-convexity price change approximation."""
        dc = DurationConvexity()

        duration = 4.5
        convexity = 25.0
        yield_change = 0.01  # 1% increase
        current_price = 1000

        price_change = dc.price_change_approximation(duration, convexity, yield_change, current_price)

        # Should be negative (price decreases when yields increase)
        assert price_change < 0

        # Duration effect should be larger than convexity effect for small changes
        duration_effect = -duration * yield_change * current_price
        convexity_effect = 0.5 * convexity * yield_change ** 2 * current_price

        assert abs(price_change - (duration_effect + convexity_effect)) < 1e-10
