"""
Pytest configuration and fixtures for ACTUNEO testing.
"""

import numpy as np
import pytest
from actuneo.mortality import MortalityTable
from actuneo.finance import YieldCurve


def makeham_qx(ages, a=0.0007, b=0.00005, c=1.09):
    """One-year mortality rates under Makeham's law, mu_x = a + b * c**x."""
    ages = np.asarray(ages, dtype=float)
    return 1 - np.exp(-(a + b * c ** ages * (c - 1) / np.log(c)))


@pytest.fixture
def sample_mortality_table():
    """A closed Makeham mortality table for ages 20-110 (qx = 1 at age 110)."""
    ages = np.arange(20, 111)
    qx = makeham_qx(ages)
    qx[-1] = 1.0
    return MortalityTable(ages, qx, name="Test Table")


@pytest.fixture
def open_mortality_table():
    """The same table cut off at age 100, where qx is still below 1."""
    ages = np.arange(20, 101)
    return MortalityTable(ages, makeham_qx(ages), name="Open Test Table")


@pytest.fixture
def sample_yield_curve():
    """Create a sample yield curve for testing."""
    maturities = [1, 2, 3, 5, 10, 20, 30]
    yields = [0.03, 0.035, 0.04, 0.045, 0.055, 0.065, 0.07]
    return YieldCurve(maturities, yields, interpolation_method='linear')


@pytest.fixture
def sample_zero_rates():
    """Sample zero-coupon rates for testing."""
    maturities = [1, 2, 3, 5, 10]
    zero_rates = [0.025, 0.032, 0.038, 0.045, 0.052]
    return maturities, zero_rates


@pytest.fixture
def tolerance():
    """Default tolerance for floating point comparisons."""
    return 1e-6


@pytest.fixture
def large_tolerance():
    """Larger tolerance for less precise calculations."""
    return 1e-4
