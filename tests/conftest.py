"""
Pytest configuration and fixtures for ACTUNEO testing.
"""

import numpy as np
import pytest
from actuneo.mortality import MortalityTable
from actuneo.finance import YieldCurve


@pytest.fixture
def sample_mortality_table():
    """Create a sample mortality table for testing."""
    ages = np.arange(20, 101)
    # Simplified mortality rates - increasing with age
    qx = 0.001 + 0.00005 * (ages - 20)
    qx = np.clip(qx, 0, 0.1)  # Cap at 10% for reasonableness
    return MortalityTable(ages, qx, name="Test Table")


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
