"""
Yield Curve Construction and Analysis

Provides tools for constructing and analyzing yield curves,
including spot rates, forward rates, and various interpolation methods.
"""

import numpy as np
from typing import List, Union, Optional, Dict, Callable
import matplotlib.pyplot as plt


class YieldCurve:
    """
    A class for constructing and analyzing yield curves.

    Supports various interpolation methods and provides tools for
    spot rates, forward rates, and yield curve analysis.
    """

    def __init__(self,
                 maturities: Union[List[float], np.ndarray],
                 yields: Union[List[float], np.ndarray],
                 interpolation_method: str = 'linear'):
        """
        Initialize YieldCurve with maturity and yield data.

        Args:
            maturities: Time to maturity (in years)
            yields: Yield rates (decimal)
            interpolation_method: Method for interpolation ('linear', 'cubic', 'nelson_siegel')
        """
        self.maturities = np.array(maturities, dtype=float)
        self.yields = np.array(yields, dtype=float)
        self.interpolation_method = interpolation_method

        # Sort by maturity
        sort_idx = np.argsort(self.maturities)
        self.maturities = self.maturities[sort_idx]
        self.yields = self.yields[sort_idx]

        # Validate inputs
        if len(self.maturities) != len(self.yields):
            raise ValueError("maturities and yields must have the same length")

        if not np.all(self.maturities > 0):
            raise ValueError("All maturities must be positive")

        # Set up interpolation function
        self._setup_interpolation()

    def _setup_interpolation(self):
        """Set up the interpolation method."""
        if self.interpolation_method == 'linear':
            self._interp_func = self._linear_interpolation
        elif self.interpolation_method == 'cubic':
            from scipy.interpolate import interp1d
            self._interp_func = interp1d(self.maturities, self.yields,
                                       kind='cubic', bounds_error=False,
                                       fill_value='extrapolate')
        elif self.interpolation_method == 'nelson_siegel':
            self._fit_nelson_siegel()
        else:
            raise ValueError(f"Unknown interpolation method: {self.interpolation_method}")

    def _linear_interpolation(self, t: float) -> float:
        """Linear interpolation for yields."""
        if t <= self.maturities[0]:
            return self.yields[0]
        elif t >= self.maturities[-1]:
            # Linear extrapolation
            slope = (self.yields[-1] - self.yields[-2]) / (self.maturities[-1] - self.maturities[-2])
            return self.yields[-1] + slope * (t - self.maturities[-1])
        else:
            # Linear interpolation
            idx = np.searchsorted(self.maturities, t) - 1
            t1, t2 = self.maturities[idx], self.maturities[idx + 1]
            y1, y2 = self.yields[idx], self.yields[idx + 1]
            return y1 + (y2 - y1) * (t - t1) / (t2 - t1)

    def _fit_nelson_siegel(self):
        """Fit Nelson-Siegel model to the yield curve."""
        from scipy.optimize import minimize

        def nelson_siegel(params, t):
            beta0, beta1, beta2, tau = params
            return beta0 + beta1 * (1 - np.exp(-t/tau)) / (t/tau) + beta2 * ((1 - np.exp(-t/tau)) / (t/tau) - np.exp(-t/tau))

        def objective(params):
            predicted = nelson_siegel(params, self.maturities)
            return np.sum((predicted - self.yields) ** 2)

        # Initial guess
        initial_params = [self.yields[0], -0.01, 0.01, 2.0]

        # Fit parameters
        result = minimize(objective, initial_params, bounds=[(None, None), (None, None), (None, None), (0.1, 10)])
        self.ns_params = result.x

        def interp_func(t):
            return nelson_siegel(self.ns_params, t)

        self._interp_func = interp_func

    def get_yield(self, maturity: float) -> float:
        """
        Get yield for a specific maturity using interpolation.

        Args:
            maturity: Time to maturity in years

        Returns:
            Interpolated yield rate
        """
        if callable(self._interp_func):
            return float(self._interp_func(maturity))
        else:
            return self._interp_func(maturity)

    def get_spot_rate(self, maturity: float) -> float:
        """
        Get spot rate for a specific maturity.
        For simplicity, assumes spot rate equals yield for same maturity.

        Args:
            maturity: Time to maturity in years

        Returns:
            Spot rate
        """
        return self.get_yield(maturity)

    def get_forward_rate(self, start_time: float, end_time: float) -> float:
        """
        Calculate forward rate between two time periods.

        Args:
            start_time: Start time in years
            end_time: End time in years

        Returns:
            Forward rate from start_time to end_time
        """
        if start_time >= end_time:
            raise ValueError("start_time must be less than end_time")

        spot_start = self.get_spot_rate(start_time)
        spot_end = self.get_spot_rate(end_time)

        # Forward rate calculation
        forward_rate = ((1 + spot_end) ** end_time / (1 + spot_start) ** start_time) ** (1 / (end_time - start_time)) - 1

        return forward_rate

    def get_discount_factor(self, maturity: float) -> float:
        """
        Calculate discount factor for a given maturity.

        Args:
            maturity: Time to maturity in years

        Returns:
            Discount factor
        """
        spot_rate = self.get_spot_rate(maturity)
        return (1 + spot_rate) ** (-maturity)

    def bootstrap_spot_rates(self) -> Dict[float, float]:
        """
        Bootstrap spot rates from coupon bond prices.
        This is a simplified version assuming the yields are already spot rates.

        Returns:
            Dictionary of maturity -> spot rate
        """
        spot_rates = {}
        for maturity, yield_rate in zip(self.maturities, self.yields):
            spot_rates[maturity] = yield_rate
        return spot_rates

    def plot_yield_curve(self, show_forward_rates: bool = False, **kwargs):
        """
        Plot the yield curve.

        Args:
            show_forward_rates: Whether to show forward rate curve
            **kwargs: Additional arguments for plotting
        """
        plt.figure(figsize=(10, 6))

        # Plot yield curve
        plt.plot(self.maturities, self.yields * 100, 'b-', label='Yield Curve', linewidth=2)

        if show_forward_rates:
            # Calculate and plot forward rates
            forward_maturities = []
            forward_rates = []

            for i in range(1, len(self.maturities)):
                start_t = self.maturities[i-1]
                end_t = self.maturities[i]
                forward_rate = self.get_forward_rate(start_t, end_t)
                forward_maturities.append((start_t + end_t) / 2)
                forward_rates.append(forward_rate)

            plt.plot(forward_maturities, np.array(forward_rates) * 100, 'r--', label='Forward Rates', linewidth=2)

        plt.xlabel('Maturity (Years)')
        plt.ylabel('Yield (%)')
        plt.title('Yield Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if 'save_path' in kwargs:
            plt.savefig(kwargs['save_path'], dpi=300, bbox_inches='tight')

        if kwargs.get('show', True):
            plt.show()

    @classmethod
    def from_zero_rates(cls,
                       maturities: List[float],
                       zero_rates: List[float]) -> 'YieldCurve':
        """
        Create YieldCurve from zero rates.

        Args:
            maturities: Time to maturity
            zero_rates: Zero coupon rates

        Returns:
            YieldCurve instance
        """
        return cls(maturities, zero_rates)

    @classmethod
    def from_par_rates(cls,
                      maturities: List[float],
                      par_rates: List[float],
                      coupon_freq: int = 2) -> 'YieldCurve':
        """
        Create YieldCurve from par bond yields.

        Args:
            maturities: Time to maturity
            par_rates: Par bond yields
            coupon_freq: Coupon payment frequency per year

        Returns:
            YieldCurve instance
        """
        # For simplicity, assume par rates are approximately equal to yields
        # In practice, this would require bootstrapping
        return cls(maturities, par_rates)

    def __repr__(self) -> str:
        return f"YieldCurve(maturities={len(self.maturities)}, range=({self.maturities[0]:.1f}-{self.maturities[-1]:.1f} years), method='{self.interpolation_method}')"
