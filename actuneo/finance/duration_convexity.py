"""
Duration and Convexity Measures

Provides calculations for duration, convexity, and related risk measures
for bonds and fixed income securities.
"""

import numpy as np
from typing import List, Union, Optional
from .yield_curve import YieldCurve


class DurationConvexity:
    """
    A class for calculating duration and convexity measures for bonds
    and fixed income portfolios.
    """

    def __init__(self, yield_curve: Optional[YieldCurve] = None):
        """
        Initialize DurationConvexity calculator.

        Args:
            yield_curve: YieldCurve instance for spot rate calculations
        """
        self.yield_curve = yield_curve

    def macaulay_duration(self,
                         cash_flows: List[float],
                         times: List[float],
                         yield_rate: float) -> float:
        """
        Calculate Macaulay duration.

        Args:
            cash_flows: List of cash flow amounts
            times: List of times when cash flows occur
            yield_rate: Yield rate (decimal)

        Returns:
            Macaulay duration
        """
        if len(cash_flows) != len(times):
            raise ValueError("cash_flows and times must have the same length")

        # Calculate present values
        pv_cash_flows = []
        for cf, t in zip(cash_flows, times):
            pv = cf * (1 + yield_rate) ** (-t)
            pv_cash_flows.append(pv)

        total_pv = sum(pv_cash_flows)

        if total_pv == 0:
            return 0.0

        # Calculate weighted average time
        duration = sum(pv * t for pv, t in zip(pv_cash_flows, times)) / total_pv

        return duration

    def modified_duration(self,
                         cash_flows: List[float],
                         times: List[float],
                         yield_rate: float) -> float:
        """
        Calculate modified duration.

        Args:
            cash_flows: List of cash flow amounts
            times: List of times when cash flows occur
            yield_rate: Yield rate (decimal)

        Returns:
            Modified duration
        """
        macaulay_dur = self.macaulay_duration(cash_flows, times, yield_rate)
        return macaulay_dur / (1 + yield_rate)

    def convexity(self,
                 cash_flows: List[float],
                 times: List[float],
                 yield_rate: float) -> float:
        """
        Calculate convexity.

        Args:
            cash_flows: List of cash flow amounts
            times: List of times when cash flows occur
            yield_rate: Yield rate (decimal)

        Returns:
            Convexity measure
        """
        if len(cash_flows) != len(times):
            raise ValueError("cash_flows and times must have the same length")

        # Calculate present values
        pv_cash_flows = []
        for cf, t in zip(cash_flows, times):
            pv = cf * (1 + yield_rate) ** (-t)
            pv_cash_flows.append(pv)

        total_pv = sum(pv_cash_flows)

        if total_pv == 0:
            return 0.0

        # Calculate convexity
        convexity = sum(pv * t * (t + 1) for pv, t in zip(pv_cash_flows, times)) / (total_pv * (1 + yield_rate) ** 2)

        return convexity

    def bond_duration(self,
                     face_value: float,
                     coupon_rate: float,
                     maturity: float,
                     yield_rate: float,
                     frequency: int = 2) -> float:
        """
        Calculate Macaulay duration for a standard bond.

        Args:
            face_value: Face value of the bond
            coupon_rate: Annual coupon rate (decimal)
            maturity: Time to maturity in years
            yield_rate: Yield to maturity (decimal)
            frequency: Coupon payment frequency per year

        Returns:
            Macaulay duration
        """
        # Generate cash flows
        periods = int(maturity * frequency)
        coupon_payment = face_value * coupon_rate / frequency

        cash_flows = [coupon_payment] * periods
        cash_flows[-1] += face_value  # Add face value to final payment

        times = [(i + 1) / frequency for i in range(periods)]

        return self.macaulay_duration(cash_flows, times, yield_rate)

    def bond_convexity(self,
                      face_value: float,
                      coupon_rate: float,
                      maturity: float,
                      yield_rate: float,
                      frequency: int = 2) -> float:
        """
        Calculate convexity for a standard bond.

        Args:
            face_value: Face value of the bond
            coupon_rate: Annual coupon rate (decimal)
            maturity: Time to maturity in years
            yield_rate: Yield to maturity (decimal)
            frequency: Coupon payment frequency per year

        Returns:
            Convexity
        """
        # Generate cash flows
        periods = int(maturity * frequency)
        coupon_payment = face_value * coupon_rate / frequency

        cash_flows = [coupon_payment] * periods
        cash_flows[-1] += face_value  # Add face value to final payment

        times = [(i + 1) / frequency for i in range(periods)]

        return self.convexity(cash_flows, times, yield_rate)

    def price_change_approximation(self,
                                 duration: float,
                                 convexity: float,
                                 yield_change: float,
                                 current_price: float) -> float:
        """
        Approximate price change using duration-convexity approximation.

        Args:
            duration: Modified duration
            convexity: Convexity measure
            yield_change: Change in yield (decimal)
            current_price: Current bond price

        Returns:
            Approximate price change
        """
        duration_effect = -duration * yield_change * current_price
        convexity_effect = 0.5 * convexity * yield_change ** 2 * current_price

        return duration_effect + convexity_effect

    def portfolio_duration(self,
                          positions: List[dict],
                          yield_rate: Optional[float] = None) -> float:
        """
        Calculate portfolio duration.

        Args:
            positions: List of position dictionaries with keys:
                      'cash_flows', 'times', 'weight' or 'market_value'
            yield_rate: Common yield rate for all positions

        Returns:
            Portfolio duration (modified duration)
        """
        total_value = 0
        weighted_duration = 0

        for position in positions:
            cash_flows = position['cash_flows']
            times = position['times']

            # Calculate present value of position
            if yield_rate is None:
                # Assume positions have their own yields
                position_yield = position.get('yield_rate', 0.05)
            else:
                position_yield = yield_rate

            pv = sum(cf * (1 + position_yield) ** (-t) for cf, t in zip(cash_flows, times))

            # Get weight
            if 'weight' in position:
                weight = position['weight']
                position_value = weight
            elif 'market_value' in position:
                position_value = position['market_value']
            else:
                position_value = pv

            total_value += position_value

            # Calculate position duration
            duration = self.modified_duration(cash_flows, times, position_yield)
            weighted_duration += position_value * duration

        if total_value == 0:
            return 0.0

        return weighted_duration / total_value

    def key_rate_duration(self,
                         cash_flows: List[float],
                         times: List[float],
                         key_rates: List[float],
                         yield_curve: YieldCurve) -> dict:
        """
        Calculate key rate durations.

        Args:
            cash_flows: List of cash flow amounts
            times: List of times when cash flows occur
            key_rates: List of key rate maturities
            yield_curve: YieldCurve instance

        Returns:
            Dictionary of key rate durations
        """
        base_price = self._calculate_pv_with_yield_curve(cash_flows, times, yield_curve)
        key_rate_durations = {}

        for key_rate in key_rates:
            # Shift yield curve at key rate
            shifted_curve = self._shift_yield_curve(yield_curve, key_rate, 0.0001)  # 1 bp shift
            shifted_price = self._calculate_pv_with_yield_curve(cash_flows, times, shifted_curve)

            # Calculate duration
            duration = -(shifted_price - base_price) / (base_price * 0.0001)
            key_rate_durations[key_rate] = duration

        return key_rate_durations

    def _calculate_pv_with_yield_curve(self,
                                      cash_flows: List[float],
                                      times: List[float],
                                      yield_curve: YieldCurve) -> float:
        """Calculate present value using yield curve."""
        pv = 0
        for cf, t in zip(cash_flows, times):
            discount_factor = yield_curve.get_discount_factor(t)
            pv += cf * discount_factor
        return pv

    def _shift_yield_curve(self,
                          yield_curve: YieldCurve,
                          shift_maturity: float,
                          shift_amount: float) -> YieldCurve:
        """Create a shifted copy of the yield curve."""
        # This is a simplified implementation
        new_maturities = yield_curve.maturities.copy()
        new_yields = yield_curve.yields.copy()

        # Find closest maturity and shift it
        idx = np.argmin(np.abs(new_maturities - shift_maturity))
        new_yields[idx] += shift_amount

        return YieldCurve(new_maturities, new_yields, yield_curve.interpolation_method)
