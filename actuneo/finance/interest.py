"""
Interest Theory

Provides functions for interest calculations, present value, future value,
and various interest rate conversions.
"""

import numpy as np
from typing import Union, Optional


class InterestTheory:
    """
    A class for performing various interest rate calculations and
    time value of money computations.
    """

    def __init__(self, interest_rate: float = 0.05, compounding_frequency: int = 1):
        """
        Initialize InterestTheory with interest rate and compounding frequency.

        Args:
            interest_rate: Annual interest rate (decimal)
            compounding_frequency: Compounding frequency per year (1=annual, 2=semi-annual, etc.)
        """
        if interest_rate <= -1:
            raise ValueError("interest_rate must be greater than -1.0")
        if compounding_frequency < 1:
            raise ValueError("compounding_frequency must be at least 1")

        self.i = interest_rate
        self.m = compounding_frequency
        self.i_m = interest_rate / compounding_frequency  # Periodic interest rate
        self.v = 1 / (1 + interest_rate)  # Annual discount factor
        self.d = interest_rate / (1 + interest_rate)  # Annual effective rate of discount
        self.delta = float(np.log1p(interest_rate))  # Force of interest

    def future_value(self,
                    present_value: float,
                    periods: Union[int, float],
                    interest_rate: Optional[float] = None) -> float:
        """
        Calculate future value of a present amount.

        Args:
            present_value: Present value amount
            periods: Number of periods
            interest_rate: Override default interest rate

        Returns:
            Future value
        """
        i = interest_rate if interest_rate is not None else self.i
        return present_value * (1 + i) ** periods

    def present_value(self,
                     future_value: float,
                     periods: Union[int, float],
                     interest_rate: Optional[float] = None) -> float:
        """
        Calculate present value of a future amount.

        Args:
            future_value: Future value amount
            periods: Number of periods
            interest_rate: Override default interest rate

        Returns:
            Present value
        """
        i = interest_rate if interest_rate is not None else self.i
        return future_value * (1 + i) ** (-periods)

    def annuity_present_value(self,
                            payment: float,
                            periods: int,
                            interest_rate: Optional[float] = None,
                            immediate: bool = True) -> float:
        """
        Calculate present value of an annuity.

        Args:
            payment: Periodic payment amount
            periods: Number of periods
            interest_rate: Override default interest rate
            immediate: True for immediate annuity, False for annuity-due

        Returns:
            Present value of annuity
        """
        i = interest_rate if interest_rate is not None else self.i

        if i == 0:
            return payment * periods

        if immediate:
            # Immediate annuity: payments at end of each period
            return payment * ((1 - (1 + i) ** (-periods)) / i)
        else:
            # Annuity-due: payments at beginning of each period
            return payment * ((1 - (1 + i) ** (-periods)) / i) * (1 + i)

    def annuity_future_value(self,
                           payment: float,
                           periods: int,
                           interest_rate: Optional[float] = None,
                           immediate: bool = True) -> float:
        """
        Calculate future value of an annuity.

        Args:
            payment: Periodic payment amount
            periods: Number of periods
            interest_rate: Override default interest rate
            immediate: True for immediate annuity, False for annuity-due

        Returns:
            Future value of annuity
        """
        i = interest_rate if interest_rate is not None else self.i

        if i == 0:
            return payment * periods

        if immediate:
            # Immediate annuity
            return payment * (((1 + i) ** periods - 1) / i)
        else:
            # Annuity-due
            return payment * (((1 + i) ** periods - 1) / i) * (1 + i)

    def loan_payment(self,
                    principal: float,
                    periods: int,
                    interest_rate: Optional[float] = None) -> float:
        """
        Calculate periodic loan payment amount.

        Args:
            principal: Loan principal amount
            periods: Number of periods
            interest_rate: Override default interest rate

        Returns:
            Periodic payment amount
        """
        i = interest_rate if interest_rate is not None else self.i

        if i == 0:
            return principal / periods

        return principal * (i * (1 + i) ** periods) / ((1 + i) ** periods - 1)

    def loan_balance(self,
                    principal: float,
                    periods: int,
                    payments_made: int,
                    interest_rate: Optional[float] = None) -> float:
        """
        Calculate remaining loan balance after certain number of payments.

        Args:
            principal: Original loan principal
            periods: Total number of periods
            payments_made: Number of payments already made
            interest_rate: Override default interest rate

        Returns:
            Remaining balance
        """
        i = interest_rate if interest_rate is not None else self.i
        payment = self.loan_payment(principal, periods, i)

        # Future value of remaining payments
        remaining_periods = periods - payments_made
        remaining_annuity_pv = self.annuity_present_value(payment, remaining_periods, i, immediate=True)

        return remaining_annuity_pv

    def effective_annual_rate(self,
                            nominal_rate: float,
                            compounding_freq: int) -> float:
        """
        Convert nominal rate to effective annual rate.

        Args:
            nominal_rate: Nominal annual interest rate
            compounding_freq: Compounding frequency per year

        Returns:
            Effective annual rate
        """
        return (1 + nominal_rate / compounding_freq) ** compounding_freq - 1

    def nominal_rate(self,
                   effective_rate: float,
                   compounding_freq: int) -> float:
        """
        Convert effective annual rate to nominal rate.

        Args:
            effective_rate: Effective annual interest rate
            compounding_freq: Compounding frequency per year

        Returns:
            Nominal annual rate
        """
        return compounding_freq * ((1 + effective_rate) ** (1 / compounding_freq) - 1)

    def discount_rate(self, effective_rate: Optional[float] = None) -> float:
        """
        Convert an effective annual interest rate to the effective rate of discount d.

        Args:
            effective_rate: Effective annual interest rate (defaults to the instance rate)

        Returns:
            d = i / (1 + i)
        """
        i = effective_rate if effective_rate is not None else self.i
        return i / (1 + i)

    def force_of_interest(self, effective_rate: Optional[float] = None) -> float:
        """
        Convert an effective annual interest rate to the force of interest.

        Args:
            effective_rate: Effective annual interest rate (defaults to the instance rate)

        Returns:
            delta = ln(1 + i)
        """
        i = effective_rate if effective_rate is not None else self.i
        return float(np.log1p(i))

    def nominal_discount_rate(self,
                              effective_rate: float,
                              compounding_freq: int) -> float:
        """
        Convert effective annual interest rate to a nominal rate of discount d^(m).

        Args:
            effective_rate: Effective annual interest rate
            compounding_freq: Compounding frequency per year

        Returns:
            Nominal annual rate of discount convertible m times a year
        """
        return compounding_freq * (1 - (1 + effective_rate) ** (-1 / compounding_freq))

    def real_interest_rate(self,
                          nominal_rate: float,
                          inflation_rate: float) -> float:
        """
        Calculate real interest rate using Fisher equation.

        Args:
            nominal_rate: Nominal interest rate
            inflation_rate: Inflation rate

        Returns:
            Real interest rate
        """
        return (1 + nominal_rate) / (1 + inflation_rate) - 1

    def inflation_adjusted_value(self,
                               nominal_value: float,
                               inflation_rate: float,
                               periods: Union[int, float]) -> float:
        """
        Calculate inflation-adjusted (real) value.

        Args:
            nominal_value: Nominal value
            inflation_rate: Annual inflation rate
            periods: Number of periods

        Returns:
            Real value adjusted for inflation
        """
        return nominal_value / (1 + inflation_rate) ** periods
