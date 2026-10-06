"""
Loan Schedules

Repayment of a loan by instalments: the level instalment, the split of each
instalment between interest and capital, the loan outstanding at any time,
and the annual percentage rate of charge (APR).
"""

import numpy as np
import pandas as pd
from typing import Sequence


def loan_schedule(principal: float, payments: Sequence[float],
                  rate_per_period: float) -> pd.DataFrame:
    """
    Schedule of a loan repaid by any series of payments in arrears.

    Args:
        principal: Amount of the loan
        payments: Payment at the end of each period
        rate_per_period: Effective rate of interest per period

    Returns:
        DataFrame indexed by payment number with the loan outstanding before
        the payment, the payment, its interest and capital elements, and the
        loan outstanding after it
    """
    payments = np.atleast_1d(np.asarray(payments, dtype=float))
    opening = np.zeros(len(payments))
    interest = np.zeros(len(payments))
    balance = float(principal)
    for k, payment in enumerate(payments):
        opening[k] = balance
        interest[k] = balance * rate_per_period
        balance = balance + interest[k] - payment
    capital = payments - interest
    return pd.DataFrame({
        "opening": opening,
        "payment": payments,
        "interest": interest,
        "capital": capital,
        "closing": opening - capital,
    }, index=pd.RangeIndex(1, len(payments) + 1, name="payment_number"))


class Loan:
    """
    A loan repaid by level instalments in arrears.

    Attributes:
        payment: Amount of each instalment
        rate_per_period: Effective rate of interest per payment period
        n_payments: Number of instalments
    """

    def __init__(self,
                 principal: float,
                 rate: float,
                 term: float,
                 payments_per_year: int = 1,
                 final_repayment: float = 0.0):
        """
        Args:
            principal: Amount of the loan
            rate: Effective annual rate of interest
            term: Term in years
            payments_per_year: Number of instalments a year
            final_repayment: Capital left to be repaid as a lump sum at the
                end of the term, on top of the last instalment (0 for a loan
                repaid in full by the instalments)
        """
        if principal <= 0 or term <= 0:
            raise ValueError("principal and term must be positive")
        if rate <= -1:
            raise ValueError("rate must be greater than -1")
        n = term * payments_per_year
        if abs(n - round(n)) > 1e-9:
            raise ValueError("the term must be a whole number of payment periods")

        self.principal = float(principal)
        self.rate = float(rate)
        self.term = float(term)
        self.payments_per_year = int(payments_per_year)
        self.final_repayment = float(final_repayment)
        self.n_payments = int(round(n))
        self.rate_per_period = (1 + rate) ** (1 / payments_per_year) - 1

        j, v = self.rate_per_period, 1 / (1 + self.rate_per_period)
        factor = self.n_payments if j == 0 else (1 - v ** self.n_payments) / j
        self.payment = (self.principal - self.final_repayment * v ** self.n_payments) / factor

    def schedule(self) -> pd.DataFrame:
        """Full schedule of the loan, one row per instalment."""
        return loan_schedule(self.principal, np.full(self.n_payments, self.payment),
                             self.rate_per_period)

    def outstanding(self, after_payment: int) -> float:
        """
        Loan outstanding immediately after an instalment (prospectively: the
        present value of the instalments still to be made).

        Args:
            after_payment: Number of instalments made (0 for the start)
        """
        k = self._check(after_payment, allow_zero=True)
        remaining = self.n_payments - k
        j, v = self.rate_per_period, 1 / (1 + self.rate_per_period)
        factor = remaining if j == 0 else (1 - v ** remaining) / j
        return float(self.payment * factor + self.final_repayment * v ** remaining)

    def interest_element(self, payment_number: int) -> float:
        """Interest in a given instalment."""
        k = self._check(payment_number)
        return self.outstanding(k - 1) * self.rate_per_period

    def capital_element(self, payment_number: int) -> float:
        """Capital repaid by a given instalment."""
        return self.payment - self.interest_element(payment_number)

    def capital_repaid(self, first: int, last: int) -> float:
        """Capital repaid by instalments ``first`` to ``last`` inclusive."""
        first, last = self._check(first), self._check(last)
        if last < first:
            raise ValueError("last must not be before first")
        return self.outstanding(first - 1) - self.outstanding(last)

    def interest_paid(self, first: int, last: int) -> float:
        """Interest paid in instalments ``first`` to ``last`` inclusive."""
        return (last - first + 1) * self.payment - self.capital_repaid(first, last)

    @property
    def total_interest(self) -> float:
        """Total interest paid over the term."""
        return self.n_payments * self.payment + self.final_repayment - self.principal

    def _check(self, k, allow_zero: bool = False) -> int:
        lowest = 0 if allow_zero else 1
        if k != int(k) or not lowest <= k <= self.n_payments:
            raise ValueError(f"payment number must be a whole number from {lowest} to "
                             f"{self.n_payments}")
        return int(k)

    def to_excel(self, path: str, title: str = "Loan schedule", currency: str = "") -> None:
        """
        Write the schedule to a formatted Excel workbook (needs openpyxl).

        Args:
            path: File name of the workbook, ending in .xlsx
            title: Heading for the report
            currency: Currency of the amounts
        """
        from ..utils import Sheet, write_report, DECIMAL_FORMAT, RATIO_FORMAT
        schedule = self.schedule()
        schedule.loc["Total"] = [np.nan, schedule["payment"].sum(), schedule["interest"].sum(),
                                 schedule["capital"].sum(), np.nan]
        unit = f"In {currency}" if currency else ""
        write_report(
            path,
            {"Schedule": Sheet(schedule, "Repayment schedule", unit,
                               number_format=DECIMAL_FORMAT, total_rows=["Total"],
                               chart={"kind": "line", "title": "Interest and capital",
                                      "columns": ["interest", "capital"]},
                               description="Interest, capital and loan outstanding by instalment")},
            report_title=title,
            report_subtitle=f"{self.n_payments} instalments, {self.payments_per_year} a year",
            highlights={"Loan": (self.principal, DECIMAL_FORMAT),
                        "Instalment": (self.payment, DECIMAL_FORMAT),
                        "Total interest": (self.total_interest, DECIMAL_FORMAT),
                        "Effective annual rate": (self.rate, RATIO_FORMAT)},
        )

    def __repr__(self) -> str:
        return (f"Loan(principal={self.principal:,.2f}, rate={self.rate:.4%}, "
                f"{self.n_payments} payments of {self.payment:,.2f})")


def annual_percentage_rate(principal: float, payment: float, n_payments: int,
                           payments_per_year: int = 12,
                           fees: float = 0.0) -> float:
    """
    Annual percentage rate of charge (APR): the effective annual rate at
    which the instalments repay the amount advanced.

    Quoted APRs are rounded down to one decimal place of a percentage point;
    this returns the unrounded rate.

    Args:
        principal: Amount of the loan
        payment: Level instalment paid in arrears
        n_payments: Number of instalments
        payments_per_year: Number of instalments a year
        fees: Charges paid at the start, deducted from the amount advanced
    """
    from scipy import optimize
    advanced = principal - fees
    if payment * n_payments <= advanced:
        raise ValueError("the instalments do not repay the loan")

    def difference(j):
        return payment * (1 - (1 + j) ** -n_payments) / j - advanced

    period_rate = optimize.brentq(difference, 1e-12, 10.0, xtol=1e-14)
    return float((1 + period_rate) ** payments_per_year - 1)


def flat_rate(principal: float, total_repaid: float, term: float) -> float:
    """
    Flat rate of interest: total interest divided by the loan and the term
    in years. It understates the true cost, since it ignores the capital
    repaid along the way.
    """
    return (total_repaid - principal) / (principal * term)
