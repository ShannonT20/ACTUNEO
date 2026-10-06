"""
Cashflow Models of Standard Instruments

Each function returns the cashflows of a financial instrument from the point
of view of the investor (or lender): the price paid is negative and the
payments received are positive. The result is a
:class:`~actuneo.finance.Cashflows` object, so it can be valued, or its
yield found, directly.

Only instruments whose payments are fixed in amount and timing are covered.
Cashflows that depend on survival or on claims are in the life and loss
reserving modules.
"""

import numpy as np
from typing import Sequence
from .cashflows import Cashflows


def zero_coupon_bond(price: float, redemption: float, term: float) -> Cashflows:
    """A single payment at the end of the term, bought for a price now."""
    return Cashflows([0.0, term], [-price, redemption])


def fixed_interest_security(price: float, coupon_rate: float, term: float,
                            frequency: int = 2, redemption: float = 100.0,
                            nominal: float = 100.0) -> Cashflows:
    """Regular coupons in arrears and a redemption payment at the end of the term."""
    flows = Cashflows([0.0], [-price])
    flows.add_level(0.0, term, coupon_rate * nominal, p=frequency)
    return flows.add(term, redemption * nominal / 100.0)


def index_linked_security(price: float, times: Sequence[float],
                          nominal_amounts: Sequence[float], base_index: float,
                          index_values: Sequence[float]) -> Cashflows:
    """
    Payments that rise with a price index: each nominal amount is scaled by
    the index value applying to it over the base index.
    """
    amounts = (np.asarray(nominal_amounts, dtype=float)
               * np.asarray(index_values, dtype=float) / base_index)
    return Cashflows([0.0], [-price]) + Cashflows(times, amounts)


def annuity_certain(price: float, annual_amount: float, term: float, p: int = 1,
                    timing: str = "arrears") -> Cashflows:
    """Level payments for a fixed term in return for a single price."""
    return Cashflows([0.0], [-price]).add_level(0.0, term, annual_amount, p=p, timing=timing)


def interest_only_loan(amount: float, rate: float, term: float,
                       payments_per_year: int = 1) -> Cashflows:
    """
    Lender's cashflows on a loan where interest is paid each period and the
    whole amount is repaid at the end.

    Args:
        amount: Amount lent
        rate: Effective annual rate of interest
        term: Term in years
        payments_per_year: Number of interest payments a year
    """
    interest = amount * ((1 + rate) ** (1 / payments_per_year) - 1) * payments_per_year
    flows = Cashflows([0.0], [-amount]).add_level(0.0, term, interest, p=payments_per_year)
    return flows.add(term, amount)


def repayment_loan(amount: float, rate: float, term: float,
                   payments_per_year: int = 1) -> Cashflows:
    """
    Lender's cashflows on a loan repaid by level instalments of interest and
    capital (a repayment mortgage).
    """
    from .loans import Loan
    loan = Loan(amount, rate, term, payments_per_year)
    return Cashflows([0.0], [-amount]).add_level(0.0, term, loan.payment * payments_per_year,
                                                 p=payments_per_year)


def equity(price: float, dividends: Sequence[float], sale_proceeds: float = 0.0) -> Cashflows:
    """
    Dividends at the end of each year and the proceeds of sale after the
    last one. The amounts are expectations: dividends are not guaranteed.
    """
    dividends = np.atleast_1d(np.asarray(dividends, dtype=float))
    times = np.arange(1, len(dividends) + 1, dtype=float)
    flows = Cashflows([0.0], [-price]) + Cashflows(times, dividends)
    return flows.add(float(len(dividends)), sale_proceeds) if sale_proceeds else flows
