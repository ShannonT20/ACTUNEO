"""
Liability for Incurred Claims

The liability for incurred claims (LIC) is the obligation to pay claims for
insured events that have already happened, whether reported or not, together
with the related expenses. Under every IFRS 17 measurement model, including
the premium allocation approach, it is measured as the fulfilment cash flows
(IFRS 17.40(b), 59(b)):

- an estimate of the future cash flows,
- adjusted for the time value of money (discounted at current rates), plus
- a risk adjustment for non-financial risk.

Discounting may be omitted when the claims are expected to be paid within
one year of being incurred (IFRS 17.59(b)).
"""

import numpy as np
import pandas as pd
from typing import Optional, Sequence
from ._common import discount_factors


def risk_adjustment_confidence_level(model,
                                     confidence_level: float = 0.75,
                                     distribution: str = "lognormal") -> float:
    """
    Risk adjustment as the margin between a percentile of the reserve
    distribution and its mean (the confidence level technique).

    Args:
        model: A fitted :class:`~actuneo.loss_reserving.MackChainLadder` or
            :class:`~actuneo.loss_reserving.BootChainLadder`
        confidence_level: Probability that the reserve plus the risk
            adjustment is sufficient, for example 0.75
        distribution: Shape assumed for a Mack model, "lognormal" or "normal"

    Returns:
        Undiscounted risk adjustment for the total reserve (not less than zero)
    """
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    if hasattr(model, "total_ibnr_simulations"):
        quantile = float(model.total_ibnr_simulations.quantile(confidence_level))
        mean = model.mean_ibnr
    elif hasattr(model, "reserve_quantile"):
        quantile = float(model.reserve_quantile(confidence_level, distribution).loc["Total"].iloc[0])
        mean = model.total_ibnr
    else:
        raise TypeError("model must be a MackChainLadder or BootChainLadder")
    return max(0.0, quantile - mean)


def implied_confidence_level(model, risk_adjustment: float,
                             distribution: str = "lognormal") -> float:
    """
    Confidence level that a given risk adjustment corresponds to.

    IFRS 17.119 requires this disclosure when the risk adjustment is set by a
    technique other than the confidence level technique.

    Args:
        model: A fitted MackChainLadder or BootChainLadder
        risk_adjustment: Undiscounted risk adjustment for the total reserve
        distribution: Shape assumed for a Mack model, "lognormal" or "normal"

    Returns:
        Probability that the reserve plus the risk adjustment is sufficient
    """
    if hasattr(model, "total_ibnr_simulations"):
        simulations = model.total_ibnr_simulations.to_numpy()
        return float(np.mean(simulations <= model.mean_ibnr + risk_adjustment))
    if not hasattr(model, "total_mack_se"):
        raise TypeError("model must be a MackChainLadder or BootChainLadder")

    from scipy import stats
    mean, se = model.total_ibnr, model.total_mack_se
    target = mean + risk_adjustment
    if distribution == "normal":
        return float(stats.norm.cdf((target - mean) / se))
    if distribution != "lognormal":
        raise ValueError("distribution must be 'lognormal' or 'normal'")
    log_var = np.log1p((se / mean) ** 2)
    z = (np.log(target) - (np.log(mean) - log_var / 2)) / np.sqrt(log_var)
    return float(stats.norm.cdf(z))


def risk_adjustment_cost_of_capital(capital_requirements: Sequence[float],
                                    cost_of_capital: float = 0.06,
                                    discount_rate=0.0,
                                    periods_per_year: int = 1) -> float:
    """
    Risk adjustment as the present value of the cost of holding capital for
    non-financial risk until the claims are settled (the cost of capital
    technique).

    Args:
        capital_requirements: Capital held for non-financial risk at the
            start of each future period, the first being the valuation date
        cost_of_capital: Annual rate of return required on that capital
            above the risk-free rate
        discount_rate: Annual effective rate, or a yield curve object
        periods_per_year: Number of periods in a year

    Returns:
        Risk adjustment. The cost for each period is taken at its end.
    """
    capital = np.asarray(capital_requirements, dtype=float)
    years = np.arange(1, len(capital) + 1) / periods_per_year
    cost = capital * cost_of_capital / periods_per_year
    return float(np.sum(cost * discount_factors(discount_rate, years)))


class LiabilityForIncurredClaims:
    """
    Measurement of a liability for incurred claims.

    Attributes:
        cash_flows: Expected future payments by period
        undiscounted: Sum of the expected future payments
        present_value: Estimate of the present value of the future cash flows
        discounting: Undiscounted amount less present value
        risk_adjustment: Risk adjustment for non-financial risk
        total: Present value plus risk adjustment
    """

    def __init__(self,
                 cash_flows: Sequence[float],
                 discount_rate=0.0,
                 risk_adjustment: float = 0.0,
                 timing: float = 0.5,
                 periods_per_year: int = 1,
                 discount_risk_adjustment: bool = False):
        """
        Args:
            cash_flows: Expected payments in each future period, nearest first
            discount_rate: Annual effective rate, or a yield curve object
                with a ``get_discount_factor(years)`` method. Use 0 where
                claims are paid within a year and discounting is not applied.
            risk_adjustment: Risk adjustment for non-financial risk
            timing: When payments fall within each period: 0.5 for
                mid-period, 0 for the start, 1 for the end
            periods_per_year: Number of periods in a year
            discount_risk_adjustment: Scale the risk adjustment by the ratio
                of the present value to the undiscounted cash flows, for a
                risk adjustment that was calculated on undiscounted amounts
        """
        if not 0 <= timing <= 1:
            raise ValueError("timing must be between 0 and 1")
        if risk_adjustment < 0:
            raise ValueError("risk_adjustment must not be negative")

        self.cash_flows = np.atleast_1d(np.asarray(cash_flows, dtype=float))
        self.discount_rate = discount_rate
        self.timing = timing
        self.periods_per_year = periods_per_year

        years = (np.arange(len(self.cash_flows)) + timing) / periods_per_year
        self.undiscounted = float(self.cash_flows.sum())
        self.present_value = float(np.sum(self.cash_flows * discount_factors(discount_rate, years)))
        self.discounting = self.undiscounted - self.present_value

        if discount_risk_adjustment and self.undiscounted != 0:
            risk_adjustment = risk_adjustment * self.present_value / self.undiscounted
        self.risk_adjustment = float(risk_adjustment)
        self.total = self.present_value + self.risk_adjustment

    @classmethod
    def from_reserving(cls,
                       model,
                       discount_rate=0.0,
                       confidence_level: Optional[float] = None,
                       risk_adjustment: Optional[float] = None,
                       distribution: str = "lognormal",
                       timing: float = 0.5,
                       periods_per_year: int = 1) -> 'LiabilityForIncurredClaims':
        """
        Measure the liability from a fitted reserving model.

        The expected payments are the model's ``cash_flows()``. For a paid
        claims triangle these are all outstanding payments. For an incurred
        claims triangle they are only the development beyond the case
        estimates, so the case estimates must be added separately.

        Args:
            model: A fitted reserving model from :mod:`actuneo.loss_reserving`
            discount_rate: Annual effective rate, or a yield curve object
            confidence_level: Confidence level for the risk adjustment; the
                model must then be a MackChainLadder or BootChainLadder
            risk_adjustment: Risk adjustment on undiscounted amounts, as an
                alternative to ``confidence_level``
            distribution: Shape assumed for a Mack model
            timing: When payments fall within each period
            periods_per_year: Number of development periods in a year

        Returns:
            LiabilityForIncurredClaims with the risk adjustment discounted in
            proportion to the cash flows
        """
        if confidence_level is not None and risk_adjustment is not None:
            raise ValueError("Give either confidence_level or risk_adjustment, not both")

        if hasattr(model, "cash_flows"):
            flows = model.cash_flows().to_numpy()
            source = model
        elif hasattr(model, "chain_ladder"):
            flows = model.chain_ladder.cash_flows().to_numpy()
            source = model
        else:
            raise TypeError("model must be a fitted reserving model")

        if confidence_level is not None:
            risk_adjustment = risk_adjustment_confidence_level(source, confidence_level,
                                                               distribution)
        return cls(flows, discount_rate, risk_adjustment or 0.0, timing, periods_per_year,
                   discount_risk_adjustment=True)

    def summary(self) -> pd.Series:
        """The components of the liability."""
        return pd.Series({
            "undiscounted_cash_flows": self.undiscounted,
            "discounting": -self.discounting,
            "present_value_of_future_cash_flows": self.present_value,
            "risk_adjustment": self.risk_adjustment,
            "liability_for_incurred_claims": self.total,
        }, name="amount")

    def __repr__(self) -> str:
        return f"LiabilityForIncurredClaims\n{self.summary().to_string()}"


def lic_analysis_of_change(opening_cash_flows: Sequence[float],
                           closing_cash_flows_prior: Sequence[float],
                           closing_cash_flows_current: Sequence[float],
                           paid_prior: float,
                           paid_current: float,
                           opening_rate=0.0,
                           closing_rate=None,
                           opening_risk_adjustment: float = 0.0,
                           closing_risk_adjustment_prior: float = 0.0,
                           closing_risk_adjustment_current: float = 0.0,
                           timing: float = 0.5,
                           periods_per_year: int = 1) -> pd.Series:
    """
    Split the movement of a liability for incurred claims over one reporting
    period into the amounts IFRS 17 presents separately.

    - **Incurred claims** of the period: payments on claims incurred in the
      period plus the closing estimate for them.
    - **Adjustments to the liability for incurred claims**: the change in
      estimates for claims of earlier periods that is not due to interest
      (prior-year development).
    - **Insurance finance expenses**: interest accreted on the opening
      liability, and the effect of the change in discount rates.

    The amounts satisfy::

        closing = opening + incurred + adjustments + finance - paid

    Simplifications: interest is not accreted on claims incurred during the
    period, the opening rates are applied to the closing cash flows of prior
    claims without rolling the curve forward, and changes in the risk
    adjustment are all presented in the insurance service result
    (IFRS 17.81 permits not disaggregating them).

    Args:
        opening_cash_flows: Expected payments at the start of the period
        closing_cash_flows_prior: Expected payments at the end of the period
            for claims incurred before the period
        closing_cash_flows_current: Expected payments at the end of the
            period for claims incurred during the period
        paid_prior: Paid in the period on claims incurred before the period
        paid_current: Paid in the period on claims incurred during the period
        opening_rate: Discount rate (or curve) at the start of the period
        closing_rate: Discount rate (or curve) at the end of the period
            (defaults to the opening rate)
        opening_risk_adjustment: Risk adjustment at the start of the period
        closing_risk_adjustment_prior: Closing risk adjustment for claims
            incurred before the period
        closing_risk_adjustment_current: Closing risk adjustment for claims
            incurred during the period
        timing: When payments fall within each period
        periods_per_year: Number of periods in a year

    Returns:
        Series of the components, named to match the inputs of
        :class:`~actuneo.ifrs17.PAAGroup`
    """
    if closing_rate is None:
        closing_rate = opening_rate
    opening = np.atleast_1d(np.asarray(opening_cash_flows, dtype=float))
    prior = np.atleast_1d(np.asarray(closing_cash_flows_prior, dtype=float))
    current = np.atleast_1d(np.asarray(closing_cash_flows_current, dtype=float))

    def present_value(flows, rate, shift=0):
        years = (np.arange(len(flows)) - shift + timing) / periods_per_year
        return float(np.sum(flows * discount_factors(rate, years)))

    opening_pv = present_value(opening, opening_rate)
    # Opening estimate one period on: the first payment is made, the rest are one period nearer
    expected_end = opening[0] + present_value(opening[1:], opening_rate) if len(opening) else 0.0
    interest = expected_end - opening_pv

    prior_at_opening_rate = present_value(prior, opening_rate)
    prior_at_closing_rate = present_value(prior, closing_rate)
    current_pv = present_value(current, closing_rate)

    experience = prior_at_opening_rate + paid_prior - expected_end
    rate_change = prior_at_closing_rate - prior_at_opening_rate

    return pd.Series({
        "opening_lic_pv": opening_pv,
        "opening_lic_ra": float(opening_risk_adjustment),
        "incurred_claims": paid_current + current_pv,
        "incurred_risk_adjustment": float(closing_risk_adjustment_current),
        "adjustments_to_lic_pv": experience,
        "adjustments_to_lic_ra": float(closing_risk_adjustment_prior - opening_risk_adjustment),
        "interest_accreted": interest,
        "effect_of_rate_changes": rate_change,
        "finance_expenses": interest + rate_change,
        "claims_paid": float(paid_prior + paid_current),
        "closing_lic_pv": prior_at_closing_rate + current_pv,
        "closing_lic_ra": float(closing_risk_adjustment_prior + closing_risk_adjustment_current),
    }, name="amount")
