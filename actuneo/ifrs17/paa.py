"""
Premium Allocation Approach (PAA)

The simplified IFRS 17 measurement model for groups of insurance contracts
with a coverage period of one year or less, or where it would give a result
not materially different from the general model (IFRS 17.53-59).

The carrying amount of a group is the sum of two liabilities:

**Liability for remaining coverage (LRC)**, for cover not yet provided. It is
rolled forward each period (IFRS 17.55)::

    closing LRC = opening LRC
                  + premiums received
                  - insurance acquisition cash flows paid
                  + amortisation of insurance acquisition cash flows
                  - insurance revenue

plus a loss component if the group is onerous (IFRS 17.57-58).

**Liability for incurred claims (LIC)**, for claims that have happened,
measured as the present value of the future cash flows plus a risk
adjustment (IFRS 17.59(b)).

Liabilities are positive. In the profit or loss tables income is positive
and expenses are negative.
"""

import numpy as np
import pandas as pd
from typing import Optional, Sequence
from ._common import as_series, period_labels


def paa_eligibility(coverage_period_months: float,
                    lrc_paa: Optional[Sequence[float]] = None,
                    lrc_general_model: Optional[Sequence[float]] = None,
                    threshold: float = 0.05) -> dict:
    """
    Test whether a group of contracts may be measured under the PAA
    (IFRS 17.53).

    A group qualifies automatically if every contract has a coverage period
    of one year or less. Otherwise the PAA may be used only if the liability
    for remaining coverage it produces is not expected to differ materially
    from the general model, which is tested by projecting both.

    Args:
        coverage_period_months: Longest coverage period of the contracts
        lrc_paa: Projected liability for remaining coverage under the PAA at
            each future reporting date
        lrc_general_model: The same projection under the general model
        threshold: Largest relative difference regarded as immaterial. This
            is a judgement for the entity and its auditors.

    Returns:
        Dictionary with ``eligible``, the ``basis`` for the conclusion and,
        where projections were compared, the ``max_relative_difference``
    """
    if coverage_period_months <= 0:
        raise ValueError("coverage_period_months must be positive")
    if coverage_period_months <= 12:
        return {"eligible": True,
                "basis": "Coverage period of one year or less (IFRS 17.53(b))",
                "max_relative_difference": None}
    if lrc_paa is None or lrc_general_model is None:
        return {"eligible": False,
                "basis": "Coverage period exceeds one year; projections of the liability "
                         "for remaining coverage under both models are needed (IFRS 17.53(a))",
                "max_relative_difference": None}

    paa = np.asarray(lrc_paa, dtype=float)
    general = np.asarray(lrc_general_model, dtype=float)
    if paa.shape != general.shape:
        raise ValueError("lrc_paa and lrc_general_model must have the same length")
    scale = np.where(np.abs(general) > 0, np.abs(general), np.nan)
    relative = np.abs(paa - general) / scale
    largest = float(np.nanmax(relative)) if np.any(~np.isnan(relative)) else 0.0
    eligible = largest <= threshold
    verdict = "does not differ" if eligible else "differs"
    return {"eligible": bool(eligible),
            "basis": f"Liability for remaining coverage {verdict} materially from the "
                     f"general model at a threshold of {threshold:.1%} (IFRS 17.53(a))",
            "max_relative_difference": largest}


class PAAGroup:
    """
    A group of insurance contracts measured under the premium allocation
    approach, followed over its reporting periods from initial recognition.

    Every per-period input is a sequence with one value per reporting period;
    a single number is used for every period.
    """

    def __init__(self,
                 premiums_received: Sequence[float],
                 insurance_revenue: Optional[Sequence[float]] = None,
                 expected_premium: Optional[float] = None,
                 revenue_pattern: Optional[Sequence[float]] = None,
                 acquisition_cash_flows=None,
                 expense_acquisition_cash_flows: bool = False,
                 fulfilment_cash_flows_remaining: Optional[Sequence[float]] = None,
                 incurred_claims=None,
                 incurred_risk_adjustment=None,
                 claims_paid=None,
                 closing_lic_pv: Optional[Sequence[float]] = None,
                 closing_lic_ra: Optional[Sequence[float]] = None,
                 finance_expenses=None,
                 finance_expenses_in_oci=None,
                 other_insurance_service_expenses=None,
                 opening_lrc: float = 0.0,
                 opening_loss_component: float = 0.0,
                 opening_lic_pv: float = 0.0,
                 opening_lic_ra: float = 0.0,
                 opening_deferred_acquisition_cash_flows: float = 0.0,
                 lrc_discount_rate: Optional[float] = None,
                 periods_per_year: int = 1,
                 periods: Optional[Sequence] = None,
                 name: str = "Group",
                 portfolio: Optional[str] = None):
        """
        Args:
            premiums_received: Premiums received in each period
            insurance_revenue: Insurance revenue of each period. If omitted
                it is ``expected_premium`` spread by ``revenue_pattern``.
            expected_premium: Total premium still to be recognised as revenue
                from the first period onwards: the expected premium receipts
                of a new group, or the unearned premium of a group already in
                force (defaults to the total of the revenue given, or of the
                premiums received)
            revenue_pattern: Share of the cover provided in each period, by
                the passage of time or by the expected pattern of incurred
                claims (defaults to equal periods)
            acquisition_cash_flows: Insurance acquisition cash flows paid in
                each period (commission and other directly attributable costs)
            expense_acquisition_cash_flows: Expense acquisition cash flows
                when incurred instead of deferring them, allowed when the
                coverage period is one year or less (IFRS 17.59(a))
            fulfilment_cash_flows_remaining: Fulfilment cash flows for the
                remaining coverage at each period end (expected future claims
                and expenses plus risk adjustment). Where they exceed the
                liability for remaining coverage the group is onerous and a
                loss component is recognised. Omit when facts and
                circumstances do not indicate that the group is onerous.
            incurred_claims: Claims and other insurance service expenses
                incurred in each period, at present value where discounted
            incurred_risk_adjustment: Risk adjustment on the claims incurred
                in each period
            claims_paid: Claims and other insurance service expenses paid
            closing_lic_pv: Present value of the future cash flows of the
                liability for incurred claims at each period end. If omitted
                it follows from incurred claims, payments and finance
                expenses, with no change in estimates.
            closing_lic_ra: Risk adjustment of the liability for incurred
                claims at each period end. If omitted, the risk adjustment
                on incurred claims accumulates without release.
            finance_expenses: Insurance finance expenses of each period
                (interest accreted and the effect of discount rate changes)
            finance_expenses_in_oci: The part of the finance expenses that is
                presented in other comprehensive income, when the entity
                chooses to disaggregate them (IFRS 17.88(b)). For incurred
                claims under the PAA this is the effect of changes in
                discount rates since the claims were incurred (IFRS 17.B133).
            other_insurance_service_expenses: Directly attributable expenses
                other than claims and acquisition costs (policy
                administration, claims handling), taken as paid when incurred
            opening_lrc: Liability for remaining coverage, excluding any loss
                component, at the start of the first period, for a group
                already in force
            opening_loss_component: Loss component at the start
            opening_lic_pv: Present value of the future cash flows of the
                liability for incurred claims at the start
            opening_lic_ra: Risk adjustment of the liability for incurred
                claims at the start
            opening_deferred_acquisition_cash_flows: Acquisition cash flows
                included in the opening liability for remaining coverage that
                have not yet been amortised
            lrc_discount_rate: Annual effective discount rate fixed at
                initial recognition, used to adjust the liability for
                remaining coverage for the time value of money when the group
                has a significant financing component (IFRS 17.56). Leave
                unset when premiums are due within a year of the cover they
                pay for, where no adjustment is required. When set, interest
                accretes on the liability each period, is reported as an
                insurance finance expense, and revenue is the premium plus
                that interest. Premiums are taken as received at the start of
                each period and revenue as earned at its end, and the periods
                given must cover the whole of the remaining cover.
            periods_per_year: Number of reporting periods in a year, used
                with ``lrc_discount_rate``
            periods: Labels of the reporting periods
            name: Name of the group
            portfolio: Portfolio the group belongs to. Groups of a portfolio
                are presented net in the statement of financial position.
        """
        premiums = np.atleast_1d(np.asarray(premiums_received, dtype=float))
        if premiums.ndim != 1 or premiums.size == 0:
            raise ValueError("premiums_received must be a sequence with one value per period")
        n = premiums.size

        self.name = name
        self.portfolio = portfolio if portfolio is not None else name
        self.periods = period_labels(periods, n)
        self.expense_acquisition_cash_flows = bool(expense_acquisition_cash_flows)

        self.premiums_received = as_series(premiums, n, "premiums_received")
        self.acquisition_cash_flows = as_series(acquisition_cash_flows, n, "acquisition_cash_flows")

        if insurance_revenue is not None:
            if revenue_pattern is not None:
                raise ValueError("Give either insurance_revenue or revenue_pattern, not both")
            revenue = as_series(insurance_revenue, n, "insurance_revenue")
            total_expected = float(expected_premium) if expected_premium is not None \
                else float(revenue.sum())
        else:
            total_expected = float(expected_premium) if expected_premium is not None \
                else float(premiums.sum())
            if revenue_pattern is None:
                pattern = np.full(n, 1.0 / n)
            else:
                pattern = as_series(revenue_pattern, n, "revenue_pattern")
                if np.any(pattern < 0) or pattern.sum() <= 0:
                    raise ValueError("revenue_pattern must be non-negative with a positive total")
                if pattern.sum() > 1 + 1e-9:
                    pattern = pattern / pattern.sum()
            revenue = total_expected * pattern
        # Significant financing component: revenue is scaled so that premiums
        # accumulated at the locked-in rate are released in full over the cover
        self.lrc_interest = np.zeros(n)
        if lrc_discount_rate:
            if insurance_revenue is not None:
                raise ValueError("Use revenue_pattern, not insurance_revenue, with "
                                 "lrc_discount_rate")
            if lrc_discount_rate <= -1:
                raise ValueError("lrc_discount_rate must be greater than -1")
            if revenue.sum() <= 0 or abs(revenue.sum() - total_expected) > 1e-9 * abs(
                    total_expected):
                raise ValueError("With lrc_discount_rate the periods must cover the whole of "
                                 "the remaining cover")
            rate = (1 + lrc_discount_rate) ** (1 / periods_per_year) - 1
            v = 1 / (1 + rate)
            weights = revenue / revenue.sum()
            opening_gross = float(opening_lrc) + float(opening_deferred_acquisition_cash_flows)
            funded = opening_gross + np.sum(self.premiums_received * v ** np.arange(n))
            revenue = weights * funded / np.sum(weights * v ** np.arange(1, n + 1))
            balance = opening_gross
            for t in range(n):
                self.lrc_interest[t] = (balance + self.premiums_received[t]) * rate
                balance += self.premiums_received[t] + self.lrc_interest[t] - revenue[t]
            total_expected = float(revenue.sum())
        self.insurance_revenue = revenue
        self.expected_premium = total_expected

        self.opening_lrc = float(opening_lrc)
        self.opening_loss_component = float(opening_loss_component)
        self.opening_lic_pv = float(opening_lic_pv)
        self.opening_lic_ra = float(opening_lic_ra)
        self.opening_deferred_acquisition_cash_flows = float(
            opening_deferred_acquisition_cash_flows
        )
        if self.opening_loss_component < 0 or self.opening_lic_ra < 0:
            raise ValueError("opening loss component and risk adjustment must not be negative")

        # Unamortised acquisition cash flows are spread over the cover still to
        # be given, in proportion to the revenue of each period
        self.amortisation = np.zeros(n)
        self.deferred_acquisition_cash_flows = np.zeros(n)
        self.unearned_premium = total_expected - np.cumsum(revenue)
        if not self.expense_acquisition_cash_flows:
            unamortised = self.opening_deferred_acquisition_cash_flows
            for t in range(n):
                unamortised += self.acquisition_cash_flows[t]
                remaining = total_expected - revenue[:t].sum()
                share = 1.0 if remaining <= revenue[t] or remaining <= 0 \
                    else revenue[t] / remaining
                self.amortisation[t] = unamortised * share
                unamortised -= self.amortisation[t]
                self.deferred_acquisition_cash_flows[t] = unamortised
        deferred = np.zeros(n) if self.expense_acquisition_cash_flows else self.acquisition_cash_flows

        movement = (self.premiums_received - deferred - revenue + self.amortisation
                    + self.lrc_interest)
        self.lrc_excluding_loss_component = self.opening_lrc + np.cumsum(movement)

        # Loss component: fulfilment cash flows in excess of the carrying amount
        if fulfilment_cash_flows_remaining is None:
            self.loss_component = np.zeros(n)
        else:
            fulfilment = as_series(fulfilment_cash_flows_remaining, n,
                                   "fulfilment_cash_flows_remaining")
            self.loss_component = np.maximum(0.0, fulfilment - self.lrc_excluding_loss_component)
        self.loss_on_onerous_contracts = np.diff(
            np.concatenate(([self.opening_loss_component], self.loss_component))
        )

        # Liability for incurred claims
        self.incurred_claims = as_series(incurred_claims, n, "incurred_claims")
        self.incurred_risk_adjustment = as_series(incurred_risk_adjustment, n,
                                                  "incurred_risk_adjustment")
        self.claims_paid = as_series(claims_paid, n, "claims_paid")
        self.finance_expenses = as_series(finance_expenses, n, "finance_expenses")
        self.finance_expenses_in_oci = as_series(finance_expenses_in_oci, n,
                                                 "finance_expenses_in_oci")
        self.other_insurance_service_expenses = as_series(
            other_insurance_service_expenses, n, "other_insurance_service_expenses"
        )

        expected_pv = self.opening_lic_pv + np.cumsum(
            self.incurred_claims + self.finance_expenses - self.claims_paid
        )
        if closing_lic_pv is None:
            self.lic_present_value = expected_pv
            self.adjustments_to_lic_pv = np.zeros(n)
        else:
            self.lic_present_value = as_series(closing_lic_pv, n, "closing_lic_pv")
            opening = np.concatenate(([self.opening_lic_pv], self.lic_present_value[:-1]))
            self.adjustments_to_lic_pv = (self.lic_present_value - opening - self.incurred_claims
                                          - self.finance_expenses + self.claims_paid)

        if closing_lic_ra is None:
            self.lic_risk_adjustment = self.opening_lic_ra + np.cumsum(
                self.incurred_risk_adjustment
            )
            self.adjustments_to_lic_ra = np.zeros(n)
        else:
            self.lic_risk_adjustment = as_series(closing_lic_ra, n, "closing_lic_ra")
            opening = np.concatenate(([self.opening_lic_ra], self.lic_risk_adjustment[:-1]))
            self.adjustments_to_lic_ra = (self.lic_risk_adjustment - opening
                                          - self.incurred_risk_adjustment)

    # ------------------------------------------------------------------
    # Balances
    # ------------------------------------------------------------------
    @property
    def n_periods(self) -> int:
        return len(self.periods)

    @property
    def lrc(self) -> np.ndarray:
        """Liability for remaining coverage including the loss component."""
        return self.lrc_excluding_loss_component + self.loss_component

    @property
    def lic(self) -> np.ndarray:
        """Liability for incurred claims: present value plus risk adjustment."""
        return self.lic_present_value + self.lic_risk_adjustment

    @property
    def carrying_amount(self) -> np.ndarray:
        """Net carrying amount of the group; negative when the group is an asset."""
        return self.lrc + self.lic

    def _frame(self, data: dict) -> pd.DataFrame:
        # Adding zero turns any negative zero into a plain zero
        return pd.DataFrame(data, index=pd.Index(self.periods, name="period")) + 0.0

    @staticmethod
    def _opening(closing: np.ndarray, first: float = 0.0) -> np.ndarray:
        return np.concatenate(([first], closing[:-1]))

    @property
    def opening_carrying_amount(self) -> float:
        """Net carrying amount of the group at the start of the first period."""
        return (self.opening_lrc + self.opening_loss_component
                + self.opening_lic_pv + self.opening_lic_ra)

    def supplementary_position(self) -> pd.DataFrame:
        """
        The carrying amount analysed into the balances insurers reported
        under IFRS 4, as supplementary information.

        IFRS 17 presents a single carrying amount for a group of contracts.
        The familiar components can still be shown to explain it::

            LRC = unearned premium - premiums receivable
                  - deferred acquisition costs + loss component
            LIC = outstanding claims (discounted) + risk adjustment

        Premiums receivable are the premiums for cover already recognised
        that have not yet been received. Where premiums must be paid before
        cover starts they are nil or negative (premiums received in advance).
        """
        receivable = (self.unearned_premium - self.deferred_acquisition_cash_flows
                      - self.lrc_excluding_loss_component)
        return self._frame({
            "unearned_premium": self.unearned_premium,
            "premiums_receivable": -receivable,
            "deferred_acquisition_costs": -self.deferred_acquisition_cash_flows,
            "loss_component": self.loss_component,
            "liability_for_remaining_coverage": self.lrc,
            "outstanding_claims": self.lic_present_value,
            "risk_adjustment": self.lic_risk_adjustment,
            "liability_for_incurred_claims": self.lic,
            "carrying_amount": self.carrying_amount,
        })

    def financial_position(self) -> pd.DataFrame:
        """Components of the carrying amount at each period end."""
        return self._frame({
            "lrc_excluding_loss_component": self.lrc_excluding_loss_component,
            "loss_component": self.loss_component,
            "lic_present_value": self.lic_present_value,
            "lic_risk_adjustment": self.lic_risk_adjustment,
            "carrying_amount": self.carrying_amount,
        })

    # ------------------------------------------------------------------
    # Roll-forwards
    # ------------------------------------------------------------------
    def lrc_rollforward(self) -> pd.DataFrame:
        """Movement of the liability for remaining coverage in each period."""
        deferred = (np.zeros(self.n_periods) if self.expense_acquisition_cash_flows
                    else self.acquisition_cash_flows)
        return self._frame({
            "opening": self._opening(self.lrc, self.opening_lrc + self.opening_loss_component),
            "premiums_received": self.premiums_received,
            "acquisition_cash_flows": -deferred,
            "amortisation_of_acquisition_cash_flows": self.amortisation,
            "interest_accreted": self.lrc_interest,
            "insurance_revenue": -self.insurance_revenue,
            "loss_component_movement": self.loss_on_onerous_contracts,
            "closing": self.lrc,
        })

    def lic_rollforward(self) -> pd.DataFrame:
        """Movement of the liability for incurred claims in each period."""
        return self._frame({
            "opening": self._opening(self.lic, self.opening_lic_pv + self.opening_lic_ra),
            "incurred_claims": self.incurred_claims + self.incurred_risk_adjustment,
            "adjustments_to_lic": self.adjustments_to_lic_pv + self.adjustments_to_lic_ra,
            "finance_expenses": self.finance_expenses,
            "claims_paid": -self.claims_paid,
            "closing": self.lic,
        })

    # ------------------------------------------------------------------
    # Performance and cash
    # ------------------------------------------------------------------
    def profit_or_loss(self) -> pd.DataFrame:
        """
        Amounts recognised in profit or loss in each period, and insurance
        finance expenses presented in other comprehensive income. Income is
        positive and expenses negative.

        When acquisition cash flows are expensed as incurred they appear in
        the ``amortisation_of_acquisition_cash_flows`` column.
        """
        acquisition = (self.acquisition_cash_flows if self.expense_acquisition_cash_flows
                       else self.amortisation)
        table = self._frame({
            "insurance_revenue": self.insurance_revenue,
            "incurred_claims_and_other_expenses": -(self.incurred_claims
                                                    + self.incurred_risk_adjustment
                                                    + self.other_insurance_service_expenses),
            "amortisation_of_acquisition_cash_flows": -acquisition,
            "losses_on_onerous_contracts": -self.loss_on_onerous_contracts,
            "adjustments_to_lic": -(self.adjustments_to_lic_pv + self.adjustments_to_lic_ra),
        })
        table["insurance_service_expenses"] = table.iloc[:, 1:].sum(axis=1)
        table["insurance_service_result"] = (table["insurance_revenue"]
                                             + table["insurance_service_expenses"])
        table["insurance_finance_expenses"] = -(self.finance_expenses + self.lrc_interest
                                                - self.finance_expenses_in_oci)
        table["insurance_finance_expenses_oci"] = -self.finance_expenses_in_oci
        return table

    def cash_flows(self) -> pd.DataFrame:
        """Cash received (positive) and paid (negative) in each period."""
        table = self._frame({
            "premiums_received": self.premiums_received,
            "claims_paid": -(self.claims_paid + self.other_insurance_service_expenses),
            "acquisition_cash_flows": -self.acquisition_cash_flows,
        })
        table["net_cash_flow"] = table.sum(axis=1)
        return table

    def reconciliation(self, period=None) -> pd.DataFrame:
        """
        Reconciliation of the opening and closing liabilities for a period,
        analysed by remaining coverage and incurred claims (IFRS 17.100).

        Args:
            period: Label of the period (defaults to the last period)

        Returns:
            DataFrame of movements. Increases in the liability are positive.
        """
        k = self.n_periods - 1 if period is None else self.periods.index(period)
        deferred = 0.0 if self.expense_acquisition_cash_flows else self.acquisition_cash_flows[k]
        columns = ["lrc_excluding_loss_component", "loss_component", "lic_present_value",
                   "lic_risk_adjustment"]

        balances = [self.lrc_excluding_loss_component, self.loss_component,
                    self.lic_present_value, self.lic_risk_adjustment]
        first = [self.opening_lrc, self.opening_loss_component,
                 self.opening_lic_pv, self.opening_lic_ra]
        other = self.other_insurance_service_expenses[k]
        rows = {
            "Opening liabilities": [b[k - 1] if k > 0 else f for b, f in zip(balances, first)],
            "Insurance revenue": [-self.insurance_revenue[k], 0, 0, 0],
            "Incurred claims and other insurance service expenses":
                [0, 0, self.incurred_claims[k] + other, self.incurred_risk_adjustment[k]],
            "Amortisation of insurance acquisition cash flows": [self.amortisation[k], 0, 0, 0],
            "Losses on onerous contracts and reversals":
                [0, self.loss_on_onerous_contracts[k], 0, 0],
            "Adjustments to liabilities for incurred claims":
                [0, 0, self.adjustments_to_lic_pv[k], self.adjustments_to_lic_ra[k]],
        }
        table = pd.DataFrame(rows, index=columns, dtype=float).T
        service_rows = list(rows)[1:]
        table.loc["Insurance service result"] = table.loc[service_rows].sum()
        table.loc["Net finance expenses from insurance contracts"] = \
            [self.lrc_interest[k], 0, self.finance_expenses[k], 0]
        table.loc["Total changes in profit or loss"] = table.loc[
            ["Insurance service result", "Net finance expenses from insurance contracts"]
        ].sum()
        table.loc["Premiums received"] = [self.premiums_received[k], 0, 0, 0]
        table.loc["Claims and other insurance service expenses paid"] = \
            [0, 0, -(self.claims_paid[k] + other), 0]
        table.loc["Insurance acquisition cash flows"] = [-deferred, 0, 0, 0]
        cash_rows = ["Premiums received", "Claims and other insurance service expenses paid",
                     "Insurance acquisition cash flows"]
        table.loc["Total cash flows"] = table.loc[cash_rows].sum()
        table.loc["Closing liabilities"] = [b[k] for b in balances]
        table["total"] = table.sum(axis=1)
        table.columns.name = self.periods[k]
        return table + 0.0

    def __repr__(self) -> str:
        return (f"PAAGroup(name='{self.name}', periods={self.n_periods})\n"
                f"{self.financial_position().to_string()}")


class PAAReinsuranceHeld:
    """
    A group of reinsurance contracts held, measured under the premium
    allocation approach (IFRS 17.69-70A).

    It mirrors an insurance group: the cedant holds an asset for remaining
    coverage (reinsurance premiums paid for cover not yet received) and an
    asset for incurred claims (recoveries due on claims that have happened).
    Assets are positive.
    """

    def __init__(self,
                 premiums_paid: Sequence[float],
                 allocation_of_premiums: Optional[Sequence[float]] = None,
                 expected_premium: Optional[float] = None,
                 allocation_pattern: Optional[Sequence[float]] = None,
                 recoveries_incurred=None,
                 recoveries_risk_adjustment=None,
                 recoveries_received=None,
                 closing_aic_pv: Optional[Sequence[float]] = None,
                 closing_aic_ra: Optional[Sequence[float]] = None,
                 finance_income=None,
                 finance_income_in_oci=None,
                 loss_recovery_component: Optional[Sequence[float]] = None,
                 opening_arc: float = 0.0,
                 opening_loss_recovery_component: float = 0.0,
                 opening_aic_pv: float = 0.0,
                 opening_aic_ra: float = 0.0,
                 periods: Optional[Sequence] = None,
                 name: str = "Reinsurance",
                 portfolio: Optional[str] = None):
        """
        Args:
            premiums_paid: Reinsurance premiums paid in each period, net of
                any commission received from the reinsurer
            allocation_of_premiums: Reinsurance premiums expensed in each
                period for the cover received. If omitted it is
                ``expected_premium`` spread by ``allocation_pattern``.
            expected_premium: Total expected reinsurance premiums
            allocation_pattern: Share of the cover received in each period
            recoveries_incurred: Recoveries on claims incurred in each period
            recoveries_risk_adjustment: Risk adjustment ceded on those claims
            recoveries_received: Recoveries received from the reinsurer
            closing_aic_pv: Present value of the future cash flows of the
                asset for incurred claims at each period end
            closing_aic_ra: Risk adjustment of the asset for incurred claims
                at each period end
            finance_income: Finance income from the reinsurance contracts
            finance_income_in_oci: The part of the finance income presented
                in other comprehensive income
            opening_arc: Asset for remaining coverage, excluding any
                loss-recovery component, at the start of the first period
            opening_loss_recovery_component: Loss-recovery component at the start
            opening_aic_pv: Present value of the future cash flows of the
                asset for incurred claims at the start
            opening_aic_ra: Risk adjustment of the asset for incurred claims
                at the start
            loss_recovery_component: Loss-recovery component at each period
                end, where the reinsurance covers onerous underlying
                contracts: the loss on those contracts multiplied by the
                percentage of claims expected to be recovered (IFRS 17.66A-B)
            periods: Labels of the reporting periods
            name: Name of the group
            portfolio: Portfolio the group belongs to
        """
        mirror = PAAGroup(
            premiums_received=premiums_paid,
            insurance_revenue=allocation_of_premiums,
            expected_premium=expected_premium,
            revenue_pattern=allocation_pattern,
            incurred_claims=recoveries_incurred,
            incurred_risk_adjustment=recoveries_risk_adjustment,
            claims_paid=recoveries_received,
            closing_lic_pv=closing_aic_pv,
            closing_lic_ra=closing_aic_ra,
            finance_expenses=finance_income,
            finance_expenses_in_oci=finance_income_in_oci,
            opening_lrc=opening_arc,
            opening_lic_pv=opening_aic_pv,
            opening_lic_ra=opening_aic_ra,
            periods=periods,
            name=name,
            portfolio=portfolio,
        )
        self._mirror = mirror
        self.name = name
        self.portfolio = mirror.portfolio
        self.periods = mirror.periods
        n = mirror.n_periods

        self.premiums_paid = mirror.premiums_received
        self.allocation_of_premiums = mirror.insurance_revenue
        self.recoveries_incurred = mirror.incurred_claims
        self.recoveries_risk_adjustment = mirror.incurred_risk_adjustment
        self.recoveries_received = mirror.claims_paid
        self.finance_income = mirror.finance_expenses
        self.finance_income_in_oci = mirror.finance_expenses_in_oci
        self.opening_loss_recovery_component = float(opening_loss_recovery_component)
        self.adjustments_to_aic = mirror.adjustments_to_lic_pv + mirror.adjustments_to_lic_ra

        self.arc_excluding_loss_recovery = mirror.lrc_excluding_loss_component
        self.loss_recovery_component = as_series(loss_recovery_component, n,
                                                 "loss_recovery_component")
        if np.any(self.loss_recovery_component < 0):
            raise ValueError("loss_recovery_component must not be negative")
        self.loss_recovery_income = np.diff(
            np.concatenate(([self.opening_loss_recovery_component], self.loss_recovery_component))
        )
        self.aic_present_value = mirror.lic_present_value
        self.aic_risk_adjustment = mirror.lic_risk_adjustment

    @property
    def n_periods(self) -> int:
        return len(self.periods)

    @property
    def opening_carrying_amount(self) -> float:
        """Net carrying amount of the group at the start of the first period."""
        return self._mirror.opening_carrying_amount + self.opening_loss_recovery_component

    @property
    def carrying_amount(self) -> np.ndarray:
        """Net carrying amount of the group as an asset; negative when a liability."""
        return (self.arc_excluding_loss_recovery + self.loss_recovery_component
                + self.aic_present_value + self.aic_risk_adjustment)

    def _frame(self, data: dict) -> pd.DataFrame:
        # Adding zero turns any negative zero into a plain zero
        return pd.DataFrame(data, index=pd.Index(self.periods, name="period")) + 0.0

    def financial_position(self) -> pd.DataFrame:
        """Components of the carrying amount at each period end."""
        return self._frame({
            "arc_excluding_loss_recovery": self.arc_excluding_loss_recovery,
            "loss_recovery_component": self.loss_recovery_component,
            "aic_present_value": self.aic_present_value,
            "aic_risk_adjustment": self.aic_risk_adjustment,
            "carrying_amount": self.carrying_amount,
        })

    def profit_or_loss(self) -> pd.DataFrame:
        """Amounts recognised in profit or loss; income positive, expenses negative."""
        table = self._frame({
            "allocation_of_reinsurance_premiums": -self.allocation_of_premiums,
            "recoveries_of_incurred_claims": (self.recoveries_incurred
                                              + self.recoveries_risk_adjustment),
            "recoveries_of_losses_on_onerous_contracts": self.loss_recovery_income,
            "adjustments_to_assets_for_incurred_claims": self.adjustments_to_aic,
        })
        table["net_expenses_from_reinsurance_contracts"] = table.sum(axis=1)
        table["reinsurance_finance_income"] = self.finance_income - self.finance_income_in_oci
        table["reinsurance_finance_income_oci"] = self.finance_income_in_oci
        return table

    def cash_flows(self) -> pd.DataFrame:
        """Cash received (positive) and paid (negative) in each period."""
        table = self._frame({
            "reinsurance_premiums_paid": -self.premiums_paid,
            "recoveries_received": self.recoveries_received,
        })
        table["net_cash_flow"] = table.sum(axis=1)
        return table

    def __repr__(self) -> str:
        return (f"PAAReinsuranceHeld(name='{self.name}', periods={self.n_periods})\n"
                f"{self.financial_position().to_string()}")
