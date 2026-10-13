"""
IAS 19: accounting for a defined benefit plan

:class:`IAS19Period` takes the defined benefit obligation and plan assets at
the start and end of a reporting period, with the cash flows between, and
produces the amounts that IAS 19 *Employee Benefits* requires: the cost in
profit or loss, the remeasurements in other comprehensive income, and the
reconciliations of the obligation, the assets and the net liability.

The obligation itself is measured by the projected unit credit method
(IAS 19.67), which is ``method="projected_unit"`` of
:class:`~actuneo.pensions.DBValuation`.
"""

from typing import Optional

import pandas as pd


class IAS19Period:
    """
    One reporting period of a defined benefit plan under IAS 19.

    Sign convention: the obligation, costs and losses are positive; gains
    are negative. The net defined benefit liability is the obligation less
    the plan assets, plus any effect of the asset ceiling.
    """

    def __init__(self,
                 opening_obligation: float,
                 opening_assets: float,
                 discount_rate: float,
                 current_service_cost: float,
                 closing_obligation: float,
                 closing_assets: float,
                 *,
                 benefits_paid: float = 0.0,
                 employer_contributions: float = 0.0,
                 member_contributions: float = 0.0,
                 past_service_cost: float = 0.0,
                 administration_expenses: float = 0.0,
                 cash_flow_timing: float = 0.5,
                 closing_obligation_before_assumption_changes: Optional[float] = None,
                 opening_asset_ceiling_effect: float = 0.0,
                 asset_ceiling: Optional[float] = None):
        """
        Args:
            opening_obligation: Defined benefit obligation at the start
            opening_assets: Fair value of plan assets at the start
            discount_rate: Discount rate at the start of the period
            current_service_cost: The employer's current service cost,
                measured at the end of the period
            closing_obligation: Obligation at the end, from the valuation
            closing_assets: Fair value of plan assets at the end
            benefits_paid: Benefits paid out of plan assets
            employer_contributions: Contributions paid by the employer
            member_contributions: Contributions paid by members
            past_service_cost: Cost of plan amendments and curtailments
                (negative for a gain)
            administration_expenses: Costs of running the plan paid from
                its assets
            cash_flow_timing: Point in the period, from 0 (start) to 1
                (end), at which the cash flows fall on average. It decides
                how much interest they carry.
            closing_obligation_before_assumption_changes: Closing obligation
                on the assumptions held at the start. When given, the
                actuarial gain or loss is split between experience and
                changes in assumptions.
            opening_asset_ceiling_effect: Reduction in the net asset at the
                start because of the asset ceiling
            asset_ceiling: Economic benefit available from a surplus, in
                refunds or reduced contributions. If omitted, a surplus is
                recognised in full.
        """
        if discount_rate <= -1:
            raise ValueError("discount_rate must be greater than -1")
        if not 0 <= cash_flow_timing <= 1:
            raise ValueError("cash_flow_timing must be between 0 and 1")
        if opening_asset_ceiling_effect < 0:
            raise ValueError("opening_asset_ceiling_effect must be non-negative")
        if asset_ceiling is not None and asset_ceiling < 0:
            raise ValueError("asset_ceiling must be non-negative")
        self.opening_obligation = float(opening_obligation)
        self.opening_assets = float(opening_assets)
        self.discount_rate = float(discount_rate)
        self.current_service_cost = float(current_service_cost)
        self.closing_obligation = float(closing_obligation)
        self.closing_assets = float(closing_assets)
        self.benefits_paid = float(benefits_paid)
        self.employer_contributions = float(employer_contributions)
        self.member_contributions = float(member_contributions)
        self.past_service_cost = float(past_service_cost)
        self.administration_expenses = float(administration_expenses)
        self.cash_flow_timing = float(cash_flow_timing)
        self._before_assumptions = closing_obligation_before_assumption_changes
        self.opening_asset_ceiling_effect = float(opening_asset_ceiling_effect)
        self.asset_ceiling = asset_ceiling

    @classmethod
    def from_valuations(cls, opening, closing, **cash_flows) -> 'IAS19Period':
        """
        Build the period from two projected unit valuations.

        The opening discount rate and normal cost come from the opening
        valuation. The current service cost is the normal cost with a year's
        interest, less the members' contributions.

        Args:
            opening: :class:`~actuneo.pensions.DBValuation` at the start
            closing: :class:`~actuneo.pensions.DBValuation` at the end
            **cash_flows: The other arguments of :class:`IAS19Period`, such
                as ``benefits_paid`` and ``employer_contributions``
        """
        for valuation in (opening, closing):
            if valuation.method != "projected_unit":
                raise ValueError("IAS 19 requires the projected unit credit method")
        rate = opening.basis.discount_rate
        members = cash_flows.get("member_contributions", 0.0)
        return cls(
            opening.total_liability, opening.assets, rate,
            opening.normal_cost * (1 + rate) - members,
            closing.total_liability, closing.assets, **cash_flows,
        )

    # ------------------------------------------------------------------
    # Obligation
    # ------------------------------------------------------------------
    @property
    def _weight(self) -> float:
        """Fraction of the period for which the cash flows carry interest."""
        return 1.0 - self.cash_flow_timing

    @property
    def interest_cost(self) -> float:
        """Interest on the obligation, allowing for benefits paid in the period."""
        return self.discount_rate * (self.opening_obligation
                                     - self._weight * self.benefits_paid)

    @property
    def expected_closing_obligation(self) -> float:
        """Obligation at the end had experience followed the opening assumptions."""
        return (self.opening_obligation + self.interest_cost + self.current_service_cost
                + self.past_service_cost + self.member_contributions - self.benefits_paid)

    @property
    def actuarial_loss(self) -> float:
        """Actuarial loss on the obligation (negative for a gain)."""
        return self.closing_obligation - self.expected_closing_obligation

    @property
    def experience_loss(self) -> float:
        """Part of the actuarial loss that comes from experience."""
        if self._before_assumptions is None:
            return self.actuarial_loss
        return float(self._before_assumptions) - self.expected_closing_obligation

    @property
    def assumption_loss(self) -> float:
        """Part of the actuarial loss that comes from changes in assumptions."""
        if self._before_assumptions is None:
            return 0.0
        return self.closing_obligation - float(self._before_assumptions)

    # ------------------------------------------------------------------
    # Assets
    # ------------------------------------------------------------------
    @property
    def interest_income(self) -> float:
        """Interest on plan assets at the discount rate."""
        net_cash = (self.employer_contributions + self.member_contributions
                    - self.benefits_paid - self.administration_expenses)
        return self.discount_rate * (self.opening_assets + self._weight * net_cash)

    @property
    def expected_closing_assets(self) -> float:
        """Assets at the end had they earned the discount rate."""
        return (self.opening_assets + self.interest_income + self.employer_contributions
                + self.member_contributions - self.benefits_paid
                - self.administration_expenses)

    @property
    def return_on_assets_excluding_interest(self) -> float:
        """Return on plan assets above (or below) interest at the discount rate."""
        return self.closing_assets - self.expected_closing_assets

    # ------------------------------------------------------------------
    # Asset ceiling and net liability
    # ------------------------------------------------------------------
    @property
    def closing_asset_ceiling_effect(self) -> float:
        """Reduction in the net asset at the end because of the asset ceiling."""
        if self.asset_ceiling is None:
            return 0.0
        return max(0.0, self.closing_assets - self.closing_obligation - self.asset_ceiling)

    @property
    def interest_on_asset_ceiling(self) -> float:
        """Interest on the effect of the asset ceiling."""
        return self.discount_rate * self.opening_asset_ceiling_effect

    @property
    def change_in_asset_ceiling(self) -> float:
        """Change in the effect of the asset ceiling, excluding interest."""
        return (self.closing_asset_ceiling_effect - self.opening_asset_ceiling_effect
                - self.interest_on_asset_ceiling)

    @property
    def net_interest(self) -> float:
        """Net interest on the net defined benefit liability."""
        return self.interest_cost - self.interest_income + self.interest_on_asset_ceiling

    @property
    def opening_net_liability(self) -> float:
        """Net defined benefit liability at the start (negative for an asset)."""
        return (self.opening_obligation - self.opening_assets
                + self.opening_asset_ceiling_effect)

    @property
    def closing_net_liability(self) -> float:
        """Net defined benefit liability at the end (negative for an asset)."""
        return (self.closing_obligation - self.closing_assets
                + self.closing_asset_ceiling_effect)

    # ------------------------------------------------------------------
    # Statements
    # ------------------------------------------------------------------
    def profit_or_loss(self) -> pd.Series:
        """Defined benefit cost recognised in profit or loss."""
        items = pd.Series({
            "Current service cost": self.current_service_cost,
            "Past service cost": self.past_service_cost,
            "Administration expenses": self.administration_expenses,
            "Net interest": self.net_interest,
        })
        items["Defined benefit cost in profit or loss"] = items.sum()
        return items

    def other_comprehensive_income(self) -> pd.Series:
        """Remeasurements recognised in other comprehensive income (losses positive)."""
        items = pd.Series({
            "Actuarial loss / (gain): experience": self.experience_loss,
            "Actuarial loss / (gain): changes in assumptions": self.assumption_loss,
            "Return on plan assets excluding interest": -self.return_on_assets_excluding_interest,
            "Change in the effect of the asset ceiling": self.change_in_asset_ceiling,
        })
        items["Remeasurement loss / (gain) in other comprehensive income"] = items.sum()
        return items

    def obligation_reconciliation(self) -> pd.Series:
        """Movement in the defined benefit obligation over the period."""
        return pd.Series({
            "Opening obligation": self.opening_obligation,
            "Current service cost": self.current_service_cost,
            "Past service cost": self.past_service_cost,
            "Interest cost": self.interest_cost,
            "Contributions by members": self.member_contributions,
            "Benefits paid": -self.benefits_paid,
            "Actuarial loss / (gain): experience": self.experience_loss,
            "Actuarial loss / (gain): changes in assumptions": self.assumption_loss,
            "Closing obligation": self.closing_obligation,
        })

    def asset_reconciliation(self) -> pd.Series:
        """Movement in the fair value of plan assets over the period."""
        return pd.Series({
            "Opening plan assets": self.opening_assets,
            "Interest income": self.interest_income,
            "Return on plan assets excluding interest": self.return_on_assets_excluding_interest,
            "Contributions by the employer": self.employer_contributions,
            "Contributions by members": self.member_contributions,
            "Benefits paid": -self.benefits_paid,
            "Administration expenses": -self.administration_expenses,
            "Closing plan assets": self.closing_assets,
        })

    def net_liability_reconciliation(self) -> pd.Series:
        """Movement in the net defined benefit liability over the period."""
        return pd.Series({
            "Opening net liability / (asset)": self.opening_net_liability,
            "Defined benefit cost in profit or loss":
                self.profit_or_loss()["Defined benefit cost in profit or loss"],
            "Remeasurement loss / (gain) in other comprehensive income":
                self.other_comprehensive_income().iloc[-1],
            "Contributions by the employer": -self.employer_contributions,
            "Closing net liability / (asset)": self.closing_net_liability,
        })

    def financial_position(self) -> pd.Series:
        """The amount recognised in the statement of financial position at the end."""
        return pd.Series({
            "Defined benefit obligation": self.closing_obligation,
            "Fair value of plan assets": -self.closing_assets,
            "Deficit / (surplus)": self.closing_obligation - self.closing_assets,
            "Effect of the asset ceiling": self.closing_asset_ceiling_effect,
            "Net defined benefit liability / (asset)": self.closing_net_liability,
        })

    def __repr__(self) -> str:
        cost = self.profit_or_loss().iloc[-1]
        return (f"IAS19Period(profit_or_loss={cost:,.0f}, "
                f"oci={self.other_comprehensive_income().iloc[-1]:,.0f}, "
                f"closing_net_liability={self.closing_net_liability:,.0f})")
