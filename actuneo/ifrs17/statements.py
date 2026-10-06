"""
IFRS 17 Financial Statements

Builds the primary statements of an insurer from its groups of insurance
contracts and reinsurance contracts held: the statement of profit or loss,
the statement of financial position and the statement of cash flows.

The presentation follows IFRS 17.78-92 and IAS 1:

- Insurance revenue, insurance service expenses and the net result of
  reinsurance contracts held make up the **insurance service result**.
- Insurance finance income or expenses are shown separately, next to the
  investment return.
- Portfolios of contracts that are assets are presented separately from
  portfolios that are liabilities, for insurance contracts issued and for
  reinsurance contracts held.

The statements are those of a simple insurer whose only other items are cash
and investments, operating expenses that are not attributable to insurance
contracts, investment return and tax, all settled in cash in the period.
"""

import numpy as np
import pandas as pd
from typing import Sequence
from ._common import as_series
from .paa import PAAGroup, PAAReinsuranceHeld


class IFRS17Statements:
    """
    Primary financial statements from groups of insurance contracts.

    By construction the statement of financial position balances: assets
    less liabilities equals the opening equity plus the profits to date.
    """

    def __init__(self,
                 groups: Sequence[PAAGroup],
                 reinsurance: Sequence[PAAReinsuranceHeld] = (),
                 opening_equity: float = 0.0,
                 investment_return=None,
                 other_operating_expenses=None,
                 tax_rate: float = 0.0):
        """
        Args:
            groups: Groups of insurance contracts issued
            reinsurance: Groups of reinsurance contracts held
            opening_equity: Equity at the start of the first period. Cash and
                investments at that date are this equity plus the opening
                insurance contract liabilities, less the opening reinsurance
                contract assets.
            investment_return: Investment return of each period, received in cash
            other_operating_expenses: Expenses of each period that are not
                directly attributable to insurance contracts, paid in cash
            tax_rate: Rate of income tax on the profit of each period, paid
                in the period. No tax is charged or credited on a loss.
        """
        self.groups = list(groups)
        self.reinsurance = list(reinsurance)
        if not self.groups:
            raise ValueError("At least one group of insurance contracts is needed")
        self.periods = list(self.groups[0].periods)
        for item in self.groups + self.reinsurance:
            if list(item.periods) != self.periods:
                raise ValueError("All groups must cover the same reporting periods")
        if not 0 <= tax_rate < 1:
            raise ValueError("tax_rate must be between 0 and 1")

        n = len(self.periods)
        self.opening_equity = float(opening_equity)
        self.investment_return = as_series(investment_return, n, "investment_return")
        self.other_operating_expenses = as_series(other_operating_expenses, n,
                                                  "other_operating_expenses")
        self.tax_rate = float(tax_rate)
        self.opening_cash = (self.opening_equity
                             + sum(g.opening_carrying_amount for g in self.groups)
                             - sum(r.opening_carrying_amount for r in self.reinsurance))

    # ------------------------------------------------------------------
    def _sum(self, items, table: str, column: str) -> np.ndarray:
        total = np.zeros(len(self.periods))
        for item in items:
            total = total + getattr(item, table)()[column].to_numpy()
        return total

    def _table(self, rows: dict) -> pd.DataFrame:
        # Adding zero turns any negative zero into a plain zero
        return pd.DataFrame(rows, index=pd.Index(self.periods, name="period")).T + 0.0

    def profit_or_loss(self, detailed: bool = False) -> pd.DataFrame:
        """
        Statement of profit or loss and other comprehensive income, one
        column per reporting period. Income is positive and expenses are
        negative.

        Other comprehensive income holds the insurance finance income and
        expenses that the entity has chosen to present there
        (IFRS 17.88(b)). No tax is attributed to it.

        Args:
            detailed: Show the components of insurance service expenses on
                the face of the statement
        """
        revenue = self._sum(self.groups, "profit_or_loss", "insurance_revenue")
        service_expenses = self._sum(self.groups, "profit_or_loss", "insurance_service_expenses")
        reinsurance = self._sum(self.reinsurance, "profit_or_loss",
                                "net_expenses_from_reinsurance_contracts")
        service_result = revenue + service_expenses + reinsurance

        insurance_finance = self._sum(self.groups, "profit_or_loss", "insurance_finance_expenses")
        reinsurance_finance = self._sum(self.reinsurance, "profit_or_loss",
                                        "reinsurance_finance_income")
        financial_result = self.investment_return + insurance_finance + reinsurance_finance

        before_tax = service_result + financial_result - self.other_operating_expenses
        tax = -self.tax_rate * np.maximum(before_tax, 0.0)
        profit = before_tax + tax
        oci = (self._sum(self.groups, "profit_or_loss", "insurance_finance_expenses_oci")
               + self._sum(self.reinsurance, "profit_or_loss", "reinsurance_finance_income_oci"))

        rows = {"Insurance revenue": revenue}
        if detailed:
            for label, column in self._EXPENSE_LINES:
                rows[label] = self._sum(self.groups, "profit_or_loss", column)
        rows.update({
            "Insurance service expenses": service_expenses,
            "Net expenses from reinsurance contracts": reinsurance,
            "Insurance service result": service_result,
            "Investment return": self.investment_return,
            "Net finance expenses from insurance contracts": insurance_finance,
            "Net finance income from reinsurance contracts": reinsurance_finance,
            "Net financial result": financial_result,
            "Other operating expenses": -self.other_operating_expenses,
            "Profit before tax": before_tax,
            "Income tax expense": tax,
            "Profit for the period": profit,
            "Other comprehensive income": oci,
            "Total comprehensive income": profit + oci,
        })
        return self._table(rows)

    _EXPENSE_LINES = (
        ("Incurred claims and other insurance service expenses",
         "incurred_claims_and_other_expenses"),
        ("Amortisation of insurance acquisition cash flows",
         "amortisation_of_acquisition_cash_flows"),
        ("Losses on onerous contracts and reversals", "losses_on_onerous_contracts"),
        ("Adjustments to liabilities for incurred claims", "adjustments_to_lic"),
    )

    def insurance_service_expenses(self) -> pd.DataFrame:
        """Note analysing insurance service expenses by nature (IFRS 17.103(b))."""
        rows = {label: self._sum(self.groups, "profit_or_loss", column)
                for label, column in self._EXPENSE_LINES}
        rows["Insurance service expenses"] = np.sum(list(rows.values()), axis=0)
        return self._table(rows)

    def finance_income_and_expenses(self) -> pd.DataFrame:
        """
        Note on insurance finance income and expenses, showing the amounts
        recognised in profit or loss and in other comprehensive income
        (IFRS 17.110).
        """
        insurance_pl = self._sum(self.groups, "profit_or_loss", "insurance_finance_expenses")
        insurance_oci = self._sum(self.groups, "profit_or_loss",
                                  "insurance_finance_expenses_oci")
        reinsurance_pl = self._sum(self.reinsurance, "profit_or_loss",
                                   "reinsurance_finance_income")
        reinsurance_oci = self._sum(self.reinsurance, "profit_or_loss",
                                    "reinsurance_finance_income_oci")
        return self._table({
            "Insurance contracts: recognised in profit or loss": insurance_pl,
            "Insurance contracts: recognised in OCI": insurance_oci,
            "Net finance expenses from insurance contracts": insurance_pl + insurance_oci,
            "Reinsurance contracts: recognised in profit or loss": reinsurance_pl,
            "Reinsurance contracts: recognised in OCI": reinsurance_oci,
            "Net finance income from reinsurance contracts": reinsurance_pl + reinsurance_oci,
            "Investment return": self.investment_return,
            "Net financial result including OCI": (insurance_pl + insurance_oci + reinsurance_pl
                                                   + reinsurance_oci + self.investment_return),
        })

    def cash_flows(self) -> pd.DataFrame:
        """Statement of cash flows (direct method), one column per reporting period."""
        profit = self.profit_or_loss()
        rows = {
            "Premiums received": self._sum(self.groups, "cash_flows", "premiums_received"),
            "Claims and other insurance service expenses paid":
                self._sum(self.groups, "cash_flows", "claims_paid"),
            "Insurance acquisition cash flows paid":
                self._sum(self.groups, "cash_flows", "acquisition_cash_flows"),
            "Reinsurance premiums paid":
                self._sum(self.reinsurance, "cash_flows", "reinsurance_premiums_paid"),
            "Recoveries received from reinsurers":
                self._sum(self.reinsurance, "cash_flows", "recoveries_received"),
            "Other operating expenses paid": -self.other_operating_expenses,
            "Investment return received": self.investment_return,
            "Income tax paid": profit.loc["Income tax expense"].to_numpy(),
        }
        net = np.sum(list(rows.values()), axis=0)
        closing = self.opening_cash + np.cumsum(net)
        rows["Net increase in cash and investments"] = net
        rows["Cash and investments at start of period"] = np.concatenate(
            ([self.opening_cash], closing[:-1])
        )
        rows["Cash and investments at end of period"] = closing
        return self._table(rows)

    def _by_portfolio(self, items) -> tuple:
        """Totals of portfolios in a positive position and in a negative position."""
        n = len(self.periods)
        portfolios = {}
        for item in items:
            portfolios[item.portfolio] = (portfolios.get(item.portfolio, np.zeros(n))
                                          + item.carrying_amount)
        positive, negative = np.zeros(n), np.zeros(n)
        for amount in portfolios.values():
            positive = positive + np.maximum(amount, 0.0)
            negative = negative + np.maximum(-amount, 0.0)
        return positive, negative

    def financial_position(self) -> pd.DataFrame:
        """Statement of financial position at each reporting date."""
        cash = self.cash_flows().loc["Cash and investments at end of period"].to_numpy()
        insurance_liabilities, insurance_assets = self._by_portfolio(self.groups)
        reinsurance_assets, reinsurance_liabilities = self._by_portfolio(self.reinsurance)
        performance = self.profit_or_loss()
        retained = np.cumsum(performance.loc["Profit for the period"].to_numpy())
        finance_reserve = np.cumsum(performance.loc["Other comprehensive income"].to_numpy())

        total_assets = cash + insurance_assets + reinsurance_assets
        total_liabilities = insurance_liabilities + reinsurance_liabilities
        total_equity = self.opening_equity + retained + finance_reserve
        return self._table({
            "Cash and investments": cash,
            "Insurance contract assets": insurance_assets,
            "Reinsurance contract assets": reinsurance_assets,
            "Total assets": total_assets,
            "Insurance contract liabilities": insurance_liabilities,
            "Reinsurance contract liabilities": reinsurance_liabilities,
            "Total liabilities": total_liabilities,
            "Share capital and opening reserves": np.full(len(self.periods), self.opening_equity),
            "Retained earnings": retained,
            "Insurance finance reserve": finance_reserve,
            "Total equity": total_equity,
            "Total liabilities and equity": total_liabilities + total_equity,
        })

    def changes_in_equity(self) -> pd.DataFrame:
        """Statement of changes in equity, one column per reporting period."""
        performance = self.profit_or_loss()
        profit = performance.loc["Profit for the period"].to_numpy()
        oci = performance.loc["Other comprehensive income"].to_numpy()
        closing = self.opening_equity + np.cumsum(profit + oci)
        return self._table({
            "Equity at start of period": np.concatenate(([self.opening_equity], closing[:-1])),
            "Profit for the period": profit,
            "Other comprehensive income": oci,
            "Total comprehensive income": profit + oci,
            "Equity at end of period": closing,
        })

    def supplementary_position(self) -> pd.DataFrame:
        """
        Insurance contract balances analysed into the components reported
        under IFRS 4 (unearned premium, premiums receivable, deferred
        acquisition costs, outstanding claims), as supplementary information
        for readers used to that presentation.
        """
        columns = ["unearned_premium", "premiums_receivable", "deferred_acquisition_costs",
                   "loss_component", "outstanding_claims", "risk_adjustment", "carrying_amount"]
        labels = ["Unearned premium", "Premiums receivable", "Deferred acquisition costs",
                  "Loss component", "Outstanding claims (present value)", "Risk adjustment",
                  "Net insurance contract liabilities"]
        return self._table({
            label: self._sum(self.groups, "supplementary_position", column)
            for label, column in zip(labels, columns)
        })

    def balance_check(self) -> float:
        """Largest difference between total assets and total liabilities plus equity."""
        position = self.financial_position()
        difference = position.loc["Total assets"] - position.loc["Total liabilities and equity"]
        return float(difference.abs().max())

    def key_ratios(self) -> pd.DataFrame:
        """
        Performance ratios on insurance revenue for each period: claims
        ratio, expense ratio, reinsurance ratio and combined ratio.
        """
        revenue = self._sum(self.groups, "profit_or_loss", "insurance_revenue")
        claims = -(self._sum(self.groups, "profit_or_loss", "incurred_claims_and_other_expenses")
                   + self._sum(self.groups, "profit_or_loss", "adjustments_to_lic")
                   + self._sum(self.groups, "profit_or_loss", "losses_on_onerous_contracts"))
        acquisition = -self._sum(self.groups, "profit_or_loss",
                                 "amortisation_of_acquisition_cash_flows")
        reinsurance = -self._sum(self.reinsurance, "profit_or_loss",
                                 "net_expenses_from_reinsurance_contracts")
        with np.errstate(divide="ignore", invalid="ignore"):
            rows = {
                "Claims ratio": claims / revenue,
                "Acquisition expense ratio": acquisition / revenue,
                "Other expense ratio": self.other_operating_expenses / revenue,
                "Net reinsurance ratio": reinsurance / revenue,
            }
            rows["Combined ratio"] = np.sum(list(rows.values()), axis=0)
        return self._table(rows)
