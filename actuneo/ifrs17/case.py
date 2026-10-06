"""
PAA Case Files

Runs a full premium allocation approach calculation from input files, so
that the figures can be prepared in a spreadsheet instead of in code.

A case is either a folder of four CSV files or one Excel workbook with four
sheets of the same names:

``settings``
    Two columns, ``key`` and ``value``. Recognised keys: ``entity``,
    ``currency``, ``opening_equity``, ``tax_rate``.

``groups``
    One row per group of contracts. Columns:

    - ``group``: name of the group (required)
    - ``type``: ``insurance`` or ``reinsurance`` (required)
    - ``portfolio``: portfolio the group belongs to
    - ``expected_premium``: total premium to be recognised as revenue
    - ``expense_acquisition_cash_flows``: TRUE to expense them when incurred
    - ``lrc_discount_rate``, ``periods_per_year``: financing component
    - ``opening_lrc``, ``opening_loss_component``, ``opening_lic_pv``,
      ``opening_lic_ra``, ``opening_deferred_acquisition_cash_flows``:
      balances of a group already in force

``data``
    One row per group and reporting period. Columns ``group`` and ``period``
    are required; the rest are optional and a column left empty for a group
    is treated as not supplied:

    - ``premiums``: premiums received (reinsurance: premiums paid)
    - ``insurance_revenue`` or ``revenue_pattern`` (reinsurance: the
      allocation of reinsurance premiums, or its pattern)
    - ``premiums_written``: premium of the policies added to the group in
      the period, for a group that grows during the year
    - ``acquisition_cash_flows``
    - ``incurred_claims``, ``incurred_risk_adjustment``
      (reinsurance: recoveries incurred and the risk adjustment ceded)
    - ``claims_paid`` (reinsurance: recoveries received)
    - ``closing_lic_pv``, ``closing_lic_ra``
      (reinsurance: the asset for incurred claims)
    - ``finance_expenses``, ``finance_expenses_in_oci``
      (reinsurance: finance income)
    - ``other_insurance_service_expenses``
    - ``fulfilment_cash_flows_remaining``: for the onerous test
    - ``loss_recovery_component``: reinsurance only

``entity``
    One row per reporting period with ``period``, ``investment_return`` and
    ``other_operating_expenses``.

Periods are taken in the order they appear in the ``entity`` sheet (or the
``data`` sheet if there is none) and every group must have every period.
"""

import shutil
from importlib import resources
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .paa import PAAGroup, PAAReinsuranceHeld
from .statements import IFRS17Statements

_SHEETS = ("settings", "groups", "data", "entity")
_INSURANCE_SERIES = {
    "premiums": "premiums_received",
    "insurance_revenue": "insurance_revenue",
    "revenue_pattern": "revenue_pattern",
    "premiums_written": "premiums_written",
    "acquisition_cash_flows": "acquisition_cash_flows",
    "incurred_claims": "incurred_claims",
    "incurred_risk_adjustment": "incurred_risk_adjustment",
    "claims_paid": "claims_paid",
    "closing_lic_pv": "closing_lic_pv",
    "closing_lic_ra": "closing_lic_ra",
    "finance_expenses": "finance_expenses",
    "finance_expenses_in_oci": "finance_expenses_in_oci",
    "other_insurance_service_expenses": "other_insurance_service_expenses",
    "fulfilment_cash_flows_remaining": "fulfilment_cash_flows_remaining",
}
_REINSURANCE_SERIES = {
    "premiums": "premiums_paid",
    "insurance_revenue": "allocation_of_premiums",
    "revenue_pattern": "allocation_pattern",
    "incurred_claims": "recoveries_incurred",
    "incurred_risk_adjustment": "recoveries_risk_adjustment",
    "claims_paid": "recoveries_received",
    "closing_lic_pv": "closing_aic_pv",
    "closing_lic_ra": "closing_aic_ra",
    "finance_expenses": "finance_income",
    "finance_expenses_in_oci": "finance_income_in_oci",
    "loss_recovery_component": "loss_recovery_component",
}
_INSURANCE_SCALARS = ("expected_premium", "lrc_discount_rate", "opening_lrc",
                      "opening_loss_component", "opening_lic_pv", "opening_lic_ra",
                      "opening_deferred_acquisition_cash_flows")
_REINSURANCE_SCALARS = {"expected_premium": "expected_premium", "opening_lrc": "opening_arc",
                        "opening_loss_component": "opening_loss_recovery_component",
                        "opening_lic_pv": "opening_aic_pv", "opening_lic_ra": "opening_aic_ra"}


def example_case_path() -> Path:
    """Folder of the example case shipped with ACTUNEO (a fictional insurer)."""
    return Path(str(resources.files("actuneo.ifrs17") / "data" / "example_case"))


def copy_example_case(destination: str) -> Path:
    """
    Copy the example case files to a folder, as a template to edit.

    Args:
        destination: Folder to create. It must not already exist.

    Returns:
        Path of the new folder
    """
    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"{target} already exists")
    shutil.copytree(example_case_path(), target)
    return target


def _read_sheets(path: Path) -> Dict[str, pd.DataFrame]:
    if path.is_dir():
        sheets = {}
        for name in _SHEETS:
            file = path / f"{name}.csv"
            if file.exists():
                sheets[name] = pd.read_csv(file)
        return sheets
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        try:
            import openpyxl  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Reading Excel files requires openpyxl. Install it with: "
                "pip install actuneo[excel]"
            ) from exc
        book = pd.read_excel(path, sheet_name=None)
        return {name: frame for name, frame in book.items() if name in _SHEETS}
    raise ValueError("A case must be a folder of CSV files or an .xlsx workbook")


def _is_true(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "y", "1")
    return bool(value) and not (isinstance(value, float) and np.isnan(value))


class PAACase:
    """
    A complete premium allocation approach calculation loaded from files.

    Attributes:
        settings: The settings read from the case
        periods: Reporting periods, in order
        groups: Groups of insurance contracts by name
        reinsurance: Groups of reinsurance contracts held by name
        statements: The financial statements of the entity
    """

    def __init__(self, path: str):
        """
        Args:
            path: Folder of CSV files, or an Excel workbook
        """
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"{source} does not exist")
        sheets = _read_sheets(source)
        for required in ("groups", "data"):
            if required not in sheets:
                raise ValueError(f"The case has no '{required}' sheet")
        groups, data = sheets["groups"], sheets["data"]
        for frame, columns, name in ((groups, ("group", "type"), "groups"),
                                     (data, ("group", "period"), "data")):
            missing = [c for c in columns if c not in frame.columns]
            if missing:
                raise ValueError(f"The '{name}' sheet has no column(s) {missing}")
        unknown = set(data.columns) - {"group", "period"} - set(_INSURANCE_SERIES) \
            - set(_REINSURANCE_SERIES)
        if unknown:
            raise ValueError(f"The 'data' sheet has unknown column(s) {sorted(unknown)}")

        self.settings = {}
        if "settings" in sheets:
            self.settings = dict(zip(sheets["settings"]["key"].astype(str),
                                     sheets["settings"]["value"]))
        entity = sheets.get("entity")
        if entity is not None:
            self.periods = entity["period"].astype(str).tolist()
        else:
            self.periods = list(dict.fromkeys(data["period"].astype(str)))
        data = data.assign(period=data["period"].astype(str))

        self.groups: Dict[str, PAAGroup] = {}
        self.reinsurance: Dict[str, PAAReinsuranceHeld] = {}
        for _, row in groups.iterrows():
            name = str(row["group"])
            kind = str(row["type"]).strip().lower()
            if kind not in ("insurance", "reinsurance"):
                raise ValueError(f"Group '{name}': type must be 'insurance' or 'reinsurance'")
            rows = data[data["group"].astype(str) == name].set_index("period")
            if rows.index.duplicated().any() or set(rows.index) != set(self.periods):
                raise ValueError(f"Group '{name}' must have exactly one row for every period")
            rows = rows.loc[self.periods]

            series_map = _INSURANCE_SERIES if kind == "insurance" else _REINSURANCE_SERIES
            arguments = {}
            for column, argument in series_map.items():
                if column not in rows.columns or rows[column].isna().all():
                    continue
                if rows[column].isna().any():
                    raise ValueError(f"Group '{name}': column '{column}' has empty cells")
                arguments[argument] = rows[column].to_numpy(dtype=float)
            premium_argument = series_map["premiums"]
            if premium_argument not in arguments:
                arguments[premium_argument] = np.zeros(len(self.periods))

            def scalar(column):
                value = row.get(column)
                return None if value is None or pd.isna(value) else float(value)

            if kind == "insurance":
                for column in _INSURANCE_SCALARS:
                    if scalar(column) is not None:
                        arguments[column] = scalar(column)
                if scalar("periods_per_year") is not None:
                    arguments["periods_per_year"] = int(scalar("periods_per_year"))
                if "expense_acquisition_cash_flows" in row.index:
                    arguments["expense_acquisition_cash_flows"] = _is_true(
                        row["expense_acquisition_cash_flows"]
                    )
            else:
                for column, argument in _REINSURANCE_SCALARS.items():
                    if scalar(column) is not None:
                        arguments[argument] = scalar(column)

            portfolio = row.get("portfolio")
            arguments.update(periods=self.periods, name=name,
                             portfolio=None if portfolio is None or pd.isna(portfolio)
                             else str(portfolio))
            if kind == "insurance":
                self.groups[name] = PAAGroup(**arguments)
            else:
                self.reinsurance[name] = PAAReinsuranceHeld(**arguments)

        investment = other = None
        if entity is not None:
            if "investment_return" in entity.columns:
                investment = entity["investment_return"].fillna(0).to_numpy(dtype=float)
            if "other_operating_expenses" in entity.columns:
                other = entity["other_operating_expenses"].fillna(0).to_numpy(dtype=float)
        self.statements = IFRS17Statements(
            list(self.groups.values()), list(self.reinsurance.values()),
            opening_equity=float(self.settings.get("opening_equity", 0) or 0),
            investment_return=investment, other_operating_expenses=other,
            tax_rate=float(self.settings.get("tax_rate", 0) or 0),
        )

    def report(self, total: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Every table of the calculation, with reporting periods in columns
        and line items in rows.

        Args:
            total: Add a "Total" column to the statements of profit or loss,
                cash flows, changes in equity and key ratios

        Returns:
            Dictionary of DataFrames: the financial statements and notes,
            then the roll-forward tables and reconciliation of each group
        """
        s = self.statements
        tables = {
            "Profit or loss": s.profit_or_loss(detailed=True, total=total),
            "Financial position": s.financial_position(),
            "Cash flows": s.cash_flows(total=total),
            "Changes in equity": s.changes_in_equity(total=total),
            "Key ratios": s.key_ratios(total=total),
            "Supplementary position": s.supplementary_position(),
        }
        for name, group in self.groups.items():
            tables[f"{name} - LRC"] = group.lrc_rollforward().T
            tables[f"{name} - LIC"] = group.lic_rollforward().T
            tables[f"{name} - Reconciliation"] = group.reconciliation("all")
        for name, group in self.reinsurance.items():
            tables[f"{name} - ARC"] = group.arc_rollforward().T
            tables[f"{name} - AIC"] = group.aic_rollforward().T
            tables[f"{name} - Reconciliation"] = group.reconciliation("all")
        return tables

    def to_excel(self, path: str) -> None:
        """
        Write the whole calculation to a formatted Excel workbook (needs
        openpyxl): a contents sheet, the financial statements and notes, and
        for each group its roll-forwards and IFRS 17.100 reconciliation.

        Args:
            path: File name of the workbook, ending in .xlsx
        """
        from ..utils import Sheet, write_report
        from .statements import _REPORT_NOTES

        entity = str(self.settings.get("entity", "") or "")
        currency = str(self.settings.get("currency", "") or "")
        unit = f"In {currency}" if currency else ""
        sheets = self.statements.presentation(entity, currency)

        def movement(table, title, closing_label="closing"):
            return Sheet(table, title, unit, total_rows=[closing_label])

        for name, group in self.groups.items():
            sheets[f"{name} - LRC"] = movement(
                group.lrc_rollforward().T, f"{name}: liability for remaining coverage")
            sheets[f"{name} - LIC"] = movement(
                group.lic_rollforward().T, f"{name}: liability for incurred claims")
            sheets[f"{name} - Reconciliation"] = Sheet(
                group.reconciliation("all"),
                f"{name}: reconciliation of insurance contract liabilities",
                ". ".join(x for x in (f"{self.periods[0]} to {self.periods[-1]}", unit) if x),
                total_rows=["Insurance service result", "Total changes in profit or loss",
                            "Total cash flows", "Closing liabilities"],
                references={"Opening liabilities": "IFRS 17.100",
                            "Insurance revenue": "IFRS 17.103(a)",
                            "Insurance service result": "IFRS 17.103",
                            "Premiums received": "IFRS 17.105(a)(i)"},
            )
        for name, group in self.reinsurance.items():
            sheets[f"{name} - ARC"] = movement(
                group.arc_rollforward().T, f"{name}: asset for remaining coverage")
            sheets[f"{name} - AIC"] = movement(
                group.aic_rollforward().T, f"{name}: asset for incurred claims")
            sheets[f"{name} - Reconciliation"] = Sheet(
                group.reconciliation("all"),
                f"{name}: reconciliation of reinsurance contract assets",
                ". ".join(x for x in (f"{self.periods[0]} to {self.periods[-1]}", unit) if x),
                total_rows=["Net expenses from reinsurance contracts",
                            "Total changes in profit or loss", "Total cash flows",
                            "Closing assets"],
                references={"Opening assets": "IFRS 17.100"},
            )
        write_report(path, sheets, report_title=entity or "PAA case",
                     report_subtitle="IFRS 17 premium allocation approach",
                     notes=_REPORT_NOTES)

    def __repr__(self) -> str:
        entity = self.settings.get("entity", "PAA case")
        return (f"PAACase('{entity}', {len(self.groups)} insurance group(s), "
                f"{len(self.reinsurance)} reinsurance group(s), {len(self.periods)} periods)")


def load_paa_case(path: str) -> PAACase:
    """
    Load and run a premium allocation approach case from files.

    Args:
        path: Folder of CSV files, or an Excel workbook, in the format
            described in :mod:`actuneo.ifrs17.case`

    Returns:
        PAACase with the groups measured and the statements built
    """
    return PAACase(path)
