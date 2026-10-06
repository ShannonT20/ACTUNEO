"""
Formatted Excel Reports

Writes tables to an Excel workbook laid out for reading: a contents sheet
with links, a title block on every sheet, a shaded header row, bold totals
with a rule above them, negatives in brackets and sensible column widths.

Needs openpyxl (``pip install actuneo[excel]``).
"""

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

NUMBER_FORMAT = '#,##0;(#,##0);"-"'
DECIMAL_FORMAT = '#,##0.00;(#,##0.00);"-"'
RATIO_FORMAT = '0.0%;(0.0%);"-"'
FACTOR_FORMAT = '0.0000'

_HEADER_FILL = "1F3864"
_SECTION_FILL = "D9E2F3"
_INVALID = '[]:*?/\\'


class Sheet:
    """
    One sheet of a report.

    Attributes:
        table: The data, with line items in the index
        title: Heading shown at the top of the sheet
        subtitle: Second line of the heading (for example the period)
        number_format: Excel number format for the figures
        total_rows: Index labels to show in bold with a rule above
        section_rows: Index labels that are section headings without figures
        references: Text for a reference column (for example the paragraph
            of the standard), keyed by index label
        note: Text placed under the table
    """

    def __init__(self,
                 table: pd.DataFrame,
                 title: str,
                 subtitle: str = "",
                 number_format: str = NUMBER_FORMAT,
                 total_rows: Iterable[str] = (),
                 section_rows: Iterable[str] = (),
                 references: Optional[Dict[str, str]] = None,
                 note: str = ""):
        if isinstance(table, pd.Series):
            table = table.to_frame()
        self.table = table
        self.title = title
        self.subtitle = subtitle
        self.number_format = number_format
        self.total_rows = set(total_rows)
        self.section_rows = set(section_rows)
        self.references = references or {}
        self.note = note


def _sheet_name(title: str, used: set) -> str:
    name = "".join(c for c in title if c not in _INVALID)[:31] or "Sheet"
    base, k = name, 2
    while name.lower() in used:
        suffix = f" {k}"
        name, k = base[:31 - len(suffix)] + suffix, k + 1
    used.add(name.lower())
    return name


def write_report(path: str,
                 sheets: Dict[str, Sheet],
                 report_title: str = "Report",
                 report_subtitle: str = "",
                 notes: Sequence[str] = ()) -> None:
    """
    Write a formatted workbook.

    Args:
        path: File name of the workbook, ending in .xlsx
        sheets: Sheets to write, keyed by the name to show in the contents
        report_title: Heading of the contents sheet (for example the entity)
        report_subtitle: Second line of the contents heading
        notes: Lines of text shown under the contents list, for example the
            basis of preparation and its limitations
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
        from openpyxl.utils import get_column_letter
    except ImportError as exc:
        raise ImportError(
            "Writing Excel files requires openpyxl. Install it with: pip install actuneo[excel]"
        ) from exc

    book = Workbook()
    contents = book.active
    contents.title = "Contents"
    used = {"contents"}
    names = {key: _sheet_name(key, used) for key in sheets}

    bold = Font(bold=True)
    white_bold = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor=_HEADER_FILL)
    section_fill = PatternFill("solid", fgColor=_SECTION_FILL)
    rule = Border(top=Side(style="thin"))
    double_rule = Border(top=Side(style="thin"), bottom=Side(style="double"))

    # Contents
    contents["A1"] = report_title
    contents["A1"].font = Font(bold=True, size=16)
    contents["A2"] = report_subtitle
    contents["A2"].font = Font(italic=True, color="595959")
    contents["A4"] = "Contents"
    contents["A4"].font = white_bold
    contents["A4"].fill = header_fill
    row = 5
    for key in sheets:
        cell = contents.cell(row=row, column=1, value=key)
        cell.hyperlink = f"#'{names[key]}'!A1"
        cell.font = Font(color="0563C1", underline="single")
        row += 1
    row += 1
    for line in notes:
        cell = contents.cell(row=row, column=1, value=line)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        cell.font = Font(italic=True, color="595959")
        row += 1
    contents.column_dimensions["A"].width = 90

    for key, sheet in sheets.items():
        ws = book.create_sheet(names[key])
        table = sheet.table
        has_reference = bool(sheet.references)
        first_value_column = 3 if has_reference else 2

        ws["A1"] = report_title
        ws["A1"].font = Font(bold=True, size=11, color="595959")
        ws["A2"] = sheet.title
        ws["A2"].font = Font(bold=True, size=14)
        ws["A3"] = sheet.subtitle
        ws["A3"].font = Font(italic=True, color="595959")
        back = ws.cell(row=1, column=first_value_column + len(table.columns), value="Contents")
        back.hyperlink = "#'Contents'!A1"
        back.font = Font(color="0563C1", underline="single")

        header_row = 5
        label = ws.cell(row=header_row, column=1, value=table.index.name or "")
        cells = [label]
        if has_reference:
            cells.append(ws.cell(row=header_row, column=2, value="Reference"))
        for j, column in enumerate(table.columns):
            cells.append(ws.cell(row=header_row, column=first_value_column + j, value=str(column)))
        for cell in cells:
            cell.font = white_bold
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center" if cell.column > 1 else "left",
                                       vertical="center", wrap_text=True)

        for i, (index, values) in enumerate(table.iterrows()):
            r = header_row + 1 + i
            text = str(index)
            is_total = text in sheet.total_rows
            is_section = text in sheet.section_rows
            name_cell = ws.cell(row=r, column=1, value=text)
            if is_section:
                for c in range(1, first_value_column + len(table.columns)):
                    ws.cell(row=r, column=c).fill = section_fill
                name_cell.font = bold
                continue
            if has_reference:
                ref = ws.cell(row=r, column=2, value=sheet.references.get(text, ""))
                ref.font = Font(size=8, color="7F7F7F")
            name_cell.font = bold if is_total else Font()
            if not is_total:
                name_cell.alignment = Alignment(indent=1)
            for j, value in enumerate(values):
                cell = ws.cell(row=r, column=first_value_column + j)
                if isinstance(value, (int, float, np.integer, np.floating)):
                    if not (isinstance(value, float) and np.isnan(value)):
                        cell.value = float(value)
                        cell.number_format = sheet.number_format
                elif value is not None and not pd.isna(value):
                    cell.value = str(value)
                cell.alignment = Alignment(horizontal="right")
                if is_total:
                    cell.font = bold
                    cell.border = double_rule if text.lower().startswith("total") else rule

        if sheet.note:
            note_cell = ws.cell(row=header_row + len(table) + 2, column=1, value=sheet.note)
            note_cell.font = Font(italic=True, size=9, color="595959")
            note_cell.alignment = Alignment(wrap_text=True, vertical="top")

        longest = max([len(str(i)) for i in table.index] + [12])
        ws.column_dimensions["A"].width = min(max(longest + 4, 22), 62)
        if has_reference:
            ws.column_dimensions["B"].width = 16
        for j in range(len(table.columns)):
            ws.column_dimensions[get_column_letter(first_value_column + j)].width = 15
        ws.freeze_panes = ws.cell(row=header_row + 1, column=first_value_column)
        ws.sheet_view.showGridLines = False

    book.save(path)
