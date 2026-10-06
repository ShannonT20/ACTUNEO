"""
Formatted Excel Reports

Writes tables to an Excel workbook designed for reading and printing:

- a cover sheet with the title, key figures and a linked list of contents;
- a coloured title band, shaded header and banded rows on every sheet;
- bold totals on a tinted row with rules above and below;
- negatives in red brackets, zeros as a dash;
- optional charts beside the tables;
- frozen headings, coloured tabs and landscape pages that fit the width.

Needs openpyxl (``pip install actuneo[excel]``).
"""

from datetime import date
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

NUMBER_FORMAT = '#,##0;[Red](#,##0);"-"'
DECIMAL_FORMAT = '#,##0.00;[Red](#,##0.00);"-"'
RATIO_FORMAT = '0.0%;[Red](0.0%);"-"'
FACTOR_FORMAT = '0.0000'

NAVY = "1F3864"
BLUE = "2E75B6"
TEAL = "1B7F79"
GREY = "7F7F7F"
_BAND = "F3F7FC"
_TOTAL = "DDEBF7"
_SECTION = "BDD7EE"
_CARD = "EAF1FB"
_INVALID = '[]:*?/\\'


class Sheet:
    """
    One sheet of a report.

    Attributes:
        table: The data, with line items in the index
        title: Heading shown at the top of the sheet
        subtitle: Second line of the heading (for example the period)
        number_format: Excel number format for the figures
        total_rows: Index labels to show as totals
        section_rows: Index labels that are section headings without figures
        references: Text for a reference column (for example the paragraph
            of the standard), keyed by index label
        note: Text placed under the table
        chart: A chart to draw beside the table, as a dictionary with
            ``kind`` ("line" or "bar"), ``title``, and either ``rows`` (index
            labels to plot across the columns) or ``columns`` (column names
            to plot down the index). ``exclude`` lists column or index
            labels to leave out, such as "Total".
        tab_color: Colour of the sheet tab, as a hex string
        description: Short description shown in the list of contents
    """

    def __init__(self,
                 table: pd.DataFrame,
                 title: str,
                 subtitle: str = "",
                 number_format: str = NUMBER_FORMAT,
                 total_rows: Iterable[str] = (),
                 section_rows: Iterable[str] = (),
                 references: Optional[Dict[str, str]] = None,
                 note: str = "",
                 chart: Optional[dict] = None,
                 tab_color: str = NAVY,
                 description: str = ""):
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
        self.chart = chart
        self.tab_color = tab_color
        self.description = description


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
                 notes: Sequence[str] = (),
                 highlights: Optional[Dict[str, tuple]] = None) -> None:
    """
    Write a formatted workbook.

    Args:
        path: File name of the workbook, ending in .xlsx
        sheets: Sheets to write, keyed by the name to show in the contents
        report_title: Heading of the cover sheet (for example the entity)
        report_subtitle: Second line of the cover heading
        notes: Lines of text shown under the contents list, for example the
            basis of preparation and its limitations
        highlights: Key figures for the cover sheet, as a dictionary of
            label to ``(value, number_format)``
    """
    try:
        from openpyxl import Workbook
        from openpyxl.chart import BarChart, LineChart, Reference
        from openpyxl.chart.series import SeriesLabel
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
        from openpyxl.utils import get_column_letter
    except ImportError as exc:
        raise ImportError(
            "Writing Excel files requires openpyxl. Install it with: pip install actuneo[excel]"
        ) from exc

    def fill(colour):
        return PatternFill("solid", fgColor=colour)

    thin = Side(style="thin", color="8EA9DB")
    hair = Side(style="hair", color="D0D7E5")
    white = Font(bold=True, color="FFFFFF")
    link = Font(color="0563C1", underline="single")

    book = Workbook()
    cover = book.active
    cover.title = "Contents"
    cover.sheet_properties.tabColor = NAVY
    used = {"contents"}
    names = {key: _sheet_name(key, used) for key in sheets}

    # ---------------- Cover sheet ----------------
    for column, width in zip("ABCDEF", (46, 52, 4, 30, 20, 4)):
        cover.column_dimensions[column].width = width
    for r in (1, 2):
        for c in range(1, 7):
            cover.cell(row=r, column=c).fill = fill(NAVY)
    cover.row_dimensions[1].height = 34
    cover.row_dimensions[2].height = 20
    cover["A1"] = report_title
    cover["A1"].font = Font(bold=True, size=20, color="FFFFFF")
    cover["A1"].alignment = Alignment(vertical="center", indent=1)
    cover["A2"] = report_subtitle
    cover["A2"].font = Font(italic=True, size=11, color="DDEBF7")
    cover["A2"].alignment = Alignment(vertical="center", indent=1)
    cover["E2"] = f"Prepared {date.today():%d %B %Y}"
    cover["E2"].font = Font(size=9, color="DDEBF7")
    cover["E2"].alignment = Alignment(horizontal="right", vertical="center")

    cover["A4"] = "Contents"
    cover["B4"] = "What it shows"
    for cell in (cover["A4"], cover["B4"]):
        cell.font = white
        cell.fill = fill(BLUE)
        cell.alignment = Alignment(indent=1, vertical="center")
    row = 5
    for i, (key, sheet) in enumerate(sheets.items()):
        cell = cover.cell(row=row, column=1, value=key)
        cell.hyperlink = f"#'{names[key]}'!A1"
        cell.font = link
        cell.alignment = Alignment(indent=1)
        detail = cover.cell(row=row, column=2, value=sheet.description or sheet.title)
        detail.font = Font(color="404040")
        for c in (cell, detail):
            c.border = Border(bottom=hair)
            if i % 2:
                c.fill = fill(_BAND)
        row += 1

    if highlights:
        cover["D4"] = "Key figures"
        cover["E4"] = ""
        for cell in (cover["D4"], cover["E4"]):
            cell.font = white
            cell.fill = fill(TEAL)
            cell.alignment = Alignment(indent=1, vertical="center")
        r = 5
        for label, (value, number_format) in highlights.items():
            name = cover.cell(row=r, column=4, value=label)
            figure = cover.cell(row=r, column=5, value=float(value))
            figure.number_format = number_format
            name.alignment = Alignment(indent=1, vertical="center")
            figure.alignment = Alignment(horizontal="right", vertical="center")
            figure.font = Font(bold=True, size=12, color=NAVY)
            for c in (name, figure):
                c.fill = fill(_CARD)
                c.border = Border(bottom=Side(style="thin", color="FFFFFF"))
            cover.row_dimensions[r].height = 22
            r += 1
        row = max(row, r)

    row += 1
    for line in notes:
        cover.merge_cells(start_row=row, start_column=1, end_row=row, end_column=2)
        cell = cover.cell(row=row, column=1, value=line)
        cell.alignment = Alignment(wrap_text=True, vertical="top", indent=1)
        cell.font = Font(italic=True, size=9, color=GREY)
        cover.row_dimensions[row].height = 30
        row += 1
    cover.sheet_view.showGridLines = False

    # ---------------- Data sheets ----------------
    for key, sheet in sheets.items():
        ws = book.create_sheet(names[key])
        ws.sheet_properties.tabColor = sheet.tab_color
        table = sheet.table
        has_reference = bool(sheet.references)
        first_value = 3 if has_reference else 2
        last_column = first_value + len(table.columns) - 1

        # Title band
        for r in (1, 2, 3):
            for c in range(1, last_column + 1):
                ws.cell(row=r, column=c).fill = fill(NAVY)
        ws.row_dimensions[2].height = 26
        ws["A1"] = report_title
        ws["A1"].font = Font(size=9, color="DDEBF7")
        ws["A2"] = sheet.title
        ws["A2"].font = Font(bold=True, size=15, color="FFFFFF")
        ws["A3"] = sheet.subtitle
        ws["A3"].font = Font(italic=True, size=9, color="DDEBF7")
        for ref in ("A1", "A2", "A3"):
            ws[ref].alignment = Alignment(indent=1, vertical="center")
        back = ws.cell(row=1, column=last_column + 1, value="Contents")
        back.hyperlink = "#'Contents'!A1"
        back.font = link

        # Header
        header_row = 5
        ws.row_dimensions[header_row].height = 30
        headers = [ws.cell(row=header_row, column=1, value=table.index.name or "")]
        if has_reference:
            headers.append(ws.cell(row=header_row, column=2, value="Reference"))
        for j, column in enumerate(table.columns):
            headers.append(ws.cell(row=header_row, column=first_value + j, value=str(column)))
        for cell in headers:
            cell.font = white
            cell.fill = fill(BLUE)
            cell.alignment = Alignment(horizontal="left" if cell.column == 1 else "right",
                                       vertical="center", wrap_text=True,
                                       indent=1 if cell.column == 1 else 0)
            cell.border = Border(bottom=thin)
        total_label = "Total" if "Total" in [str(c) for c in table.columns] else None

        # Body
        band = False
        for i, (index, values) in enumerate(table.iterrows()):
            r = header_row + 1 + i
            text = str(index)
            is_total = text in sheet.total_rows
            is_section = text in sheet.section_rows
            name_cell = ws.cell(row=r, column=1, value=text)
            row_cells = [ws.cell(row=r, column=c) for c in range(1, last_column + 1)]

            if is_section:
                for cell in row_cells:
                    cell.fill = fill(_SECTION)
                name_cell.font = Font(bold=True, color=NAVY)
                name_cell.alignment = Alignment(indent=1, vertical="center")
                band = False
                continue

            if has_reference:
                ref = ws.cell(row=r, column=2, value=sheet.references.get(text, ""))
                ref.font = Font(size=8, color=GREY)
                ref.alignment = Alignment(vertical="center")
            name_cell.font = Font(bold=True, color=NAVY) if is_total else Font(color="262626")
            name_cell.alignment = Alignment(indent=1 if is_total else 2, vertical="center")

            for j, value in enumerate(values):
                cell = ws.cell(row=r, column=first_value + j)
                if isinstance(value, (int, float, np.integer, np.floating)):
                    if not (isinstance(value, float) and np.isnan(value)):
                        cell.value = float(value)
                        cell.number_format = sheet.number_format
                elif value is not None and not pd.isna(value):
                    cell.value = str(value)
                cell.alignment = Alignment(horizontal="right", vertical="center")
                if is_total or str(table.columns[j]) == total_label:
                    cell.font = Font(bold=True, color=NAVY)

            if is_total:
                closing = text.lower().startswith("total") or text.lower().startswith("closing")
                for cell in row_cells:
                    cell.fill = fill(_TOTAL)
                    cell.border = Border(top=thin,
                                         bottom=Side(style="double", color="8EA9DB")
                                         if closing else thin)
                band = False
            else:
                for cell in row_cells:
                    cell.border = Border(bottom=hair)
                    if band:
                        cell.fill = fill(_BAND)
                band = not band

        if total_label:
            position = first_value + [str(c) for c in table.columns].index(total_label)
            for r in range(header_row, header_row + 1 + len(table)):
                cell = ws.cell(row=r, column=position)
                cell.border = Border(left=thin, top=cell.border.top, bottom=cell.border.bottom)

        end_row = header_row + len(table)
        if sheet.note:
            ws.merge_cells(start_row=end_row + 2, start_column=1, end_row=end_row + 2,
                           end_column=max(last_column, 4))
            note_cell = ws.cell(row=end_row + 2, column=1, value=sheet.note)
            note_cell.font = Font(italic=True, size=9, color=GREY)
            note_cell.alignment = Alignment(wrap_text=True, vertical="top", indent=1)
            ws.row_dimensions[end_row + 2].height = 30

        # Chart
        if sheet.chart:
            spec = sheet.chart
            exclude = {str(x) for x in spec.get("exclude", ("Total",))}
            chart = LineChart() if spec.get("kind", "line") == "line" else BarChart()
            chart.title = spec.get("title", sheet.title)
            chart.height, chart.width = 8.5, 20
            chart.legend.position = "b"
            labels = [str(x) for x in table.index]
            columns = [str(c) for c in table.columns]
            if "rows" in spec:
                keep = [j for j, c in enumerate(columns) if c not in exclude]
                if keep:
                    lo, hi = first_value + keep[0], first_value + keep[-1]
                    for label in spec["rows"]:
                        if label in labels:
                            r = header_row + 1 + labels.index(label)
                            chart.add_data(Reference(ws, min_col=lo, max_col=hi, min_row=r,
                                                     max_row=r),
                                           from_rows=True, titles_from_data=False)
                            chart.series[-1].tx = SeriesLabel(v=label)
                    chart.set_categories(Reference(ws, min_col=lo, max_col=hi,
                                                   min_row=header_row, max_row=header_row))
            else:
                keep = [i for i, x in enumerate(labels) if x not in exclude]
                if keep:
                    lo, hi = header_row + 1 + keep[0], header_row + 1 + keep[-1]
                    for column in spec.get("columns", columns):
                        if column in columns:
                            c = first_value + columns.index(column)
                            chart.add_data(Reference(ws, min_col=c, min_row=header_row,
                                                     max_row=hi), titles_from_data=True)
                    chart.set_categories(Reference(ws, min_col=1, min_row=lo, max_row=hi))
            if chart.series:
                ws.add_chart(chart, f"A{end_row + (5 if sheet.note else 3)}")

        # Widths, panes and printing
        longest = max([len(str(i)) for i in table.index] + [12])
        ws.column_dimensions["A"].width = min(max(longest + 6, 26), 64)
        if has_reference:
            ws.column_dimensions["B"].width = 15
        for j, column in enumerate(table.columns):
            width = max(14, min(len(str(column)) + 4, 24))
            ws.column_dimensions[get_column_letter(first_value + j)].width = width
        ws.column_dimensions[get_column_letter(last_column + 1)].width = 11
        ws.freeze_panes = ws.cell(row=header_row + 1, column=first_value)
        ws.sheet_view.showGridLines = False
        ws.page_setup.orientation = "landscape"
        ws.page_setup.fitToWidth = 1
        ws.page_setup.fitToHeight = 0
        ws.sheet_properties.pageSetUpPr.fitToPage = True
        ws.print_title_rows = f"{header_row}:{header_row}"

    book.save(path)
