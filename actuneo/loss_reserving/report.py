"""
Reserving Reports

Compares the results of several reserving methods and writes them, with the
triangle they were fitted to, to a formatted Excel workbook.
"""

from typing import Dict

import numpy as np
import pandas as pd

from .triangle import Triangle

_NOTES = (
    "Prepared with ACTUNEO from the triangle supplied. Reserves are undiscounted unless "
    "stated otherwise.",
    "ACTUNEO has not been independently reviewed and is not certified for statutory or "
    "audited reporting. The choice of method, development factors and tail is a matter "
    "of actuarial judgement. Check these figures before relying on them.",
)


def compare_methods(models: Dict[str, object]) -> pd.DataFrame:
    """
    Ultimate claims and reserves of several methods side by side.

    Args:
        models: Fitted reserving models keyed by the name to show for each,
            for example ``{"Chain ladder": cl, "Bornhuetter-Ferguson": bf}``

    Returns:
        DataFrame by origin period with a "Total" row: latest claims, then
        the ultimate and the reserve (IBNR) of each method
    """
    if not models:
        raise ValueError("At least one model is needed")
    first = next(iter(models.values()))
    latest = first.latest
    table = pd.DataFrame({"Latest": latest})
    for name, model in models.items():
        if hasattr(model, "ibnr_simulations"):
            ibnr = model.ibnr_simulations.mean()
            ibnr.index = latest.index
            ultimate = latest + ibnr
        else:
            ultimate, ibnr = model.ultimate, model.ibnr
        if not ultimate.index.equals(latest.index):
            raise ValueError(f"'{name}' was fitted to a triangle with different origin periods")
        table[f"{name} ultimate"] = ultimate
        table[f"{name} IBNR"] = ibnr
    table.loc["Total"] = table.sum()
    return table


def export_reserving_report(path: str,
                            triangle: Triangle,
                            models: Dict[str, object],
                            title: str = "Reserving report",
                            currency: str = "") -> None:
    """
    Write a formatted reserving workbook (needs openpyxl).

    Sheets: contents, the cumulative and incremental triangles, link
    ratios, a comparison of the methods, and for each method its summary and,
    where it has them, its projected triangle, development factors and
    expected cash flows.

    Args:
        path: File name of the workbook, ending in .xlsx
        triangle: The triangle the models were fitted to
        models: Fitted reserving models keyed by the name to show for each
        title: Heading for the report
        currency: Currency of the amounts
    """
    from ..utils import (Sheet, write_report, DECIMAL_FORMAT, FACTOR_FORMAT, NUMBER_FORMAT,
                         BLUE, TEAL, GREY)

    unit = f"In {currency}" if currency else ""
    cumulative = triangle.to_cumulative()

    def frame(table):
        table = table.copy()
        table.index.name = table.index.name or "origin"
        return table

    comparison = frame(compare_methods(models))
    sheets = {
        "Comparison of methods": Sheet(
            comparison, "Ultimate claims and reserves by method", unit, total_rows=["Total"],
            chart={"kind": "bar", "title": "Reserve (IBNR) by origin period and method",
                   "columns": [c for c in comparison.columns if c.endswith("IBNR")]},
            description="Latest claims, ultimate and reserve for each method"),
        "Cumulative triangle": Sheet(
            frame(cumulative.to_frame()), "Cumulative claims",
            unit or "Origin period by development period", tab_color=GREY,
            description="The triangle the methods were fitted to"),
        "Incremental triangle": Sheet(
            frame(cumulative.to_incremental().to_frame()), "Incremental claims", unit,
            tab_color=GREY, description="Claims in each development period"),
        "Link ratios": Sheet(
            frame(cumulative.age_to_age()), "Link ratios (age-to-age factors)",
            "Individual ratios with simple and volume-weighted averages",
            number_format=FACTOR_FORMAT, total_rows=["simple", "volume"], tab_color=GREY,
            description="Development from one period to the next, with averages"),
    }
    highlights = {"Latest claims": (comparison.loc["Total", "Latest"], NUMBER_FORMAT)}
    for name in models:
        highlights[f"{name} reserve"] = (comparison.loc["Total", f"{name} IBNR"], NUMBER_FORMAT)

    for name, model in models.items():
        summary = frame(model.summary())
        ratio_columns = [c for c in summary.columns
                         if c in ("dev_to_date", "cv_ibnr", "cdf", "ultimate_average_cost")]
        # Ratios would be lost in a whole-number format, so those tables keep decimals
        sheets[f"{name} - Summary"] = Sheet(
            summary, f"{name}: results by origin period", unit,
            number_format=DECIMAL_FORMAT if ratio_columns else NUMBER_FORMAT,
            total_rows=["Total"], tab_color=BLUE,
            description="Latest, ultimate and reserve by origin period",
        )
        if hasattr(model, "full_triangle"):
            sheets[f"{name} - Projection"] = Sheet(
                frame(model.full_triangle), f"{name}: projected cumulative claims", unit,
                tab_color=BLUE, description="The triangle completed to ultimate")
        if hasattr(model, "factors") and hasattr(model, "cdf"):
            factors = pd.DataFrame({"factor": model.factors})
            factors["to_ultimate"] = model.cdf.to_numpy()[:-1]
            for column in ("sigma", "f_se"):
                if hasattr(model, column):
                    factors[column] = getattr(model, column)
            if getattr(model, "tail", 1.0) != 1.0:
                factors.loc["tail"] = [model.tail, model.tail] + [np.nan] * (factors.shape[1] - 2)
            factors.index.name = "development step"
            sheets[f"{name} - Factors"] = Sheet(
                factors, f"{name}: development factors", "", number_format=FACTOR_FORMAT,
                tab_color=BLUE, description="Development factors and factors to ultimate")
        if hasattr(model, "cash_flows"):
            flows = model.cash_flows().to_frame("expected payments")
            flows.loc["Total"] = flows.sum()
            flows.index.name = "future period"
            sheets[f"{name} - Cash flows"] = Sheet(
                flows, f"{name}: expected future payments",
                "By period after the valuation date" + (f". {unit}" if unit else ""),
                total_rows=["Total"], tab_color=TEAL,
                chart={"kind": "bar", "title": "Expected future payments",
                       "columns": ["expected payments"]},
                description="Expected payments by future period")

    write_report(path, sheets, report_title=title,
                 report_subtitle=f"Triangle: {triangle.name}" + (f" | {currency}" if currency
                                                                 else ""),
                 notes=_NOTES, highlights=highlights)
