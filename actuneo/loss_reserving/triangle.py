"""
Claims Development Triangles

A :class:`Triangle` holds claims (paid, incurred or counts) by origin period
and development period, and provides the basic operations every reserving
method starts from:

- conversion between cumulative and incremental form
- conversion between wide (matrix) and long (one row per cell) layouts
- the latest diagonal
- individual link ratios (age-to-age factors) and their averages

The functionality mirrors the triangle helpers of the R ``ChainLadder``
package (``as.triangle``, ``incr2cum``, ``cum2incr``, ``ata``,
``getLatestCumulative``), see Gesmann et al., *ChainLadder: Statistical
Methods and Models for Claims Reserving in General Insurance*.
"""

import csv
import re
import warnings

import numpy as np
import pandas as pd
from typing import List, Optional, Sequence, Union

# Column names recognised in long format files, after lower-casing and
# replacing spaces and hyphens with underscores
_ORIGIN_NAMES = ("origin", "origin_year", "origin_period", "accident_year", "accident_period",
                 "ay", "loss_year", "year_of_loss", "underwriting_year", "uw_year",
                 "policy_year", "occurrence_year")
_DEVELOPMENT_NAMES = ("development", "development_year", "development_period", "dev",
                      "dev_year", "dev_period", "lag", "age", "development_month",
                      "development_months", "dev_month", "dev_months")
_CALENDAR_NAMES = ("calendar", "calendar_year", "calendar_period", "cy", "payment_year",
                   "paid_year", "transaction_year", "valuation_year", "financial_year")
_VALUE_NAMES = ("value", "amount", "paid", "incurred", "claims", "claim_amount",
                "paid_amount", "incurred_amount", "paid_claims", "incurred_claims", "loss",
                "losses", "count", "claim_count", "number")
_TOTAL_LABEL = re.compile(r"^\s*(grand\s+)?(total|totals|sum)\s*$", re.IGNORECASE)


class Triangle:
    """
    A claims development triangle.

    Rows are origin periods (accident, underwriting or reporting periods) and
    columns are development periods. Cells that have not yet been observed
    are ``NaN``.
    """

    def __init__(self,
                 values: Union[np.ndarray, pd.DataFrame, Sequence[Sequence[float]]],
                 origin: Optional[Sequence] = None,
                 development: Optional[Sequence] = None,
                 cumulative: bool = True,
                 name: str = "Triangle",
                 allow_missing: bool = False):
        """
        Initialize a triangle.

        Args:
            values: 2-dimensional data, origin periods in rows and development
                periods in columns, with NaN for cells not yet observed. A
                DataFrame supplies its index and columns as labels.
            origin: Labels of the origin periods (default 1, 2, ...)
            development: Labels of the development periods (default 1, 2, ...)
            cumulative: True if the values are cumulative, False if incremental
            name: Name/description of the triangle
            allow_missing: Accept cells with no value inside the observed
                part of the triangle (for example a lost year of data). Link
                ratios that need a missing cell are left out. Only for
                cumulative triangles; the bootstrap and the
                inflation-adjusted method do not support missing cells.
        """
        if isinstance(values, pd.DataFrame):
            if origin is None:
                origin = values.index.tolist()
            if development is None:
                development = values.columns.tolist()
            values = values.to_numpy()

        arr = np.array(values, dtype=float)
        if arr.ndim != 2:
            raise ValueError("values must be 2-dimensional (origin x development)")
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError("values must not be empty")
        if np.all(np.isnan(arr[:, 0])):
            raise ValueError("the first development period has no observations")

        n_origin, n_dev = arr.shape
        self.origin = list(origin) if origin is not None else list(range(1, n_origin + 1))
        self.development = (
            list(development) if development is not None else list(range(1, n_dev + 1))
        )
        if len(self.origin) != n_origin:
            raise ValueError("origin must have one label per row of values")
        if len(self.development) != n_dev:
            raise ValueError("development must have one label per column of values")

        # Each row must be observed up to some development period and unknown after
        observed = ~np.isnan(arr)
        for i in range(n_origin):
            if not observed[i].any():
                raise ValueError(f"origin period {self.origin[i]} has no observations")
            last = np.flatnonzero(observed[i])[-1]
            if not allow_missing and not observed[i, :last + 1].all():
                raise ValueError(
                    f"origin period {self.origin[i]} has missing values before its "
                    "latest observation"
                )

        if allow_missing and not cumulative:
            raise ValueError("missing cells are only supported for cumulative triangles")
        self.values = arr
        self.is_cumulative = bool(cumulative)
        self.name = name
        self.allow_missing = bool(allow_missing)
        self.has_missing = bool(allow_missing and any(
            not observed[i, :np.flatnonzero(observed[i])[-1] + 1].all()
            for i in range(n_origin)))

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_long(cls,
                  df: pd.DataFrame,
                  origin: str = "origin",
                  development: str = "development",
                  value: str = "value",
                  cumulative: bool = True,
                  name: str = "Triangle",
                  calendar: Optional[str] = None) -> 'Triangle':
        """
        Create a Triangle from long format data, one row per cell.

        Several rows for the same cell (for example individual claim payments)
        are added together. For incremental data, an empty cell before the
        latest cell with data in its origin period is treated as zero. An
        origin period with no movement in its most recent development periods
        therefore needs explicit zero rows for those periods.

        Args:
            df: DataFrame with origin, development and value columns
            origin: Column name of the origin period
            development: Column name of the development period
            value: Column name of the claims amount
            cumulative: True if the values are cumulative, False if incremental
            name: Name for the triangle
            calendar: Column with the calendar year (year of payment or
                valuation) to use instead of a development column. Origin
                and calendar periods must be whole years (or other whole
                numbers of the same period); development period 1 is the
                origin year itself. Pass ``development=None`` with it.

        Returns:
            Triangle instance
        """
        if calendar is not None:
            if development is not None and development in df.columns:
                raise ValueError("Give either development or calendar, not both")
            years = pd.to_numeric(df[origin], errors="coerce")
            paid = pd.to_numeric(df[calendar], errors="coerce")
            if years.isna().any() or paid.isna().any() or (years % 1 != 0).any() \
                    or (paid % 1 != 0).any():
                raise ValueError("origin and calendar periods must be whole numbers such as "
                                 "years; for dates use Triangle.from_transactions")
            lag = (paid - years + 1).astype(int)
            if (lag < 1).any():
                raise ValueError("a calendar period is before its origin period")
            df = df.assign(_development=lag.to_numpy(), **{origin: years.astype(int).to_numpy()})
            development = "_development"
        wide = df.groupby([origin, development])[value].sum(min_count=1).unstack()
        wide = wide.sort_index(axis=0).sort_index(axis=1)
        if not cumulative:
            # An incremental cell with no transactions inside the observed
            # part of the triangle is a zero, not an unknown
            arr = wide.to_numpy(dtype=float, copy=True)
            observed = ~np.isnan(arr)
            for i in range(arr.shape[0]):
                if observed[i].any():
                    last = np.flatnonzero(observed[i])[-1]
                    arr[i, :last + 1] = np.nan_to_num(arr[i, :last + 1])
            wide = pd.DataFrame(arr, index=wide.index, columns=wide.columns)
        if calendar is not None:
            # Every development period up to the longest seen gets a column
            wide = wide.reindex(columns=range(1, int(wide.columns.max()) + 1))
            wide.columns.name = None
        return cls(wide, cumulative=cumulative, name=name)

    @classmethod
    def from_transactions(cls,
                          df: pd.DataFrame,
                          origin_date: str,
                          transaction_date: str,
                          value: str,
                          origin_grain: str = "Y",
                          development_grain: str = "Y",
                          valuation_date=None,
                          cumulative: bool = True,
                          name: str = "Triangle") -> 'Triangle':
        """
        Build a triangle from a listing of dated transactions.

        Each row is one movement on a claim, for example a payment with the
        date of loss and the date of payment, or a claim count of 1 with the
        date of loss and the date reported. Rows are grouped into origin
        periods by ``origin_date`` and into development periods by the time
        from the start of the origin period to ``transaction_date``.

        Periods with no transactions up to the valuation date are zero, so
        quiet periods and origin periods without claims are handled.

        Args:
            df: DataFrame with one row per transaction
            origin_date: Column with the date that fixes the origin period
                (date of loss, policy inception or reporting date)
            transaction_date: Column with the date of the transaction
            value: Column with the amount (or count) of the transaction
            origin_grain: Length of the origin periods: "Y" (year),
                "Q" (quarter) or "M" (month)
            development_grain: Length of the development periods: "Y", "Q"
                or "M". Must not be longer than the origin periods.
            valuation_date: Date up to which the data is complete (default:
                the end of the development period of the latest transaction)
            cumulative: Return a cumulative triangle (True) or incremental
            name: Name for the triangle

        Returns:
            Triangle with origin periods labelled by period (for example
            "2023" or "2023Q2") and development periods numbered from 1
        """
        months = {"Y": 12, "Q": 3, "M": 1}
        if origin_grain not in months or development_grain not in months:
            raise ValueError("grains must be 'Y', 'Q' or 'M'")
        if months[development_grain] > months[origin_grain]:
            raise ValueError("development periods must not be longer than origin periods")
        if len(df) == 0:
            raise ValueError("df has no transactions")

        origin_dates = pd.to_datetime(df[origin_date])
        transaction_dates = pd.to_datetime(df[transaction_date])
        if origin_dates.isna().any() or transaction_dates.isna().any():
            raise ValueError("origin and transaction dates must not be missing")

        def month_number(dates):
            return dates.dt.year * 12 + dates.dt.month - 1

        origin_len, dev_len = months[origin_grain], months[development_grain]
        origin_month = month_number(origin_dates)
        origin_idx = origin_month // origin_len
        # Development lag counted from the start of the origin period
        lag = month_number(transaction_dates) // dev_len - (origin_idx * origin_len) // dev_len
        if (transaction_dates < origin_dates).any() or (lag < 0).any():
            raise ValueError("a transaction is dated before its origin date")

        if valuation_date is None:
            valuation_period = int((month_number(transaction_dates) // dev_len).max())
        else:
            valuation = pd.Timestamp(valuation_date)
            valuation_period = (valuation.year * 12 + valuation.month - 1) // dev_len
            if (month_number(transaction_dates) // dev_len > valuation_period).any():
                raise ValueError("a transaction is dated after the valuation date")

        # Origin periods run up to the valuation date, with or without claims
        first = int(origin_idx.min())
        last = max(int(origin_idx.max()), (valuation_period * dev_len) // origin_len)
        n_origin = last - first + 1
        latest_lag = valuation_period - (np.arange(first, last + 1) * origin_len) // dev_len
        n_dev = int(latest_lag.max()) + 1

        values = np.zeros((n_origin, n_dev))
        np.add.at(values, (origin_idx.to_numpy() - first, lag.to_numpy()),
                  df[value].to_numpy(dtype=float))
        for i in range(n_origin):
            values[i, latest_lag[i] + 1:] = np.nan

        def label(idx):
            year, part = divmod(idx * origin_len, 12)
            if origin_grain == "Y":
                return str(year)
            if origin_grain == "Q":
                return f"{year}Q{part // 3 + 1}"
            return f"{year}-{part + 1:02d}"

        triangle = cls(values, origin=[label(i) for i in range(first, last + 1)],
                       development=range(1, n_dev + 1), cumulative=False, name=name)
        return triangle.to_cumulative() if cumulative else triangle

    @staticmethod
    def _number(cell, decimal: str = ".") -> float:
        """Read one cell as a number: "1,234", "(56)", "-" and blanks are handled."""
        if cell is None:
            return np.nan
        if isinstance(cell, (int, float, np.integer, np.floating)) and not isinstance(cell, bool):
            return float(cell)
        text = str(cell).strip()
        negative = text.startswith("(") and text.endswith(")")
        text = re.sub(r"[^0-9,.\-eE+]", "", text.strip("()"))
        if text.lower() in ("", "-", "--", ".", "e"):
            return np.nan
        if decimal == ",":
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
        try:
            number = float(text)
        except ValueError:
            return np.nan
        return -number if negative else number

    @staticmethod
    def _label(label):
        """Use whole numbers for row and column labels where the file has them."""
        if isinstance(label, str):
            label = label.strip()
        try:
            number = float(label)
        except (TypeError, ValueError):
            return label
        if np.isnan(number):
            return label
        return int(number) if number == int(number) else number

    @staticmethod
    def _key(name) -> str:
        return re.sub(r"[\s\-]+", "_", str(name).strip().lower())

    @classmethod
    def _find_column(cls, columns, wanted, candidates, what):
        if wanted is not None:
            if wanted not in columns:
                raise ValueError(f"no column '{wanted}' in the file; it has {list(columns)}")
            return wanted
        for column in columns:
            if cls._key(column) in candidates:
                return column
        return None

    @classmethod
    def _from_rows(cls, rows: list, cumulative: bool, name: str, layout: str,
                   origin, development, calendar, value, decimal: str,
                   calendar_columns: Optional[bool],
                   origin_in_columns: Optional[bool] = None) -> 'Triangle':
        """Build a triangle from the cells of a sheet or CSV file."""
        width = max((len(r) for r in rows), default=0)
        cells = pd.DataFrame([list(r) + [None] * (width - len(r)) for r in rows], dtype=object)
        blank = cells.map(lambda c: c is None or (isinstance(c, float) and np.isnan(c))
                          or (isinstance(c, str) and not c.strip()))
        cells = cells.loc[~blank.all(axis=1), ~blank.all(axis=0)]
        blank = blank.loc[cells.index, cells.columns]
        if cells.empty:
            raise ValueError("the file has no data")

        # The header is the first row about as wide as the table; titles and
        # notes above it fill one or two cells
        filled = (~blank).sum(axis=1)
        header_at = int(np.flatnonzero(filled.to_numpy() >= max(2, 0.5 * filled.max()))[0])
        header = [None if b else c for c, b in zip(cells.iloc[header_at], blank.iloc[header_at])]
        body = cells.iloc[header_at + 1:]

        if layout not in ("auto", "wide", "long"):
            raise ValueError("layout must be 'auto', 'wide' or 'long'")
        names = [str(h).strip() if h is not None else f"column_{i}" for i, h in enumerate(header)]
        if layout in ("auto", "long"):
            origin_col = cls._find_column(names, origin, _ORIGIN_NAMES, "origin")
            dev_col = cls._find_column(names, development, _DEVELOPMENT_NAMES, "development")
            cal_col = cls._find_column(names, calendar, _CALENDAR_NAMES, "calendar")
            value_col = cls._find_column(names, value, _VALUE_NAMES, "value")
            is_long = origin_col is not None and value_col is not None and (
                dev_col is not None or cal_col is not None)
            if layout == "long" and not is_long:
                raise ValueError(
                    "could not find the origin, development (or calendar) and value columns; "
                    f"the file has {names}. Name them with origin=, development= (or "
                    "calendar=) and value=")
            if is_long:
                frame = pd.DataFrame(body.to_numpy(), columns=names)
                frame = frame[frame[origin_col].map(
                    lambda c: c is not None and str(c).strip() != ""
                    and not _TOTAL_LABEL.match(str(c)))]
                frame[value_col] = frame[value_col].map(lambda c: cls._number(c, decimal))
                frame[origin_col] = frame[origin_col].map(cls._label)
                use_calendar = dev_col is None
                period = cal_col if use_calendar else dev_col
                frame[period] = frame[period].map(cls._label)
                return cls.from_long(frame, origin=origin_col,
                                     development=None if use_calendar else dev_col,
                                     calendar=cal_col if use_calendar else None,
                                     value=value_col, cumulative=cumulative, name=name)

        # Wide layout: origin periods down the first column
        labels = [cls._label(c) for c in body.iloc[:, 0]]
        columns = [cls._label(h) for h in header[1:]]
        values = body.iloc[:, 1:].map(lambda c: cls._number(c, decimal)).to_numpy(dtype=float)
        frame = pd.DataFrame(values, index=labels, columns=columns)
        keep_rows = [not _TOTAL_LABEL.match(str(r)) for r in frame.index]
        keep_cols = [c is not None and not _TOTAL_LABEL.match(str(c)) for c in frame.columns]
        frame = frame.loc[keep_rows, keep_cols]
        frame = frame.loc[frame.notna().any(axis=1), frame.notna().any(axis=0)]

        # Published claims development tables (IFRS 17.130) put the accident
        # years across and the years of development down
        def is_year(c):
            return isinstance(c, int) and 1900 <= c <= 2200
        if origin_in_columns is None:
            numbered = [r for r in frame.index if isinstance(r, int)]
            origin_in_columns = (all(is_year(c) for c in frame.columns) and len(numbered) >= 2
                                 and all(0 <= r < 100 for r in numbered))
        if origin_in_columns:
            frame = frame.loc[[isinstance(r, int) for r in frame.index]].T
            frame = frame.loc[:, frame.notna().any(axis=0)]
            calendar_columns = False
        if calendar_columns is None:
            calendar_columns = (all(isinstance(c, int) and 1900 <= c <= 2200 for c in frame.columns)
                                and all(isinstance(r, int) and 1900 <= r <= 2200
                                        for r in frame.index))
        if calendar_columns:
            long = frame.stack().dropna().rename("value").rename_axis(["origin", "calendar"]).reset_index()
            return cls.from_long(long, origin="origin", development=None, calendar="calendar",
                                 value="value", cumulative=cumulative, name=name)
        return cls._tidy(frame.to_numpy(), list(frame.index), list(frame.columns),
                         cumulative, name)

    @classmethod
    def _tidy(cls, arr, origin, development, cumulative, name) -> 'Triangle':
        """
        Read the empty and zero cells of a triangle loaded from data.

        Incremental: an empty cell before the latest one with data is a
        zero. Cumulative: zeros after the latest positive amount of an origin
        period are cells not yet observed (spreadsheets often fill them with
        zeros), since cumulative claims do not fall back to nil.
        """
        arr = np.array(arr, dtype=float)
        changed = []
        for i in range(arr.shape[0]):
            seen = np.flatnonzero(~np.isnan(arr[i]))
            if not seen.size:
                continue
            if cumulative:
                nonzero = np.flatnonzero(np.nan_to_num(arr[i]) != 0)
                if nonzero.size and (arr[i, nonzero[-1] + 1:] == 0).any():
                    arr[i, nonzero[-1] + 1:] = np.nan
                    changed.append(origin[i])
            else:
                arr[i, :seen[-1] + 1] = np.nan_to_num(arr[i, :seen[-1] + 1])
        if changed:
            warnings.warn(
                f"Zeros after the latest amount of origin periods {changed} were read as not "
                "yet observed. Pass a triangle with empty cells if they are genuine.",
                UserWarning, stacklevel=4)
        return cls(arr, origin=origin, development=development, cumulative=cumulative, name=name)

    @classmethod
    def from_excel(cls,
                   path: str,
                   sheet_name=0,
                   cumulative: bool = True,
                   name: Optional[str] = None,
                   layout: str = "auto",
                   origin: Optional[str] = None,
                   development: Optional[str] = None,
                   calendar: Optional[str] = None,
                   value: Optional[str] = None,
                   calendar_columns: Optional[bool] = None,
                   origin_in_columns: Optional[bool] = None) -> 'Triangle':
        """
        Read a triangle from an Excel sheet (needs openpyxl).

        Two layouts are read:

        - **wide**: origin periods down the first column, development periods
          (or calendar years, see ``calendar_columns``) across the header row,
          and empty cells where claims have not yet developed;
        - **long**: one row per cell or per transaction, with columns for the
          origin period, the development period (or calendar year) and the
          amount. Common column names such as ``accident_year``, ``dev``,
          ``payment_year`` and ``paid`` are recognised; otherwise name them.

        Titles and notes above the table, Total rows and columns, amounts
        stored as text ("1,234", "(56)") and dashes for empty cells are
        handled. In a cumulative triangle, zeros after the latest amount of
        an origin period are read as not yet observed, with a warning.

        Args:
            path: Excel workbook
            sheet_name: Name or position of the sheet
            cumulative: True if the values are cumulative, False if incremental
            name: Name for the triangle (defaults to the sheet name)
            layout: "auto" (long when the origin, period and value columns are
                found, wide otherwise), "wide" or "long"
            origin, development, calendar, value: Column names for the long
                layout, where they are not recognised automatically. Give
                ``calendar`` (the year of payment or valuation) instead of
                ``development`` when the file has calendar years.
            calendar_columns: In the wide layout, True if the columns are
                calendar years rather than development periods. By default
                this is assumed when both rows and columns are years.
            origin_in_columns: True if the origin years run across the header
                and the development years down the first column, as in the
                claims development tables of published financial statements.
                Rows whose label is not a whole number (totals, current
                estimate) are then ignored. By default this is assumed when
                the columns are years and the rows are numbered 0, 1, 2, ...

        Returns:
            Triangle instance
        """
        try:
            sheet = pd.read_excel(path, sheet_name=sheet_name, header=None, dtype=object)
        except ImportError as exc:
            raise ImportError(
                "Reading Excel files requires openpyxl. Install it with: "
                "pip install actuneo[excel]"
            ) from exc
        label = name or (sheet_name if isinstance(sheet_name, str) else "Triangle")
        rows = sheet.where(sheet.notna(), None).values.tolist()
        return cls._from_rows(rows, cumulative, label, layout, origin, development, calendar,
                              value, ".", calendar_columns, origin_in_columns)

    @classmethod
    def from_csv(cls,
                 path: str,
                 cumulative: bool = True,
                 name: str = "Triangle",
                 layout: str = "auto",
                 origin: Optional[str] = None,
                 development: Optional[str] = None,
                 calendar: Optional[str] = None,
                 value: Optional[str] = None,
                 calendar_columns: Optional[bool] = None,
                 origin_in_columns: Optional[bool] = None,
                 sep: Optional[str] = None,
                 decimal: Optional[str] = None) -> 'Triangle':
        """
        Read a triangle from a CSV file, in the wide or long layout described
        under :meth:`from_excel`.

        Args:
            path: CSV file
            cumulative: True if the values are cumulative, False if incremental
            name: Name for the triangle
            layout, origin, development, calendar, value, calendar_columns,
            origin_in_columns: As for :meth:`from_excel`
            sep: Field separator (detected from the file by default: comma,
                semicolon, tab or bar)
            decimal: Decimal mark, "." or "," (default "," for files
                separated by semicolons, "." otherwise)

        Returns:
            Triangle instance
        """
        with open(path, newline="", encoding="utf-8-sig") as handle:
            text = handle.read()
        if sep is None:
            try:
                sep = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|").delimiter
            except csv.Error:
                sep = ","
        if decimal is None:
            decimal = "," if sep == ";" else "."
        rows = list(csv.reader(text.splitlines(), delimiter=sep))
        return cls._from_rows(rows, cumulative, name, layout, origin, development, calendar,
                              value, decimal, calendar_columns, origin_in_columns)

    def to_excel(self, path: str, sheet_name: str = "Triangle") -> None:
        """
        Write the triangle to an Excel sheet in the layout that
        :meth:`from_excel` reads (needs openpyxl). Useful as a template.

        Args:
            path: File name of the workbook, ending in .xlsx
            sheet_name: Name of the sheet
        """
        try:
            self.to_frame().to_excel(path, sheet_name=sheet_name)
        except ImportError as exc:
            raise ImportError(
                "Writing Excel files requires openpyxl. Install it with: "
                "pip install actuneo[excel]"
            ) from exc

    def to_csv(self, path: str) -> None:
        """Write the triangle to a CSV file in the layout that :meth:`from_csv` reads."""
        self.to_frame().to_csv(path)

    @classmethod
    def from_dataframe(cls,
                       df: pd.DataFrame,
                       cumulative: bool = True,
                       name: str = "Triangle") -> 'Triangle':
        """
        Create a Triangle from a wide DataFrame (origin index, development columns).
        """
        return cls(df, cumulative=cumulative, name=name)

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------
    @property
    def shape(self) -> tuple:
        """(number of origin periods, number of development periods)"""
        return self.values.shape

    @property
    def n_origin(self) -> int:
        return self.values.shape[0]

    @property
    def n_development(self) -> int:
        return self.values.shape[1]

    def _with_values(self, values: np.ndarray, cumulative: bool) -> 'Triangle':
        return Triangle(values, self.origin, self.development, cumulative, self.name)

    def _require_complete(self, what: str) -> None:
        if getattr(self, "has_missing", False):
            raise ValueError(f"{what} does not support triangles with missing cells")

    def _latest_index(self) -> np.ndarray:
        """Column position of the latest observation of each origin period."""
        observed = ~np.isnan(self.values)
        return observed.shape[1] - 1 - np.argmax(observed[:, ::-1], axis=1)

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------
    def to_cumulative(self) -> 'Triangle':
        """Cumulative triangle (R: ``incr2cum``)."""
        if self.is_cumulative:
            return self
        return self._with_values(np.cumsum(self.values, axis=1), True)

    def to_incremental(self) -> 'Triangle':
        """Incremental triangle (R: ``cum2incr``)."""
        if not self.is_cumulative:
            return self
        self._require_complete("Conversion to incremental claims")
        incremental = self.values.copy()
        incremental[:, 1:] = np.diff(self.values, axis=1)
        return self._with_values(incremental, False)

    def to_frame(self) -> pd.DataFrame:
        """Wide DataFrame with origin periods as index and development periods as columns."""
        return pd.DataFrame(
            self.values,
            index=pd.Index(self.origin, name="origin"),
            columns=pd.Index(self.development, name="development"),
        )

    def to_long(self, dropna: bool = True) -> pd.DataFrame:
        """
        Long format DataFrame, one row per cell (R: ``as.LongTriangle``).

        Args:
            dropna: Leave out cells that have not been observed
        """
        long = pd.DataFrame(
            {
                "origin": np.repeat(np.array(self.origin, dtype=object), self.n_development),
                "development": np.tile(np.array(self.development, dtype=object), self.n_origin),
                "value": self.values.ravel(),
            }
        ).infer_objects()
        if dropna:
            long = long.dropna(subset=["value"]).reset_index(drop=True)
        return long

    # ------------------------------------------------------------------
    # Diagonals
    # ------------------------------------------------------------------
    def latest_diagonal(self) -> pd.Series:
        """
        Latest observed value of each origin period (R: ``getLatestCumulative``).

        For an incremental triangle this is the latest incremental amount.
        """
        idx = self._latest_index()
        latest = self.values[np.arange(self.n_origin), idx]
        return pd.Series(latest, index=pd.Index(self.origin, name="origin"), name="latest")

    def latest_development(self) -> pd.Series:
        """Development period reached by each origin period."""
        idx = self._latest_index()
        return pd.Series([self.development[j] for j in idx],
                         index=pd.Index(self.origin, name="origin"), name="development")

    # ------------------------------------------------------------------
    # Link ratios
    # ------------------------------------------------------------------
    def _link_labels(self) -> List[str]:
        dev = self.development
        return [f"{dev[k]}-{dev[k + 1]}" for k in range(len(dev) - 1)]

    def link_ratios(self) -> pd.DataFrame:
        """
        Individual link ratios (age-to-age factors) C[i, k+1] / C[i, k].

        Returns:
            DataFrame of link ratios, origin periods in rows
        """
        cum = self.to_cumulative().values
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = cum[:, 1:] / cum[:, :-1]
        ratios[~np.isfinite(ratios)] = np.nan
        return pd.DataFrame(ratios,
                            index=pd.Index(self.origin, name="origin"),
                            columns=self._link_labels())

    def development_factors(self,
                            average: str = "volume",
                            n_periods: Optional[int] = None) -> pd.Series:
        """
        Average link ratio for each development step.

        Args:
            average: "volume" for the volume-weighted average (the chain-ladder
                factor, sum of C[i, k+1] over sum of C[i, k]), "simple" for
                the arithmetic mean of the individual link ratios
            n_periods: Use only the latest n origin periods available for
                each development step (None for all)

        Returns:
            Series of development factors
        """
        if average not in ("volume", "simple"):
            raise ValueError("average must be 'volume' or 'simple'")
        if n_periods is not None and n_periods < 1:
            raise ValueError("n_periods must be at least 1")

        cum = self.to_cumulative().values
        n_steps = cum.shape[1] - 1
        factors = np.full(n_steps, np.nan)
        for k in range(n_steps):
            rows = np.flatnonzero(~np.isnan(cum[:, k]) & ~np.isnan(cum[:, k + 1]))
            if n_periods is not None:
                rows = rows[-n_periods:]
            if len(rows) == 0:
                continue
            start, end = cum[rows, k], cum[rows, k + 1]
            if average == "volume":
                if start.sum() != 0:
                    factors[k] = end.sum() / start.sum()
            else:
                usable = start != 0
                if usable.any():
                    factors[k] = np.mean(end[usable] / start[usable])
        return pd.Series(factors, index=self._link_labels(), name=f"{average}")

    def age_to_age(self) -> pd.DataFrame:
        """
        Link ratio table with simple and volume-weighted averages (R: ``ata``).

        Returns:
            DataFrame of individual link ratios with two summary rows,
            "simple" and "volume"
        """
        table = self.link_ratios()
        table.loc["simple"] = self.development_factors("simple")
        table.loc["volume"] = self.development_factors("volume")
        return table

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def plot(self, ax=None):
        """
        Plot the development of each origin period (needs matplotlib).

        Args:
            ax: Matplotlib axes to draw on (a new figure if omitted)

        Returns:
            The matplotlib axes
        """
        from ._plotting import plot_development
        kind = "Cumulative" if self.is_cumulative else "Incremental"
        return plot_development(self.to_frame(), f"{self.name}: {kind.lower()} development",
                                f"{kind} claims", ax)

    def __repr__(self) -> str:
        kind = "cumulative" if self.is_cumulative else "incremental"
        return (f"Triangle(name='{self.name}', {kind}, "
                f"{self.n_origin} origin x {self.n_development} development)\n"
                f"{self.to_frame().to_string(na_rep='')}")
