"""
Actuarial valuation of a defined benefit pension scheme

:class:`DBValuation` values the benefits of active members, deferred
pensioners and pensioners, compares the total with the assets, and gives the
contribution rate that the funding method calls for.

Timing conventions
------------------
Ages of active members are whole numbers at the valuation date. Members who
leave during a year of age (by withdrawal, death, ill health or early
retirement) are taken to leave half way through it. Those still in service at
the normal retirement age retire on that birthday. Salaries are reviewed once
a year, at the valuation date and on each anniversary, and are earned evenly
over the year.

Funding methods
---------------
``projected_unit``
    The liability is the value of the benefits earned by service to date,
    based on salaries projected to the date of leaving. A benefit is spread
    evenly over the service that earns it. The normal cost is the value of
    the benefit earned in the coming year. This is the method required by
    IAS 19.
``current_unit``
    As the projected unit method, with salaries held at their present level.
    Future salary increases on past service then show up as a cost each
    year instead of being funded in advance.
``entry_age``
    The normal cost is the level percentage of salary that would pay for all
    of a member's benefits if contributed from the age at entry. The
    liability is the value of all benefits less the value of future normal
    costs.
``attained_age``
    The liability is that of the projected unit method. The normal cost is
    the level percentage of future salaries that pays for the benefits of
    future service.

The aggregate method, which has no liability for individual members, is given
by :meth:`DBValuation.aggregate_contribution_rate`.
"""

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .annuity import pension_annuity, survival_probabilities
from .scheme import DefinedBenefitScheme, ValuationBasis, normalise_sex, pick_by_sex
from .service_table import CAUSES, SalaryScale, ServiceTable

METHODS = ("projected_unit", "current_unit", "entry_age", "attained_age")


def _frame(data, required: Sequence[str], name: str) -> pd.DataFrame:
    """A copy of the member data with a sex column, after checking it."""
    if data is None:
        return pd.DataFrame(columns=list(required) + ["sex"])
    frame = pd.DataFrame(data).copy()
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} data needs the columns {missing}")
    if "sex" not in frame.columns:
        frame["sex"] = "M"
    frame["sex"] = [normalise_sex(sex) for sex in frame["sex"]]
    for column in required:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or (values < 0).any():
            raise ValueError(f"{name} column '{column}' must hold non-negative numbers")
        frame[column] = values.astype(float)
    return frame.reset_index(drop=True)


def _optional(row, column: str, default):
    """A value from a row, or the default if the column is absent or empty."""
    if column not in row.index:
        return default
    value = row[column]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    return value


class DBValuation:
    """
    Actuarial valuation of a defined benefit scheme.

    Member data is given as tables (anything pandas can turn into a
    DataFrame), one row per member.

    Active members need the columns ``age`` (a whole number), ``service``
    (years of pensionable service to date) and ``salary`` (the annual rate of
    pensionable salary from the valuation date). Optional: ``sex`` ("M" or
    "F", "M" if omitted) and ``contributions`` (the member's own
    contributions with interest to date, used when leavers take a refund).

    Pensioners need ``age`` and ``pension`` (the annual pension now in
    payment). Optional: ``sex``; ``spouse_age`` (if known; otherwise the
    assumed age difference and proportion married are used); ``reversion``
    (False for a pension that dies with the pensioner, such as one already
    being paid to a widow or widower); ``guarantee_remaining`` (years of
    guarantee left); ``ill_health`` (True to use the ill-health mortality
    table).

    Deferred pensioners need ``age`` and ``pension`` (the annual pension
    including revaluation to the valuation date). Optional: ``sex``.
    """

    def __init__(self,
                 scheme: DefinedBenefitScheme,
                 basis: ValuationBasis,
                 actives=None,
                 pensioners=None,
                 deferreds=None,
                 assets: float = 0.0,
                 method: str = "projected_unit",
                 reserves: float = 0.0):
        """
        Args:
            scheme: The benefits promised
            basis: The valuation assumptions
            actives: Data for active members
            pensioners: Data for pensioners
            deferreds: Data for deferred pensioners
            assets: Value of the scheme's net assets
            method: Funding method, one of ``METHODS``
            reserves: Contingency and other reserves held in addition to the
                members' liabilities
        """
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.scheme = scheme
        self.basis = basis
        self.method = method
        self.assets = float(assets)
        self.reserves = float(reserves)
        self._actives = _frame(actives, ["age", "service", "salary"], "active member")
        self._pensioners = _frame(pensioners, ["age", "pension"], "pensioner")
        self._deferreds = _frame(deferreds, ["age", "pension"], "deferred pensioner")
        if (self._actives["age"] != np.floor(self._actives["age"])).any():
            raise ValueError("ages of active members must be whole numbers")
        tables = basis.service_table
        if tables is not None:
            for table in (tables.values() if isinstance(tables, Mapping) else [tables]):
                if table.retirement_age != scheme.normal_retirement_age:
                    raise ValueError(
                        f"the service table retires members at {table.retirement_age} but the "
                        f"scheme's normal retirement age is {scheme.normal_retirement_age}"
                    )
        self._factors: Dict[Tuple, float] = {}
        self._no_decrements: Dict[int, ServiceTable] = {}
        self._active_results: Optional[pd.DataFrame] = None
        self._pensioner_results: Optional[pd.DataFrame] = None
        self._deferred_results: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Annuity factors
    # ------------------------------------------------------------------
    def _annuity(self, table, age: float, sex: str, *, guarantee: float = 0.0,
                 reversion: bool = True, spouse_age: Optional[float] = None,
                 proportion_married: Optional[float] = None) -> float:
        scheme, basis = self.scheme, self.basis
        fraction = scheme.spouse_fraction if reversion else 0.0
        if proportion_married is None:
            proportion_married = basis.proportion_married
        if fraction > 0 and spouse_age is None:
            spouse_age = basis.spouse_age(age, sex)
        return pension_annuity(
            table, age, basis.post_rate,
            increase=basis.pension_increase,
            frequency=scheme.pension_frequency,
            timing=scheme.pension_timing,
            guarantee=guarantee,
            spouse_fraction=fraction,
            spouse_age=spouse_age,
            spouse_table=basis.spouse_table(sex) if fraction > 0 else None,
            proportion_married=proportion_married,
        )

    def retirement_factor(self, age: float, sex: str = "M", ill_health: bool = False) -> float:
        """
        Value at retirement of a pension of 1 a year before commutation,
        including the spouse's pension and guarantee in the rules. Where a
        commutation factor is set, the cash taken is allowed for.
        """
        sex = normalise_sex(sex)
        key = ("retirement", float(age), sex, bool(ill_health))
        if key not in self._factors:
            tables = self.basis.mortality_post
            if ill_health and self.basis.mortality_ill_health is not None:
                tables = self.basis.mortality_ill_health
            annuity = self._annuity(pick_by_sex(tables, sex), age, sex,
                                    guarantee=self.scheme.guarantee_years)
            cash = self.scheme.commutation_fraction
            if cash > 0 and self.scheme.commutation_factor is not None:
                annuity = (1 - cash) * annuity + cash * self.scheme.commutation_factor
            self._factors[key] = annuity
        return self._factors[key]

    def _spouse_factor(self, member_age: float, sex: str) -> float:
        """Value of a pension of 1 a year to the spouse from the member's death in service."""
        key = ("spouse", float(member_age), sex)
        if key not in self._factors:
            basis, scheme = self.basis, self.scheme
            self._factors[key] = pension_annuity(
                basis.spouse_table(sex), basis.spouse_age(member_age, sex), basis.post_rate,
                increase=basis.pension_increase, frequency=scheme.pension_frequency,
                timing=scheme.pension_timing,
            )
        return self._factors[key]

    def _deferment(self, age: float, sex: str) -> float:
        """
        Factor that takes a pension earned at ``age`` to its value at the
        same date as a pension from the normal retirement age: revaluation,
        discounting and survival over the period of deferment.
        """
        key = ("deferment", float(age), sex)
        if key not in self._factors:
            basis = self.basis
            years = self.scheme.normal_retirement_age - age
            factor = ((1 + basis.deferred_revaluation) / (1 + basis.discount_rate)) ** years
            if basis.mortality_deferment is not None:
                table = pick_by_sex(basis.mortality_deferment, sex)
                factor *= float(survival_probabilities(table, age, [years])[0])
            self._factors[key] = factor
        return self._factors[key]

    def _service_table(self, sex: str, age: int) -> ServiceTable:
        retirement_age = self.scheme.normal_retirement_age
        if self.basis.service_table is None:
            if age not in self._no_decrements:
                self._no_decrements[age] = ServiceTable.no_decrements(age, retirement_age)
            return self._no_decrements[age]
        table = pick_by_sex(self.basis.service_table, sex)
        if age < table.min_age:
            raise ValueError(f"the service table starts at age {table.min_age}; "
                             f"a member aged {age} cannot be valued")
        return table

    # ------------------------------------------------------------------
    # One active member
    # ------------------------------------------------------------------
    def _exits(self, age: int, service: float, salary: float, sex: str,
               contributions: float, scale: SalaryScale) -> Dict[str, object]:
        """
        Every way the member can leave: when, with what probability, and the
        value at that date of the benefit then due. Also the value of the
        member's future salaries.
        """
        scheme, basis = self.scheme, self.basis
        retirement_age = scheme.normal_retirement_age
        v = 1.0 / (1.0 + basis.discount_rate)
        averaging = scheme.final_average_years
        years_to_go = retirement_age - age
        times: List[float] = []
        probabilities: List[float] = []
        values: List[float] = []
        causes: List[str] = []
        salaries = 0.0
        first_year_salary = 0.0

        def record(time, probability, value, cause):
            times.append(time)
            probabilities.append(probability)
            values.append(value)
            causes.append(cause)

        if years_to_go <= 0:
            counted = scheme.capped_service(service)
            average = scale.final_average(salary, age, 0.0, averaging)
            value = (scheme.accrual_rate * counted * average * self.retirement_factor(age, sex)
                     + scheme.lump_sum_accrual * counted * average)
            record(0.0, 1.0, value, "retirement")
        else:
            table = self._service_table(sex, age)
            at_retirement = self.retirement_factor(retirement_age, sex)
            full_service = scheme.capped_service(service + years_to_go)
            paid_in = float(contributions)
            for year in range(years_to_go):
                attained = age + year
                staying = table.survival(age, year)
                if staying <= 0:
                    break
                time = year + 0.5
                leaving = table.total_rate(attained)
                rate_of_pay = scale.projected(salary, age, year)
                earned = staying * (1 - 0.5 * leaving) * rate_of_pay * v ** time
                salaries += earned
                if year == 0:
                    first_year_salary = earned
                if leaving > 0:
                    exit_age = attained + 0.5
                    average = scale.final_average(salary, age, time, averaging)
                    counted = scheme.capped_service(service + time)
                    pension = scheme.accrual_rate * counted * average
                    lump_sum = scheme.lump_sum_accrual * counted * average
                    for cause in CAUSES:
                        q = table.rate(attained, cause)
                        if q <= 0:
                            continue
                        if cause == "retirement":
                            reduction = max(0.0, 1 - scheme.early_retirement_reduction
                                            * (retirement_age - exit_age))
                            value = (pension * reduction * self.retirement_factor(exit_age, sex)
                                     + lump_sum)
                        elif cause == "ill_health":
                            credited = (full_service if scheme.ill_health_service == "prospective"
                                        else counted)
                            value = (scheme.accrual_rate * credited * average
                                     * self.retirement_factor(exit_age, sex, ill_health=True)
                                     + lump_sum)
                        elif cause == "death":
                            value = scheme.death_in_service_multiple * rate_of_pay
                            if scheme.death_in_service_spouse_fraction > 0:
                                credited = (full_service
                                            if scheme.death_in_service_service == "prospective"
                                            else counted)
                                value += (scheme.death_in_service_spouse_fraction
                                          * scheme.accrual_rate * credited * average
                                          * basis.proportion_married
                                          * self._spouse_factor(exit_age, sex))
                        elif scheme.withdrawal_benefit == "deferred":
                            value = (self._deferment(exit_age, sex)
                                     * (pension * at_retirement + lump_sum))
                        elif scheme.withdrawal_benefit == "refund":
                            growth = 1 + basis.contribution_interest
                            value = (paid_in * growth ** 0.5
                                     + scheme.member_contribution_rate * rate_of_pay * 0.5
                                     * growth ** 0.25)
                        else:
                            value = 0.0
                        record(time, staying * q, value, cause)
                paid_in = (paid_in * (1 + basis.contribution_interest)
                           + scheme.member_contribution_rate * rate_of_pay
                           * (1 + basis.contribution_interest) ** 0.5)
            staying = table.survival(age, years_to_go)
            if staying > 0:
                average = scale.final_average(salary, age, float(years_to_go), averaging)
                value = (scheme.accrual_rate * full_service * average * at_retirement
                         + scheme.lump_sum_accrual * full_service * average)
                record(float(years_to_go), staying, value, "retirement")

        times_arr = np.array(times)
        present_values = np.array(probabilities) * v ** times_arr * np.array(values)
        return {
            "times": times_arr,
            "present_values": present_values,
            "causes": causes,
            "benefits": float(present_values.sum()),
            "salaries": salaries,
            "first_year_salary": first_year_salary,
        }

    def _measure(self, age: int, service: float, salary: float, sex: str,
                 contributions: float, method: str) -> Dict[str, float]:
        """Liability and normal cost of one active member under a funding method."""
        scheme = self.scheme
        scale = self.basis.salary_scale
        projected = self._exits(age, service, salary, sex, contributions, scale)
        benefits = projected["benefits"]
        salaries = projected["salaries"]
        first_year = projected["first_year_salary"]

        def unit_credit(exits):
            times = exits["times"]
            past = scheme.capped_service(service)
            at_exit = np.array([scheme.capped_service(service + t) for t in times])
            after_a_year = np.array([scheme.capped_service(service + min(1.0, t))
                                     for t in times])
            with np.errstate(divide="ignore", invalid="ignore"):
                earned = np.where(at_exit > 0, past / at_exit, 1.0)
                accruing = np.where(at_exit > 0, (after_a_year - past) / at_exit, 0.0)
            return (float(np.sum(exits["present_values"] * earned)),
                    float(np.sum(exits["present_values"] * accruing)))

        if method == "projected_unit":
            liability, normal_cost = unit_credit(projected)
        elif method == "current_unit":
            level = self._exits(age, service, salary, sex, contributions, SalaryScale(0.0))
            liability, normal_cost = unit_credit(level)
        elif method == "attained_age":
            liability, _ = unit_credit(projected)
            normal_cost = (benefits - liability) / salaries * first_year if salaries > 0 else 0.0
        else:
            entry_age = int(round(age - service))
            if self.basis.service_table is not None:
                entry_age = max(entry_age, self._service_table(sex, age).min_age)
            entry_age = min(entry_age, age)
            if salaries <= 0:
                rate = 0.0
            elif entry_age == age:
                rate = benefits / salaries
            else:
                at_entry = self._exits(entry_age, 0.0,
                                       scale.projected(salary, age, entry_age - age),
                                       sex, 0.0, scale)
                rate = at_entry["benefits"] / at_entry["salaries"]
            liability = benefits - rate * salaries
            normal_cost = rate * first_year
        return {
            "value_of_benefits": benefits,
            "value_of_salaries": salaries,
            "salary_next_year": first_year,
            "accrued_liability": liability,
            "normal_cost": normal_cost,
        }

    # ------------------------------------------------------------------
    # Results by member
    # ------------------------------------------------------------------
    @property
    def active_results(self) -> pd.DataFrame:
        """
        One row per active member: the value of all benefits, the value of
        future salaries, the liability and the normal cost.
        """
        if self._active_results is None:
            rows = []
            for _, member in self._actives.iterrows():
                rows.append(self._measure(
                    int(member["age"]), float(member["service"]), float(member["salary"]),
                    member["sex"], float(_optional(member, "contributions", 0.0)), self.method,
                ))
            columns = ["value_of_benefits", "value_of_salaries", "salary_next_year",
                       "accrued_liability", "normal_cost"]
            results = pd.DataFrame(rows, columns=columns, index=self._actives.index)
            self._active_results = pd.concat([self._actives, results], axis=1)
        return self._active_results

    @property
    def pensioner_results(self) -> pd.DataFrame:
        """One row per pensioner with the annuity factor and the liability."""
        if self._pensioner_results is None:
            basis = self.basis
            factors = []
            for _, member in self._pensioners.iterrows():
                sex = member["sex"]
                tables = basis.mortality_post
                if bool(_optional(member, "ill_health", False)) \
                        and basis.mortality_ill_health is not None:
                    tables = basis.mortality_ill_health
                spouse_age = _optional(member, "spouse_age", None)
                factors.append(self._annuity(
                    pick_by_sex(tables, sex), float(member["age"]), sex,
                    guarantee=float(_optional(member, "guarantee_remaining", 0.0)),
                    reversion=bool(_optional(member, "reversion", True)),
                    spouse_age=None if spouse_age is None else float(spouse_age),
                    proportion_married=None if spouse_age is None else 1.0,
                ))
            results = self._pensioners.copy()
            results["annuity_factor"] = np.array(factors, dtype=float)
            results["liability"] = results["pension"] * results["annuity_factor"]
            self._pensioner_results = results
        return self._pensioner_results

    @property
    def deferred_results(self) -> pd.DataFrame:
        """One row per deferred pensioner with the liability."""
        if self._deferred_results is None:
            retirement_age = self.scheme.normal_retirement_age
            liabilities = []
            for _, member in self._deferreds.iterrows():
                age, sex = float(member["age"]), member["sex"]
                if age >= retirement_age:
                    factor = self.retirement_factor(age, sex)
                else:
                    factor = self._deferment(age, sex) * self.retirement_factor(retirement_age,
                                                                                sex)
                liabilities.append(member["pension"] * factor)
            results = self._deferreds.copy()
            results["liability"] = np.array(liabilities, dtype=float)
            self._deferred_results = results
        return self._deferred_results

    # ------------------------------------------------------------------
    # Totals
    # ------------------------------------------------------------------
    @property
    def active_liability(self) -> float:
        """Liability for active members."""
        return float(self.active_results["accrued_liability"].sum())

    @property
    def pensioner_liability(self) -> float:
        """Liability for pensions in payment."""
        return float(self.pensioner_results["liability"].sum())

    @property
    def deferred_liability(self) -> float:
        """Liability for deferred pensioners."""
        return float(self.deferred_results["liability"].sum())

    @property
    def total_liability(self) -> float:
        """Total actuarial liabilities, including reserves."""
        return (self.active_liability + self.deferred_liability + self.pensioner_liability
                + self.reserves)

    @property
    def surplus(self) -> float:
        """Assets less total liabilities (negative for a deficit)."""
        return self.assets - self.total_liability

    @property
    def funding_level(self) -> float:
        """Assets as a proportion of total liabilities."""
        total = self.total_liability
        return self.assets / total if total > 0 else float("nan")

    @property
    def payroll(self) -> float:
        """Total annual pensionable salaries of active members."""
        return float(self._actives["salary"].sum())

    @property
    def normal_cost(self) -> float:
        """Normal cost of the coming year, valued at the valuation date."""
        return float(self.active_results["normal_cost"].sum())

    @property
    def standard_contribution_rate(self) -> float:
        """
        Normal cost as a proportion of the salaries expected to be paid over
        the coming year (both valued at the valuation date).
        """
        salaries = float(self.active_results["salary_next_year"].sum())
        return self.normal_cost / salaries if salaries > 0 else 0.0

    def summary(self) -> pd.Series:
        """The financial position of the scheme, as set out in a valuation report."""
        return pd.Series({
            "Net assets": self.assets,
            "Active members": self.active_liability,
            "Deferred pensioners": self.deferred_liability,
            "Pensioners": self.pensioner_liability,
            "Reserves": self.reserves,
            "Total actuarial liabilities": self.total_liability,
            "Surplus / (deficit)": self.surplus,
            "Funding level": self.funding_level,
        }, name=self.method)

    # ------------------------------------------------------------------
    # Contributions
    # ------------------------------------------------------------------
    def contribution_rate(self,
                          recovery_period: Optional[int] = None,
                          expenses: float = 0.0,
                          spread_surplus: bool = False) -> pd.Series:
        """
        Recommended contribution rate as a proportion of pensionable salaries.

        Args:
            recovery_period: Years over which a deficit is paid off. If
                omitted, nothing is added for the deficit.
            expenses: Allowance for expenses, as a proportion of salaries
            spread_surplus: If True, a surplus reduces the rate over the
                same period

        Returns:
            The standard rate, the adjustment for the deficit or surplus,
            expenses, the total, and its split between members and employer.
            The adjustment assumes that total salaries grow at the general
            rate of salary increase and are paid evenly over each year.
        """
        adjustment = 0.0
        shortfall = -self.surplus
        if recovery_period is not None and (shortfall > 0 or spread_surplus):
            if recovery_period <= 0 or recovery_period != int(recovery_period):
                raise ValueError("recovery_period must be a positive whole number of years")
            if self.payroll <= 0:
                raise ValueError("there are no salaries from which to recover a deficit")
            ratio = ((1 + self.basis.salary_scale.general_increase)
                     / (1 + self.basis.discount_rate))
            annuity = float(np.sum(ratio ** (np.arange(int(recovery_period)) + 0.5)))
            adjustment = shortfall / (self.payroll * annuity)
        standard = self.standard_contribution_rate
        total = standard + adjustment + expenses
        members = self.scheme.member_contribution_rate
        return pd.Series({
            "Standard contribution rate": standard,
            "Deficit / (surplus) adjustment": adjustment,
            "Expenses": expenses,
            "Total contribution rate": total,
            "Members": members,
            "Employer": total - members,
        })

    def aggregate_contribution_rate(self) -> float:
        """
        Contribution rate under the aggregate method: the value of all
        benefits and reserves, less the assets, spread over the value of the
        future salaries of the present active members.
        """
        results = self.active_results
        salaries = float(results["value_of_salaries"].sum())
        if salaries <= 0:
            raise ValueError("there are no future salaries to spread the cost over")
        benefits = (float(results["value_of_benefits"].sum()) + self.deferred_liability
                    + self.pensioner_liability + self.reserves)
        return (benefits - self.assets) / salaries

    # ------------------------------------------------------------------
    # Other bases and methods
    # ------------------------------------------------------------------
    def revalue(self, basis: Optional[ValuationBasis] = None, method: Optional[str] = None,
                assets: Optional[float] = None) -> 'DBValuation':
        """The same members valued on another basis, method or asset value."""
        return DBValuation(
            self.scheme, basis or self.basis, self._actives, self._pensioners, self._deferreds,
            assets=self.assets if assets is None else assets,
            method=method or self.method, reserves=self.reserves,
        )

    def method_comparison(self) -> pd.DataFrame:
        """Liability, normal cost and funding level under each funding method."""
        rows = {}
        for method in METHODS:
            valuation = self if method == self.method else self.revalue(method=method)
            rows[method] = {
                "Active liability": valuation.active_liability,
                "Total liabilities": valuation.total_liability,
                "Funding level": valuation.funding_level,
                "Normal cost": valuation.normal_cost,
                "Standard contribution rate": valuation.standard_contribution_rate,
            }
        return pd.DataFrame(rows).T

    def sensitivities(self, shifts: Optional[Dict[str, Dict[str, float]]] = None) -> pd.DataFrame:
        """
        Effect on the liabilities of changing one assumption at a time.

        Args:
            shifts: Mapping from a description to the keyword arguments of
                :meth:`ValuationBasis.shifted`. By default the discount
                rate, salary increases and pension increases are moved up
                and down by 1% and pensioner mortality is reduced by 10%.
        """
        if shifts is None:
            shifts = {
                "Discount rate +1%": {"discount_rate": 0.01},
                "Discount rate -1%": {"discount_rate": -0.01},
                "Salary increases +1%": {"salary_increase": 0.01},
                "Salary increases -1%": {"salary_increase": -0.01},
                "Pension increases +1%": {"pension_increase": 0.01},
                "Pension increases -1%": {"pension_increase": -0.01},
                "Pensioner mortality -10%": {"mortality_factor": 0.9},
            }
        base = self.total_liability
        rows = {"Valuation basis": {"Total liabilities": base, "Change": 0.0,
                                    "Change %": 0.0, "Funding level": self.funding_level}}
        for label, change in shifts.items():
            valuation = self.revalue(basis=self.basis.shifted(**change))
            total = valuation.total_liability
            rows[label] = {
                "Total liabilities": total,
                "Change": total - base,
                "Change %": (total / base - 1) if base else float("nan"),
                "Funding level": valuation.funding_level,
            }
        return pd.DataFrame(rows).T

    def duration(self, shift: float = 0.001) -> float:
        """
        Modified duration of the liabilities: the proportionate fall in the
        liabilities for a rise of 1 (that is, 100%) in the discount rate,
        measured over a small change either way.
        """
        up = self.revalue(basis=self.basis.shifted(discount_rate=shift)).total_liability
        down = self.revalue(basis=self.basis.shifted(discount_rate=-shift)).total_liability
        return (down - up) / (2 * shift * self.total_liability)

    def __repr__(self) -> str:
        return (f"DBValuation(method='{self.method}', actives={len(self._actives)}, "
                f"deferreds={len(self._deferreds)}, pensioners={len(self._pensioners)}, "
                f"liabilities={self.total_liability:,.0f}, "
                f"funding_level={self.funding_level:.1%})")
