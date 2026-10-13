"""
Scheme rules and valuation basis

:class:`DefinedBenefitScheme` describes what the rules of a scheme promise.
:class:`ValuationBasis` holds the assumptions used to put a value on it.
"""

from dataclasses import dataclass, field, replace
from typing import Mapping, Optional, Union

import numpy as np

from ..mortality import MortalityTable
from .service_table import SalaryScale, ServiceTable

TableBySex = Union[MortalityTable, Mapping[str, MortalityTable]]
ServiceTableBySex = Union[ServiceTable, Mapping[str, ServiceTable]]


def normalise_sex(sex) -> str:
    """'M' or 'F' from the usual ways of writing it."""
    text = str(sex).strip().upper()
    if text in ("M", "MALE"):
        return "M"
    if text in ("F", "FEMALE"):
        return "F"
    raise ValueError(f"sex must be 'M' or 'F', not '{sex}'")


def pick_by_sex(item, sex: str):
    """The entry for a sex from a mapping keyed 'M' and 'F', or the item itself."""
    if isinstance(item, Mapping):
        if sex not in item:
            raise ValueError(f"no table was supplied for sex '{sex}'")
        return item[sex]
    return item


@dataclass(frozen=True)
class DefinedBenefitScheme:
    """
    The benefits promised by a defined benefit scheme.

    Attributes:
        accrual_rate: Pension earned for each year of service, as a fraction
            of final average salary (for example 1/60)
        normal_retirement_age: Age at which members retire
        final_average_years: Years over which salary is averaged before
            leaving. Zero gives the salary rate at the date of leaving.
        lump_sum_accrual: Lump sum paid on retirement for each year of
            service, as a fraction of final average salary
        max_service: Most years of service that count for benefits
        commutation_fraction: Fraction of the pension given up for cash at
            retirement
        commutation_factor: Cash paid for each 1 a year of pension given up.
            If omitted, commutation is taken to be on terms that do not
            change the value of the benefit.
        spouse_fraction: Pension paid to the spouse after a pensioner dies,
            as a fraction of the pensioner's pension
        guarantee_years: Years for which a pension is paid in any event
        pension_frequency: Number of pension payments a year
        pension_timing: "advance" or "arrears"
        early_retirement_reduction: Reduction in pension for each year that
            retirement is before the normal retirement age
        ill_health_service: Service that counts on ill-health retirement:
            "accrued" (to the date of retirement) or "prospective" (as if
            the member had stayed to the normal retirement age)
        death_in_service_multiple: Lump sum on death in service as a
            multiple of salary
        death_in_service_spouse_fraction: Spouse's pension on death in
            service as a fraction of the member's pension
        death_in_service_service: Service that counts for that spouse's
            pension: "accrued" or "prospective"
        withdrawal_benefit: Benefit on leaving the employer: "deferred" (a
            pension from the normal retirement age based on service and
            salary at leaving), "refund" (the member's contributions with
            interest) or "none"
        member_contribution_rate: Members' contributions as a fraction of
            salary
    """
    accrual_rate: float = 1 / 60
    normal_retirement_age: int = 60
    final_average_years: float = 1.0
    lump_sum_accrual: float = 0.0
    max_service: Optional[float] = None
    commutation_fraction: float = 0.0
    commutation_factor: Optional[float] = None
    spouse_fraction: float = 0.0
    guarantee_years: float = 0.0
    pension_frequency: int = 12
    pension_timing: str = "advance"
    early_retirement_reduction: float = 0.0
    ill_health_service: str = "accrued"
    death_in_service_multiple: float = 0.0
    death_in_service_spouse_fraction: float = 0.0
    death_in_service_service: str = "accrued"
    withdrawal_benefit: str = "deferred"
    member_contribution_rate: float = 0.0

    def __post_init__(self):
        if self.accrual_rate < 0 or self.lump_sum_accrual < 0:
            raise ValueError("accrual rates must be non-negative")
        if self.final_average_years < 0:
            raise ValueError("final_average_years must be non-negative")
        if self.max_service is not None and self.max_service <= 0:
            raise ValueError("max_service must be positive")
        if not 0 <= self.commutation_fraction <= 1:
            raise ValueError("commutation_fraction must be between 0 and 1")
        if self.spouse_fraction < 0 or self.death_in_service_spouse_fraction < 0:
            raise ValueError("spouse fractions must be non-negative")
        if self.guarantee_years < 0 or self.early_retirement_reduction < 0:
            raise ValueError("guarantee_years and early_retirement_reduction must be "
                             "non-negative")
        if self.pension_timing not in ("advance", "arrears"):
            raise ValueError("pension_timing must be 'advance' or 'arrears'")
        for name in ("ill_health_service", "death_in_service_service"):
            if getattr(self, name) not in ("accrued", "prospective"):
                raise ValueError(f"{name} must be 'accrued' or 'prospective'")
        if self.withdrawal_benefit not in ("deferred", "refund", "none"):
            raise ValueError("withdrawal_benefit must be 'deferred', 'refund' or 'none'")
        if not 0 <= self.member_contribution_rate < 1:
            raise ValueError("member_contribution_rate must be between 0 and 1")

    def capped_service(self, service: float) -> float:
        """Service that counts for benefits."""
        return service if self.max_service is None else min(service, self.max_service)


@dataclass(frozen=True)
class ValuationBasis:
    """
    The assumptions of an actuarial valuation.

    Attributes:
        discount_rate: Rate of interest used to discount benefits, before
            retirement and (unless ``post_retirement_discount_rate`` is
            given) after it
        mortality_post: Mortality of pensioners: one table, or a mapping
            with a table for each sex under "M" and "F"
        service_table: Decrements of active members: one table or a mapping
            by sex. If omitted, members are assumed to stay to the normal
            retirement age.
        salary_scale: Future salary increases: a :class:`SalaryScale` or a
            single annual rate
        pension_increase: Annual rate of increase of pensions in payment
        deferred_revaluation: Annual rate at which a deferred pension is
            increased between leaving and retirement
        post_retirement_discount_rate: Rate of interest after retirement
        mortality_ill_health: Mortality of ill-health pensioners (that of
            other pensioners if omitted)
        mortality_deferment: Mortality of deferred pensioners before
            retirement. If omitted, no deaths are assumed in deferment.
        mortality_spouse: Mortality of spouses. If omitted, the table of the
            other sex is used when tables are given by sex, otherwise the
            pensioner's table.
        proportion_married: Probability that a member leaves a spouse
        spouse_age_difference: Years by which husbands are older than wives
        contribution_interest: Interest credited to members' contributions
            when they are refunded
    """
    discount_rate: float
    mortality_post: TableBySex
    service_table: Optional[ServiceTableBySex] = None
    salary_scale: Union[SalaryScale, float] = field(default_factory=SalaryScale)
    pension_increase: float = 0.0
    deferred_revaluation: float = 0.0
    post_retirement_discount_rate: Optional[float] = None
    mortality_ill_health: Optional[TableBySex] = None
    mortality_deferment: Optional[TableBySex] = None
    mortality_spouse: Optional[TableBySex] = None
    proportion_married: float = 1.0
    spouse_age_difference: float = 0.0
    contribution_interest: float = 0.0

    def __post_init__(self):
        if not isinstance(self.salary_scale, SalaryScale):
            object.__setattr__(self, "salary_scale", SalaryScale(float(self.salary_scale)))
        for name in ("discount_rate", "pension_increase", "deferred_revaluation",
                     "contribution_interest"):
            if getattr(self, name) <= -1:
                raise ValueError(f"{name} must be greater than -1")
        if self.post_retirement_discount_rate is not None \
                and self.post_retirement_discount_rate <= -1:
            raise ValueError("post_retirement_discount_rate must be greater than -1")
        if not 0 <= self.proportion_married <= 1:
            raise ValueError("proportion_married must be between 0 and 1")

    @property
    def post_rate(self) -> float:
        """Rate of interest after retirement."""
        if self.post_retirement_discount_rate is None:
            return self.discount_rate
        return self.post_retirement_discount_rate

    def spouse_table(self, member_sex: str) -> MortalityTable:
        """Mortality table of the spouse of a member of the given sex."""
        other = "F" if member_sex == "M" else "M"
        if self.mortality_spouse is not None:
            return pick_by_sex(self.mortality_spouse, other)
        if isinstance(self.mortality_post, Mapping):
            return pick_by_sex(self.mortality_post, other)
        return self.mortality_post

    def spouse_age(self, member_age: float, member_sex: str) -> float:
        """Assumed age of the spouse of a member."""
        if member_sex == "M":
            return member_age - self.spouse_age_difference
        return member_age + self.spouse_age_difference

    def shifted(self,
                discount_rate: float = 0.0,
                salary_increase: float = 0.0,
                pension_increase: float = 0.0,
                deferred_revaluation: float = 0.0,
                mortality_factor: float = 1.0) -> 'ValuationBasis':
        """
        The basis with some assumptions changed, for sensitivity tests.

        Args:
            discount_rate: Amount added to the discount rate (before and
                after retirement)
            salary_increase: Amount added to the general salary increase
            pension_increase: Amount added to the pension increase
            deferred_revaluation: Amount added to the revaluation rate
            mortality_factor: Factor applied to the mortality rates of
                pensioners, ill-health pensioners and spouses
        """
        changes = {
            "discount_rate": self.discount_rate + discount_rate,
            "salary_scale": self.salary_scale.shifted(salary_increase),
            "pension_increase": self.pension_increase + pension_increase,
            "deferred_revaluation": self.deferred_revaluation + deferred_revaluation,
        }
        if self.post_retirement_discount_rate is not None:
            changes["post_retirement_discount_rate"] = (
                self.post_retirement_discount_rate + discount_rate
            )
        if mortality_factor != 1.0:
            for name in ("mortality_post", "mortality_ill_health", "mortality_spouse"):
                changes[name] = _scale_tables(getattr(self, name), mortality_factor)
        return replace(self, **changes)


def _scale_table(table: MortalityTable, factor: float) -> MortalityTable:
    rates = np.minimum(table.qx_values * factor, 1.0)
    if table.is_closed:
        rates[-1] = 1.0
    return MortalityTable(table.ages, rates, name=f"{table.name} x {factor:g}",
                          metadata=table.metadata)


def _scale_tables(tables, factor: float):
    if tables is None:
        return None
    if isinstance(tables, Mapping):
        return {sex: _scale_table(table, factor) for sex, table in tables.items()}
    return _scale_table(tables, factor)
