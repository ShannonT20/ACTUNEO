"""
Solvers

Each solver is one calculation that can be asked for in plain words. It says
what it needs (its parameters, and the wording that identifies each one in a
question), which phrases point to it, and how to work the answer out step by
step with ACTUNEO.

Add a solver by writing a function that returns a
:class:`~actuneo.ai.solutions.Solution` and registering it with
:func:`register`.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .solutions import Solution, Step, money, number, percent

NUM = r"(\d+(?:\.\d+)?)"
RADIX = 100_000         # lives shown at the first age of a table


@dataclass(frozen=True)
class Param:
    """
    One quantity a solver needs.

    Attributes:
        name: Argument name
        kind: "age", "years", "rate", "amount" or "choice"
        label: How to name it when asking the user for it
        required: Whether the calculation can go ahead without it
        default: Value used when it is not given
        cues: Regular expressions, each with one group, that pick this
            quantity out of a question when several numbers are present
        choices: For a "choice": each value with the words that mean it
        monthly_to_annual: An amount stated per month is multiplied by 12
        cue_only: Read this quantity only where a cue matches, never by
            taking the next unused number in the question
        low, high: Range outside which a value is refused
    """
    name: str
    kind: str
    label: str
    required: bool = True
    default: Any = None
    cues: Tuple[str, ...] = ()
    choices: Optional[Dict[Any, Tuple[str, ...]]] = None
    monthly_to_annual: bool = False
    cue_only: bool = False
    low: Optional[float] = None
    high: Optional[float] = None


@dataclass(frozen=True)
class Solver:
    """A calculation that can be asked for in plain words."""
    name: str
    title: str
    description: str
    keywords: Tuple[Tuple[str, int], ...]
    params: Tuple[Param, ...]
    solve: Callable[..., Solution] = field(compare=False)
    example: str = ""
    uses: Tuple[str, ...] = ()      # kinds of the user's own data it can take

    def tool(self) -> Dict[str, Any]:
        """The solver described as a tool for a language model."""
        properties = {}
        for param in self.params:
            if param.kind == "choice":
                values = list(param.choices)
                kind = "integer" if all(isinstance(v, int) for v in values) else "string"
                properties[param.name] = {"type": kind, "enum": values,
                                          "description": param.label}
            elif param.kind in ("age", "years"):
                properties[param.name] = {"type": "integer", "description": param.label}
            elif param.kind == "rate":
                properties[param.name] = {
                    "type": "number",
                    "description": f"{param.label}, as a decimal, for example 0.05 for 5%"}
            else:
                properties[param.name] = {"type": "number", "description": param.label}
        return {"name": self.name, "description": self.description,
                "parameters": {"type": "object", "properties": properties,
                               "required": [p.name for p in self.params if p.required]}}


SOLVERS: Dict[str, Solver] = {}


def register(solver: Solver) -> Solver:
    """Add a solver to the registry (replacing one of the same name)."""
    SOLVERS[solver.name] = solver
    return solver


# ----------------------------------------------------------------------
# Parameters used by several solvers
# ----------------------------------------------------------------------
SEX_WORDS = {"F": ("woman", "women", "female", "lady", "she", "her", "widow", "wife", "girl",
                   "mother"),
             "M": ("man", "men", "male", "he", "his", "him", "widower", "husband", "boy",
                   "father", "gentleman")}

AGE = Param("age", "age", "the age", low=0, high=120)
SEX = Param("sex", "choice", "the sex (M or F)", required=False, default="M", choices=SEX_WORDS)
RATE = Param("interest_rate", "rate", "the interest rate", low=-0.5, high=1.0)
BASIS = Param("basis", "choice", "the mortality basis (zimbabwe, south_africa or am92)",
              required=False, default="zimbabwe",
              choices={"zimbabwe": ("zimbabwe", "zimbabwean", "zim", "ipec"),
                       "south_africa": ("south africa", "south african", "sa85", "sa 85",
                                        "saiml98", "saifl98", "assa"),
                       "am92": ("am92", "am 92", "uk table", "uk tables", "institute table"),
                       "own": ("my table", "my own table", "own table", "this table",
                               "the table supplied", "my mortality")})
TRIANGLE = Param("triangle", "choice", "the triangle (RAA, GenIns, or your own data)",
                 choices={"RAA": ("raa",), "GenIns": ("genins", "gen ins", "taylor", "ashe"),
                          "own": ("my triangle", "my own triangle", "my data", "this triangle",
                                  "the triangle supplied", "my claims", "own triangle",
                                  "this data", "the data")})


def _years(label: str = "the number of years", cues: Tuple[str, ...] = ()) -> Param:
    return Param("years", "years", label, cues=cues, low=1, high=100)


# ----------------------------------------------------------------------
# Compound interest
# ----------------------------------------------------------------------
def _present_value(amount: float, years: int, interest_rate: float) -> Solution:
    v = 1 / (1 + interest_rate)
    factor = v ** years
    value = amount * factor
    return Solution(
        "Present value of a single payment",
        {"amount S": money(amount), "time n": f"{years} years", "interest i": percent(interest_rate)},
        [Step("Discount factor for one year",
              ["v = 1 / (1 + i)", f"  = 1 / {1 + interest_rate:.6g}", f"  = {number(v)}"]),
         Step(f"Discount factor for {years} years",
              ["v^n", f"  = {number(v)}^{years}", f"  = {number(factor)}"]),
         Step("Present value",
              ["PV = S x v^n", f"   = {money(amount)} x {number(factor)}", f"   = {money(value)}"])],
        f"The present value is {money(value)}.", value,
        ["Interest is compounded once a year at an effective rate."],
        f"Accumulating {money(value)} for {years} years at {percent(interest_rate)} gives "
        f"{money(value * (1 + interest_rate) ** years)}, the amount due.")


register(Solver(
    "present_value", "Present value of a single payment",
    "Present value: what a single amount due or payable in a number of years is worth today",
    ((r"present value", 3), (r"worth today|worth now|value today|\bpv\b", 3),
     (r"\bdiscount", 2), (r"\b(due|payable|receivable) in\b", 1)),
    (Param("amount", "amount", "the amount due", low=0), _years("the number of years until it is due"),
     RATE),
    _present_value, "What is the present value of 1000 due in 5 years at 10% interest?"))


def _future_value(amount: float, years: int, interest_rate: float) -> Solution:
    factor = (1 + interest_rate) ** years
    value = amount * factor
    return Solution(
        "Accumulated value of a single payment",
        {"amount P": money(amount), "time n": f"{years} years", "interest i": percent(interest_rate)},
        [Step(f"Accumulation factor for {years} years",
              ["(1 + i)^n", f"  = {1 + interest_rate:.6g}^{years}", f"  = {number(factor)}"]),
         Step("Accumulated value",
              ["A = P x (1 + i)^n", f"  = {money(amount)} x {number(factor)}", f"  = {money(value)}"])],
        f"The amount grows to {money(value)}.", value,
        ["Interest is compounded once a year at an effective rate."],
        f"Interest earned is {money(value - amount)}, which is "
        f"{value / amount - 1:.1%} of the amount invested." if amount else None)


register(Solver(
    "future_value", "Accumulated value of a single payment",
    "Future or accumulated value: what an amount invested today grows to after a number of years",
    ((r"future value|accumulated value|accumulat", 3), (r"grows? to|grow into|amount to", 3),
     (r"\binvest", 2), (r"will .* be worth", 2)),
    (Param("amount", "amount", "the amount invested", low=0), _years(), RATE),
    _future_value, "If I invest 5000 today at 9%, what will it grow to in 12 years?"))


def _annuity_certain(payment: float, years: int, interest_rate: float,
                     timing: str = "arrears") -> Solution:
    i = interest_rate
    v = 1 / (1 + i)
    vn = v ** years
    if abs(i) < 1e-12:
        factor = float(years)
        lines = ["With no interest the factor is the number of payments", f"  = {years}"]
        symbol = "a_n"
    elif timing == "advance":
        d = i / (1 + i)
        factor = (1 - vn) / d
        symbol = "a-due_n"
        lines = ["a-due_n = (1 - v^n) / d,  where d = i / (1 + i)",
                 f"  d = {i:.6g} / {1 + i:.6g} = {number(d)}",
                 f"  a-due_n = (1 - {number(vn)}) / {number(d)}", f"          = {number(factor)}"]
    else:
        factor = (1 - vn) / i
        symbol = "a_n"
        lines = ["a_n = (1 - v^n) / i", f"    = (1 - {number(vn)}) / {i:.6g}",
                 f"    = {number(factor)}"]
    value = payment * factor
    when = "start" if timing == "advance" else "end"
    return Solution(
        "Present value of an annuity-certain",
        {"payment P": f"{money(payment)} a year, at the {when} of each year",
         "term n": f"{years} years", "interest i": percent(i)},
        [Step("Discount factors", ["v = 1 / (1 + i)", f"  = {number(v)}",
                                   f"v^n = {number(v)}^{years} = {number(vn)}"]),
         Step("Annuity factor", lines),
         Step("Present value", [f"PV = P x {symbol}", f"   = {money(payment)} x {number(factor)}",
                                f"   = {money(value)}"])],
        f"The present value is {money(value)}.", value,
        [f"Payments are certain (not dependent on survival) and made at the {when} of each year.",
         "Interest is an effective annual rate."],
        f"The {years} payments total {money(payment * years)} before discounting.")


register(Solver(
    "annuity_certain", "Present value of an annuity-certain",
    "Present value of level payments made every year for a fixed number of years (annuity-certain)",
    ((r"annuity[- ]certain", 5), (r"level payments?|series of payments|payments? of", 3),
     (r"(each|every|per) year|a year for \d+ years", 2)),
    (Param("payment", "amount", "the yearly payment", low=0),
     _years("the number of years of payments"), RATE,
     Param("timing", "choice", "whether payments are in advance or in arrears", required=False,
           default="arrears",
           choices={"advance": ("in advance", "start of each", "beginning of each", "annuity-due",
                                "annuity due"),
                    "arrears": ("in arrears", "end of each", "end of every")})),
    _annuity_certain, "Present value of payments of 1000 at the end of each year for 10 years at 6%"))


def _loan_repayment(amount: float, interest_rate: float, years: int) -> Solution:
    from ..finance import Loan
    loan = Loan(amount, interest_rate, years, payments_per_year=12)
    j = (1 + interest_rate) ** (1 / 12) - 1
    n = 12 * years
    factor = (1 - (1 + j) ** -n) / j if j else float(n)
    payment = loan.payment
    interest_first = amount * j
    nominal_j = interest_rate / 12
    nominal_payment = amount * nominal_j / (1 - (1 + nominal_j) ** -n) if nominal_j else amount / n
    return Solution(
        "Level monthly repayment on a loan",
        {"loan L": money(amount), "interest i": f"{percent(interest_rate)} a year",
         "term": f"{years} years, so n = {n} monthly payments"},
        [Step("Monthly rate of interest equivalent to the annual rate",
              ["j = (1 + i)^(1/12) - 1", f"  = {1 + interest_rate:.6g}^(1/12) - 1", f"  = {number(j, 8)}"]),
         Step(f"Annuity factor for {n} monthly payments",
              ["a_n at rate j = (1 - (1 + j)^-n) / j",
               f"  = (1 - {number((1 + j) ** -n)}) / {number(j, 8)}", f"  = {number(factor, 4)}"]),
         Step("Repayment, from: loan = repayment x annuity factor",
              ["X = L / a_n", f"  = {money(amount)} / {number(factor, 4)}", f"  = {money(payment)}"]),
         Step("First repayment split into interest and capital",
              [f"interest = L x j = {money(amount)} x {number(j, 8)} = {money(interest_first)}",
               f"capital  = X - interest = {money(payment)} - {money(interest_first)} "
               f"= {money(payment - interest_first)}"])],
        f"The repayment is {money(payment)} a month.", payment,
        [f"{percent(interest_rate)} is taken as an effective annual rate, repayments are monthly "
         "in arrears, and the loan is repaid in full by the end.",
         f"If {percent(interest_rate)} is instead a nominal rate convertible monthly, the monthly "
         f"rate is i/12 = {number(nominal_j, 8)} and the repayment is {money(nominal_payment)}."],
        f"Total repaid is {n} x {money(payment)} = {money(n * payment)}, of which "
        f"{money(n * payment - amount)} is interest.")


register(Solver(
    "loan_repayment", "Level monthly repayment on a loan",
    "Monthly repayment or instalment on a loan or mortgage of an amount borrowed over a number of years",
    ((r"\bloan\b|mortgage", 3), (r"borrow|\blent\b|\blend", 3),
     (r"repayment|instal+ment|pay each month|monthly payment", 2)),
    (Param("amount", "amount", "the amount borrowed", low=0), RATE,
     _years("the term of the loan in years")),
    _loan_repayment, "What is the monthly repayment on a loan of 50000 over 10 years at 8%?"))


def _bond_price(coupon_rate: float, years: int, yield_rate: float) -> Solution:
    from ..finance import Bond
    price = Bond(coupon_rate, years, frequency=2).price(yield_rate)
    coupon = 100 * coupon_rate / 2
    j = (1 + yield_rate) ** 0.5 - 1
    n = 2 * years
    factor = (1 - (1 + j) ** -n) / j if j else float(n)
    redemption = 100 * (1 + yield_rate) ** -years
    nominal_j = yield_rate / 2
    nominal = (coupon * (1 - (1 + nominal_j) ** -n) / nominal_j if nominal_j else coupon * n) \
        + 100 * (1 + nominal_j) ** -n
    return Solution(
        "Price of a fixed-interest bond",
        {"coupon": f"{percent(coupon_rate)} a year, paid half-yearly", "term n": f"{years} years",
         "yield i": f"{percent(yield_rate)} a year", "redemption": "100 per 100 nominal (at par)"},
        [Step("Each half-yearly coupon per 100 nominal",
              ["coupon = 100 x coupon rate / 2", f"       = 100 x {coupon_rate:.6g} / 2",
               f"       = {number(coupon, 4)}"]),
         Step("Half-yearly rate equivalent to the yield",
              ["j = (1 + i)^(1/2) - 1", f"  = {1 + yield_rate:.6g}^0.5 - 1", f"  = {number(j, 8)}"]),
         Step(f"Present value of the {n} coupons",
              ["a_2n at rate j = (1 - (1 + j)^-2n) / j", f"  = {number(factor, 4)}",
               f"coupons = {number(coupon, 4)} x {number(factor, 4)} = {number(coupon * factor, 4)}"]),
         Step("Present value of the redemption payment",
              ["100 x v^n = 100 x (1 + i)^-n", f"  = 100 x {number((1 + yield_rate) ** -years)}",
               f"  = {number(redemption, 4)}"]),
         Step("Price", ["P = coupons + redemption",
                        f"  = {number(coupon * factor, 4)} + {number(redemption, 4)}",
                        f"  = {number(price, 4)}"])],
        f"The price is {price:.2f} per 100 nominal.", price,
        ["Coupons are paid half-yearly in arrears, the next one in six months, and the bond is "
         "redeemed at par.", f"{percent(yield_rate)} is an effective annual yield. No tax.",
         f"If the yield is instead a nominal rate convertible half-yearly, the half-yearly rate is "
         f"i/2 = {number(nominal_j, 6)} and the price is {nominal:.2f}."],
        "The price is " + ("below" if price < 100 else "above" if price > 100 else "equal to")
        + " par because the coupon rate is " + ("below" if price < 100 else "above"
                                                if price > 100 else "equal to") + " the yield.")


register(Solver(
    "bond_price", "Price of a fixed-interest bond",
    "Price of a bond from its coupon rate, its term in years and the yield",
    ((r"\bbonds?\b", 3), (r"\bcoupon", 3), (r"\byields?\b|redemption", 1), (r"\bprice\b|trade at", 1)),
    (Param("coupon_rate", "rate", "the coupon rate", low=0, high=1,
           cues=(NUM + r"\s?%\s*(?:annual\s+)?coupon", r"coupon(?:\s+rate)?(?:\s+of)?\s+" + NUM + r"\s?%",
                 r"pays?\s+(?:a\s+)?" + NUM + r"\s?%")),
     _years("the term of the bond in years"),
     Param("yield_rate", "rate", "the yield", low=-0.5, high=1,
           cues=(r"yields?(?:\s+\w+){0,3}?\s+" + NUM + r"\s?%", NUM + r"\s?%\s*yield",
                 r"to yield\s+" + NUM + r"\s?%"))),
    _bond_price, "Price a 10-year bond with a 6% coupon at a yield of 7%."))


def _effective_rate(nominal_rate: float, periods: int = 12) -> Solution:
    per_period = nominal_rate / periods
    effective = (1 + per_period) ** periods - 1
    names = {1: "yearly", 2: "half-yearly", 4: "quarterly", 12: "monthly", 365: "daily"}
    return Solution(
        "Effective annual rate from a nominal rate",
        {"nominal rate i(p)": f"{percent(nominal_rate)} a year",
         "convertible": f"{names.get(periods, str(periods) + ' times a year')} (p = {periods})"},
        [Step("Rate for each period", ["i(p) / p", f"  = {nominal_rate:.6g} / {periods}",
                                       f"  = {number(per_period, 8)}"]),
         Step("Compound over a year",
              ["i = (1 + i(p)/p)^p - 1", f"  = {number(1 + per_period, 8)}^{periods} - 1",
               f"  = {number(effective, 8)}"])],
        f"The effective annual rate is {effective:.4%}.", effective,
        ["Interest is added at the end of each period and earns interest thereafter."],
        f"The effective rate exceeds the nominal rate by {effective - nominal_rate:.4%} "
        "because interest is earned on interest within the year.")


register(Solver(
    "effective_rate", "Effective annual rate from a nominal rate",
    "Effective annual interest rate equivalent to a nominal rate convertible or compounded several times a year",
    ((r"effective", 3), (r"nominal", 3), (r"convertible|compounded|compounding", 2)),
    (Param("nominal_rate", "rate", "the nominal rate", low=-0.5, high=5),
     Param("periods", "choice", "how often interest is compounded each year", required=False,
           default=12,
           choices={12: ("monthly", "every month", "12 times"), 4: ("quarterly", "every quarter"),
                    2: ("half-yearly", "half yearly", "semi-annual", "semiannual", "twice a year"),
                    365: ("daily", "every day"), 1: ("annually", "yearly")})),
    _effective_rate, "What is the effective annual rate of 12% convertible monthly?"))


# ----------------------------------------------------------------------
# Mortality and life contingencies
# ----------------------------------------------------------------------
def _check_age(table, label: str, age: float) -> None:
    if not table.min_age <= age <= table.max_age:
        raise ValueError(f"{label} covers ages {table.min_age} to {table.max_age}, so age "
                         f"{age:g} cannot be valued on it. Name another basis in the question "
                         "(Zimbabwe, South African or AM92) or pass your own table.")


def _table(basis: str, sex: str, age: float, annuitant: bool, data: Optional[dict] = None):
    """
    The mortality table for a basis, with a line describing it.

    The default is Zimbabwe 2023. South African tables and AM92 are used only
    when the question names them, and the user's own table when one is passed.
    """
    from ..mortality import MortalityTable, load_am92
    who = "males" if sex == "M" else "females"
    if basis == "own":
        if not data or data.get("table") is None:
            raise ValueError("You asked for your own table but did not pass one. Pass it as "
                             "data, for example ask(question, data=my_table).")
        table = data["table"]
        label = f"your own table ({table.name})"
    elif basis == "am92":
        table = load_am92().ultimate
        label = ("AM92 ultimate (UK assured lives, males, 1991-94), the usual table of UK "
                 "actuarial examinations. It is not a basis for African lives")
    elif basis == "south_africa":
        if annuitant or age >= 40:
            key = "saiml98" if sex == "M" else "saifl98"
            table = MortalityTable.from_south_africa(key)
            label = f"South African annuitants 1996-2000 ({key.upper()}, {who})"
        else:
            table = MortalityTable.from_south_africa("sa8590")
            label = "South African assured lives 1985-90 (SA85-90, not split by sex)"
    elif annuitant:
        pensioners = MortalityTable.from_zimbabwe_2023(
            f"post_retirement_pensions_{'male' if sex == 'M' else 'female'}")
        if age >= pensioners.min_age:
            table = pensioners
            label = f"Zimbabwe 2023 post-retirement pensions, {who}"
        else:
            assured = MortalityTable.from_zimbabwe_2023("male_assured_lives")
            keep = assured.ages < pensioners.min_age
            table = MortalityTable(
                list(assured.ages[keep]) + list(pensioners.ages),
                list(assured.qx_values[keep]) + list(pensioners.qx_values),
                name="Zimbabwe 2023 assured lives to 75, pensioners from 76")
            label = (f"Zimbabwe 2023. The published pensioner tables start at age "
                     f"{pensioners.min_age}, so the male assured lives rates are used up to age "
                     f"{pensioners.min_age - 1} and the post-retirement pensioner rates ({who}) "
                     "after that. This join is ACTUNEO's, not a published table")
    else:
        table = MortalityTable.from_zimbabwe_2023("male_assured_lives")
        label = "Zimbabwe 2023 male assured lives"
        if sex == "F":
            label += " (no female assured lives table is published, so the male table is used)"
    _check_age(table, label, age)
    return table, label


def _assured_table(basis: str, age: float, data: Optional[dict] = None):
    return _table(basis, "M", age, annuitant=False, data=data) if basis != "south_africa" \
        else _south_african_assured(age)


def _south_african_assured(age: float):
    from ..mortality import MortalityTable
    table = MortalityTable.from_south_africa("sa8590")
    label = "South African assured lives 1985-90 (SA85-90)"
    _check_age(table, label, age)
    return table, label


def _life_expectancy(age: int, sex: str = "M", basis: str = "zimbabwe",
                     data: Optional[dict] = None) -> Solution:
    table, label = _table(basis, sex, age, annuitant=False, data=data)
    frame = table.to_dataframe().set_index("age")
    scale = RADIX / table.radix
    lx, Tx = frame.loc[age, "lx"] * scale, frame.loc[age, "Tx"] * scale
    complete = table.life_expectancy(age)
    curtate = table.life_expectancy(age, "curtate")
    assumptions = [f"Mortality: {label}.",
                   "Deaths are spread evenly over each year of age."]
    if not table.is_closed:
        assumptions.append(f"The table stops at age {table.max_age}. Years lived beyond it are "
                           "counted only if the table publishes them, so the figure may be "
                           "slightly low.")
    return Solution(
        "Expectation of life",
        {"age x": str(age), "sex": "male" if sex == "M" else "female"},
        [Step(f"Survivors at age {age}, from the table (out of {RADIX:,} at age {table.min_age})",
              [f"l_{age} = {money(lx, 1)}"]),
         Step("Total years lived after that age by those survivors",
              ["T_x = L_x + L_(x+1) + ...   where L_y = l_y - d_y / 2",
               f"T_{age} = {money(Tx, 1)}"]),
         Step("Complete expectation of life",
              ["e_x (complete) = T_x / l_x", f"  = {money(Tx, 1)} / {money(lx, 1)}",
               f"  = {complete:.2f} years"]),
         Step("Curtate expectation (whole years only), for comparison",
              ["e_x = p_x + 2p_x + 3p_x + ...", f"    = {curtate:.2f} years",
               "The complete expectation is about half a year more."])],
        f"A {'man' if sex == 'M' else 'woman'} aged {age} is expected to live another "
        f"{complete:.1f} years, to about age {age + complete:.0f}.", complete, assumptions,
        "This is an average over the lives in the table, not a prediction for one person.")


register(Solver(
    "life_expectancy", "Expectation of life",
    "Expectation of life: how many more years a person of an age is expected to live",
    ((r"life expectancy|expectation of life", 4),
     (r"expect(ed)? to live|how long .*\blive|more years|years (left|remaining)", 3)),
    (AGE, SEX, BASIS), _life_expectancy, "What is the life expectancy of a 45 year old woman?",
    uses=("table",)))


def _survival_probability(age: int, years: int, basis: str = "zimbabwe",
                          data: Optional[dict] = None) -> Solution:
    table, label = _assured_table(basis, age, data)
    if age + years > table.max_age + 1:
        raise ValueError(f"{label} stops at age {table.max_age}, so survival to age "
                         f"{age + years} cannot be read from it.")
    lx = float(table.lx(age)) * RADIX / table.radix
    probability = table.npx(age, years)
    later = lx * probability
    return Solution(
        "Probability of survival",
        {"age x": str(age), "period n": f"{years} years (to age {age + years})"},
        [Step(f"Survivors at each age, from the table (out of {RADIX:,} at age {table.min_age})",
              [f"l_{age} = {money(lx, 1)}", f"l_{age + years} = {money(later, 1)}"]),
         Step("Probability of surviving n years",
              ["n_p_x = l_(x+n) / l_x", f"  = {money(later, 1)} / {money(lx, 1)}",
               f"  = {number(probability)}"]),
         Step("Probability of dying within the period",
              ["n_q_x = 1 - n_p_x", f"  = {number(1 - probability)}"])],
        f"The probability that a life aged {age} survives {years} years is {probability:.2%}.",
        probability, [f"Mortality: {label}."],
        f"Of every 1,000 lives aged {age}, about {1000 * probability:.0f} reach age {age + years}.")


register(Solver(
    "survival_probability", "Probability of survival",
    "Probability or chance that a person of an age survives, or is still alive after, a number of years",
    ((r"surviv", 3), (r"still alive|alive at|alive in|lives? another|live (to|until)|reach(es)? age", 3),
     (r"probability|chance|odds|likelihood", 1), (r"\bdies?\b|death", 1)),
    (AGE, _years("the number of years to survive",
                 (r"surviv\w*\s+(?:for\s+|another\s+)?" + r"(\d{1,3})\s*(?:years?|yrs?)",
                  r"(?:another|next|within|in)\s+(\d{1,3})\s*(?:years?|yrs?)")), BASIS),
    _survival_probability, "What is the probability that a 30 year old survives 20 years?",
    uses=("table",)))


def _life_annuity(age: int, annual_amount: float, interest_rate: float, sex: str = "M",
                  basis: str = "zimbabwe", data: Optional[dict] = None) -> Solution:
    from ..mortality import SurvivalFunctions
    from ..pensions import pension_annuity
    table, label = _table(basis, sex, age, annuitant=True, data=data)
    v = 1 / (1 + interest_rate)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        annual = SurvivalFunctions(table, interest_rate).annuity_due(age)
        monthly = pension_annuity(table, age, interest_rate, frequency=12)
    terms = [f"k={k}: v^{k} x {k}_p_x = {number(v ** k)} x {number(table.npx(age, k))} "
             f"= {number(v ** k * table.npx(age, k))}" for k in range(3)]
    value = annual_amount * monthly
    return Solution(
        "Value of a pension payable for life",
        {"age x": str(age), "sex": "male" if sex == "M" else "female",
         "pension": f"{money(annual_amount)} a year, paid monthly in advance",
         "interest i": percent(interest_rate)},
        [Step("Discount factor", ["v = 1 / (1 + i)", f"  = {number(v)}"]),
         Step("Annuity of 1 a year paid yearly in advance",
              ["a-due_x = sum over k of v^k x k_p_x   (k = 0, 1, 2, ...)"] + terms
              + ["...", f"a-due_x = {number(annual, 4)}"]),
         Step("Adjust for monthly payment",
              ["a-due_x(12) is approximately a-due_x - 11/24",
               f"  = {number(annual, 4)} - 0.4583 = {number(annual - 11 / 24, 4)}",
               f"Exact, with deaths spread evenly over each year: {number(monthly, 4)}"]),
         Step("Value of the pension",
              ["value = pension x a-due_x(12)", f"      = {money(annual_amount)} x {number(monthly, 4)}",
               f"      = {money(value)}"])],
        f"The pension is worth {money(value)}.", value,
        [f"Mortality: {label}.", "Paid monthly in advance for life, with no increases, no "
         "guarantee period and no spouse's pension.", "Interest is an effective annual rate."],
        f"The factor {monthly:.2f} means the pension is worth about {monthly:.0f} years' "
        f"payments. Expectation of life at {age} on this table is {table.life_expectancy(age):.1f} "
        "years; the factor is lower because of discounting.")


register(Solver(
    "life_annuity", "Value of a pension payable for life",
    "Value, worth or cost of an annuity, pension or income of an annual amount paid for life from an age",
    ((r"for life|rest of (his|her|their|my) life|life annuity|lifetime", 3),
     (r"\bannuity\b|\bpension\b", 2), (r"\bincome\b", 1), (r"(a|per) year from age|from age", 1)),
    (AGE,
     Param("annual_amount", "amount", "the yearly amount of the pension", low=0,
           monthly_to_annual=True),
     RATE, SEX, BASIS),
    _life_annuity, "Value a pension of 6000 a year from age 65 at 5% interest.",
    uses=("table",)))


def _term_assurance_premium(age: int, term: int, sum_assured: float, interest_rate: float,
                            basis: str = "zimbabwe", data: Optional[dict] = None) -> Solution:
    from ..mortality import SurvivalFunctions
    table, label = _assured_table(basis, age, data)
    if age + term > table.max_age + 1:
        raise ValueError(f"{label} stops at age {table.max_age}, so a {term}-year term from age "
                         f"{age} cannot be valued on it.")
    v = 1 / (1 + interest_rate)
    functions = SurvivalFunctions(table, interest_rate)
    assurance = functions.assurance(age, term)
    annuity = functions.annuity_due(age, term)
    premium = sum_assured * assurance / annuity
    benefit_terms = []
    for k in range(min(2, term)):
        survive, die = table.npx(age, k), float(table.qx(age + k))
        benefit_terms.append(f"k={k}: v^{k + 1} x {k}_p_x x q_{age + k} = {number(v ** (k + 1))} x "
                             f"{number(survive)} x {number(die)} = {number(v ** (k + 1) * survive * die, 8)}")
    return Solution(
        "Net annual premium for a term assurance",
        {"age x": str(age), "term n": f"{term} years", "sum assured S": money(sum_assured),
         "interest i": percent(interest_rate)},
        [Step("Discount factor", ["v = 1 / (1 + i)", f"  = {number(v)}"]),
         Step("Expected present value of 1 paid at the end of the year of death, if within the term",
              ["A = sum over k of v^(k+1) x k_p_x x q_(x+k)   (k = 0 to n-1)"] + benefit_terms
              + ["...", f"A = {number(assurance)}"]),
         Step("Expected present value of 1 a year paid in advance while alive, for the term",
              ["a-due = sum over k of v^k x k_p_x   (k = 0 to n-1)", f"a-due = {number(annuity, 4)}"]),
         Step("Equation of value: premiums = benefits",
              ["P x a-due = S x A", "P = S x A / a-due",
               f"  = {money(sum_assured)} x {number(assurance)} / {number(annuity, 4)}",
               f"  = {money(premium)}"])],
        f"The net premium is {money(premium)} a year.", premium,
        [f"Mortality: {label}.", "The sum assured is paid at the end of the year of death. "
         "Premiums are paid yearly in advance for the term, or until earlier death.",
         "This is a net premium: no expenses, profit or tax."],
        f"The premium is {premium / sum_assured:.3%} of the sum assured each year.")


register(Solver(
    "term_assurance_premium", "Net annual premium for a term assurance",
    "Yearly premium or cost of term assurance: life cover of a sum assured for a term of years",
    ((r"term (assurance|insurance)|life cover|life insurance|death cover|sum assured", 3),
     (r"\bpremium", 2)),
    (AGE, Param("term", "years", "the term of the cover in years", low=1, high=100),
     Param("sum_assured", "amount", "the sum assured", low=0), RATE, BASIS),
    _term_assurance_premium,
    "Premium for a 20-year term assurance of 100000 for a 35 year old at 6% interest",
    uses=("table",)))


# ----------------------------------------------------------------------
# Reserving
# ----------------------------------------------------------------------
def _load_triangle(triangle: str, data: Optional[dict] = None):
    from ..loss_reserving import load_genins, load_raa
    if triangle == "own":
        if not data or data.get("triangle") is None:
            raise ValueError("You asked about your own triangle but did not pass one. Pass it "
                             "as data, for example ask(question, data=my_triangle).")
        own = data["triangle"]
        return own, f"your own data ({own.name})"
    loaders = {"RAA": (load_raa, "RAA (reinsurance general liability, a standard example)"),
               "GenIns": (load_genins, "GenIns (the Taylor and Ashe data, a standard example)")}
    load, label = loaders[triangle]
    return load(), label


def _chain_ladder(triangle: str, data: Optional[dict] = None) -> Solution:
    from ..loss_reserving import ChainLadder
    data, label = _load_triangle(triangle, data)
    model = ChainLadder(data)
    summary = model.summary().drop(index="Total", errors="ignore")
    factors = [f"{period}: {factor:.4f}" for period, factor in model.factors.items()]
    rows = [f"{'origin':>8} {'latest':>14} {'ultimate':>14} {'IBNR':>14}"]
    rows += [f"{str(origin):>8} {row['latest']:>14,.0f} {row['ultimate']:>14,.0f} {row['ibnr']:>14,.0f}"
             for origin, row in summary.iterrows()]
    last = summary.index[-1]
    return Solution(
        "Chain ladder reserve",
        {"triangle": label, "origin years": str(len(summary))},
        [Step("Development factors, one for each step from one development year to the next",
              ["f_j = (sum of claims at development j+1) / (sum of claims at development j),",
               "      over the origin years observed at both"] + factors),
         Step("Project each origin year to its ultimate",
              ["ultimate = latest claims x the factors still to come",
               f"For {last}: {summary.loc[last, 'latest']:,.0f} / {summary.loc[last, 'dev_to_date']:.6f}"
               f" = {summary.loc[last, 'ultimate']:,.0f}",
               "(dividing by the proportion developed is the same as multiplying by the factors)"]),
         Step("Reserve for each origin year", ["IBNR = ultimate - latest"] + rows),
         Step("Total", [f"IBNR = {model.total_ibnr:,.0f}"])],
        f"The chain ladder reserve is {model.total_ibnr:,.0f}.", float(model.total_ibnr),
        ["Development factors are volume-weighted averages over all origin years.",
         "No tail factor: claims are taken as fully developed at the last development year.",
         "The method assumes later years will develop as earlier years did. Check that before "
         "relying on the figure: see actuneo.loss_reserving.diagnose."],
        f"The most recent origin year carries {summary.loc[last, 'ibnr'] / model.total_ibnr:.0%} "
        "of the reserve, and is the least certain.")


register(Solver(
    "chain_ladder_reserve", "Chain ladder reserve",
    "IBNR claims reserve for a run-off triangle by the chain ladder method",
    ((r"chain[- ]ladder", 3), (r"\bibnr\b", 3), (r"\breserves?\b|outstanding claims", 1),
     (r"triangle", 1)),
    (TRIANGLE,), _chain_ladder, "Estimate the IBNR for the RAA triangle with the chain ladder method.",
    uses=("triangle",)))


def _mack(triangle: str, data: Optional[dict] = None) -> Solution:
    from ..loss_reserving import MackChainLadder
    data, label = _load_triangle(triangle, data)
    model = MackChainLadder(data, est_sigma="mack")
    summary = model.summary().drop(index="Total", errors="ignore")
    rows = [f"{'origin':>8} {'IBNR':>14} {'std error':>14} {'cv':>8}"]
    rows += [f"{str(origin):>8} {row['ibnr']:>14,.0f} {row['mack_se']:>14,.0f} "
             + (f"{row['cv_ibnr']:>8.0%}" if row["ibnr"] > 0 else f"{'-':>8}")
             for origin, row in summary.iterrows()]
    process, parameter = float(model.total_process_risk), float(model.total_parameter_risk)
    total = float(model.total_mack_se)
    reserve = float(model.total_ibnr)
    return Solution(
        "Mack standard error of the chain ladder reserve",
        {"triangle": label},
        [Step("The reserve, from the chain ladder", [f"IBNR = {reserve:,.0f}"]),
         Step("Standard error for each origin year, from Mack's formula",
              ["It combines the variance of each development factor with the size of the",
               "claims still to develop."] + rows),
         Step("Total standard error, split into its two sources",
              ["process risk (claims are random)          = " + f"{process:,.0f}",
               "parameter risk (factors are estimates)    = " + f"{parameter:,.0f}",
               "standard error = sqrt(process^2 + parameter^2)",
               f"               = sqrt({process:,.0f}^2 + {parameter:,.0f}^2)",
               f"               = {total:,.0f}"]),
         Step("As a proportion of the reserve",
              ["cv = standard error / IBNR", f"   = {total:,.0f} / {reserve:,.0f}",
               f"   = {total / reserve:.1%}"])],
        f"The standard error is {total:,.0f}, which is {total / reserve:.0%} of the reserve of "
        f"{reserve:,.0f}.", total,
        ["Mack's 1993 distribution-free model, with his estimate of the last variance parameter.",
         "The total is not the sum of the rows: origin years share the same estimated factors, "
         "so their errors are correlated.",
         "It measures uncertainty if the chain ladder assumptions hold. It does not allow for "
         "the method itself being wrong."],
        f"Roughly, a reserve of {reserve:,.0f} plus or minus {total:,.0f}.")


register(Solver(
    "mack_standard_error", "Mack standard error of the chain ladder reserve",
    "Mack standard error: the uncertainty or variability of the chain ladder reserve for a triangle",
    ((r"\bmack\b", 6), (r"standard error|prediction error", 6),
     (r"uncertain|variability|how reliable|how sure|confidence", 6)),
    (TRIANGLE,), _mack, "What is the Mack standard error for the GenIns triangle?",
    uses=("triangle",)))


# ----------------------------------------------------------------------
# Pensions and currency
# ----------------------------------------------------------------------
def _retirement_fund(age: int, retirement_age: int, salary: float, contribution_rate: float,
                     investment_return: float, salary_increase: float = 0.0) -> Solution:
    from ..pensions import DCProjection
    if retirement_age <= age:
        raise ValueError("The retirement age must be above the present age.")
    years = retirement_age - age
    fund = DCProjection(age, retirement_age, salary, member_rate=contribution_rate,
                        investment_return=investment_return,
                        salary_increase=salary_increase).fund_at_retirement
    j, e = investment_return, salary_increase
    monthly = (1 + j) ** (1 / 12) - 1
    within = j / (12 * monthly) if monthly else 1.0
    first = contribution_rate * salary
    if abs(j - e) > 1e-12:
        growth = ((1 + j) ** years - (1 + e) ** years) / (j - e)
        growth_lines = ["sum = ((1 + j)^n - (1 + e)^n) / (j - e)",
                        f"    = ({number((1 + j) ** years, 4)} - {number((1 + e) ** years, 4)}) / "
                        f"{j - e:.6g}", f"    = {number(growth, 4)}"]
    else:
        growth = years * (1 + j) ** (years - 1)
        growth_lines = ["With j = e: sum = n x (1 + j)^(n-1)", f"    = {number(growth, 4)}"]
    return Solution(
        "Projected retirement fund",
        {"age now": str(age), "retirement age": f"{retirement_age}, so n = {years} years",
         "salary": f"{money(salary)} a year", "contributions": f"{percent(contribution_rate)} of salary",
         "investment return j": percent(j), "salary increases e": percent(e)},
        [Step("Contributions in the first year",
              ["C = contribution rate x salary", f"  = {contribution_rate:.6g} x {money(salary)}",
               f"  = {money(first)}"]),
         Step("Value at the end of a year of 1 paid in twelve monthly instalments",
              ["s = j / (12 x ((1 + j)^(1/12) - 1))", f"  = {j:.6g} / (12 x {number(monthly, 8)})",
               f"  = {number(within)}"]),
         Step("Accumulate every year's contributions to retirement",
              ["Year t's contributions are C x (1 + e)^t x s at the end of that year,",
               "and then grow for the n - 1 - t years left."] + growth_lines),
         Step("Fund at retirement",
              ["F = C x s x sum", f"  = {money(first)} x {number(within)} x {number(growth, 4)}",
               f"  = {money(fund, 0)}"])],
        f"The projected fund at age {retirement_age} is {money(fund, 0)}.", fund,
        ["Contributions are paid monthly in arrears. Salary is reviewed once a year.",
         "No charges, no existing fund, and the return is the same every year. The result is "
         "one outcome for one set of assumptions, not a forecast."],
        f"Contributions paid in total {money(first * (((1 + e) ** years - 1) / e if e else years), 0)}; "
        "the rest is investment return.")


register(Solver(
    "retirement_fund", "Projected retirement fund",
    "Size of a retirement fund or pot at retirement from salary, contributions and investment return",
    ((r"retirement (fund|pot|savings)|\bpot\b|fund at retirement|provident fund", 3),
     (r"\bretire", 2), (r"contribut|put in|employer adds|pay in", 2)),
    (Param("age", "age", "your present age", low=15, high=80),
     Param("retirement_age", "age", "the retirement age", low=40, high=85,
           cues=(r"retir\w*\s+(?:at|age|aged|from)\s+(?:age\s+)?(\d{2})", r"until\s+(?:age\s+)?(\d{2})",
                 r"(?:to|at)\s+age\s+(\d{2})")),
     Param("salary", "amount", "the yearly salary", low=0,
           cues=(r"(?:salary|earn\w*|paid|income)(?:\s+\w+){0,2}?\s+\$?" + NUM,)),
     Param("contribution_rate", "rate", "the total contribution rate", low=0, high=1,
           cues=(r"contribut\w*(?:\s+\w+){0,2}?\s+" + NUM + r"\s?%", NUM + r"\s?%\s+contribution",
                 r"put(?:ting)?\s+in\s+" + NUM + r"\s?%", r"pay(?:ing)?\s+in\s+" + NUM + r"\s?%")),
     Param("investment_return", "rate", "the investment return", low=-0.5, high=1,
           cues=(r"returns?(?:\s+\w+){0,3}?\s+" + NUM + r"\s?%", NUM + r"\s?%\s+(?:investment\s+)?return",
                 r"grow\w*\s+at\s+" + NUM + r"\s?%", r"earn\w*\s+" + NUM + r"\s?%")),
     Param("salary_increase", "rate", "the rate of salary increases", required=False, default=0.0,
           low=-0.5, high=1,
           cues=(r"salary\s+(?:increas\w+|growth|rising|rises?)(?:\s+\w+){0,2}?\s+" + NUM + r"\s?%",
                 r"(?:increas\w+|rais\w+)\s+of\s+" + NUM + r"\s?%"))),
    _retirement_fund,
    "I am 30 and retire at 60 on a salary of 12000, contributing 10% with a 7% return. How big is my pot?"))


def _zwl_to_zig(amount: float) -> Solution:
    from .. import zimbabwe
    value = zimbabwe.zwl_to_zwg(amount)
    return Solution(
        "Zimbabwe dollars to Zimbabwe Gold",
        {"amount": f"ZWL {money(amount)}"},
        [Step("Conversion rate set in April 2024",
              [f"1 ZiG = {zimbabwe.ZWL_PER_ZWG:,.4f} ZWL"]),
         Step("Convert", ["ZiG = ZWL / rate", f"    = {money(amount)} / {zimbabwe.ZWL_PER_ZWG:,.4f}",
                          f"    = {money(value)}"])],
        f"ZWL {money(amount)} is ZiG {money(value)}.", value,
        ["The rate at which Zimbabwe dollar balances were converted when ZiG replaced the "
         "Zimbabwe dollar (Statutory Instrument 60 of 2024). It is not a market exchange rate."])


register(Solver(
    "zwl_to_zig", "Zimbabwe dollars to Zimbabwe Gold",
    "Convert an amount of Zimbabwe dollars (ZWL) to Zimbabwe Gold (ZiG)",
    ((r"\bzwl\b|zimbabwe(an)? dollars?", 3), (r"\bzig\b|\bzwg\b|zimbabwe gold", 3), (r"convert", 1)),
    (Param("amount", "amount", "the amount in ZWL", low=0),),
    _zwl_to_zig, "Convert 1,000,000 ZWL to ZiG."))


# ----------------------------------------------------------------------
# Investment appraisal
# ----------------------------------------------------------------------
def _net_present_value(interest_rate: Optional[float] = None,
                       data: Optional[dict] = None) -> Solution:
    from ..finance import Cashflows
    flows = (data or {}).get("cashflows")
    if not flows or len(flows) < 2:
        raise ValueError("I need the cash flows: list them in the question (\"cash flows of "
                         "-1000, 300, 400, 500\") or pass them as data, one amount per year "
                         "starting now.")
    flows = [float(amount) for amount in flows]
    times = list(range(len(flows)))
    project = Cashflows(times, flows)
    given = {"cash flows": ", ".join(money(amount) for amount in flows) + "  (at times 0, 1, 2, ...)"}
    steps, assumptions = [], ["One cash flow at the start (time 0) and one at the end of each "
                              "year after it.", "Interest is an effective annual rate."]
    value = float("nan")
    answer = []
    if interest_rate is not None:
        v = 1 / (1 + interest_rate)
        given["interest i"] = percent(interest_rate)
        rows = [f"{'t':>3} {'cash flow':>14} {'v^t':>10} {'present value':>15}"]
        rows += [f"{t:>3} {amount:>14,.2f} {v ** t:>10.6f} {amount * v ** t:>15,.2f}"
                 for t, amount in zip(times, flows)]
        value = project.net_present_value(interest_rate)
        steps.append(Step("Discount each cash flow",
                          ["PV of a cash flow at time t = cash flow x v^t,  v = 1 / (1 + i) = "
                           + number(v)] + rows))
        steps.append(Step("Net present value: add them up", [f"NPV = {money(value)}"]))
        answer.append(f"The net present value at {percent(interest_rate)} is {money(value)}.")
    changes = sum(1 for a, b in zip(flows, flows[1:]) if a * b < 0)
    if changes >= 1:
        rate = project.internal_rate_of_return()
        steps.append(Step("Internal rate of return: the rate at which the NPV is zero",
                          ["Solve  sum of cash flow x (1 + i)^-t = 0  for i (by trial)",
                           f"i = {rate:.4%}",
                           f"Check: NPV at {rate:.4%} = {money(project.net_present_value(rate))}"]))
        answer.append(f"The internal rate of return is {rate:.2%}.")
        if interest_rate is None:
            value = rate
        if changes > 1:
            assumptions.append("The cash flows change sign more than once, so there may be "
                               "more than one rate at which the NPV is zero.")
    check = None
    if interest_rate is not None and changes >= 1:
        verdict = "above" if rate > interest_rate else "below"
        check = (f"The internal rate of return is {verdict} {percent(interest_rate)}, which agrees "
                 f"with a {'positive' if value > 0 else 'negative'} NPV at that rate.")
    if not steps:
        raise ValueError("I need an interest rate to work out a net present value, and the cash "
                         "flows must change sign for there to be an internal rate of return.")
    return Solution("Net present value and internal rate of return", given, steps,
                    " ".join(answer), value, assumptions, check)


register(Solver(
    "net_present_value", "Net present value and internal rate of return",
    "Net present value (NPV) and internal rate of return (IRR) of a series of yearly cash flows, "
    "for appraising a project or investment",
    ((r"\bnpv\b|net present value", 6), (r"\birr\b|internal rate of return", 6),
     (r"cash\s?flows?", 4), (r"\bproject\b|appraisal|payback", 2)),
    (Param("interest_rate", "rate", "the interest rate", required=False, default=None,
           low=-0.5, high=1.0),),
    _net_present_value, "What is the NPV at 8% of cash flows of -1000, 300, 400, 500?",
    uses=("cashflows",)))


# ----------------------------------------------------------------------
# IFRS 17, premium allocation approach
# ----------------------------------------------------------------------
def _paa_liability(premium: float, elapsed_months: float, coverage_months: float = 12,
                   acquisition_costs: float = 0.0) -> Solution:
    if elapsed_months > coverage_months:
        raise ValueError("The months elapsed cannot exceed the coverage period.")
    share = elapsed_months / coverage_months
    revenue = premium * share
    amortised = acquisition_costs * share
    liability = premium - acquisition_costs - revenue + amortised
    steps = [
        Step("Proportion of the coverage provided so far (passage of time)",
             ["elapsed / coverage", f"  = {elapsed_months:g} / {coverage_months:g}",
              f"  = {number(share, 4)}"]),
        Step("Insurance revenue recognised to date",
             ["revenue = premium x proportion", f"        = {money(premium)} x {number(share, 4)}",
              f"        = {money(revenue)}"]),
    ]
    if acquisition_costs:
        steps.append(Step("Acquisition cash flows amortised to date (they are deferred)",
                          ["amortised = acquisition costs x proportion",
                           f"          = {money(acquisition_costs)} x {number(share, 4)}",
                           f"          = {money(amortised)}"]))
    steps.append(Step("Liability for remaining coverage",
                      ["LRC = premiums received - acquisition cash flows paid",
                       "      + amortisation of acquisition cash flows - revenue recognised",
                       f"    = {money(premium)} - {money(acquisition_costs)} + {money(amortised)} "
                       f"- {money(revenue)}", f"    = {money(liability)}"]))
    return Solution(
        "IFRS 17 premium allocation approach: liability for remaining coverage",
        {"premium received": money(premium), "coverage period": f"{coverage_months:g} months",
         "elapsed": f"{elapsed_months:g} months",
         "acquisition cash flows": money(acquisition_costs)},
        steps, f"The liability for remaining coverage is {money(liability)}, after insurance "
               f"revenue of {money(revenue)} to date.", liability,
        ["The whole premium was received at the start, and acquisition cash flows were paid "
         "then and deferred (not expensed).",
         "Revenue is recognised evenly with the passage of time (IFRS 17.B126(a)).",
         "No significant financing component, so no interest is accreted, and the group is not "
         "onerous, so there is no loss component.",
         "The liability for incurred claims is separate and is not included."],
        f"The same as (premium - acquisition costs) x unexpired proportion = "
        f"{money(premium - acquisition_costs)} x {number(1 - share, 4)} = "
        f"{money((premium - acquisition_costs) * (1 - share))}.")


register(Solver(
    "paa_liability", "IFRS 17 premium allocation approach: liability for remaining coverage",
    "IFRS 17 premium allocation approach (PAA): liability for remaining coverage (LRC) and "
    "insurance revenue from the premium, the coverage period and the time elapsed",
    ((r"\bpaa\b|premium allocation", 6), (r"\blrc\b|remaining coverage", 6),
     (r"ifrs\s?17", 3), (r"insurance revenue|unearned", 3), (r"\bpremium", 1)),
    (Param("premium", "amount", "the premium received", low=0,
           cues=(r"premiums?\s+(?:received\s+)?(?:of|is|was|=)\s+\$?" + NUM,
                 r"\$?" + NUM + r"\s+(?:of\s+)?premium")),
     Param("elapsed_months", "amount", "the number of months elapsed", cue_only=True, low=0,
           high=120,
           cues=(r"(?:after|at the end of|first)\s+" + NUM + r"\s+months?",
                 NUM + r"\s+months?\s+(?:have\s+|has\s+)?(?:elapsed|passed|in\b|later|gone)",
                 r"month\s+" + NUM + r"\b")),
     Param("coverage_months", "amount", "the coverage period in months", required=False,
           default=12, cue_only=True, low=1, high=120,
           cues=(NUM + r"[\s-]months?[\s-](?:polic\w+|cover\w*|contract|term)",
                 r"(?:coverage|cover|policy|contract)\s+(?:period\s+)?(?:of|is|for)\s+" + NUM
                 + r"\s+months?")),
     Param("acquisition_costs", "amount", "the acquisition cash flows", required=False,
           default=0.0, cue_only=True, low=0,
           cues=(r"(?:acquisition\s+(?:costs?|cash\s?flows?|expenses?)|commission)\s+"
                 r"(?:paid\s+)?(?:of|is|was|=)\s+\$?" + NUM,
                 r"\$?" + NUM + r"\s+(?:of\s+)?(?:acquisition|commission)"))),
    _paa_liability,
    "IFRS 17 PAA: premium of 1200 for a 12-month policy with acquisition costs of 180. "
    "What is the LRC after 3 months?"))
