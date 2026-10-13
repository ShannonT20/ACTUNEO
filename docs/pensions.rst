Pension Scheme Valuation
========================

The ``actuneo.pensions`` module values defined benefit pension schemes,
produces the IAS 19 accounting figures, projects defined contribution
accounts, and carries a few calculations from Zimbabwe's Pensions and
Provident Funds Act [Chapter 24:32].

It does the arithmetic of a valuation. It does not choose the assumptions:
the discount rate, salary increases, mortality and rates of leaving service
are judgements for the actuary, and the result is only as good as they are.
Read :doc:`status` before relying on any figure.

A valuation, step by step
-------------------------

The example uses the South African tables shipped with ACTUNEO: SA85-90 for
deaths in service and the SAIML98 and SAIFL98 annuitant tables after
retirement. The members are invented.

**1. The rules of the scheme.** A pension of one sixtieth of final average
salary for each year of service from age 60, half of it continuing to a
spouse, guaranteed for five years, with twice salary paid on death in
service. Members contribute 6% of salary.

.. code-block:: python

   import pandas as pd
   from actuneo.mortality import MortalityTable
   from actuneo.pensions import (
       DBValuation, DefinedBenefitScheme, ValuationBasis, ServiceTable, zimbabwe,
   )

   scheme = DefinedBenefitScheme(
       accrual_rate=1 / 60,
       normal_retirement_age=60,
       final_average_years=3,
       spouse_fraction=0.5,
       guarantee_years=5,
       death_in_service_multiple=2,
       member_contribution_rate=0.06,
   )

**2. The assumptions.** The service table says how members leave before
retirement. Here deaths follow SA85-90, withdrawals fall from 8% a year at
age 20, and ill-health retirements start at age 45.

.. code-block:: python

   in_service = MortalityTable.from_south_africa("sa8590")
   pensioners_mortality = {
       "M": MortalityTable.from_south_africa("saiml98"),
       "F": MortalityTable.from_south_africa("saifl98"),
   }
   service_table = ServiceTable.from_mortality(
       in_service, retirement_age=60, start_age=20,
       withdrawal=lambda age: max(0.0, 0.08 - 0.002 * (age - 20)),
       ill_health=lambda age: 0.001 if age >= 45 else 0.0,
   )

   basis = ValuationBasis(
       discount_rate=0.10,
       mortality_post=pensioners_mortality,
       service_table=service_table,
       salary_scale=0.07,
       pension_increase=0.04,
       deferred_revaluation=0.04,
       proportion_married=0.8,
       spouse_age_difference=4,
   )

**3. The members.** One row each. Active members need ``age``, ``service``
and ``salary``; pensioners and deferred pensioners need ``age`` and
``pension``.

.. code-block:: python

   actives = pd.DataFrame({
       "age":     [28, 35, 42, 50, 57],
       "service": [3, 8, 15, 22, 30],
       "salary":  [14_400, 19_200, 26_400, 33_600, 40_800],
       "sex":     ["F", "M", "M", "F", "M"],
   })
   pensioners = pd.DataFrame({
       "age": [63, 70, 78], "pension": [9_600, 7_200, 4_800], "sex": ["M", "F", "M"],
   })
   deferreds = pd.DataFrame({"age": [45], "pension": [2_400], "sex": ["M"]})

**4. The result.** The summary is laid out like the financial position in a
valuation report: net assets, the liabilities by class of member, reserves,
the surplus or deficit and the funding level.

.. code-block:: python

   valuation = DBValuation(scheme, basis, actives, pensioners, deferreds,
                           assets=420_000, reserves=10_000)
   print(valuation.summary().round(3))
   print(valuation.active_results.round(0))
   print(zimbabwe.funding_level_check(valuation.funding_level)["status"])

On these figures the liabilities are 579,824 against assets of 420,000, a
funding level of 72.4%, which is below the 75% that IPEC treats as the
minimum.

**5. The contribution rate.** The standard rate pays for the benefits earned
in the coming year. A deficit is paid off over a recovery period as a level
percentage of salaries.

.. code-block:: python

   print(valuation.contribution_rate(recovery_period=10, expenses=0.01).round(4))

Here the standard rate is 12.1% of salaries and paying off the deficit over
ten years adds 13.6%. With 1% for expenses the total is 26.8%, of which
members pay 6% and the employer 20.8%.

**6. Other methods and assumptions.**

.. code-block:: python

   print(valuation.method_comparison().round(3))
   print(valuation.sensitivities().round(3))
   print(f"Duration of the liabilities: {valuation.duration():.1f} years")
   print(f"Aggregate method rate: {valuation.aggregate_contribution_rate():.1%}")

How benefits are valued
-----------------------

Conventions
~~~~~~~~~~~

* Ages of active members are whole numbers at the valuation date.
* A member who leaves during a year of age, by withdrawal, death, ill health
  or early retirement, leaves half way through it. Those still in service at
  the normal retirement age retire on that birthday.
* Salaries are reviewed once a year, at the valuation date and on each
  anniversary. The ``salary`` supplied is the rate from the valuation date.
  Final average salary is the exact average of the salary rate over the
  averaging period before leaving.
* Survival to a fractional age is by linear interpolation of ``lx``.
* Benefits are discounted to the date of leaving at the post-retirement rate
  inside the annuity factor, and from there to the valuation date at the
  pre-retirement rate. The two are the same unless
  ``post_retirement_discount_rate`` is set.

The benefits
~~~~~~~~~~~~

For a member who leaves at time *t* with service *s* and final average
salary *F*, the value at that date of the benefit is:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Cause
     - Value at the date of leaving
   * - Retirement at the normal age
     - ``accrual_rate × s × F × annuity``, plus ``lump_sum_accrual × s × F``
   * - Early retirement
     - The same, with the pension reduced by ``early_retirement_reduction``
       for each year before the normal age and the annuity taken at the
       age of retirement
   * - Ill-health retirement
     - The pension without reduction, on service to the date of retirement
       or, if ``ill_health_service="prospective"``, to the normal retirement
       age. Valued on the ill-health mortality table if one is given
   * - Death in service
     - ``death_in_service_multiple`` times the salary rate, plus a spouse's
       pension of ``death_in_service_spouse_fraction`` of the member's
       pension, weighted by the proportion married
   * - Withdrawal
     - ``"deferred"``: the pension and lump sum earned to the date of
       leaving, revalued to the normal retirement age and discounted back,
       with no deaths in deferment unless ``mortality_deferment`` is given.
       ``"refund"``: the member's contributions with interest

The annuity is :func:`~actuneo.pensions.pension_annuity`. It allows for the
payment frequency, increases on each anniversary, the guarantee period and
the spouse's pension. During a guarantee the full pension is paid whether
the member is alive or not, and the spouse's pension starts when the
guarantee ends. If a commutation factor is set, the cash taken at retirement
replaces that part of the pension; if not, commutation is taken to be on
terms that leave the value unchanged.

The value of all a member's benefits is the sum, over every cause and year
of leaving, of the probability of leaving then, the discount factor and the
value above.

Funding methods
~~~~~~~~~~~~~~~

The methods differ in how much of that value they treat as already earned.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Method
     - Liability
     - Normal cost
   * - ``projected_unit``
     - Each benefit times past service over service at the date of leaving,
       with salaries projected
     - Each benefit times the service of the coming year over service at
       leaving
   * - ``current_unit``
     - As projected unit, with salaries held at today's level
     - As projected unit, with salaries held at today's level
   * - ``entry_age``
     - Value of all benefits less the value of future normal costs
     - The level percentage of salary that would pay for all benefits from
       the age at entry
   * - ``attained_age``
     - The projected unit liability
     - The level percentage of future salaries that pays for the benefits
       of future service
   * - Aggregate
     - None for individual members
     - Value of all benefits less the assets, spread over future salaries

Under the unit methods a benefit is earned evenly over the service that
qualifies for it, which is how IAS 19 paragraph 70 attributes benefit when
later service earns more than earlier service. Where ``max_service`` caps
pensionable service, nothing is earned after the cap is reached
(paragraph 73).

The standard contribution rate is the normal cost divided by the value of
the salaries expected to be paid over the coming year.

Under the current unit method, salary increases on service already worked
are not funded in advance. They appear as a cost each year, on top of the
normal cost shown.

IAS 19
------

IAS 19 requires the projected unit credit method. :class:`~actuneo.pensions.IAS19Period`
takes the obligation and plan assets at the start and end of a period and
the cash flows between, and returns the cost in profit or loss, the
remeasurements in other comprehensive income and the reconciliations.

.. code-block:: python

   from actuneo.pensions import IAS19Period

   period = IAS19Period(
       opening_obligation=1_000, opening_assets=800, discount_rate=0.10,
       current_service_cost=90, closing_obligation=1_200, closing_assets=930,
       benefits_paid=60, employer_contributions=100, member_contributions=20,
   )
   print(period.profit_or_loss())
   print(period.other_comprehensive_income())
   print(period.net_liability_reconciliation())

The current service cost is measured at the end of the period: the normal
cost from a valuation at the start, with a year's interest, less the
members' contributions. :meth:`~actuneo.pensions.IAS19Period.from_valuations`
does this from two valuations.

Interest is at the opening discount rate on the opening balances, adjusted
for cash flows at the point in the period set by ``cash_flow_timing``. With
an ``asset_ceiling``, a net asset is limited to the economic benefit
available and the change in that limit goes to other comprehensive income.

Defined contribution
--------------------

:class:`~actuneo.pensions.DCProjection` accumulates contributions and
investment returns to retirement and converts the account to a pension.

.. code-block:: python

   from actuneo.pensions import DCProjection, DCFundValuation

   member = DCProjection(
       age=35, retirement_age=60, salary=15_000, fund=8_000,
       member_rate=0.05, employer_rate=0.075,
       salary_increase=0.04, investment_return=0.07, inflation=0.03,
       asset_charge=0.01,
   )
   factor = member.annuity_factor(pensioners_mortality["M"], 0.05, increase=0.02)
   print(f"Fund at retirement:    {member.fund_at_retirement:,.0f}")
   print(f"In today's money:      {member.real_fund_at_retirement:,.0f}")
   print(f"Replacement ratio:     {member.replacement_ratio(factor):.1%}")
   print(f"Rate needed for 60%:   {member.required_contribution_rate(0.6, factor):.1%}")

:class:`~actuneo.pensions.DCFundValuation` sets a fund's assets against its
members' accounts and reserves, and gives the bonus that a surplus would
support.

.. code-block:: python

   fund = DCFundValuation(assets=1_100_000, member_accounts=900_000,
                          pensioner_liability=50_000, investment_reserve=40_000,
                          contingency_reserve=10_000)
   print(fund.summary())
   print(f"Bonus that keeps a 5% margin: {fund.bonus_rate(target_funding_level=1.05):.2%}")

Zimbabwe: the Pensions and Provident Funds Act
----------------------------------------------

``actuneo.pensions.zimbabwe`` holds what could be taken from the Act
(No. 3 of 2022) and from IPEC's publications.

.. list-table::
   :header-rows: 1
   :widths: 32 38 30

   * - Item
     - What it is
     - Source
   * - ``funding_level_check``
     - Compares a funding level with 100% and with the 75% minimum
     - IPEC trustee training material and quarterly pensions reports. The
       instrument that sets 75% was not located
   * - ``dissolution_reserves``
     - Reduction of minimum individual reserves in proportion when assets
       fall short on dissolution
     - Section 18(1)(b) of the Act
   * - ``surplus_share_on_exit``
     - Leavers' share of surplus and reserves, in proportion to past
       service liability
     - Section 18(3) of the Act
   * - ``PENSION_INCREASE_INTERVAL_YEARS``
     - Pension increases at least every three years for defined benefit
       funds and every year for defined contribution funds
     - Section 18(1)(d) of the Act
   * - ``SURPLUS_USES``
     - The purposes for which a board may use a surplus, with the
       Commission's approval
     - Section 20(2) of the Act

.. code-block:: python

   shortfall = zimbabwe.dissolution_reserves(
       assets=800_000,
       minimum_individual_reserves=[400_000, 200_000],
       benefits_previously_paid=[100_000, 0],
       annuity_cost=500_000,
   )
   print(shortfall, shortfall.attrs["ratio"])

   share = zimbabwe.surplus_share_on_exit(
       leaver_past_service_liability=50_000, total_past_service_liability=1_000_000,
       actuarial_surplus=200_000, investment_reserve=60_000,
   )
   print(share)

Two amounts that the Act relies on are **not** implemented. The *minimum
individual reserve* and the *minimum pension increase* are each "determined
in accordance with the formula approved by the Commission" (section 2).
Those formulas are not in the Act, so the reserves have to be supplied.

How it was checked
------------------

* **Published figures.** The projected unit method reproduces the example
  illustrating paragraph 68 of IAS 19: a lump sum of 1% of final salary for
  each of five years of service, salary rising at 7%. Opening and closing
  obligations, current service cost and interest agree to the unit for every
  year at discount rates of 10%, 4% and 7%, as printed in IFRS
  Interpretations Committee agenda paper 9 of November 2012.
* **Existing functions.** Annual annuity factors, increasing pensions,
  guarantees and spouse's pensions agree with the life contingency functions
  already in ACTUNEO. The service table agrees with the multiple decrement
  table.
* **Hand calculation.** One year from retirement, with every cause of
  leaving active, the value of each benefit is written out in the tests and
  compared.
* **Identities.** If experience follows the assumptions, the liability and
  the normal cost together pay for the benefits of those who leave in the
  year and the liability of those who stay. This holds for the projected
  unit and entry age methods at several ages. The analysis of surplus adds
  up, and shows no gain or loss when nothing differs from the assumptions.
  The IAS 19 reconciliations tie.

This is internal evidence. No valuation produced by this module has been
compared with one signed by an actuary.

Sources
-------

* IAS 19 *Employee Benefits*, paragraphs 67 to 73 and 123.
* IFRS Interpretations Committee, agenda paper 9, November 2012:
  *Measurement of the net DBO for post-employment benefit plans with
  employee contributions* (reproduces the example illustrating paragraph 68).
* Pensions and Provident Funds Act [Chapter 24:32], No. 3 of 2022, Zimbabwe.
* Insurance and Pensions Commission, *Understanding Actuarial Valuation
  Reports* (trustee training presentation), and quarterly pensions reports.
