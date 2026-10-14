Examples
========

Worked examples using ACTUNEO. Every example on this page is run by the test
suite.

Example 1: Zimbabwe Mortality Comparison
----------------------------------------

Compare mortality between the Zimbabwe 2023 funeral assurance tables.

.. code-block:: python

   import pandas as pd
   from actuneo.mortality import MortalityTable

   names = ["funeral_principal_members", "funeral_spouses", "funeral_adult_dependents"]
   tables = {name: MortalityTable.from_zimbabwe_2023(name) for name in names}

   comparison = pd.DataFrame({
       name: {
           "q50": mt.qx(50),
           "q70": mt.qx(70),
           "survival 50 to 70": mt.npx(50, 20),
           "life expectancy at 50": mt.life_expectancy(50),
       }
       for name, mt in tables.items()
   })
   print(comparison.round(4))

Example 2: Pricing a Term Assurance
-----------------------------------

Net and office premium for a 20-year term assurance on a man aged 35.

.. code-block:: python

   from actuneo.mortality import MortalityTable
   from actuneo.life import LifeAssurance

   mt = MortalityTable.from_zimbabwe_2023("male_assured_lives")
   la = LifeAssurance(mt, interest_rate=0.08, expense_loading=0.15)

   age, term, sum_assured = 35, 20, 50_000

   single = sum_assured * la.term_assurance(age, term)
   annuity = la.temporary_life_annuity(age, term, due=True)
   net_annual = la.annual_premium(single, annuity)
   office_annual = la.annual_premium(single, annuity, gross_margin=0.25)

   print(f"Net single premium:    {single:,.2f}")
   print(f"Net annual premium:    {net_annual:,.2f}")
   print(f"Office annual premium: {office_annual:,.2f}")

Example 3: Endowment Reserves
-----------------------------

Net premium reserves at each fifth policy anniversary, and the check that
prospective and retrospective reserves agree.

.. code-block:: python

   from actuneo.mortality import MortalityTable
   from actuneo.life import LifeAssurance, Reserves

   mt = MortalityTable.from_zimbabwe_2023("male_assured_lives")
   la = LifeAssurance(mt, interest_rate=0.08)
   res = Reserves(mt, interest_rate=0.08)

   age, term, sum_assured = 30, 20, 100_000
   premium = sum_assured * la.net_annual_premium(age, term, "endowment")
   print(f"Net annual premium: {premium:,.2f}")

   for t in range(0, term + 1, 5):
       reserve = res.prospective_reserve_endowment(age, term, t, premium, sum_assured)
       print(f"Duration {t:2d}, age {age + t}: reserve {reserve:,.2f}")

Example 4: Pension Valuation
----------------------------

Value of a pension of 6,000 a year paid monthly in advance from age 76, with
a 50% spouse's pension, on the Zimbabwe post-retirement tables.

.. code-block:: python

   from actuneo.mortality import MortalityTable
   from actuneo.life import Annuities

   male = MortalityTable.from_zimbabwe_2023("post_retirement_pensions_male")
   female = MortalityTable.from_zimbabwe_2023("post_retirement_pensions_female")

   ann = Annuities(male, interest_rate=0.07)
   pension = 6_000

   member = ann.monthly_life_annuity(76, pension)
   spouse = ann.reversionary_annuity(76, 76, 0.5 * pension, table_y=female)

   print(f"Member's pension:  {member:,.2f}")
   print(f"Spouse's pension:  {spouse:,.2f}")
   print(f"Total liability:   {member + spouse:,.2f}")

Example 5: Bond Duration and Convexity
--------------------------------------

.. code-block:: python

   from actuneo.finance import DurationConvexity

   dc = DurationConvexity()
   cash_flows = [80, 80, 80, 80, 1080]   # 5-year bond, 8% annual coupon
   times = [1, 2, 3, 4, 5]
   yield_rate = 0.10

   price = sum(cf / (1 + yield_rate) ** t for cf, t in zip(cash_flows, times))
   duration = dc.modified_duration(cash_flows, times, yield_rate)
   convexity = dc.convexity(cash_flows, times, yield_rate)

   change = dc.price_change_approximation(duration, convexity, 0.01, price)
   print(f"Price {price:.2f}, modified duration {duration:.3f}, convexity {convexity:.2f}")
   print(f"Estimated price change for +1% yield: {change:.2f}")

Example 6: Chain-Ladder Reserve with Mack's Standard Error
-----------------------------------------------------------

The Taylor and Ashe triangle, as analysed in Mack (1993) and England and
Verrall (2002).

.. code-block:: python

   from actuneo.loss_reserving import MackChainLadder, load_genins

   triangle = load_genins()
   mack = MackChainLadder(triangle, est_sigma="mack")

   print(mack.factors.round(4))        # development factors
   print(mack.sigma.round(2))          # Mack's sigma by development step
   print(mack.summary().round(3))

   print(f"Total reserve:   {mack.total_ibnr:,.0f}")
   print(f"Process risk:    {mack.total_process_risk:,.0f}")
   print(f"Parameter risk:  {mack.total_parameter_risk:,.0f}")
   print(f"Standard error:  {mack.total_mack_se:,.0f}")

Example 7: Checking Whether the Chain-Ladder Can Be Trusted
-----------------------------------------------------------

Simulate a portfolio in which claims inflation jumps from 3% to 40% in the
last two calendar years, run the diagnostic tests on the triangle, and
compare the chain-ladder with a model that has calendar year effects. The
simulator knows what the claims finally cost, so the error of each method
can be measured.

.. code-block:: python

   from actuneo.loss_reserving import ChainLadder, GLMReserving, diagnose
   from actuneo.simulation import ClaimsSimulator, benchmark_reserving

   portfolio = ClaimsSimulator(
       n_years=10, first_year=2015, claims_per_year=400, claim_cv=1.0,
       inflation=0.03, pattern_concentration=3, seed=1,
   ).with_inflation_shock(2022, 0.40)

   triangle = portfolio.triangle()
   report = diagnose(triangle)
   print(report[["flag", "finding"]])

   calendar = GLMReserving(triangle, structure="calendar", trend_periods=2)
   print(calendar.calendar_inflation.round(2))

   methods = {
       "Chain-ladder": ChainLadder,
       "Calendar GLM": lambda t: GLMReserving(t, structure="calendar", trend_periods=2),
   }
   print(benchmark_reserving(portfolio, methods, n_simulations=10).round(3))

The calendar model does better here because it is told, in effect, that
recent inflation will continue. If inflation fell back, it would overstate
the reserve. The benchmark describes the simulated portfolio only.

References
----------

* Mack, T. (1993). Distribution-free calculation of the standard error of
  chain ladder reserve estimates. *ASTIN Bulletin* 23(2), 213-225.
* Mack, T. (1999). The standard error of chain ladder reserve estimates:
  recursive calculation and inclusion of a tail factor. *ASTIN Bulletin*
  29(2), 361-366.
* England, P.D. and Verrall, R.J. (2002). Stochastic claims reserving in
  general insurance. *British Actuarial Journal* 8(3), 443-518.
* Taylor, G.C. and Ashe, F.R. (1983). Second moments of estimates of
  outstanding claims. *Journal of Econometrics* 23(1), 37-61.
* Gesmann, M. et al. *ChainLadder: Statistical Methods and Models for Claims
  Reserving in General Insurance*. R package.
