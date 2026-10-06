Quick Start Guide
=================

This guide will help you get started with ACTUNEO. Every example on this
page is run by the test suite.

Conventions
-----------

ACTUNEO follows the International Actuarial Notation:

* Values are for a unit benefit (sum assured of 1, annuity of 1 per year).
  Multiply by the sum assured or annual amount.
* Interest rates are annual effective rates written as decimals (0.05 for 5%).
* Death benefits are paid at the end of the year of death unless you ask for
  payment at the moment of death (``discrete=False``).
* An *immediate* annuity is paid in arrears, an *annuity-due* in advance.
  Premiums are paid annually in advance.

Mortality Tables
----------------

Zimbabwe 2023 Tables
~~~~~~~~~~~~~~~~~~~~

The mortality tables launched by the Insurance and Pensions Commission (IPEC)
in July 2023 are shipped with the package:

.. code-block:: python

   from actuneo import zimbabwe
   from actuneo.mortality import MortalityTable

   print(zimbabwe.mortality_tables())

   mt = MortalityTable.from_zimbabwe_2023("male_assured_lives")
   print(mt)
   print(f"q40 = {mt.qx(40):.6f}")
   print(f"Life expectancy at 40: {mt.life_expectancy(40):.1f} years")
   print(f"20-year survival from 40: {mt.npx(40, 20):.4f}")

   # Life table derived from qx, and the columns exactly as published
   life_table = mt.to_dataframe()
   published = mt.published

Some of the published tables stop before the end of life (group life and
pre-retirement pension tables stop at age 70, the others at age 100). Term
calculations inside the tabulated ages are unaffected. Whole life calculations
assume all survivors die in the year after the last tabulated age and issue a
warning.

Your Own Table
~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from actuneo.mortality import MortalityTable

   ages = np.arange(20, 111)
   qx = np.minimum(0.0005 * 1.09 ** (ages - 20), 1.0)
   qx[-1] = 1.0  # close the table at the last age
   table = MortalityTable(ages, qx, name="Example Table")

   # Or from a file with "age" and "qx" columns
   # table = MortalityTable.from_csv("my_table.csv")

Life Contingencies
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from actuneo.mortality import SurvivalFunctions, CommutationFunctions

   sf = SurvivalFunctions(table, interest_rate=0.05)

   print(f"10p30        = {sf.npx(30, 10):.5f}")
   print(f"0.5p30 (UDD) = {sf.tpx(30, 0.5):.5f}")
   print(f"A30          = {sf.assurance(30):.5f}")
   print(f"A30:20 term  = {sf.assurance(30, 20):.5f}")
   print(f"ä30          = {sf.annuity_due(30):.4f}")
   print(f"a30:20       = {sf.annuity_immediate(30, 20):.4f}")
   print(f"20E30        = {sf.pure_endowment(30, 20):.5f}")

   # Commutation columns Dx, Nx, Sx, Cx, Mx, Rx
   cf = CommutationFunctions(table, interest_rate=0.05)
   print(cf.to_dataframe().head())

Financial Calculations
----------------------

Interest Theory
~~~~~~~~~~~~~~~

.. code-block:: python

   from actuneo.finance import InterestTheory

   it = InterestTheory(interest_rate=0.05)

   print(f"Accumulation of 1000 over 10 years: {it.future_value(1000, 10):.2f}")
   print(f"Present value of 1000 due in 10 years: {it.present_value(1000, 10):.2f}")
   print(f"Rate of discount d: {it.discount_rate():.5f}")
   print(f"Force of interest: {it.force_of_interest():.5f}")
   print(f"Nominal rate convertible monthly: {it.nominal_rate(0.05, 12):.5f}")
   print(f"Level instalment on a 10-year loan of 1000: {it.loan_payment(1000, 10):.2f}")

Yield Curves
~~~~~~~~~~~~

.. code-block:: python

   from actuneo.finance import YieldCurve

   yc = YieldCurve([1, 2, 5, 10, 30], [0.03, 0.035, 0.045, 0.055, 0.065])

   print(f"15-year spot rate: {yc.get_yield(15):.3%}")
   print(f"5-year discount factor: {yc.get_discount_factor(5):.4f}")
   print(f"Forward rate from year 5 to year 10: {yc.get_forward_rate(5, 10):.3%}")

   # Zero rates bootstrapped from par yields
   par_curve = YieldCurve.from_par_rates([1, 2, 3, 5], [0.03, 0.035, 0.04, 0.045],
                                         coupon_freq=1)

Life Insurance
--------------

.. code-block:: python

   from actuneo.life import LifeAssurance

   la = LifeAssurance(table, interest_rate=0.05)
   sum_assured = 100_000

   single = sum_assured * la.whole_life_assurance(30)
   print(f"Whole life single premium at 30: {single:,.2f}")

   term_premium = sum_assured * la.net_annual_premium(30, 20, "term")
   endowment_premium = sum_assured * la.net_annual_premium(30, 20, "endowment")
   print(f"20-year term annual premium: {term_premium:,.2f}")
   print(f"20-year endowment annual premium: {endowment_premium:,.2f}")

   # Net premium reserve per unit sum assured after 10 years
   reserve = sum_assured * la.reserve_endowment(30, 20, 10)
   print(f"Endowment reserve at duration 10: {reserve:,.2f}")

   # Two lives: benefit on the first death
   print(f"Joint life assurance (40, 45): {la.joint_life_assurance(40, 45):.5f}")

Annuities
---------

.. code-block:: python

   from actuneo.life import Annuities

   ann = Annuities(table, interest_rate=0.05)
   pension = 12_000

   print(f"Pension from 65, annually in arrears: {ann.life_annuity_immediate(65, pension):,.2f}")
   print(f"Pension from 65, monthly in advance:  {ann.monthly_life_annuity(65, pension):,.2f}")
   print(f"With 5 years guaranteed:              {ann.guaranteed_annuity(65, 5, pension):,.2f}")
   print(f"Deferred from age 45:                 {ann.deferred_life_annuity(45, 20, pension):,.2f}")
   print(f"Spouse's reversion (65, 60):          {ann.reversionary_annuity(65, 60, pension):,.2f}")

Claims Triangles
----------------

.. code-block:: python

   from actuneo.loss_reserving import Triangle, ChainLadder, MackChainLadder, load_raa

   raa = load_raa()            # a well-known example triangle
   print(raa)

   print(raa.latest_diagonal())
   print(raa.to_incremental())
   print(raa.age_to_age().round(3))   # link ratios with simple and volume averages

   cl = ChainLadder(raa)
   print(cl.summary().round(0))

   mack = MackChainLadder(raa, est_sigma="mack")
   print(mack.summary().round(3))
   print(f"Total IBNR {mack.total_ibnr:,.0f}, standard error {mack.total_mack_se:,.0f}")

Build a triangle from your own data, either as a table or one row per
payment:

.. code-block:: python

   import pandas as pd

   payments = pd.DataFrame({
       "accident_year": [2021, 2021, 2021, 2022, 2022, 2023],
       "development":   [1, 2, 3, 1, 2, 1],
       "paid":          [100.0, 50.0, 15.0, 110.0, 66.0, 120.0],
   })
   tri = Triangle.from_long(payments, origin="accident_year", development="development",
                            value="paid", cumulative=False)
   print(ChainLadder(tri).summary())

Reserving Methods
-----------------

The methods below all start from a ``Triangle``. The example is a triangle of
cumulative claim payments for accident years 2008 to 2012.

.. code-block:: python

   import numpy as np
   from actuneo.loss_reserving import (
       Triangle, ChainLadder, InflationAdjustedChainLadder, BornhuetterFerguson,
       AverageCostPerClaim, MackChainLadder, BootChainLadder,
   )

   nan = np.nan
   paid = Triangle(
       [[786, 1410, 2216, 2440, 2519],
        [904, 1575, 2515, 2796, nan],
        [995, 1814, 2880, nan, nan],
        [1220, 2142, nan, nan, nan],
        [1182, nan, nan, nan, nan]],
       origin=range(2008, 2013), development=range(5),
   )

   # Basic chain-ladder, and a check of how well it fits the past
   cl = ChainLadder(paid)
   print(cl.factors.round(3))
   print(f"Reserve: {cl.reserve():,.0f}")
   print(cl.fit_errors().round(0))        # actual less fitted payments

   # Explicit inflation: past rates by calendar year, 10% a year in future
   inflated = InflationAdjustedChainLadder(
       paid, past_inflation=[0.051, 0.064, 0.073, 0.054], future_inflation=0.10
   )
   print(f"Inflation-adjusted reserve: {inflated.reserve():,.0f}")

   # Bornhuetter-Ferguson: premium and an expected loss ratio
   bf = BornhuetterFerguson(paid, premium=[3300, 3600, 4100, 4700, 4800], loss_ratio=0.80)
   print(bf.summary().round(0))

   # A tail factor, supplied or estimated from the decay of the factors
   print(f"With a 2% tail: {ChainLadder(paid, tail=1.02).reserve():,.0f}")

Average cost per claim needs a triangle of claim numbers on the same basis as
the claims triangle:

.. code-block:: python

   claims = Triangle([[632, 714, 788, 822], [729, 784, 803, nan],
                      [800, 855, nan, nan], [824, nan, nan, nan]])
   numbers = Triangle([[52, 60, 66, 70], [54, 63, 65, nan],
                       [60, 70, nan, nan], [65, nan, nan, nan]])

   acpc = AverageCostPerClaim(claims, numbers, method="chain_ladder")
   print(acpc.summary().round(3))
   # The triangle holds incurred claims, so deduct what has been paid
   print(f"Reserve: {acpc.reserve(paid_to_date=1902):,.0f}")

Uncertainty of the reserve, from Mack's formula or by simulation:

.. code-block:: python

   from actuneo.loss_reserving import load_genins

   triangle = load_genins()

   mack = MackChainLadder(triangle, est_sigma="mack")
   print(f"Mack: reserve {mack.total_ibnr:,.0f}, standard error {mack.total_mack_se:,.0f}")

   boot = BootChainLadder(triangle, n_simulations=2000, seed=1)
   print(boot.summary(quantiles=[0.75, 0.995]).round(0))

Working with Real Data
----------------------

Claims usually arrive as a listing, one row per payment. ``from_transactions``
builds the triangle from the dates, on a yearly, quarterly or monthly basis:

.. code-block:: python

   import pandas as pd
   from actuneo.loss_reserving import Triangle, ChainLadder

   listing = pd.DataFrame({
       "loss_date": ["2021-03-10", "2021-03-10", "2021-11-02", "2022-06-30",
                     "2022-06-30", "2023-01-15"],
       "paid_date": ["2021-05-01", "2022-02-01", "2023-08-20", "2022-09-09",
                     "2023-03-03", "2023-12-31"],
       "amount":    [60.0, 50.0, 15.0, 110.0, 66.0, 120.0],
   })

   tri = Triangle.from_transactions(listing, origin_date="loss_date",
                                    transaction_date="paid_date", value="amount",
                                    valuation_date="2023-12-31")
   print(tri)

   # Accident years developed quarterly
   quarterly = Triangle.from_transactions(listing, "loss_date", "paid_date", "amount",
                                          development_grain="Q")
   print(quarterly.shape)

Future payments, discounting and a loss ratio estimated from the data:

.. code-block:: python

   from actuneo.finance import YieldCurve
   from actuneo.loss_reserving import CapeCod, MackChainLadder, load_raa

   raa = load_raa()
   mack = MackChainLadder(raa, est_sigma="mack")

   print(mack.cash_flows().round(0))                  # expected payments by future year
   print(f"Undiscounted reserve: {mack.reserve():,.0f}")
   print(f"Discounted at 8%:     {mack.discounted_reserve(0.08):,.0f}")

   curve = YieldCurve([1, 3, 5, 10], [0.07, 0.08, 0.085, 0.09])
   print(f"Discounted on a curve: {mack.discounted_reserve(curve):,.0f}")

   # Reserve at a confidence level, assuming a lognormal distribution
   print(mack.reserve_quantile([0.75, 0.995]).round(0))

   # Diagnostics: standardised residuals should show no pattern
   print(mack.residuals().round(2))

   # Cape Cod: Bornhuetter-Ferguson with the loss ratio estimated from the triangle
   cape_cod = CapeCod(raa, premium=[25000] * 10)
   print(f"Cape Cod loss ratio {cape_cod.loss_ratio:.1%}, reserve {cape_cod.reserve():,.0f}")

Plots need matplotlib (``pip install actuneo[viz]``) and Excel output needs
openpyxl (``pip install actuneo[excel]``):

.. code-block:: py

   raa.plot()                      # development of each origin year
   mack.plot()                     # latest claims and reserve with error bars
   mack.plot_residuals()           # residuals by development step
   BootChainLadder(raa, seed=1).plot()   # histogram of the simulated reserve

   mack.to_excel("raa_reserve.xlsx")     # summary, triangle, projection, factors, cash flows

IFRS 17: Premium Allocation Approach
------------------------------------

A group of one-year motor policies followed over four quarters. The premium
of 1,200 is received at the start (as SI 81 of 2023 requires in Zimbabwe) and
commission of 180 is paid.

.. code-block:: python

   from actuneo.ifrs17 import PAAGroup, PAAReinsuranceHeld, IFRS17Statements

   quarters = ["Q1", "Q2", "Q3", "Q4"]

   motor = PAAGroup(
       premiums_received=[1200, 0, 0, 0],
       acquisition_cash_flows=[180, 0, 0, 0],
       incurred_claims=[200, 210, 190, 220],        # claims incurred each quarter
       incurred_risk_adjustment=[10, 10, 10, 10],
       claims_paid=[100, 180, 200, 150],
       closing_lic_pv=[100, 125, 120, 185],         # reserve at each quarter end
       closing_lic_ra=[10, 14, 12, 15],
       finance_expenses=[0, 2, 3, 3],               # unwind of discount
       periods=quarters, name="Motor 2026", portfolio="Motor",
   )

   print(motor.lrc_rollforward())     # liability for remaining coverage
   print(motor.lic_rollforward())     # liability for incurred claims
   print(motor.profit_or_loss())
   print(motor.reconciliation("Q2"))  # IFRS 17.100 reconciliation for the quarter

   # 20% quota share reinsurance
   quota_share = PAAReinsuranceHeld(
       premiums_paid=[240, 0, 0, 0],
       recoveries_incurred=[40, 42, 38, 44],
       recoveries_received=[20, 36, 40, 30],
       periods=quarters,
   )

   statements = IFRS17Statements(
       [motor], [quota_share], opening_equity=500,
       investment_return=[10, 12, 12, 13], other_operating_expenses=20, tax_rate=0.25,
   )
   print(statements.profit_or_loss())
   print(statements.financial_position())
   print(statements.cash_flows())
   print(statements.key_ratios().round(3))

The liability for incurred claims comes from a reserving model, discounted
and with a risk adjustment at a chosen confidence level:

.. code-block:: python

   from actuneo.ifrs17 import LiabilityForIncurredClaims, implied_confidence_level
   from actuneo.loss_reserving import MackChainLadder, load_raa

   mack = MackChainLadder(load_raa(), est_sigma="mack")
   lic = LiabilityForIncurredClaims.from_reserving(mack, discount_rate=0.08,
                                                   confidence_level=0.75)
   print(lic)

Premiums are earned by days of cover from a policy listing:

.. code-block:: python

   import pandas as pd
   from actuneo.ifrs17 import earned_premium_by_period, unearned_premium

   policies = pd.DataFrame({
       "premium":    [365.0, 730.0, 120.0],
       "start_date": ["2025-01-01", "2025-07-01", "2025-11-01"],
       "end_date":   ["2025-12-31", "2026-06-30", "2025-11-30"],
   })
   print(earned_premium_by_period(policies, ["2025-06-30", "2025-12-31"]))
   print(unearned_premium(policies["premium"], policies["start_date"],
                          policies["end_date"], "2025-12-31"))

Grouping contracts, groups already in force, and the notes to the accounts:

.. code-block:: python

   from actuneo.ifrs17 import (
       group_contracts, claims_development_table, maturity_analysis, lic_sensitivity,
   )

   # Portfolio, annual cohort and profitability group for each policy
   book = pd.DataFrame({
       "portfolio": ["Motor", "Motor", "Fire"],
       "start_date": ["2025-03-01", "2026-01-10", "2025-06-01"],
       "expected_combined_ratio": [0.70, 1.10, 0.95],
   })
   print(group_contracts(book)[["portfolio", "cohort", "profitability", "group"]])

   # A group part way through its cover, starting from last year's closing balances
   in_force = PAAGroup(
       premiums_received=[0, 0], expected_premium=600,
       opening_lrc=510, opening_deferred_acquisition_cash_flows=90,
       opening_lic_pv=300, incurred_claims=[220, 230], claims_paid=[250, 260],
       finance_expenses=[6, 5], finance_expenses_in_oci=[2, 1],   # OCI option
       periods=["H1", "H2"],
   )
   accounts = IFRS17Statements([in_force], opening_equity=400)
   print(accounts.profit_or_loss(detailed=True))   # with other comprehensive income
   print(accounts.changes_in_equity())
   print(accounts.supplementary_position())        # unearned premium, DAC, claims reserves

   # Notes on incurred claims
   print(claims_development_table(load_raa(), discount_rate=0.08,
                                  risk_adjustment=lic.risk_adjustment).round(0))
   print(maturity_analysis(mack.cash_flows(), discount_rate=0.08).round(0))
   print(lic_sensitivity(mack.cash_flows(), discount_rate=0.08,
                         risk_adjustment=lic.risk_adjustment).round(0))

Next Steps
----------

* See :doc:`examples` for worked examples
* See the API reference for every class and function
