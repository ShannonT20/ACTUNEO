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

Next Steps
----------

* See :doc:`examples` for worked examples
* See the API reference for every class and function
