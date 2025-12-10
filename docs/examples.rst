Examples
========

This page contains comprehensive examples demonstrating ACTUNEO's capabilities.

Example 1: Complete Mortality Analysis
---------------------------------------

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from actuneo.mortality import MortalityTable, SurvivalFunctions

   # Create a mortality table with realistic rates
   ages = np.arange(0, 121)
   
   # Gompertz-Makeham formula parameters
   A = 0.0001
   B = 0.00035
   c = 1.085
   
   qx = A + B * c ** ages
   qx = np.clip(qx, 0, 1)  # Ensure rates are between 0 and 1
   
   mt = MortalityTable(ages, qx, name="Gompertz-Makeham Table")
   
   # Calculate various metrics
   print(f"Life expectancy at birth: {mt.life_expectancy(0):.2f} years")
   print(f"Life expectancy at age 65: {mt.life_expectancy(65):.2f} years")
   
   # Plot survival curve
   sf = SurvivalFunctions(mt)
   ages_plot = np.arange(0, 101)
   survival_probs = [sf.npx(0, age) for age in ages_plot]
   
   plt.figure(figsize=(10, 6))
   plt.plot(ages_plot, survival_probs)
   plt.xlabel('Age')
   plt.ylabel('Survival Probability')
   plt.title('Survival Curve')
   plt.grid(True)
   plt.show()

Example 2: Pension Valuation
-----------------------------

.. code-block:: python

   from actuneo.mortality import MortalityTable, SurvivalFunctions
   from actuneo.life import Annuities
   import numpy as np

   # Create a mortality table
   ages = np.arange(20, 101)
   qx = 0.001 * (ages - 20) / 80
   mt = MortalityTable(ages, qx, name="Pension Table")
   
   # Initialize annuity calculator
   ann = Annuities(mt, interest_rate=0.05)
   
   # Calculate pension liability for a member retiring at 65
   # Annual pension: $50,000
   retirement_age = 65
   annual_pension = 50000
   
   # Present value of whole life annuity
   annuity_factor = ann.whole_life_annuity_due(retirement_age)
   pension_liability = annual_pension * annuity_factor
   
   print(f"Annuity factor: {annuity_factor:.4f}")
   print(f"Pension liability: ${pension_liability:,.2f}")
   
   # Calculate liability for different interest rates
   print("\nSensitivity to interest rates:")
   for rate in [0.03, 0.04, 0.05, 0.06, 0.07]:
       ann_temp = Annuities(mt, interest_rate=rate)
       factor = ann_temp.whole_life_annuity_due(retirement_age)
       liability = annual_pension * factor
       print(f"  Rate {rate:.1%}: ${liability:,.2f}")

Example 3: Life Insurance Premium Calculation
----------------------------------------------

.. code-block:: python

   from actuneo.mortality import MortalityTable
   from actuneo.life import LifeAssurance
   import numpy as np

   # Create mortality table
   ages = np.arange(20, 101)
   qx = 0.001 * (ages - 20) / 80
   mt = MortalityTable(ages, qx, name="Standard Table")
   
   # Initialize life assurance calculator
   la = LifeAssurance(mt, interest_rate=0.05)
   
   # Policyholder details
   age = 35
   sum_assured = 500000
   term = 20
   
   # Calculate premiums for different product types
   print(f"Premiums for age {age}, sum assured ${sum_assured:,}")
   print(f"\nProduct comparisons:")
   
   # Whole life
   wl_premium = la.whole_life_assurance(age, sum_assured=sum_assured)
   print(f"  Whole Life: ${wl_premium:,.2f} per year")
   
   # Term assurance
   term_premium = la.term_assurance(age, term, sum_assured=sum_assured)
   print(f"  {term}-Year Term: ${term_premium:,.2f} per year")
   
   # Endowment
   endow_premium = la.endowment_assurance(age, term, sum_assured=sum_assured)
   print(f"  {term}-Year Endowment: ${endow_premium:,.2f} per year")
   
   # Calculate total premiums paid
   print(f"\nTotal premiums over {term} years:")
   print(f"  Term: ${term_premium * term:,.2f}")
   print(f"  Endowment: ${endow_premium * term:,.2f}")

Example 4: Yield Curve Analysis
--------------------------------

.. code-block:: python

   from actuneo.finance import YieldCurve, DurationConvexity
   import numpy as np
   import matplotlib.pyplot as plt

   # Create a yield curve
   maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
   yields = [0.025, 0.028, 0.030, 0.035, 0.038, 0.042, 0.045, 0.048, 0.052, 0.053]
   
   yc = YieldCurve(maturities, yields)
   
   # Interpolate for all maturities
   all_maturities = np.linspace(0.25, 30, 100)
   interpolated_yields = [yc.get_yield(m) for m in all_maturities]
   
   # Plot yield curve
   plt.figure(figsize=(12, 6))
   plt.plot(all_maturities, [y * 100 for y in interpolated_yields], 'b-', linewidth=2)
   plt.scatter(maturities, [y * 100 for y in yields], color='red', s=50, zorder=5)
   plt.xlabel('Maturity (years)')
   plt.ylabel('Yield (%)')
   plt.title('Yield Curve')
   plt.grid(True, alpha=0.3)
   plt.show()
   
   # Calculate discount factors
   print("\nDiscount Factors:")
   for mat in [1, 5, 10, 20, 30]:
       df = yc.discount_factor(mat)
       print(f"  {mat}-year: {df:.6f}")

Example 5: Reserve Calculation
-------------------------------

.. code-block:: python

   from actuneo.mortality import MortalityTable
   from actuneo.life import LifeAssurance, Reserves
   import numpy as np

   # Setup
   ages = np.arange(20, 101)
   qx = 0.001 * (ages - 20) / 80
   mt = MortalityTable(ages, qx, name="Reserve Table")
   
   la = LifeAssurance(mt, interest_rate=0.05)
   res = Reserves(mt, interest_rate=0.05)
   
   # Policy details
   issue_age = 30
   sum_assured = 100000
   term = 20
   
   # Calculate annual premium
   annual_premium = la.endowment_assurance(issue_age, term, sum_assured=sum_assured)
   
   print(f"Endowment Assurance Policy")
   print(f"Issue age: {issue_age}")
   print(f"Term: {term} years")
   print(f"Sum assured: ${sum_assured:,}")
   print(f"Annual premium: ${annual_premium:,.2f}\n")
   
   # Calculate reserves at each policy anniversary
   print("Policy Reserves:")
   print("Year | Age | Reserve")
   print("-" * 30)
   
   for t in range(0, term + 1, 5):
       current_age = issue_age + t
       reserve = res.reserve_endowment(issue_age, term, t, annual_premium)
       print(f" {t:2d}  | {current_age} | ${reserve:,.2f}")

More Examples
-------------

For more examples and tutorials, visit:

* `GitHub Repository <https://github.com/ShannonT20/ACTUNEO>`_
* `Examples Directory <https://github.com/ShannonT20/ACTUNEO/tree/main/examples>`_

Contributing Examples
---------------------

If you have examples you'd like to share, please submit them via pull request or open an issue on GitHub!

