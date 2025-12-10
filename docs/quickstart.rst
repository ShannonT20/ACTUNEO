Quick Start Guide
=================

This guide will help you get started with ACTUNEO.

Basic Usage
-----------

Import the library:

.. code-block:: python

   import actuneo
   from actuneo.mortality import MortalityTable, SurvivalFunctions
   from actuneo.finance import InterestTheory, YieldCurve
   from actuneo.life import LifeAssurance, Annuities

Mortality Tables
----------------

Creating a Mortality Table
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from actuneo.mortality import MortalityTable

   # Create a simple mortality table
   ages = np.arange(20, 101)
   qx = 0.001 * (ages - 20) / 80  # Simplified mortality rates
   
   mt = MortalityTable(ages, qx, name="Example Table")

Life Expectancy
~~~~~~~~~~~~~~~

.. code-block:: python

   # Calculate life expectancy at age 30
   le = mt.life_expectancy(30)
   print(f"Life expectancy at age 30: {le:.1f} years")

Survival Probabilities
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from actuneo.mortality import SurvivalFunctions

   sf = SurvivalFunctions(mt)
   
   # 20-year survival probability from age 30
   survival_prob = sf.npx(30, 20)
   print(f"20-year survival probability: {survival_prob:.3f}")
   
   # 5-year death probability from age 40
   death_prob = sf.nqx(40, 5)
   print(f"5-year death probability: {death_prob:.3f}")

Financial Calculations
----------------------

Interest Theory
~~~~~~~~~~~~~~~

.. code-block:: python

   from actuneo.finance import InterestTheory

   it = InterestTheory(interest_rate=0.05)
   
   # Future value
   fv = it.future_value(1000, 10)
   print(f"Future value: ${fv:.2f}")
   
   # Present value
   pv = it.present_value(1000, 10)
   print(f"Present value: ${pv:.2f}")
   
   # Discount factor
   v = it.discount_factor(10)
   print(f"Discount factor: {v:.4f}")

Yield Curves
~~~~~~~~~~~~

.. code-block:: python

   from actuneo.finance import YieldCurve

   # Create a yield curve
   maturities = [1, 2, 5, 10, 30]
   yields = [0.03, 0.035, 0.045, 0.055, 0.065]
   
   yc = YieldCurve(maturities, yields)
   
   # Get yield at any maturity
   yield_15y = yc.get_yield(15)
   print(f"15-year yield: {yield_15y:.3%}")
   
   # Get discount factor
   df_10y = yc.discount_factor(10)
   print(f"10-year discount factor: {df_10y:.4f}")

Life Insurance
--------------

Whole Life Assurance
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from actuneo.life import LifeAssurance

   la = LifeAssurance(mt, interest_rate=0.05)
   
   # Calculate premium for whole life assurance
   premium = la.whole_life_assurance(30, sum_assured=100000)
   print(f"Annual premium: ${premium:.2f}")

Term Assurance
~~~~~~~~~~~~~~

.. code-block:: python

   # 20-year term assurance
   term_premium = la.term_assurance(30, 20, sum_assured=100000)
   print(f"20-year term premium: ${term_premium:.2f}")

Endowment Assurance
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 25-year endowment
   endowment_premium = la.endowment_assurance(30, 25, sum_assured=100000)
   print(f"25-year endowment premium: ${endowment_premium:.2f}")

Annuities
---------

Life Annuities
~~~~~~~~~~~~~~

.. code-block:: python

   from actuneo.life import Annuities

   ann = Annuities(mt, interest_rate=0.05)
   
   # Whole life annuity
   annuity_value = ann.whole_life_annuity_due(65)
   print(f"Whole life annuity value: {annuity_value:.2f}")
   
   # Temporary annuity
   temp_annuity = ann.temporary_annuity_due(65, 20)
   print(f"20-year temporary annuity: {temp_annuity:.2f}")

Deferred Annuities
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Annuity deferred 10 years
   deferred = ann.deferred_annuity(55, 10)
   print(f"Deferred annuity value: {deferred:.2f}")

Next Steps
----------

* Explore the :doc:`examples` for more advanced use cases
* Check the :doc:`api/mortality` for complete API documentation
* Learn about :doc:`contributing` to the project

