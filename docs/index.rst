ACTUNEO: African Actuarial Python Library
==========================================

**ACTUNEO** is an open-source, community-driven actuarial Python library that empowers African and Zimbabwean actuaries to perform core actuarial, financial, and statistical computations with ease.

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://badge.fury.io/py/actuneo.svg
   :target: https://badge.fury.io/py/actuneo
   :alt: PyPI version

Vision and Objective
--------------------

The goal is to build a localized yet globally compatible toolkit that supports insurance, pensions, and investment analytics, while integrating with modern data science tools.

The library aims to:

* Provide accessible actuarial computation tools for African practitioners
* Bridge the gap between global methodologies and African market realities
* Foster community-driven development and knowledge sharing
* Support both traditional actuarial work and modern data science approaches

Key Features
------------

African Market Focus
~~~~~~~~~~~~~~~~~~~~

* **Localized Data Support**: Mortality, inflation, and interest rate tables calibrated to African markets
* **Regulatory Alignment**: Built-in parameters for IPEC Zimbabwe, PASA, SAM reporting
* **Currency Handling**: Multi-currency modeling (USD, ZWL, Rand, etc.) with inflation adjustment
* **Socioeconomic Context**: Assumptions for informal sector, microinsurance, and low-coverage environments

Technical Advantages
~~~~~~~~~~~~~~~~~~~~

* **Integration**: Works seamlessly with pandas, numpy, scikit-learn, plotly
* **Open Development**: Community-driven through GitHub
* **Modern Approach**: Combines traditional actuarial methods with data science

Installation
------------

From PyPI:

.. code-block:: bash

   pip install actuneo

From source:

.. code-block:: bash

   git clone https://github.com/ShannonT20/ACTUNEO.git
   cd actuneo
   pip install -e .

Quick Start
-----------

Mortality Analysis
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from actuneo.mortality import MortalityTable, SurvivalFunctions

   # Create a mortality table
   ages = np.arange(20, 101)
   qx = 0.001 * (ages - 20) / 80  # Simplified mortality rates
   mt = MortalityTable(ages, qx, name="Example Table")

   # Calculate life expectancy
   le = mt.life_expectancy(30)
   print(f"Life expectancy at age 30: {le:.1f} years")

   # Survival functions
   sf = SurvivalFunctions(mt)
   survival_prob = sf.npx(30, 20)  # 20-year survival probability
   print(f"20-year survival probability: {survival_prob:.3f}")

Financial Calculations
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from actuneo.finance import InterestTheory, YieldCurve

   # Interest theory
   it = InterestTheory(interest_rate=0.05)
   fv = it.future_value(1000, 10)  # Future value of $1000 in 10 years
   print(f"Future value: ${fv:.2f}")

   # Yield curve
   maturities = [1, 2, 5, 10, 30]
   yields = [0.03, 0.035, 0.045, 0.055, 0.065]
   yc = YieldCurve(maturities, yields)
   yield_15y = yc.get_yield(15)
   print(f"15-year yield: {yield_15y:.3%}")

Life Insurance Calculations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from actuneo.life import LifeAssurance

   # Life assurance calculations
   la = LifeAssurance(mt, interest_rate=0.05)
   premium = la.whole_life_assurance(30, sum_assured=100000)
   print(f"Whole life premium at age 30: ${premium:.2f}")

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/mortality
   api/finance
   api/life
   api/pensions
   api/ifrs17
   api/loss_reserving
   api/macro_africa
   api/simulation
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

