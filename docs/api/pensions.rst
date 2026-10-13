Pensions Module
===============

Valuation of defined benefit schemes, IAS 19 accounting, defined
contribution projections and calculations from Zimbabwe's Pensions and
Provident Funds Act. See :doc:`../pensions` for a worked valuation and the
method.

.. automodule:: actuneo.pensions
   :no-members:

Defined benefit valuation
-------------------------

.. autoclass:: actuneo.pensions.DBValuation
   :members:

.. automodule:: actuneo.pensions.valuation
   :no-members:

Scheme rules and assumptions
----------------------------

.. autoclass:: actuneo.pensions.DefinedBenefitScheme
   :members:

.. autoclass:: actuneo.pensions.ValuationBasis
   :members:

.. autoclass:: actuneo.pensions.ServiceTable
   :members:

.. autoclass:: actuneo.pensions.SalaryScale
   :members:

Annuity factors
---------------

.. autofunction:: actuneo.pensions.pension_annuity

.. autofunction:: actuneo.pensions.survival_probabilities

IAS 19
------

.. autoclass:: actuneo.pensions.IAS19Period
   :members:

Defined contribution
--------------------

.. autoclass:: actuneo.pensions.DCProjection
   :members:

.. autoclass:: actuneo.pensions.DCFundValuation
   :members:

Analysis of surplus
-------------------

.. autofunction:: actuneo.pensions.analysis_of_surplus

Zimbabwe: Pensions and Provident Funds Act
------------------------------------------

.. automodule:: actuneo.pensions.zimbabwe
   :members:
