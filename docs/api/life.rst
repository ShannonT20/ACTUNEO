Life Module
===========

The life module provides tools for life insurance calculations including assurance, annuities, and reserves.

.. automodule:: actuneo.life
   :members:
   :undoc-members:
   :show-inheritance:

LifeAssurance
-------------

.. autoclass:: actuneo.life.LifeAssurance
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Annuities
---------

.. autoclass:: actuneo.life.Annuities
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Reserves
--------

.. autoclass:: actuneo.life.Reserves
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__


Policies with Expenses
----------------------

.. automodule:: actuneo.life.policy

.. autoclass:: actuneo.life.LifePolicy
   :members:
   :special-members: __init__

.. autoclass:: actuneo.life.Expenses
   :special-members: __init__

.. autofunction:: actuneo.life.simple_bonus_benefits

.. autofunction:: actuneo.life.compound_bonus_benefits

Competing Risks
---------------

.. automodule:: actuneo.life.decrements

.. autoclass:: actuneo.life.MultipleDecrementTable
   :members:
   :special-members: __init__

.. autoclass:: actuneo.life.MultiStateModel
   :members:
   :special-members: __init__

.. autofunction:: actuneo.life.dependent_from_independent

.. autofunction:: actuneo.life.independent_from_dependent

Profit Testing and Unit-Linked Policies
---------------------------------------

.. automodule:: actuneo.life.profit_testing

.. autoclass:: actuneo.life.ProfitTest
   :members:
   :special-members: __init__

.. autoclass:: actuneo.life.ProfitSignature
   :members:
   :special-members: __init__

.. autoclass:: actuneo.life.UnitLinkedPolicy
   :members:
   :special-members: __init__

.. autofunction:: actuneo.life.zeroise_negative_cashflows
