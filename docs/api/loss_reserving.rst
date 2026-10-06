Loss Reserving Module
=====================

Claims development triangles and chain-ladder reserving for general insurance.

.. automodule:: actuneo.loss_reserving

Triangle
--------

.. autoclass:: actuneo.loss_reserving.Triangle
   :members:
   :special-members: __init__

ChainLadder
-----------

.. autoclass:: actuneo.loss_reserving.ChainLadder
   :members:
   :special-members: __init__

InflationAdjustedChainLadder
----------------------------

.. autoclass:: actuneo.loss_reserving.InflationAdjustedChainLadder
   :members:
   :show-inheritance:
   :special-members: __init__

AverageCostPerClaim
-------------------

.. autoclass:: actuneo.loss_reserving.AverageCostPerClaim
   :members:
   :special-members: __init__

.. autofunction:: actuneo.loss_reserving.grossing_up

BornhuetterFerguson
-------------------

.. autoclass:: actuneo.loss_reserving.BornhuetterFerguson
   :members:
   :show-inheritance:
   :special-members: __init__

CapeCod
-------

.. autoclass:: actuneo.loss_reserving.CapeCod
   :members:
   :show-inheritance:
   :special-members: __init__

MackChainLadder
---------------

.. autoclass:: actuneo.loss_reserving.MackChainLadder
   :members:
   :show-inheritance:
   :special-members: __init__

BootChainLadder
---------------

.. autoclass:: actuneo.loss_reserving.BootChainLadder
   :members:
   :special-members: __init__

Reports and Files
-----------------

``Triangle.from_excel``, ``Triangle.from_csv``, ``Triangle.to_excel`` and
``Triangle.to_csv`` read and write triangles as spreadsheets.

.. autofunction:: actuneo.loss_reserving.compare_methods

.. autofunction:: actuneo.loss_reserving.export_reserving_report

Tail Factors
------------

.. autofunction:: actuneo.loss_reserving.estimate_tail_factor

Example Triangles
-----------------

.. autofunction:: actuneo.loss_reserving.load_raa

.. autofunction:: actuneo.loss_reserving.load_genins

Planned Features
----------------

* Munich chain-ladder
* One-year reserve risk (Merz-Wuthrich)
* Triangles with missing cells
