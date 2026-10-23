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

MunichChainLadder
-----------------

.. automodule:: actuneo.loss_reserving.munich

.. autoclass:: actuneo.loss_reserving.MunichChainLadder
   :members:
   :special-members: __init__

One-year reserve risk is ``MackChainLadder.cdr()``. Triangles with missing
cells are created with ``Triangle(..., allow_missing=True)``.

BootChainLadder
---------------

.. autoclass:: actuneo.loss_reserving.BootChainLadder
   :members:
   :special-members: __init__

Diagnostics: when the chain-ladder fails
----------------------------------------

.. automodule:: actuneo.loss_reserving.diagnostics

.. autofunction:: actuneo.loss_reserving.diagnose

.. autofunction:: actuneo.loss_reserving.calendar_year_effect_test

.. autofunction:: actuneo.loss_reserving.development_factor_correlation_test

.. autofunction:: actuneo.loss_reserving.intercept_test

.. autofunction:: actuneo.loss_reserving.link_ratio_trend_test

.. autofunction:: actuneo.loss_reserving.link_ratio_outliers

.. autofunction:: actuneo.loss_reserving.backtest

GLMReserving
------------

.. automodule:: actuneo.loss_reserving.glm

.. autoclass:: actuneo.loss_reserving.GLMReserving
   :members:
   :special-members: __init__

MLChainLadder
-------------

.. automodule:: actuneo.loss_reserving.ml

.. autoclass:: actuneo.loss_reserving.MLChainLadder
   :members:
   :special-members: __init__

Reports and Files
-----------------

``Triangle.from_excel``, ``Triangle.from_csv``, ``Triangle.to_excel`` and
``Triangle.to_csv`` read and write triangles as spreadsheets.

``from_csv`` and ``from_excel`` read the layouts insurers' systems and
spreadsheets usually produce:

* the triangle as a table (origin periods down, development periods across),
  or one row per cell or transaction (long format, with columns such as
  ``accident_year``, ``dev`` and ``paid`` recognised by name);
* calendar (payment) years instead of development periods, in either layout;
* titles and notes above the table, Total rows and columns, amounts stored
  as text (``1,234``, ``(56)``), dashes for empty cells, and semicolon
  separated files with decimal commas.

In a cumulative triangle, zeros after an origin period's latest amount are
read as not yet observed, with a warning.

.. code-block:: python

   paid = Triangle.from_csv("claims_listing.csv")      # long format, found by column names
   paid = Triangle.from_csv("by_year.csv", layout="long", origin="AY",
                            calendar="Payment year", value="Amount")
   paid = Triangle.from_excel("board_pack.xlsx", sheet_name="Paid")

.. autofunction:: actuneo.loss_reserving.compare_methods

.. autofunction:: actuneo.loss_reserving.export_reserving_report

Tail Factors
------------

.. autofunction:: actuneo.loss_reserving.estimate_tail_factor

Example Triangles
-----------------

.. autofunction:: actuneo.loss_reserving.load_raa

.. autofunction:: actuneo.loss_reserving.load_genins

.. autofunction:: actuneo.loss_reserving.load_mw2008

.. autofunction:: actuneo.loss_reserving.load_mcl

Planned Features
----------------

* Generalised linear models for reserving
* Bootstrap with a tail factor
