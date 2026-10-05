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

MackChainLadder
---------------

.. autoclass:: actuneo.loss_reserving.MackChainLadder
   :members:
   :show-inheritance:
   :special-members: __init__

Example Triangles
-----------------

.. autofunction:: actuneo.loss_reserving.load_raa

.. autofunction:: actuneo.loss_reserving.load_genins

Planned Features
----------------

* Tail factor estimation and its standard error in Mack's model
* Bornhuetter-Ferguson and Cape Cod methods
* Bootstrap and other stochastic reserving models
