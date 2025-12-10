Installation
============

Requirements
------------

ACTUNEO requires Python 3.8 or higher and has the following dependencies:

* numpy >= 1.21.0
* pandas >= 1.3.0
* scipy >= 1.7.0
* matplotlib >= 3.4.0

Installing from PyPI
--------------------

The easiest way to install ACTUNEO is from PyPI:

.. code-block:: bash

   pip install actuneo

Installing from Source
----------------------

To install from source:

.. code-block:: bash

   git clone https://github.com/ShannonT20/ACTUNEO.git
   cd ACTUNEO
   pip install -e .

Development Installation
------------------------

For development, install with additional dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

This includes testing and documentation tools:

* pytest >= 6.2.0
* pytest-cov >= 2.12.0
* black >= 21.0.0
* flake8 >= 3.9.0
* sphinx >= 4.0.0
* sphinx-rtd-theme >= 1.0.0

Optional Dependencies
---------------------

Machine Learning
~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install actuneo[ml]

Includes:

* scikit-learn >= 1.0.0
* statsmodels >= 0.12.0

Visualization
~~~~~~~~~~~~~

.. code-block:: bash

   pip install actuneo[viz]

Includes:

* plotly >= 5.0.0
* seaborn >= 0.11.0

Verifying Installation
----------------------

To verify the installation:

.. code-block:: python

   import actuneo
   print(actuneo.__version__)
   print(actuneo.__author__)

