Contributing to ACTUNEO
=======================

We welcome contributions from the actuarial community! This guide will help you get started.

Getting Started
---------------

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature
4. Make your changes
5. Run tests
6. Submit a pull request

Development Setup
-----------------

.. code-block:: bash

   # Clone your fork
   git clone https://github.com/YOUR_USERNAME/ACTUNEO.git
   cd ACTUNEO
   
   # Create a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install development dependencies
   pip install -e ".[dev]"

Code Style
----------

We follow PEP 8 style guidelines. Format your code using:

.. code-block:: bash

   black actuneo/
   flake8 actuneo/

Running Tests
-------------

Before submitting a pull request, ensure all tests pass:

.. code-block:: bash

   pytest
   
   # With coverage
   pytest --cov=actuneo --cov-report=html

Writing Tests
-------------

All new features should include tests. Place tests in the ``tests/`` directory:

.. code-block:: python

   # tests/test_new_feature.py
   import pytest
   from actuneo.module import NewFeature
   
   def test_new_feature():
       nf = NewFeature()
       assert nf.method() == expected_result

Documentation
-------------

Update documentation for any API changes:

1. Update docstrings in the code
2. Update relevant ``.rst`` files in ``docs/``
3. Build documentation locally to verify:

.. code-block:: bash

   cd docs
   make html
   # Open _build/html/index.html in your browser

Areas for Contribution
----------------------

African Mortality Tables
~~~~~~~~~~~~~~~~~~~~~~~~

We need contributions of:

* Localized mortality data
* Country-specific mortality tables
* Mortality improvement models for African populations

Regulatory Frameworks
~~~~~~~~~~~~~~~~~~~~~

Help implement:

* IPEC Zimbabwe reporting requirements
* PASA compliance tools
* SAM (Statutory Actuarial Memorandum) templates
* Other African regulatory frameworks

Pension Systems
~~~~~~~~~~~~~~~

Develop models for:

* Defined benefit schemes
* Defined contribution schemes
* Hybrid pension plans
* Contribution schedules

IFRS 17
~~~~~~~

Implement:

* GMM (General Measurement Model)
* VFA (Variable Fee Approach)
* PAA (Premium Allocation Approach)
* CSM (Contractual Service Margin) calculations

Loss Reserving
~~~~~~~~~~~~~~

Add methods for:

* Chain-ladder technique
* Bornhuetter-Ferguson method
* Stochastic reserving models
* Loss development triangles

Documentation
~~~~~~~~~~~~~

* Improve API documentation
* Add tutorials and examples
* Create video tutorials
* Translate documentation

Testing
~~~~~~~

* Expand test coverage
* Add integration tests
* Create benchmark tests
* Test on different platforms

Pull Request Process
--------------------

1. Update the README.md with details of changes if applicable
2. Update documentation
3. Ensure tests pass
4. Update the CHANGELOG.md
5. The PR will be merged once reviewed by maintainers

Community Guidelines
--------------------

* Be respectful and inclusive
* Provide constructive feedback
* Help others learn and grow
* Follow the Code of Conduct

Contact
-------

* GitHub Issues: https://github.com/ShannonT20/ACTUNEO/issues
* Email: shannonsikadi@gmail.com
* LinkedIn: https://www.linkedin.com/in/shannon-sikadi-9370b3196/

Thank you for contributing to ACTUNEO!

