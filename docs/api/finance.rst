Finance Module
==============

Compound interest and its applications: the financial mathematics that the
rest of the library builds on.

.. list-table:: Where to find things
   :header-rows: 1
   :widths: 38 62

   * - Topic
     - Use
   * - Effective, nominal and simple rates; discount; force of interest
     - ``InterestRate``, ``simple_accumulation``, ``simple_discount_value``
   * - Force of interest varying with time; continuous payment streams
     - ``ForceOfInterest``
   * - Real and money rates, inflation indices
     - ``real_rate``, ``money_rate``, ``inflation_from_index``
   * - Level annuities: in arrears, in advance, p-thly, continuous, deferred,
       perpetual
     - ``annuity``, ``accumulated_annuity``, ``perpetuity``
   * - Increasing and decreasing annuities (also deferred), compound growth,
       stepped payments
     - ``increasing_annuity``, ``decreasing_annuity``,
       ``continuously_increasing_annuity``, ``geometric_annuity``,
       ``stepped_annuity``
   * - Present and accumulated values of any cashflows; equations of value;
       yields
     - ``Cashflows``, ``present_value``, ``accumulated_value``,
       ``internal_rate_of_return``
   * - Uncertain payments (probability of a cashflow); whether a yield is
       unique
     - ``Cashflows(..., probabilities=...)``, ``Cashflows.has_unique_yield``
   * - Project appraisal: net present value, internal rate of return,
       payback periods, accumulated profit, comparing two projects
     - ``Cashflows``, ``crossover_rate``
   * - Cashflow models of standard instruments
     - ``instruments`` (zero-coupon bond, fixed-interest and index-linked
       securities, annuity-certain, interest-only and repayment loans, equity)
   * - Loan schedules, interest and capital, loan outstanding, APR
     - ``Loan``, ``loan_schedule``, ``annual_percentage_rate``, ``flat_rate``
   * - Bond prices and yields with income tax and capital gains tax; purchase
       between coupon dates; tax paid later; optional redemption dates and
       price bounds
     - ``Bond``, ``bond_price_with_optional_redemption``
   * - Index-linked bonds, equities, property, real returns
     - ``index_linked_cashflows``, ``equity_price``, ``equity_yield``,
       ``present_value_of_dividends``, ``property_value``, ``real_yield``
   * - Spot rates, forward rates, par yields (discrete and continuous time)
     - ``forward_rate``, ``spot_rates_from_forwards``, ``par_yield``,
       ``spot_rates_from_bonds``, ``continuous_spot_rate``,
       ``continuous_forward_rate``, ``instantaneous_forward_rate``,
       ``YieldCurve``
   * - Duration, convexity, sensitivity and immunisation
     - ``discounted_mean_term``, ``volatility``, ``convexity``,
       ``estimated_value_change``, ``immunisation_check``,
       ``immunising_amounts``
   * - Hyperinflation restatement
     - ``restate``, ``net_monetary_gain_or_loss``

Conventions: rates are effective annual rates written as decimals unless a
function says otherwise, times are in years, and annuity functions value 1
per year.

Interest Rates
--------------

.. automodule:: actuneo.finance.rates

.. autoclass:: actuneo.finance.InterestRate
   :members:
   :special-members: __init__

.. autoclass:: actuneo.finance.ForceOfInterest
   :members:
   :special-members: __init__

.. autofunction:: actuneo.finance.simple_accumulation

.. autofunction:: actuneo.finance.simple_discount_value

.. autofunction:: actuneo.finance.real_rate

.. autofunction:: actuneo.finance.money_rate

.. autofunction:: actuneo.finance.inflation_from_index

Annuities Certain
-----------------

.. automodule:: actuneo.finance.annuities

.. autofunction:: actuneo.finance.annuity

.. autofunction:: actuneo.finance.accumulated_annuity

.. autofunction:: actuneo.finance.perpetuity

.. autofunction:: actuneo.finance.increasing_annuity

.. autofunction:: actuneo.finance.continuously_increasing_annuity

.. autofunction:: actuneo.finance.decreasing_annuity

.. autofunction:: actuneo.finance.accumulated_increasing_annuity

.. autofunction:: actuneo.finance.geometric_annuity

.. autofunction:: actuneo.finance.stepped_annuity

Cashflows and Project Appraisal
-------------------------------

.. automodule:: actuneo.finance.cashflows

.. autoclass:: actuneo.finance.Cashflows
   :members:
   :special-members: __init__

.. autofunction:: actuneo.finance.present_value

.. autofunction:: actuneo.finance.accumulated_value

.. autofunction:: actuneo.finance.internal_rate_of_return

.. autofunction:: actuneo.finance.crossover_rate

Instrument Cashflow Models
--------------------------

.. automodule:: actuneo.finance.instruments
   :members:

Loans
-----

.. automodule:: actuneo.finance.loans

.. autoclass:: actuneo.finance.Loan
   :members:
   :special-members: __init__

.. autofunction:: actuneo.finance.loan_schedule

.. autofunction:: actuneo.finance.annual_percentage_rate

.. autofunction:: actuneo.finance.flat_rate

Bonds, Equities and Real Returns
--------------------------------

.. automodule:: actuneo.finance.securities

.. autoclass:: actuneo.finance.Bond
   :members:
   :special-members: __init__

.. autofunction:: actuneo.finance.bond_price_with_optional_redemption

.. autofunction:: actuneo.finance.equity_price

.. autofunction:: actuneo.finance.equity_yield

.. autofunction:: actuneo.finance.present_value_of_dividends

.. autofunction:: actuneo.finance.real_yield

.. autofunction:: actuneo.finance.index_linked_cashflows

.. autofunction:: actuneo.finance.property_value

Term Structure, Duration and Immunisation
-----------------------------------------

.. automodule:: actuneo.finance.term_structure

.. autofunction:: actuneo.finance.forward_rate

.. autofunction:: actuneo.finance.spot_rates_from_forwards

.. autofunction:: actuneo.finance.forwards_from_spot_rates

.. autofunction:: actuneo.finance.par_yield

.. autofunction:: actuneo.finance.spot_rates_from_bonds

.. autofunction:: actuneo.finance.discounted_mean_term

.. autofunction:: actuneo.finance.volatility

.. autofunction:: actuneo.finance.convexity

.. autofunction:: actuneo.finance.immunisation_check

.. autofunction:: actuneo.finance.immunising_amounts

.. autofunction:: actuneo.finance.estimated_value_change

.. autofunction:: actuneo.finance.continuous_spot_rate

.. autofunction:: actuneo.finance.continuous_forward_rate

.. autofunction:: actuneo.finance.continuous_forward_from_spots

.. autofunction:: actuneo.finance.instantaneous_forward_rate

.. autofunction:: actuneo.finance.effective_to_continuous

.. autofunction:: actuneo.finance.continuous_to_effective

Earlier Classes
---------------

These classes predate the functions above and remain available.

.. autoclass:: actuneo.finance.InterestTheory
   :members:
   :special-members: __init__

.. autoclass:: actuneo.finance.YieldCurve
   :members:
   :special-members: __init__

.. autoclass:: actuneo.finance.DurationConvexity
   :members:
   :special-members: __init__

Hyperinflation
--------------

.. automodule:: actuneo.finance.hyperinflation

.. autofunction:: actuneo.finance.restate

.. autofunction:: actuneo.finance.cumulative_inflation

.. autofunction:: actuneo.finance.exceeds_hyperinflation_indicator

.. autofunction:: actuneo.finance.net_monetary_gain_or_loss
