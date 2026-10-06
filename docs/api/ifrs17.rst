IFRS 17 Module
==============

Measurement and presentation of insurance contracts under IFRS 17. The
premium allocation approach (PAA) used for short-term business is
implemented; the general measurement model and the variable fee approach are
planned.

.. automodule:: actuneo.ifrs17

How the pieces fit
------------------

Under the PAA the carrying amount of a group of contracts has two parts.

**Liability for remaining coverage (LRC)**: cover not yet provided. It starts
from the premiums received and is released to insurance revenue as cover is
given, much like an unearned premium reserve net of deferred acquisition
costs. If the cover still to be given is expected to cost more than the
liability, the group is onerous and a loss component is added.

**Liability for incurred claims (LIC)**: claims that have already happened,
reported or not. It is the present value of the expected payments plus a risk
adjustment for non-financial risk. The expected payments come straight from
the reserving models in :mod:`actuneo.loss_reserving`.

``PAAGroup`` rolls both forward period by period, ``PAAReinsuranceHeld`` does
the same for reinsurance, and ``IFRS17Statements`` combines them into the
statement of profit or loss, the statement of financial position and the
statement of cash flows.

From IFRS 4 to IFRS 17
----------------------

For readers used to the IFRS 4 accounts of a short-term insurer.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - IFRS 4
     - IFRS 17
   * - Unearned premium reserve
     - Liability for remaining coverage (LRC)
   * - Deferred acquisition costs (an asset)
     - Not shown separately: netted in the LRC
   * - Premiums receivable (an asset)
     - Not shown separately: netted in the LRC
   * - Outstanding claims and IBNR
     - Liability for incurred claims (LIC): discounted best estimate plus an
       explicit risk adjustment
   * - Gross written premium on the income statement
     - Insurance revenue, recognised as cover is provided. Premiums written
       or received appear only in the notes
   * - Claims and change in insurance liabilities
     - Insurance service expenses: incurred claims, amortisation of
       acquisition cash flows, adjustments to the LIC, losses on onerous
       contracts
   * - Premiums ceded and claims ceded as separate lines
     - One line: net expenses from reinsurance contracts held
   * - Investment income
     - Net financial result: investment return less insurance finance
       expenses (unwinding of discount and changes in discount rates)

``supplementary_position()`` on a group or on the statements shows the
IFRS 17 carrying amount analysed back into unearned premium, premiums
receivable, deferred acquisition costs and outstanding claims.

Presentation choices
--------------------

* **Acquisition cash flows** may be expensed when incurred if the coverage
  period is one year or less (``expense_acquisition_cash_flows=True``).
  Otherwise they are deferred in the LRC and amortised with revenue.
* **Insurance finance income and expenses** may be presented wholly in profit
  or loss, or disaggregated with part in other comprehensive income
  (``finance_expenses_in_oci``). For incurred claims under the PAA, the part
  in OCI is the effect of changes in discount rates since the claims were
  incurred.
* **Insurance service expenses** may be shown as one line or by component
  (``profit_or_loss(detailed=True)``).
* **Portfolios** in an asset position are presented separately from those in
  a liability position; groups within a portfolio are netted.

Practice varies. KPMG's review of insurers' 2024 annual reports found that
19 of 47 insurers using the PAA expensed some acquisition cash flows when
incurred, and that a little over half presented part of insurance finance
income and expenses in other comprehensive income.

Key ratios
----------

``key_ratios()`` gives the combined ratio as insurance service expenses
divided by insurance revenue, the base calculation most non-life insurers
now use, with two common variants: net of reinsurance, and including
expenses not directly attributable to insurance contracts. The same review
found that insurers adjust this base in different ways and some still use a
premium measure as the denominator, so a ratio should always be quoted with
its definition.

IFRS 17 in Zimbabwe
-------------------

Points that affect how the standard is applied locally.

* **Effective date.** IFRS 17 applied from 1 January 2023. The Insurance and
  Pensions Commission (IPEC) allowed regulatory returns for the first two
  quarters of 2023 to be prepared under IFRS 4.
* **No premium, no cover.** Statutory Instrument 81 of 2023 makes receipt of
  the premium a condition for a valid short-term insurance contract, with no
  cover unless the premium is paid in advance. Premiums are therefore
  received at or before the start of cover, which is the first case of the
  IASB's PAA example: the liability for remaining coverage is the unearned
  premium less unamortised acquisition cash flows, and groups are not
  expected to be in an asset position for unpaid premiums.
* **Discount rates.** There is no observable local yield curve for short or
  long terms. A discount rate has to be built up by the entity, so the
  functions here accept either a single rate or a curve supplied by the user.
  Where claims are paid within a year of being incurred, IFRS 17.59(b) allows
  the liability for incurred claims to be left undiscounted.
* **Currency and inflation.** Zimbabwe Gold (ZiG) replaced the Zimbabwe
  dollar in April 2024 and many insurers have the US dollar as functional
  currency. First Mutual Holdings, for example, changed its functional and
  presentation currency to the US dollar from 1 January 2024. Hyperinflation accounting under IAS 29 and translation under
  IAS 21 are not covered by this module; amounts are in whatever single
  currency the inputs are given in.
* **Data.** Claims triangles for IFRS 17 need reliable historical data by
  accident period. ``Triangle.from_transactions`` builds them from a claims
  listing.

Sources: KPMG, "Real-time IFRS 17: Insurers' 2024 annual financial
statements" (2025); IPEC and Insurance Council of Zimbabwe notices on SI 81 of 2023;
Claxon Actuaries, "Implementing IFRS 17 in Zimbabwe: Challenges" and
"Presentation of Financial Accounts under IFRS 17: A Non-Life Insurance
Perspective" (June 2020); First Mutual Holdings Limited, 2024 Annual Report.

Premium Allocation Approach
---------------------------

.. autoclass:: actuneo.ifrs17.PAAGroup
   :members:
   :special-members: __init__

.. autoclass:: actuneo.ifrs17.PAAReinsuranceHeld
   :members:
   :special-members: __init__

.. autofunction:: actuneo.ifrs17.paa_eligibility

Liability for Incurred Claims
-----------------------------

.. autoclass:: actuneo.ifrs17.LiabilityForIncurredClaims
   :members:
   :special-members: __init__

.. autofunction:: actuneo.ifrs17.lic_analysis_of_change

Risk Adjustment
---------------

.. autofunction:: actuneo.ifrs17.risk_adjustment_confidence_level

.. autofunction:: actuneo.ifrs17.risk_adjustment_cost_of_capital

.. autofunction:: actuneo.ifrs17.implied_confidence_level

Financial Statements
--------------------

.. autoclass:: actuneo.ifrs17.IFRS17Statements
   :members:
   :special-members: __init__

Case Files
----------

.. automodule:: actuneo.ifrs17.case

.. autofunction:: actuneo.ifrs17.load_paa_case

.. autoclass:: actuneo.ifrs17.PAACase
   :members:

.. autofunction:: actuneo.ifrs17.example_case_path

.. autofunction:: actuneo.ifrs17.copy_example_case

Level of Aggregation
--------------------

.. automodule:: actuneo.ifrs17.aggregation

.. autofunction:: actuneo.ifrs17.group_contracts

Disclosures
-----------

.. autofunction:: actuneo.ifrs17.claims_development_table

.. autofunction:: actuneo.ifrs17.maturity_analysis

.. autofunction:: actuneo.ifrs17.lic_sensitivity

Earning Premiums
----------------

.. autofunction:: actuneo.ifrs17.earned_fraction

.. autofunction:: actuneo.ifrs17.unearned_premium

.. autofunction:: actuneo.ifrs17.earned_premium_by_period

References
----------

* IFRS 17 *Insurance Contracts*, paragraphs 53-59 (premium allocation
  approach), 69-70A (reinsurance contracts held), 78-92 (presentation) and
  97-109 (disclosure).
* IASB, *Premium allocation approach example*, Transition Resource Group for
  IFRS 17, May 2018.
* KPMG, *Illustrative disclosures for insurers: Guide to annual financial
  statements, IFRS 17 and IFRS 9*, October 2024.

Planned Features
----------------

* General measurement model with the contractual service margin
* Variable fee approach
* Hyperinflation restatement of the statements (IAS 29) and currency translation
* Transition approaches

See :doc:`../status` for the limitations of what is implemented.
