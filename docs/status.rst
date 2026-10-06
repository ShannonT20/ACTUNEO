Status and Limitations
======================

ACTUNEO is an early-stage library. This page states plainly what is
implemented, how each part has been checked, and what is missing, so that
nobody relies on it for more than it can do.

**Read this first**

* The library has not been reviewed by an independent actuary or auditor.
* It has not yet been run on a real insurer's or pension fund's data.
* It was developed with heavy use of an AI coding assistant. The tests were
  written alongside the code by the same process, which is weaker evidence
  than independent testing.
* It is not a substitute for professional judgement and it is not certified
  for statutory, regulatory or audited reporting. Check its output against
  an independent calculation before relying on it.
* Interfaces may change between versions.

How results have been checked
-----------------------------

The level of evidence differs by module. Three levels are used below.

**Published figures**
  The code reproduces numbers printed in a paper, standard-setter example or
  textbook.

**Hand calculation and identities**
  The code agrees with small examples worked by hand and with relationships
  that must hold (for example ``A = 1 - d * ä``, or assets equal liabilities
  plus equity). This shows internal consistency, not that the method is the
  right one for a given purpose.

**Runs without error only**
  The code is exercised by a test but its numerical output has not been
  compared with anything.

What is implemented
-------------------

.. list-table::
   :header-rows: 1
   :widths: 22 43 35

   * - Area
     - What it does
     - How it has been checked
   * - Mortality tables
     - Life table columns from ``qx``, survival probabilities, life
       expectancy, ten Zimbabwe 2023 tables
     - Hand calculation. The shipped tables agree with the columns in their
       source files (life expectancy to within rounding)
   * - Life contingencies
     - Assurances, annuities, net premiums, net premium reserves,
       commutation functions, two-life functions assuming independence
     - Hand calculation and identities. No published table of values has
       been reproduced
   * - Finance
     - Interest rate conversions, annuities certain, loans, yield curve
       interpolation, bootstrapping from par rates, duration and convexity
     - Hand calculation and identities
   * - Run-off triangles
     - Triangle handling, basic and inflation-adjusted chain ladder, average
       cost per claim, Bornhuetter-Ferguson
     - Published figures: the worked examples of a standard run-off
       triangles text, to within its rounding
   * - Mack's model
     - Standard error of the chain-ladder reserve, by origin year and in total
     - Published figures for two standard triangles with
       ``est_sigma="mack"`` (Mack 1993)
   * - Bootstrap
     - Simulated reserve distribution (over-dispersed Poisson)
     - Published scale parameter for the Taylor/Ashe data. The simulated
       prediction error is close to, not equal to, the published analytic
       figure, as expected of a simulation
   * - Other reserving tools
     - Cape Cod, tail factors, cash flows, discounting, quantiles,
       triangles from claim listings, Excel output, plots
     - Hand calculation and identities. Plots: runs without error only
   * - IFRS 17, premium allocation approach
     - Liability for remaining coverage, revenue, acquisition cash flows,
       loss component, liability for incurred claims, reinsurance held
     - Published figures for the liability for remaining coverage (the three
       cases of the IASB example). Everything else: hand calculation and
       identities
   * - IFRS 17 statements and notes
     - Statement of profit or loss and OCI, financial position, cash flows,
       changes in equity, reconciliations, claims development, maturity
       analysis, sensitivities
     - Identities only (the statements balance and tie to the group
       roll-forwards). The layouts follow published illustrative statements
       but no published set of accounts has been reproduced
   * - Hyperinflation helpers
     - Restating amounts by a price index, gain or loss on the net monetary
       position
     - Hand calculation

Known limitations
-----------------

Mortality and life
~~~~~~~~~~~~~~~~~~

* Tables must have single-year ages. Select tables, mortality improvement
  and graduation are not supported.
* The Zimbabwe 2023 data shipped has gaps: there is no female assured-lives
  table, group life and pre-retirement pension tables stop at age 70,
  post-retirement tables start at age 76 (ages 71 to 75 are missing for
  pensions), and the male pre-retirement table starts at age 46. The files
  were supplied to the project; they have not been checked against an
  official publication by anyone other than the author.
* Whole life values on a table that stops before the end of life assume
  everyone dies in the following year, with a warning. For a table that
  stops at age 70 this makes whole life values meaningless.
* Two-life functions assume the lives are independent.
* Premiums and reserves are net (no expenses, lapses, surrender values,
  bonuses or tax). There is no gross premium or profit-testing model.

Finance
~~~~~~~

* Yield curves are simple: interpolation of supplied rates, a basic
  Nelson-Siegel fit and par-rate bootstrapping. There are no stochastic
  interest rate models.
* Bond duration functions discount at an annual effective yield.

Loss reserving
~~~~~~~~~~~~~~

* Triangles with a missing cell inside the observed area are rejected.
* Mack's ``est_sigma="log-linear"`` option and the automatic tail
  uncertainty follow the algorithm in the R ChainLadder source code but
  their output has not been compared numerically with R.
* Tail factors and their uncertainty are judgemental. The automatic
  estimates are starting points.
* Reserve quantiles from Mack's model assume a lognormal or normal shape.
* The bootstrap follows the R approach of resampling all residuals. It has
  no tail factor and does not handle negative development well.
* Munich chain ladder, one-year reserve risk (Merz-Wuthrich) and
  generalised linear models are not implemented.
* The inflation-adjusted method treats payments as made mid-period and
  applies a tail factor to money-terms claims without further inflation.

IFRS 17
~~~~~~~

* **Only the premium allocation approach is implemented.** There is no
  general measurement model, no variable fee approach and no contractual
  service margin. Life business, and short-term business that does not
  qualify for the premium allocation approach, cannot be measured.
* The module does the arithmetic of the standard. It does not decide
  whether a group is eligible, onerous, or correctly grouped: those
  judgements stay with the preparer, and the thresholds in
  ``paa_eligibility`` and ``group_contracts`` are placeholders.
* The loss component is recalculated at each period end from the fulfilment
  cash flows supplied. It is not released by a systematic allocation as
  some entities do.
* The analysis of change in the liability for incurred claims uses
  simplifications: no interest on claims incurred during the period,
  opening rates applied to closing cash flows without rolling the curve
  forward, and all changes in the risk adjustment presented in the
  insurance service result.
* The financing component in the liability for remaining coverage assumes
  premiums are received at the start of each period and revenue is earned
  at its end. Acquisition cash flows are amortised without interest.
* Reinsurance held is a mirror of the insurance model with a loss-recovery
  component that the user supplies. Reinsurer default risk, commissions
  contingent on claims and reinstatement premiums are not modelled.
* The claims development table rebuilds past estimates from today's
  development factors when recorded estimates are not supplied. That is a
  substitute for the history the standard asks for.
* The Excel workbooks are laid out like published insurer accounts but they
  are working papers, not financial statements: there are no accounting
  policies, comparatives or most of the required disclosures.
* The case files (``load_paa_case``) are a convenience format of this
  library, not an industry standard. The example case shipped with it is a
  fictional insurer with invented figures, for demonstration only.
* The financial statements are those of a very simple insurer: cash and
  investments are one line, other expenses and tax are settled in cash in
  the period, there is no deferred tax, no tax on other comprehensive
  income, no fixed assets, and no investment accounting under IFRS 9.
* Transition (IFRS 17 Appendix C), contract modification, business
  combinations, investment components and multi-currency groups are not
  covered.
* Hyperinflation (IAS 29) is not applied to the statements. The helpers in
  ``actuneo.finance`` restate individual amounts only.

Zimbabwe-specific content
~~~~~~~~~~~~~~~~~~~~~~~~~

* The Zimbabwean content is limited to the 2023 mortality tables, the April
  2024 ZWL to ZiG conversion rate, and notes on how local rules affect
  IFRS 17 (for example SI 81 of 2023).
* No IPEC return templates, ZICARP capital calculations, tax rules or local
  price indices are included.
* The notes on local practice come from public sources found online. No
  IPEC guideline on IFRS 17 was located, and the practice of individual
  Zimbabwean insurers has not been verified.

Not started
~~~~~~~~~~~

The ``pensions``, ``macro_africa`` and ``simulation`` modules are empty
placeholders. ``utils`` holds only the Excel report writer.

What would raise confidence
---------------------------

* Running real triangles and policy data through the library and comparing
  with existing valuations.
* Review of the formulas and tests by a second actuary.
* Comparison of the Mack and bootstrap output with the R ChainLadder
  package on the same data.
* Reproducing a published set of IFRS 17 accounts, or a worked example from
  a firm's training material, end to end.

Reports of errors are welcome on the project's issue tracker.
