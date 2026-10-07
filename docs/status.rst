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
       expectancy, select mortality, ten Zimbabwe 2023 tables, AM92
     - Hand calculation. The Zimbabwe tables agree with the columns in
       their source files. AM92 agrees with its graduation formula and
       reproduces published annuity and assurance values
   * - Life contingencies
     - Assurances, annuities, net premiums, net premium reserves,
       commutation functions, two-life functions assuming independence
     - Published figures for single-life values on AM92. Two-life
       functions: hand calculation and identities only
   * - Gross premiums and reserves
     - Premiums and reserves with expenses, with-profits bonuses, loss
       distribution, mortality profit
     - Published figures: worked answers of a standard life contingencies
       text on AM92, to within their rounding
   * - Competing risks
     - Multiple decrement tables, multi-state models with constant
       intensities
     - Published figures for the multiple decrement table. Multi-state
       model: closed-form cases only
   * - Profit testing
     - Conventional and unit-linked profit tests, profit signature, net
       present value, zeroising reserves
     - Published figures for three worked profit tests. Zeroising: hand
       calculation
   * - Finance
     - Interest rate conversions, force of interest, annuities certain,
       cashflows and project appraisal, loan schedules, bonds and equities,
       spot and forward rates, duration, convexity and immunisation
     - Published figures: the worked answers of a standard compound interest
       text, to within their rounding. Yield curve interpolation and the
       Nelson-Siegel fit: hand calculation only
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
   * - One-year reserve risk
     - Standard error of the claims development result
     - Published figures: the Merz and Wuthrich (2008) triangle, as printed
       by the R ChainLadder package
   * - Munich chain ladder
     - Joint projection of paid and incurred triangles
     - The correlation parameters match those published for Quarg and
       Mack's data. The projected ultimates were not checked against a
       published table
   * - Reserving diagnostics
     - Mack's calendar year and correlation tests; intercept, trend and
       outlier checks; back-test of the latest diagonals
     - Mack's two tests: published figures for the RAA triangle, as printed
       by the R ChainLadder package. The other checks: hand calculation and
       behaviour on simulated triangles
   * - GLM reserving
     - Over-dispersed Poisson model with origin or calendar effects and an
       analytic prediction error
     - Published figures for the Taylor/Ashe data (England and Verrall
       2002): scale parameter, reserves and prediction errors by year and
       in total, to within 0.002%. Calendar structure: recovers the
       inflation built into simulated data; no published result reproduced
   * - Machine learning reserving
     - Chain-ladder factors adjusted by a scikit-learn model
     - Identity only (no adjustment gives the chain-ladder) and behaviour
       on simulated triangles. Experimental
   * - Other reserving tools
     - Cape Cod, tail factors, cash flows, discounting, quantiles,
       triangles from claim listings, missing cells, Excel output, plots
     - Hand calculation and identities. Plots: runs without error only
   * - Simulation
     - Aggregate claims; simulated claim payments and triangles; benchmark
       of reserving methods against the simulated outcome
     - Simulated moments against their theoretical values. The claims
       simulator is a simple model of reality, not a calibrated one
   * - Macro data
     - World Bank indicators downloaded on request
     - Tested on a stored response only. The live download was not
       exercised by the tests
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

* Tables must have single-year ages. Mortality improvement and graduation
  are not supported.
* AM92 is a UK table of assured male lives from 1991-94. It is included for
  education and testing, not as a suitable basis for African lives. It
  belongs to the Continuous Mortality Investigation; check their terms
  before redistributing it. Its ultimate rates at ages 17 and 18 come from
  the graduation formula, which leaves ``lx`` about 0.01 in 10,000 away
  from the published column; ratios of ``lx`` are unaffected.
* The pensioner tables used in UK examples (PMA92, PFA92) and ELT15 are
  not included, so questions that need them cannot be reproduced.
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
* ``LifePolicy`` has premiums paid annually in advance only. Premiums and
  benefits paid monthly, deferred annuities with expenses, and benefits
  that return premiums are not handled by it.
* Benefits paid at the moment of death use the approximation of half a
  year's interest in ``LifePolicy`` and the uniform-deaths factor in
  ``SurvivalFunctions``. The two differ slightly.
* With-profits covers reversionary bonuses added at the end of each year.
  Terminal bonus, asset shares and accumulating with-profits are not
  modelled.
* The multi-state model has constant transition intensities, so rates do
  not vary with age or duration of sickness.
* Profit tests take the reserves and decrement rates as inputs. They do
  not derive reserves from a basis, and unit-linked charges are limited to
  allocation, bid/offer spread and an annual management charge.

Finance
~~~~~~~

* Yield curves are simple: interpolation of supplied rates, a basic
  Nelson-Siegel fit and par-rate bootstrapping. There are no stochastic
  interest rate models.
* Bond dates are measured in years from issue. There are no calendar
  dates, day-count conventions, ex-dividend periods or settlement lags, and
  accrued interest is in simple proportion to time.
* Index-linked bonds take the index value that applies to each payment as
  an input. Looking up a lagged index, or projecting an index forward at an
  assumed rate of inflation, is left to the user.
* Uncertain payments are allowed for by a probability on each payment (the
  expected value). There is no model of default, recovery or correlation.
* The theories of the term structure (expectations, liquidity preference,
  market segmentation) are explanations, not calculations, and are not
  represented.
* A yield is found by search. Cashflows that change sign more than once can
  have several yields; the one nearest to the ``guess`` is returned.
* Immunisation is Redington's theory: it protects only against a small,
  uniform change in a flat rate of interest.
* Two styles coexist: the earlier classes (``InterestTheory``,
  ``DurationConvexity``) and the newer functions. The older bond duration
  functions discount at an annual effective yield.

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
* ``GLMReserving`` fits the over-dispersed Poisson model only, with origin
  and development effects or development and calendar effects. There are no
  other distributions, no smoothing of the effects and no user-defined
  model formula. It cannot be fitted when a development period has negative
  total incremental claims. Under the calendar structure the prediction
  error ignores the uncertainty of future inflation, which is usually the
  largest uncertainty of all, and all origin periods are assumed to have
  the same volume unless an exposure is given.
* The diagnostic tests have little power on a triangle of ordinary size and
  they also raise false alarms. On simulated ten-year triangles with nothing
  wrong, individual checks in ``diagnose`` flagged between 1% and 18% of
  triangles and at least one check flagged about a third. A clean report
  does not show that the chain-ladder is appropriate, and a flag is a
  reason to look, not a finding. The intercept, trend and outlier checks
  are this library's own choices of test, not published procedures.
* ``MLChainLadder`` is an experiment. It has no measure of uncertainty, no
  tail factor and no published result to check against. On the Taylor and
  Ashe triangle its default model gives a reserve 18% above the
  chain-ladder, and on the RAA triangle 38% above, without any evidence
  that it is nearer the truth. A more flexible model (gradient boosting)
  more than doubled the Taylor and Ashe reserve. See the benchmark below.
* No reserving method here uses individual claims data, case estimates
  together with payments (other than Munich), or claim counts in a
  machine learning model.
* The one-year reserve risk formula does not allow for a tail factor.
* The Munich chain ladder fills the parameter of the last development step
  by a log-linear trend, and has no standard errors.
* Missing cells are supported by the chain-ladder and Mack only.
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
  2024 ZWL to ZiG conversion rate, minimum capital figures as reported for
  SI 67 of 2025, and notes on how local rules affect IFRS 17 (for example
  SI 81 of 2023).
* The minimum capital figures were taken from published summaries of the
  instrument (a law firm's article), not from the gazetted text.
* No IPEC return templates, ZICARP capital calculations, tax rules or local
  price indices are included. The risk-based solvency rules reported for
  SI 44 of 2026 (best estimate plus risk margin, a solvency capital
  requirement at 99.5% over one year, a minimum capital requirement, own
  funds in tiers, an own risk and solvency assessment) are not implemented
  in any form: the standard formula and its calibration were not available
  to build from.
* The notes on local practice come from public sources found online. No
  IPEC guideline on IFRS 17 was located, and the practice of individual
  Zimbabwean insurers has not been verified.

Not started
~~~~~~~~~~~

The ``pensions`` module is an empty placeholder. ``utils`` holds only the
Excel report writer.

Simulation and macro data
~~~~~~~~~~~~~~~~~~~~~~~~~

* The claims simulator pays every claim over the development years in
  fixed proportions, or in proportions that vary at random from claim to
  claim, with no reporting delay, case estimates or reopened claims. It is
  a test bed for reserving methods, not a model of a real portfolio.
* ``benchmark_reserving`` ranks methods on the simulated portfolio only. In
  the runs made while building it (100 ten-year triangles of about 400
  claims a year), the error of the total reserve was:

  .. list-table::
     :header-rows: 1
     :widths: 34 22 22 22

     * - Method (bias / root mean squared error)
       - Nothing wrong
       - Faster settlement from year 7
       - Inflation 3% to 40% in the last two years
     * - Chain-ladder
       - +1% / 6%
       - +183% / 184%
       - -44% / 44%
     * - Chain-ladder, latest three years
       - 0% / 6%
       - +138% / 139%
       - -26% / 26%
     * - GLM, calendar effects, all years
       - +1% / 7%
       - +141% / 142%
       - -41% / 42%
     * - GLM, calendar effects, last two years
       - +1% / 11%
       - +100% / 102%
       - +2% / 13%
     * - ML chain-ladder (default)
       - +1% / 10%
       - -34% / 37%
       - -1% / 10%

  No method was best throughout: the methods that cope with a change cost
  accuracy when nothing has changed, and none handled faster settlement
  well. These figures come from one simple simulator with large, abrupt
  changes and should not be read as the performance to expect on real
  business.
* Macro data comes only from the World Bank's annual indicators. Recent
  years are often missing, and for countries with very high inflation or
  several exchange rates the official series may not reflect the rates
  actually experienced. No national statistics office or central bank
  source is connected.

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
