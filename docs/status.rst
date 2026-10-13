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
       expectancy, select mortality, ten Zimbabwe 2023 tables, AM92,
       and seven South African ASSA tables (SA85-90 in three variants,
       SA Pensioner 2005-10 male and female, SAIML98 and SAIFL98
       annuitant standard tables)
     - Hand calculation. The Zimbabwe tables agree with the columns in
       their source files. AM92 agrees with its graduation formula and
       reproduces published annuity and assurance values. The SA tables
       are loaded directly from the ASSA source files or digitised from
       the published paper, with spot checks against published figures
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
   * - Pensions, defined benefit valuation
     - Liabilities for active members, deferred pensioners and pensioners;
       funding level; contribution rate; sensitivities; projected unit,
       current unit, entry age, attained age and aggregate methods
     - Published figures for the projected unit method: the example
       illustrating paragraph 68 of IAS 19 at three discount rates, as
       printed in an IFRS Interpretations Committee paper. Everything else:
       hand calculation, agreement with the life functions, and identities
   * - Pensions, IAS 19
     - Cost in profit or loss, remeasurements, asset ceiling and the
       reconciliations of the obligation, assets and net liability
     - Hand calculation and identities (the reconciliations tie). The
       interest and service cost of the IAS 19 example are reproduced
   * - Pensions, defined contribution
     - Projection of an account to retirement, replacement ratio, required
       contribution rate; financial position of a fund and the bonus a
       surplus supports
     - Hand calculation (closed-form accumulations)
   * - Pensions, Zimbabwean Act
     - Reduction of minimum reserves on dissolution, leavers' share of
       surplus, IPEC's minimum funding level
     - Hand calculation. The rules were read from the text of the Act; the
       75% funding level comes from IPEC's publications

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
* The South African ASSA tables cover three investigation periods:
  SA85-90 (assured lives, 1985-90, ages 15-104, three risk variants),
  SA Pensioner 2005-10 (ages 50-110, male and female), and SAIML98 /
  SAIFL98 individual annuitant tables 1996-2000 (ages 40-110, digitised
  from Dorrington & Tootla 2007). The SA85-90 is an aggregate table with
  no select period effect; the three variants (standard, light, heavy)
  represent different underwriting risk classes. None of these is suitable
  for other African countries without adjustment. The investigation periods
  are 15-40 years old. The 1991-94, 1995-98, and 1999-2002 ASSA XLS
  files are A/E investigation reports (actual vs. expected ratios against
  SA85-90), not graduated tables, and have not been encoded.
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

Pensions
~~~~~~~~

* **No valuation from this module has been compared with one signed by an
  actuary.** The only published figures reproduced are those of a five-year
  lump sum example with no decrements. Benefits with decrements, pensions,
  spouses and guarantees are checked by hand calculation and identities.
* The module does not choose assumptions. Discount rate, salary increases,
  mortality, rates of leaving and the proportion married are inputs, and
  the answer is very sensitive to them.
* Timing is approximate: active members' ages are whole numbers, leavers go
  at mid-year, and salaries change once a year at the valuation date. A
  scheme with a different salary review date or exact ages will differ
  slightly.
* One accrual rate applies to all service. Schemes with different rates for
  different periods, career average benefits, integration with a state
  pension, or benefits bought by transfers in cannot be described.
* The Zimbabwe 2023 pension tables cannot support a full valuation on their
  own: the pre-retirement tables stop at age 70, the post-retirement tables
  start at 76 and the male pre-retirement table starts at 46. The gaps have
  to be filled by the user.
* A table that stops while mortality is below 1 is closed by assuming that
  nobody survives its last year of age. For a table ending at 100 this
  understates annuities at high ages. A warning is given when more than
  1 life in 1,000 is affected.
* The spouse's pension assumes independent lives, one age difference for
  everyone and a single proportion married applied at the member's death.
  Children's pensions and remarriage are not modelled.
* Deferred pensioners are assumed not to die before retirement unless a
  table is given, and no death benefit in deferment is valued.
* On death in service, the lump sum is spread over service like every other
  benefit. Some actuaries cost it separately as a one-year risk premium.
* A refund of contributions on withdrawal is valued from the balance given
  and future contributions. It is not compared with a minimum benefit.
* The current unit method's normal cost leaves out the cost of salary
  increases on past service, which falls due each year in addition.
* The entry age method rounds service to whole years to find the age at
  entry, and takes the entry age no lower than the start of the service
  table.
* The deficit recovery rate assumes total salaries grow at the general rate
  of salary increase, with no change in the number of members.
* There is no projection of future cash flows, funding level or
  contributions, no asset model, no stochastic projection and no matching of
  assets to liabilities.
* IAS 19: settlements, the detailed split of actuarial gains between
  demographic and financial assumptions, the disclosures of risks and
  maturity, and other long-term employee benefits are not covered. Interest
  on cash flows uses simple weighting by the timing supplied.
* Defined contribution projections are deterministic. They give one outcome
  for one set of assumptions and say nothing about the range of outcomes.
* The defined contribution fund valuation takes the reserves as given. It
  does not model a bonus smoothing account or the vested and non-vested
  bonuses of an insured fund.
* The analysis of surplus works on totals. It does not separate salary,
  mortality, withdrawal and retirement experience.

AI module
~~~~~~~~~

* ``ask`` covers sixteen calculations, each on one set of conventions
  (stated in the answer's assumptions). It is not a general problem solver.
  A question in several parts is split and each part answered on its own:
  a part that depends on the answer to an earlier part cannot be done.
* **Answers in words are written by the language model and can be wrong.**
  The model is given matching passages of ACTUNEO's documentation and told
  to answer as an actuary without inventing figures, but nothing checks what
  it writes. With the smallest model (Qwen3 0.6B) answers about ACTUNEO were
  accurate where a passage held the facts, and answers on general topics
  contained errors (it called IFRS 17 a European Union standard). Larger
  models have not been tried. Every such answer is marked as the model's.
* The passages are found by matching words, not meaning, over the existing
  documentation and docstrings only. There is no body of actuarial reference
  text behind it: general actuarial knowledge comes from the model alone.
* Parts are found from labels such as (a) and (b), separate questions, and
  "and what is". Other ways of writing a multi-part question are read as one.
* Your own data is recognised by its shape: a column named ``qx`` makes a
  mortality table, a single row or column makes cash flows, anything else a
  cumulative triangle. An incremental triangle passed this way is misread.
* IFRS 17 is one calculation: the liability for remaining coverage under the
  premium allocation approach, with the premium received at the start,
  revenue by passage of time, no financing component and no loss component.
* The question is read by pattern rules. They were written against about
  forty test questions and get those right; wording unlike them can be
  misread or not recognised. *How the question was read* shows what was
  taken from the question: check it, because a misread number still gives a
  worked answer that looks correct.
* Each solver's final figure is tested against a direct call to the
  library. The intermediate lines of working are produced by the same code
  and have been read through, not independently checked line by line.
* The working assumes one convention where practice varies (an effective
  annual rate for loans and bonds, half-yearly coupons, monthly pensions in
  advance). The alternative is noted for loans and bonds only.
* Mortality questions use the Zimbabwe 2023 tables unless another basis is
  named. Those tables have gaps, and the answers work around them in ways
  that are judgements of this library: female assured lives are valued on
  the male table, and a pension starting below age 76 is valued on the male
  assured lives rates to age 75 joined to the pensioner rates from 76. The
  assured lives rates are much lighter than pensioner rates at those ages,
  so such a pension is valued higher than on a pensioner table. Each answer
  states the table used.
* With the smallest model (Qwen3 0.6B), two of three unusually worded test
  calculations were still not recognised.
* Apart from ``ask``, no calculation in ACTUNEO uses a language model.
* The memory each model needs is an estimate: the size of the file plus
  about 1 GB. The rule that keeps models above about 5 GB off processors is
  a judgement about speed, not a measurement.
* Hardware detection covers NVIDIA cards and Apple Silicon. AMD and Intel
  graphics are treated as absent, so those machines run on the processor.
  Free memory is measured at the moment of setup: with other applications
  open, a smaller model is recommended than the machine could otherwise run.
* Constraining the output guarantees a valid function name and arguments
  of the right shape, not the right values. On the module's own benchmark
  (twenty requests, twelve tools, one Apple Silicon machine) the smallest
  tier, Qwen3 0.6B, chose the right function 75% of the time and the right
  function and arguments 62% of the time; every miss was a real request
  answered with "none". The earlier Qwen2.5 0.5B scored 65% and 50%.
* Only Qwen3 0.6B has been run. The 1.7B, 4B, 8B, 14B and 30B-A3B tiers and
  the four challengers have been checked for their download details, not
  loaded or benchmarked, so the claim that larger tiers do better is
  untested here. Run ``actuneo.ai.benchmark()`` on your own machine.
* ``shortlist_tools`` ranks tools by shared words, not by meaning. A request
  phrased in words the tool descriptions do not use can drop the right tool
  before the model sees it.
* There is no router that answers common questions without the model, no
  search over ACTUNEO's own functions, and no worked explanation of
  results. ``call_tools`` returns a function name and arguments; calling
  the function is left to the caller.
* The local backend was run on one machine only (Apple Silicon, macOS).
  Windows, Linux and NVIDIA cards are covered by tests with mock hardware.
* Models are fetched by plain HTTP, because Hugging Face's newer transfer
  method stalled on the connection used for the trial.
* A metered connection cannot be detected. The warning is always shown.
* The Gemini, OpenAI, Anthropic and Ollama backends are tested against
  recorded request and reply shapes only. They have not been run against
  the live services, and the default model names will go out of date.
* The catalog's model addresses, sizes and checksums were read from Hugging
  Face in October 2026. Models can be moved or withdrawn.

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
* Pensions: sections 18 and 20 of the Pensions and Provident Funds Act
  [Chapter 24:32] were read from the text of the Act (No. 3 of 2022). The
  regulations made under it were not available. The minimum individual
  reserve and the minimum pension increase follow formulas approved by the
  Commission that are not in the Act, and are not implemented. The 75%
  minimum funding level is quoted from IPEC's trustee training material and
  quarterly reports, not from the instrument that sets it. The guidance on
  converting pension values between currencies is not implemented.

Not started
~~~~~~~~~~~

``utils`` holds only the Excel report writer.

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
* Running the data and basis of a real pension valuation through the
  pensions module and comparing the liabilities, member by member, with the
  actuary's report.

Reports of errors are welcome on the project's issue tracker.
