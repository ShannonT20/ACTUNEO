Changelog
=========

All notable changes to ACTUNEO will be documented here.

Version 0.4.0 (2026-10-07)
--------------------------

Beta-quality release (first published as 0.4.0b1). See the Status and Limitations page for what has and has not been
checked.

Added
~~~~~

* ``actuneo.ifrs17``, premium allocation approach:

  * ``PAAGroup``: liability for remaining coverage, insurance revenue, deferral or expensing
    of acquisition cash flows, loss component for onerous groups, liability for incurred
    claims, and the IFRS 17.100 reconciliation
  * ``LiabilityForIncurredClaims``: discounting and risk adjustment, taken directly from a
    reserving model; ``lic_analysis_of_change`` splits its movement into incurred claims,
    prior-period adjustments and finance expenses
  * Risk adjustment by confidence level or cost of capital, and the implied confidence level
  * ``PAAReinsuranceHeld`` with a loss-recovery component
  * ``IFRS17Statements``: statement of profit or loss, financial position and cash flows,
    and key ratios
  * ``earned_premium_by_period`` and ``unearned_premium`` from a policy listing
  * ``paa_eligibility`` test
  * Opening balances for groups already in force, other attributable expenses, and the
    option to present part of insurance finance expenses in other comprehensive income
  * ``group_contracts``: portfolio, cohort and profitability groups
  * Disclosure tables: ``claims_development_table`` (IFRS 17.130), ``maturity_analysis``
    (IFRS 17.132) and ``lic_sensitivity`` (IFRS 17.128)
  * Statement of changes in equity, notes on insurance service expenses and finance
    income and expenses, and ``supplementary_position`` showing the carrying amount as
    unearned premium, premiums receivable, deferred acquisition costs and claims reserves
  * ``lrc_discount_rate`` for a significant financing component in the liability for
    remaining coverage (IFRS 17.56)
  * ``load_paa_case``: run a whole calculation from a folder of CSV files or an Excel
    workbook, with a fictional example case and a report with periods in columns
  * ``premiums_written`` for groups that grow as policies are sold, and roll-forward
    tables for reinsurance held
  * Totals for all periods on the statements, a reconciliation for the whole span of
    periods, a reinsurance reconciliation, and a formatted Excel workbook of the statements
    and notes (``IFRS17Statements.to_excel``, ``PAACase.to_excel``)

  The liability for remaining coverage reproduces the IASB premium allocation approach example
* Life, with expenses and beyond net premiums:

  * ``SelectMortalityTable`` and the AM92 select and ultimate table (``load_am92``)
  * ``LifePolicy`` with ``Expenses``: gross and net premiums, prospective and retrospective
    reserves, valuation of policies in force, the distribution of the insurer's loss, the
    premium for a given probability of loss, death strain and mortality profit
  * With-profits benefits with simple or compound bonuses
  * ``MultipleDecrementTable`` (dependent and independent rates, revised tables) and
    ``MultiStateModel`` (constant transition intensities)
  * ``ProfitTest``, ``UnitLinkedPolicy`` and ``zeroise_negative_cashflows``: profit vector,
    profit signature, net present value, internal rate of return and profit margin

  These reproduce published answers worked on AM92
* Reserving diagnostics (``actuneo.loss_reserving.diagnostics``): Mack's calendar year
  effect and development factor correlation tests, an intercept test, a test for trends in
  link ratios, outlying link ratios, ``backtest`` of the latest diagonals, and ``diagnose``
  to run them all
* ``GLMReserving``: over-dispersed Poisson model with an analytic prediction error; with
  ``structure="calendar"`` it estimates calendar year inflation from the triangle
* ``MLChainLadder``: chain-ladder factors adjusted by a scikit-learn model, for experiments
* ``benchmark_reserving``: compare reserving methods with the true outcome on simulated
  triangles; ``ClaimsSimulator(pattern_concentration=...)`` lets each claim be paid in its
  own proportions
* ``actuneo.zimbabwe.minimum_capital``: minimum capital by class of insurer as reported for
  SI 67 of 2025
* Reserving: one-year reserve risk (``MackChainLadder.cdr``, Merz and Wuthrich 2008),
  ``MunichChainLadder`` for paid and incurred triangles together, and triangles with
  missing cells (``allow_missing=True``)
* ``actuneo.simulation``: aggregate claims from a frequency and severity distribution, and
  ``ClaimsSimulator``, which produces triangles whose true ultimate cost is known
* ``actuneo.macro_africa``: World Bank indicators for African countries
* ``ChainLadder.cash_flows_by_origin``
* ``Triangle.from_excel``, ``from_csv``, ``to_excel`` and ``to_csv``; ``compare_methods`` and
  ``export_reserving_report`` for a formatted reserving workbook
* ``actuneo.utils``: ``write_report`` for formatted Excel workbooks, with a cover sheet of key
  figures and contents, banded tables, highlighted totals, coloured tabs and charts
* ``actuneo.templates``: Excel templates for a claims triangle and for a premium allocation
  approach case, each with a guide sheet (``save_template``)
* ``actuneo.finance``, compound interest:

  * ``InterestRate``: conversion between effective and nominal rates of interest and
    discount, simple rates and the force of interest; ``ForceOfInterest`` for a force of
    interest that varies with time, and continuous payment streams
  * Annuities certain: ``annuity``, ``accumulated_annuity``, ``perpetuity``,
    ``increasing_annuity``, ``decreasing_annuity``, ``continuously_increasing_annuity``
    and ``geometric_annuity``
  * ``Cashflows``: present and accumulated values, the yield on an equation of value,
    payback periods and accumulated profit
  * ``Loan``, ``loan_schedule``, ``annual_percentage_rate`` and ``flat_rate``
  * ``Bond`` with income tax, capital gains tax and optional redemption dates;
    ``equity_price``, ``equity_yield``, ``present_value_of_dividends`` and ``real_yield``
  * Spot rates, forward rates and par yields; ``discounted_mean_term``, ``volatility``,
    ``convexity``, ``immunisation_check`` and ``immunising_amounts``

  * Bonds bought between coupon dates (dirty and clean price, accrued interest), tax paid
    after a delay, price bounds for optional redemption dates, ``index_linked_cashflows``
    and ``property_value``
  * Uncertain payments (``probabilities``), ``has_unique_yield``, ``crossover_rate``,
    deferred increasing annuities, ``stepped_annuity``, continuous-time spot and forward
    rates, ``estimated_value_change``, and cashflow models of standard instruments
    (``actuneo.finance.instruments``)

  These are tested against the published answers to standard compound interest questions
* ``actuneo.finance``: helpers for hyperinflation restatement (``restate``,
  ``net_monetary_gain_or_loss``, ``cumulative_inflation``)
* A Status and Limitations page stating what is implemented, how each part was checked and
  what is missing
* ``Triangle.from_transactions``: build a triangle from a listing of dated payments or
  claims, with yearly, quarterly or monthly origin and development periods
* ``cash_flows``, ``future_incremental`` and ``discounted_reserve`` (flat rate or yield curve)
  on every projection method
* ``CapeCod`` method
* ``factors=`` on ``ChainLadder``, ``BornhuetterFerguson`` and ``InflationAdjustedChainLadder``
  to use selected development factors, and ``tail=`` on ``InflationAdjustedChainLadder``
* ``MackChainLadder``: ``residuals``, ``reserve_quantile`` (lognormal or normal) and the
  ``mse_method="independence"`` estimation error of Buchwalder et al. (2006)
* ``to_excel`` output and plots of triangles, reserves, residuals and bootstrap results
* ``InflationAdjustedChainLadder(development_per_origin=...)`` for development periods
  shorter than the origin periods

Changed
~~~~~~~

* ``MackChainLadder`` skips link ratios that start from zero claims instead of rejecting
  the triangle; ``BootChainLadder`` handles fitted values of zero
* The Mack sigma extrapolation and tail uncertainty were compared with the source of the
  R ChainLadder package and follow the same algorithm
* ``key_ratios`` now gives the combined ratio as insurance service expenses over insurance
  revenue, with the net-of-reinsurance and including-other-expenses variants as separate lines
* README and documentation no longer describe unbuilt features (regulatory templates,
  multi-currency modelling, calibrated local data) as available

Version 0.3.0 (2026-10-05)
--------------------------

Added
~~~~~

* ``InflationAdjustedChainLadder``: chain-ladder in real terms with explicit past and
  future claims inflation
* ``AverageCostPerClaim`` and ``grossing_up``: separate projection of claim numbers and
  average claim amounts, with grossing-up or development factors
* ``BornhuetterFerguson``: from premium and loss ratio, or any initial estimate of ultimate claims
* ``BootChainLadder``: bootstrap of the over-dispersed Poisson chain-ladder with gamma or
  over-dispersed Poisson process error, giving the reserve distribution and its quantiles
* Tail factors: ``estimate_tail_factor``, ``tail=True`` in ``ChainLadder``, and a tail with
  its own standard error and sigma in ``MackChainLadder``
* ``ChainLadder.fitted_triangle`` and ``fit_errors`` for checking the model against the past,
  and ``reserve(paid_to_date)`` for triangles of incurred claims

The new methods are tested against standard worked examples and against the scale parameter
and prediction error published by England and Verrall (2002) for the Taylor/Ashe data.

Version 0.2.0 (2026-10-05)
--------------------------

Added
~~~~~

* ``actuneo.loss_reserving``: ``Triangle`` (wide and long data, cumulative and incremental
  conversion, latest diagonal, link ratios and their averages), ``ChainLadder`` and
  ``MackChainLadder``, with the RAA and Taylor/Ashe example triangles. Results are tested
  against the figures published in Mack (1993) and by the R ChainLadder package
* ``actuneo.zimbabwe``: catalogue of the Zimbabwe 2023 mortality tables and ZWL to ZiG conversion
* ``CommutationFunctions`` (Dx, Nx, Sx, Cx, Mx, Rx)
* ``SurvivalFunctions``: pure endowment, endowment assurance, deferred and m-thly annuities,
  increasing assurance and annuity, net annual premiums, joint life and contingent functions,
  payment of claims at the moment of death
* ``LifeAssurance.last_survivor_assurance`` and ``net_annual_premium``;
  ``Annuities.last_survivor_annuity``, ``reversionary_annuity``, ``monthly_life_annuity`` and
  arithmetic increasing/decreasing annuities; ``Reserves.zillmer_reserve``
* ``InterestTheory``: rate of discount, force of interest, nominal rate of discount
* Tests that run every code example in the README and documentation

Fixed
~~~~~

* Assurance values ignored survival to the year of death and were overstated
* Temporary annuities in arrears were calculated as the annuity-due less 1
* Net premium reserves were calculated as a ratio of assurance to annuity instead of
  benefits less premiums; retrospective reserves were not per surviving policy
* Joint life and last survivor values used fixed multipliers instead of joint survival probabilities
* A guaranteed annuity was valued as an ordinary life annuity
* ``YieldCurve.from_par_rates`` returned the par rates unchanged; it now bootstraps zero rates
* A one-element list passed to ``MortalityTable.qx`` returned a scalar
* Mortality data files were not declared as package data
* Documentation examples called functions that did not exist

Changed
~~~~~~~

* ``MortalityTable`` requires consecutive integer ages and derives ``lx``, ``dx``, ``Lx``,
  ``Tx`` and ``ex`` from ``qx``; columns from the source file are available in ``published``.
  ``life_expectancy`` no longer returns the rounded published figure
* Survival beyond the last age of a table is zero; whole life functions on a table that ends
  with ``qx < 1`` close the table and warn
* Invalid ages and negative terms raise ``ValueError`` instead of returning 0
* Reserves may be negative; pass ``floor_at_zero=True`` to floor them
* ``Annuities.increasing_annuity`` and ``decreasing_annuity`` take a compound rate of
  change, default 0; ``annuity_with_withdrawal`` projects a level withdrawal from a fund
* ``LifeAssurance.reserve_*`` return net premium reserves per unit sum assured
* matplotlib is an optional dependency (``pip install actuneo[viz]``)
* Python 3.9 or later is required

Version 0.1.2 (2026-05-12)
--------------------------

Added
~~~~~

* Added JOSS submission materials (paper.md and paper.bib)

Changed
~~~~~~~

* Clarified public-facing descriptions to distinguish implemented modules from planned modules
* Corrected source installation instructions in README

Fixed
~~~~~

* Fixed a typo in the project title in README

Version 0.1.1 (2025-01-10)
--------------------------

Changed
~~~~~~~

* Updated documentation links to GitHub
* Removed Read the Docs badge until documentation is set up
* Added GitHub repository badge

Fixed
~~~~~

* Fixed package structure to include all submodules

Version 0.1.0 (2025-01-10)
--------------------------

Added
~~~~~

* Initial release of ACTUNEO
* Core mortality module with MortalityTable and SurvivalFunctions
* Finance module with InterestTheory, YieldCurve, and DurationConvexity
* Life module with LifeAssurance, Annuities, and Reserves
* Basic test suite
* Documentation structure
* PyPI package publication

Features
~~~~~~~~

* Mortality table creation and manipulation
* Life expectancy calculations
* Survival probability functions
* Interest rate calculations
* Yield curve construction and interpolation
* Duration and convexity measures
* Life assurance premium calculations
* Annuity valuations
* Reserve calculations

Modules
~~~~~~~

* ``actuneo.mortality``: Mortality tables and survival functions
* ``actuneo.finance``: Financial mathematics and yield curves
* ``actuneo.life``: Life insurance and annuity calculations
* ``actuneo.pensions``: Placeholder for future development
* ``actuneo.ifrs17``: Placeholder for future development
* ``actuneo.loss_reserving``: Placeholder for future development
* ``actuneo.macro_africa``: Placeholder for future development
* ``actuneo.simulation``: Placeholder for future development
* ``actuneo.utils``: Placeholder for future development

Upcoming
--------

Version 0.2.0 (Planned)
~~~~~~~~~~~~~~~~~~~~~~~

* African mortality tables
* IPEC Zimbabwe regulatory templates
* Multi-currency support
* Pension calculations
* Expanded test coverage
* Complete documentation on Read the Docs

Version 0.3.0 (Planned)
~~~~~~~~~~~~~~~~~~~~~~~

* IFRS 17 implementation
* Loss reserving methods
* Macroeconomic data connectors
* Stochastic simulation tools
* Web interface prototype

