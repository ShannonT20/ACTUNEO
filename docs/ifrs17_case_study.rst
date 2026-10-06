IFRS 17 Case Study: Kopje Insurance
===================================

A full premium allocation approach calculation run from input files. Kopje
Insurance is a fictional company and every figure is invented for the
example.

The question
------------

Kopje Insurance is a short-term insurer that started writing business on
1 January 2026 with equity of USD 600,000. It reports monthly and measures
all its contracts under the premium allocation approach. All policies give
one year of cover and, as SI 81 of 2023 requires, premiums are received
before cover starts.

**Motor portfolio.** New policies are sold every month, with premiums from
USD 90,000 to USD 130,000 a month (USD 1,280,000 for the year). Commission
of 15% is paid when the premium is received and is deferred. Premiums are
earned by the 24ths method: half a month of cover in the month of sale and a
full month after that. Claims are incurred at between 57% and 71% of earned
premium and are paid over four months (45%, 30%, 15%, 10%). Claims
liabilities are discounted, with interest accreting at 0.4% a month, and
carry a risk adjustment of 8%. Estimates of earlier claims were revised in
April, July and October. Policy administration costs are 4% of earned
premium. The expected cost of the unexpired cover is 72% of unearned premium.

**Fire portfolio.** Premiums are USD 36,000 to USD 50,000 a month
(USD 511,000 for the year) with commission of 20%. Claims are incurred at
between 70% and 110% of earned premium and paid over four months. The risk
adjustment is 10% and administration costs are 5% of earned premium. The
expected cost of the unexpired cover is 86% of unearned premium.

**Reinsurance.** A 30% quota share covers the motor portfolio. The
reinsurer pays 25% commission, which is deducted from the premium.

**Other.** Investment return rises from USD 4,000 to USD 11,800 a month.
Expenses not attributable to insurance contracts are USD 14,000 a month.
Tax is 25% of profit, with no relief for losses.

Required, for each month of 2026:

1. the roll-forward of the liability for remaining coverage and of the
   liability for incurred claims for each portfolio;
2. whether either portfolio is onerous, and any loss component;
3. the roll-forward of the reinsurance assets;
4. the statement of profit or loss, the statement of financial position and
   the statement of cash flows;
5. the claims, expense and combined ratios;
6. the insurance liabilities analysed into unearned premium, deferred
   acquisition costs and claims reserves.

The input files
---------------

The case is four CSV files (or one Excel workbook with four sheets):
``settings``, ``groups``, ``data`` and ``entity``. Their columns are
described in the API reference under *Case Files*. To get a copy to open in
Excel and edit:

.. code-block:: py

   from actuneo.templates import save_template
   save_template("paa_case")        # paa_case_template.xlsx, with a guide sheet

   from actuneo.ifrs17 import copy_example_case
   copy_example_case("my_case")     # or a folder with the four CSV files

The solution
------------

.. code-block:: python

   import pandas as pd
   from actuneo.ifrs17 import load_paa_case, example_case_path

   pd.set_option("display.width", 250)
   pd.set_option("display.max_columns", 20)

   case = load_paa_case(example_case_path())     # or load_paa_case("my_case")
   print(case)

   tables = case.report()        # months in columns, line items in rows
   print(list(tables))

   # 1. Insurance roll-forwards
   print(tables["Motor 2026 - LRC"].round(0))
   print(tables["Motor 2026 - LIC"].round(0))
   print(tables["Fire 2026 - LRC"].round(0))
   print(tables["Fire 2026 - LIC"].round(0))

   # 2. Onerous contracts: the loss component at each month end
   for name, group in case.groups.items():
       print(name, group.loss_component.round(0))

   # 3. Reinsurance roll-forwards
   print(tables["Motor quota share - ARC"].round(0))
   print(tables["Motor quota share - AIC"].round(0))

   # 4. Financial statements
   print(tables["Profit or loss"].round(0))
   print(tables["Financial position"].round(0))
   print(tables["Cash flows"].round(0))
   print("Balance check:", case.statements.balance_check())

   # 5 and 6. Ratios and the IFRS 4 style analysis
   print(tables["Key ratios"].round(3))
   print(tables["Supplementary position"].round(0))

To write the whole calculation to a formatted Excel workbook:

.. code-block:: py

   case.to_excel("kopje_results.xlsx")

The workbook opens on a cover sheet with the key figures (insurance
revenue, insurance service result, profit, combined ratio, insurance
contract liabilities and equity) and a linked list of contents. The
statements come first, laid out as an insurer publishes them: a statement of
profit or loss and other comprehensive income leading to the insurance
service result and the net financial result; a statement of financial
position with assets, liabilities and equity; then cash flows and changes in
equity. Each has a column per month and a total for the year, negatives in
red brackets, totals highlighted, charts of the main trends, and the
paragraph of IFRS 17 beside the main lines. The notes, key ratios, roll-forwards and the IFRS 17.100
reconciliation of each group follow.

What the answer shows
---------------------

* The motor portfolio is not onerous. Its liability for remaining coverage
  grows through the year as policies are sold faster than they are earned.
* The fire portfolio is onerous: the expected cost of the unexpired cover is
  more than the unearned premium less deferred commission, so a loss
  component is recognised and grows with the book.
* The company makes losses in the first months, when little premium has been
  earned against fixed monthly expenses, and becomes profitable later in the
  year.
* The statement of financial position balances in every month.

This case tests that the calculation runs end to end and ties together. Its
figures have not been checked against an independent calculation.
