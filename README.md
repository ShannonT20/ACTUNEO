# ACTUNEO: African Actuarial Python Library

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Documentation Status](https://readthedocs.org/projects/actuneo/badge/?version=latest)](https://actuneo.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/actuneo.svg)](https://pypi.org/project/actuneo/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue.svg)](https://github.com/ShannonT20/ACTUNEO)

## Vision and Objective

ACTUNEO is an open-source, community-driven actuarial Python library that empowers African and Zimbabwean actuaries to perform core actuarial, financial, and statistical computations with ease. The goal is to build a localized yet globally compatible toolkit that supports insurance, pensions, and investment analytics, while integrating with modern data science tools.

## Status

ACTUNEO is an early-stage library. It has not been reviewed by an independent actuary and has not yet been run on a real insurer's data. Some parts reproduce published figures; others are checked only by hand calculation. It is not certified for statutory or audited reporting: check its output independently before relying on it.

**Read [Status and Limitations](docs/status.rst) before using it.** It lists what is implemented, how each part was checked, and what is missing.

## Features

### Implemented Modules

- **mortality**: Mortality tables, select mortality, life contingencies and commutation functions, with the Zimbabwe 2023 tables and AM92 included
- **finance**: Interest theory, yield curve construction/interpolation, duration, and convexity
- **life**: Life assurance, annuities, net and gross premiums and reserves with expenses, with-profits benefits, mortality profit, multiple decrement and multi-state models, unit-linked policies and profit testing
- **loss_reserving**: Claims triangles, chain-ladder (basic and inflation-adjusted), average cost per claim, Bornhuetter-Ferguson, Mack's standard error and bootstrap
- **pensions**: Defined benefit valuations (projected unit, entry age and other funding methods), funding level and contribution rates, IAS 19 accounting, defined contribution projections, and calculations from Zimbabwe's Pensions and Provident Funds Act
- **ifrs17**: Premium allocation approach, liability for incurred claims with discounting and risk adjustment, reinsurance held, and the IFRS 17 statement of profit or loss, financial position and cash flows
- **simulation**: Aggregate claims simulation, and a claims simulator that produces triangles with a known true cost for testing reserving methods
- **macro_africa**: World Bank economic indicators (inflation, growth, exchange and interest rates) for African countries, downloaded on request
- **zimbabwe**: Catalogue of the Zimbabwe 2023 mortality tables and ZWL to ZiG conversion

### In Development (Scaffolding Present)

- **ifrs17** (further): General measurement model (GMM) and variable fee approach (VFA), with the contractual service margin
- **utils**: Data input-output helpers, validation, and reporting utilities

### African Market Focus

- **Localized Assumptions (Roadmap)**: Support for market-specific mortality, inflation, and interest rate tables
- **Regulatory Alignment (Roadmap)**: Extensible templates for regulatory reporting (e.g., IPEC Zimbabwe, PASA, SAM)
- **Currency Handling (Roadmap)**: Multi-currency modelling (USD, ZWL, Rand, etc.) with inflation-adjusted projections
- **Socioeconomic Context (Roadmap)**: Assumptions relevant to informal sector, microinsurance, and low-coverage environments

## Installation

### From PyPI

```bash
pip install actuneo
```

ACTUNEO is in beta. Read the [Status and Limitations](docs/status.rst) page before
relying on any result. Releases before 0.4.0 contain calculation errors that have since
been corrected: upgrade with `pip install --upgrade actuneo`.

### From Source

```bash
git clone https://github.com/ShannonT20/ACTUNEO.git
cd ACTUNEO
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### Mortality Analysis

```python
from actuneo.mortality import MortalityTable, SurvivalFunctions

# Zimbabwe 2023 mortality tables are included
print(MortalityTable.zimbabwe_2023_tables())
mt = MortalityTable.from_zimbabwe_2023("male_assured_lives")

le = mt.life_expectancy(30)
print(f"Life expectancy at age 30: {le:.1f} years")

# Survival and life contingencies at 8% interest
sf = SurvivalFunctions(mt, interest_rate=0.08)
print(f"20-year survival probability from 30: {sf.npx(30, 20):.3f}")
print(f"20-year term assurance, A30:20 = {sf.assurance(30, 20):.5f}")
print(f"20-year annuity-due, a30:20 = {sf.annuity_due(30, 20):.4f}")
```

### Financial Calculations

```python
from actuneo.finance import InterestTheory, YieldCurve

# Interest theory
it = InterestTheory(interest_rate=0.05)
fv = it.future_value(1000, 10)  # Future value of $1000 in 10 years
print(f"Future value: ${fv:.2f}")

# Yield curve
maturities = [1, 2, 5, 10, 30]
yields = [0.03, 0.035, 0.045, 0.055, 0.065]
yc = YieldCurve(maturities, yields)
yield_15y = yc.get_yield(15)
print(f"15-year yield: {yield_15y:.3%}")
```

### Life Insurance Calculations

Values are per unit sum assured; premiums are annual in advance.

```python
from actuneo.life import LifeAssurance

la = LifeAssurance(mt, interest_rate=0.08)
sum_assured = 100_000

premium = sum_assured * la.net_annual_premium(30, 20, "endowment")
print(f"20-year endowment, annual premium at age 30: ${premium:,.2f}")

reserve = sum_assured * la.reserve_endowment(30, 20, 5)
print(f"Reserve after 5 years: ${reserve:,.2f}")
```

### Claims Triangles and Chain-Ladder

```python
from actuneo.loss_reserving import MackChainLadder, load_raa

triangle = load_raa()                  # or Triangle(...) / Triangle.from_long(...)
print(triangle.age_to_age().round(3))  # link ratios and their averages

mack = MackChainLadder(triangle, est_sigma="mack")
print(mack.summary().round(3))
print(f"IBNR {mack.total_ibnr:,.0f} with standard error {mack.total_mack_se:,.0f}")
```

### Ask a Question, Get a Worked Solution

Type an actuarial question in plain words and get the answer set out like a textbook
solution: what was given, each step with its formula, the substitution and the result,
the answer, and the assumptions behind it.

```bash
actuneo-ai ask "What is the monthly repayment on a loan of 50,000 over 10 years at 8%?"
```

```bash
actuneo-ai chat
```

```bash
actuneo-ai topics
```

In Python or a notebook:

```py
import actuneo.ai as ai

answer = ai.ask("If I am 45 now, what are the odds I am still alive at 70?")
answer              # displays the worked solution in a notebook
answer.value        # 0.6937
print(answer.explain())
```

Fixed rules read the numbers out of the question ("120k", "eight and a half percent",
"750 a month", "alive at 70") and recognise the calculation. ACTUNEO does the arithmetic,
and the working is written from fixed templates, so no language model writes any figure
or any step. If something needed is missing, the answer says what, instead of guessing.

It covers present and accumulated values, annuities-certain, loan repayments, bond
prices, effective rates, net present value and internal rate of return, expectation of
life, survival probabilities, pensions for life, term assurance premiums, chain ladder
reserves, Mack's standard error, the IFRS 17 premium allocation approach, retirement fund
projections and ZWL to ZiG.

- **Several parts.** "A man is aged 40. (a) What is his life expectancy? (b) What is the
  probability he survives 20 years?" is split and each part answered.
- **Your own data.** `ai.ask("Chain ladder reserve for my triangle?", data=my_triangle)`
  takes a triangle, a mortality table with columns `age` and `qx`, or a list of yearly
  cash flows. In a terminal: `actuneo-ai ask "IBNR reserve?" --data my_triangle.csv`.
- **Zimbabwe tables by default.** Mortality questions use the Zimbabwe 2023 tables unless
  the question says "South African" or "AM92", or you pass a table.
- **Questions in words.** "What is the premium allocation approach?" or "What does
  ACTUNEO do for pensions?" is answered by the language model, speaking as an actuary. It
  is given the matching passages of ACTUNEO's own documentation and uses them with its
  own knowledge; the passages are listed under the answer, which is marked as written by
  the model. This needs a model installed (below), and its quality depends on the model.

`examples/actuneo_ai_setup.ipynb` has questions to try at three levels of difficulty.

#### The local model

Clearly worded calculations need no model. A language model that runs on your own
machine answers questions in words and chooses the calculation when the wording is
unusual. It is offline and free after the download. The smallest model is often wrong
about general actuarial ideas; the 4B model or larger is much better.

```bash
pip install "actuneo[ai]"
```

```bash
actuneo-ai models
```

```bash
actuneo-ai setup
```

`models` lists every model and whether it fits your machine. `setup` shows the
recommendation, asks, and downloads with a progress bar. Other commands: `status`,
`benchmark`, `remove`, and `setup --model KEY --force` to pick a model yourself. Use
`python -m actuneo.ai` if the `actuneo-ai` command is not found.

| Key | Model | Download | Free memory needed |
|---|---|---|---|
| `minimal` | Qwen3 0.6B | 0.4 GB | about 1.5 GB (a 4 GB laptop with no GPU) |
| `tiny` | Qwen3 1.7B | 1.1 GB | about 3 GB |
| `small` | Qwen3 4B | 2.5 GB | about 5 GB |
| `default` | Qwen3 8B | 5.0 GB | about 9.5 GB |
| `strong` | Qwen3 14B | 9.0 GB | about 16 GB, or a 13 GB graphics card |
| `max` | Qwen3 30B-A3B | 18.6 GB | about 31 GB, or a 24 GB graphics card when asked for as `largest` |

When the model does choose, the answer says so and asks you to check. A number the model
supplies is accepted only if it appears in your question. If the machine is too small,
use your own key instead: `actuneo-ai configure gemini` (or `openai`, `anthropic`,
`ollama`). The key is never written to ACTUNEO's configuration file.

## Documentation

📚 **Full documentation is available at [https://actuneo.readthedocs.io/](https://actuneo.readthedocs.io/)**

The documentation includes:

- **Installation Guide**: Step-by-step installation instructions
- **Quick Start**: Get started with ACTUNEO in minutes
- **Examples**: Worked examples for the implemented modules
- **Status and Limitations**: What is and is not implemented, and how it was checked
- **API Reference**: Complete API documentation for all classes and functions
- **Contributing Guide**: How to contribute to the project
- **Changelog**: Version history and release notes

### Documentation Sections

- [Installation Guide](https://actuneo.readthedocs.io/en/latest/installation.html)
- [Quick Start Tutorial](https://actuneo.readthedocs.io/en/latest/quickstart.html)
- [Examples](https://actuneo.readthedocs.io/en/latest/examples.html)
- [API Reference](https://actuneo.readthedocs.io/en/latest/api/mortality.html)
- [Contributing](https://actuneo.readthedocs.io/en/latest/contributing.html)

## Contributing

We welcome contributions from the actuarial community, especially from African and Zimbabwean actuaries and students. Here's how you can contribute:

### Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/ShannonT20/ACTUNEO.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate the environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Unix)
5. Install development dependencies: `pip install -e ".[dev]"`
6. Run tests: `pytest`

### Guidelines

- Follow PEP 8 style guidelines
- Write comprehensive tests for new features
- Update documentation for API changes
- Add examples for new functionality
- Ensure cross-platform compatibility

### Areas for Contribution

- **African Mortality Tables**: Contribute localized mortality data and improvement models
- **Regulatory Frameworks**: Implement calculations for specific African regulatory requirements
- **Pension Systems**: Develop models for various pension scheme structures
- **IFRS 17**: Implement insurance contract measurement models
- **Loss Reserving**: Add stochastic reserving methods
- **Documentation**: Improve documentation and add tutorials
- **Testing**: Expand test coverage and add integration tests

## Roadmap

### Phase 1 (Current): Core Implementation
- [x] Basic package structure
- [x] Mortality tables and survival functions
- [x] Financial calculations (interest, yield curves)
- [x] Life assurance and annuity calculations
- [x] Unit testing framework
- [x] Documentation setup

### Phase 2: Expansion
- [x] Pension calculations
- [ ] IFRS 17 implementation
- [ ] Loss reserving methods
- [ ] African economic data connectors
- [ ] Stochastic simulation tools

### Phase 3: Advanced Features
- [ ] Machine learning integration
- [ ] Web application interfaces
- [ ] Regulatory reporting tools
- [ ] Multi-language support

## Testing

Run the test suite:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=actuneo --cov-report=html
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use ACTUNEO in your research or work, please cite:

```bibtex
@software{sikadi_actuneo_2025,
  author = {Sikadi, Shannon Tafadzwa},
  title = {ACTUNEO: African Actuarial Python Library},
  url = {https://github.com/ShannonT20/ACTUNEO},
  year = {2025}
}
```

## Contact

- **Author**: Shannon Tafadzwa Sikadi
- **Email**: shannonsikadi@gmail.com
- **GitHub**: https://github.com/ShannonT20/ACTUNEO
- **LinkedIn**: https://www.linkedin.com/in/shannon-sikadi-9370b3196/

## Acknowledgments

- Inspired by existing actuarial libraries like `lifecontingencies`, `actuarialmath`, and `pandas`
- Special thanks to the African actuarial community for their support and feedback
- Built with modern Python data science tools (NumPy, Pandas, SciPy, Matplotlib)

---

**ACTUNEO**: Empowering African actuaries through open-source technology.
