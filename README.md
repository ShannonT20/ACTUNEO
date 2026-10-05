# ACTUNEO: Comprehensive Actuarial Python Library

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Documentation Status](https://readthedocs.org/projects/actuneo/badge/?version=latest)](https://actuneo.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/actuneo.svg)](https://badge.fury.io/py/actuneo)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue.svg)](https://github.com/ShannonT20/ACTUNEO)

## Vision and Objective

ACTUNEO is an open-source, community-driven actuarial Python library that empowers African and Zimbabwean actuaries to perform core actuarial, financial, and statistical computations with ease. The goal is to build a localized yet globally compatible toolkit that supports insurance, pensions, and investment analytics, while integrating with modern data science tools.

## Features

### Implemented Modules

- **mortality**: Mortality tables, life contingencies and commutation functions, with the Zimbabwe 2023 tables included
- **finance**: Interest theory, yield curve construction/interpolation, duration, and convexity
- **life**: Life assurance, annuities, premiums and reserves, for one or two lives
- **loss_reserving**: Claims triangles, chain-ladder projection and Mack's standard error
- **zimbabwe**: Catalogue of the Zimbabwe 2023 mortality tables and ZWL to ZiG conversion

### In Development (Scaffolding Present)

- **pensions**: Contribution schedules, benefit projections, and actuarial valuations for pension schemes
- **ifrs17**: Insurance contract measurement models (GMM, VFA, PAA), CSM, risk adjustment, discounting
- **macro_africa**: Country-specific economic data connectors (inflation, GDP, currency exchange)
- **simulation**: Monte Carlo simulations for stochastic actuarial models
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

## Documentation

📚 **Full documentation is available at [https://actuneo.readthedocs.io/](https://actuneo.readthedocs.io/)**

The documentation includes:

- **Installation Guide**: Step-by-step installation instructions
- **Quick Start**: Get started with ACTUNEO in minutes
- **Examples**: Comprehensive code examples for all modules
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
- [ ] Pension calculations
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
