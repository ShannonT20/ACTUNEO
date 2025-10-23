# ACTUNEO: African Actuarial Python Library

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/actuneo/badge/?version=latest)](https://actuneo.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/actuneo.svg)](https://badge.fury.io/py/actuneo)

## Vision and Objective

ACTUNEO is an open-source, community-driven actuarial Python library that empowers African and Zimbabwean actuaries to perform core actuarial, financial, and statistical computations with ease. The goal is to build a localized yet globally compatible toolkit that supports insurance, pensions, and investment analytics, while integrating with modern data science tools.

## Features

### Core Modules

- **mortality**: Mortality tables, survival functions, graduation, and mortality improvement models
- **life**: Life assurance, annuities, reserves, and premium calculations
- **pensions**: Contribution schedules, benefit projections, and actuarial valuations for pension schemes
- **ifrs17**: Measurement models (GMM, VFA, PAA), CSM, risk adjustment, discounting
- **loss_reserving**: Chain-ladder, Bornhuetter-Ferguson, and stochastic reserving models
- **finance**: Interest theory, yield curve construction, duration, and convexity measures
- **macro_africa**: Country-specific economic data connectors (inflation, GDP, currency exchange)
- **simulation**: Monte Carlo simulations for stochastic actuarial models
- **utils**: Excel/CSV input-output functions, validation, and reporting

### African Market Focus

- **Localized Data Support**: Includes mortality, inflation, and interest rate tables calibrated to African and Zimbabwean markets
- **Regulatory Alignment**: Built-in parameters for regulatory reporting (e.g., IPEC Zimbabwe, PASA, SAM)
- **Currency Handling**: Supports multi-currency modeling (USD, ZWL, Rand, etc.) with inflation-adjusted projections
- **Socioeconomic Context**: Incorporates assumptions relevant to informal sector, microinsurance, and low-coverage environments

## Installation

### From PyPI (Coming Soon)

```bash
pip install actuneo
```

### From Source

```bash
git clone https://github.com/yourusername/actuneo.git
cd actuneo
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### Mortality Analysis

```python
import numpy as np
from actuneo.mortality import MortalityTable, SurvivalFunctions

# Create a mortality table
ages = np.arange(20, 101)
qx = 0.001 * (ages - 20) / 80  # Simplified mortality rates
mt = MortalityTable(ages, qx, name="Example Table")

# Calculate life expectancy
le = mt.life_expectancy(30)
print(f"Life expectancy at age 30: {le:.1f} years")

# Survival functions
sf = SurvivalFunctions(mt)
survival_prob = sf.npx(30, 20)  # Probability of surviving 20 years from age 30
print(f"20-year survival probability: {survival_prob:.3f}")
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

```python
from actuneo.life import LifeAssurance

# Life assurance calculations
la = LifeAssurance(mt, interest_rate=0.05)
premium = la.whole_life_assurance(30, sum_assured=100000)
print(f"Whole life premium at age 30: ${premium:.2f}")

# Reserve calculation
reserve = la.reserve_whole_life(30, 5, annual_premium=1500)
print(f"Reserve after 5 years: ${reserve:.2f}")
```

## Documentation

Full documentation is available at [https://actuneo.readthedocs.io/](https://actuneo.readthedocs.io/)

## Contributing

We welcome contributions from the actuarial community, especially from African and Zimbabwean actuaries and students. Here's how you can contribute:

### Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/actuneo.git`
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
- [ ] Unit testing framework
- [ ] Documentation setup

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
@software{sikadi_actuneo_2024,
  author = {Sikadi, Shannon Tafadzwa},
  title = {ACTUNEO: African Actuarial Python Library},
  url = {https://github.com/yourusername/actuneo},
  year = {2024}
}
```

## Contact

- **Author**: Shannon Tafadzwa Sikadi
- **Email**: [Your email here]
- **GitHub**: [https://github.com/yourusername/actuneo](https://github.com/yourusername/actuneo)
- **LinkedIn**: [Your LinkedIn profile]

## Acknowledgments

- Inspired by existing actuarial libraries like `lifecontingencies`, `actuarialmath`, and `pandas`
- Special thanks to the African actuarial community for their support and feedback
- Built with modern Python data science tools (NumPy, Pandas, SciPy, Matplotlib)

---

**ACTUNEO**: Empowering African actuaries through open-source technology.
