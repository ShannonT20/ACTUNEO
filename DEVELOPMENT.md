# ACTUNEO Development Guide

This guide provides detailed instructions for setting up a development environment and contributing to ACTUNEO.

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Git**: Latest version
- **Operating System**: Windows, macOS, or Linux

### Recommended Tools
- **IDE**: VS Code, PyCharm, or any Python-compatible editor
- **Virtual Environment**: `venv` (built-in) or `conda`
- **Version Control**: Git with GitHub account

## Quick Setup

### 1. Fork and Clone
```bash
# Fork the repository on GitHub first
git clone https://github.com/yourusername/actuneo.git
cd actuneo
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install ACTUNEO in development mode with all extras
pip install -e ".[dev,ml,viz]"

# Verify installation
python -c "import actuneo; print(f'ACTUNEO {actuneo.__version__} installed successfully')"
```

### 4. Run Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=actuneo --cov-report=html

# Run specific test file
pytest tests/test_mortality.py
```

## Development Workflow

### Daily Development
```bash
# Activate environment (if not already active)
source venv/bin/activate  # Windows: venv\Scripts\activate

# Make your changes...

# Run tests frequently
pytest

# Format code
black actuneo/ tests/

# Check linting
flake8 actuneo/
```

### Before Committing
```bash
# Run full test suite
pytest --cov=actuneo --cov-report=term-missing

# Format all code
black .

# Check for linting issues
flake8 actuneo/ tests/

# Update documentation if needed
# (We'll add docs later)
```

### Working with Git
```bash
# Check status
git status

# Stage changes
git add .

# Commit with conventional format
git commit -m "feat: add new mortality model

- Implement Gompertz-Makeham model
- Add parameter validation
- Include comprehensive tests"

# Push to your fork
git push origin feature/your-feature-name
```

## Project Structure

```
ACTUNEO/
├── actuneo/                    # Main package
│   ├── __init__.py            # Package initialization
│   ├── mortality/             # Mortality calculations
│   │   ├── __init__.py
│   │   ├── mortality_table.py
│   │   └── survival_functions.py
│   ├── finance/               # Financial mathematics
│   ├── life/                  # Life insurance calculations
│   └── [other modules...]
├── tests/                     # Test suite
│   ├── conftest.py           # Test configuration and fixtures
│   ├── test_*.py             # Individual test files
├── docs/                     # Documentation
├── .github/                  # GitHub configuration
│   └── workflows/            # CI/CD workflows
├── CONTRIBUTING.md           # Contribution guidelines
├── CODE_OF_CONDUCT.md        # Community standards
├── DEVELOPMENT.md            # This file
├── LICENSE                   # MIT License
├── pyproject.toml            # Modern Python packaging
├── setup.py                  # Traditional packaging
├── requirements.txt          # Dependencies
└── README.md                 # Project overview
```

## Module Development Guidelines

### Adding a New Module

1. **Create Module Directory**
```bash
mkdir actuneo/new_module
```

2. **Create `__init__.py`**
```python
# actuneo/new_module/__init__.py
"""
New Module

Brief description of what this module does.
"""

from .main_class import MainClass

__all__ = ['MainClass']
```

3. **Implement Core Classes**
```python
# actuneo/new_module/main_class.py
"""
Main implementation for new module.
"""

class MainClass:
    """Main class documentation."""

    def __init__(self):
        pass

    def method_name(self, param):
        """Method documentation."""
        pass
```

4. **Update Main Package**
```python
# actuneo/__init__.py
# Add import for new module
from . import new_module
```

5. **Add Tests**
```python
# tests/test_new_module.py
import pytest
from actuneo.new_module import MainClass

class TestMainClass:
    def test_initialization(self):
        obj = MainClass()
        assert obj is not None

    def test_method_name(self):
        obj = MainClass()
        result = obj.method_name("test")
        assert result is not None
```

### African Market Considerations

When developing modules, consider:

- **Data Availability**: African markets may have limited historical data
- **Regulatory Frameworks**: Different countries have different requirements
- **Economic Context**: Consider inflation, currency fluctuations, informal sectors
- **Cultural Factors**: Risk perceptions and insurance uptake patterns

## Testing Strategy

### Test Organization
- **Unit Tests**: Test individual functions and methods
- **Integration Tests**: Test module interactions
- **Regression Tests**: Ensure existing functionality still works

### Test Coverage Goals
- **Minimum Coverage**: 80% overall
- **Critical Paths**: 95% for core actuarial calculations
- **New Features**: 100% for new code

### Running Tests
```bash
# Basic test run
pytest

# With coverage report
pytest --cov=actuneo --cov-report=html
open htmlcov/index.html  # View coverage report

# Specific module tests
pytest tests/test_mortality.py -v

# Tests matching pattern
pytest -k "test_mortality_table"

# Stop on first failure
pytest -x
```

### Writing Tests
```python
import pytest
import numpy as np
from actuneo.mortality import MortalityTable

class TestMortalityTable:
    def test_creation_with_valid_data(self):
        """Test creating a mortality table with valid inputs."""
        ages = [20, 30, 40]
        qx = [0.001, 0.005, 0.01]

        mt = MortalityTable(ages, qx)

        assert len(mt.ages) == 3
        assert mt.name == "Unnamed Table"

    def test_invalid_qx_values(self):
        """Test error handling for invalid mortality rates."""
        ages = [20, 30, 40]
        invalid_qx = [0.001, 1.5, 0.01]  # 1.5 > 1.0

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            MortalityTable(ages, invalid_qx)

    @pytest.mark.parametrize("age,qx_expected", [
        (20, 0.001),
        (30, 0.005),
        (40, 0.01),
    ])
    def test_qx_retrieval(self, age, qx_expected):
        """Test getting mortality rates for different ages."""
        ages = [20, 30, 40]
        qx = [0.001, 0.005, 0.01]
        mt = MortalityTable(ages, qx)

        assert mt.get_qx(age) == pytest.approx(qx_expected)
```

## Documentation

### Code Documentation
- Use Google-style docstrings
- Document all public functions, classes, and methods
- Include parameter types and return types
- Provide usage examples where helpful

```python
def calculate_premium(self, age: int, sum_assured: float) -> float:
    """
    Calculate insurance premium for given age and sum assured.

    This method uses the net single premium approach with safety margins.

    Args:
        age: Age of the insured person
        sum_assured: Insurance coverage amount

    Returns:
        Annual premium amount

    Raises:
        ValueError: If age is outside valid range

    Example:
        >>> calc = PremiumCalculator()
        >>> premium = calc.calculate_premium(30, 100000)
        >>> print(f"Annual premium: ${premium:.2f}")
        Annual premium: $1250.00
    """
    pass
```

### Building Documentation
```bash
# Install documentation dependencies
pip install -e ".[dev]"

# Build HTML documentation
cd docs
make html

# View documentation
open _build/html/index.html
```

## Performance Considerations

### Profiling Code
```python
import cProfile
import pstats

def profile_function():
    # Your code here
    pass

# Profile the function
profiler = cProfile.Profile()
profiler.enable()
profile_function()
profiler.disable()

# Print statistics
stats = pstats.Stats(profiler).sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
```

### Optimization Tips
- Use NumPy arrays for numerical computations
- Avoid loops where possible - use vectorized operations
- Cache expensive calculations when appropriate
- Consider memory usage for large datasets

## Security Considerations

### Data Handling
- Never store sensitive personal data in tests
- Use anonymized or synthetic data for examples
- Be careful with financial calculations precision
- Validate all inputs thoroughly

### Code Security
- Avoid `eval()` and `exec()` functions
- Use safe YAML/JSON parsing
- Be cautious with file operations
- Follow secure coding practices

## Getting Help

### Resources
- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and community discussion
- **Documentation**: Check existing docs first
- **Tests**: Look at existing test patterns

### Communication
- **Primary Language**: English
- **Response Time**: We aim to respond within 48 hours
- **Timezone Consideration**: Team spans multiple African timezones

---

Happy coding! May your contributions make actuarial science more accessible across Africa. 🇿🇼✨
