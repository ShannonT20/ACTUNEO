"""
Setup script for ACTUNEO - African Actuarial Python Library
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = '''
ACTUNEO: African Actuarial Python Library

An open-source, community-driven actuarial Python library that empowers African and
Zimbabwean actuaries to perform core actuarial, financial, and statistical computations.

Modules:
- mortality: Mortality tables, survival functions, graduation, and mortality improvement models
- life: Life assurance, annuities, reserves, and premium calculations
- pensions: Contribution schedules, benefit projections, and actuarial valuations for pension schemes
- ifrs17: Measurement models (GMM, VFA, PAA), CSM, risk adjustment, discounting
- loss_reserving: Chain-ladder, Bornhuetter-Ferguson, and stochastic reserving models
- finance: Interest theory, yield curve construction, duration, and convexity measures
- macro_africa: Country-specific economic data connectors (inflation, GDP, currency exchange)
- simulation: Monte Carlo simulations for stochastic actuarial models
- utils: Excel/CSV input-output functions, validation, and reporting
'''

setup(
    name='actuneo',
    version='0.1.2',
    author='Shannon Tafadzwa Sikadi',
    author_email='shannonsikadi@gmail.com',
    description='African Actuarial Python Library for insurance, pensions, and investment analytics',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ShannonT20/ACTUNEO',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Office/Business :: Financial :: Accounting',
    ],
    keywords='actuarial insurance pensions finance mortality africa zimbabwe',
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.2.0',
            'pytest-cov>=2.12.0',
            'black>=21.0.0',
            'flake8>=3.9.0',
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=1.0.0',
        ],
        'ml': [
            'scikit-learn>=1.0.0',
            'statsmodels>=0.12.0',
        ],
        'viz': [
            'plotly>=5.0.0',
            'seaborn>=0.11.0',
        ],
    },
    project_urls={
        'Bug Reports': 'https://github.com/ShannonT20/ACTUNEO/issues',
        'Source': 'https://github.com/ShannonT20/ACTUNEO',
        'Documentation': 'https://actuneo.readthedocs.io/',
    },
)
