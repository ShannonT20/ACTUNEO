"""
Utils Module

Input and output helpers.

Currently provides formatted Excel reports. Validation and other reporting
utilities are planned.
"""

from .excel import (
    Sheet, write_report, NUMBER_FORMAT, DECIMAL_FORMAT, RATIO_FORMAT, FACTOR_FORMAT,
    NAVY, BLUE, TEAL, GREY,
)

__all__ = [
    'Sheet',
    'write_report',
    'NUMBER_FORMAT',
    'DECIMAL_FORMAT',
    'RATIO_FORMAT',
    'FACTOR_FORMAT',
    'NAVY',
    'BLUE',
    'TEAL',
    'GREY',
]
