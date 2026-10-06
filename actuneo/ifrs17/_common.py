"""Shared helpers for the IFRS 17 module."""

import numpy as np
from typing import Optional, Sequence


def as_series(values, n: int, name: str) -> np.ndarray:
    """A per-period input as an array of length n; a single number or None fills every period."""
    if values is None:
        return np.zeros(n)
    arr = np.atleast_1d(np.asarray(values, dtype=float))
    if arr.size == 1 and n != 1:
        arr = np.full(n, arr[0])
    if arr.shape != (n,):
        raise ValueError(f"{name} needs one value for each of the {n} periods")
    if np.any(np.isnan(arr)):
        raise ValueError(f"{name} must not contain missing values")
    return arr


def period_labels(periods: Optional[Sequence], n: int) -> list:
    if periods is None:
        return list(range(1, n + 1))
    labels = list(periods)
    if len(labels) != n:
        raise ValueError(f"periods needs one label for each of the {n} periods")
    return labels


def discount_factors(discount_rate, years: np.ndarray) -> np.ndarray:
    """
    Discount factors at the given times in years, from an annual effective
    rate or from a curve with a ``get_discount_factor(years)`` method.
    """
    years = np.asarray(years, dtype=float)
    if hasattr(discount_rate, "get_discount_factor"):
        return np.array([1.0 if t <= 0 else discount_rate.get_discount_factor(t) for t in years])
    if discount_rate <= -1:
        raise ValueError("discount_rate must be greater than -1")
    return (1 + discount_rate) ** -years
