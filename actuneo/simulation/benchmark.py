"""
Benchmark of Reserving Methods

Runs reserving methods on simulated triangles and compares their reserves
with what the simulated claims finally cost.
"""

from typing import Callable, Dict

import numpy as np
import pandas as pd

from .claims import ClaimsSimulator


def benchmark_reserving(simulator: ClaimsSimulator,
                        methods: Dict[str, Callable],
                        n_simulations: int = 50,
                        seed: int = 0) -> pd.DataFrame:
    """
    Compare reserving methods against the truth on simulated triangles.

    For each simulation the claims are generated to final settlement, the
    triangle a reserving actuary would see is built, each method is applied
    to it, and its reserve is compared with the payments still to come.

    Args:
        simulator: Configured :class:`ClaimsSimulator` (its seed is ignored)
        methods: Name of each method and a function that takes the
            triangle and returns either the total reserve or a fitted model
            with a ``total_ibnr``
        n_simulations: Number of triangles
        seed: Seed of the first simulation; later ones use seed + 1, ...

    Returns:
        DataFrame by method with the ``mean_error`` (bias) and the root
        mean squared error ``rmse`` of the reserve as proportions of the
        true reserve, the worst under- and over-statement, and the number of
        simulations in which the method failed. The results describe the
        simulated portfolio only.
    """
    if n_simulations < 1:
        raise ValueError("n_simulations must be at least 1")
    errors = {name: [] for name in methods}
    failed = {name: 0 for name in methods}
    original_seed = simulator.seed
    try:
        for run in range(n_simulations):
            simulator.seed = seed + run
            payments = simulator.simulate()
            triangle = simulator.triangle(payments)
            outstanding = float(simulator.true_ultimate(payments).sum()
                                - triangle.latest_diagonal().sum())
            for name, method in methods.items():
                try:
                    result = method(triangle)
                    reserve = float(getattr(result, "total_ibnr", result))
                except (ValueError, RuntimeError, np.linalg.LinAlgError):
                    failed[name] += 1
                    continue
                errors[name].append(reserve / outstanding - 1)
    finally:
        simulator.seed = original_seed

    rows = []
    for name in methods:
        e = np.asarray(errors[name])
        if len(e) == 0:
            rows.append({"method": name, "mean_error": np.nan, "rmse": np.nan,
                         "worst_under": np.nan, "worst_over": np.nan, "failed": failed[name]})
            continue
        rows.append({"method": name, "mean_error": e.mean(), "rmse": np.sqrt(np.mean(e ** 2)),
                     "worst_under": e.min(), "worst_over": e.max(), "failed": failed[name]})
    return pd.DataFrame(rows).set_index("method")
