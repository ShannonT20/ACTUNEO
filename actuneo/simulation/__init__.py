"""
Simulation Module

Monte Carlo simulation for general insurance.

This module provides tools for:

- Simulating aggregate claims from a frequency and a severity distribution
  (the collective risk model), with risk measures
- Simulating individual claims with payment delays, inflation and changes
  in settlement speed, to produce claims triangles whose true ultimate cost
  is known. These are for testing reserving methods.
- Benchmarking reserving methods against that known cost

The results depend on the random seed. Fix it to reproduce a run.
"""

from .aggregate import simulate_aggregate_claims, AggregateClaims
from .claims import ClaimsSimulator
from .benchmark import benchmark_reserving

__all__ = [
    'simulate_aggregate_claims',
    'AggregateClaims',
    'ClaimsSimulator',
    'benchmark_reserving',
]
