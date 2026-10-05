"""
Cape Cod Method

A Bornhuetter-Ferguson projection in which the expected loss ratio is not
supplied but estimated from the triangle itself: claims to date are divided
by the premium "used up" so far, that is the premium weighted by the
proportion of claims expected to have developed.

References
----------
- Buhlmann, H. (1983). Estimation of IBNR reserves by the methods chain
  ladder, Cape Cod and complementary loss ratio. International Summer School.
- Stanard, J.N. (1985). A simulation test of prediction errors of loss
  reserve estimation techniques. Proceedings of the Casualty Actuarial
  Society 72, 124-148.
"""

import numpy as np
import pandas as pd
from typing import Optional, Sequence, Union
from .triangle import Triangle
from .chain_ladder import ChainLadder
from .bornhuetter_ferguson import BornhuetterFerguson


class CapeCod(BornhuetterFerguson):
    """
    Cape Cod projection of a claims triangle.

    With ``f`` the chain-ladder factor to ultimate of each origin period::

        used-up premium     = premium / f
        expected loss ratio = sum(latest claims) / sum(used-up premium)

    and the reserve then follows the Bornhuetter-Ferguson formula with that
    loss ratio. The premium should be on-level (adjusted to a common rate
    level) for the single loss ratio to be meaningful.

    Attributes:
        loss_ratio: Expected loss ratio estimated from the triangle
        used_up_premium: Premium weighted by the proportion developed
    """

    def __init__(self,
                 triangle: Triangle,
                 premium: Sequence[float],
                 average: str = "volume",
                 n_periods: Optional[int] = None,
                 tail: Union[float, bool] = 1.0):
        """
        Fit the Cape Cod method.

        Args:
            triangle: Claims triangle (cumulative or incremental), paid or incurred
            premium: Earned premium (or another exposure measure) for each
                origin period
            average: "volume" or "simple" averaging of the link ratios
            n_periods: Average only the latest n origin periods (None for all)
            tail: Tail factor, or True to estimate it
        """
        chain_ladder = ChainLadder(triangle, average=average, n_periods=n_periods, tail=tail)
        premium = np.asarray(premium, dtype=float)
        if premium.shape != (chain_ladder.triangle.n_origin,):
            raise ValueError(
                f"One premium is needed for each of the {chain_ladder.triangle.n_origin} "
                "origin periods"
            )

        to_ultimate = chain_ladder.cdf.to_numpy()[chain_ladder._latest_idx]
        used_up = premium / to_ultimate
        if used_up.sum() <= 0:
            raise ValueError("The used-up premium must be positive")
        loss_ratio = float(chain_ladder.latest.sum() / used_up.sum())

        super().__init__(triangle, premium=premium, loss_ratio=loss_ratio, average=average,
                         n_periods=n_periods, tail=chain_ladder.tail)
        self.loss_ratio = loss_ratio
        self.used_up_premium = pd.Series(used_up, index=self.latest.index,
                                         name="used_up_premium")
