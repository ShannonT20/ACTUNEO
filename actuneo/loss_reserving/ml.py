"""
Machine Learning Reserving

A chain-ladder whose development factors are adjusted by a machine learning
model. The model is trained on how far each observed link ratio lies from
the chain-ladder factor of its step, given the development step and the
position of the origin period, so it can let development differ between
older and newer origin periods where the data support it. Where it finds no
pattern, the result is the ordinary chain-ladder.

This is a baseline for experiments, not a recommended reserving method. A
triangle of ten years holds 45 link ratios, which is very little for a
flexible model to learn from, and no measure of uncertainty is produced.
Compare it with the standard methods on simulated data, where the true
answer is known, before relying on it.

Requires scikit-learn (``pip install actuneo[ml]``).

References
----------
- Wüthrich, M.V. (2018). Machine learning in individual claims reserving.
  Scandinavian Actuarial Journal 2018(6), 465-480.
- Kuo, K. (2019). DeepTriangle: a deep learning approach to loss reserving.
  Risks 7(3), 97.
"""

from typing import Optional

import numpy as np
import pandas as pd

from .triangle import Triangle
from .chain_ladder import ChainLadder


class MLChainLadder:
    """
    Chain-ladder with development factors predicted by a regression model.

    Each observed link ratio is a training example. The features are the
    development step and the position of the origin period; the target is
    the logarithm of the link ratio divided by the volume-weighted
    chain-ladder factor of its step; examples are weighted by claims to
    date. Unobserved cells are projected one step at a time with the
    chain-ladder factor multiplied by the predicted adjustment.

    With a model that always predicts zero, this is the ordinary
    chain-ladder.

    Attributes:
        model: The fitted regression model
        factors: Volume-weighted chain-ladder factors the model adjusts
        predicted_factors: Factor used for each projected cell
        full_triangle: Triangle with the unobserved cells projected
        latest, ultimate, ibnr: By origin period
        total_ibnr: Total reserve
    """

    def __init__(self, triangle: Triangle, model=None, random_state: Optional[int] = 0):
        """
        Args:
            triangle: Claims triangle (cumulative or incremental), without
                missing cells
            model: Any scikit-learn regressor whose ``fit`` accepts
                ``sample_weight``. Defaults to a random forest with at least
                three link ratios in each leaf, which smooths heavily. More
                flexible models, such as gradient boosting, fit the noise in
                a triangle of this size and can give absurd reserves.
            random_state: Seed of the default model
        """
        if not isinstance(triangle, Triangle):
            raise TypeError("triangle must be a Triangle")
        if model is None:
            try:
                from sklearn.ensemble import RandomForestRegressor
            except ImportError as error:
                raise ImportError("MLChainLadder needs scikit-learn: "
                                  "pip install actuneo[ml]") from error
            model = RandomForestRegressor(n_estimators=100, min_samples_leaf=3,
                                          random_state=random_state)
        self.triangle = triangle.to_cumulative()
        self.triangle._require_complete("Machine learning reserving")
        cum = self.triangle.values
        n_origin, n_dev = cum.shape

        base = ChainLadder(self.triangle).factors.to_numpy()
        features, target, weight = [], [], []
        for i in range(n_origin):
            for k in range(n_dev - 1):
                if np.isnan(cum[i, k + 1]) or cum[i, k] <= 0 or cum[i, k + 1] <= 0:
                    continue
                features.append(self._features(k, i, n_origin))
                target.append(np.log(cum[i, k + 1] / cum[i, k] / base[k]))
                weight.append(cum[i, k])
        if len(target) < 3:
            raise ValueError("The triangle is too small to train a model")
        self.training = pd.DataFrame(features, columns=["development_step", "origin_position"])
        self.training["log_ratio_to_chain_ladder"] = target
        self.training["weight"] = weight
        self.model = model.fit(np.asarray(features), np.asarray(target),
                               sample_weight=np.asarray(weight))

        full = cum.copy()
        latest_index = self.triangle._latest_index()
        factors = np.full((n_origin, n_dev - 1), np.nan)
        for i, j0 in enumerate(latest_index):
            steps = np.arange(j0, n_dev - 1)
            if len(steps) == 0:
                continue
            x = np.asarray([self._features(k, i, n_origin) for k in steps])
            factors[i, steps] = base[steps] * np.exp(self.model.predict(x))
            for k in steps:
                full[i, k + 1] = full[i, k] * factors[i, k]
        origin_index = pd.Index(self.triangle.origin, name="origin")
        self.factors = pd.Series(base, index=self.triangle._link_labels(), name="factor")
        self.predicted_factors = pd.DataFrame(factors, index=origin_index,
                                              columns=self.triangle._link_labels())
        self.full_triangle = pd.DataFrame(
            full, index=origin_index,
            columns=pd.Index(self.triangle.development, name="development"))
        self.latest = self.triangle.latest_diagonal()
        self.ultimate = pd.Series(full[:, -1], index=origin_index, name="ultimate")
        self.ibnr = (self.ultimate - self.latest).rename("ibnr")

    @staticmethod
    def _features(step: int, origin: int, n_origin: int):
        return [float(step), origin / max(n_origin - 1, 1)]

    @property
    def total_ibnr(self) -> float:
        """Total reserve over all origin periods."""
        return float(self.ibnr.sum())

    def summary(self, total: bool = True) -> pd.DataFrame:
        """Latest, ultimate and reserve by origin period."""
        table = pd.DataFrame({"latest": self.latest, "ultimate": self.ultimate,
                              "ibnr": self.ibnr})
        if total:
            table.loc["Total"] = table.sum()
        return table

    def __repr__(self) -> str:
        return (f"MLChainLadder(model={type(self.model).__name__}, "
                f"total_ibnr={self.total_ibnr:,.0f})")
