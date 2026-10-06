"""
Competing Risks

Models for lives exposed to more than one cause of leaving a population:
death, withdrawal, retirement, sickness and so on.

**Multiple decrement tables** work in whole years of age. A *dependent*
probability ``(aq)`` is the chance of leaving by a cause while the other
causes are also operating. An *independent* probability ``q`` is what the
chance would be if that cause operated alone. The two are linked here by
assuming each force of decrement is constant over the year of age.

**Multiple state models** work in continuous time with constant transition
intensities between states, for example healthy, sick and dead.
"""

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


def dependent_from_independent(independent: Dict[str, float]) -> Dict[str, float]:
    """
    Dependent probabilities of decrement for one year of age from the
    independent probabilities, with a constant force for each cause.

    Args:
        independent: Independent probability for each cause

    Returns:
        Dependent probability for each cause
    """
    forces = {k: -np.log1p(-q) for k, q in independent.items()}
    total = sum(forces.values())
    if total == 0:
        return {k: 0.0 for k in independent}
    leaving = 1 - np.exp(-total)
    return {k: float(force / total * leaving) for k, force in forces.items()}


def independent_from_dependent(dependent: Dict[str, float]) -> Dict[str, float]:
    """
    Independent probabilities of decrement for one year of age from the
    dependent probabilities, with a constant force for each cause:
    ``q = 1 - (ap) ** ((aq) / (aq total))``.

    Args:
        dependent: Dependent probability for each cause

    Returns:
        Independent probability for each cause
    """
    total = sum(dependent.values())
    if total >= 1:
        raise ValueError("the dependent probabilities must sum to less than 1")
    if total == 0:
        return {k: 0.0 for k in dependent}
    staying = 1 - total
    return {k: float(1 - staying ** (aq / total)) for k, aq in dependent.items()}


class MultipleDecrementTable:
    """
    A multiple decrement table: the number of lives at each age and the
    numbers leaving by each cause during the year of age.
    """

    def __init__(self, ages: Sequence[int], dependent_rates: Dict[str, Sequence[float]],
                 radix: float = 100000.0):
        """
        Args:
            ages: Consecutive integer ages
            dependent_rates: Dependent probability of decrement at each age,
                for each cause
            radix: Number of lives at the first age
        """
        self.ages = np.asarray(ages, dtype=int)
        if np.any(np.diff(self.ages) != 1):
            raise ValueError("ages must be consecutive integers")
        self.causes = list(dependent_rates)
        self.rates = {}
        for cause, values in dependent_rates.items():
            arr = np.asarray(values, dtype=float)
            if arr.shape != self.ages.shape or np.any((arr < 0) | (arr > 1)):
                raise ValueError(f"'{cause}' needs one probability between 0 and 1 per age")
            self.rates[cause] = arr
        self.total_rate = np.sum(list(self.rates.values()), axis=0)
        if np.any(self.total_rate > 1 + 1e-12):
            raise ValueError("the dependent probabilities at an age must not exceed 1 in total")
        self.radix = float(radix)
        self.al = self.radix * np.concatenate(([1.0], np.cumprod(1 - self.total_rate[:-1])))

    @classmethod
    def from_decrements(cls, ages: Sequence[int], lives: Sequence[float],
                        decrements: Dict[str, Sequence[float]]) -> 'MultipleDecrementTable':
        """
        Build a table from the numbers of lives and of decrements at each age.

        Args:
            ages: Consecutive integer ages
            lives: Number of lives at each age, ``(al)x``
            decrements: Number leaving by each cause at each age, ``(ad)x``
        """
        lives = np.asarray(lives, dtype=float)
        rates = {cause: np.asarray(d, dtype=float) / lives for cause, d in decrements.items()}
        return cls(ages, rates, radix=lives[0])

    @classmethod
    def from_independent_rates(cls, ages: Sequence[int],
                               independent_rates: Dict[str, Sequence[float]],
                               radix: float = 100000.0) -> 'MultipleDecrementTable':
        """Build a table from independent probabilities, assuming constant forces."""
        causes = list(independent_rates)
        columns = {c: np.asarray(independent_rates[c], dtype=float) for c in causes}
        dependent = {c: np.zeros(len(ages)) for c in causes}
        for k in range(len(ages)):
            row = dependent_from_independent({c: columns[c][k] for c in causes})
            for c in causes:
                dependent[c][k] = row[c]
        return cls(ages, dependent, radix)

    def _index(self, age: int) -> int:
        if age < self.ages[0] or age > self.ages[-1]:
            raise ValueError(f"age {age} is not in the table")
        return int(age - self.ages[0])

    def decrements(self, cause: str) -> np.ndarray:
        """Number leaving by a cause at each age, ``(ad)x``."""
        return self.al * self.rates[cause]

    def dependent_rate(self, age: int, cause: str) -> float:
        """Dependent probability of leaving by a cause during the year of age."""
        return float(self.rates[cause][self._index(age)])

    def independent_rate(self, age: int, cause: str) -> float:
        """Independent probability for a cause at an age, assuming constant forces."""
        k = self._index(age)
        return independent_from_dependent({c: self.rates[c][k] for c in self.causes})[cause]

    def force(self, age: int, cause: str) -> float:
        """Constant force of decrement for a cause over the year of age."""
        return float(-np.log1p(-self.independent_rate(age, cause)))

    def survival_probability(self, age: int, years: int) -> float:
        """Probability that a life stays in the population for a number of years."""
        k = self._index(age)
        if k + years > len(self.ages):
            return 0.0
        return float(np.prod(1 - self.total_rate[k:k + years]))

    def probability_of_decrement(self, age: int, cause: str, deferred: int = 0,
                                 years: int = 1) -> float:
        """
        Probability that a life now aged ``age`` leaves by a cause during a
        period of ``years`` starting ``deferred`` years from now.
        """
        k = self._index(age)
        start = k + deferred
        counts = self.decrements(cause)[start:start + years]
        return float(counts.sum() / self.al[k])

    def with_scaled_force(self, cause: str, factor: float) -> 'MultipleDecrementTable':
        """
        A new table in which the independent force of one cause is multiplied
        by a factor and the independent forces of the others are unchanged.
        """
        independent = {c: np.zeros(len(self.ages)) for c in self.causes}
        for k, age in enumerate(self.ages):
            row = independent_from_dependent({c: self.rates[c][k] for c in self.causes})
            for c in self.causes:
                q = row[c]
                if c == cause:
                    q = 1 - (1 - q) ** factor
                independent[c][k] = q
        return MultipleDecrementTable.from_independent_rates(self.ages, independent, self.radix)

    def epv_of_decrement_benefit(self, age: int, cause: str, interest: float,
                                 benefit: float = 1.0, years: Optional[int] = None,
                                 timing: float = 0.5) -> float:
        """
        Expected present value of a benefit paid on leaving by a cause.

        Args:
            age: Present age of the life
            cause: Cause on which the benefit is paid
            interest: Effective annual rate of interest
            benefit: Amount paid
            years: Number of years of cover (to the end of the table if omitted)
            timing: When in the year of decrement the benefit is paid: 0.5
                for mid-year (decrements spread over the year), 1 for the
                end of the year
        """
        k = self._index(age)
        end = len(self.ages) if years is None else min(k + years, len(self.ages))
        counts = self.decrements(cause)[k:end]
        times = np.arange(len(counts)) + timing
        return float(benefit * np.sum(counts * (1 + interest) ** -times) / self.al[k])

    def to_dataframe(self) -> pd.DataFrame:
        """Lives and decrements at each age."""
        data = {"age": self.ages, "al": self.al}
        for cause in self.causes:
            data[f"ad_{cause}"] = self.decrements(cause)
        return pd.DataFrame(data)


class MultiStateModel:
    """
    A continuous-time model of movement between states with constant
    transition intensities (a time-homogeneous Markov model).
    """

    def __init__(self, states: Sequence[str], intensities: Dict[tuple, float]):
        """
        Args:
            states: Names of the states, for example ["able", "sick", "dead"]
            intensities: Transition intensity per year for each pair of
                states, as ``{("able", "sick"): 0.05, ...}``. Pairs not given
                have no transition.
        """
        self.states = list(states)
        n = len(self.states)
        self.generator = np.zeros((n, n))
        for (origin, destination), rate in intensities.items():
            if rate < 0:
                raise ValueError("intensities must not be negative")
            i, j = self.states.index(origin), self.states.index(destination)
            if i == j:
                raise ValueError("a state cannot move to itself")
            self.generator[i, j] = rate
        self.generator -= np.diag(self.generator.sum(axis=1))

    def transition_probabilities(self, t: float) -> pd.DataFrame:
        """Probability of being in each state after t years, from each starting state."""
        from scipy.linalg import expm
        matrix = expm(self.generator * t)
        return pd.DataFrame(matrix, index=self.states, columns=self.states)

    def probability(self, origin: str, destination: str, t: float) -> float:
        """Probability of being in ``destination`` after t years, starting in ``origin``."""
        return float(self.transition_probabilities(t).loc[origin, destination])

    def probability_of_staying(self, state: str, t: float) -> float:
        """Probability of remaining in a state continuously for t years."""
        k = self.states.index(state)
        return float(np.exp(self.generator[k, k] * t))

    def epv_annuity(self, start: str, state: str, term: float, force_of_interest: float,
                    rate: float = 1.0) -> float:
        """
        Expected present value of a payment made continuously at ``rate``
        per year while the life is in ``state``, for a life starting in
        ``start``.
        """
        from scipy import integrate
        i, j = self.states.index(start), self.states.index(state)
        from scipy.linalg import expm
        value, _ = integrate.quad(
            lambda t: np.exp(-force_of_interest * t) * expm(self.generator * t)[i, j],
            0, term, limit=200)
        return float(rate * value)

    def epv_transition_benefit(self, start: str, origin: str, destination: str, term: float,
                               force_of_interest: float, benefit: float = 1.0) -> float:
        """
        Expected present value of a benefit paid at the moment of each move
        from ``origin`` to ``destination``, for a life starting in ``start``.
        """
        intensity = self.generator[self.states.index(origin), self.states.index(destination)]
        return benefit * intensity * self.epv_annuity(start, origin, term, force_of_interest)
