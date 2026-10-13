"""
Worked solutions

A :class:`Solution` is an answer set out the way a textbook sets it out: what
was given, the working step by step with the formula, the substitution and
the result of each step, the answer, and the assumptions behind it.

Every figure in a solution is calculated by ACTUNEO. The wording comes from
fixed templates. No language model writes any part of it.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


def money(value: float, places: int = 2) -> str:
    """An amount with thousands separators."""
    return f"{value:,.{places}f}"


def percent(rate: float) -> str:
    """A rate as a percentage without trailing zeros, for example 8.5%."""
    return f"{rate * 100:.6g}%"


def number(value: float, places: int = 6) -> str:
    """A factor or probability to a fixed number of places."""
    return f"{value:.{places}f}"


@dataclass
class Step:
    """One step of the working: a title and its lines (formula, substitution, result)."""
    title: str
    lines: List[str] = field(default_factory=list)


@dataclass
class Solution:
    """
    A worked answer.

    Attributes:
        topic: What was calculated
        given: The quantities taken from the question, by name
        steps: The working, in order
        answer: The answer in words
        value: The answer as a number
        assumptions: The basis and conventions the answer rests on
        check: A way to see that the answer is reasonable
    """
    topic: str
    given: Dict[str, str]
    steps: List[Step]
    answer: str
    value: float
    assumptions: List[str] = field(default_factory=list)
    check: Optional[str] = None

    def explain(self) -> str:
        """The solution as plain text, for a terminal."""
        out = [self.topic, "=" * len(self.topic), "", "Given"]
        width = max(len(name) for name in self.given) if self.given else 0
        out.extend(f"  {name:<{width}}  {value}" for name, value in self.given.items())
        out.extend(["", "Working"])
        for index, step in enumerate(self.steps, start=1):
            out.append(f"  {index}. {step.title}")
            out.extend(f"       {line}" for line in step.lines)
            out.append("")
        out.extend(["Answer", f"  {self.answer}"])
        if self.check:
            out.extend(["", "Check", f"  {self.check}"])
        if self.assumptions:
            out.extend(["", "Assumptions"])
            out.extend(f"  - {item}" for item in self.assumptions)
        return "\n".join(out)

    def to_markdown(self) -> str:
        """The solution as Markdown, for a notebook."""
        out = [f"### {self.topic}", "", "**Given**", ""]
        out.extend(f"- {name}: `{value}`" for name, value in self.given.items())
        out.extend(["", "**Working**", ""])
        for index, step in enumerate(self.steps, start=1):
            out.append(f"{index}. **{step.title}**")
            out.append("")
            out.append("   ```")
            out.extend(f"   {line}" for line in step.lines)
            out.append("   ```")
        out.extend(["", f"**Answer:** {self.answer}"])
        if self.check:
            out.extend(["", f"**Check:** {self.check}"])
        if self.assumptions:
            out.extend(["", "**Assumptions**", ""])
            out.extend(f"- {item}" for item in self.assumptions)
        return "\n".join(out)

    def _repr_markdown_(self) -> str:
        return self.to_markdown()

    def __str__(self) -> str:
        return self.explain()
