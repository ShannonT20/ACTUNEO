"""
Ask ACTUNEO a question in plain words

:func:`ask` answers two kinds of question.

**A calculation** ("What is the monthly repayment on a loan of 50,000 over
10 years at 8%?") is answered with a worked solution set out like a textbook.
Fixed rules read the numbers out of the sentence and recognise the
calculation, ACTUNEO does the arithmetic, and the working is written from
templates. The language model never does arithmetic and never writes a step.

**A question in words** ("What is the premium allocation approach?", "What
does ACTUNEO do for pensions?", "What is IPEC?") is answered by the language
model, speaking as an actuary. It is given the passages of ACTUNEO's own
documentation and docstrings that match the question, found by
:mod:`actuneo.ai.knowledge`, and uses them together with its own knowledge.
The passages used are listed with the answer. Without a model, the passages
themselves are returned.

Also
----
- A question in several parts is split and each part answered in turn.
- Your own data can be passed with ``data=``: a claims triangle, a mortality
  table (columns ``age`` and ``qx``), or a list of yearly cash flows.
- Mortality questions use the Zimbabwe 2023 tables unless the question names
  another basis ("South African", "AM92") or a table is passed.
- If something needed is missing, the answer says what, instead of guessing.

How good the answers in words are depends on the model installed. A small
model states wrong things with confidence. Calculations do not depend on it.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import knowledge
from .solutions import Solution, percent
from .solvers import SOLVERS, Param, Solver

RULE_SCORE = 2          # weight of matched phrases needed to choose without the model
DATA_BOOST = 3          # added to a calculation that can use the data passed

#: How the model is told to speak when it answers in words
ACTUARY_PROMPT = (
    "You are the assistant of ACTUNEO, an open-source actuarial Python library built for "
    "Africa, starting with Zimbabwe. You answer as a qualified actuary would explain to a "
    "student or a colleague: precise, practical, using the terms actuaries use (expected "
    "present value, reserve, funding level, development factor, risk adjustment) and tying "
    "the idea to how it is used in insurance, pensions or investment.\n"
    "Rules:\n"
    "- Use the numbered passages from ACTUNEO's documentation when they bear on the question, "
    "and name ACTUNEO's functions only if they appear there.\n"
    "- For general actuarial ideas, answer from your own knowledge of actuarial science.\n"
    "- Never invent figures, table values, regulations, section numbers or sources. If you "
    "are not sure, say so plainly.\n"
    "- Do not do arithmetic. For a calculation, tell the user to ask it with the numbers and "
    "ACTUNEO will work it out.\n"
    "- Give the answer directly, in at most three short paragraphs. No preamble."
)

# ----------------------------------------------------------------------
# Reading numbers
# ----------------------------------------------------------------------
_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
          "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
          "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
          "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}
_FRACTIONS = {"half": 0.5, "quarter": 0.25}
_NUMBER_WORD = "|".join(list(_UNITS) + list(_TENS))
_PERCENT_WORDS = re.compile(
    r"\b((?:(?:%s)[\s-]+){1,3})((?:and\s+)?(?:a\s+)?(?:half|quarter)\s+|and\s+three\s+quarters\s+|"
    r"point\s+(?:%s)\s+)?(?:percent|per\s+cent)\b" % (_NUMBER_WORD, _NUMBER_WORD), flags=re.I)


def _spoken(match: "re.Match") -> str:
    total = 0.0
    for word in re.split(r"[\s-]+", match.group(1).strip().lower()):
        total += _UNITS.get(word, 0) + _TENS.get(word, 0)
    extra = (match.group(2) or "").lower()
    if "three quarters" in extra:
        total += 0.75
    elif "point" in extra:
        total += _UNITS.get(extra.split()[1], 0) / 10
    else:
        total += sum(value for word, value in _FRACTIONS.items() if word in extra)
    return f"{total:g}%"


def normalise(text: str) -> str:
    """
    Rewrite the numbers in a question in one plain form: no thousands
    separators, "120k" and "5 million" written out, and percentages given in
    words ("eight and a half percent") as digits.
    """
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)
    text = _PERCENT_WORDS.sub(_spoken, text)
    text = re.sub(r"(\d)\s*(?:percent|per\s+cent|pct)\b", r"\1%", text, flags=re.I)

    def scale(match):
        factor = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6,
                  "bn": 1e9, "billion": 1e9}[match.group(2).lower()]
        return f"{float(match.group(1)) * factor:.10g}"
    text = re.sub(r"(\d+(?:\.\d+)?)\s+(thousand|million|billion|mn|bn)\b", scale, text, flags=re.I)
    text = re.sub(r"(\d+(?:\.\d+)?)(k|m|mn|bn)\b", scale, text, flags=re.I)
    return text


@dataclass
class _Token:
    value: float
    is_percent: bool
    start: int
    end: int
    used: bool = False


_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(\s?%)?")
_AGE_PATTERNS = (
    r"\baged?\s+(\d{1,3})\b", r"\bage\s+of\s+(\d{1,3})\b",
    r"(?<![\w.])(\d{1,3})[\s-]*(?:years?|yrs?)[\s-]*old",
    r"\b(?:i\s+am|i'm|im)\s+(\d{1,3})\b",
    r"\b(?:man|woman|lady|male|female|person|someone|member|life|client|widow|widower|teacher|"
    r"employee|pensioner)\s+(?:of|aged)\s+(\d{1,3})\b",
    r"\b(?:from|at)\s+age\s+(\d{1,3})\b",
)
_YEARS_PATTERN = r"(?<![\w.])(\d{1,3})[\s-]*(?:years?|yrs?)\b(?![\s-]*old)"
_UNTIL_AGE = (r"(?:alive\s+at|live[sd]?\s+(?:to|until)|surviv\w*\s+(?:to|until)|reach\w*|until|"
              r"to\s+age|at\s+age)\s+(?:age\s+)?(\d{2,3})\b")
_PER_MONTH = r"^\s*(?:(?:a|per|each|every)\s+month|/\s*month|monthly|pm\b|p\.m\.)"


_LABEL_BEFORE = re.compile(r"(?:\bifrs|\bias|\bsi|\bsection|\bchapter|\bcap\.?|\bno\.?|\bexample|"
                           r"\bpart|\bquestion)\s*$", flags=re.I)


def _tokens(text: str) -> List[_Token]:
    """The numbers in a question, leaving out those that are part of a name such as IFRS 17."""
    return [_Token(float(m.group(1)), bool(m.group(2)), m.start(1), m.end())
            for m in _NUMBER.finditer(text) if not _LABEL_BEFORE.search(text[:m.start(1)])]


def has_quantities(text: str) -> bool:
    """True if the question gives at least one number to calculate with."""
    return bool(_tokens(text.lower()))


_WANTS_A_FIGURE = re.compile(r"\b(my|our|mine|i|me|we)\b|^\s*(value|calculate|compute|price|"
                             r"find|estimate|work\s+out|project|convert)\b", flags=re.I)


def _mark(tokens: List[_Token], position: int) -> Optional[_Token]:
    for token in tokens:
        if token.start == position:
            token.used = True
            return token
    return None


def extract_arguments(solver: Solver, text: str) -> Tuple[Dict[str, Any], List[str], List[Param]]:
    """
    Read a solver's arguments out of a question by fixed rules.

    Args:
        solver: The calculation
        text: The question, already passed through :func:`normalise`

    Returns:
        The arguments found, notes on anything that was worked out rather
        than read directly, and the required parameters still missing
    """
    low = text.lower()
    tokens = _tokens(low)
    found: Dict[str, Any] = {}
    notes: List[str] = []

    for param in solver.params:
        if param.kind != "choice":
            continue
        hits = []
        for value, words in param.choices.items():
            for word in words:
                match = re.search(r"(?<![\w])" + re.escape(word) + r"(?![\w])", low)
                if match:
                    hits.append((match.start(), value))
        if hits:
            hits.sort(key=lambda hit: hit[0])
            found[param.name] = hits[0][1]

    def read(match: "re.Match", param: Param) -> float:
        token = _mark(tokens, match.start(1))
        value = float(match.group(1))
        if param.kind == "rate":
            value /= 100
        elif param.monthly_to_annual and token and re.match(_PER_MONTH, low[token.end:]):
            notes.append(f"{param.label}: {value:g} a month x 12 = {value * 12:g} a year")
            value *= 12
        return value

    for param in solver.params:
        if param.kind == "choice":
            continue
        for cue in param.cues:
            match = re.search(cue, low)
            if match and not _LABEL_BEFORE.search(low[:match.start(1)]):
                found[param.name] = read(match, param)
                break

    first_age = next((p for p in solver.params if p.kind == "age"), None)
    if first_age and first_age.name not in found:
        for pattern in _AGE_PATTERNS:
            match = re.search(pattern, low)
            if match and not any(t.start == match.start(1) and t.used for t in tokens):
                found[first_age.name] = read(match, first_age)
                break

    for param in solver.params:
        if param.name in found or param.kind == "choice":
            continue
        if param.kind == "years":
            for match in re.finditer(_YEARS_PATTERN, low):
                token = next((t for t in tokens if t.start == match.start(1)), None)
                if token and not token.used:
                    found[param.name] = read(match, param)
                    break
            if param.name not in found and first_age and first_age.name in found:
                match = re.search(_UNTIL_AGE, low)
                age = found[first_age.name]
                if match and float(match.group(1)) > age:
                    target = float(match.group(1))
                    _mark(tokens, match.start(1))
                    found[param.name] = target - age
                    notes.append(f"{param.label}: age {target:g} - age {age:g} = "
                                 f"{target - age:g} years")
        elif param.kind == "rate":
            token = next((t for t in tokens if t.is_percent and not t.used), None)
            if token:
                token.used = True
                found[param.name] = token.value / 100
        elif param.kind == "amount" and not param.cue_only:
            token = next((t for t in tokens if not t.is_percent and not t.used), None)
            if token:
                token.used = True
                value = token.value
                if param.monthly_to_annual and re.match(_PER_MONTH, low[token.end:]):
                    notes.append(f"{param.label}: {value:g} a month x 12 = {value * 12:g} a year")
                    value *= 12
                found[param.name] = value

    _special_cases(solver, low, found, notes)
    for name in list(found):
        param = next(p for p in solver.params if p.name == name)
        if param.kind in ("age", "years") and float(found[name]).is_integer():
            found[name] = int(found[name])
    missing = [p for p in solver.params if p.required and p.name not in found]
    return found, notes, missing


def _special_cases(solver: Solver, low: str, found: Dict[str, Any], notes: List[str]) -> None:
    """Wording that changes a value already read."""
    if solver.name == "retirement_fund" and "contribution_rate" in found:
        match = re.search(r"employer\s+(?:adds?|contributes?|pays?|puts?\s+in|matches\s+with)\s+"
                          r"(?:another\s+|a\s+further\s+)?(\d+(?:\.\d+)?)\s?%", low)
        if match:
            extra = float(match.group(1)) / 100
            own = found["contribution_rate"]
            if abs(own - extra) > 1e-12 or low.count("%") > 2:
                found["contribution_rate"] = own + extra
                notes.append(f"the total contribution rate: {percent(own)} from the member + "
                             f"{percent(extra)} from the employer = {percent(own + extra)}")
    if solver.name == "bond_price" and "years" in found:
        term = re.search(r"(\d{1,3})[\s-]*years?[\s-]*(?:term|bond|maturity)", low)
        ago = re.search(r"issued\s+(\d{1,3})\s+years?\s+ago", low)
        if term and ago and float(term.group(1)) > float(ago.group(1)):
            original, elapsed = float(term.group(1)), float(ago.group(1))
            found["years"] = original - elapsed
            notes.append(f"the term of the bond in years: {original:g}-year term - {elapsed:g} "
                         f"years since issue = {original - elapsed:g} years left")


# ----------------------------------------------------------------------
# Choosing the calculation
# ----------------------------------------------------------------------
def choice_values(solver: Solver, text: str) -> Dict[str, List[Any]]:
    """Every value of each choice that the question mentions, in the order mentioned."""
    low = text.lower()
    mentioned: Dict[str, List[Any]] = {}
    for param in solver.params:
        if param.kind != "choice":
            continue
        hits = []
        for value, words in param.choices.items():
            for word in words:
                match = re.search(r"(?<![\w])" + re.escape(word) + r"(?![\w])", low)
                if match:
                    hits.append((match.start(), value))
                    break
        mentioned[param.name] = [value for _, value in sorted(hits, key=lambda hit: hit[0])]
    return mentioned


def rank_solvers(text: str, data_kinds: Tuple[str, ...] = ()) -> List[Tuple[int, Solver]]:
    """
    Solvers whose key phrases appear in the question, best match first. A
    solver that can use the kind of data passed with the question is favoured.
    """
    low = text.lower()
    ranked = []
    for solver in SOLVERS.values():
        score = sum(weight for pattern, weight in solver.keywords if re.search(pattern, low))
        if score > 0 and any(kind in solver.uses for kind in data_kinds):
            score += DATA_BOOST
        if score > 0:
            ranked.append((score, solver))
    ranked.sort(key=lambda item: -item[0])
    return ranked


_GREETING = re.compile(r"^\s*(hi|hello|hey|good\s+(morning|afternoon|evening|day)|thanks|"
                       r"thank\s+you|bye|goodbye)\b[^?]*$", flags=re.I)
_CASHFLOW_LIST = re.compile(r"cash\s?flows?(?:\s+(?:of|are|is|were))?\s*:?\s*"
                            r"((?:-?\$?\d+(?:\.\d+)?\s*(?:,|;|\band\b)?\s*){2,})", flags=re.I)


# ----------------------------------------------------------------------
# The user's own data
# ----------------------------------------------------------------------
def read_data(data: Any) -> Dict[str, Any]:
    """
    Work out what the data passed with a question is.

    Args:
        data: One of: a :class:`~actuneo.loss_reserving.Triangle`, a
            :class:`~actuneo.mortality.MortalityTable`, a DataFrame, a list,
            the path of a CSV or Excel file, or a mapping with any of the
            keys ``triangle``, ``table`` and ``cashflows``.

            - a table with a ``qx`` column is a mortality table (ages in an
              ``age`` column or the index);
            - a single row or column of numbers is a list of yearly cash
              flows, the first one now;
            - any other table is a claims triangle, origin years in rows and
              development years in columns, cumulative, blank where not yet
              observed.

    Returns:
        A mapping with whichever of ``triangle``, ``table`` and
        ``cashflows`` was recognised
    """
    from ..loss_reserving import Triangle
    from ..mortality import MortalityTable
    if data is None:
        return {}
    if isinstance(data, dict) and set(data) <= {"triangle", "table", "cashflows"}:
        out: Dict[str, Any] = {}
        for value in data.values():
            out.update(read_data(value))
        return out
    if isinstance(data, Triangle):
        return {"triangle": data}
    if isinstance(data, MortalityTable):
        return {"table": data}
    if isinstance(data, (str, Path)):
        path = Path(data).expanduser()
        if not path.exists():
            raise ValueError(f"The data file {path} was not found.")
        frame = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") \
            else pd.read_csv(path)
        return read_data(frame)
    if isinstance(data, pd.Series):
        return {"cashflows": [float(value) for value in data.dropna()]}
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
        lowered = {str(column).strip().lower(): column for column in frame.columns}
        if "qx" in lowered:
            ages = frame[lowered["age"]] if "age" in lowered else frame.index
            return {"table": MortalityTable(list(ages), list(frame[lowered["qx"]]),
                                            name="your table")}
        first = frame.columns[0]
        labels = frame[first]
        named = str(first).lower().startswith(("origin", "year", "accident", "unnamed",
                                               "period", "underwriting"))
        years = pd.api.types.is_numeric_dtype(labels) and labels.between(1900, 2100).all() \
            and (labels.diff().dropna() == 1).all()
        if frame.shape[1] > 1 and (named or years
                                   or not pd.api.types.is_numeric_dtype(labels)):
            frame = frame.set_index(first)
        frame = frame.apply(pd.to_numeric, errors="coerce")
        if 1 in frame.shape:
            return {"cashflows": [float(value) for value in frame.to_numpy().ravel()
                                  if not np.isnan(value)]}
        return {"triangle": Triangle(frame, name="your triangle")}
    rows = list(data)
    if rows and all(isinstance(row, (list, tuple, np.ndarray)) for row in rows):
        width = max(len(row) for row in rows)
        grid = np.full((len(rows), width), np.nan)
        for index, row in enumerate(rows):
            grid[index, :len(row)] = [np.nan if value is None else value for value in row]
        return {"triangle": Triangle(grid, name="your triangle")}
    return {"cashflows": [float(value) for value in rows]}


# ----------------------------------------------------------------------
# Splitting a question into parts
# ----------------------------------------------------------------------
_LETTERED = re.compile(r"(?:(?<=\s)|^)\(?(?:[a-h]|i{1,3}|iv|v|vi{1,3}|\d)\)\s+", flags=re.I)
_NUMBERED = re.compile(r"(?:(?<=\s)|^)\d\.\s+(?=[A-Za-z])")
_AND_ASK = re.compile(r"[,;]?\s+(?:and|also|then)\s+(?=(?:what|how|find|calculate|estimate|"
                      r"value|price|work\s+out|give)\b)", flags=re.I)


def split_parts(question: str) -> Tuple[str, List[str]]:
    """
    Split a question into its parts.

    Recognises parts labelled (a), (b) or 1., 2.; two or more sentences that
    each end in a question mark; and "... and what is ...".

    Returns:
        The opening statement that applies to every part (which may be
        empty), and the parts. A question with one part comes back whole.
    """
    text = question.strip()
    for pattern in (_LETTERED, _NUMBERED):
        marks = list(pattern.finditer(text))
        if len(marks) >= 2:
            parts = [text[m.end():(marks[i + 1].start() if i + 1 < len(marks) else len(text))]
                     .strip(" ;") for i, m in enumerate(marks)]
            return text[:marks[0].start()].strip(), [part for part in parts if part]
    sentences = [piece.strip() for piece in re.split(r"(?<=[.?!])\s+(?=[A-Z(])", text)
                 if piece.strip()]
    asking = [index for index, sentence in enumerate(sentences) if sentence.endswith("?")]
    if len(asking) >= 2:
        preamble = " ".join(sentences[:asking[0]])
        parts, current = [], []
        for sentence in sentences[asking[0]:]:
            if current and current[-1].endswith("?"):
                if sentence.endswith("?"):
                    parts.append(" ".join(current))
                    current = [sentence]
                else:           # a statement after a question belongs to that question
                    current.append(sentence)
            else:
                current.append(sentence)
        parts.append(" ".join(current))
        return preamble, parts
    pieces = [piece.strip() for piece in _AND_ASK.split(text) if piece.strip()]
    if len(pieces) >= 2:
        return "", pieces
    return "", [text]


# ----------------------------------------------------------------------
# The answer
# ----------------------------------------------------------------------
@dataclass
class Answer:
    """
    The reply to a question.

    Attributes:
        question: The question as asked
        kind: "solution" (calculated by ACTUNEO), "explanation" (written by
            the language model), "reference" (passages of ACTUNEO's
            documentation, when there is no model), "need_more" (something
            needed was not in the question), "parts" (a question in several
            parts), "greeting" or "unknown"
        text: The words of the reply, for anything other than a solution
        solution: The worked solution
        solver: Name of the calculation used
        arguments: The values the calculation was given
        notes: How the question was read, including anything worked out
        sources: The passages of ACTUNEO's documentation the reply drew on
        parts: The answers to each part of a question in several parts
        used_model: Whether the language model had a part in the reply
    """
    question: str
    kind: str
    text: str = ""
    solution: Optional[Solution] = None
    solver: Optional[str] = None
    arguments: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    sources: List[knowledge.Passage] = field(default_factory=list)
    parts: List["Answer"] = field(default_factory=list)
    used_model: bool = False

    @property
    def solved(self) -> bool:
        """True if a calculation was carried out (in every part, if several)."""
        if self.parts:
            return all(part.solved for part in self.parts)
        return self.solution is not None

    @property
    def value(self) -> Optional[float]:
        """The numerical answer, if there is one."""
        return self.solution.value if self.solution else None

    def _source_lines(self) -> List[str]:
        seen, lines = set(), []
        for passage in self.sources:
            label = f"{passage.title} ({passage.source})" if passage.title != passage.source \
                else passage.title
            if label not in seen:
                seen.add(label)
                lines.append(label)
        return lines

    def explain(self) -> str:
        """The answer as plain text."""
        if self.parts:
            out = [f"Question: {self.question}", ""]
            for index, part in enumerate(self.parts, start=1):
                out.append(f"----- Part {index} -----")
                out.append(part.explain())
                out.append("")
            return "\n".join(out).rstrip()
        out = [f"Question: {self.question}", ""]
        if self.notes:
            out.append("How the question was read")
            out.extend(f"  - {note}" for note in self.notes)
            out.append("")
        out.append(self.solution.explain() if self.solution else self.text)
        if self.kind == "explanation":
            out.extend(["", "This was written by the language model, not calculated by ACTUNEO. "
                            "A model can be wrong: check anything you will rely on."])
        if self.sources:
            out.extend(["", "Passages of ACTUNEO's documentation it drew on"])
            out.extend(f"  - {line}" for line in self._source_lines())
        return "\n".join(out)

    def to_markdown(self) -> str:
        """The answer as Markdown."""
        if self.parts:
            out = [f"> {self.question}", ""]
            for index, part in enumerate(self.parts, start=1):
                out.extend([f"---", f"**Part {index}**", "", part.to_markdown(), ""])
            return "\n".join(out)
        out = [f"> {self.question}", ""]
        if self.notes:
            out.extend(["*How the question was read:*", ""])
            out.extend(f"- {note}" for note in self.notes)
            out.append("")
        out.append(self.solution.to_markdown() if self.solution else self.text)
        if self.kind == "explanation":
            out.extend(["", "*Written by the language model, not calculated by ACTUNEO. A model "
                            "can be wrong: check anything you will rely on.*"])
        if self.sources:
            out.extend(["", "*Passages of ACTUNEO's documentation it drew on:*", ""])
            out.extend(f"- {line}" for line in self._source_lines())
        return "\n".join(out)

    def _repr_markdown_(self) -> str:
        return self.to_markdown()

    def __repr__(self) -> str:
        return self.explain()


# ----------------------------------------------------------------------
# The language model
# ----------------------------------------------------------------------
def _backend(use_model: bool):
    if not use_model:
        return None
    try:
        from .setup import get_backend
        return get_backend()
    except Exception:           # nothing installed or configured
        return None


def _choose_with_model(backend, text: str, candidates: List[Solver]):
    """Let the model choose between the candidates, all of them shown to it."""
    limit = getattr(backend, "max_tools", None)
    if limit is not None:
        backend.max_tools = max(limit, len(candidates))
    try:
        return backend.call_tools(text, [solver.tool() for solver in candidates])
    finally:
        if limit is not None:
            backend.max_tools = limit


def _grounded(value: Any, text: str) -> bool:
    """True if a number the model gave can be traced to numbers in the question."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    numbers = [token.value for token in _tokens(text.lower())]
    candidates = set()
    for a in numbers:
        candidates.update((a, a / 100, a * 12))
        for b in numbers:
            candidates.update((a - b, (a + b) / 100))
    return any(abs(value - c) <= 1e-9 * max(1.0, abs(c)) for c in candidates)


def _what_i_can_do() -> str:
    lines = ["I can work these out, with the working shown:"]
    lines.extend(f"  - {solver.title}" for solver in SOLVERS.values())
    lines.extend(["", 'For example: "' + next(iter(SOLVERS.values())).example + '"', "",
                  "I can also answer questions in words about actuarial work and about ACTUNEO "
                  "itself, if a language model is installed (actuneo-ai setup)."])
    return "\n".join(lines)


def _passages_for(question: str) -> List[knowledge.Passage]:
    found = [passage for _, passage in knowledge.retrieve(question, limit=4)]
    if not found and re.search(r"\bactuneo\b|this (package|library)|the (package|library)",
                               question, flags=re.I):
        wanted = ("Vision and Objective", "Status", "Implemented Modules",
                  "ACTUNEO: African Actuarial Python Library")
        found = [p for p in knowledge.corpus() if p.source == "README.md" and p.title in wanted][:3]
    return found


def _answer_in_words(question: str, backend, notes: List[str]) -> Answer:
    """Answer from the model's knowledge and ACTUNEO's own documentation."""
    passages = _passages_for(question)
    if backend is not None:
        context = "\n".join(f"[{index}] {passage.title}: {passage.text}"
                            for index, passage in enumerate(passages, start=1))
        prompt = (f"Passages from ACTUNEO's documentation:\n{context}\n\n" if context else
                  "No passage of ACTUNEO's documentation matches this question, so answer from "
                  "your own actuarial knowledge.\n\n") + f"Question: {question}"
        try:
            reply = backend.generate(prompt, system=ACTUARY_PROMPT, max_tokens=450,
                                     temperature=0.2)
            if reply:
                return Answer(question, "explanation", reply, notes=notes, sources=passages,
                              used_model=True)
        except Exception as problem:
            notes = notes + [f"The language model could not be used ({str(problem)[:80]})."]
    if passages:
        body = ["No language model is installed, so here is what ACTUNEO's own documentation "
                "says. Install a model (actuneo-ai setup) for an answer in words.", ""]
        for passage in passages[:2]:
            body.extend([f"{passage.title}:", passage.text, ""])
        return Answer(question, "reference", "\n".join(body).rstrip(), notes=notes,
                      sources=passages[:2])
    return Answer(question, "unknown",
                  "I could not tell which calculation you want, and answering a general "
                  "question needs a language model (actuneo-ai setup).\n\n" + _what_i_can_do(),
                  notes=notes)


# ----------------------------------------------------------------------
# Answering
# ----------------------------------------------------------------------
def _ask_one(question: str, context: str, data: Dict[str, Any], backend) -> Answer:
    """Answer one question, or one part of a question."""
    if _GREETING.match(question):
        return Answer(question, "greeting",
                      "Hello. Ask me an actuarial question in plain words.\n\n" + _what_i_can_do())
    text = normalise(question.strip())
    full_text = normalise(f"{context} {question}".strip()) if context else text
    ranked = rank_solvers(text, tuple(data))
    notes: List[str] = []
    used_model = False
    model_arguments: Dict[str, Any] = {}

    solver: Optional[Solver] = None
    clear = bool(ranked) and ranked[0][0] >= RULE_SCORE \
        and (len(ranked) == 1 or ranked[0][0] > ranked[1][0])
    has_numbers = has_quantities(full_text) or bool(data)
    if clear:
        solver = ranked[0][1]
    elif backend is not None and has_numbers:
        candidates = [s for _, s in ranked[:6]] or list(SOLVERS.values())
        try:
            call = _choose_with_model(backend, text, candidates)
            if call.name in SOLVERS:
                solver, model_arguments, used_model = SOLVERS[call.name], call.arguments, True
        except Exception as problem:
            notes.append(f"The language model could not be used ({str(problem)[:80]}).")

    if solver is None:
        return _answer_in_words(question, backend, notes)

    found, read_notes, missing = extract_arguments(solver, text)
    if context:
        earlier, earlier_notes, _ = extract_arguments(solver, full_text)
        taken = [p for p in solver.params if p.name not in found and p.name in earlier]
        for param in taken:
            found[param.name] = earlier[param.name]
        if taken:
            read_notes.extend(earlier_notes)
            read_notes.append("Taken from the opening of the question: "
                              + ", ".join(p.label for p in taken) + ".")
        missing = [p for p in solver.params if p.required and p.name not in found]

    local = dict(data)
    if "triangle" in solver.uses and local.get("triangle") is not None:
        mentioned = choice_values(solver, text).get("triangle", [])
        if not [v for v in mentioned if v != "own"]:
            found["triangle"] = "own"
            read_notes.append("The triangle is the data you passed.")
    if "table" in solver.uses and local.get("table") is not None:
        mentioned = choice_values(solver, full_text).get("basis", [])
        if not [v for v in mentioned if v != "own"]:
            found["basis"] = "own"
            read_notes.append("Mortality is from the table you passed.")
    if "cashflows" in solver.uses and local.get("cashflows") is None:
        listed = _CASHFLOW_LIST.search(text)
        if listed:
            local["cashflows"] = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", listed.group(1))]
            # The amounts in the list are cash flows, not an interest rate
            rate = re.search(r"(\d+(?:\.\d+)?)\s?%", text)
            if rate:
                found["interest_rate"] = float(rate.group(1)) / 100
            else:
                found.pop("interest_rate", None)
    missing = [p for p in solver.params if p.required and p.name not in found]

    if missing and backend is not None and has_numbers:
        try:
            if not model_arguments:
                model_arguments = backend.call_tools(text, [solver.tool()],
                                                     allow_none=False).arguments
            for param in list(missing):
                value = model_arguments.get(param.name)
                if param.kind != "choice" and _grounded(value, full_text):
                    found[param.name] = int(value) if param.kind in ("age", "years") else value
                    missing.remove(param)
                    used_model = True
                    read_notes.append(f"{param.label}: {value:g} was picked out by the "
                                      "language model. Check it.")
        except Exception as problem:
            notes.append(f"The language model could not be used ({str(problem)[:80]}).")

    if missing:
        if not has_quantities(full_text) and not data and not _WANTS_A_FIGURE.search(text):
            return _answer_in_words(question, backend, notes)
        needed = ", ".join(param.label for param in missing)
        return Answer(question, "need_more",
                      f"This looks like: {solver.title}.\nTo work it out I also need: {needed}."
                      f'\n\nFor example: "{solver.example}"',
                      solver=solver.name, arguments=found, notes=read_notes,
                      used_model=used_model)

    for param in solver.params:
        if param.name in found:
            value = found[param.name]
            if param.kind != "choice" and value is not None and (
                    (param.low is not None and value < param.low)
                    or (param.high is not None and value > param.high)):
                shown = percent(value) if param.kind == "rate" else f"{value:g}"
                return Answer(question, "need_more",
                              f"I read {param.label} as {shown}, which is outside the range I "
                              "can work with. Please check the question.",
                              solver=solver.name, arguments=found, notes=read_notes)
        elif not param.required:
            found[param.name] = param.default
    how = ("chosen by the language model from your wording. Check that it is the one you "
           "meant." if used_model and not clear else "recognised from your wording.")
    notes.append(f"Calculation: {solver.title}, {how}")
    notes.extend(read_notes)

    # One answer for each value of a choice the question names more than once
    variants = [dict(found)]
    for name, values in choice_values(solver, text).items():
        param = next(p for p in solver.params if p.name == name)
        values = [v for v in values if v != "own"]
        if param.required and len(values) > 1:
            variants = [dict(found, **{name: value}) for value in values]
    answers = []
    for arguments in variants:
        try:
            solution = solver.solve(**arguments, data=local) if solver.uses \
                else solver.solve(**arguments)
        except (ValueError, KeyError) as problem:
            answers.append(Answer(question, "need_more", str(problem), solver=solver.name,
                                  arguments=arguments, notes=notes, used_model=used_model))
            continue
        answers.append(Answer(question, "solution", solution=solution, solver=solver.name,
                              arguments=arguments, notes=notes, used_model=used_model))
    if len(answers) == 1:
        return answers[0]
    return Answer(question, "parts", parts=answers, used_model=used_model)


def ask(question: str, data: Any = None, use_model: bool = True) -> Answer:
    """
    Answer a question: a calculation with its working, or a question in words.

    Args:
        question: The question in plain words. It may have several parts.
        data: Your own data for the calculation: a claims triangle, a
            mortality table (columns ``age`` and ``qx``), a list of yearly
            cash flows, or the path of a CSV or Excel file holding one of
            these. See :func:`read_data`.
        use_model: Let the installed language model help. With False, only
            the fixed rules and ACTUNEO's documentation are used.

    Returns:
        An :class:`Answer`. In a notebook it displays itself; elsewhere
        ``print(answer)`` gives the text, ``answer.value`` the number, and
        ``answer.parts`` the answers to each part.
    """
    if not question or not question.strip():
        return Answer(question, "unknown", _what_i_can_do())
    try:
        own = read_data(data)
    except (ValueError, TypeError, KeyError) as problem:
        return Answer(question, "need_more",
                      f"The data could not be read: {problem}\nPass a triangle (rows are origin "
                      "years, columns development years), a mortality table with columns age "
                      "and qx, or a list of yearly cash flows.")
    backend = _backend(use_model)
    preamble, parts = split_parts(question)
    if len(parts) == 1:
        return _ask_one(parts[0] if not preamble else question, "", own, backend)
    answers = [_ask_one(part, preamble, own, backend) for part in parts]
    flattened: List[Answer] = []
    for answer in answers:
        flattened.extend(answer.parts or [answer])
    return Answer(question, "parts", parts=flattened,
                  used_model=any(part.used_model for part in flattened))


def topics() -> pd.DataFrame:
    """The calculations :func:`ask` can do, each with an example question."""
    return pd.DataFrame([{"calculation": solver.title, "example": solver.example}
                         for solver in SOLVERS.values()])
