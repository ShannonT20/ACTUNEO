"""
Ask ACTUNEO a question in plain words

:func:`ask` takes a question such as "What is the monthly repayment on a loan
of 50,000 over 10 years at 8%?" and returns a worked answer set out like a
textbook solution.

How a question is answered
--------------------------
1. **Rules first.** The wording is matched against each calculation's key
   phrases, and the numbers are read out of the sentence by fixed rules
   (ages, terms, percentages, amounts, "120k", "eight and a half percent",
   "750 a month"). Clearly worded questions are answered this way with no
   language model at all.
2. **The model only when needed.** If the wording does not settle which
   calculation is meant, the installed language model chooses between the
   likely ones. If a number cannot be found by the rules, the model is asked
   for it, and its answer is accepted only if that number appears in the
   question.
3. **ACTUNEO calculates** and the working is written from fixed templates.
   The model never does arithmetic and never writes the explanation.

If something needed is missing, the answer says what, instead of guessing.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .solutions import Solution, percent
from .solvers import GLOSSARY, SOLVERS, Param, Solver

RULE_SCORE = 2          # weight of matched phrases needed to choose without the model

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


def _tokens(text: str) -> List[_Token]:
    return [_Token(float(m.group(1)), bool(m.group(2)), m.start(1), m.end())
            for m in _NUMBER.finditer(text)]


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
            others = {value for _, value in hits} - {hits[0][1]}
            if others and param.required:
                notes.append(f"More than one was mentioned for {param.label}; "
                             f"{hits[0][1]} was used. Ask about the others separately.")

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
            if match:
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
        elif param.kind == "amount":
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
def rank_solvers(text: str) -> List[Tuple[int, Solver]]:
    """Solvers whose key phrases appear in the question, best match first."""
    low = text.lower()
    ranked = []
    for solver in SOLVERS.values():
        score = sum(weight for pattern, weight in solver.keywords if re.search(pattern, low))
        if score > 0:
            ranked.append((score, solver))
    ranked.sort(key=lambda item: -item[0])
    return ranked


_CONCEPT = re.compile(r"^\s*(explain|define|describe|what\s+(is|are|does)|what's|meaning\s+of|"
                      r"why\b|which\s+is\s+better|difference\s+between|how\s+does)", flags=re.I)


def _definitions(text: str) -> List[Tuple[str, str]]:
    low = text.lower().replace("-", " ")
    hits = []
    for term in sorted(GLOSSARY, key=len, reverse=True):
        key = term.replace("-", " ")
        if re.search(r"(?<![\w])" + re.escape(key) + r"(?![\w])", low) \
                and not any(key in earlier for earlier, _ in hits):
            hits.append((key, GLOSSARY[term]))
    return hits


# ----------------------------------------------------------------------
# The answer
# ----------------------------------------------------------------------
@dataclass
class Answer:
    """
    The reply to a question.

    Attributes:
        question: The question as asked
        kind: "solution" (calculated), "definition", "need_more" (something
            needed was not in the question) or "unknown"
        text: The message, for anything other than a solution
        solution: The worked solution
        solver: Name of the calculation used
        arguments: The values the calculation was given
        notes: How the question was read, including anything worked out
        used_model: Whether the language model had a part in reading it
    """
    question: str
    kind: str
    text: str = ""
    solution: Optional[Solution] = None
    solver: Optional[str] = None
    arguments: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    used_model: bool = False

    @property
    def solved(self) -> bool:
        """True if a calculation was carried out."""
        return self.solution is not None

    @property
    def value(self) -> Optional[float]:
        """The numerical answer, if there is one."""
        return self.solution.value if self.solution else None

    def explain(self) -> str:
        """The answer as plain text."""
        out = [f"Question: {self.question}", ""]
        if self.notes:
            out.append("How the question was read")
            out.extend(f"  - {note}" for note in self.notes)
            out.append("")
        out.append(self.solution.explain() if self.solution else self.text)
        return "\n".join(out)

    def to_markdown(self) -> str:
        """The answer as Markdown."""
        out = [f"> {self.question}", ""]
        if self.notes:
            out.append("*How the question was read:*")
            out.append("")
            out.extend(f"- {note}" for note in self.notes)
            out.append("")
        out.append(self.solution.to_markdown() if self.solution else self.text)
        return "\n".join(out)

    def _repr_markdown_(self) -> str:
        return self.to_markdown()

    def __repr__(self) -> str:
        return self.explain()


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
    lines = ["I could not tell which calculation you want. I can work out:"]
    lines.extend(f"  - {solver.title}" for solver in SOLVERS.values())
    lines.append("")
    lines.append('For example: "' + next(iter(SOLVERS.values())).example + '"')
    return "\n".join(lines)


def ask(question: str, use_model: bool = True) -> Answer:
    """
    Answer a question with a worked solution.

    Args:
        question: The question in plain words
        use_model: Let the installed language model help when the wording
            is unclear. With False, only the fixed rules are used.

    Returns:
        An :class:`Answer`. In a notebook it displays itself; elsewhere
        ``print(answer)`` or ``answer.explain()`` gives the text, and
        ``answer.value`` the number.
    """
    if not question or not question.strip():
        return Answer(question, "unknown", _what_i_can_do())
    text = normalise(question.strip())
    ranked = rank_solvers(text)
    backend = _backend(use_model)
    notes: List[str] = []
    used_model = False
    model_arguments: Dict[str, Any] = {}

    solver: Optional[Solver] = None
    clear = bool(ranked) and ranked[0][0] >= RULE_SCORE \
        and (len(ranked) == 1 or ranked[0][0] > ranked[1][0])
    if clear:
        solver = ranked[0][1]
    elif backend is not None:
        candidates = [s for _, s in ranked[:5]] or list(SOLVERS.values())
        try:
            call = _choose_with_model(backend, text, candidates)
            if call.name in SOLVERS:
                solver, model_arguments, used_model = SOLVERS[call.name], call.arguments, True
        except Exception as problem:
            notes.append(f"The language model could not be used ({str(problem)[:80]}).")

    definitions = _definitions(text)
    looks_like_concept = bool(_CONCEPT.match(text)) and not re.search(r"\d", text)

    if solver is not None:
        found, read_notes, missing = extract_arguments(solver, text)
        if missing and backend is not None:
            try:
                if not model_arguments:
                    model_arguments = backend.call_tools(text, [solver.tool()],
                                                         allow_none=False).arguments
                for param in list(missing):
                    value = model_arguments.get(param.name)
                    if param.kind != "choice" and _grounded(value, text):
                        found[param.name] = int(value) if param.kind in ("age", "years") else value
                        missing.remove(param)
                        used_model = True
                        read_notes.append(f"{param.label}: {value:g} was picked out by the "
                                          "language model. Check it.")
            except Exception as problem:
                notes.append(f"The language model could not be used ({str(problem)[:80]}).")
        if not missing:
            for param in solver.params:
                if param.name in found:
                    value = found[param.name]
                    if param.kind != "choice" and (
                            (param.low is not None and value < param.low)
                            or (param.high is not None and value > param.high)):
                        shown = percent(value) if param.kind == "rate" else f"{value:g}"
                        return Answer(question, "need_more",
                                      f"I read {param.label} as {shown}, which is outside the "
                                      "range I can work with. Please check the question.",
                                      solver=solver.name, arguments=found, notes=read_notes)
                elif not param.required:
                    found[param.name] = param.default
            how = ("chosen by the language model from your wording. Check that it is the one "
                   "you meant." if used_model and not clear else "recognised from your wording.")
            notes.append(f"Calculation: {solver.title}, {how}")
            notes.extend(read_notes)
            try:
                solution = solver.solve(**found)
            except (ValueError, KeyError) as problem:
                return Answer(question, "need_more", str(problem), solver=solver.name,
                              arguments=found, notes=notes, used_model=used_model)
            return Answer(question, "solution", solution=solution, solver=solver.name,
                          arguments=found, notes=notes, used_model=used_model)
        if not (looks_like_concept and definitions):
            needed = ", ".join(param.label for param in missing)
            message = (f"This looks like: {solver.title}.\nTo work it out I also need: {needed}."
                       f'\n\nFor example: "{solver.example}"')
            return Answer(question, "need_more", message, solver=solver.name, arguments=found,
                          notes=read_notes, used_model=used_model)

    if definitions:
        body = "\n\n".join(definition for _, definition in definitions[:3])
        if re.search(r"which\s+is\s+better|should\s+i\s+use|best\s+method", text, flags=re.I):
            body += ("\n\nWhich to use is a matter of judgement that depends on the data. "
                     "ACTUNEO can calculate each one; it does not choose between them.")
        return Answer(question, "definition", body, notes=notes)
    return Answer(question, "unknown", _what_i_can_do(), notes=notes)


def topics() -> pd.DataFrame:
    """The calculations :func:`ask` can do, each with an example question."""
    return pd.DataFrame([{"calculation": solver.title, "example": solver.example}
                         for solver in SOLVERS.values()])
