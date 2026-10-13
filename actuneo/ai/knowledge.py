"""
Search over what ACTUNEO already says

The assistant answers questions in words from two things: the language
model's own knowledge, and passages found here. Nothing in this index was
written for it. It is built from what the library already contains:

- the documentation (README, quick start, examples, the pensions guide, the
  IFRS 17 case study, and the Status and Limitations page), stored in
  ``corpus.json`` beside this file by :func:`build_corpus_file`;
- the docstring of every public module, class and function in ACTUNEO, read
  when the index is first used.

Search is by shared words (BM25), not by meaning. It needs no extra model
and no download.
"""

import importlib
import inspect
import json
import math
import pkgutil
import re
from collections import Counter
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MAX_WORDS = 170
DOCUMENTS = ("README.md", "docs/quickstart.rst", "docs/examples.rst", "docs/pensions.rst",
             "docs/ifrs17_case_study.rst", "docs/status.rst", "docs/installation.rst")

_STOP = frozenset("""a an and are as at be by can do does for from has have how i in is it its me my
of on or that the their them then there these this to was what when where which who why will with
you your""".split())
_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Passage:
    """A piece of ACTUNEO's documentation or a docstring."""
    title: str
    text: str
    source: str


def _terms(text: str) -> List[str]:
    words = []
    for word in _WORD.findall(text.lower().replace("_", " ")):
        if word in _STOP or len(word) < 2:
            continue
        words.append(word[:-1] if len(word) > 3 and word.endswith("s") else word)
    return words


def _clip(text: str, words: int = MAX_WORDS) -> str:
    parts = text.split()
    return " ".join(parts[:words]) + (" ..." if len(parts) > words else "")


# ----------------------------------------------------------------------
# Documentation
# ----------------------------------------------------------------------
def _clean(text: str) -> str:
    """Strip the commonest reStructuredText and Markdown markup."""
    text = re.sub(r":(?:class|func|meth|mod|doc|attr|data):`~?([^`]+)`", r"\1", text)
    text = re.sub(r"\.\. (?:code-block|automodule|autoclass|autofunction|image|toctree|note|"
                  r"list-table)::.*", "", text)
    text = re.sub(r"^\s*:(?:\w|-)+:.*$", "", text, flags=re.M)
    text = re.sub(r"\[!\[.*?\)\]\(.*?\)|!\[.*?\]\(.*?\)", "", text)
    text = text.replace("``", "").replace("**", "").replace("`", "")
    return re.sub(r"[ \t]+", " ", text)


def _sections(text: str, is_markdown: bool) -> List[Tuple[str, str]]:
    """Split a document into (heading, body) pairs."""
    lines = text.splitlines()
    sections, title, body = [], "", []
    index = 0
    while index < len(lines):
        line = lines[index]
        following = lines[index + 1] if index + 1 < len(lines) else ""
        heading = None
        if is_markdown and re.match(r"#{1,4}\s+\S", line) and not line.startswith("#!"):
            heading = line.lstrip("#").strip()
        elif not is_markdown and line.strip() and len(following) >= 3 \
                and re.fullmatch(r"([=\-~^\"'])\1+", following.strip()) \
                and len(following.strip()) >= len(line.strip()) - 1:
            heading = line.strip()
            index += 1
        if heading is not None:
            sections.append((title, "\n".join(body)))
            title, body = heading, []
        else:
            body.append(line)
        index += 1
    sections.append((title, "\n".join(body)))
    return [(t, b) for t, b in sections if b.strip()]


def passages_from_document(path: Path, name: str) -> List[Passage]:
    """Cut one documentation file into passages by heading."""
    text = path.read_text(encoding="utf-8")
    found = []
    for title, body in _sections(text, path.suffix == ".md"):
        body = _clean(body)
        blocks = [block.strip() for block in re.split(r"\n\s*\n", body) if block.strip()]
        current: List[str] = []
        count = 0
        for block in blocks + [None]:
            size = len(block.split()) if block else 0
            if block is None or (current and count + size > MAX_WORDS):
                piece = " ".join(" ".join(current).split())
                if len(piece.split()) >= 8:
                    found.append(Passage(title or name, _clip(piece), name))
                current, count = [], 0
            if block:
                current.append(block)
                count += size
    return found


def build_corpus_file(root: Optional[Path] = None) -> Path:
    """
    Rebuild ``corpus.json`` from the documentation in a source checkout.

    Run it after changing the documentation::

        python -c "from actuneo.ai.knowledge import build_corpus_file; build_corpus_file()"

    Args:
        root: The repository root (found from this file if omitted)

    Returns:
        Path of the file written
    """
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    passages = []
    for name in DOCUMENTS:
        path = root / name
        if path.exists():
            passages.extend(passages_from_document(path, name))
    target = Path(__file__).with_name("corpus.json")
    target.write_text(json.dumps(
        [{"title": p.title, "text": p.text, "source": p.source} for p in passages],
        indent=1, ensure_ascii=False), encoding="utf-8")
    return target


def _documentation() -> List[Passage]:
    try:
        text = (resources.files("actuneo.ai") / "corpus.json").read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return []
    return [Passage(**entry) for entry in json.loads(text)]


# ----------------------------------------------------------------------
# Docstrings
# ----------------------------------------------------------------------
def _signature(obj) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return ""


def _docstrings() -> List[Passage]:
    import actuneo
    found = []
    for info in pkgutil.walk_packages(actuneo.__path__, "actuneo."):
        name = info.name
        if name.endswith("__main__") or ".ai." in name:
            continue
        try:
            module = importlib.import_module(name)
        except Exception:           # an optional dependency is missing
            continue
        doc = inspect.getdoc(module)
        if doc and not info.ispkg or (doc and info.ispkg):
            found.append(Passage(f"The {name} module", _clip(doc), name))
        for attribute, obj in vars(module).items():
            if attribute.startswith("_") or getattr(obj, "__module__", None) != name:
                continue
            if not (inspect.isclass(obj) or inspect.isfunction(obj)):
                continue
            doc = inspect.getdoc(obj)
            if not doc:
                continue
            label = f"{name}.{attribute}"
            found.append(Passage(label, _clip(f"{attribute}{_signature(obj)}. {doc}"), label))
            if inspect.isclass(obj):
                for method, member in vars(obj).items():
                    target = getattr(member, "__func__", member)
                    if method.startswith("_") or not inspect.isfunction(target):
                        continue
                    doc = inspect.getdoc(target)
                    if doc and len(doc.split()) >= 6:
                        found.append(Passage(f"{label}.{method}",
                                             _clip(f"{attribute}.{method}{_signature(target)}. {doc}"),
                                             f"{label}.{method}"))
    return found


# ----------------------------------------------------------------------
# Index
# ----------------------------------------------------------------------
class _Index:
    def __init__(self, passages: List[Passage]):
        self.passages = passages
        self.counts = []
        document_frequency: Counter = Counter()
        for passage in passages:
            # Words in the title count three times
            terms = Counter(_terms(passage.title) * 3 + _terms(passage.text))
            self.counts.append(terms)
            document_frequency.update(terms.keys())
        self.lengths = [sum(count.values()) for count in self.counts]
        self.average = (sum(self.lengths) / len(self.lengths)) if self.lengths else 1.0
        total = len(passages)
        self.idf = {term: math.log(1 + (total - n + 0.5) / (n + 0.5))
                    for term, n in document_frequency.items()}

    def search(self, question: str, limit: int) -> List[Tuple[float, Passage]]:
        asked = set(_terms(question))
        scored = []
        for position, counts in enumerate(self.counts):
            score = 0.0
            for term in asked:
                frequency = counts.get(term)
                if frequency:
                    norm = frequency + 1.5 * (0.25 + 0.75 * self.lengths[position] / self.average)
                    score += self.idf[term] * frequency * 2.5 / norm
            if score > 0:
                scored.append((score, position))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [(score, self.passages[position]) for score, position in scored[:limit]]


_INDEX: Dict[str, _Index] = {}


def corpus() -> List[Passage]:
    """Every passage the assistant can draw on."""
    return _index().passages


def _index() -> _Index:
    if "index" not in _INDEX:
        _INDEX["index"] = _Index(_documentation() + _docstrings())
    return _INDEX["index"]


def retrieve(question: str, limit: int = 4, minimum: float = 4.0) -> List[Tuple[float, Passage]]:
    """
    The passages of ACTUNEO's documentation and docstrings most relevant to
    a question.

    Args:
        question: The question
        limit: Most passages to return
        minimum: Lowest score worth returning. Below it a passage shares
            only incidental words with the question.

    Returns:
        (score, passage) pairs, best first. Empty when nothing is relevant.
    """
    return [(score, passage) for score, passage in _index().search(question, limit)
            if score >= minimum]
