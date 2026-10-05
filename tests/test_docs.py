"""
Run the code examples in the README and documentation.
"""

import re
import textwrap
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _rst_blocks(text):
    blocks = re.findall(r"\.\. code-block:: python\n\n((?:[ \t]+.*\n|\n)+)", text)
    return [textwrap.dedent(block) for block in blocks]


def _markdown_blocks(text):
    return re.findall(r"```python\n(.*?)```", text, flags=re.S)


@pytest.mark.parametrize("path, extract", [
    ("README.md", _markdown_blocks),
    ("docs/index.rst", _rst_blocks),
    ("docs/quickstart.rst", _rst_blocks),
    ("docs/examples.rst", _rst_blocks),
])
def test_documentation_examples_run(path, extract, capsys):
    blocks = extract((ROOT / path).read_text(encoding="utf-8"))
    assert blocks, f"no python examples found in {path}"

    namespace = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for block in blocks:
            exec(compile(block, path, "exec"), namespace)
    capsys.readouterr()
