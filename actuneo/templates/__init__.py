"""
Templates

Excel workbooks to fill in with your own data, shipped with the package:

``triangle``
    A claims triangle laid out for
    :meth:`actuneo.loss_reserving.Triangle.from_excel`.

``paa_case``
    The four input sheets of an IFRS 17 premium allocation approach case,
    for :func:`actuneo.ifrs17.load_paa_case`.

Each workbook has a "Guide" sheet explaining what to enter and is filled
with invented example figures to overwrite.
"""

import shutil
from importlib import resources
from pathlib import Path
from typing import List

_FILES = {
    "triangle": "triangle_template.xlsx",
    "paa_case": "paa_case_template.xlsx",
}


def list_templates() -> List[str]:
    """Names of the templates shipped with ACTUNEO."""
    return sorted(_FILES)


def template_path(name: str) -> Path:
    """
    Location of a template inside the installed package.

    Args:
        name: Template name, see :func:`list_templates`
    """
    if name not in _FILES:
        raise ValueError(f"Unknown template '{name}'. Available: {list_templates()}")
    return Path(str(resources.files("actuneo.templates") / _FILES[name]))


def save_template(name: str, destination: str = ".", overwrite: bool = False) -> Path:
    """
    Copy a template to a folder or file of your choice, ready to fill in.

    Args:
        name: Template name, see :func:`list_templates`
        destination: Folder to copy into (keeping the template's file name),
            or the full name of the new .xlsx file
        overwrite: Replace the file if it already exists

    Returns:
        Path of the copy
    """
    source = template_path(name)
    target = Path(destination)
    if target.suffix.lower() != ".xlsx":
        target = target / source.name
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} already exists. Pass overwrite=True to replace it.")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


__all__ = ['list_templates', 'template_path', 'save_template']
