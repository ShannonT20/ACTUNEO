"""
Catalog of local models

The models that :func:`actuneo.ai.setup` can offer are listed in
``models.json`` beside this file. The catalog is data: change the file, or
set the ``ACTUNEO_AI_CATALOG`` environment variable to the path of your own
copy, and the selection logic picks up the new list.
"""

import json
import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import List, Optional, Union

GB = 1e9

_REQUIRED = ("key", "name", "repo", "filename", "size_bytes", "min_ram_gb", "min_vram_gb")


@dataclass(frozen=True)
class ModelSpec:
    """One downloadable model."""
    key: str
    name: str
    repo: str
    filename: str
    size_bytes: int
    min_ram_gb: float
    min_vram_gb: float
    layers: int = 32
    quality: str = ""
    tool_calling: bool = False
    license: str = ""
    parameters: str = ""
    sha256: Optional[str] = None

    @property
    def size_gb(self) -> float:
        """Download size in gigabytes (10^9 bytes)."""
        return self.size_bytes / GB

    def to_dict(self) -> dict:
        """The entry as plain data."""
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def load_catalog(path: Union[str, Path, None] = None) -> List[ModelSpec]:
    """
    Read the model catalog, smallest model first.

    Args:
        path: A catalog file. If omitted, the file named by the
            ``ACTUNEO_AI_CATALOG`` environment variable is used, or the
            catalog shipped with ACTUNEO.

    Returns:
        The models, ordered by download size
    """
    path = path or os.environ.get("ACTUNEO_AI_CATALOG")
    if path:
        text = Path(path).read_text(encoding="utf-8")
    else:
        text = (resources.files("actuneo.ai") / "models.json").read_text(encoding="utf-8")
    data = json.loads(text)
    entries = data["models"] if isinstance(data, dict) else data
    models = []
    for entry in entries:
        missing = [name for name in _REQUIRED if name not in entry]
        if missing:
            raise ValueError(f"catalog entry {entry.get('key', entry)} is missing {missing}")
        known = {k: v for k, v in entry.items() if k in ModelSpec.__dataclass_fields__}
        models.append(ModelSpec(**known))
    keys = [model.key for model in models]
    if len(set(keys)) != len(keys):
        raise ValueError("catalog keys must be unique")
    if not models:
        raise ValueError("the catalog holds no models")
    return sorted(models, key=lambda model: model.size_bytes)
