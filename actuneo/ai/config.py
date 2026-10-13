"""
Where ACTUNEO keeps its AI settings and models

Settings go in the user's configuration folder and models in the user's
cache folder (found with ``platformdirs``), never inside the package. Set
the ``ACTUNEO_AI_HOME`` environment variable to keep both in a folder of
your choice.

API keys are never written to the configuration file.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("actuneo.ai")

CONFIG_NAME = "ai.json"


def _home_override():
    home = os.environ.get("ACTUNEO_AI_HOME")
    return Path(home).expanduser() if home else None


def config_dir() -> Path:
    """Folder that holds the configuration file."""
    home = _home_override()
    if home:
        return home
    try:
        import platformdirs
        return Path(platformdirs.user_config_dir("actuneo"))
    except ImportError:
        return Path.home() / ".actuneo"


def models_dir() -> Path:
    """Folder that holds downloaded models."""
    home = _home_override()
    if home:
        return home / "models"
    try:
        import platformdirs
        return Path(platformdirs.user_cache_dir("actuneo")) / "models"
    except ImportError:
        return Path.home() / ".actuneo" / "models"


def config_path() -> Path:
    """Path of the configuration file."""
    return config_dir() / CONFIG_NAME


def load_config() -> Dict[str, Any]:
    """The saved configuration, or an empty mapping if there is none."""
    path = config_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        logger.warning("Ignoring unreadable configuration %s: %s", path, error)
        return {}


def save_config(config: Dict[str, Any]) -> Path:
    """Write the configuration file and return its path."""
    for forbidden in ("api_key", "key", "token"):
        if forbidden in config:
            raise ValueError("API keys must not be written to the configuration file")
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path
