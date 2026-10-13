"""
AI Module

A language model for ACTUNEO that runs on the user's own machine, or a
provider reached with the user's own key.

- :func:`setup`: detect the hardware, recommend the largest local model the
  machine can run comfortably, ask, download it and remember the choice
- :func:`status` and :func:`remove_model`: see what is installed, and free
  the disk space
- :func:`configure`: use Gemini, OpenAI, Anthropic or Ollama instead
- :func:`get_backend`: the configured model, with ``generate()`` and
  ``call_tools()``

The local model needs the optional packages: ``pip install actuneo[ai]``.
Importing this module needs none of them.

This module supplies the model only. Nothing in ACTUNEO's calculations uses
it, and a language model's answers about actuarial work must be checked.
"""

from .assistant import Answer, ask, topics
from .backends import (
    AnthropicBackend,
    Backend,
    BackendError,
    GeminiBackend,
    LocalBackend,
    OllamaBackend,
    OpenAIBackend,
    ToolCall,
    shortlist_tools,
    tool_choice_schema,
)
from .benchmark import benchmark
from .catalog import ModelSpec, load_catalog
from .hardware import GPUInfo, HardwareProfile, detect_hardware
from .solutions import Solution, Step
from .setup import (
    SetupError,
    configure,
    download_model,
    get_backend,
    remove_model,
    select_model,
    setup,
    status,
)

__all__ = [
    'ask',
    'topics',
    'Answer',
    'Solution',
    'setup',
    'status',
    'remove_model',
    'configure',
    'get_backend',
    'detect_hardware',
    'select_model',
    'load_catalog',
    'download_model',
    'HardwareProfile',
    'GPUInfo',
    'ModelSpec',
    'Backend',
    'LocalBackend',
    'OpenAIBackend',
    'AnthropicBackend',
    'GeminiBackend',
    'OllamaBackend',
    'ToolCall',
    'tool_choice_schema',
    'shortlist_tools',
    'benchmark',
    'SetupError',
    'BackendError',
]
