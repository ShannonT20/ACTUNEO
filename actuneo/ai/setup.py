"""
Set up a language model for ACTUNEO

:func:`setup` looks at the machine, picks the largest local model it can run
comfortably, says what will be downloaded and how big it is, asks before
downloading, then saves the choice so the question is asked only once. After
the download everything runs offline and costs nothing.

:func:`configure` is the alternative: use a provider (Gemini, OpenAI,
Anthropic or Ollama) with your own key.

How a model is chosen
---------------------
A model fits when its memory need, with headroom left for the operating
system and Python, is within what is free:

- on an NVIDIA card, within free video memory less 20%, all layers on the GPU;
- on Apple Silicon, within free memory less 30% (the GPU shares it);
- otherwise within free memory less 30%, on the processor. An NVIDIA card
  that is too small for the whole model takes as many layers as it can hold.

The download plus 1 GB must also fit on the disk. The recommended model is
the largest that fits, except that a model which cannot sit wholly on a GPU
is recommended only up to about 5.5 GB, because larger ones answer slowly on a
processor. Those are still offered as the "largest" alternative.
"""

import datetime
import errno
import hashlib
import logging
import os
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from . import backends as _backends
from . import config as _config
from .backends import PROVIDERS, Backend, BackendError, LocalBackend
from .catalog import ModelSpec, load_catalog, tiers
from .hardware import GIB, HardwareProfile, detect_hardware

logger = logging.getLogger("actuneo.ai")

RAM_HEADROOM = 0.30
VRAM_HEADROOM = 0.20
RELAXED_HEADROOM = 0.10
DISK_BUFFER_GB = 1.0
CPU_COMFORT_SIZE_GB = 5.5
LARGE_DOWNLOAD_GB = 3.0
BUSY_MEMORY_SHARE = 0.4
CONTEXT_TOKENS = 4096
KEYRING_SERVICE = "actuneo"

COLAB_ADVICE = (
    "This machine does not have enough free memory to run even the smallest model safely.\n"
    "Two ways forward, both free to start:\n"
    "  1. Use a provider with your own key, for example Gemini's free tier:\n"
    "       actuneo.ai.configure(provider=\"gemini\", api_key=\"...\")\n"
    "  2. Run ACTUNEO in a free Google Colab notebook, which has more memory,\n"
    "     and call actuneo.ai.setup() there."
)

DISK_ADVICE = (
    "There is not enough free disk space for even the smallest model (about 1.5 GB is\n"
    "needed). Free some space and run actuneo.ai.setup() again, or use a provider with\n"
    "your own key:  actuneo.ai.configure(provider=\"gemini\", api_key=\"...\")"
)

_SESSION_KEYS: Dict[str, str] = {}
_BACKEND_CACHE: Dict[str, Backend] = {}


class SetupError(RuntimeError):
    """Setup could not be completed. The message says what to do next."""


@dataclass
class Choice:
    """A model and how it would run on this machine."""
    model: ModelSpec
    device: str             # "gpu", "gpu+cpu" or "cpu"
    n_gpu_layers: int       # -1 for all layers, 0 for none
    comfortable: bool = True

    def runs_on(self) -> str:
        """Where the model runs, in words."""
        if self.device == "gpu":
            return "GPU (all layers)"
        if self.device == "gpu+cpu":
            return f"GPU and CPU ({self.n_gpu_layers} of {self.model.layers} layers on the GPU)"
        return "CPU only"


@dataclass
class Selection:
    """The outcome of choosing a model for a machine."""
    recommended: Optional[Choice] = None
    smallest: Optional[Choice] = None
    largest: Optional[Choice] = None
    too_weak: bool = False
    fallback: bool = False
    disk_limited: bool = False


# ----------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------
def placement(model: ModelSpec, hardware: HardwareProfile,
              relaxed: bool = False) -> Optional[Choice]:
    """
    How a model would run on the machine, or None if it does not fit.

    Args:
        model: The model
        hardware: The machine
        relaxed: Leave only 10% headroom instead of the usual 20% to 30%.
            The model runs, but the machine may slow down.
    """
    if hardware.disk_free_gb is not None \
            and model.size_bytes / GIB + DISK_BUFFER_GB > hardware.disk_free_gb:
        return None
    ram_headroom = RELAXED_HEADROOM if relaxed else RAM_HEADROOM
    vram_headroom = RELAXED_HEADROOM if relaxed else VRAM_HEADROOM
    ram_budget = (hardware.ram_available_gb or 0.0) * (1 - ram_headroom)
    comfortable_on_cpu = model.size_gb <= CPU_COMFORT_SIZE_GB
    gpu = hardware.gpu

    if gpu is not None and gpu.unified:
        if model.min_ram_gb <= ram_budget:
            return Choice(model, "gpu", -1)
        return None
    if gpu is not None:
        vram_budget = gpu.vram_free_gb * (1 - vram_headroom)
        if model.min_vram_gb <= vram_budget:
            return Choice(model, "gpu", -1)
        if model.min_ram_gb <= ram_budget:
            layers = int(model.layers * vram_budget / model.min_vram_gb)
            if layers > 0:
                return Choice(model, "gpu+cpu", layers, comfortable_on_cpu)
            return Choice(model, "cpu", 0, comfortable_on_cpu)
        return None
    if model.min_ram_gb <= ram_budget:
        return Choice(model, "cpu", 0, comfortable_on_cpu)
    return None


def select_model(hardware: HardwareProfile,
                 catalog: Optional[List[ModelSpec]] = None) -> Selection:
    """
    Choose models for a machine.

    Args:
        hardware: The detected hardware
        catalog: The models to choose from (the shipped catalog if omitted)

    Returns:
        The recommended model, the smallest that fits and the largest that
        fits with reduced headroom. ``too_weak`` is True when nothing fits.
        ``fallback`` is True when the hardware could not be measured and the
        smallest model was chosen for safety.
    """
    catalog = tiers(catalog or load_catalog())
    if not hardware.detection_ok:
        choice = Choice(catalog[0], "cpu", 0)
        return Selection(choice, choice, choice, fallback=True)
    fits = [c for c in (placement(m, hardware) for m in catalog) if c is not None]
    if not fits:
        roomy = replace(hardware, disk_free_gb=None)
        disk_limited = any(placement(m, roomy) is not None for m in catalog)
        return Selection(too_weak=True, disk_limited=disk_limited)
    relaxed = [c for c in (placement(m, hardware, relaxed=True) for m in catalog)
               if c is not None]
    comfortable = [c for c in fits if c.comfortable]
    recommended = comfortable[-1] if comfortable else fits[0]
    return Selection(recommended, fits[0], relaxed[-1] if relaxed else fits[-1])


def _options(selection: Selection) -> List[Dict[str, Any]]:
    """The numbered alternatives shown to the user, without repeats."""
    entries = [
        ("smallest", "Smallest", "fastest", selection.smallest),
        ("recommended", "Recommended", "", selection.recommended),
        ("largest", "Largest your PC can handle", "may be slow", selection.largest),
    ]
    options, seen = [], set()
    for key, label, remark, choice in entries:
        if choice is None:
            continue
        if choice.model.key in seen and key != "recommended":
            continue
        if key == "recommended":
            # The recommended entry replaces an earlier duplicate
            options = [o for o in options if o["choice"].model.key != choice.model.key]
        seen.add(choice.model.key)
        if key == "largest" and choice.model.key == selection.recommended.model.key:
            continue
        options.append({"key": key, "label": label, "remark": remark, "choice": choice})
    for number, option in enumerate(options, start=1):
        option["number"] = number
    return options


def render_summary(hardware: HardwareProfile, selection: Selection,
                   models_dir: Union[str, Path, None] = None) -> str:
    """The text shown to the user before anything is downloaded."""
    lines = [f"Detected: {hardware.describe()}"]
    lines.extend(f"Note: {note}" for note in hardware.notes)
    if selection.too_weak:
        lines.append("")
        lines.append(DISK_ADVICE if selection.disk_limited else COLAB_ADVICE)
        return "\n".join(lines)
    choice = selection.recommended
    model = choice.model
    lines.append(f"Recommended model: {model.name} - {model.size_gb:.1f} GB download")
    if selection.fallback:
        lines.append("Why: the hardware could not be measured, so the smallest model "
                     "was chosen to be safe")
    else:
        lines.append("Why: largest model your machine can run comfortably")
        if (hardware.ram_total_gb and hardware.ram_available_gb is not None
                and hardware.ram_available_gb < BUSY_MEMORY_SHARE * hardware.ram_total_gb):
            lines.append(f"Tip: only {hardware.ram_available_gb:.1f} of "
                         f"{hardware.ram_total_gb:.0f} GB of memory is free. Closing other "
                         "applications before setup may let a larger model fit.")
    lines.append(f"Runs on: {choice.runs_on()}")
    details = [f"quality: {model.quality}" if model.quality else "",
               f"licence: {model.license}" if model.license else ""]
    lines.append("Details: " + ", ".join(d for d in details if d))
    if model.note:
        lines.append(f"Note: {model.note}")
    if models_dir is not None:
        lines.append(f"Stored in: {models_dir}")
    parts = []
    for option in _options(selection):
        remark = f", {option['remark']}" if option["remark"] else ""
        parts.append(f"[{option['number']}] {option['label']} "
                     f"({option['choice'].model.size_gb:.1f} GB{remark})")
    parts.append("[s] Skip and use an API key instead")
    lines.append("Alternatives: " + "  ".join(parts))
    if model.size_gb > LARGE_DOWNLOAD_GB:
        lines.append(f"Warning: this is a large download ({model.size_gb:.1f} GB).")
    lines.append("If you are on mobile data or a metered connection, this download "
                 "will use your data allowance.")
    return "\n".join(lines)


def _say(text: str, title: Optional[str] = None) -> None:
    """Print for the user, with rich if it is installed."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
        console = Console()
        if title:
            console.print(Panel(Text(text), title=title, expand=False))
        else:
            console.print(text, markup=False, highlight=False)
    except ImportError:
        if title:
            print(f"== {title} ==")
        print(text)


# ----------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_message(error: BaseException, model: ModelSpec) -> str:
    """A plain explanation of a failed download."""
    text = f"{type(error).__name__}: {error}"
    if isinstance(error, OSError) and error.errno == errno.ENOSPC:
        return ("The disk filled up during the download. Free some space and run "
                "actuneo.ai.setup() again: the download resumes where it stopped.")
    lowered = text.lower()
    if any(word in lowered for word in ("404", "repositorynotfound", "entrynotfound",
                                        "revisionnotfound")):
        return (f"{model.repo}/{model.filename} was not found on Hugging Face. The catalog "
                "entry may be out of date.")
    if any(word in lowered for word in ("connection", "timeout", "timed out", "offline",
                                        "localentrynotfound", "max retries", "resolve",
                                        "network", "ssl")):
        return ("Could not reach Hugging Face. Check the internet connection and run "
                "actuneo.ai.setup() again: the download resumes where it stopped.\n"
                f"({text[:300]})")
    return f"The download failed: {text[:400]}"


def download_model(model: ModelSpec, directory: Union[str, Path, None] = None,
                   verify: bool = True) -> Path:
    """
    Download a model from Hugging Face, resuming a partial download.

    Args:
        model: The catalog entry
        directory: Where to store it (the user's cache folder if omitted)
        verify: Check the size and SHA-256 of the file against the catalog

    Returns:
        Path of the model file
    """
    directory = Path(directory) if directory else _config.models_dir()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / model.filename
    if target.exists() and target.stat().st_size == model.size_bytes:
        logger.info("Model already downloaded: %s", target)
    else:
        # Plain HTTP downloads resume reliably on slow or unsteady connections,
        # where Hugging Face's xet transfer has been seen to stall. The setting
        # is read when huggingface_hub is first imported; set it yourself to
        # "0" to use xet.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise SetupError("huggingface_hub is not installed. "
                             "Install the AI extras with: pip install actuneo[ai]") from error
        free = shutil.disk_usage(directory).free / GIB
        if model.size_bytes / GIB + DISK_BUFFER_GB > free:
            raise SetupError(f"Not enough disk space: {model.size_gb:.1f} GB plus 1 GB spare is "
                             f"needed and {free:.1f} GB is free in {directory}.")
        logger.info("Downloading %s/%s to %s", model.repo, model.filename, directory)
        try:
            target = Path(hf_hub_download(repo_id=model.repo, filename=model.filename,
                                          local_dir=str(directory)))
        except KeyboardInterrupt:
            raise SetupError("Download interrupted. Run actuneo.ai.setup() again to resume "
                             "where it stopped.") from None
        except Exception as error:
            raise SetupError(_download_message(error, model)) from error
    if verify:
        size = target.stat().st_size
        if size != model.size_bytes:
            raise SetupError(f"The file is {size:,} bytes but {model.size_bytes:,} were expected. "
                             "Run actuneo.ai.setup() again to finish the download.")
        if model.sha256:
            _say("Checking the downloaded file...")
            if _sha256(target) != model.sha256:
                target.unlink()
                raise SetupError("The downloaded file failed its integrity check and was "
                                 "removed. Run actuneo.ai.setup() again.")
    return target


# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------
def _resolve(model: str, selection: Selection, hardware: HardwareProfile,
             catalog: List[ModelSpec], force: bool) -> Optional[Choice]:
    """The choice a ``model=`` argument names. None means skip."""
    name = model.strip().lower()
    if name in ("skip", "api", "byok", "s"):
        return None
    matches = [m for m in catalog if name in (m.key.lower(), m.name.lower())]
    named = {"smallest": selection.smallest, "fastest": selection.smallest,
             "recommended": selection.recommended, "auto": selection.recommended,
             "largest": selection.largest, "max": selection.largest}
    if not matches and name in named:
        if named[name] is None:
            raise SetupError(DISK_ADVICE if selection.disk_limited else COLAB_ADVICE)
        return named[name]
    if not matches:
        keys = ", ".join(m.key for m in catalog)
        raise SetupError(f"Unknown model '{model}'. Use smallest, recommended, largest, "
                         f"skip, or one of: {keys}.")
    choice = placement(matches[0], hardware, relaxed=True)
    if choice is None:
        if not force:
            raise SetupError(
                f"{matches[0].name} needs about {matches[0].min_ram_gb:.0f} GB of free memory "
                f"and {matches[0].size_gb:.1f} GB of disk, more than this machine has free. "
                "Pass force=True to download it anyway.")
        choice = Choice(matches[0], "cpu", 0, comfortable=False)
    return choice


def _ask(selection: Selection, input_func: Callable[[str], str]) -> Optional[Choice]:
    """Ask the user what to do. Returns the choice, or None to skip."""
    options = {str(o["number"]): o["choice"] for o in _options(selection)}
    numbers = "/".join(options)
    question = f"Continue with the recommended model? [y/N, {numbers} to choose, s to skip]: "
    while True:
        try:
            answer = input_func(question).strip().lower()
        except EOFError:
            _say("No answer could be read. To run without questions use "
                 "actuneo.ai.setup(yes=True), setup(auto=True) or setup(model=\"small\").")
            return None
        if answer in ("y", "yes"):
            return selection.recommended
        if answer in options:
            return options[answer]
        if answer in ("", "n", "no", "s", "skip", "q", "quit"):
            return None
        _say(f"Please answer y, n, s or one of {numbers}.")


def setup(auto: bool = False,
          model: Optional[str] = None,
          yes: bool = False,
          force: bool = False,
          models_dir: Union[str, Path, None] = None,
          hardware: Optional[HardwareProfile] = None,
          input_func: Callable[[str], str] = input) -> Dict[str, Any]:
    """
    Choose, download and configure a local model for this machine.

    Run with no arguments to be shown the recommendation and asked before
    anything is downloaded. Setup runs once: later calls report the model
    already in place unless ``force`` is set or a different ``model`` is
    named.

    Args:
        auto: Take the recommended model without asking
        model: Take a particular model without asking: a key from the
            catalog ("minimal", "tiny", "small", "default", "strong", "max",
            or a challenger such as "phi4-mini"), or "smallest",
            "recommended" or "largest" for this machine, or "skip" to
            download nothing
        yes: Answer yes to the confirmation (takes the recommended model)
        force: Set up again even if a model is in place, and allow a named
            model that does not fit the machine
        models_dir: Folder for the model (the user's cache folder if omitted)
        hardware: A hardware profile to use instead of detecting one
        input_func: Function used to ask the question (for testing)

    Returns:
        The status, as from :func:`status`, with ``"result"`` set to
        "ready", "already_set_up", "skipped" or "too_weak"
    """
    directory = Path(models_dir) if models_dir else _config.models_dir()
    saved = _config.load_config()
    local = saved.get("local") or {}
    in_place = bool(local.get("path")) and Path(local["path"]).exists()
    if in_place and not force and model is None:
        if saved.get("backend") != "local":
            saved["backend"] = "local"
            _config.save_config(saved)
            _BACKEND_CACHE.clear()
        _say(f"Already set up: {local['model']['name']}. "
             "Use setup(force=True) to choose again, or remove_model() to free the space.")
        return {**status(quiet=True), "result": "already_set_up"}

    catalog = load_catalog()
    hardware = hardware or detect_hardware(directory)
    selection = select_model(hardware, catalog)
    _say(render_summary(hardware, selection, directory), title="ACTUNEO AI setup")
    if selection.too_weak and model is None:
        return {**status(quiet=True), "result": "too_weak"}

    if model is not None:
        choice = _resolve(model, selection, hardware, catalog, force)
    elif auto or yes:
        choice = selection.recommended
    else:
        choice = _ask(selection, input_func)
    if choice is None:
        _say("Nothing was downloaded. To use a provider with your own key:\n"
             "  actuneo.ai.configure(provider=\"gemini\", api_key=\"...\")")
        return {**status(quiet=True), "result": "skipped"}

    spec = choice.model
    if spec.role != "tier":
        _say(f"{spec.name} is a challenger model, offered for comparison."
             + (f" {spec.note}" if spec.note else "") + f" Licence: {spec.license}.")
    _say(f"Downloading {spec.name} ({spec.size_gb:.1f} GB) from huggingface.co/{spec.repo}")
    path = download_model(spec, directory)
    saved["backend"] = "local"
    saved["local"] = {
        "model": spec.to_dict(),
        "path": str(path),
        "device": choice.device,
        "n_gpu_layers": choice.n_gpu_layers,
        "n_ctx": CONTEXT_TOKENS,
        "n_threads": hardware.cpu_physical,
        "no_think": spec.no_think,
        "hardware": hardware.to_dict(),
        "configured_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    _config.save_config(saved)
    _BACKEND_CACHE.clear()
    _say(f"Ready: {spec.name}, running on {choice.runs_on()}.\n"
         "It now works offline. Try:  actuneo.ai.get_backend().generate(\"Hello\")")
    return {**status(quiet=True), "result": "ready"}


# ----------------------------------------------------------------------
# Status, removal, keys
# ----------------------------------------------------------------------
def _find_key(provider: str) -> Optional[str]:
    """A provider's key from this session, the environment or the system keyring."""
    if provider in _SESSION_KEYS:
        return _SESSION_KEYS[provider]
    key = _backends.api_key_from_environment(provider)
    if key:
        return key
    try:
        import keyring
        return keyring.get_password(KEYRING_SERVICE, provider)
    except Exception:       # keyring missing or no usable store
        return None


def status(quiet: bool = False) -> Dict[str, Any]:
    """
    What is configured: the backend in use, the local model and its size on
    disk, the hardware recorded at setup, and the provider if one is set.

    Args:
        quiet: Return the information without printing it

    Returns:
        A mapping. API keys are never included, only whether one was found.
    """
    saved = _config.load_config()
    local = saved.get("local") or {}
    api = saved.get("api") or {}
    info: Dict[str, Any] = {
        "configured": bool(saved.get("backend")),
        "backend": saved.get("backend"),
        "config_file": str(_config.config_path()),
        "models_dir": str(_config.models_dir()),
        "local_model": None,
        "provider": None,
    }
    lines = []
    if local.get("path"):
        path = Path(local["path"])
        on_disk = path.stat().st_size / 1e9 if path.exists() else 0.0
        info["local_model"] = {
            "name": local["model"]["name"], "key": local["model"]["key"],
            "path": str(path), "present": path.exists(), "size_gb": round(on_disk, 2),
            "device": local.get("device"), "n_gpu_layers": local.get("n_gpu_layers"),
            "hardware": local.get("hardware"), "configured_at": local.get("configured_at"),
        }
        state = f"{on_disk:.1f} GB on disk" if path.exists() else "file missing"
        lines.append(f"Local model: {local['model']['name']} ({state}), {local.get('device')}")
        lines.append(f"  File: {path}")
        hardware = local.get("hardware") or {}
        if hardware.get("ram_total_gb"):
            gpu = (hardware.get("gpu") or {}).get("name", "no supported GPU")
            lines.append(f"  Hardware at setup: {hardware['ram_total_gb']:.0f} GB RAM, {gpu}")
    if api.get("provider"):
        provider = api["provider"]
        found = provider == "ollama" or _find_key(provider) is not None
        info["provider"] = {"name": provider, "model": api.get("model"),
                            "api_key_found": found}
        lines.append(f"Provider: {provider} ({api.get('model')}), "
                     f"key {'found' if found else 'NOT found'}")
    if not lines:
        lines.append("Nothing is set up yet. Run actuneo.ai.setup() for a local model, or "
                     "actuneo.ai.configure(provider=...) to use your own API key.")
    else:
        lines.insert(0, f"Backend in use: {saved.get('backend')}")
    if not quiet:
        _say("\n".join(lines), title="ACTUNEO AI status")
    return info


def remove_model(yes: bool = False, input_func: Callable[[str], str] = input) -> Dict[str, Any]:
    """
    Delete the downloaded model to free disk space.

    Args:
        yes: Delete without asking
        input_func: Function used to ask the question (for testing)

    Returns:
        Whether a model was removed and how many gigabytes were freed
    """
    saved = _config.load_config()
    local = saved.get("local") or {}
    if not local.get("path"):
        _say("There is no local model to remove.")
        return {"removed": False, "freed_gb": 0.0}
    path = Path(local["path"])
    size = path.stat().st_size / 1e9 if path.exists() else 0.0
    if not yes:
        try:
            answer = input_func(f"Delete {local['model']['name']} ({size:.1f} GB)? [y/N]: ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            _say("Nothing was removed.")
            return {"removed": False, "freed_gb": 0.0}
    if path.exists():
        path.unlink()
    bookkeeping = path.parent / ".cache" / "huggingface"
    if bookkeeping.exists():
        shutil.rmtree(bookkeeping, ignore_errors=True)
    saved.pop("local", None)
    if saved.get("backend") == "local":
        saved["backend"] = "api" if saved.get("api") else None
    _config.save_config(saved)
    _BACKEND_CACHE.clear()
    logger.info("Removed model %s", path)
    _say(f"Removed {local['model']['name']} and freed {size:.1f} GB.")
    return {"removed": True, "freed_gb": round(size, 2)}


def configure(provider: str,
              api_key: Optional[str] = None,
              model: Optional[str] = None,
              base_url: Optional[str] = None,
              save_key: bool = False) -> Dict[str, Any]:
    """
    Use a provider with your own key instead of a local model.

    The key is held in memory for this Python session only. For later
    sessions set the provider's environment variable (``GEMINI_API_KEY`` or
    ``GOOGLE_API_KEY``, ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``), or pass
    ``save_key=True`` to store it in the operating system's keychain. The
    key is never written to ACTUNEO's configuration file and never logged.

    Args:
        provider: "gemini", "openai", "anthropic" or "ollama"
        api_key: The key. If omitted it is read from the environment.
            Ollama needs none.
        model: Model to use (the provider's default in ACTUNEO if omitted)
        base_url: Address of the service, for a proxy or an Ollama server
            on another machine
        save_key: Store the key in the system keychain (needs the
            ``keyring`` package)

    Returns:
        The status, as from :func:`status`
    """
    provider = str(provider).strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(f"provider must be one of {PROVIDERS}")
    if api_key:
        _SESSION_KEYS[provider] = api_key
        if save_key:
            try:
                import keyring
                keyring.set_password(KEYRING_SERVICE, provider, api_key)
            except ImportError as error:
                raise SetupError("Saving a key needs the keyring package (pip install keyring). "
                                 "Or set the provider's environment variable instead.") from error
    saved = _config.load_config()
    saved["backend"] = "api"
    saved["api"] = {"provider": provider,
                    "model": model or _backends.DEFAULT_MODELS[provider]}
    if base_url:
        saved["api"]["base_url"] = base_url
    _config.save_config(saved)
    _BACKEND_CACHE.clear()
    logger.info("Configured provider %s with model %s", provider, saved["api"]["model"])
    if provider not in ("ollama", "anthropic") and _find_key(provider) is None:
        variables = " or ".join(_backends.KEY_VARIABLES[provider])
        _say(f"{provider} is selected, but no key was given. Set {variables} before use.")
    return status(quiet=True)


def get_backend() -> Backend:
    """
    The configured backend, ready to use.

    Returns:
        A backend with ``generate()`` and ``call_tools()``. A local model is
        loaded on its first use and kept for the session.
    """
    saved = _config.load_config()
    kind = saved.get("backend")
    if kind == "local":
        local = saved.get("local") or {}
        key = f"local:{local.get('path')}:{local.get('n_gpu_layers')}"
        if key not in _BACKEND_CACHE:
            if not local.get("path") or not Path(local["path"]).exists():
                raise SetupError("The local model file is missing. Run actuneo.ai.setup().")
            _BACKEND_CACHE[key] = LocalBackend(
                local["path"], local.get("n_gpu_layers", 0),
                local.get("n_ctx", CONTEXT_TOKENS), local.get("n_threads"),
                no_think=bool(local.get("no_think", False)))
        return _BACKEND_CACHE[key]
    if kind == "api":
        api = saved["api"]
        try:
            return _backends.make_api_backend(api["provider"], _find_key(api["provider"]),
                                              api.get("model"), api.get("base_url"))
        except BackendError as error:
            raise SetupError(str(error)) from error
    raise SetupError("No model is set up. Run actuneo.ai.setup() for a local model, or "
                     "actuneo.ai.configure(provider=...) to use your own API key.")
