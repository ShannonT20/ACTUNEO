"""
Hardware detection

Finds the memory, processor, graphics card and free disk space of the
machine, on Windows, macOS and Linux. Every probe is allowed to fail: what
could not be detected is recorded in :attr:`HardwareProfile.notes` and the
caller falls back to a safe choice.
"""

import logging
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

logger = logging.getLogger("actuneo.ai")

GIB = 1024 ** 3


@dataclass
class GPUInfo:
    """A graphics processor that can run a model."""
    name: str
    vendor: str                 # "nvidia" or "apple"
    vram_total_gb: float
    vram_free_gb: float
    unified: bool = False       # memory shared with the system (Apple Silicon)


@dataclass
class HardwareProfile:
    """What was detected about the machine."""
    system: str = ""
    machine: str = ""
    cpu_count: int = 1
    cpu_physical: Optional[int] = None
    ram_total_gb: Optional[float] = None
    ram_available_gb: Optional[float] = None
    gpu: Optional[GPUInfo] = None
    disk_free_gb: Optional[float] = None
    detection_ok: bool = True
    notes: List[str] = field(default_factory=list)

    def describe(self) -> str:
        """One line for the user, for example '16 GB RAM, NVIDIA RTX 3050 (4 GB VRAM), 120 GB free disk'."""
        parts = []
        if self.ram_total_gb is None:
            parts.append("RAM unknown")
        else:
            ram = f"{self.ram_total_gb:.0f} GB RAM"
            if self.ram_available_gb is not None:
                ram += f" ({self.ram_available_gb:.1f} GB free)"
            parts.append(ram)
        parts.append(f"{self.cpu_physical or self.cpu_count} CPU cores")
        if self.gpu is None:
            parts.append("no supported GPU")
        elif self.gpu.unified:
            parts.append(f"{self.gpu.name} (unified memory)")
        else:
            parts.append(f"{self.gpu.name} ({self.gpu.vram_total_gb:.0f} GB VRAM, "
                         f"{self.gpu.vram_free_gb:.1f} GB free)")
        if self.disk_free_gb is not None:
            parts.append(f"{self.disk_free_gb:.0f} GB free disk")
        return ", ".join(parts)

    def to_dict(self) -> dict:
        """The profile as plain data, for the configuration file."""
        return asdict(self)


def _detect_memory() -> Tuple[Optional[float], Optional[float]]:
    """Total and available memory in GiB, or None where unknown."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return memory.total / GIB, memory.available / GIB
    except Exception as error:      # psutil missing or failing
        logger.debug("psutil memory detection failed: %s", error)
    try:
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / GIB
            available = None
            if "SC_AVPHYS_PAGES" in os.sysconf_names:
                available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / GIB
            return total, available
    except (ValueError, OSError) as error:
        logger.debug("sysconf memory detection failed: %s", error)
    return None, None


def _physical_cores() -> Optional[int]:
    """Physical processor cores, not counting hyperthreads."""
    try:
        import psutil
        return psutil.cpu_count(logical=False)
    except Exception as error:
        logger.debug("physical core detection failed: %s", error)
        return None


def _detect_nvidia() -> Optional[GPUInfo]:
    """The NVIDIA card with the most free memory, from nvidia-smi or pynvml."""
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
        cards = []
        for line in output.strip().splitlines():
            name, total, free = [part.strip() for part in line.split(",")]
            cards.append(GPUInfo(name, "nvidia", float(total) / 1024, float(free) / 1024))
        if cards:
            return max(cards, key=lambda card: card.vram_free_gb)
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        logger.debug("nvidia-smi detection failed: %s", error)
    try:
        import pynvml
        pynvml.nvmlInit()
        cards = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            cards.append(GPUInfo(name, "nvidia", memory.total / GIB, memory.free / GIB))
        pynvml.nvmlShutdown()
        if cards:
            return max(cards, key=lambda card: card.vram_free_gb)
    except Exception as error:      # pynvml missing, or no driver
        logger.debug("pynvml detection failed: %s", error)
    return None


def _detect_apple(ram_total: Optional[float], ram_available: Optional[float]) -> Optional[GPUInfo]:
    """Apple Silicon: the GPU shares the system's memory."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None
    name = "Apple Silicon"
    try:
        brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        if brand:
            name = brand
    except (OSError, subprocess.SubprocessError) as error:
        logger.debug("sysctl brand detection failed: %s", error)
    if ram_total is None:
        return None
    return GPUInfo(name, "apple", ram_total,
                   ram_available if ram_available is not None else ram_total / 2,
                   unified=True)


def _disk_free(path: Union[str, Path]) -> Optional[float]:
    """Free space in GiB on the drive that holds (or will hold) ``path``."""
    folder = Path(path).expanduser()
    while not folder.exists() and folder != folder.parent:
        folder = folder.parent
    try:
        return shutil.disk_usage(folder).free / GIB
    except OSError as error:
        logger.debug("disk detection failed: %s", error)
        return None


def detect_hardware(models_dir: Union[str, Path, None] = None) -> HardwareProfile:
    """
    Detect the machine's memory, processor, GPU and free disk space.

    AMD and Intel graphics are treated as absent: models then run on the
    processor.

    Args:
        models_dir: Folder where models will be stored, used to measure free
            disk space (the home folder if omitted)

    Returns:
        The profile. ``detection_ok`` is False when memory could not be
        measured, in which case nothing larger than the smallest model
        should be chosen.
    """
    profile = HardwareProfile(system=platform.system(), machine=platform.machine(),
                              cpu_count=os.cpu_count() or 1)
    profile.cpu_physical = _physical_cores()
    total, available = _detect_memory()
    profile.ram_total_gb = total
    profile.ram_available_gb = available
    if total is None:
        profile.detection_ok = False
        profile.notes.append("Memory could not be detected.")
    elif available is None:
        profile.ram_available_gb = total / 2
        profile.notes.append("Free memory could not be measured; half of the total is assumed.")

    profile.gpu = _detect_nvidia() or _detect_apple(profile.ram_total_gb,
                                                    profile.ram_available_gb)
    profile.disk_free_gb = _disk_free(models_dir or Path.home())
    if profile.disk_free_gb is None:
        profile.notes.append("Free disk space could not be measured.")
    logger.info("Detected hardware: %s", profile.describe())
    return profile
