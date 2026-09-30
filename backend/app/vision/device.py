"""Execution device selection.

CUDA is **probed, never assumed**. A deployment without a GPU, without a CUDA
build of the framework, or with a driver mismatch must fall back to CPU and say
so, rather than raising at first inference.

The resolved device travels in every inference result. That matters beyond
curiosity: CPU and GPU inference can produce marginally different floating-point
outputs, and a detection count that changes between machines is only debuggable
if you know which device produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from app.core.logging import get_logger

logger = get_logger(__name__)


class DeviceKind(StrEnum):
    """Execution device actually in use."""

    CPU = "cpu"
    CUDA = "cuda"


@dataclass(frozen=True, slots=True)
class ResolvedDevice:
    """The device selected, and why."""

    kind: DeviceKind
    name: str
    """Human-readable identity, e.g. ``NVIDIA RTX 4070`` or ``cpu``."""

    reason: str
    """Why this device was chosen. Present so a surprising CPU fallback on a
    machine that has a GPU is explained rather than merely observed."""

    @property
    def is_gpu(self) -> bool:
        """Whether inference runs on a GPU."""
        return self.kind is DeviceKind.CUDA

    @property
    def torch_device(self) -> str:
        """Device string in the form torch and ultralytics expect."""
        return "cuda:0" if self.is_gpu else "cpu"


def _probe_cuda() -> tuple[bool, str]:
    """Return whether CUDA is genuinely usable, and a description.

    Importing torch is enough to tell us whether a CUDA build is present, but
    not whether a device is actually reachable -- a CUDA-enabled build on a
    machine with no GPU reports ``is_available() == False``, which is exactly
    the case that must fall back rather than fail.
    """
    try:
        import torch
    except ImportError:
        return False, "torch is not installed, so no CUDA runtime is available"

    try:
        if not torch.cuda.is_available():
            return False, "torch is installed but reports no available CUDA device"
        name = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - driver-dependent
        return False, f"CUDA probe failed ({type(exc).__name__}); falling back to CPU"
    return True, name


def resolve_device(preference: str = "auto") -> ResolvedDevice:
    """Choose an execution device.

    Parameters
    ----------
    preference:
        ``auto`` probes for CUDA and falls back to CPU. ``cuda`` requests a GPU
        but still falls back with a stated reason rather than failing -- a
        missing GPU should degrade throughput, not deny perception entirely.
        ``cpu`` forces CPU.
    """
    if preference == "cpu":
        return ResolvedDevice(DeviceKind.CPU, "cpu", "CPU explicitly requested by configuration")

    available, description = _probe_cuda()

    if preference == "cuda":
        if available:
            return ResolvedDevice(DeviceKind.CUDA, description, "CUDA requested and available")
        logger.warning(
            "CUDA requested but unavailable; falling back to CPU",
            extra={"reason": description},
        )
        return ResolvedDevice(
            DeviceKind.CPU,
            "cpu",
            f"CUDA was requested but is unavailable: {description}",
        )

    if available:
        return ResolvedDevice(
            DeviceKind.CUDA, description, "CUDA detected and selected automatically"
        )
    return ResolvedDevice(DeviceKind.CPU, "cpu", f"No CUDA device detected: {description}")
