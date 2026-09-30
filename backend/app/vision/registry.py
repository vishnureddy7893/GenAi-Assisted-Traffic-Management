"""Detector backend registry and construction.

No module outside this one names a concrete detector class. Everything else
depends on the :class:`~app.vision.detector.VehicleDetector` interface and
receives an instance built here, which is what makes the backend genuinely
swappable through configuration rather than nominally so.

``auto`` resolution picks the first backend whose dependencies actually import.
It is a *probe*, not a guess: a backend is only offered once its import has
succeeded, so selecting one can never fail later with an ImportError at first
inference.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from pydantic import Field

from app.core.config import Settings
from app.core.errors import NotAvailableYetError, ValidationFailedError
from app.core.logging import get_logger
from app.domain.base import DomainModel
from app.vision.detector import VehicleDetector

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BackendSpec:
    """One registered detector backend."""

    name: str
    description: str
    required_modules: tuple[str, ...]
    default_model: str
    factory: Callable[[Settings, Path], VehicleDetector]

    @property
    def missing_modules(self) -> tuple[str, ...]:
        """Required modules that are not importable in this environment."""
        return tuple(
            module
            for module in self.required_modules
            if importlib.util.find_spec(module) is None
        )

    @property
    def is_available(self) -> bool:
        """Whether every dependency this backend needs is installed."""
        return not self.missing_modules


class BackendStatus(DomainModel):
    """API-facing availability of one backend."""

    name: Annotated[str, Field(min_length=1, max_length=40)]
    description: Annotated[str, Field(max_length=300)]
    available: bool
    missing_modules: tuple[str, ...] = ()
    default_model: Annotated[str, Field(max_length=120)]
    is_selected: bool = False


def _build_ultralytics(settings: Settings, models_dir: Path) -> VehicleDetector:
    from app.vision.backends.ultralytics_detector import UltralyticsYoloDetector

    return UltralyticsYoloDetector(
        models_dir=models_dir,
        model_name=settings.detector_model,
        device_preference=settings.detector_device,
        confidence_threshold=settings.detector_confidence_threshold,
        iou_threshold=settings.detector_iou_threshold,
        image_size=settings.detector_image_size,
        max_detections=settings.detector_max_detections,
    )


def _build_onnx(settings: Settings, models_dir: Path) -> VehicleDetector:
    from app.vision.backends.onnx_detector import OnnxYoloDetector

    model = settings.detector_model
    # The configured model may name .pt weights while this backend needs a
    # graph. Correct the extension rather than failing obscurely inside the
    # session constructor.
    if model.endswith(".pt"):
        model = model[: -len(".pt")] + ".onnx"

    return OnnxYoloDetector(
        models_dir=models_dir,
        model_name=model,
        device_preference=settings.detector_device,
        confidence_threshold=settings.detector_confidence_threshold,
        iou_threshold=settings.detector_iou_threshold,
        image_size=settings.detector_image_size,
        max_detections=settings.detector_max_detections,
    )


#: Registered backends, in ``auto`` preference order. Ultralytics first: it can
#: fetch its own weights, so it works from a clean checkout, whereas the ONNX
#: backend needs a graph to be exported first.
BACKENDS: tuple[BackendSpec, ...] = (
    BackendSpec(
        name="ultralytics",
        description=(
            "Pretrained YOLO via the ultralytics package. Fetches its own weights "
            "and runs on PyTorch, using CUDA when available."
        ),
        required_modules=("ultralytics", "torch"),
        default_model="yolov8s.pt",
        factory=_build_ultralytics,
    ),
    BackendSpec(
        name="onnx",
        description=(
            "Pretrained YOLO graph on ONNX Runtime. Much smaller to install than "
            "PyTorch; requires an exported .onnx model in the models directory."
        ),
        required_modules=("onnxruntime", "cv2", "numpy"),
        default_model="yolov8n.onnx",
        factory=_build_onnx,
    ),
)


def backend_by_name(name: str) -> BackendSpec | None:
    """Look up a registered backend."""
    for spec in BACKENDS:
        if spec.name == name:
            return spec
    return None


def resolve_backend(settings: Settings) -> BackendSpec | None:
    """Choose the backend named by configuration, or the first available one.

    Returns ``None`` when detection is disabled or nothing is installed. The
    caller then reports detection as unavailable -- it must not substitute an
    empty detector, because "no detections" and "no detector" are different
    claims and only one of them is about the road.
    """
    preference = settings.detector_backend

    if preference == "none":
        return None

    if preference != "auto":
        spec = backend_by_name(preference)
        if spec is None:
            known = ", ".join(b.name for b in BACKENDS)
            msg = f"Unknown detector backend {preference!r}. Available: {known}."
            raise ValidationFailedError(msg, backend=preference)
        return spec

    for spec in BACKENDS:
        if spec.is_available:
            return spec
    return None


def backend_statuses(settings: Settings) -> tuple[BackendStatus, ...]:
    """Availability of every registered backend, for the status endpoint."""
    try:
        selected = resolve_backend(settings)
    except ValidationFailedError:
        selected = None
    selected_name = selected.name if selected else None

    return tuple(
        BackendStatus(
            name=spec.name,
            description=spec.description,
            available=spec.is_available,
            missing_modules=spec.missing_modules,
            default_model=spec.default_model,
            is_selected=spec.name == selected_name,
        )
        for spec in BACKENDS
    )


def create_detector(settings: Settings, models_dir: Path) -> VehicleDetector:
    """Construct the configured detector.

    Raises :class:`~app.core.errors.NotAvailableYetError` when no backend is
    usable, so an API caller receives an accurate 501 naming what is missing
    rather than an empty detection list.
    """
    spec = resolve_backend(settings)
    if spec is None:
        if settings.detector_backend == "none":
            msg = (
                "Object detection is disabled by configuration "
                "(detector_backend='none')."
            )
        else:
            names = ", ".join(f"{b.name} (needs {', '.join(b.missing_modules)})" for b in BACKENDS)
            msg = (
                "No detector backend is available in this environment. "
                f"Install the dependencies for one of: {names}."
            )
        raise NotAvailableYetError(msg, phase=3, phase_name="Computer Vision")

    if not spec.is_available:
        msg = (
            f"Detector backend {spec.name!r} is selected but its dependencies are "
            f"missing: {', '.join(spec.missing_modules)}."
        )
        raise NotAvailableYetError(msg, phase=3, phase_name="Computer Vision")

    return spec.factory(settings, models_dir)
