"""Pretrained YOLO detector via ultralytics.

Implements the Phase-1 :class:`~app.vision.detector.VehicleDetector` contract.
No traffic logic lives here: this class answers *"what objects are visible in
this frame?"* and nothing else. It does not count, does not associate across
frames, and does not interpret.

**Pretrained only.** No training, no fine-tuning. The weights are stock YOLOv8
COCO weights, and the only project-specific step is mapping COCO labels onto our
road-user vocabulary (see :mod:`app.vision.classes`).

Class filtering happens *inside* the model rather than afterwards. Detecting all
80 COCO classes and discarding 74 wastes work, and — more importantly — lets
non-road-user boxes compete in non-maximum suppression, where a handbag
overlapping a pedestrian can suppress the pedestrian.
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
from typing import Any

from app.core.errors import DataSourceError
from app.core.logging import get_logger
from app.vision.classes import map_label, road_user_class_indices
from app.vision.detector import (
    BoundingBox,
    Detection,
    DetectionResult,
    DetectorInfo,
    VehicleDetector,
)
from app.vision.device import ResolvedDevice, resolve_device
from app.vision.models import ModelIdentity, describe_weights, ensure_weights

logger = get_logger(__name__)

BACKEND_NAME = "ultralytics"


def _as_ultralytics_input(frame: Any) -> Any:
    """Convert an ingestion frame into the channel order ultralytics expects.

    This project's ingestion layer yields **RGB** arrays, which is the right
    contract: it is what PIL produces, what the ONNX backend consumes directly,
    and what every consumer downstream reads. The ultralytics Python API is the
    odd one out -- it documents raw numpy input as **BGR** and flips the
    channels itself before inference, because it was built around OpenCV.

    Feeding it an RGB array therefore delivers a channel-swapped image to the
    network. The model still detects *something*, which is exactly why this is
    easy to miss: it degrades recall rather than failing. Measured on the
    reference junction frame at 1280 px, correcting the order took the detection
    count from 49 to 56 -- and the corrected numbers are byte-identical to
    passing the PIL image, which the library converts properly. That equality is
    the proof that this conversion is a fix and not a tuning knob.

    Anything that is not an HWC three-channel array is passed through untouched:
    a PIL image or a file path is already handled correctly by the library, and
    a test double's payload is not ours to reinterpret.
    """
    shape = getattr(frame, "shape", None)
    if shape is None or len(shape) != 3 or shape[2] != 3:
        return frame

    import numpy as np

    return np.ascontiguousarray(frame[:, :, ::-1])


class UltralyticsYoloDetector(VehicleDetector):
    """YOLO object detector backed by the ultralytics package."""

    def __init__(
        self,
        *,
        models_dir: Path,
        model_name: str = "yolov8s.pt",
        device_preference: str = "auto",
        confidence_threshold: float = 0.20,
        iou_threshold: float = 0.45,
        image_size: int = 1280,
        max_detections: int = 300,
    ) -> None:
        self._models_dir = Path(models_dir)
        self._model_name = model_name
        self._device_preference = device_preference
        self._confidence = confidence_threshold
        self._iou = iou_threshold
        self._image_size = image_size
        self._max_detections = max_detections

        self._model: Any = None
        self._device: ResolvedDevice | None = None
        self._identity: ModelIdentity | None = None
        self._class_indices: tuple[int, ...] = ()

    # --- identity ---------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        """Whether weights are genuinely loaded. Never optimistic."""
        return self._model is not None

    @property
    def device(self) -> ResolvedDevice | None:
        """Resolved execution device, once loaded."""
        return self._device

    @property
    def identity(self) -> ModelIdentity | None:
        """Measured identity of the loaded weights, once loaded."""
        return self._identity

    @property
    def info(self) -> DetectorInfo:
        """Identity of this detector, for provenance."""
        supported = frozenset(
            cls
            for cls in (map_label(name) for name in self._label_names().values())
            if cls is not None
        )
        return DetectorInfo(
            name=f"{BACKEND_NAME}:{self._model_name}",
            version=(self._identity.short_hash or "unloaded") if self._identity else "unloaded",
            weights_id=self._identity.sha256 if self._identity else None,
            input_width=self._image_size,
            input_height=self._image_size,
            supported_classes=supported,
            device=self._device.name if self._device else None,
            confidence_threshold=self._confidence,
        )

    def _label_names(self) -> dict[int, str]:
        """Class index to label, from the loaded model when available."""
        if self._model is None:
            from app.vision.classes import COCO_CLASS_NAMES

            return dict(enumerate(COCO_CLASS_NAMES))
        names = getattr(self._model, "names", None)
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, list | tuple):
            return {i: str(v) for i, v in enumerate(names)}
        from app.vision.classes import COCO_CLASS_NAMES

        return dict(enumerate(COCO_CLASS_NAMES))

    # --- lifecycle --------------------------------------------------------
    def load(self) -> None:
        """Load weights and prepare for inference. Idempotent."""
        if self._model is not None:
            return

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            msg = (
                "The ultralytics backend is selected but the package is not "
                "installed. Install it with: pip install ultralytics"
            )
            raise DataSourceError(msg, backend=BACKEND_NAME) from exc

        weights = ensure_weights(self._models_dir, self._model_name)
        self._device = resolve_device(self._device_preference)

        started = time.perf_counter()
        try:
            model = YOLO(str(weights))
            model.to(self._device.torch_device)
        except Exception as exc:
            msg = (
                f"Failed to load model {self._model_name!r} on "
                f"{self._device.kind.value}: {type(exc).__name__}: {exc}"
            )
            raise DataSourceError(msg, backend=BACKEND_NAME, model=self._model_name) from exc

        self._model = model
        # Torch initialises lazily, so the first inference costs seconds while
        # steady-state costs tens of milliseconds. Paying that here means every
        # reported inference_ms is comparable, instead of the first frame of
        # every video carrying a cold-start cost that looks like a slow model.
        self._warmup(model)
        self._identity = describe_weights(
            weights,
            name=self._model_name,
            backend=BACKEND_NAME,
            source="ultralytics release asset",
        )
        self._class_indices = road_user_class_indices(self._label_names())

        logger.info(
            "detector loaded",
            extra={
                "backend": BACKEND_NAME,
                "model": self._model_name,
                "sha256": self._identity.short_hash,
                "device": self._device.name,
                "device_kind": self._device.kind.value,
                "load_ms": round((time.perf_counter() - started) * 1000, 1),
                "road_user_classes": len(self._class_indices),
            },
        )

    def _warmup(self, model: Any) -> None:
        """Run one throwaway inference on a blank frame.

        Failure here is logged and swallowed: a warmup is an optimisation, and
        refusing to load an otherwise working model because the warmup pass
        stumbled would trade a real capability for a performance nicety.
        """
        try:
            import numpy as np

            blank = np.zeros((self._image_size, self._image_size, 3), dtype=np.uint8)
            model.predict(
                blank,
                imgsz=self._image_size,
                device=self._device.torch_device if self._device else "cpu",
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("detector warmup failed", extra={"error": repr(exc)})

    def close(self) -> None:
        """Release model resources. Idempotent."""
        self._model = None
        self._identity = None
        self._device = None

    # --- inference --------------------------------------------------------
    def detect(
        self,
        frame: Any,
        *,
        source_id: str,
        sequence: int,
        captured_at: dt.datetime,
    ) -> DetectionResult:
        """Detect road users in one RGB frame."""
        if self._model is None:
            msg = "Detector is not loaded. Call load() before detect()."
            raise DataSourceError(msg, backend=BACKEND_NAME)

        started = time.perf_counter()
        try:
            raw = self._model.predict(
                _as_ultralytics_input(frame),
                conf=self._confidence,
                iou=self._iou,
                imgsz=self._image_size,
                max_det=self._max_detections,
                classes=list(self._class_indices) or None,
                device=self._device.torch_device if self._device else "cpu",
                verbose=False,
            )
        except Exception as exc:
            msg = f"Inference failed: {type(exc).__name__}: {exc}"
            raise DataSourceError(msg, backend=BACKEND_NAME, sequence=sequence) from exc

        inference_ms = (time.perf_counter() - started) * 1000
        detections = self._convert(raw)

        has_shape = hasattr(frame, "shape")
        height = frame.shape[0] if has_shape else None
        width = frame.shape[1] if has_shape else None
        return DetectionResult(
            source_id=source_id,
            sequence=sequence,
            captured_at=captured_at,
            detections=detections,
            inference_ms=round(inference_ms, 2),
            frame_width=width,
            frame_height=height,
            metadata={
                "backend": BACKEND_NAME,
                "model": self._model_name,
                "model_sha256": self._identity.sha256 if self._identity else None,
                "device": self._device.name if self._device else None,
                "device_kind": self._device.kind.value if self._device else None,
                "image_size": self._image_size,
                "confidence_threshold": self._confidence,
                "iou_threshold": self._iou,
            },
        )

    def _convert(self, raw: Any) -> tuple[Detection, ...]:
        """Convert ultralytics output into our detection records.

        Unmappable classes are dropped rather than coerced: a detection we
        cannot name as a road user is not one we should reason about.
        """
        labels = self._label_names()
        detections: list[Detection] = []

        for result in raw:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                try:
                    x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                    confidence = float(box.conf[0].item())
                    class_index = int(box.cls[0].item())
                except (AttributeError, IndexError, TypeError, ValueError):
                    continue

                road_user = map_label(labels.get(class_index, ""))
                if road_user is None:
                    continue

                detections.append(
                    Detection(
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        vehicle_class=road_user,
                        confidence=min(max(confidence, 0.0), 1.0),
                    )
                )

        return tuple(detections)
