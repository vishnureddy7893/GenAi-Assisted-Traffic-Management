"""Pretrained YOLO detector via ONNX Runtime.

A second real backend, not a stub. Its existence is what makes the "detector is
replaceable through configuration" requirement true rather than aspirational:
the registry can swap between this and ultralytics with a config value, and
everything downstream is unaffected because both satisfy the same interface.

Why keep a second backend at all
--------------------------------
ONNX Runtime is roughly two orders of magnitude smaller to install than
PyTorch, which matters for a deployment target that only needs inference. The
cost is that pre- and post-processing are ours to implement, because an ONNX
graph is just the network: no letterboxing, no NMS, no label table.

Those steps are where subtle bugs live, so each is documented at the point it
happens. In particular the letterbox transform must be *inverted* exactly when
mapping boxes back, or every coordinate is offset by the padding.
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
from typing import Any

from app.core.errors import DataSourceError
from app.core.logging import get_logger
from app.vision.classes import COCO_CLASS_NAMES, map_label
from app.vision.detector import (
    BoundingBox,
    Detection,
    DetectionResult,
    DetectorInfo,
    VehicleDetector,
)
from app.vision.device import DeviceKind, ResolvedDevice
from app.vision.models import ModelIdentity, describe_weights, resolve_model_file

logger = get_logger(__name__)

BACKEND_NAME = "onnx"

#: Grey used to pad a letterboxed frame. The standard YOLO value; changing it
#: shifts the input distribution away from what the model was trained on.
_PAD_VALUE = 114


class OnnxYoloDetector(VehicleDetector):
    """YOLOv8-family detector running on ONNX Runtime."""

    def __init__(
        self,
        *,
        models_dir: Path,
        model_name: str = "yolov8n.onnx",
        device_preference: str = "auto",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        max_detections: int = 300,
    ) -> None:
        self._models_dir = Path(models_dir)
        self._model_name = model_name
        self._device_preference = device_preference
        self._confidence = confidence_threshold
        self._iou = iou_threshold
        self._image_size = image_size
        self._max_detections = max_detections

        self._session: Any = None
        self._input_name: str = ""
        self._device: ResolvedDevice | None = None
        self._identity: ModelIdentity | None = None

    # --- identity ---------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        """Whether a session is genuinely open."""
        return self._session is not None

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
            cls for cls in (map_label(name) for name in COCO_CLASS_NAMES) if cls is not None
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

    # --- lifecycle --------------------------------------------------------
    def load(self) -> None:
        """Open an inference session. Idempotent."""
        if self._session is not None:
            return

        try:
            import onnxruntime as ort
        except ImportError as exc:
            msg = (
                "The onnx backend is selected but onnxruntime is not installed. "
                "Install it with: pip install onnxruntime"
            )
            raise DataSourceError(msg, backend=BACKEND_NAME) from exc

        weights = resolve_model_file(self._models_dir, self._model_name)
        if not weights.exists():
            msg = (
                f"ONNX model {self._model_name!r} is not present at {weights}. "
                "Export one (for example: yolo export model=yolov8n.pt format=onnx) "
                "and place it in the models directory."
            )
            raise DataSourceError(msg, backend=BACKEND_NAME, model=self._model_name)

        available = list(ort.get_available_providers())
        wants_gpu = self._device_preference in ("auto", "cuda")
        use_cuda = wants_gpu and "CUDAExecutionProvider" in available
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_cuda
            else ["CPUExecutionProvider"]
        )

        try:
            session = ort.InferenceSession(str(weights), providers=providers)
        except Exception as exc:
            msg = f"Failed to open ONNX session: {type(exc).__name__}: {exc}"
            raise DataSourceError(msg, backend=BACKEND_NAME, model=self._model_name) from exc

        self._session = session
        self._input_name = session.get_inputs()[0].name
        self._device = (
            ResolvedDevice(DeviceKind.CUDA, "CUDAExecutionProvider", "CUDA provider available")
            if use_cuda
            else ResolvedDevice(
                DeviceKind.CPU,
                "CPUExecutionProvider",
                "CUDA execution provider not available"
                if wants_gpu
                else "CPU explicitly requested",
            )
        )
        self._identity = describe_weights(
            weights, name=self._model_name, backend=BACKEND_NAME, source="local file"
        )

        logger.info(
            "detector loaded",
            extra={
                "backend": BACKEND_NAME,
                "model": self._model_name,
                "sha256": self._identity.short_hash,
                "device": self._device.name,
            },
        )

    def close(self) -> None:
        """Release the session. Idempotent."""
        self._session = None
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
        if self._session is None:
            msg = "Detector is not loaded. Call load() before detect()."
            raise DataSourceError(msg, backend=BACKEND_NAME)

        import numpy as np

        started = time.perf_counter()
        tensor, scale, pad_x, pad_y = self._preprocess(frame)
        try:
            outputs = self._session.run(None, {self._input_name: tensor})
        except Exception as exc:
            msg = f"Inference failed: {type(exc).__name__}: {exc}"
            raise DataSourceError(msg, backend=BACKEND_NAME, sequence=sequence) from exc
        inference_ms = (time.perf_counter() - started) * 1000

        height, width = frame.shape[0], frame.shape[1]
        detections = self._postprocess(
            np.asarray(outputs[0]), scale, pad_x, pad_y, width, height
        )

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

    def _preprocess(self, frame: Any) -> tuple[Any, float, float, float]:
        """Letterbox to the model input size and return the inverse transform.

        Aspect ratio is preserved and the remainder padded, rather than the
        frame being squashed to a square. Squashing changes object proportions
        away from the training distribution and measurably costs recall on the
        elongated boxes traffic cameras produce.

        The scale and padding are returned so post-processing can invert this
        exactly. Forgetting the padding offset is the classic YOLO integration
        bug: boxes land plausibly but are consistently displaced.
        """
        import cv2
        import numpy as np

        height, width = frame.shape[0], frame.shape[1]
        size = self._image_size
        scale = min(size / width, size / height)
        new_w, new_h = round(width * scale), round(height * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size, size, 3), _PAD_VALUE, dtype=np.uint8)
        pad_x = (size - new_w) / 2
        pad_y = (size - new_h) / 2
        top, left = round(pad_y), round(pad_x)
        canvas[top : top + new_h, left : left + new_w] = resized

        tensor = canvas.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]
        return np.ascontiguousarray(tensor), scale, float(left), float(top)

    def _postprocess(
        self,
        output: Any,
        scale: float,
        pad_x: float,
        pad_y: float,
        width: int,
        height: int,
    ) -> tuple[Detection, ...]:
        """Decode raw output into detections, with NMS applied."""
        import cv2
        import numpy as np

        predictions = np.squeeze(output)
        if predictions.ndim != 2:
            return ()
        # YOLOv8 emits (4 + num_classes, num_anchors); transpose to per-anchor rows.
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.transpose()

        if predictions.shape[1] < 5:
            return ()

        class_scores = predictions[:, 4:]
        confidences = class_scores.max(axis=1)
        class_indices = class_scores.argmax(axis=1)

        keep = confidences >= self._confidence
        if not keep.any():
            return ()

        boxes_xywh = predictions[keep, :4]
        confidences = confidences[keep]
        class_indices = class_indices[keep]

        # Centre-form to corner-form, then invert the letterbox.
        cx, cy, w, h = (boxes_xywh[:, i] for i in range(4))
        x1 = (cx - w / 2 - pad_x) / scale
        y1 = (cy - h / 2 - pad_y) / scale
        x2 = (cx + w / 2 - pad_x) / scale
        y2 = (cy + h / 2 - pad_y) / scale

        x1 = np.clip(x1, 0, width)
        y1 = np.clip(y1, 0, height)
        x2 = np.clip(x2, 0, width)
        y2 = np.clip(y2, 0, height)

        nms_boxes = [
            [float(a), float(b), float(c - a), float(d - b)]
            for a, b, c, d in zip(x1, y1, x2, y2, strict=True)
        ]
        indices = cv2.dnn.NMSBoxes(
            nms_boxes,
            confidences.astype(float).tolist(),
            self._confidence,
            self._iou,
        )
        if len(indices) == 0:
            return ()

        flat = np.array(indices).flatten()[: self._max_detections]
        detections: list[Detection] = []
        for index in flat:
            label = (
                COCO_CLASS_NAMES[class_indices[index]]
                if class_indices[index] < len(COCO_CLASS_NAMES)
                else ""
            )
            road_user = map_label(label)
            if road_user is None:
                continue
            if x2[index] <= x1[index] or y2[index] <= y1[index]:
                continue
            detections.append(
                Detection(
                    bbox=BoundingBox(
                        x1=float(x1[index]),
                        y1=float(y1[index]),
                        x2=float(x2[index]),
                        y2=float(y2[index]),
                    ),
                    vehicle_class=road_user,
                    confidence=float(min(max(confidences[index], 0.0), 1.0)),
                )
            )
        return tuple(detections)
