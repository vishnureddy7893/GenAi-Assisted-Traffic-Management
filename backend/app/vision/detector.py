"""Vehicle detection interface. Backends are implemented in Phase 3.

Technology is deliberately not chosen here
------------------------------------------
No detector backend is named in this module. The choice between a YOLO family
model, an RT-DETR variant, or a classical background-subtraction pipeline cannot
be made responsibly before the dataset exists, because the criteria that decide
it are properties of the data and the deployment, not preferences:

* **Accuracy on this footage.** Traffic-camera video is small-object, oblique,
  heavily occluded and often low-resolution. COCO mAP is close to uninformative
  for it; what matters is recall on distant and partially occluded vehicles in
  the specific camera geometry.
* **Throughput against the hardware actually available.** A model that needs a
  discrete GPU is not a candidate if the deployment target is a laptop CPU.
  Whether that constraint binds is a fact about the environment, and it is
  measured, not assumed.
* **Class coverage.** If the dataset's vehicle taxonomy is unusual -- an
  auto-rickshaw class, say -- a pretrained model may need fine-tuning, which
  changes the cost of the whole option.
* **Licence and deployment weight.** Some strong models carry licences that are
  awkward for an academic deliverable.

So this module fixes the *contract* -- what a detector consumes and produces --
and Phase 3 selected a backend against measured numbers on the real dataset. The
contract is intentionally narrow enough that swapping backends is a
constructor change, which is the property that makes an honest comparison
affordable at all.

Frames are numpy arrays and this module does not import numpy: typing the
payload loosely keeps the interface importable in a deployment that ingests only
tabular data and has no CV stack installed.
"""

from __future__ import annotations

import abc
import datetime as dt
from dataclasses import dataclass, field
from typing import Annotated, Any

from pydantic import Field

from app.domain.base import DomainModel
from app.domain.enums import VehicleClass


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned box in *pixel* coordinates of the source frame.

    Pixels, not metres, and the distinction is load-bearing: converting to ground
    coordinates requires calibration that may not exist. Keeping detector output
    in its native units means an uncalibrated pipeline cannot accidentally
    present pixel measurements as metric ones.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            msg = f"degenerate bounding box: ({self.x1},{self.y1})-({self.x2},{self.y2})"
            raise ValueError(msg)

    @property
    def width(self) -> float:
        """Box width in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Box height in pixels."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Box area in square pixels."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """Box centre in pixels."""
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def ground_contact_point(self) -> tuple[float, float]:
        """Bottom-centre of the box: the best available proxy for where the
        vehicle meets the road.

        Used instead of the centroid for ground-plane projection. A box centroid
        sits roughly at vehicle mid-height, so projecting it through a
        ground-plane homography places the vehicle metres beyond its true
        position -- an error that grows with distance from the camera and biases
        every derived speed and queue length.
        """
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def iou(self, other: BoundingBox) -> float:
        """Intersection over union with another box."""
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        intersection = (ix2 - ix1) * (iy2 - iy1)
        union = self.area + other.area - intersection
        return 0.0 if union <= 0 else intersection / union


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected object in one frame."""

    bbox: BoundingBox
    vehicle_class: VehicleClass
    confidence: float
    class_scores: dict[VehicleClass, float] = field(default_factory=dict)
    """Full score distribution when the backend exposes one. Retained because a
    car/truck confusion at 0.51/0.49 is materially different from 0.99/0.01 for
    anything that weights by vehicle type, and the argmax alone destroys that."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            msg = f"confidence must be in [0,1], got {self.confidence}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Detector output for one frame."""

    source_id: str
    sequence: int
    captured_at: dt.datetime
    detections: tuple[Detection, ...]
    inference_ms: float | None = None
    frame_width: int | None = None
    frame_height: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        """Number of detections in this frame."""
        return len(self.detections)

    def above_confidence(self, threshold: float) -> tuple[Detection, ...]:
        """Detections at or above a confidence threshold."""
        return tuple(d for d in self.detections if d.confidence >= threshold)


class DetectorInfo(DomainModel):
    """Identity of a detector implementation.

    Serialised into observation provenance, so a count can always be attributed
    to the exact model and weights that produced it. Without this, a change in
    measured traffic cannot be distinguished from a change in the detector.
    """

    name: Annotated[str, Field(min_length=1, max_length=100)]
    version: Annotated[str, Field(min_length=1, max_length=50)]
    weights_id: Annotated[str, Field(max_length=200)] | None = None
    """Checksum or registry id of the weights file. The version string alone is
    not enough: two runs of the same 'v1' can use different fine-tuned weights."""

    input_width: Annotated[int, Field(gt=0)] | None = None
    input_height: Annotated[int, Field(gt=0)] | None = None
    supported_classes: frozenset[VehicleClass] = frozenset()
    device: Annotated[str, Field(max_length=50)] | None = None
    confidence_threshold: Annotated[float, Field(ge=0.0, le=1.0)] | None = None


class VehicleDetector(abc.ABC):
    """Detects vehicles in a single frame.

    Deliberately stateless with respect to time: a detector sees one frame and
    knows nothing about the previous one. Temporal association is the tracker's
    job, and keeping the two apart is what allows either to be replaced -- or
    bypassed entirely, when the input is a pre-annotated detection dataset.
    """

    @property
    @abc.abstractmethod
    def info(self) -> DetectorInfo:
        """Identity of this detector, for provenance."""

    @abc.abstractmethod
    def load(self) -> None:
        """Load weights and prepare for inference. Idempotent."""

    @abc.abstractmethod
    def detect(
        self,
        frame: Any,
        *,
        source_id: str,
        sequence: int,
        captured_at: dt.datetime,
    ) -> DetectionResult:
        """Detect vehicles in one frame.

        ``captured_at`` is passed in rather than read from a clock: the detector
        must not decide which time base it is operating on, or recorded replay
        would be stamped with wall-clock time.
        """

    def warmup(self, frame: Any) -> None:
        """Run one throwaway inference to trigger lazy initialisation.

        Optional but worth having: the first inference on most backends is
        several times slower than steady state, and without a warmup that cost
        lands on the first real frame and distorts the first latency figures.
        """
        return None

    @abc.abstractmethod
    def close(self) -> None:
        """Release model resources. Idempotent."""
