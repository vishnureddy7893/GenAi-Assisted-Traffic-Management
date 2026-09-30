"""Computer vision pipeline interfaces.

Three stages, deliberately separable:

    frame -> VehicleDetector    -> Detection
          -> VehicleTracker     -> Track          (identity across time)
          -> TrafficStateExtractor -> JunctionObservation

Separable because a pre-annotated detection dataset can enter at stage two and a
tabular count dataset can bypass all three, while every path still produces the
same JunctionObservation. No backend technology is named by these interfaces;
Phase 3 selected one against measured numbers on the real dataset, and
Phase 4 implements TrafficStateExtractor in :mod:`app.analytics`.
"""

from app.vision.detector import (
    BoundingBox,
    Detection,
    DetectionResult,
    DetectorInfo,
    VehicleDetector,
)
from app.vision.extractor import (
    CountingLine,
    ExtractionContext,
    ImagePoint,
    QueueZone,
    TrafficStateExtractor,
    ZoneConfiguration,
)
from app.vision.tracker import (
    Track,
    TrackerInfo,
    TrackerStats,
    TrackPoint,
    VehicleTracker,
)

__all__ = [
    "BoundingBox",
    "CountingLine",
    "Detection",
    "DetectionResult",
    "DetectorInfo",
    "ExtractionContext",
    "ImagePoint",
    "QueueZone",
    "Track",
    "TrackPoint",
    "TrackerInfo",
    "TrackerStats",
    "TrafficStateExtractor",
    "VehicleDetector",
    "VehicleTracker",
    "ZoneConfiguration",
]
