"""Detection quality assessment: which boxes may be counted as vehicles.

Why this layer exists
---------------------
A detector answers "does this look like a car?". It does not answer "is this a
distinct road vehicle that a traffic count may include?", and on a real junction
frame those come apart in specific, repeatable ways: a building detected as a
bus, one vehicle detected twice under two class labels, a box clipped by the
frame edge with nothing identifiable inside it.

Raising the detector's confidence threshold does not fix this. On the reference
frame the false positives are spread from 0.20 to 0.37 and real vehicles run
from 0.26 to 0.87, so the two populations overlap: every threshold that removes
the last false positive also removes ten real vehicles. That measurement is what
motivated separating **reporting** from **confirmation**.

Two tiers, and nothing is deleted
---------------------------------
The detector reports everything above its threshold. This layer sorts what it
reported into three tiers and attaches the reason:

* :attr:`DetectionQuality.CONFIRMED` -- may be counted as a road user.
* :attr:`DetectionQuality.PROVISIONAL` -- genuinely detected, but not evidence a
  count may rest on: below its class's confirmation floor, a duplicate of
  another box, or clipped by the frame edge without the confidence to carry it.
* :attr:`DetectionQuality.REJECTED` -- geometrically impossible for a road
  vehicle in this view.

Nothing is dropped. Every detection is still returned, still drawn on the
overlay, and still carries its own confidence -- what changes is that the
downstream measurement layer consumes the confirmed tier only. An operator who
disagrees with a verdict can see the box and the reason and lower a threshold,
which is not possible when a filter silently deletes.

What this deliberately does not do
----------------------------------
No rule here says "small boxes are not vehicles". The smallest confirmed vehicle
on the reference frame is 15x10 px -- a real car far up the avenue -- and a size
cut sharp enough to remove the 8x8 false positive would remove it too. Size
enters only through the *maximum* area check, which encodes something physical:
one road vehicle cannot fill a large fraction of a wide-area junction view.

Nor is anything here tuned to a location. There are no coordinates, no regions
of interest and no per-image exceptions: every rule is a property of a box, a
class, or a pair of boxes, and every threshold comes from configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated

from pydantic import Field

from app.core.config import Settings
from app.domain.base import DomainModel
from app.domain.enums import VehicleClass
from app.vision.detector import BoundingBox, Detection


class DetectionQuality(StrEnum):
    """Whether a detection may be counted as a road user."""

    CONFIRMED = "confirmed"
    """Passed every configured check. The only tier a measurement may count."""

    PROVISIONAL = "provisional"
    """Detected, reported, and not countable. A judgement about evidence
    strength rather than about physics: the object may well be there."""

    REJECTED = "rejected"
    """Geometrically impossible for a road vehicle in this view. Still
    reported, because a rejected box is a fact about the detector that an
    operator tuning the system needs to see."""


class QualityFlag(StrEnum):
    """Why a detection is not confirmed. Codes, not prose."""

    BELOW_CLASS_CONFIDENCE = "below_class_confidence"
    """Under the confirmation floor configured for its class. Classes are not
    equally reliable in an oblique traffic view, and one global number cannot
    express that."""

    DUPLICATE = "duplicate"
    """Overlaps a higher-confidence detection enough to be the same object.
    Non-maximum suppression runs per class inside the model, so one vehicle
    labelled ``car`` by one head and ``truck`` by another survives it."""

    IMPLAUSIBLE_AREA = "implausible_area"
    """Covers more of the frame than one road vehicle can in this view."""

    IMPLAUSIBLE_ASPECT = "implausible_aspect"
    """Degenerate proportions -- a sliver or a column, not a vehicle."""

    EDGE_TRUNCATED_LOW_CONFIDENCE = "edge_truncated_low_confidence"
    """Clipped by the frame border without the confidence to carry it. A
    truncated box shows part of an object, so the same score is weaker
    evidence; a confident vehicle entering the frame is unaffected."""


class ClassConfidenceFloor(DomainModel):
    """One class's confirmation floor, and where the number came from."""

    road_user_class: VehicleClass
    floor: Annotated[float, Field(ge=0.0, le=1.0)]
    is_class_specific: bool
    """``False`` when the class falls back to the detector's global threshold
    because no per-class evidence exists for it."""


class QualityPolicy(DomainModel):
    """The quality rules in force, as an inspectable value.

    Recorded alongside results rather than read from configuration when they are
    displayed: a count produced under one policy and read against another is
    unreproducible, and the drift would be invisible.
    """

    enabled: bool = True

    confirm_confidence: dict[VehicleClass, float] = Field(default_factory=dict)
    """Per-class confirmation floors. A class absent from this map uses
    :attr:`default_confidence`."""

    default_confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    """Floor for classes with no per-class evidence. Equal to the detector's own
    threshold, so those classes are unaffected by this layer."""

    max_area_fraction: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    """Largest share of the frame one vehicle may cover. ``0`` disables."""

    min_aspect_ratio: Annotated[float, Field(ge=0.0)] = 0.0
    max_aspect_ratio: Annotated[float, Field(ge=0.0)] = 0.0
    """Width-over-height band. ``0`` on either disables that side."""

    duplicate_iou: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    duplicate_containment: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    """Two boxes are one object at or above either measure. Containment catches
    what IoU misses: a small box nested inside a large one scores a low IoU
    while still being a second count of one vehicle."""

    edge_margin_px: Annotated[int, Field(ge=0)] = 0
    edge_confidence_margin: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0

    def floor_for(self, road_user_class: VehicleClass) -> float:
        """Confirmation floor for one class."""
        return self.confirm_confidence.get(road_user_class, self.default_confidence)

    def floors(self) -> tuple[ClassConfidenceFloor, ...]:
        """Every class's floor, and whether it is class-specific."""
        return tuple(
            ClassConfidenceFloor(
                road_user_class=item,
                floor=self.floor_for(item),
                is_class_specific=item in self.confirm_confidence,
            )
            for item in VehicleClass
        )


@dataclass(frozen=True, slots=True)
class AssessedDetection:
    """One detection, its verdict, and the geometry the verdict rests on."""

    detection: Detection
    quality: DetectionQuality
    flags: tuple[QualityFlag, ...]
    reason: str | None
    area_fraction: float
    aspect_ratio: float | None
    touches_frame_edge: bool
    duplicate_of_index: int | None = None
    """Index of the detection this one duplicates, so the pairing is checkable
    rather than asserted."""

    @property
    def is_confirmed(self) -> bool:
        """Whether this may be counted as a road user."""
        return self.quality is DetectionQuality.CONFIRMED


def policy_from(settings: Settings) -> QualityPolicy:
    """Build the policy in force from configuration.

    A class-specific floor is set only where measurement supports one. Classes
    without evidence fall back to the detector's own threshold rather than to an
    invented number -- a floor nobody measured is a guess wearing a policy's
    clothes.
    """
    specific: dict[VehicleClass, float] = {}
    for road_user_class, configured in (
        (VehicleClass.CAR, settings.detection_confirm_confidence_car),
        (VehicleClass.MOTORCYCLE, settings.detection_confirm_confidence_motorcycle),
        (VehicleClass.BUS, settings.detection_confirm_confidence_bus),
        (VehicleClass.TRUCK, settings.detection_confirm_confidence_truck),
        (VehicleClass.BICYCLE, settings.detection_confirm_confidence_bicycle),
        (VehicleClass.PERSON, settings.detection_confirm_confidence_person),
    ):
        if configured is not None:
            specific[road_user_class] = configured

    return QualityPolicy(
        enabled=settings.detection_quality_enabled,
        confirm_confidence=specific,
        default_confidence=settings.detector_confidence_threshold,
        max_area_fraction=settings.detection_max_area_fraction,
        min_aspect_ratio=settings.detection_min_aspect_ratio,
        max_aspect_ratio=settings.detection_max_aspect_ratio,
        duplicate_iou=settings.detection_duplicate_iou,
        duplicate_containment=settings.detection_duplicate_containment,
        edge_margin_px=settings.detection_edge_margin_px,
        edge_confidence_margin=settings.detection_edge_confidence_margin,
    )


def _containment(a: BoundingBox, b: BoundingBox) -> float:
    """Share of the smaller box lying inside the larger one."""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    overlap = (ix2 - ix1) * (iy2 - iy1)
    smaller = min(a.area, b.area)
    return 0.0 if smaller <= 0 else overlap / smaller


def _touches_edge(box: BoundingBox, width: int, height: int, margin: int) -> bool:
    """Whether the box reaches the frame border within ``margin`` pixels."""
    return (
        box.x1 <= margin
        or box.y1 <= margin
        or box.x2 >= width - 1 - margin
        or box.y2 >= height - 1 - margin
    )


def assess(
    detections: tuple[Detection, ...],
    *,
    frame_width: int,
    frame_height: int,
    policy: QualityPolicy,
) -> tuple[AssessedDetection, ...]:
    """Sort detections into quality tiers, in the order they were given.

    Deterministic throughout. Duplicate resolution ranks by confidence and
    breaks ties on the detector's own ordering, so the same input always yields
    the same verdicts -- a quality layer whose output depended on dictionary
    order would make every count downstream irreproducible.

    A frame with no dimensions cannot be assessed against area or edge rules;
    those are skipped rather than evaluated against a guess.
    """
    if not detections:
        return ()

    frame_area = float(frame_width * frame_height) if frame_width and frame_height else 0.0
    geometry: list[tuple[float, float | None, bool]] = []
    for detection in detections:
        box = detection.bbox
        area_fraction = box.area / frame_area if frame_area > 0 else 0.0
        aspect = box.width / box.height if box.height > 0 else None
        edge = (
            _touches_edge(box, frame_width, frame_height, policy.edge_margin_px)
            if frame_area > 0
            else False
        )
        geometry.append((area_fraction, aspect, edge))

    if not policy.enabled:
        return tuple(
            AssessedDetection(
                detection=detection,
                quality=DetectionQuality.CONFIRMED,
                flags=(),
                reason=None,
                area_fraction=round(area, 6),
                aspect_ratio=round(aspect, 4) if aspect is not None else None,
                touches_frame_edge=edge,
            )
            for detection, (area, aspect, edge) in zip(detections, geometry, strict=True)
        )

    # --- geometric impossibility -------------------------------------------
    rejected: dict[int, tuple[QualityFlag, str]] = {}
    for index, (area, aspect, _edge) in enumerate(geometry):
        if policy.max_area_fraction > 0 and frame_area > 0 and area >= policy.max_area_fraction:
            rejected[index] = (
                QualityFlag.IMPLAUSIBLE_AREA,
                f"Covers {area * 100:.1f}% of the frame, at or above the "
                f"{policy.max_area_fraction * 100:.1f}% ceiling for a single road "
                "vehicle in this view. A box this large is a structure or a crowd.",
            )
            continue
        if aspect is None:
            continue
        if policy.min_aspect_ratio > 0 and aspect < policy.min_aspect_ratio:
            rejected[index] = (
                QualityFlag.IMPLAUSIBLE_ASPECT,
                f"Width-over-height {aspect:.2f} is below the configured "
                f"{policy.min_aspect_ratio:.2f}: a column, not a vehicle.",
            )
        elif policy.max_aspect_ratio > 0 and aspect > policy.max_aspect_ratio:
            rejected[index] = (
                QualityFlag.IMPLAUSIBLE_ASPECT,
                f"Width-over-height {aspect:.2f} is above the configured "
                f"{policy.max_aspect_ratio:.2f}: a sliver, not a vehicle.",
            )

    # --- one object, one detection -----------------------------------------
    # Ranked by confidence, ties broken on detector order, so the survivor of a
    # duplicate pair is the same on every run.
    duplicate_of: dict[int, int] = {}
    order = sorted(
        (i for i in range(len(detections)) if i not in rejected),
        key=lambda i: (-detections[i].confidence, i),
    )
    kept: list[int] = []
    for index in order:
        box = detections[index].bbox
        for survivor in kept:
            other = detections[survivor].bbox
            iou = box.iou(other)
            contained = _containment(box, other)
            if (policy.duplicate_iou > 0 and iou >= policy.duplicate_iou) or (
                policy.duplicate_containment > 0
                and contained >= policy.duplicate_containment
            ):
                duplicate_of[index] = survivor
                break
        else:
            kept.append(index)

    # --- verdicts ----------------------------------------------------------
    assessed: list[AssessedDetection] = []
    for index, (detection, (area, aspect, edge)) in enumerate(
        zip(detections, geometry, strict=True)
    ):
        flags: list[QualityFlag] = []
        reasons: list[str] = []
        quality = DetectionQuality.CONFIRMED

        if index in rejected:
            flag, reason = rejected[index]
            flags.append(flag)
            reasons.append(reason)
            quality = DetectionQuality.REJECTED
        else:
            if index in duplicate_of:
                survivor = duplicate_of[index]
                flags.append(QualityFlag.DUPLICATE)
                reasons.append(
                    f"Overlaps detection {survivor} "
                    f"({detections[survivor].vehicle_class.value} at "
                    f"{detections[survivor].confidence:.2f}) closely enough to be the "
                    "same object; the more confident box is counted."
                )
                quality = DetectionQuality.PROVISIONAL

            floor = policy.floor_for(detection.vehicle_class)
            if detection.confidence < floor:
                flags.append(QualityFlag.BELOW_CLASS_CONFIDENCE)
                reasons.append(
                    f"Confidence {detection.confidence:.2f} is below the "
                    f"{floor:.2f} confirmation floor for "
                    f"{detection.vehicle_class.value}. Reported, not counted."
                )
                quality = DetectionQuality.PROVISIONAL
            elif (
                edge
                and policy.edge_confidence_margin > 0
                and detection.confidence < floor + policy.edge_confidence_margin
            ):
                flags.append(QualityFlag.EDGE_TRUNCATED_LOW_CONFIDENCE)
                reasons.append(
                    f"Clipped by the frame edge at {detection.confidence:.2f}, under "
                    f"the {floor + policy.edge_confidence_margin:.2f} required of a "
                    "truncated box. Part of an object is weaker evidence than all "
                    "of it."
                )
                quality = DetectionQuality.PROVISIONAL

        assessed.append(
            AssessedDetection(
                detection=detection,
                quality=quality,
                flags=tuple(flags),
                reason=" ".join(reasons) if reasons else None,
                area_fraction=round(area, 6),
                aspect_ratio=round(aspect, 4) if aspect is not None else None,
                touches_frame_edge=edge,
                duplicate_of_index=duplicate_of.get(index),
            )
        )

    return tuple(assessed)


def confirmed(assessed: tuple[AssessedDetection, ...]) -> tuple[Detection, ...]:
    """The detections a measurement may count."""
    return tuple(item.detection for item in assessed if item.is_confirmed)
