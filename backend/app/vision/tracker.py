"""Vehicle tracking interface. Implemented in Phase 3 by the ByteTrack-style tracker.

Why tracking is not optional
----------------------------
Detection alone can count objects *in a frame*. Almost every quantity this
system actually needs requires identity across frames:

* a **count** of vehicles that passed, rather than vehicles currently visible --
  without identity, one car sitting at a red light is recounted every frame;
* **speed**, which is displacement of the same vehicle over time;
* **queue length** and **waiting time**, which require knowing a vehicle has
  been stationary, not merely that something is stationary there;
* **turning-movement counts**, which are the fundamental unit of junction
  analysis and are defined entirely by where one vehicle entered and exited.

Time, not frame index
---------------------
:meth:`VehicleTracker.update` takes a timestamp rather than a frame number. Frame
indices assume a constant frame rate, and neither mode delivers that: live
sources drop frames under backpressure, and recorded video is routinely
variable-frame-rate. A tracker that assumes uniform spacing computes velocity
by dividing by the wrong dt, and the resulting speed error propagates silently
into congestion classification and ETA.

Association quality is a first-class output
-------------------------------------------
An identity switch -- a track hopping between two vehicles under occlusion --
produces an enormous apparent displacement, and therefore an absurd speed. That
is a data-quality event, so trackers report their own instability rather than
letting it surface downstream as an unexplained 300 km/h reading.
"""

from __future__ import annotations

import abc
import datetime as dt
from dataclasses import dataclass, field
from typing import Annotated

from pydantic import Field

from app.domain.base import DomainModel
from app.domain.enums import VehicleClass
from app.vision.detector import BoundingBox, Detection


@dataclass(frozen=True, slots=True)
class TrackPoint:
    """One observation of a track at one instant."""

    timestamp: dt.datetime
    bbox: BoundingBox
    confidence: float
    ground_position_m: tuple[float, float] | None = None
    """Ground-plane position, when calibration allows the projection. ``None``
    on an uncalibrated camera -- and the honest consequence is that no metric
    speed can be produced for this track."""


@dataclass(frozen=True, slots=True)
class Track:
    """One vehicle followed across frames."""

    track_id: int
    vehicle_class: VehicleClass
    history: tuple[TrackPoint, ...]
    age_frames: int
    hits: int
    """Frames in which this track was matched to a detection. A track with many
    frames of age but few hits is mostly extrapolation and should not be trusted
    for speed."""

    time_since_update_s: float
    is_confirmed: bool
    """Whether the track has survived the confirmation threshold. Unconfirmed
    tracks are excluded from counts: including them turns every detector false
    positive into a counted vehicle."""

    velocity_px_per_s: tuple[float, float] | None = None
    velocity_mps: tuple[float, float] | None = None
    """Ground-plane velocity. ``None`` without calibration. Never derived from
    the pixel velocity by a scalar: perspective makes pixel-to-metre scale
    position-dependent, so a single factor is wrong nearly everywhere in frame."""

    id_switch_suspected: bool = False
    """Set when an implausible jump suggests the identity changed. Propagates to
    :attr:`~app.domain.enums.DataQualityFlag.LOW_DETECTION_CONFIDENCE` on the
    observation rather than being silently smoothed away."""

    metadata: dict[str, float] = field(default_factory=dict)

    @property
    def current(self) -> TrackPoint:
        """Most recent observation of this track."""
        return self.history[-1]

    @property
    def first_seen_at(self) -> dt.datetime:
        """Timestamp of the first observation."""
        return self.history[0].timestamp

    @property
    def duration_s(self) -> float:
        """Observed lifetime of the track in seconds."""
        return (self.history[-1].timestamp - self.history[0].timestamp).total_seconds()

    @property
    def is_stationary_candidate(self) -> bool:
        """Whether this track currently looks stopped.

        A *candidate* only: the queue estimator decides what counts as stopped,
        using a threshold calibrated for the site. Parked vehicles and vehicles
        queued at a red light are both stationary, and telling them apart needs
        zone context this type does not have.
        """
        if self.velocity_mps is not None:
            vx, vy = self.velocity_mps
            return (vx * vx + vy * vy) ** 0.5 < 0.5
        return False


class TrackerInfo(DomainModel):
    """Identity and configuration of a tracker implementation."""

    name: Annotated[str, Field(min_length=1, max_length=100)]
    version: Annotated[str, Field(min_length=1, max_length=50)]
    max_age_frames: Annotated[int, Field(ge=0)] | None = None
    min_hits_to_confirm: Annotated[int, Field(ge=0)] | None = None
    association_metric: Annotated[str, Field(max_length=50)] | None = None
    """e.g. ``iou``, ``iou+appearance``. Appearance features cost throughput and
    buy robustness through occlusion; which trade is right depends on the
    footage and is measured in Phase 3, not assumed here."""


class TrackerStats(DomainModel):
    """Per-update tracker health, surfaced as observation quality."""

    active_tracks: Annotated[int, Field(ge=0)] = 0
    confirmed_tracks: Annotated[int, Field(ge=0)] = 0
    new_tracks: Annotated[int, Field(ge=0)] = 0
    lost_tracks: Annotated[int, Field(ge=0)] = 0
    suspected_id_switches: Annotated[int, Field(ge=0)] = 0
    unmatched_detections: Annotated[int, Field(ge=0)] = 0
    """Detections that matched no track. A sustained high value means the
    association is failing, which inflates counts and destroys speed estimates."""


class VehicleTracker(abc.ABC):
    """Associates detections across frames into persistent tracks."""

    @property
    @abc.abstractmethod
    def info(self) -> TrackerInfo:
        """Identity of this tracker, for provenance."""

    @abc.abstractmethod
    def update(
        self,
        detections: tuple[Detection, ...],
        *,
        timestamp: dt.datetime,
    ) -> tuple[Track, ...]:
        """Advance the tracker by one frame and return the current tracks.

        ``timestamp`` must come from the source clock. Passing a frame index, or
        assuming uniform spacing, makes every velocity wrong under dropped
        frames or variable frame rate.
        """

    @property
    @abc.abstractmethod
    def stats(self) -> TrackerStats:
        """Statistics from the most recent update."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Discard all state.

        Required whenever the input timeline discontinues -- after a seek on a
        recorded source, or a reconnect on a live one. Without it the tracker
        associates across the discontinuity and manufactures enormous
        displacements, and therefore impossible speeds.
        """
