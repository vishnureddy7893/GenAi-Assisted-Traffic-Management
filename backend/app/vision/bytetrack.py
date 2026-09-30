"""ByteTrack-style multi-object tracker.

Implements the Phase-1 :class:`~app.vision.tracker.VehicleTracker` contract.
Answers exactly one question -- *"which detections across frames are the same
object?"* -- and nothing about what that means for traffic.

The BYTE association strategy
-----------------------------
The insight of ByteTrack (Zhang et al., 2022) is not its motion model: it is
that **low-confidence detections should not be thrown away**. A vehicle that
becomes partially occluded does not vanish; its detection confidence drops.
Discarding everything below a single threshold therefore breaks the track
precisely when occlusion makes continuity most valuable, and the object
reappears with a new id -- an identity switch that ruins any per-vehicle
measurement built on top.

So association runs in two passes:

1. **High-confidence detections** are matched against all active tracks.
2. **Low-confidence detections** are matched against only the tracks that pass
   one left unmatched. These recover occluded objects, but are never allowed to
   start a new track: a low-confidence box with no history is far more likely to
   be a false positive than a new vehicle.

Deliberate deviations, and why
------------------------------
* **Constant-velocity prediction instead of a Kalman filter.** At traffic-camera
  frame rates over the short gaps this tracker must bridge, the two are close,
  and a linear predictor is exactly reproducible. Determinism matters more here
  than the last few points of MOTA: a tracker whose output shifts between runs
  makes every downstream regression test unreliable. The interface for swapping
  in a Kalman filter is :meth:`_Track.predict`.
* **Greedy IoU matching instead of the Hungarian algorithm.** Optimal assignment
  matters when many similar objects compete; at junction-scale object counts the
  greedy choice by descending IoU agrees with the optimal one almost always, and
  it avoids a scipy dependency for a marginal gain.

Both deviations are stated in :attr:`ByteTracker.info` so a result is never
presented as stock ByteTrack.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from app.domain.enums import VehicleClass
from app.vision.detector import BoundingBox, Detection
from app.vision.tracker import Track, TrackerInfo, TrackerStats, TrackPoint, VehicleTracker

TRACKER_NAME = "bytetrack-lite"
TRACKER_VERSION = "1.0"

#: Displacement, in multiples of the box diagonal, beyond which a match is
#: treated as an identity switch rather than motion. A vehicle cannot cross
#: several of its own lengths between consecutive frames.
_IMPLAUSIBLE_JUMP_DIAGONALS = 2.0


def iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection over union of two boxes."""
    return a.iou(b)


@dataclass
class _Track:
    """Mutable internal track state."""

    track_id: int
    vehicle_class: VehicleClass
    bbox: BoundingBox
    timestamp: dt.datetime
    confidence: float

    history: list[TrackPoint] = field(default_factory=list)
    age_frames: int = 0
    hits: int = 1
    time_since_update_frames: int = 0
    last_update_time: dt.datetime | None = None
    is_confirmed: bool = False
    id_switch_suspected: bool = False
    velocity: tuple[float, float] = (0.0, 0.0)
    #: Per-class hit counts. A track's class is the majority vote over its life,
    #: not whatever the latest frame said: single-frame class flicker (car vs
    #: truck on a van) is common and should not rewrite an object's identity.
    class_votes: dict[VehicleClass, int] = field(default_factory=dict)

    def predict(self) -> BoundingBox:
        """Where this track is expected to be in the next frame.

        Constant velocity. Replace here to introduce a Kalman filter without
        touching the association logic.
        """
        dx, dy = self.velocity
        return BoundingBox(
            x1=self.bbox.x1 + dx,
            y1=self.bbox.y1 + dy,
            x2=self.bbox.x2 + dx,
            y2=self.bbox.y2 + dy,
        )

    def update(self, detection: Detection, timestamp: dt.datetime) -> None:
        """Absorb a matched detection."""
        previous_centre = self.bbox.center
        new_centre = detection.bbox.center

        dx = new_centre[0] - previous_centre[0]
        dy = new_centre[1] - previous_centre[1]
        diagonal = max((self.bbox.width**2 + self.bbox.height**2) ** 0.5, 1e-6)
        if (dx * dx + dy * dy) ** 0.5 > _IMPLAUSIBLE_JUMP_DIAGONALS * diagonal:
            # Flagged rather than smoothed away: an implausible jump means the
            # measurement is suspect, and hiding it would surface later as an
            # impossible speed with no explanation.
            self.id_switch_suspected = True

        self.velocity = (dx, dy)
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.timestamp = timestamp
        self.last_update_time = timestamp
        self.hits += 1
        self.time_since_update_frames = 0
        self.class_votes[detection.vehicle_class] = (
            self.class_votes.get(detection.vehicle_class, 0) + 1
        )
        self.vehicle_class = max(self.class_votes.items(), key=lambda kv: kv[1])[0]
        self.history.append(
            TrackPoint(
                timestamp=timestamp,
                bbox=detection.bbox,
                confidence=detection.confidence,
            )
        )

    def to_public(self, now: dt.datetime) -> Track:
        """Snapshot as the immutable public track record."""
        elapsed = 0.0
        if self.last_update_time is not None:
            elapsed = max((now - self.last_update_time).total_seconds(), 0.0)
        return Track(
            track_id=self.track_id,
            vehicle_class=self.vehicle_class,
            history=tuple(self.history),
            age_frames=self.age_frames,
            hits=self.hits,
            time_since_update_s=elapsed,
            is_confirmed=self.is_confirmed,
            velocity_px_per_s=None,
            velocity_mps=None,
            id_switch_suspected=self.id_switch_suspected,
            metadata={"last_confidence": self.confidence},
        )


class ByteTracker(VehicleTracker):
    """Two-stage IoU tracker following the BYTE association strategy."""

    def __init__(
        self,
        *,
        high_threshold: float = 0.5,
        low_threshold: float = 0.1,
        match_iou: float = 0.3,
        max_age_frames: int = 30,
        min_hits_to_confirm: int = 3,
    ) -> None:
        """
        Parameters
        ----------
        high_threshold:
            Confidence at or above which a detection may start a new track.
        low_threshold:
            Floor below which a detection is ignored entirely. Between the two
            thresholds, detections may only *continue* an existing track.
        match_iou:
            Minimum overlap for an association. Low by traffic-camera standards
            because vehicles are small and move several box-widths per second.
        max_age_frames:
            Frames a track survives unmatched before being dropped, which is
            what carries an object through a brief occlusion.
        """
        if low_threshold > high_threshold:
            msg = "low_threshold cannot exceed high_threshold"
            raise ValueError(msg)

        self._high = high_threshold
        self._low = low_threshold
        self._match_iou = match_iou
        self._max_age = max_age_frames
        self._min_hits = min_hits_to_confirm

        self._tracks: list[_Track] = []
        self._next_id = 1
        self._stats = TrackerStats()

    @property
    def info(self) -> TrackerInfo:
        """Identity of this tracker, for provenance."""
        return TrackerInfo(
            name=TRACKER_NAME,
            version=TRACKER_VERSION,
            max_age_frames=self._max_age,
            min_hits_to_confirm=self._min_hits,
            association_metric="iou (two-stage BYTE, greedy, constant-velocity)",
        )

    @property
    def stats(self) -> TrackerStats:
        """Statistics from the most recent update."""
        return self._stats

    def reset(self) -> None:
        """Discard all state.

        Required after a seek or a reconnect. Without it the tracker associates
        across the discontinuity and manufactures enormous displacements.
        """
        self._tracks.clear()
        self._next_id = 1
        self._stats = TrackerStats()

    # --- association ------------------------------------------------------
    def _match(
        self,
        tracks: list[_Track],
        detections: list[Detection],
    ) -> tuple[list[tuple[_Track, Detection]], list[_Track], list[Detection]]:
        """Greedily associate tracks to detections by descending IoU."""
        if not tracks or not detections:
            return [], list(tracks), list(detections)

        candidates: list[tuple[float, int, int]] = []
        for t_index, track in enumerate(tracks):
            predicted = track.predict()
            for d_index, detection in enumerate(detections):
                overlap = iou(predicted, detection.bbox)
                if overlap >= self._match_iou:
                    candidates.append((overlap, t_index, d_index))

        # Sort by overlap, then by index, so ties resolve identically every run.
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        matches: list[tuple[_Track, Detection]] = []

        for _overlap, t_index, d_index in candidates:
            if t_index in used_tracks or d_index in used_detections:
                continue
            used_tracks.add(t_index)
            used_detections.add(d_index)
            matches.append((tracks[t_index], detections[d_index]))

        unmatched_tracks = [t for i, t in enumerate(tracks) if i not in used_tracks]
        unmatched_detections = [d for i, d in enumerate(detections) if i not in used_detections]
        return matches, unmatched_tracks, unmatched_detections

    def update(
        self,
        detections: tuple[Detection, ...],
        *,
        timestamp: dt.datetime,
    ) -> tuple[Track, ...]:
        """Advance one frame and return the current tracks."""
        for track in self._tracks:
            track.age_frames += 1
            track.time_since_update_frames += 1

        high = [d for d in detections if d.confidence >= self._high]
        low = [d for d in detections if self._low <= d.confidence < self._high]

        # Pass 1: high-confidence detections against every active track.
        matches, unmatched_tracks, unmatched_high = self._match(list(self._tracks), high)

        # Pass 2: low-confidence detections against tracks pass 1 left behind.
        # These recover occluded objects; they may never start a new track.
        low_matches, _still_unmatched, _unmatched_low = self._match(unmatched_tracks, low)
        matches.extend(low_matches)

        for track, detection in matches:
            track.update(detection, timestamp)
            if not track.is_confirmed and track.hits >= self._min_hits:
                track.is_confirmed = True

        new_tracks = 0
        for detection in unmatched_high:
            self._tracks.append(self._spawn(detection, timestamp))
            new_tracks += 1

        before = len(self._tracks)
        self._tracks = [t for t in self._tracks if t.time_since_update_frames <= self._max_age]
        lost = before - len(self._tracks)

        self._stats = TrackerStats(
            active_tracks=len(self._tracks),
            confirmed_tracks=sum(1 for t in self._tracks if t.is_confirmed),
            new_tracks=new_tracks,
            lost_tracks=lost,
            suspected_id_switches=sum(1 for t in self._tracks if t.id_switch_suspected),
            unmatched_detections=len(unmatched_high),
        )

        return tuple(
            track.to_public(timestamp)
            for track in self._tracks
            if track.time_since_update_frames == 0
        )

    def _spawn(self, detection: Detection, timestamp: dt.datetime) -> _Track:
        """Create a track from an unmatched high-confidence detection."""
        track = _Track(
            track_id=self._next_id,
            vehicle_class=detection.vehicle_class,
            bbox=detection.bbox,
            timestamp=timestamp,
            confidence=detection.confidence,
            last_update_time=timestamp,
            class_votes={detection.vehicle_class: 1},
        )
        track.history.append(
            TrackPoint(
                timestamp=timestamp,
                bbox=detection.bbox,
                confidence=detection.confidence,
            )
        )
        # A track is confirmed immediately when one hit is enough, so a
        # single-frame source (an image) still yields a confirmed track.
        if track.hits >= self._min_hits:
            track.is_confirmed = True
        self._next_id += 1
        return track

    @property
    def all_tracks(self) -> tuple[Track, ...]:
        """Every live track, including ones not updated this frame."""
        return tuple(track.to_public(track.timestamp) for track in self._tracks)
