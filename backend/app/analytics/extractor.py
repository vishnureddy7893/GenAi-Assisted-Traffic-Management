"""TrafficStateExtractor implementation.

Bridges Phase 4 analytics onto the Phase 1 domain model, producing a
:class:`~app.domain.observation.JunctionObservation` from tracked objects.

Why this is a separate step from the analytics engine
-----------------------------------------------------
:class:`~app.analytics.engine.TrafficAnalyticsEngine` measures **one camera's
field of view**. A ``JunctionObservation`` describes **a surveyed junction with
named approaches**. Those are not the same thing, and the gap between them is a
zone configuration mapping image regions onto real road elements.

When that mapping is absent, this extractor emits a single approach covering the
whole field of view, with an identifier that says exactly that. It does not
invent approaches, and it does not guess which vehicles belong to which arm of a
junction — an unmapped camera genuinely does not know.

Metric fields are populated only when calibration exists. The
``JunctionObservation`` validator enforces this independently, so a bug here
fails loudly at construction rather than silently publishing pixel counts as
metres.
"""

from __future__ import annotations

import datetime as dt

from app.analytics.calibration import GroundProjector
from app.analytics.engine import (
    METHOD,
    METHOD_VERSION,
    AnalyticsContext,
    ObservedFrame,
    ObservedObject,
    TrafficAnalyticsEngine,
)
from app.analytics.schemas import TrafficAnalytics
from app.core.config import Settings
from app.domain.enums import VEHICLE_CLASSES, DataQualityFlag, VehicleClass
from app.domain.observation import (
    ApproachObservation,
    JunctionObservation,
    ObservationQuality,
    PedestrianObservation,
)
from app.domain.provenance import Measurement, Provenance, Unit, ValueSource
from app.vision.extractor import ExtractionContext, TrafficStateExtractor
from app.vision.models import ModelIdentity
from app.vision.tracker import Track

#: Approach identifier used when no zone configuration maps the view onto a
#: junction. Named so it is unmistakable in any downstream consumer.
WHOLE_VIEW_APPROACH_ID = "whole-field-of-view"


class AnalyticsTrafficStateExtractor(TrafficStateExtractor):
    """Produces a ``JunctionObservation`` from tracks and their analytics."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._engine = TrafficAnalyticsEngine(self._settings)

    @property
    def name(self) -> str:
        """Identifier of this extractor, recorded in provenance."""
        return METHOD

    @property
    def version(self) -> str:
        """Version of the extraction logic, recorded in provenance."""
        return METHOD_VERSION

    def reset(self) -> None:
        """Clear accumulated windowed state.

        Stateless by construction: every extraction is computed from the tracks
        it is given. A stateful extractor would carry counts across a seek and
        report traffic that is no longer in view.
        """
        return None

    # -----------------------------------------------------------------
    def extract(
        self,
        tracks: tuple[Track, ...],
        *,
        context: ExtractionContext,
        captured_at: dt.datetime,
        ingested_at: dt.datetime,
    ) -> JunctionObservation:
        """Produce one observation from the current tracks."""
        calibration = context.zone_config.calibration
        projector = GroundProjector.from_calibration(calibration)

        frames = self._frames_from_tracks(tracks, context)
        analytics: TrafficAnalytics | None = None
        if frames:
            analytics = self._engine.analyse(
                frames,
                AnalyticsContext(
                    media_id=context.source_id,
                    source_id=context.source_id,
                    observation_area_id=context.junction_id,
                    model=ModelIdentity(
                        name=context.detector_info.name if context.detector_info else "unknown",
                        backend="vision",
                        sha256=(
                            context.detector_info.weights_id
                            if context.detector_info and context.detector_info.weights_id
                            else None
                        ),
                    ),
                    tracker=context.tracker_info.name if context.tracker_info else None,
                    calibration=calibration,
                    settings=self._settings,
                ),
                computed_at=ingested_at,
            )

        flags: set[DataQualityFlag] = set()
        if projector is None:
            flags.add(DataQualityFlag.UNCALIBRATED)
        if context.tracker_stats and context.tracker_stats.suspected_id_switches:
            flags.add(DataQualityFlag.LOW_DETECTION_CONFIDENCE)

        approach = self._approach(tracks, analytics, projector, context, captured_at)

        return JunctionObservation(
            observation_id=f"obs-{context.source_id}-{int(captured_at.timestamp() * 1000)}",
            junction_id=context.junction_id,
            source_id=context.source_id,
            captured_at=captured_at,
            ingested_at=ingested_at,
            schema_version=context.schema_version,
            approaches=(approach,),
            calibration=calibration,
            quality=ObservationQuality(
                flags=frozenset(flags),
                mean_detection_confidence=(
                    analytics.quality.mean_detection_confidence if analytics else None
                ),
                frames_analysed=analytics.quality.frames_analysed if analytics else None,
                analysis_window_s=context.analysis_window_s,
            ),
        )

    # -----------------------------------------------------------------
    def _frames_from_tracks(
        self, tracks: tuple[Track, ...], context: ExtractionContext
    ) -> list[ObservedFrame]:
        """Reconstruct per-frame observations from track histories."""
        width = context.zone_config.frame_width
        height = context.zone_config.frame_height

        by_time: dict[dt.datetime, list[ObservedObject]] = {}
        for track in tracks:
            for point in track.history:
                by_time.setdefault(point.timestamp, []).append(
                    ObservedObject(
                        road_user_class=track.vehicle_class,
                        confidence=point.confidence,
                        x1=point.bbox.x1,
                        y1=point.bbox.y1,
                        x2=point.bbox.x2,
                        y2=point.bbox.y2,
                        track_id=track.track_id,
                    )
                )

        return [
            ObservedFrame(
                frame_index=index,
                timestamp=timestamp,
                width=width,
                height=height,
                objects=tuple(objects),
            )
            for index, (timestamp, objects) in enumerate(sorted(by_time.items()))
        ]

    def _approach(
        self,
        tracks: tuple[Track, ...],
        analytics: TrafficAnalytics | None,
        projector: GroundProjector | None,
        context: ExtractionContext,
        captured_at: dt.datetime,
    ) -> ApproachObservation:
        """Build the single whole-view approach observation.

        One approach, not several: without a zone configuration the camera has
        no basis for splitting its view into arms of a junction.
        """
        provenance = Provenance(
            source=ValueSource.DERIVED,
            producer=f"analytics.{self.name}",
            producer_version=self.version,
            observed_at=captured_at,
            computed_at=captured_at,
            input_refs=(context.source_id,),
        )

        vehicles = [t for t in tracks if t.vehicle_class in VEHICLE_CLASSES]
        pedestrians = [t for t in tracks if t.vehicle_class is VehicleClass.PERSON]

        by_class: dict[VehicleClass, Measurement[int]] = {}
        for vehicle_class in {t.vehicle_class for t in tracks}:
            count = sum(1 for t in tracks if t.vehicle_class is vehicle_class)
            by_class[vehicle_class] = Measurement[int](
                value=count, unit=Unit.VEHICLE, provenance=provenance
            )

        stopped = None
        if analytics is not None and analytics.stationary_vehicle_count.is_available:
            stopped = Measurement[int](
                value=analytics.stationary_vehicle_count.value or 0,
                unit=Unit.VEHICLE,
                provenance=provenance,
            )

        # Metric fields stay None without calibration. The JunctionObservation
        # validator rejects them independently, so this cannot be forgotten.
        mean_speed = None
        has_physical = (
            projector is not None
            and analytics is not None
            and analytics.mean_speed_kph.is_available
        )
        if has_physical and analytics is not None:
            mean_speed = Measurement[float](
                value=analytics.mean_speed_kph.value or 0.0,
                unit=Unit.KILOMETER_PER_HOUR,
                provenance=provenance,
            )

        return ApproachObservation(
            approach_id=WHOLE_VIEW_APPROACH_ID,
            incoming_segment_id=WHOLE_VIEW_APPROACH_ID,
            vehicle_count_total=Measurement[int](
                value=len(vehicles), unit=Unit.VEHICLE, provenance=provenance
            ),
            vehicle_count_by_class=by_class,
            mean_speed_kph=mean_speed,
            stopped_vehicle_count=stopped,
            pedestrians=(
                PedestrianObservation(
                    waiting_count=Measurement[int](
                        value=len(pedestrians), unit=Unit.PERSON, provenance=provenance
                    )
                )
                if pedestrians
                else None
            ),
        )
