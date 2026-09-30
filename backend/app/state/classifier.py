"""Deterministic congestion classification.

A pure function of one :class:`~app.analytics.schemas.TrafficAnalytics` record
and one threshold set. No model, no learned parameters, no randomness, no
network access, no clock read: the same measurements and the same configuration
produce the same verdict on any machine, which is what makes a recommendation
replayable months later.

How a verdict is built
----------------------
Each available signal is normalised onto ``[0, 1]`` against a reference value,
where 0 is free-flowing and 1 is maximally congested. The signals are then
combined into a weighted **congestion index**, and the index is banded into a
:class:`~app.domain.enums.CongestionLevel`.

Weights are renormalised over the signals that are actually available. That is
the important detail: an unavailable signal contributes *nothing*, rather than
contributing a zero. Scoring a missing speed measurement as 0.0 would drag every
uncalibrated camera toward "free flow" -- the single most dangerous direction to
be wrong in, because it makes a jammed road look like the fast route.

When classification is refused
------------------------------
Three cases produce ``UNKNOWN`` with a stated reason rather than a band:

* **Poor measurement quality.** Interpreting numbers the measurement layer has
  already disowned would launder them into a confident-looking label.
* **Tracking failure in a sequence.** Objects detected but no confirmed tracks
  means the vehicle count is unreliable, even though detections exist.
* **No usable signal.** Nothing to compute an index from.

``UNKNOWN`` is not a band and never orders between ``FREE_FLOW`` and the rest.
An empty scene, by contrast, *is* classified: a zero vehicle count from a
working detector is a legitimate observation of free flow, which is why counts
survive without calibration when metric quantities do not.
"""

from __future__ import annotations

from app.analytics.schemas import (
    MetricAvailability,
    MetricValue,
    ObservationMode,
    QualityGrade,
    TrafficAnalytics,
)
from app.core.config import Settings
from app.domain.enums import CongestionLevel
from app.state.schemas import (
    CongestionAssessment,
    CongestionSignal,
    CongestionThresholds,
    EvidenceBasis,
    SignalName,
)

METHOD = "weighted-threshold-congestion"
METHOD_VERSION = "1.0"

#: Decimal places the index is rounded to before banding. Banding the *rounded*
#: value keeps the reported index and the reported band consistent: a reader who
#: recomputes the band from the number in the response gets the same answer.
_INDEX_PRECISION = 4

#: Confidence a verdict starts from, by evidence basis. Image-space evidence is
#: camera-specific and cannot be compared against published thresholds, so it is
#: worth materially less than a calibrated measurement even when complete.
_BASIS_CONFIDENCE = {
    EvidenceBasis.PHYSICAL: 0.9,
    EvidenceBasis.IMAGE_SPACE: 0.6,
}

_QUALITY_CONFIDENCE = {
    QualityGrade.GOOD: 1.0,
    QualityGrade.DEGRADED: 0.8,
    QualityGrade.POOR: 0.0,
}


def thresholds_from(settings: Settings) -> CongestionThresholds:
    """Build the threshold set in force from configuration."""
    return CongestionThresholds(
        light=settings.congestion_index_light,
        moderate=settings.congestion_index_moderate,
        heavy=settings.congestion_index_heavy,
        severe=settings.congestion_index_severe,
        gridlock=settings.congestion_index_gridlock,
        occupancy_reference=settings.congestion_occupancy_reference,
        stationary_reference=settings.congestion_stationary_reference,
        image_speed_reference=settings.congestion_image_speed_reference,
        density_reference_veh_per_km_per_lane=(
            settings.congestion_density_reference_veh_per_km_per_lane
        ),
        free_flow_speed_kph=settings.congestion_free_flow_speed_kph,
    )


def band_for(index: float, thresholds: CongestionThresholds) -> CongestionLevel:
    """Map a congestion index onto a band.

    Cut points are lower bounds, so a value sitting exactly on one falls in the
    higher band. Chosen over the alternative because the bands are read as
    severity: "at or above the heavy threshold" is the statement an operator
    expects, and a value landing precisely on a boundary should not be reported
    as the calmer of the two.
    """
    if index >= thresholds.gridlock:
        return CongestionLevel.GRIDLOCK
    if index >= thresholds.severe:
        return CongestionLevel.SEVERE
    if index >= thresholds.heavy:
        return CongestionLevel.HEAVY
    if index >= thresholds.moderate:
        return CongestionLevel.MODERATE
    if index >= thresholds.light:
        return CongestionLevel.LIGHT
    return CongestionLevel.FREE_FLOW


def _clamp(value: float) -> float:
    """Constrain a normalised score to ``[0, 1]``."""
    return max(0.0, min(1.0, value))


def _unavailable_signal(
    name: SignalName,
    metric: MetricValue[float] | MetricValue[int],
    *,
    image_space: bool,
) -> CongestionSignal:
    """Carry a measurement layer's own unavailability through to the verdict."""
    return CongestionSignal(
        name=name,
        availability=metric.availability,
        reason=metric.reason or "Measurement unavailable.",
        unit=metric.unit,
        is_image_space=image_space,
    )


def _missing_signal(
    name: SignalName,
    reason: str,
    *,
    image_space: bool,
    availability: MetricAvailability = MetricAvailability.UNAVAILABLE_INSUFFICIENT_DATA,
) -> CongestionSignal:
    """A signal this layer itself cannot form, with the reason why."""
    return CongestionSignal(
        name=name,
        availability=availability,
        reason=reason,
        is_image_space=image_space,
    )


def _available_signal(
    name: SignalName,
    *,
    raw: float,
    score: float,
    weight: float,
    unit: str | None,
    image_space: bool,
) -> CongestionSignal:
    """A contributing signal with its raw value and normalised score."""
    return CongestionSignal(
        name=name,
        availability=MetricAvailability.AVAILABLE,
        raw_value=round(raw, 6),
        score=round(_clamp(score), 6),
        weight=weight,
        unit=unit,
        is_image_space=image_space,
    )


def vehicle_count_for(analytics: TrafficAnalytics) -> int:
    """Vehicles the verdict is about.

    A sequence counts distinct tracked vehicles; a snapshot counts detections.
    Using the detection tally for a sequence would count one stopped car once
    per frame, which is precisely the inflation that would turn a quiet junction
    into a jam.
    """
    if analytics.mode is ObservationMode.SEQUENCE:
        return analytics.tracked_object_tally.vehicles
    return analytics.detection_tally.vehicles


def _stationary_signal(
    analytics: TrafficAnalytics, vehicles: int, weight: float, reference: float
) -> CongestionSignal:
    """Fraction of observed vehicles that are stationary."""
    metric = analytics.stationary_vehicle_count
    if not metric.is_available:
        return _unavailable_signal(SignalName.STATIONARY_FRACTION, metric, image_space=False)
    if vehicles <= 0:
        return _missing_signal(
            SignalName.STATIONARY_FRACTION,
            "No vehicles observed, so a stationary fraction has no denominator.",
            image_space=False,
            availability=MetricAvailability.UNAVAILABLE_NOT_APPLICABLE,
        )
    fraction = (metric.value or 0) / vehicles
    return _available_signal(
        SignalName.STATIONARY_FRACTION,
        raw=fraction,
        score=fraction / reference,
        weight=weight,
        unit="fraction",
        image_space=False,
    )


def _unknown(
    reason: str,
    thresholds: CongestionThresholds,
    signals: tuple[CongestionSignal, ...] = (),
) -> CongestionAssessment:
    """Refuse to classify, and say why."""
    return CongestionAssessment(
        level=CongestionLevel.UNKNOWN,
        index=None,
        basis=EvidenceBasis.INSUFFICIENT,
        signals=signals,
        thresholds=thresholds,
        confidence=None,
        reason=reason,
        method=METHOD,
        method_version=METHOD_VERSION,
    )


def classify(analytics: TrafficAnalytics, settings: Settings) -> CongestionAssessment:
    """Interpret one measurement set as a congestion level.

    Deterministic and side-effect free. Returns ``UNKNOWN`` with a reason rather
    than guessing whenever the evidence does not support a band.
    """
    thresholds = thresholds_from(settings)

    if analytics.quality.grade is QualityGrade.POOR:
        return _unknown(
            "Measurement quality is poor, so no congestion level is inferred. "
            "Interpreting measurements the analytics layer has already disowned "
            "would present low-confidence numbers as a confident verdict.",
            thresholds,
        )

    vehicles = vehicle_count_for(analytics)

    if (
        analytics.mode is ObservationMode.SEQUENCE
        and vehicles == 0
        and analytics.detection_tally.vehicles > 0
    ):
        return _unknown(
            "Vehicles were detected but no track was confirmed, so the vehicle "
            "count is unreliable. Treating this as an empty scene would report "
            "free flow on a road that may be queued.",
            thresholds,
        )

    if vehicles == 0:
        return CongestionAssessment(
            level=CongestionLevel.FREE_FLOW,
            index=0.0,
            basis=EvidenceBasis.IMAGE_SPACE,
            signals=(),
            thresholds=thresholds,
            confidence=round(
                _BASIS_CONFIDENCE[EvidenceBasis.IMAGE_SPACE]
                * _QUALITY_CONFIDENCE[analytics.quality.grade],
                3,
            ),
            reason=None,
            method=METHOD,
            method_version=METHOD_VERSION,
        )

    # Only a genuinely physical signal selects the physical basis. The
    # stationary fraction appears in both signal sets because it is derived from
    # counts and needs no calibration -- so letting it decide would label a
    # verdict "physical" on an uncalibrated camera, and would then discard the
    # image-space signals that were the only real evidence available.
    physical = _physical_signals(analytics, vehicles, settings, thresholds)
    if any(
        signal.availability is MetricAvailability.AVAILABLE
        and signal.name in (SignalName.DENSITY, SignalName.MEAN_SPEED)
        for signal in physical
    ):
        return _assess(
            physical,
            EvidenceBasis.PHYSICAL,
            analytics,
            thresholds,
            {
                SignalName.DENSITY: settings.congestion_weight_density,
                SignalName.MEAN_SPEED: settings.congestion_weight_speed,
                SignalName.STATIONARY_FRACTION: settings.congestion_weight_stationary,
            },
        )

    image = _image_signals(analytics, vehicles, settings, thresholds)
    if any(signal.availability is MetricAvailability.AVAILABLE for signal in image):
        return _assess(
            image,
            EvidenceBasis.IMAGE_SPACE,
            analytics,
            thresholds,
            {
                SignalName.IMAGE_OCCUPANCY: settings.congestion_weight_occupancy,
                SignalName.IMAGE_SPEED: settings.congestion_weight_speed,
                SignalName.STATIONARY_FRACTION: settings.congestion_weight_stationary,
            },
        )

    return _unknown(
        "No usable congestion signal is available: neither calibrated "
        "measurements nor image-space occupancy, speed or queue evidence could "
        "be computed from this analysis.",
        thresholds,
        image,
    )


def _physical_signals(
    analytics: TrafficAnalytics,
    vehicles: int,
    settings: Settings,
    thresholds: CongestionThresholds,
) -> tuple[CongestionSignal, ...]:
    """Signals expressed in real-world units. Require a calibrated camera.

    Image-space signals are deliberately excluded from a physical assessment.
    Occupancy and image speed measure the same phenomena as density and speed,
    at a different and camera-specific scale, so including both would count one
    observation twice and let a mounting angle move the verdict.
    """
    density = analytics.traffic_density_veh_per_km_per_lane
    speed = analytics.mean_speed_kph

    density_signal = (
        _available_signal(
            SignalName.DENSITY,
            raw=density.value or 0.0,
            score=(density.value or 0.0) / thresholds.density_reference_veh_per_km_per_lane,
            weight=settings.congestion_weight_density,
            unit=density.unit or "veh/km/lane",
            image_space=False,
        )
        if density.is_available
        else _unavailable_signal(SignalName.DENSITY, density, image_space=False)
    )

    speed_signal = (
        _available_signal(
            SignalName.MEAN_SPEED,
            raw=speed.value or 0.0,
            score=1.0 - (speed.value or 0.0) / thresholds.free_flow_speed_kph,
            weight=settings.congestion_weight_speed,
            unit=speed.unit or "km/h",
            image_space=False,
        )
        if speed.is_available
        else _unavailable_signal(SignalName.MEAN_SPEED, speed, image_space=False)
    )

    stationary = _stationary_signal(
        analytics,
        vehicles,
        settings.congestion_weight_stationary,
        thresholds.stationary_reference,
    )
    return (density_signal, speed_signal, stationary)


def _image_signals(
    analytics: TrafficAnalytics,
    vehicles: int,
    settings: Settings,
    thresholds: CongestionThresholds,
) -> tuple[CongestionSignal, ...]:
    """Frame-relative signals. Available without a calibration."""
    occupancy = analytics.spatial_distribution.image_occupancy_ratio
    image_speed = analytics.mean_image_speed_diagonals_per_s

    occupancy_signal = (
        _available_signal(
            SignalName.IMAGE_OCCUPANCY,
            raw=occupancy.value or 0.0,
            score=(occupancy.value or 0.0) / thresholds.occupancy_reference,
            weight=settings.congestion_weight_occupancy,
            unit=occupancy.unit or "fraction",
            image_space=True,
        )
        if occupancy.is_available
        else _unavailable_signal(SignalName.IMAGE_OCCUPANCY, occupancy, image_space=True)
    )

    speed_signal = (
        _available_signal(
            SignalName.IMAGE_SPEED,
            raw=image_speed.value or 0.0,
            score=1.0 - (image_speed.value or 0.0) / thresholds.image_speed_reference,
            weight=settings.congestion_weight_speed,
            unit=image_speed.unit or "diagonals/s",
            image_space=True,
        )
        if image_speed.is_available
        else _unavailable_signal(SignalName.IMAGE_SPEED, image_speed, image_space=True)
    )

    stationary = _stationary_signal(
        analytics,
        vehicles,
        settings.congestion_weight_stationary,
        thresholds.stationary_reference,
    )
    return (occupancy_signal, speed_signal, stationary)


def _assess(
    signals: tuple[CongestionSignal, ...],
    basis: EvidenceBasis,
    analytics: TrafficAnalytics,
    thresholds: CongestionThresholds,
    configured_weights: dict[SignalName, float],
) -> CongestionAssessment:
    """Combine available signals into an index, a band and a confidence.

    ``configured_weights`` carries what each signal *would* have weighed had it
    been available. An unavailable signal is stored with zero weight so it
    cannot contribute to the index, which means the stored weights alone cannot
    say how much evidence is missing -- and that is exactly what the confidence
    needs to know.
    """
    contributing = [
        signal
        for signal in signals
        if signal.availability is MetricAvailability.AVAILABLE and signal.weight > 0.0
    ]
    total_weight = sum(signal.weight for signal in contributing)
    if total_weight <= 0.0:
        return _unknown(
            "Every available congestion signal carries zero weight, so no index "
            "can be formed from the configured weighting.",
            thresholds,
            signals,
        )

    index = round(
        sum((signal.score or 0.0) * signal.weight for signal in contributing) / total_weight,
        _INDEX_PRECISION,
    )
    index = _clamp(index)

    # Coverage is measured against the weight this basis *could* have used, so a
    # verdict resting on one signal out of three reports lower confidence than
    # one resting on all three.
    possible_weight = sum(configured_weights.get(signal.name, 0.0) for signal in signals)
    coverage = min(1.0, total_weight / possible_weight) if possible_weight > 0 else 1.0

    confidence = round(
        _BASIS_CONFIDENCE[basis]
        * _QUALITY_CONFIDENCE[analytics.quality.grade]
        * coverage,
        3,
    )

    return CongestionAssessment(
        level=band_for(index, thresholds),
        index=index,
        basis=basis,
        signals=signals,
        thresholds=thresholds,
        confidence=confidence,
        reason=None,
        method=METHOD,
        method_version=METHOD_VERSION,
    )
