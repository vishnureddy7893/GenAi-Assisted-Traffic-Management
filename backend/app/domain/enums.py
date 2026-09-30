"""Controlled vocabularies for the traffic domain.

Every enum here is a closed set on the wire. Using free strings for things like
signal colour or incident type is how a system ends up with ``"Green"``,
``"green"`` and ``"GRN"`` in the same database, at which point safety logic that
tests for a green conflict silently stops matching.

Where a term has an established meaning in traffic engineering (level of
service, saturation flow, movement type), that meaning is used rather than an
invented one, so the outputs are legible to a traffic engineer.
"""

from __future__ import annotations

from enum import StrEnum


# ---------------------------------------------------------------------------
# Vehicles
# ---------------------------------------------------------------------------
class VehicleClass(StrEnum):
    """Vehicle taxonomy.

    Deliberately coarse: these are the classes a detector can distinguish
    reliably from a fixed traffic camera. Finer classes (sedan vs hatchback) are
    not separable at typical resolution and would create labels we cannot honour.

    Emergency classes are listed because a detector *may* identify them, but
    detection of an emergency vehicle is never the sole trigger for emergency
    handling -- see :mod:`app.domain.emergency`, where an emergency is a declared
    event with a destination, not merely a recognised silhouette.
    """

    CAR = "car"
    MOTORCYCLE = "motorcycle"
    BUS = "bus"
    TRUCK = "truck"
    LIGHT_COMMERCIAL = "light_commercial"
    BICYCLE = "bicycle"
    AUTO_RICKSHAW = "auto_rickshaw"
    AMBULANCE = "ambulance"
    FIRE_ENGINE = "fire_engine"
    POLICE = "police"

    PERSON = "person"
    """A pedestrian. Not a vehicle, but a *road user* a detector reports, and
    the enum is the detector's output vocabulary.

    Kept distinct from pedestrian analytics on purpose: a detected person is one
    box in one frame, whereas
    :class:`~app.domain.observation.PedestrianObservation` carries counts and
    densities at a crossing. This value must never be summed into a vehicle
    count -- consumers filter on :data:`VEHICLE_CLASSES` for that.
    """

    OTHER = "other"
    UNKNOWN = "unknown"


#: Classes that are actually vehicles. Anything counting traffic volume must
#: filter on this rather than on "every detection", or pedestrians and cyclists
#: inflate the vehicle count.
VEHICLE_CLASSES: frozenset[VehicleClass] = frozenset(
    {
        VehicleClass.CAR,
        VehicleClass.MOTORCYCLE,
        VehicleClass.BUS,
        VehicleClass.TRUCK,
        VehicleClass.LIGHT_COMMERCIAL,
        VehicleClass.AUTO_RICKSHAW,
        VehicleClass.AMBULANCE,
        VehicleClass.FIRE_ENGINE,
        VehicleClass.POLICE,
    }
)

#: Road users that are not vehicles. Bicycles sit here deliberately: they are
#: vulnerable road users whose safety treatment differs from motor traffic, and
#: conflating them with vehicles is how a cycle lane gets modelled as a lane.
VULNERABLE_ROAD_USER_CLASSES: frozenset[VehicleClass] = frozenset(
    {VehicleClass.PERSON, VehicleClass.BICYCLE}
)

#: Passenger Car Equivalent factors are intentionally NOT defined here.
#: PCE is jurisdiction- and facility-specific (IRC 106 differs from HCM), so
#: hard-coding a table would fabricate a calibration we have not performed.
#: If a PCE-weighted flow is ever required, it becomes a versioned config
#: artifact with a citation rather than a literal in this module.


# ---------------------------------------------------------------------------
# Road network topology
# ---------------------------------------------------------------------------
class RoadClass(StrEnum):
    """Functional classification of a road segment."""

    MOTORWAY = "motorway"
    ARTERIAL = "arterial"
    COLLECTOR = "collector"
    LOCAL = "local"
    SERVICE = "service"
    UNKNOWN = "unknown"


class JunctionKind(StrEnum):
    """Control regime at a junction."""

    SIGNALIZED = "signalized"
    UNSIGNALIZED = "unsignalized"
    ROUNDABOUT = "roundabout"
    PRIORITY = "priority"
    ACCESS_POINT = "access_point"
    """Non-junction node used to attach a facility (hospital entrance, depot)."""


class MovementType(StrEnum):
    """Turn movement at a junction, from an incoming to an outgoing segment."""

    THROUGH = "through"
    LEFT = "left"
    RIGHT = "right"
    U_TURN = "u_turn"


class FacilityType(StrEnum):
    """Destination facility for an emergency response."""

    HOSPITAL = "hospital"
    TRAUMA_CENTER = "trauma_center"
    FIRE_STATION = "fire_station"
    POLICE_STATION = "police_station"


class DrivingSide(StrEnum):
    """Which side of the road traffic drives on.

    Affects which turn is the conflicting one. Getting this wrong inverts every
    conflict-movement safety check, so it is explicit network metadata rather
    than an assumption baked into the code.
    """

    LEFT = "left"
    RIGHT = "right"


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------
class SignalIndication(StrEnum):
    """Observed aspect shown to a movement or signal group.

    ``UNKNOWN`` is a first-class value: a camera that cannot see the signal head
    must report ``UNKNOWN``, never a guess. Downstream logic treats ``UNKNOWN``
    as "no priority may be assumed", which is the fail-safe direction.
    """

    RED = "red"
    AMBER = "amber"
    GREEN = "green"
    FLASHING_AMBER = "flashing_amber"
    FLASHING_RED = "flashing_red"
    OFF = "off"
    UNKNOWN = "unknown"


class SignalAdvisoryAction(StrEnum):
    """An action this system may *recommend* to a human operator.

    Recommendation only. There is no actuation path from these values to any
    real signal controller; see :class:`app.domain.state.SignalAdvisory`.
    """

    HOLD_CURRENT = "hold_current"
    EXTEND_GREEN = "extend_green"
    EARLY_CUTOFF = "early_cutoff"
    GRANT_EMERGENCY_PRIORITY = "grant_emergency_priority"
    PREPARE_EMERGENCY_PRIORITY = "prepare_emergency_priority"
    CLEAR_CONFLICTING_MOVEMENTS = "clear_conflicting_movements"
    RESTORE_NORMAL_OPERATION = "restore_normal_operation"
    NO_CHANGE = "no_change"


# ---------------------------------------------------------------------------
# Traffic condition
# ---------------------------------------------------------------------------
class LevelOfService(StrEnum):
    """Highway Capacity Manual level of service, A (free flow) to F (breakdown).

    Used because it is the vocabulary a traffic engineer already reads. It is a
    *derived* quantity -- at a signalized junction LOS is a function of control
    delay per vehicle -- so it may only be produced by an analytics component
    that actually computed that delay, with matching provenance. It must never
    be assigned by eye from a vehicle count.
    """

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"


class CongestionLevel(StrEnum):
    """Coarse congestion band for presentation and for route penalty tiers."""

    FREE_FLOW = "free_flow"
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"
    SEVERE = "severe"
    GRIDLOCK = "gridlock"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class WeatherCondition(StrEnum):
    """Prevailing weather affecting capacity, stopping distance and visibility."""

    CLEAR = "clear"
    CLOUDY = "cloudy"
    LIGHT_RAIN = "light_rain"
    HEAVY_RAIN = "heavy_rain"
    FOG = "fog"
    SNOW = "snow"
    ICE = "ice"
    DUST = "dust"
    UNKNOWN = "unknown"


class RoadSurfaceCondition(StrEnum):
    """Physical state of the carriageway."""

    DRY = "dry"
    WET = "wet"
    FLOODED = "flooded"
    ICY = "icy"
    POTHOLED = "potholed"
    UNDER_CONSTRUCTION = "under_construction"
    UNPAVED = "unpaved"
    UNKNOWN = "unknown"


class VisibilityLevel(StrEnum):
    """Visibility band; degrades both detection confidence and safe speed."""

    GOOD = "good"
    MODERATE = "moderate"
    POOR = "poor"
    VERY_POOR = "very_poor"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Incidents
# ---------------------------------------------------------------------------
class IncidentType(StrEnum):
    """Category of disruptive event on the network."""

    ACCIDENT = "accident"
    FIRE = "fire"
    ROAD_BLOCKAGE = "road_blockage"
    STALLED_VEHICLE = "stalled_vehicle"
    FLOODING = "flooding"
    CONSTRUCTION = "construction"
    DEBRIS = "debris"
    PROTEST_OR_EVENT = "protest_or_event"
    SIGNAL_FAILURE = "signal_failure"
    STRUCTURAL_FAILURE = "structural_failure"
    OTHER = "other"


class IncidentSeverity(StrEnum):
    """Severity band of an incident."""

    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"


class IncidentStatus(StrEnum):
    """Lifecycle of an incident."""

    REPORTED = "reported"
    CONFIRMED = "confirmed"
    RESPONDING = "responding"
    CONTAINED = "contained"
    CLEARED = "cleared"
    FALSE_ALARM = "false_alarm"


# ---------------------------------------------------------------------------
# Emergency response
# ---------------------------------------------------------------------------
class EmergencyVehicleType(StrEnum):
    """Type of responding unit. Determines the class of destination."""

    AMBULANCE = "ambulance"
    FIRE_ENGINE = "fire_engine"
    POLICE = "police"
    RESCUE = "rescue"
    OTHER = "other"


class EmergencyPriority(StrEnum):
    """Response priority, following common EMS dispatch grading.

    P1 is immediately life-threatening and is the only tier that justifies
    recommending pre-emption of conflicting movements across a corridor.
    """

    P1_IMMEDIATE = "P1_immediate"
    P2_URGENT = "P2_urgent"
    P3_ROUTINE = "P3_routine"


class EmergencyStatus(StrEnum):
    """Lifecycle of an emergency response event."""

    DECLARED = "declared"
    EN_ROUTE = "en_route"
    ARRIVED = "arrived"
    RETURNING = "returning"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------
class FreshnessStatus(StrEnum):
    """Age classification of an observation relative to the configured budget.

    This exists because "latest observation per junction" is not safe on its own:
    a 200 ms-old reading and a 45 s-old reading look identical once the timestamp
    is discarded. Freshness travels with the data all the way to the decision.
    """

    FRESH = "fresh"
    STALE = "stale"
    """Older than the staleness budget but still usable, with the decision
    downgraded and the condition surfaced as a warning."""

    EXPIRED = "expired"
    """Older than the drop threshold. Not usable as evidence."""

    UNKNOWN = "unknown"
    """No observation has ever been received for this element."""


class DataQualityFlag(StrEnum):
    """Conditions that degrade trust in an observation."""

    OCCLUSION = "occlusion"
    LOW_LIGHT = "low_light"
    GLARE = "glare"
    RAIN_ON_LENS = "rain_on_lens"
    CAMERA_MOTION = "camera_motion"
    OUT_OF_FOCUS = "out_of_focus"
    PARTIAL_FIELD_OF_VIEW = "partial_field_of_view"
    UNCALIBRATED = "uncalibrated"
    """No camera calibration available: pixel measurements cannot be converted to
    metric units, so speed and density must be reported as unavailable."""

    CLOCK_SKEW = "clock_skew"
    LOW_DETECTION_CONFIDENCE = "low_detection_confidence"
    DROPPED_FRAMES = "dropped_frames"
