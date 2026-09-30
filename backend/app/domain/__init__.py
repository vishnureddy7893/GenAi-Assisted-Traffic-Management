"""Domain schemas.

Layering, from evidence to decision::

    JunctionObservation   one source, one junction, one instant  (observation.py)
            |
            v  fused with an explicit staleness budget
    NetworkState          the whole network at one instant       (state.py)
            |             over immutable topology                (network.py)
            |             plus incidents                         (incident.py)
            |             plus declared responses                (emergency.py)
            v
    SafetyConstraint      hard prunes / soft prices              (safety.py)
            |
            v
    RouteCandidate        admissible by construction             (routing.py)
            |
            v
    DecisionRecord        advisory output, fully auditable       (decision.py)

Every quantity travels as a :class:`~app.domain.provenance.Measurement` carrying
provenance, and absence is expressed as ``None`` rather than zero.
"""

from app.domain.base import DomainModel
from app.domain.decision import (
    CorridorAdvisory,
    DecisionRecord,
    Fact,
    FactTable,
    GenAIRecommendation,
    NarrativeValidation,
    RunManifest,
    SignalAdvisory,
    scan_for_unsupported_numbers,
)
from app.domain.emergency import (
    EmergencyDestination,
    EmergencyState,
    EmergencyVehicle,
    NetworkPosition,
)
from app.domain.incident import ExclusionZone, Incident, IncidentLocation
from app.domain.network import (
    Approach,
    Facility,
    GeoPoint,
    Junction,
    Lane,
    NetworkIndex,
    PhysicalRestrictions,
    RoadNetwork,
    RoadSegment,
    TurnMovement,
)
from app.domain.observation import (
    ApproachObservation,
    CameraCalibration,
    JunctionObservation,
    ObservationQuality,
    PedestrianObservation,
)
from app.domain.provenance import (
    Measurement,
    Provenance,
    Unit,
    ValueSource,
    make_provenance,
)
from app.domain.routing import (
    RouteCandidate,
    RouteComparison,
    RouteCostBreakdown,
    RouteLeg,
)
from app.domain.safety import (
    PHASE1_SAFETY_CATALOG,
    ConstraintScope,
    ConstraintTier,
    SafetyConstraint,
    SafetyRule,
    SafetyRuleCatalog,
)
from app.domain.site import (
    OBSERVATION_AREA_PREFIX,
    ObservationSite,
    observation_area_id_for_media,
)
from app.domain.state import (
    ApproachState,
    EnvironmentConditions,
    JunctionState,
    NetworkState,
    RoadSegmentState,
    StateAssemblyPolicy,
    TravelTimeEstimate,
)

__all__ = [
    "OBSERVATION_AREA_PREFIX",
    "PHASE1_SAFETY_CATALOG",
    "Approach",
    "ApproachObservation",
    "ApproachState",
    "CameraCalibration",
    "ConstraintScope",
    "ConstraintTier",
    "CorridorAdvisory",
    "DecisionRecord",
    "DomainModel",
    "EmergencyDestination",
    "EmergencyState",
    "EmergencyVehicle",
    "EnvironmentConditions",
    "ExclusionZone",
    "Facility",
    "Fact",
    "FactTable",
    "GenAIRecommendation",
    "GeoPoint",
    "Incident",
    "IncidentLocation",
    "Junction",
    "JunctionObservation",
    "JunctionState",
    "Lane",
    "Measurement",
    "NarrativeValidation",
    "NetworkIndex",
    "NetworkPosition",
    "NetworkState",
    "ObservationQuality",
    "ObservationSite",
    "PedestrianObservation",
    "PhysicalRestrictions",
    "Provenance",
    "RoadNetwork",
    "RoadSegment",
    "RoadSegmentState",
    "RouteCandidate",
    "RouteComparison",
    "RouteCostBreakdown",
    "RouteLeg",
    "RunManifest",
    "SafetyConstraint",
    "SafetyRule",
    "SafetyRuleCatalog",
    "SignalAdvisory",
    "StateAssemblyPolicy",
    "TravelTimeEstimate",
    "TurnMovement",
    "Unit",
    "ValueSource",
    "make_provenance",
    "observation_area_id_for_media",
    "scan_for_unsupported_numbers",
]
