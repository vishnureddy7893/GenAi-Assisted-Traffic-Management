"""Network state: Phase 4 measurements interpreted as traffic conditions.

    TrafficAnalytics  ->  classify  ->  CongestionAssessment
                      ->  interpret ->  IntersectionTrafficState
                      ->  assemble  ->  NetworkTrafficState

Answers *"what does the measured behaviour mean?"* and stops there. No travel
time, no route, no advisory, no emergency handling: those act on an interpreted
state rather than producing one, and belong to later phases.

The boundary against Phase 4 runs the other way too. Nothing here detects,
tracks or measures anything. Every number this package reads was produced by
:mod:`app.analytics`, and every number it writes is derived from those, so a
disagreement about a congestion level can always be traced to either a threshold
or a measurement -- never to a third, hidden source of traffic data.

Interpretation is refused rather than guessed. A verdict that cannot be
supported is ``CongestionLevel.UNKNOWN`` carrying the reason, never a default
band -- because the default that would look most natural, free flow, is the one
that makes a jammed road appear to be the fast route.
"""

from app.state.assembler import (
    FUSION_STRATEGY,
    STATE_SCHEMA_VERSION,
    assemble,
    freshness_for,
    interpret,
    observed_at,
    policy_from,
)
from app.state.classifier import (
    METHOD,
    METHOD_VERSION,
    band_for,
    classify,
    thresholds_from,
    vehicle_count_for,
)
from app.state.schemas import (
    CongestionAssessment,
    CongestionSignal,
    CongestionThresholds,
    EvidenceBasis,
    IntersectionStateElement,
    IntersectionTrafficState,
    NetworkTrafficState,
    SignalName,
)

__all__ = [
    "FUSION_STRATEGY",
    "METHOD",
    "METHOD_VERSION",
    "STATE_SCHEMA_VERSION",
    "CongestionAssessment",
    "CongestionSignal",
    "CongestionThresholds",
    "EvidenceBasis",
    "IntersectionStateElement",
    "IntersectionTrafficState",
    "NetworkTrafficState",
    "SignalName",
    "assemble",
    "band_for",
    "classify",
    "freshness_for",
    "interpret",
    "observed_at",
    "policy_from",
    "thresholds_from",
    "vehicle_count_for",
]
