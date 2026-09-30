"""Route optimization: structured facts in, ranked routes out.

    RoadNetwork + NetworkState + incidents  ->  hard pruning
    traffic state + observation mapping     ->  per-segment evidence
    admissible subgraph + cost weights      ->  Dijkstra / Yen's
                                            ->  RouteOptimizationResult

The optimizer chooses routes from structured data and nothing else. No language
model is consulted, and none could be: the ranking is a total order over an
additive cost in equivalent seconds, and every term in it must be reproducible
by hand from the result.

What this layer refuses to do
-----------------------------
It does not simulate, actuate, approve or narrate. It produces facts -- ranked
candidates, cost breakdowns, rejections and the evidence each cost rested on --
for the Decision Engine to act on and for Phase 12 to explain against a fact
table. Collapsing any of those into this layer would put the explanation and the
thing being explained in the same place, where they can no longer disagree.
"""

from app.routing.engine import RouteOptimizer, new_request_id, optimize
from app.routing.mapping import (
    MappingSource,
    MappingStore,
    ObservationBinding,
    ObservationMapping,
    load_mapping,
)
from app.routing.mapping import validate_against_network as validate_mapping
from app.routing.pruning import AdmissibleSubgraph, build_subgraph
from app.routing.schemas import (
    EvidenceStatus,
    OptimizationStatus,
    RankedRoute,
    ReasonCode,
    RejectionReason,
    RouteOptimizationResult,
    RouteReason,
    RouteRecommendation,
    RouteRejection,
    RouteRequest,
    SegmentEvidence,
)
from app.routing.search import (
    ALGORITHM,
    ALGORITHM_VERSION,
    SearchLimitReachedError,
    k_shortest_paths,
    shortest_path,
)
from app.routing.weights import RouteWeights, RoutingMode, weights_for

__all__ = [
    "ALGORITHM",
    "ALGORITHM_VERSION",
    "AdmissibleSubgraph",
    "EvidenceStatus",
    "MappingSource",
    "MappingStore",
    "ObservationBinding",
    "ObservationMapping",
    "OptimizationStatus",
    "RankedRoute",
    "ReasonCode",
    "RejectionReason",
    "RouteOptimizationResult",
    "RouteOptimizer",
    "RouteReason",
    "RouteRecommendation",
    "RouteRejection",
    "RouteRequest",
    "RouteWeights",
    "RoutingMode",
    "SearchLimitReachedError",
    "SegmentEvidence",
    "build_subgraph",
    "k_shortest_paths",
    "load_mapping",
    "new_request_id",
    "optimize",
    "shortest_path",
    "validate_mapping",
    "weights_for",
]
