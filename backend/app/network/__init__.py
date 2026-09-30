"""Road network loading and per-instant network state.

    network file  ->  load_network  ->  RoadNetwork (+ NetworkIndex)
                  ->  NetworkStore  ->  held for the process
    RoadNetwork + incidents  ->  assemble_network_state  ->  NetworkState

The surveyed graph is the thing every later phase is keyed by: incidents name
segments, emergencies name junctions and facilities, and route search walks
successors. Loading it is therefore the first half of Phase 6, and the half that
everything about emergency management depends on.

Nothing here interprets traffic. Congestion comes from :mod:`app.state`, which
describes *observation areas* rather than segments; binding a camera to a
segment is a survey fact this system has not recorded, so the assembler leaves
congestion ``UNKNOWN`` rather than attaching a verdict to the wrong road.
"""

from app.network.assembler import (
    FUSION_STRATEGY,
    assemble_network_state,
    degraded_segment_ids,
)
from app.network.loader import (
    MAX_NETWORK_FILE_BYTES,
    NetworkSource,
    list_available,
    load_network,
)
from app.network.store import NetworkStore

__all__ = [
    "FUSION_STRATEGY",
    "MAX_NETWORK_FILE_BYTES",
    "NetworkSource",
    "NetworkStore",
    "assemble_network_state",
    "degraded_segment_ids",
    "list_available",
    "load_network",
]
