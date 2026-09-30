"""Incidents and declared emergency responses.

    incident report  ->  IncidentRegistry   ->  Incident (lifecycle-tracked)
    centre + radius  ->  resolve_zone       ->  ExclusionZone over graph elements
    dispatch         ->  EmergencyRegistry  ->  EmergencyState (lifecycle-tracked)

An incident and an emergency response are different things, and keeping them
apart is the point of the split. A fire is an incident; the engine dispatched to
it is a response. One incident may produce several responses, and a response may
exist with no incident record at all -- a routine transfer between hospitals is
still an emergency vehicle needing a route.

Everything here is validated against the **loaded network**. An incident naming
a segment the map does not have would appear active in every listing while
pruning nothing, and a dispatch naming a missing junction fails inside route
search on a live request. Both are cheap to reject at declaration.

Nothing here plans a route or recommends an action. Producing a route is Phase 7;
this phase establishes what the route will have to avoid.
"""

from app.emergency.incidents import (
    INCIDENT_TRANSITIONS,
    IncidentRegistry,
)
from app.emergency.incidents import (
    TERMINAL_STATUSES as TERMINAL_INCIDENT_STATUSES,
)
from app.emergency.incidents import (
    validate_against_network as validate_incident,
)
from app.emergency.responses import (
    EMERGENCY_TRANSITIONS,
    EmergencyRegistry,
)
from app.emergency.responses import (
    TERMINAL_STATUSES as TERMINAL_EMERGENCY_STATUSES,
)
from app.emergency.responses import (
    validate_against_network as validate_emergency,
)
from app.emergency.zones import ZoneResolution, haversine_m, resolve_zone

__all__ = [
    "EMERGENCY_TRANSITIONS",
    "INCIDENT_TRANSITIONS",
    "TERMINAL_EMERGENCY_STATUSES",
    "TERMINAL_INCIDENT_STATUSES",
    "EmergencyRegistry",
    "IncidentRegistry",
    "ZoneResolution",
    "haversine_m",
    "resolve_zone",
    "validate_emergency",
    "validate_incident",
]
