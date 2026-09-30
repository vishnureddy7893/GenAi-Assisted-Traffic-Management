"""Version constants recorded into every decision, log line and artifact.

Reproducibility requirement
---------------------------
A recommendation is only auditable if we can answer: *which code, which schema,
which safety rules and which model produced it?*  Every version below is
stamped into the run manifest so that a decision can be replayed months later.

These are deliberately separate identifiers.  The safety rule catalog can be
revised without changing the wire schema, and a model can be retrained without
changing either.  Collapsing them into a single "app version" would destroy the
ability to attribute a behaviour change to its true cause.
"""

from __future__ import annotations

from typing import Final

#: Version of the deployable application (code).
APP_VERSION: Final[str] = "0.1.0"

#: Version of the domain wire schema (JunctionObservation / NetworkState / ...).
#: Bump on any breaking change to a field name, type or semantic.
SCHEMA_VERSION: Final[str] = "0.1.0"

#: Version of the safety rule catalog (which constraints exist and their tier).
#: Bump whenever a rule is added, removed, retiered or its threshold changes.
SAFETY_RULES_VERSION: Final[str] = "0.1.0"

#: Version of the route cost weight set (all weights expressed in equiv. seconds).
ROUTE_WEIGHTS_VERSION: Final[str] = "0.1.0"

#: Development phase currently implemented. Used by the API to tell the client
#: which capabilities genuinely exist, so the UI never implies working features.
IMPLEMENTED_PHASE: Final[int] = 8

#: Human-readable name of the current phase.
IMPLEMENTED_PHASE_NAME: Final[str] = "Phase 8 — Traffic Simulation"
