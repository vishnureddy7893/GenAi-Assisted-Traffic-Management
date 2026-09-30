"""The development phase plan, as data.

Kept in code rather than only in documentation so that the API, the dashboard
and the error messages all read the same source. When a capability is missing,
the 501 response names the phase that will provide it, and the dashboard labels
the panel with the same number -- because both are reading this table.

Two ordering decisions are worth calling out:

**Dataset inspection is part of Phase 2**, immediately after the foundation and
before any computer vision, feature engineering or model selection. Building a
CV pipeline and a feature set first, then discovering the dataset lacks the
ground truth those features assume, is the single most common way a project of
this shape loses a month. Every design choice downstream is contingent on what
the data actually contains.

**Traffic analytics (Phase 4) precedes traffic state (Phase 5).** Analytics
turns detections and tracks into measurements at a single observed scene, which
needs only perception plus a camera calibration. Traffic state interprets those
measurements and fuses them under a staleness budget, and its unit of input is
precisely what Phase 4 produces. Building the interpretation layer first would
mean interpreting a value type that did not exist yet.

**The surveyed road graph arrives in Phase 6, with emergency management.** It is
placed there rather than earlier because Phase 6 is the first phase that cannot
work without it: an incident names the segments it blocks and a dispatch names
the junction it starts from, so both are meaningless before a map exists.
Everything up to Phase 5 describes what a camera saw, which needs no survey.
"""

from __future__ import annotations

from typing import Final, NamedTuple

from app.core.versions import IMPLEMENTED_PHASE


class Phase(NamedTuple):
    """One development phase."""

    number: int
    name: str
    summary: str

    @property
    def implemented(self) -> bool:
        """Whether this phase is complete in the current build."""
        return self.number <= IMPLEMENTED_PHASE


PHASES: Final[tuple[Phase, ...]] = (
    Phase(
        1,
        "Foundation",
        "Repository, typed configuration, structured logging, domain schemas, "
        "source and vision abstractions, API skeleton, dashboard shell, tests.",
    ),
    Phase(
        2,
        "Data Ingestion",
        "Inspect the real dataset before designing features: structure, labels, "
        "ground truth, temporal properties, missingness, quality. Then build "
        "Mode A ingestion against what is actually there.",
    ),
    Phase(
        3,
        "Computer Vision",
        "Select a detector and tracker against measured accuracy and throughput "
        "on the real footage; implement detection, tracking and the video source.",
    ),
    Phase(
        4,
        "Traffic Analytics",
        "Counts, flow, density, queue evidence and image-space speeds derived "
        "from detections and tracks, each with provenance, and each metric "
        "requiring calibration reported as unavailable without one.",
    ),
    Phase(
        5,
        "Traffic State",
        "Interpret Phase 4 measurements as a congestion level per observation "
        "area, using configurable thresholds and stating the signals behind "
        "each verdict; fuse those into a snapshot under an explicit staleness "
        "budget.",
    ),
    Phase(
        6,
        "Road Network + Emergency Management",
        "Load and validate the surveyed road network graph; then incidents with "
        "resolved exclusion zones, declared emergency responses, both with "
        "explicit lifecycles, and the per-instant network state their blockages "
        "impose.",
    ),
    Phase(
        7,
        "Route Optimization",
        "Hard-constraint graph pruning, movement-aware Dijkstra with Yen's "
        "alternatives, risk-adjusted cost in equivalent seconds, and the "
        "observation-to-network mapping that lets traffic evidence reach a "
        "segment at all.",
    ),
    Phase(
        8,
        "Traffic Simulation",
        "Deterministic point-queue simulation over a configured horizon, driving "
        "a baseline and a recommended route through identical conditions and "
        "reporting travel time, delay, queueing, throughput and the difference "
        "between them. Every figure carries SIMULATED provenance.",
    ),
    Phase(
        9,
        "Decision Engine",
        "End-to-end: observation to decision to simulation to evaluation, "
        "recorded as a reproducible DecisionRecord.",
    ),
    Phase(
        10,
        "ML Models where justified",
        "Feature engineering, temporal splits, dataset-driven model selection "
        "and honest evaluation; then a versioned registry and inference in the "
        "decision path, with predictions carrying PREDICTED provenance.",
    ),
    Phase(
        11,
        "Safety & Validation",
        "Constraint evaluation, hard pruning before search, calibrated soft "
        "penalties, warnings, and the no-admissible-route outcome.",
    ),
    Phase(
        12,
        "GenAI Explanation",
        "Pretrained LLM via OpenRouter over structured state and a fact table, "
        "with mechanical post-generation validation of every number.",
    ),
    Phase(
        13,
        "Evaluation & Integration",
        "Performance, resilience, observability, documentation and the final "
        "measured evaluation of the integrated system.",
    ),
    Phase(
        14,
        "Live Traffic Extension",
        "Mode B sources with backpressure, reconnection and latency reporting, "
        "converging on the same decision pipeline.",
    ),
)


def phase(number: int) -> Phase:
    """Return one phase by number, raising ``KeyError`` if out of range."""
    for item in PHASES:
        if item.number == number:
            return item
    raise KeyError(number)
