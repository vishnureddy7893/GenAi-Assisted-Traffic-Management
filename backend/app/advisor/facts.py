"""The facts the advisor is allowed to talk about, and nothing else.

What this module is for
-----------------------
The LLM never sees the traffic system. It sees this: a flat set of values that
the deterministic pipeline already computed, each one carrying where it came
from. Anything absent from here the model has no basis to mention, and the
post-generation scan in :mod:`app.domain.decision` enforces that mechanically
rather than by asking politely.

Nothing here computes a traffic quantity. Every value is read off an object some
earlier phase produced -- an ``IntersectionTrafficState``, a ``TrafficAnalytics``
set, an ``ObservationSite``, a ``RouteOptimizationResult``, a ``SimulationRun``.
If this module ever starts deriving a number, it has become a second analysis
pipeline and the two will disagree.

Absence is a fact too
---------------------
The failure this design exists to prevent is an LLM filling a gap with a
plausible number. So a missing value is never defaulted to zero and never
silently dropped: it is recorded in :attr:`TrafficAdvisorFacts.unavailable` with
the reason it is missing, and the prompt shows that list to the model alongside
the values it does have. "Vehicle count is unavailable because no analysis has
been run" is a usable statement. A vehicle count of 0 is a lie.

Provenance survives the trip
----------------------------
Every fact carries the existing :class:`~app.domain.provenance.Provenance`, so
``observed``, ``derived``, ``predicted``, ``simulated`` and ``assumed`` mean here
exactly what they mean everywhere else in the system. Where the source object
already has provenance -- a surveyed site, a declared incident, a simulation run
-- that provenance is reused verbatim rather than re-classified, because
re-classifying it is how a simulated number quietly becomes a measured one.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Any

from pydantic import Field

from app.analytics.schemas import TrafficAnalytics
from app.core.versions import APP_VERSION
from app.domain.base import DomainModel
from app.domain.decision import Fact, FactTable
from app.domain.enums import CongestionLevel
from app.domain.incident import Incident
from app.domain.provenance import Provenance, ValueSource
from app.domain.site import ConfirmedSiteBinding, ObservationSite
from app.routing.schemas import RouteOptimizationResult
from app.simulation.schemas import SimulationRun
from app.state.schemas import IntersectionTrafficState

#: Recorded as the producer of any provenance this module has to construct,
#: which happens only where the source object carries none of its own.
ADVISOR_PRODUCER = "advisor.fact_builder"


class UnavailableFact(DomainModel):
    """A value the advisor would have liked and does not have.

    Carried explicitly so the model can say "this is unknown" instead of
    guessing, and so a reviewer can see what the advisory was working without.
    """

    fact_id: Annotated[str, Field(min_length=1, max_length=64)]
    label: Annotated[str, Field(min_length=1, max_length=200)]
    reason: Annotated[str, Field(min_length=1, max_length=300)]
    """Why it is missing. Never "0", never "n/a" -- a reason an operator can act
    on, such as "no traffic state has been interpreted for this media"."""


class TrafficAdvisorFacts(DomainModel):
    """Everything the advisor knows about one observation area at one instant.

    Named fields for what the pipeline produced, a derived
    :class:`~app.domain.decision.FactTable` for the numbers the narrative may
    cite, and an explicit list of what is missing.
    """

    # --- identity -------------------------------------------------------
    observation_area_id: Annotated[str, Field(min_length=1, max_length=120)]
    site_label: Annotated[str, Field(max_length=200)] | None = None
    latitude: float | None = None
    longitude: float | None = None

    confirmed_junction_id: Annotated[str, Field(max_length=64)] | None = None
    confirmed_segment_id: Annotated[str, Field(max_length=64)] | None = None
    confirmed_segment_direction: Annotated[str, Field(max_length=20)] | None = None
    """All three present only when an operator confirmed them. A binding is
    never inferred here -- see :mod:`app.sites.bindings`."""

    # --- measured / interpreted -----------------------------------------
    traffic_state: CongestionLevel | None = None
    congestion_index: float | None = None
    vehicle_count: int | None = None
    stationary_vehicle_count: int | None = None
    vehicle_class_counts: dict[str, int] = Field(default_factory=dict)
    pedestrian_count: int | None = None
    cyclist_count: int | None = None
    image_occupancy_ratio: float | None = None
    """Image-space. The fraction of *frame area* covered by road-user boxes.
    Not road occupancy and not density: both need geometry this has none of."""

    spatial_summary: Annotated[str, Field(max_length=300)] | None = None
    quality_grade: Annotated[str, Field(max_length=40)] | None = None
    state_as_of: dt.datetime | None = None

    # --- declared context -----------------------------------------------
    incident_summaries: tuple[str, ...] = ()
    """One line per *declared* incident. Empty means none was declared, which is
    not the same as none existing -- and the model is told so."""

    emergency_summaries: tuple[str, ...] = ()

    # --- route and simulation -------------------------------------------
    has_route_comparison: bool = False
    route_baseline_s: float | None = None
    route_alternative_s: float | None = None
    route_improvement_pct: float | None = None
    route_selected_id: Annotated[str, Field(max_length=64)] | None = None

    has_simulation: bool = False
    simulation_run_id: Annotated[str, Field(max_length=64)] | None = None
    simulation_travel_time_delta_s: float | None = None
    simulation_improvement_pct: float | None = None

    # --- the citable set -------------------------------------------------
    fact_table: FactTable = FactTable()
    unavailable: tuple[UnavailableFact, ...] = ()

    @property
    def has_state(self) -> bool:
        """Whether a traffic state was interpreted for this area."""
        return self.traffic_state is not None

    @property
    def has_incident(self) -> bool:
        """Whether any incident was *declared*. Never visually inferred."""
        return bool(self.incident_summaries)

    @property
    def has_emergency(self) -> bool:
        """Whether any emergency was *declared*. Never visually inferred."""
        return bool(self.emergency_summaries)


def _derived(
    observed_at: dt.datetime, computed_at: dt.datetime, *input_refs: str
) -> Provenance:
    """Provenance for a value read off a deterministic computation."""
    return Provenance(
        source=ValueSource.DERIVED,
        producer=ADVISOR_PRODUCER,
        producer_version=APP_VERSION,
        observed_at=observed_at,
        computed_at=computed_at,
        input_refs=tuple(input_refs),
        notes="Read from an existing pipeline result; not recomputed here.",
    )


class _Builder:
    """Accumulates facts and absences while walking the available inputs."""

    def __init__(self, now: dt.datetime) -> None:
        self.now = now
        self.facts: list[Fact] = []
        self.missing: list[UnavailableFact] = []
        self.tokens: set[str] = set()

    def add(
        self,
        fact_id: str,
        label: str,
        value: float | int | str | bool,
        provenance: Provenance,
        *,
        unit: str | None = None,
        variants: tuple[str, ...] = (),
    ) -> None:
        """Record a citable fact."""
        self.facts.append(
            Fact(
                fact_id=fact_id,
                label=label,
                value=value,
                unit=unit,
                provenance=provenance,
                display_variants=variants,
            )
        )

    def absent(self, fact_id: str, label: str, reason: str) -> None:
        """Record that a value is missing, and why."""
        self.missing.append(
            UnavailableFact(fact_id=fact_id, label=label, reason=reason)
        )

    def token(self, value: str | None) -> None:
        """Register an identifier that legitimately contains digits."""
        if value:
            self.tokens.add(value)


def build_facts(
    observation_area_id: str,
    *,
    state: IntersectionTrafficState | None = None,
    analytics: TrafficAnalytics | None = None,
    site: ObservationSite | None = None,
    binding: ConfirmedSiteBinding | None = None,
    incidents: tuple[Incident, ...] = (),
    emergencies: tuple[Any, ...] = (),
    route_result: RouteOptimizationResult | None = None,
    simulation: SimulationRun | None = None,
    now: dt.datetime,
) -> TrafficAdvisorFacts:
    """Assemble the advisor's view from whatever the pipeline has produced.

    Every argument is optional because every one of them genuinely may not
    exist: a freshly uploaded image has no traffic state, a process with no map
    has no route, and most demos have no declared incident. Each absence becomes
    an entry in ``unavailable`` rather than a zero.
    """
    builder = _Builder(now)
    builder.token(observation_area_id)

    payload: dict[str, Any] = {"observation_area_id": observation_area_id}

    _add_site(builder, payload, site, binding)
    _add_state(builder, payload, state, observation_area_id)
    _add_analytics(builder, payload, analytics, observation_area_id)
    _add_declared(builder, payload, incidents, emergencies)
    _add_route(builder, payload, route_result)
    _add_simulation(builder, payload, simulation)

    payload["fact_table"] = FactTable(
        facts=tuple(builder.facts), entity_tokens=frozenset(builder.tokens)
    )
    payload["unavailable"] = tuple(builder.missing)
    return TrafficAdvisorFacts.model_validate(payload)


def _add_site(
    builder: _Builder,
    payload: dict[str, Any],
    site: ObservationSite | None,
    binding: ConfirmedSiteBinding | None,
) -> None:
    """Location and confirmed bindings, both of which are survey facts."""
    if site is None:
        builder.absent(
            "site.location",
            "Observation site location",
            "no location has been recorded for this observation area",
        )
    else:
        payload["site_label"] = site.label
        payload["latitude"] = site.location.latitude
        payload["longitude"] = site.location.longitude
        # The site's own provenance is reused: an operator-entered coordinate
        # and a surveyed one are different evidence, and only the site knows
        # which this is.
        builder.add(
            "site.latitude",
            "Site latitude",
            site.location.latitude,
            site.provenance,
            unit="deg",
        )
        builder.add(
            "site.longitude",
            "Site longitude",
            site.location.longitude,
            site.provenance,
            unit="deg",
        )
        builder.token(site.label)

    if binding is None:
        builder.absent(
            "binding.junction",
            "Confirmed junction",
            "no operator has confirmed which junction this camera watches",
        )
        return

    payload["confirmed_junction_id"] = binding.junction_id
    builder.token(binding.junction_id)
    if binding.segment_id is not None and binding.segment_direction is not None:
        payload["confirmed_segment_id"] = binding.segment_id
        payload["confirmed_segment_direction"] = binding.segment_direction.value
        builder.token(binding.segment_id)
    else:
        builder.absent(
            "binding.segment",
            "Confirmed directed segment",
            "the junction is confirmed but no directed segment has been chosen, "
            "so evidence is not attributable to one direction of travel",
        )


def _add_state(
    builder: _Builder,
    payload: dict[str, Any],
    state: IntersectionTrafficState | None,
    area_id: str,
) -> None:
    """The interpreted traffic state -- the advisor's central fact."""
    if state is None:
        builder.absent(
            "state.level",
            "Traffic state",
            "no traffic state has been interpreted for this observation area; "
            "run the analysis and interpretation steps first",
        )
        builder.absent(
            "state.vehicle_count",
            "Vehicle count",
            "no traffic state has been interpreted, so no count exists",
        )
        return

    payload["traffic_state"] = state.congestion.level
    payload["quality_grade"] = state.quality_grade.value
    payload["state_as_of"] = state.as_of
    payload["vehicle_count"] = state.vehicle_count
    payload["stationary_vehicle_count"] = state.stationary_vehicle_count

    ref = f"analytics:{state.analytics_provenance.media_id}"
    provenance = _derived(state.as_of, state.assessed_at, ref, f"state:{state.state_id}")

    builder.add(
        "state.level",
        "Traffic state band",
        state.congestion.level.value,
        provenance,
    )
    builder.add(
        "state.vehicle_count",
        "Vehicles detected in the observation area",
        state.vehicle_count,
        provenance,
        unit="vehicles",
        variants=(f"{state.vehicle_count} vehicles",),
    )

    if state.congestion.index is None:
        builder.absent(
            "state.congestion_index",
            "Congestion index",
            "the traffic state is UNKNOWN, which by design carries no index",
        )
    else:
        payload["congestion_index"] = state.congestion.index
        builder.add(
            "state.congestion_index",
            "Congestion index (0-1)",
            round(state.congestion.index, 3),
            provenance,
            unit="ratio",
            variants=(f"{state.congestion.index:.2f}",),
        )

    if state.stationary_vehicle_count is None:
        builder.absent(
            "state.stationary_vehicle_count",
            "Stationary vehicle count",
            "stationary vehicles could not be measured for this observation",
        )
    else:
        builder.add(
            "state.stationary_vehicle_count",
            "Stationary vehicles",
            state.stationary_vehicle_count,
            provenance,
            unit="vehicles",
        )

    if state.congestion.reason:
        builder.add(
            "state.reason",
            "Why the system classified it this way",
            state.congestion.reason,
            provenance,
        )
    builder.token(area_id)


def _add_analytics(
    builder: _Builder,
    payload: dict[str, Any],
    analytics: TrafficAnalytics | None,
    area_id: str,
) -> None:
    """Class tallies and image-space occupancy, where they were computed."""
    if analytics is None:
        builder.absent(
            "analytics.class_counts",
            "Vehicle class counts",
            "no traffic analytics result is held for this observation area",
        )
        return

    ref = f"analytics:{analytics.analytics_id}"
    provenance = _derived(
        analytics.provenance.timestamp or builder.now, builder.now, ref
    )

    tally = analytics.tracked_object_tally
    counts = {
        vehicle_class.value: count
        for vehicle_class, count in tally.by_class.items()
        if count > 0
    }
    if counts:
        payload["vehicle_class_counts"] = counts
        for name, count in sorted(counts.items()):
            builder.add(
                f"analytics.count.{name}",
                f"Tracked {name} count",
                count,
                provenance,
                unit="objects",
            )

    payload["pedestrian_count"] = tally.pedestrians
    payload["cyclist_count"] = tally.cyclists
    builder.add(
        "analytics.pedestrians",
        "Pedestrians tracked in the observation area",
        tally.pedestrians,
        provenance,
        unit="people",
    )
    builder.add(
        "analytics.cyclists",
        "Cyclists tracked in the observation area",
        tally.cyclists,
        provenance,
        unit="people",
    )

    occupancy = analytics.spatial_distribution.image_occupancy_ratio
    if occupancy.is_available and occupancy.value is not None:
        payload["image_occupancy_ratio"] = occupancy.value
        builder.add(
            "analytics.image_occupancy",
            "Share of the frame covered by road users (image-space, not road occupancy)",
            round(occupancy.value, 3),
            provenance,
            unit="ratio",
            variants=(f"{occupancy.value * 100:.0f}% of the frame",),
        )
    else:
        builder.absent(
            "analytics.image_occupancy",
            "Image occupancy",
            occupancy.reason or "image occupancy was not computed",
        )

    distribution = analytics.spatial_distribution
    if distribution.centroid is not None:
        payload["spatial_summary"] = (
            f"Activity is concentrated over a {distribution.grid_rows}x"
            f"{distribution.grid_cols} image grid; this is frame geometry and "
            "says nothing about lanes or approaches."
        )
    builder.token(area_id)


def _add_declared(
    builder: _Builder,
    payload: dict[str, Any],
    incidents: tuple[Incident, ...],
    emergencies: tuple[Any, ...],
) -> None:
    """Incidents and emergencies, included only because somebody declared them.

    Neither is ever inferred from an image. A stopped vehicle is a stopped
    vehicle; calling it an accident is a claim about the world that no detector
    is in a position to make.
    """
    if incidents:
        payload["incident_summaries"] = tuple(
            f"{item.incident_type.value} ({item.severity.value}) declared at "
            f"{item.location.segment_id or item.location.junction_id or 'an unnamed location'}"
            for item in incidents
        )
        for item in incidents:
            builder.token(item.incident_id)
            builder.token(item.location.segment_id)
            builder.token(item.location.junction_id)
    else:
        builder.absent(
            "incidents.declared",
            "Declared incidents",
            "no incident has been declared; an incident is never inferred from "
            "detections, so absence here means nothing was reported",
        )

    if emergencies:
        payload["emergency_summaries"] = tuple(
            f"{getattr(item, 'emergency_id', 'emergency')} declared" for item in emergencies
        )
    else:
        builder.absent(
            "emergency.declared",
            "Declared emergency responses",
            "no emergency has been declared; an emergency is a dispatched "
            "response with a destination, never something visible in footage",
        )


def _add_route(
    builder: _Builder,
    payload: dict[str, Any],
    result: RouteOptimizationResult | None,
) -> None:
    """Route comparison, only when the optimizer actually produced one."""
    if result is None or len(result.routes) < 2:
        builder.absent(
            "route.comparison",
            "Route comparison",
            "no route optimization with at least two ranked candidates is "
            "available, so no alternative can be compared against a baseline",
        )
        return

    ranked = sorted(result.routes, key=lambda item: item.rank)
    best, runner_up = ranked[0], ranked[1]
    if not (best.travel_time_available and runner_up.travel_time_available):
        builder.absent(
            "route.comparison",
            "Route comparison",
            "ranked routes exist but at least one has no travel-time basis, so "
            "a duration comparison would be a partial sum presented as an ETA",
        )
        return

    alternative = best.estimated_travel_time_s
    baseline = runner_up.estimated_travel_time_s
    if alternative is None or baseline is None:  # pragma: no cover - guarded above
        return

    payload["has_route_comparison"] = True
    payload["route_baseline_s"] = baseline
    payload["route_alternative_s"] = alternative
    payload["route_selected_id"] = best.candidate.route_id
    builder.token(best.candidate.route_id)
    builder.token(runner_up.candidate.route_id)

    provenance = _derived(
        result.computed_at,
        result.computed_at,
        f"route_request:{result.request.request_id}",
        f"network:{result.network_id}",
    )
    builder.add(
        "route.baseline_s",
        "Baseline route predicted travel time",
        round(baseline, 1),
        provenance,
        unit="s",
        variants=(f"{baseline:.0f} s", f"{baseline / 60:.1f} min"),
    )
    builder.add(
        "route.alternative_s",
        "Recommended route predicted travel time",
        round(alternative, 1),
        provenance,
        unit="s",
        variants=(f"{alternative:.0f} s", f"{alternative / 60:.1f} min"),
    )

    if baseline > 0:
        improvement = (baseline - alternative) / baseline * 100.0
        payload["route_improvement_pct"] = improvement
        builder.add(
            "route.improvement_pct",
            "Predicted travel-time difference against the baseline",
            round(improvement, 1),
            provenance,
            unit="%",
            variants=(f"{improvement:.1f}%",),
        )


def _add_simulation(
    builder: _Builder, payload: dict[str, Any], run: SimulationRun | None
) -> None:
    """Simulation figures, carried with their own simulated provenance."""
    if run is None or run.comparison is None or not run.comparison.comparable:
        builder.absent(
            "simulation.comparison",
            "Simulation comparison",
            "no comparable simulation run is available",
        )
        return

    comparison = run.comparison
    payload["has_simulation"] = True
    payload["simulation_run_id"] = run.run_id
    builder.token(run.run_id)

    # The run's own provenance is reused verbatim. It says SIMULATED, and that
    # is the single most important word in this whole payload.
    if comparison.travel_time_delta_s is not None:
        delta = comparison.travel_time_delta_s.value
        payload["simulation_travel_time_delta_s"] = delta
        builder.add(
            "simulation.travel_time_delta_s",
            "Modelled travel-time difference (simulated, not measured)",
            round(delta, 1),
            comparison.travel_time_delta_s.provenance,
            unit="s",
            variants=(f"{delta:.0f} s",),
        )
    if comparison.improvement_pct is not None:
        payload["simulation_improvement_pct"] = comparison.improvement_pct
        builder.add(
            "simulation.improvement_pct",
            "Modelled improvement (simulated, not measured)",
            round(comparison.improvement_pct, 1),
            run.provenance,
            unit="%",
            variants=(f"{comparison.improvement_pct:.1f}%",),
        )
