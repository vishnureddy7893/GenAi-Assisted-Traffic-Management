"""Simulation request, scenario and result schemas.

What a simulated number is, and is not
--------------------------------------
Every quantity this layer produces is the outcome of a *model*, not a
measurement of any road. That is enforced rather than asserted: each headline
figure is a :class:`~app.domain.provenance.Measurement` whose provenance carries
:attr:`~app.domain.provenance.ValueSource.SIMULATED`, so
:attr:`~app.domain.provenance.Provenance.is_factual` is ``False`` and no
consumer -- including the Phase 12 narrative -- can state it as fact about the
world without stripping the modality on purpose.

The reference quantities beside them are labelled differently on purpose. A
free-flow travel time is ``DERIVED``: it comes from surveyed lengths and speeds
by a documented formula and describes an empty road. A demand read from an
observation is ``DERIVED`` from an ``OBSERVED`` measurement and says so. A demand
an operator states is ``ASSUMED`` and must carry a justification. Reading a
result therefore tells you which of the four kinds every number is, which is the
whole point of the exercise.

Absent inputs produce absent outputs
------------------------------------
There is no default demand, no default capacity and no default free-flow speed
anywhere in this package. A link missing any of them is reported as unmodelled,
with the reason, and the metrics it would have contributed are ``None``. The
alternative -- a plausible default -- would make the simulator's confident output
indistinguishable from an informed one, which is the specific failure a
simulation is most likely to cause.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from typing import Annotated

from pydantic import Field, model_validator

from app.domain.base import DomainModel
from app.domain.provenance import FloatMeasurement, Provenance
from app.domain.routing import TimeBasis
from app.routing.weights import RoutingMode

#: Name and version of the model implemented in :mod:`app.simulation.model`.
#: Recorded in every run so a figure can be attributed to the code that made it.
MODEL_NAME = "deterministic-point-queue"
MODEL_VERSION = "1.0"

#: Producer identifier stamped into every provenance record this layer writes.
SIM_PRODUCER = "simulation.point_queue"


class SimulationStatus(StrEnum):
    """Whether a run produced a modelled outcome at all."""

    COMPLETED = "completed"
    """The scenario was built and the model ran. Individual arms or links may
    still be unavailable, each saying why."""

    UNAVAILABLE = "unavailable"
    """The run could not be built. A first-class outcome carrying its causes,
    never an empty success: an empty simulation result reads as "nothing
    happened on the network", which is a claim about traffic."""


class ArmStatus(StrEnum):
    """Whether one evaluated route produced a modelled traversal."""

    SIMULATED = "simulated"
    UNAVAILABLE = "unavailable"


class LinkStatus(StrEnum):
    """Whether one segment's traffic could be modelled over the horizon."""

    MODELLED = "modelled"
    NOT_MODELLED = "not_modelled"
    """No demand basis, no capacity basis, or no traversal time. The segment is
    still listed, with the reason, so "not modelled" cannot be mistaken for
    "modelled and empty"."""


class UnavailableReason(StrEnum):
    """Why something could not be simulated. Codes, not prose."""

    NO_ROUTE_SUPPLIED = "no_route_supplied"
    NO_ADMISSIBLE_ROUTE = "no_admissible_route"
    """The optimizer found nothing to simulate. Propagated rather than
    re-derived, so the simulation and the optimizer cannot disagree."""

    ROUTE_NOT_ON_NETWORK = "route_not_on_network"
    ROUTE_DISCONTINUOUS = "route_discontinuous"
    SEGMENT_BLOCKED = "segment_blocked"
    """A leg crosses a segment an active incident blocks. Simulating a traversal
    of an impassable road would produce a travel time for a journey that cannot
    be made."""

    NO_TRAVERSAL_TIME_BASIS = "no_traversal_time_basis"
    """Some leg has neither a travel-time estimate nor a surveyed free-flow
    speed. No speed is assumed in its place, so the route has no travel time."""

    NO_CAPACITY_BASIS = "no_capacity_basis"
    NO_DEMAND_BASIS = "no_demand_basis"
    HORIZON_EXCEEDED = "horizon_exceeded"
    """The vehicle had not arrived when the horizon ended. The travel time is
    unavailable rather than truncated, because a truncated one understates the
    journey by exactly the part that was cut off."""

    ONLY_ONE_CANDIDATE = "only_one_candidate"
    """Nothing to compare against: the optimizer returned a single admissible
    route, so there is no alternative arm."""

    ARM_UNAVAILABLE = "arm_unavailable"
    """One side of a comparison produced no figure, so the difference between
    them does not exist. Reported instead of a one-sided number."""


class InputBasis(StrEnum):
    """Where one simulation input came from.

    Recorded per input, not per run, because a scenario routinely mixes them: a
    surveyed capacity on one link, a stated one on the next, and an observed
    demand on a third. A single run-level label would have to flatten that into
    the most flattering of the three.
    """

    SURVEYED = "surveyed"
    """From the road network file: a property of the map."""

    OBSERVED = "observed"
    """Derived from a measurement this system actually took, attributed to the
    segment only through an explicit observation-to-network binding."""

    STATED = "stated"
    """An assumption supplied on the request or in configuration. Carried as
    ``ASSUMED`` provenance and required to justify itself."""

    UNAVAILABLE = "unavailable"
    """Nothing supplied it. Never silently replaced by a default."""


class BaselineStrategy(StrEnum):
    """Which route the recommended one is compared against."""

    SHORTEST_DISTANCE = "shortest_distance"
    """The cheapest route by distance over the same admissible subgraph -- what
    an operator would take from a map alone, ignoring traffic evidence."""

    RUNNER_UP = "runner_up"
    """The optimizer's second-ranked candidate."""

    EXPLICIT = "explicit"
    """A route named on the request, segment by segment."""


class RouteSource(StrEnum):
    """How an evaluated route was obtained."""

    OPTIMIZER_SELECTED = "optimizer_selected"
    OPTIMIZER_ALTERNATIVE = "optimizer_alternative"
    SHORTEST_DISTANCE_SEARCH = "shortest_distance_search"
    EXPLICIT = "explicit"


class ArmRole(StrEnum):
    """What one evaluated route is in the comparison."""

    BASELINE = "baseline"
    RECOMMENDED = "recommended"
    EVALUATED = "evaluated"
    """A single route simulated on its own, with nothing to compare it to."""


class BetterArm(StrEnum):
    """Which arm the comparison favours."""

    BASELINE = "baseline"
    ALTERNATIVE = "alternative"
    NO_DIFFERENCE = "no_difference"
    UNKNOWN = "unknown"
    """At least one arm produced no figure. Never resolved to a winner by
    treating an unavailable value as a large one."""


class SimulationRejection(DomainModel):
    """One structured reason a run, an arm or a link produced nothing."""

    reason: UnavailableReason
    message: Annotated[str, Field(min_length=1, max_length=500)]
    segment_ids: tuple[str, ...] = ()
    junction_ids: tuple[str, ...] = ()
    route_ids: tuple[str, ...] = ()
    incident_ids: tuple[str, ...] = ()


class SimulationParameters(DomainModel):
    """The model parameters in force for one run.

    Embedded in the result rather than read from configuration when it is
    displayed. Two runs under different parameters are not comparable, and
    without the parameters recorded beside the numbers that difference is
    invisible.
    """

    horizon_s: Annotated[float, Field(gt=0)]
    step_s: Annotated[float, Field(gt=0)]
    step_count: Annotated[int, Field(ge=1)]
    departure_offset_s: Annotated[float, Field(ge=0)]
    emergency_queue_bypass_fraction: Annotated[float, Field(ge=0.0, le=1.0)]
    use_observed_demand: bool
    baseline_strategy: BaselineStrategy
    parameters_version: Annotated[str, Field(min_length=1, max_length=50)]

    model_name: Annotated[str, Field(min_length=1, max_length=80)] = MODEL_NAME
    model_version: Annotated[str, Field(min_length=1, max_length=40)] = MODEL_VERSION

    is_stochastic: bool = False
    """False, and there is no seed field anywhere in this package.

    The model contains no random component at all -- no sampled headways, no
    randomised departure times, no noise. Determinism is therefore a property of
    the model rather than of a seed a caller could forget to pin, and identical
    inputs produce byte-identical output. A seed parameter would imply a
    stochastic model and invite a reader to average runs that are all the same.
    """


class LinkInputs(DomainModel):
    """Everything the model knows about one directed segment, and its origin."""

    segment_id: Annotated[str, Field(min_length=1, max_length=64)]
    length_m: Annotated[float, Field(ge=0)]
    lane_count: Annotated[int, Field(ge=1)]

    free_flow_traversal_s: Annotated[float, Field(gt=0)] | None = None
    time_basis: TimeBasis = TimeBasis.UNAVAILABLE
    time_reason: Annotated[str, Field(max_length=400)] | None = None

    capacity_veh_per_hour: Annotated[float, Field(gt=0)] | None = None
    capacity_basis: InputBasis = InputBasis.UNAVAILABLE
    capacity_reason: Annotated[str, Field(max_length=400)] | None = None

    demand_veh_per_hour: Annotated[float, Field(ge=0)] | None = None
    demand_basis: InputBasis = InputBasis.UNAVAILABLE
    demand_reason: Annotated[str, Field(max_length=400)] | None = None

    initial_queue_veh: Annotated[float, Field(ge=0)] | None = None
    initial_queue_basis: InputBasis = InputBasis.UNAVAILABLE
    initial_queue_reason: Annotated[str, Field(max_length=400)] | None = None

    observation_area_id: Annotated[str, Field(max_length=120)] | None = None
    """The binding through which observed inputs reached this segment, if any."""

    is_blocked: bool = False
    blocked_by_incident_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _bases_match_content(self) -> LinkInputs:
        pairs = (
            ("capacity_veh_per_hour", self.capacity_veh_per_hour, self.capacity_basis),
            ("demand_veh_per_hour", self.demand_veh_per_hour, self.demand_basis),
            ("initial_queue_veh", self.initial_queue_veh, self.initial_queue_basis),
        )
        for name, value, basis in pairs:
            if value is None and basis is not InputBasis.UNAVAILABLE:
                msg = f"{name} is absent but claims basis {basis.value!r}"
                raise ValueError(msg)
            if value is not None and basis is InputBasis.UNAVAILABLE:
                msg = (
                    f"{name} carries a value with an unavailable basis; a number "
                    "whose origin is unstated is exactly what this type prevents"
                )
                raise ValueError(msg)
        if self.is_blocked and not self.blocked_by_incident_ids:
            msg = (
                f"link {self.segment_id!r} is blocked with no causing incident; a "
                "blockage with no provenance cannot be explained or cleared"
            )
            raise ValueError(msg)
        return self

    @property
    def is_modellable(self) -> bool:
        """Whether queueing can be evolved on this link at all."""
        return (
            self.free_flow_traversal_s is not None
            and self.capacity_veh_per_hour is not None
            and self.demand_veh_per_hour is not None
        )


class RouteArmInput(DomainModel):
    """One route to be driven through the simulated network."""

    role: ArmRole
    route_id: Annotated[str, Field(min_length=1, max_length=64)]
    label: Annotated[str, Field(min_length=1, max_length=120)]
    source: RouteSource
    mode: RoutingMode
    segment_ids: tuple[str, ...]
    origin_junction_id: Annotated[str, Field(min_length=1, max_length=64)]
    destination_junction_id: Annotated[str, Field(min_length=1, max_length=64)]

    @model_validator(mode="after")
    def _has_legs(self) -> RouteArmInput:
        if not self.segment_ids:
            msg = f"route arm {self.route_id!r} traverses no segment"
            raise ValueError(msg)
        return self


class SimulationScenario(DomainModel):
    """The complete, timestamp-free description of what was simulated.

    Deliberately carries no clock reading. The scenario is hashed to produce the
    run identifier, so anything that varies between two otherwise identical runs
    -- a wall-clock time above all -- must stay out of it. What belongs here is
    what changes the answer.
    """

    scenario_id: Annotated[str, Field(min_length=1, max_length=64)]
    network_id: Annotated[str, Field(min_length=1, max_length=64)]
    network_version: Annotated[str, Field(min_length=1, max_length=50)]
    parameters: SimulationParameters
    links: tuple[LinkInputs, ...] = ()
    arms: tuple[RouteArmInput, ...] = ()

    traffic_state_id: Annotated[str, Field(max_length=64)] | None = None
    mapping_id: Annotated[str, Field(max_length=64)] | None = None
    mapping_version: Annotated[str, Field(max_length=50)] | None = None
    active_incident_ids: tuple[str, ...] = ()

    input_fingerprint: Annotated[str, Field(min_length=8, max_length=64)]
    """SHA-256 over the canonical form of everything above. Two runs sharing a
    fingerprint were given identical inputs and must produce identical output;
    the determinism tests assert exactly that."""


class LegResult(DomainModel):
    """One segment traversal by the evaluated vehicle.

    Times are seconds on the simulation clock, which starts at zero at the top
    of the horizon. They are plain numbers rather than measurements because the
    whole object is simulated output: the route's provenance covers it, and a
    provenance record per leg field would be noise around an already-labelled
    fact.
    """

    segment_id: Annotated[str, Field(min_length=1, max_length=64)]
    from_junction_id: Annotated[str, Field(min_length=1, max_length=64)]
    to_junction_id: Annotated[str, Field(min_length=1, max_length=64)]
    entry_time_s: Annotated[float, Field(ge=0)]
    exit_time_s: Annotated[float, Field(ge=0)]
    free_flow_traversal_s: Annotated[float, Field(ge=0)]
    time_basis: TimeBasis

    queue_delay_s: Annotated[float, Field(ge=0)] | None = None
    """Time spent waiting for the queue ahead to discharge at this link's exit.
    ``None`` when this link could not be modelled, which is not the same as
    zero: zero means "modelled, and no queue was there"."""

    queue_modelled: bool = False
    queue_at_arrival_veh: Annotated[float, Field(ge=0)] | None = None
    reason: Annotated[str, Field(max_length=400)] | None = None


class RouteSimulationResult(DomainModel):
    """What happened to one evaluated route over the horizon."""

    role: ArmRole
    route_id: Annotated[str, Field(min_length=1, max_length=64)]
    label: Annotated[str, Field(min_length=1, max_length=120)]
    source: RouteSource
    mode: RoutingMode
    status: ArmStatus

    segment_ids: tuple[str, ...] = ()
    junction_sequence: tuple[str, ...] = ()
    legs: tuple[LegResult, ...] = ()

    travel_time_s: FloatMeasurement | None = None
    """Simulated door-to-door traversal: free-flow running plus queueing."""

    free_flow_travel_time_s: FloatMeasurement | None = None
    """The same route with no queueing at all: the reference the delay is
    measured against. ``DERIVED``, not ``SIMULATED`` -- it is arithmetic over
    surveyed lengths and speeds, and describes an empty road."""

    delay_s: FloatMeasurement | None = None
    """Simulated travel time minus the free-flow reference. Where some links
    could not be modelled this is a lower bound, and those links are named in
    ``unmodelled_segment_ids``."""

    completion_time_s: FloatMeasurement | None = None
    """Arrival instant on the simulation clock: departure offset plus travel
    time. Distinct from travel time whenever the vehicle departs into a horizon
    that has already been running."""

    mean_speed_kph: FloatMeasurement | None = None
    total_distance_m: FloatMeasurement | None = None

    completed_within_horizon: bool = False
    unmodelled_segment_ids: tuple[str, ...] = ()
    modelled_leg_count: Annotated[int, Field(ge=0)] = 0
    rejections: tuple[SimulationRejection, ...] = ()
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _status_matches_content(self) -> RouteSimulationResult:
        if self.status is ArmStatus.SIMULATED:
            if self.travel_time_s is None:
                msg = "a simulated arm must carry a travel time"
                raise ValueError(msg)
            if not self.legs:
                msg = "a simulated arm must carry its legs"
                raise ValueError(msg)
        else:
            if self.travel_time_s is not None:
                msg = (
                    "an unavailable arm must not carry a travel time; a partial or "
                    "truncated journey presented as an arrival is the failure this "
                    "status exists to prevent"
                )
                raise ValueError(msg)
            if not self.rejections:
                msg = "an unavailable arm must state why"
                raise ValueError(msg)
        return self


class LinkResult(DomainModel):
    """Queueing outcome on one directed segment over the horizon."""

    segment_id: Annotated[str, Field(min_length=1, max_length=64)]
    status: LinkStatus
    reason: Annotated[str, Field(max_length=400)] | None = None

    demand_veh_per_hour: Annotated[float, Field(ge=0)] | None = None
    demand_basis: InputBasis = InputBasis.UNAVAILABLE
    capacity_veh_per_hour: Annotated[float, Field(gt=0)] | None = None
    capacity_basis: InputBasis = InputBasis.UNAVAILABLE

    saturation_ratio: FloatMeasurement | None = None
    """Demand over capacity. ``DERIVED``: a ratio of two stated inputs, not an
    outcome of the run. Above 1.0 the queue grows for as long as the demand
    holds, which is the regime the horizon exists to bound."""

    peak_queue_veh: FloatMeasurement | None = None
    final_queue_veh: FloatMeasurement | None = None
    mean_queue_veh: FloatMeasurement | None = None
    throughput_veh: FloatMeasurement | None = None
    throughput_veh_per_hour: FloatMeasurement | None = None
    total_delay_veh_s: FloatMeasurement | None = None
    mean_delay_per_vehicle_s: FloatMeasurement | None = None

    queue_series_veh: tuple[float, ...] = ()
    """Queue length at the end of each step. Present so a reader can see the
    shape of the run rather than trusting a summary of it."""

    @model_validator(mode="after")
    def _not_modelled_carries_nothing(self) -> LinkResult:
        if self.status is LinkStatus.NOT_MODELLED:
            populated = [
                name
                for name in (
                    "peak_queue_veh",
                    "final_queue_veh",
                    "mean_queue_veh",
                    "throughput_veh",
                    "total_delay_veh_s",
                )
                if getattr(self, name) is not None
            ]
            if populated:
                msg = f"an unmodelled link must carry no outcome, but has {populated}"
                raise ValueError(msg)
            if not self.reason:
                msg = "an unmodelled link must state why"
                raise ValueError(msg)
        return self


class NetworkSimulationSummary(DomainModel):
    """Network-wide totals, scoped explicitly to the links that were modelled.

    Every field says ``modelled`` in its name for one reason: a total over part
    of a network reads as a total over all of it, and the segments left out are
    exactly the ones nobody has the data for. ``unmodelled_link_count`` is
    therefore not an afterthought but the number that makes the others
    interpretable.
    """

    modelled_link_count: Annotated[int, Field(ge=0)] = 0
    unmodelled_link_count: Annotated[int, Field(ge=0)] = 0
    modelled_throughput_veh: FloatMeasurement | None = None
    modelled_peak_queue_veh: FloatMeasurement | None = None
    modelled_final_queue_veh: FloatMeasurement | None = None
    modelled_total_delay_veh_s: FloatMeasurement | None = None
    oversaturated_segment_ids: tuple[str, ...] = ()
    """Links whose demand exceeds their capacity, where the queue grows for as
    long as the demand holds. Named because their figures depend on the horizon
    rather than converging within it."""

    note: Annotated[str, Field(max_length=600)]


class SimulationComparison(DomainModel):
    """Baseline against alternative, or a statement that it cannot be made."""

    baseline_route_id: Annotated[str, Field(min_length=1, max_length=64)]
    alternative_route_id: Annotated[str, Field(min_length=1, max_length=64)]
    comparable: bool
    reason: Annotated[str, Field(max_length=500)] | None = None

    travel_time_delta_s: FloatMeasurement | None = None
    """Baseline minus alternative. Positive means the alternative is faster."""

    delay_delta_s: FloatMeasurement | None = None
    completion_time_delta_s: FloatMeasurement | None = None
    distance_delta_m: FloatMeasurement | None = None
    improvement_pct: float | None = None
    """Travel-time saving as a percentage of the baseline. Negative when the
    alternative is slower, and deliberately unbounded below: a route twice as
    slow is a real outcome and clamping it would hide the worst case. ``None``
    whenever either arm is unavailable, or the baseline time is zero."""

    better: BetterArm = BetterArm.UNKNOWN
    rejections: tuple[SimulationRejection, ...] = ()

    @model_validator(mode="after")
    def _incomparable_carries_no_figures(self) -> SimulationComparison:
        if not self.comparable:
            populated = [
                name
                for name in ("travel_time_delta_s", "delay_delta_s", "improvement_pct")
                if getattr(self, name) is not None
            ]
            if populated:
                msg = (
                    f"an incomparable pair must carry no difference, but has "
                    f"{populated}; a one-sided figure presented as an improvement is "
                    "how a simulation invents a benefit"
                )
                raise ValueError(msg)
            if self.better is not BetterArm.UNKNOWN:
                msg = "an incomparable pair cannot have a winner"
                raise ValueError(msg)
            if not self.reason:
                msg = "an incomparable pair must state why"
                raise ValueError(msg)
        return self


class SimulationRun(DomainModel):
    """The complete, structured answer to one simulation request."""

    run_id: Annotated[str, Field(min_length=1, max_length=64)]
    status: SimulationStatus
    scenario: SimulationScenario

    routes: tuple[RouteSimulationResult, ...] = ()
    links: tuple[LinkResult, ...] = ()
    network: NetworkSimulationSummary | None = None
    comparison: SimulationComparison | None = None

    rejections: tuple[SimulationRejection, ...] = ()
    warnings: tuple[str, ...] = ()

    model_name: Annotated[str, Field(min_length=1, max_length=80)] = MODEL_NAME
    model_version: Annotated[str, Field(min_length=1, max_length=40)] = MODEL_VERSION
    network_id: Annotated[str, Field(min_length=1, max_length=64)]
    network_version: Annotated[str, Field(min_length=1, max_length=50)]

    computed_at: dt.datetime
    provenance: Provenance

    @model_validator(mode="after")
    def _status_matches_content(self) -> SimulationRun:
        if self.computed_at.tzinfo is None:
            msg = "computed_at must be timezone-aware"
            raise ValueError(msg)
        if self.status is SimulationStatus.UNAVAILABLE:
            if self.routes:
                msg = (
                    "status is unavailable but route results are present; a refusal "
                    "that still carries an outcome is what this status prevents"
                )
                raise ValueError(msg)
            if not self.rejections:
                msg = "an unavailable run must state why"
                raise ValueError(msg)
        elif self.network is None:
            msg = "a completed run must carry its network summary"
            raise ValueError(msg)
        return self

    def arm(self, role: ArmRole) -> RouteSimulationResult | None:
        """The result for one role, or ``None`` when that arm was not run."""
        return next((result for result in self.routes if result.role is role), None)
