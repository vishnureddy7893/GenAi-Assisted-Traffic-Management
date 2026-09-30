"""The traffic model itself: a deterministic point-queue link simulation.

Why this model
--------------
The question Phase 8 exists to answer is narrow and specific: *does the
recommended route actually beat the baseline, given the traffic we believe is
there?* The model was chosen to answer that question with the data this system
genuinely has, and no more.

A **point queue** (or vertical queue) link model gives each directed segment two
properties: a free-flow traversal time, and a discharge capacity at its exit.
Vehicles enter, spend at least the free-flow time running, then queue at the
exit if the demand ahead of them exceeds what the exit can pass. It is the
standard mesoscopic formulation used in dynamic traffic assignment, and it needs
exactly two inputs per link -- both of which are either surveyed facts about the
road or explicit, stated assumptions.

What was rejected, and why it matters
-------------------------------------
*Microscopic car-following* (the SUMO/IDM family) needs per-vehicle
acceleration, desired headway, reaction time and driver heterogeneity. Nothing
in this project measures any of them. Every one would be a number invented to
make the simulation look detailed, and the detail would be fiction with a
confidence interval attached.

*The cell transmission model* needs a jam density and a backward wave speed per
link to propagate congestion upstream. Neither is surveyed here. CTM's advantage
-- spillback into the upstream link -- is real, and its absence is the honest
limitation recorded in the results: a queue here is vertical, so it occupies no
road space and cannot block the segment behind it.

*A BPR-style volume-delay function* would produce a travel time from a
volume-to-capacity ratio in one step, but it is a steady-state equilibrium curve
with fitted exponents. Applying it to a 15-minute emergency horizon would answer
a different question than the one asked, with parameters nobody here has
calibrated.

Determinism
-----------
There is no random number generator in this module, no sampling and no seed.
Every quantity is a deterministic function of the inputs, evaluated in a fixed
order, so two runs of identical inputs produce identical output -- on any
machine, in any process. That is a requirement rather than a nicety: a simulated
comparison that cannot be reproduced cannot be reviewed.

Nothing in this module knows about HTTP, pydantic, cameras or language models.
It takes numbers and returns numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Recorded values are rounded to this many decimals. Far below any meaningful
#: traffic quantity, and it keeps accumulated floating-point noise out of the
#: byte-for-byte comparison that the determinism tests make.
ROUNDING = 9


@dataclass(frozen=True, slots=True)
class Link:
    """One directed segment as the model sees it.

    Every field is required. There is deliberately no constructor default for
    demand or capacity: a link the caller cannot supply them for is not built at
    all, and is reported as unmodelled rather than simulated against a guess.
    """

    segment_id: str
    free_flow_traversal_s: float
    capacity_veh_per_hour: float
    demand_veh_per_hour: float
    initial_queue_veh: float

    @property
    def capacity_veh_per_s(self) -> float:
        """Discharge capacity in vehicles per second."""
        return self.capacity_veh_per_hour / 3600.0

    @property
    def saturation_ratio(self) -> float:
        """Demand over capacity. Above 1.0 the queue grows while demand holds."""
        return self.demand_veh_per_hour / self.capacity_veh_per_hour


@dataclass(frozen=True, slots=True)
class LinkTrace:
    """The evolution of one link over the horizon, and its summary."""

    segment_id: str
    step_s: float
    capacity_veh_per_s: float
    initial_queue_veh: float
    queue_veh: tuple[float, ...]
    """Queue at the *end* of each step."""

    discharged_veh: tuple[float, ...]
    """Vehicles passing the exit during each step."""

    entered_veh: float
    """Total demand released onto the link over the horizon."""

    arrived_at_exit_veh: float
    """Of those, how many finished running and reached the exit in time."""

    in_transit_veh: float
    """Still running when the horizon ended. Reported rather than folded into
    the queue, because a vehicle still moving is not a vehicle waiting."""

    throughput_veh: float
    peak_queue_veh: float
    final_queue_veh: float
    mean_queue_veh: float
    total_delay_veh_s: float
    """Vehicle-seconds spent waiting in the exit queue: the queue length summed
    over steps and multiplied by the step. The standard rectangular
    approximation, and exact in the limit of a small step."""

    saturation_ratio: float

    @property
    def mean_delay_per_vehicle_s(self) -> float | None:
        """Waiting time per vehicle that got through, or ``None`` if none did."""
        if self.throughput_veh <= 0.0:
            return None
        return round(self.total_delay_veh_s / self.throughput_veh, ROUNDING)

    def queue_at(self, time_s: float) -> float:
        """Queue standing at the exit at ``time_s``.

        The most recently recorded state at or before that instant: a vehicle
        arriving mid-step joins the queue as it was at the end of the previous
        step, because that is the last state the model actually computed.
        Interpolating between steps would invent a resolution the model does not
        have.
        """
        if time_s <= 0.0 or not self.queue_veh:
            return self.initial_queue_veh
        index = math.floor(time_s / self.step_s)
        if index <= 0:
            return self.initial_queue_veh
        return self.queue_veh[min(index, len(self.queue_veh)) - 1]

    def discharge_delay_at(self, time_s: float) -> float:
        """Seconds a vehicle arriving at ``time_s`` waits for the queue ahead.

        The queue standing at the exit, divided by the rate the exit discharges
        at. This is the delay term the whole model exists to produce, and it is
        the only place a simulated route differs from the optimizer's free-flow
        estimate.
        """
        return round(self.queue_at(time_s) / self.capacity_veh_per_s, ROUNDING)


def simulate_link(link: Link, *, step_s: float, step_count: int) -> LinkTrace:
    """Evolve one link's exit queue over the horizon.

    Links are evolved independently. Vehicles discharged from a link leave the
    modelled area rather than moving onto a successor, because assigning them to
    successors would need an origin-destination matrix, and no survey in this
    project produced one. Inventing one would change every number here into a
    consequence of a fabricated travel pattern.
    """
    if step_count < 1:
        msg = "a simulation needs at least one step"
        raise ValueError(msg)
    if step_s <= 0.0:
        msg = "step_s must be positive"
        raise ValueError(msg)

    inflow_per_step = round(link.demand_veh_per_hour / 3600.0 * step_s, ROUNDING)
    capacity_per_step = round(link.capacity_veh_per_s * step_s, ROUNDING)

    # A vehicle released at step i cannot reach the exit before it has run the
    # free-flow traversal time. Rounded up to whole steps: the model's time
    # resolution is the step, and rounding down would let a vehicle arrive
    # before it physically could.
    lag_steps = max(1, math.ceil(link.free_flow_traversal_s / step_s))

    queue = link.initial_queue_veh
    queue_series: list[float] = []
    discharged_series: list[float] = []
    delay_veh_s = 0.0
    arrived = 0.0

    for index in range(step_count):
        arrivals = inflow_per_step if index >= lag_steps else 0.0
        arrived += arrivals
        queue = round(queue + arrivals, ROUNDING)
        discharge = min(queue, capacity_per_step)
        queue = round(queue - discharge, ROUNDING)
        queue_series.append(queue)
        discharged_series.append(round(discharge, ROUNDING))
        delay_veh_s += queue * step_s

    entered = round(inflow_per_step * step_count, ROUNDING)
    arrived = round(arrived, ROUNDING)
    throughput = round(sum(discharged_series), ROUNDING)

    return LinkTrace(
        segment_id=link.segment_id,
        step_s=step_s,
        capacity_veh_per_s=link.capacity_veh_per_s,
        initial_queue_veh=link.initial_queue_veh,
        queue_veh=tuple(queue_series),
        discharged_veh=tuple(discharged_series),
        entered_veh=entered,
        arrived_at_exit_veh=arrived,
        in_transit_veh=round(entered - arrived, ROUNDING),
        throughput_veh=throughput,
        peak_queue_veh=round(max([link.initial_queue_veh, *queue_series]), ROUNDING),
        final_queue_veh=queue_series[-1],
        mean_queue_veh=round(sum(queue_series) / step_count, ROUNDING),
        total_delay_veh_s=round(delay_veh_s, ROUNDING),
        saturation_ratio=round(link.saturation_ratio, ROUNDING),
    )


def simulate_network(
    links: dict[str, Link], *, step_s: float, step_count: int
) -> dict[str, LinkTrace]:
    """Evolve every modellable link. Iteration order is sorted, so the result is
    identical regardless of how the caller built the mapping."""
    return {
        segment_id: simulate_link(links[segment_id], step_s=step_s, step_count=step_count)
        for segment_id in sorted(links)
    }


@dataclass(frozen=True, slots=True)
class ProbeLeg:
    """One segment the evaluated vehicle traverses."""

    segment_id: str
    free_flow_traversal_s: float


@dataclass(frozen=True, slots=True)
class ProbeLegOutcome:
    """When the vehicle entered and left one leg, and what it waited for."""

    segment_id: str
    entry_time_s: float
    exit_time_s: float
    free_flow_traversal_s: float
    queue_delay_s: float | None
    """``None`` where this link was not modelled. Not zero: zero is a modelled
    result meaning the exit was clear."""

    queue_at_arrival_veh: float | None
    modelled: bool


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """The evaluated vehicle's journey through the simulated network."""

    legs: tuple[ProbeLegOutcome, ...]
    completed_within_horizon: bool
    completion_time_s: float | None
    travel_time_s: float | None
    free_flow_travel_time_s: float
    queue_delay_s: float
    """Total waiting across the modelled legs. A lower bound on the true delay
    whenever any leg was unmodelled -- those are named by the caller."""

    modelled_leg_count: int


def simulate_probe(
    legs: tuple[ProbeLeg, ...],
    traces: dict[str, LinkTrace],
    *,
    horizon_s: float,
    departure_offset_s: float = 0.0,
    queue_bypass_fraction: float = 0.0,
) -> ProbeResult:
    """Drive one vehicle along a route through the already-evolved network.

    The vehicle is a *probe*: it experiences the queues but does not add to
    them, and it does not consume exit capacity. One extra vehicle changes a
    link's queue by a fraction of a percent at any realistic demand, and
    modelling the feedback would require assumptions about where it came from
    that nothing here has.

    ``queue_bypass_fraction`` is the only emergency-specific term in the model,
    and it defaults to zero everywhere it is configured. A non-zero value asserts
    that a responder passes a stationary queue -- true on some geometry, false on
    others, and measured nowhere in this project. When it is zero the simulated
    emergency journey is the conservative one, which is the right direction for
    an estimate an operator will act on.
    """
    if not 0.0 <= queue_bypass_fraction <= 1.0:
        msg = "queue_bypass_fraction must lie in [0, 1]"
        raise ValueError(msg)

    clock = departure_offset_s
    outcomes: list[ProbeLegOutcome] = []
    free_flow_total = 0.0
    delay_total = 0.0
    modelled = 0

    for leg in legs:
        entry = clock
        exit_free_flow = round(entry + leg.free_flow_traversal_s, ROUNDING)
        free_flow_total += leg.free_flow_traversal_s

        trace = traces.get(leg.segment_id)
        if trace is None:
            outcomes.append(
                ProbeLegOutcome(
                    segment_id=leg.segment_id,
                    entry_time_s=entry,
                    exit_time_s=exit_free_flow,
                    free_flow_traversal_s=leg.free_flow_traversal_s,
                    queue_delay_s=None,
                    queue_at_arrival_veh=None,
                    modelled=False,
                )
            )
            clock = exit_free_flow
            continue

        queue = trace.queue_at(exit_free_flow)
        delay = round(
            trace.discharge_delay_at(exit_free_flow) * (1.0 - queue_bypass_fraction),
            ROUNDING,
        )
        exit_time = round(exit_free_flow + delay, ROUNDING)
        outcomes.append(
            ProbeLegOutcome(
                segment_id=leg.segment_id,
                entry_time_s=entry,
                exit_time_s=exit_time,
                free_flow_traversal_s=leg.free_flow_traversal_s,
                queue_delay_s=delay,
                queue_at_arrival_veh=queue,
                modelled=True,
            )
        )
        delay_total += delay
        modelled += 1
        clock = exit_time

    completed = clock <= horizon_s
    return ProbeResult(
        legs=tuple(outcomes),
        completed_within_horizon=completed,
        completion_time_s=round(clock, ROUNDING) if completed else None,
        travel_time_s=round(clock - departure_offset_s, ROUNDING) if completed else None,
        free_flow_travel_time_s=round(free_flow_total, ROUNDING),
        queue_delay_s=round(delay_total, ROUNDING),
        modelled_leg_count=modelled,
    )
