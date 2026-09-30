"""Controlled traffic simulation: a scenario in, a modelled outcome out.

    RoadNetwork + NetworkState + incidents   ->  link inputs, with their bases
    observations bound by the survey         ->  observed demand and queues
    stated assumptions from the request      ->  everything the data lacks
    route optimizer candidates               ->  the arms to compare
                                             ->  SimulationRun

The model is a deterministic point-queue link simulation. Each directed segment
runs vehicles at a free-flow traversal time and discharges them at a stated exit
capacity; where demand exceeds capacity a queue forms, and the evaluated vehicle
waits for it. There is no random component and no seed: identical inputs produce
identical output, which is what makes a comparison reviewable.

What this layer refuses to do
-----------------------------
It does not detect, track, classify, optimise, approve, actuate or narrate. It
consumes what earlier phases produced and adds one thing: what would happen,
under stated conditions, to a vehicle taking this route rather than that one.

It also refuses to fill a gap. No default demand, no default capacity, no
default speed. A link without them is reported unmodelled with the reason, and a
route whose vehicle did not arrive within the horizon reports no travel time
rather than a truncated one. Every figure it does produce carries ``SIMULATED``
provenance, so it can never be stated as a measurement of a real road.
"""

from app.simulation.model import (
    Link,
    LinkTrace,
    ProbeLeg,
    ProbeLegOutcome,
    ProbeResult,
    simulate_link,
    simulate_network,
    simulate_probe,
)
from app.simulation.runner import compare_arms, run, unavailable_run
from app.simulation.scenario import (
    ObservedSegmentInput,
    StatedSegmentInput,
    build_links,
    build_scenario,
    modellable_links,
    observed_inputs,
)
from app.simulation.schemas import (
    MODEL_NAME,
    MODEL_VERSION,
    ArmRole,
    ArmStatus,
    BaselineStrategy,
    BetterArm,
    InputBasis,
    LegResult,
    LinkInputs,
    LinkResult,
    LinkStatus,
    NetworkSimulationSummary,
    RouteArmInput,
    RouteSimulationResult,
    RouteSource,
    SimulationComparison,
    SimulationParameters,
    SimulationRejection,
    SimulationRun,
    SimulationScenario,
    SimulationStatus,
    UnavailableReason,
)
from app.simulation.store import SimulationStore

__all__ = [
    "MODEL_NAME",
    "MODEL_VERSION",
    "ArmRole",
    "ArmStatus",
    "BaselineStrategy",
    "BetterArm",
    "InputBasis",
    "LegResult",
    "Link",
    "LinkInputs",
    "LinkResult",
    "LinkStatus",
    "LinkTrace",
    "NetworkSimulationSummary",
    "ObservedSegmentInput",
    "ProbeLeg",
    "ProbeLegOutcome",
    "ProbeResult",
    "RouteArmInput",
    "RouteSimulationResult",
    "RouteSource",
    "SimulationComparison",
    "SimulationParameters",
    "SimulationRejection",
    "SimulationRun",
    "SimulationScenario",
    "SimulationStatus",
    "SimulationStore",
    "StatedSegmentInput",
    "UnavailableReason",
    "build_links",
    "build_scenario",
    "compare_arms",
    "modellable_links",
    "observed_inputs",
    "run",
    "simulate_link",
    "simulate_network",
    "simulate_probe",
    "unavailable_run",
]
