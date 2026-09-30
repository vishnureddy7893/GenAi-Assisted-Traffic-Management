"""Retention of completed simulation runs, for retrieval after the fact.

Why runs are addressable at all
-------------------------------
A comparison an operator acted on has to remain retrievable afterwards, or the
"why did we send them that way?" question has no answer. The identifier is the
scenario fingerprint rather than a random string, so re-requesting an identical
scenario addresses the same run instead of creating a second one that must then
be reconciled with the first.

Why there is no pending state
-----------------------------
The model is deterministic and bounded by the configured horizon, so a run
finishes inside the request that asked for it. There is no queue, no worker and
no progress to poll: a run's status is terminal from the moment it exists.
Modelling it as asynchronous would add a state the system never actually
occupies, and a client polling for a completion that already happened.

In-memory and per-process, like the analytics and traffic-state caches. A run is
cheap to reproduce -- that is the point of determinism -- so pinning it to a
database before the schema has settled would be premature.
"""

from __future__ import annotations

from collections import OrderedDict

from app.core.errors import ResourceNotFoundError
from app.simulation.schemas import SimulationRun


class SimulationStore:
    """Bounded, insertion-ordered store of completed runs."""

    __slots__ = ("_max_runs", "_runs")

    def __init__(self, max_runs: int = 50) -> None:
        if max_runs < 1:
            msg = "max_runs must be at least 1"
            raise ValueError(msg)
        self._runs: OrderedDict[str, SimulationRun] = OrderedDict()
        self._max_runs = max_runs

    def put(self, run: SimulationRun) -> SimulationRun:
        """Store a run, replacing any earlier run with the same identity.

        Replacement rather than duplication is deliberate: two runs sharing an
        id were given identical inputs, so keeping both would be keeping the
        same answer twice under different timestamps.
        """
        self._runs.pop(run.run_id, None)
        self._runs[run.run_id] = run
        while len(self._runs) > self._max_runs:
            self._runs.popitem(last=False)
        return run

    def get(self, run_id: str) -> SimulationRun:
        """Return one run, or explain that it was never stored or has aged out."""
        run = self._runs.get(run_id)
        if run is None:
            msg = (
                f"No simulation run {run_id!r} is held by this process. Runs are "
                "kept in memory only, so an id from an earlier process or one past "
                "the retention limit is gone; re-running the same scenario "
                "reproduces it under the same id."
            )
            raise ResourceNotFoundError(msg, run_id=run_id)
        return run

    def list(self) -> tuple[SimulationRun, ...]:
        """Every retained run, newest first."""
        return tuple(reversed(self._runs.values()))

    def clear(self) -> None:
        """Drop every retained run."""
        self._runs.clear()

    @property
    def count(self) -> int:
        """How many runs are retained."""
        return len(self._runs)

    @property
    def max_runs(self) -> int:
        """Retention ceiling in force."""
        return self._max_runs
