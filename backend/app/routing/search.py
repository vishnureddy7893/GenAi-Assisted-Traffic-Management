"""Deterministic shortest-path search over the admissible subgraph.

Why Dijkstra
------------
Every edge weight here is a non-negative number of equivalent seconds, which is
exactly Dijkstra's precondition, and the network is small enough that its cost
is irrelevant. A* was considered and rejected: an admissible heuristic needs a
lower bound on remaining cost, which needs both surveyed coordinates on every
junction *and* a defensible maximum speed. Junction positions are optional in
the schema, and a maximum speed would be an invented number of the kind this
system refuses everywhere else. A heuristic that is not admissible does not
merely slow the search down -- it silently returns the wrong route, which is the
worst possible failure for an emergency recommendation.

Alternatives come from Yen's algorithm, which is Dijkstra run repeatedly with
edges removed. Loopless by construction, deterministic, and it reuses the same
cost function, so an alternative is priced identically to the winner.

Determinism is a requirement, not a nicety
------------------------------------------
The same inputs must produce the same route on every machine and every run, or a
recommendation cannot be replayed against the state that produced it. Ties are
therefore broken explicitly -- by cost, then by the path's segment ids -- rather
than by whatever order a heap or a dictionary happened to yield.

The graph is movement-expanded
------------------------------
A search node is ``(junction, arriving segment)``, not just ``(junction)``.
Turn feasibility and turn cost both depend on how you arrived, and a node-level
graph cannot express either: it would happily route a fire engine through a
banned right turn because the junction, considered alone, is fine.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable, Iterable

from app.domain.network import NetworkIndex
from app.routing.pruning import AdmissibleSubgraph

ALGORITHM = "dijkstra+yen-k-shortest"
ALGORITHM_VERSION = "1.0"

#: A search node: the junction reached, and the segment used to reach it.
#: The empty string marks the origin, which was not reached by traversing
#: anything.
SearchNode = tuple[str, str]

#: A path is the ordered segment ids traversed. The junction sequence follows
#: from the origin plus these, so storing both would risk them disagreeing.
Path = tuple[str, ...]

#: Cost of entering ``segment_id`` at ``node``. Must be non-negative and finite;
#: ``None`` means the transition is unavailable.
EdgeCost = Callable[[SearchNode, str], float | None]


class SearchLimitReachedError(Exception):
    """The node budget was exhausted before the search completed."""


def _successors(
    node: SearchNode,
    index: NetworkIndex,
    subgraph: AdmissibleSubgraph,
    banned_transitions: frozenset[tuple[SearchNode, str]],
    banned_junctions: frozenset[str],
) -> Iterable[tuple[SearchNode, str]]:
    """Admissible transitions out of a node, in a deterministic order."""
    junction_id, arriving = node
    for segment in sorted(
        index.outgoing_segments(junction_id), key=lambda s: s.segment_id
    ):
        if not subgraph.permits_segment(segment.segment_id):
            continue
        if segment.to_junction_id in banned_junctions:
            continue
        if arriving and not subgraph.permits_movement(
            junction_id, arriving, segment.segment_id
        ):
            continue
        if (node, segment.segment_id) in banned_transitions:
            continue
        yield (segment.to_junction_id, segment.segment_id), segment.segment_id


def shortest_path(
    index: NetworkIndex,
    subgraph: AdmissibleSubgraph,
    *,
    origin_junction_id: str,
    destination_junction_id: str,
    edge_cost: EdgeCost,
    max_settled: int,
    banned_transitions: frozenset[tuple[SearchNode, str]] = frozenset(),
    banned_junctions: frozenset[str] = frozenset(),
    start_node: SearchNode | None = None,
) -> tuple[Path, float] | None:
    """Cheapest admissible path, or ``None`` when none exists.

    Ties are broken by the path's segment ids so that two equally-priced routes
    always resolve the same way. Without that, an arbitrary heap ordering would
    make the recommendation depend on dictionary iteration.
    """
    start: SearchNode = start_node or (origin_junction_id, "")
    if start[0] == destination_junction_id and start_node is None:
        return (), 0.0

    # Heap entries are (cost, tie-break path, node, path). The path in the
    # tie-break slot makes the ordering total and reproducible.
    heap: list[tuple[float, Path, SearchNode, Path]] = [(0.0, (), start, ())]
    best: dict[SearchNode, float] = {start: 0.0}
    settled = 0

    while heap:
        cost, _, node, path = heapq.heappop(heap)
        if cost > best.get(node, float("inf")):
            continue
        if node[0] == destination_junction_id and path:
            return path, cost

        settled += 1
        if settled > max_settled:
            raise SearchLimitReachedError

        for successor, segment_id in _successors(
            node, index, subgraph, banned_transitions, banned_junctions
        ):
            weight = edge_cost(node, segment_id)
            if weight is None:
                continue
            new_cost = cost + weight
            if new_cost < best.get(successor, float("inf")) - 1e-12:
                best[successor] = new_cost
                new_path = (*path, segment_id)
                heapq.heappush(heap, (new_cost, new_path, successor, new_path))

    return None


def _nodes_along(origin_junction_id: str, path: Path, index: NetworkIndex) -> list[SearchNode]:
    """Reconstruct the search nodes a path passes through."""
    nodes: list[SearchNode] = [(origin_junction_id, "")]
    for segment_id in path:
        segment = index.segment(segment_id)
        nodes.append((segment.to_junction_id, segment_id))
    return nodes


def k_shortest_paths(
    index: NetworkIndex,
    subgraph: AdmissibleSubgraph,
    *,
    origin_junction_id: str,
    destination_junction_id: str,
    edge_cost: EdgeCost,
    k: int,
    max_settled: int,
) -> list[tuple[Path, float]]:
    """Up to ``k`` cheapest loopless paths, best first (Yen's algorithm).

    Alternatives exist so an operator can ask "why not the other way?" and get
    an answer with costs attached. They are priced by the same function as the
    winner, so the comparison is like for like.
    """
    first = shortest_path(
        index,
        subgraph,
        origin_junction_id=origin_junction_id,
        destination_junction_id=destination_junction_id,
        edge_cost=edge_cost,
        max_settled=max_settled,
    )
    if first is None:
        return []

    accepted: list[tuple[Path, float]] = [first]
    if k <= 1:
        return accepted

    # Sorted by (cost, path) so extraction is deterministic under ties.
    candidates: list[tuple[float, Path]] = []

    while len(accepted) < k:
        previous_path = accepted[-1][0]
        previous_nodes = _nodes_along(origin_junction_id, previous_path, index)

        for i in range(len(previous_path)):
            spur_node = previous_nodes[i]
            root_path = previous_path[:i]

            banned_transitions = {
                (_nodes_along(origin_junction_id, accepted_path, index)[i], accepted_path[i])
                for accepted_path, _ in accepted
                if len(accepted_path) > i and accepted_path[:i] == root_path
            }
            banned_junctions = frozenset(
                node[0] for node in previous_nodes[:i] if node[0] != spur_node[0]
            )

            root_cost = 0.0
            valid_root = True
            for index_in_root, segment_id in enumerate(root_path):
                weight = edge_cost(previous_nodes[index_in_root], segment_id)
                if weight is None:
                    valid_root = False
                    break
                root_cost += weight
            if not valid_root:
                continue

            spur = shortest_path(
                index,
                subgraph,
                origin_junction_id=spur_node[0],
                destination_junction_id=destination_junction_id,
                edge_cost=edge_cost,
                max_settled=max_settled,
                banned_transitions=frozenset(banned_transitions),
                banned_junctions=banned_junctions,
                start_node=spur_node,
            )
            if spur is None:
                continue

            total_path = root_path + spur[0]
            total_cost = root_cost + spur[1]
            if any(total_path == existing for existing, _ in accepted):
                continue
            entry = (round(total_cost, 6), total_path)
            if entry not in candidates:
                candidates.append(entry)

        if not candidates:
            break

        candidates.sort()
        best_cost, best_path = candidates.pop(0)
        accepted.append((best_path, best_cost))

    return accepted
