"""Image-space geometry for traffic analytics.

Pure functions, no dependencies beyond the standard library, so every formula
here is directly testable and exactly reproducible.

Two conventions used throughout, both easy to get wrong:

* **Ground contact, not centroid.** Motion is measured from the bottom-centre of
  a box. A centroid sits at roughly mid-height, so it moves whenever the object's
  apparent size changes -- a vehicle approaching the camera appears to move
  upward in the frame even while its wheels travel straight. Bottom-centre is
  the closest thing to where the object touches the road.

* **Image y increases downward.** Every angle here is measured clockwise from
  image-right so that the arithmetic matches what a reader sees on screen. Using
  a mathematical convention would put "up" at +90 degrees while the pixels say
  otherwise.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from app.analytics.schemas import DirectionLabel

#: Direction sectors, each 45 degrees wide, clockwise from image-right.
_DIRECTION_SECTORS: tuple[DirectionLabel, ...] = (
    DirectionLabel.RIGHT,
    DirectionLabel.DOWN_RIGHT,
    DirectionLabel.DOWN,
    DirectionLabel.DOWN_LEFT,
    DirectionLabel.LEFT,
    DirectionLabel.UP_LEFT,
    DirectionLabel.UP,
    DirectionLabel.UP_RIGHT,
)

#: Resolution of the occupancy mask. 256x256 keeps a union-area estimate within
#: a fraction of a percent for traffic-sized boxes while staying trivially cheap
#: and, unlike a sampling approach, exactly deterministic.
OCCUPANCY_GRID = 256


def frame_diagonal(width: float, height: float) -> float:
    """Diagonal length of a frame, the normaliser for image-space speeds."""
    return math.hypot(width, height)


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance between two image points."""
    return math.hypot(b[0] - a[0], b[1] - a[1])


def direction_degrees(start: tuple[float, float], end: tuple[float, float]) -> float | None:
    """Heading from ``start`` to ``end``, clockwise from image-right, in [0, 360).

    Returns ``None`` for zero displacement: a stationary object has no
    direction, and returning 0 would claim it is moving rightward.
    """
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if math.isclose(dx, 0.0, abs_tol=1e-9) and math.isclose(dy, 0.0, abs_tol=1e-9):
        return None
    # atan2(dy, dx) with screen y-down already yields clockwise-from-right.
    return math.degrees(math.atan2(dy, dx)) % 360.0


def direction_label(degrees: float) -> DirectionLabel:
    """Bucket a heading into one of eight image-space sectors."""
    # Offset by half a sector so each label is centred on its nominal heading.
    index = int(((degrees + 22.5) % 360.0) // 45.0)
    return _DIRECTION_SECTORS[index]


def path_length(points: Sequence[tuple[float, float]]) -> float:
    """Total distance travelled along a polyline."""
    return sum(distance(points[i], points[i + 1]) for i in range(len(points) - 1))


def net_displacement(points: Sequence[tuple[float, float]]) -> float:
    """Straight-line distance from first to last point."""
    if len(points) < 2:
        return 0.0
    return distance(points[0], points[-1])


def union_area_fraction(
    boxes: Sequence[tuple[float, float, float, float]],
    width: float,
    height: float,
    *,
    grid: int = OCCUPANCY_GRID,
) -> float:
    """Fraction of the frame covered by boxes, counting overlaps once.

    Summing box areas would double-count every overlap and can exceed 100% in
    dense traffic, which is exactly when the number matters most. Rasterising to
    a fixed grid gives a union that is bounded, deterministic and cheap.
    """
    if width <= 0 or height <= 0 or not boxes:
        return 0.0

    covered = bytearray(grid * grid)
    for x1, y1, x2, y2 in boxes:
        # Clamp to the frame: a detector may return a box that overhangs an edge.
        col0 = max(0, min(grid - 1, int(x1 / width * grid)))
        col1 = max(0, min(grid - 1, math.ceil(x2 / width * grid) - 1))
        row0 = max(0, min(grid - 1, int(y1 / height * grid)))
        row1 = max(0, min(grid - 1, math.ceil(y2 / height * grid) - 1))
        for row in range(row0, row1 + 1):
            base = row * grid
            for col in range(col0, col1 + 1):
                covered[base + col] = 1

    return sum(covered) / (grid * grid)


def grid_counts(
    points: Sequence[tuple[float, float]],
    width: float,
    height: float,
    *,
    rows: int = 3,
    cols: int = 3,
) -> tuple[tuple[int, ...], ...]:
    """Count points per grid cell, row-major.

    A coarse read of where activity concentrates. It says nothing about lanes or
    approaches -- assigning objects to those needs a zone configuration mapping
    image regions onto real road elements.
    """
    counts = [[0] * cols for _ in range(rows)]
    if width <= 0 or height <= 0:
        return tuple(tuple(row) for row in counts)

    for x, y in points:
        col = max(0, min(cols - 1, int(x / width * cols)))
        row = max(0, min(rows - 1, int(y / height * rows)))
        counts[row][col] += 1
    return tuple(tuple(row) for row in counts)


def standard_deviation(values: Sequence[float]) -> float | None:
    """Population standard deviation, or ``None`` for fewer than two values."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def cluster_by_proximity(
    items: Sequence[tuple[int, tuple[float, float], float]],
    *,
    proximity_factor: float,
) -> tuple[tuple[int, ...], ...]:
    """Single-linkage clustering of objects by centre distance.

    Parameters
    ----------
    items:
        ``(identifier, centre_point, box_diagonal)`` triples.
    proximity_factor:
        Two objects join the same cluster when their centres are closer than
        this multiple of their *mean* box diagonal.

    Scaling the threshold by apparent size is what makes this usable on a
    perspective view: vehicles near the horizon are a few pixels apart while the
    same real spacing near the camera is hundreds of pixels, so a fixed pixel
    gap would cluster the whole far half of the frame into one queue.

    Single linkage is the right shape for this problem -- a queue is a chain of
    adjacent vehicles, not a compact blob, so transitive linking is the desired
    behaviour rather than a weakness.
    """
    if not items:
        return ()

    # Sort for determinism: identical input must always yield identical groups.
    ordered = sorted(items, key=lambda item: (item[1][0], item[1][1], item[0]))
    parent = list(range(len(ordered)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            _, centre_i, diag_i = ordered[i]
            _, centre_j, diag_j = ordered[j]
            mean_diagonal = (diag_i + diag_j) / 2.0
            if mean_diagonal <= 0:
                continue
            if distance(centre_i, centre_j) <= proximity_factor * mean_diagonal:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for index in range(len(ordered)):
        groups.setdefault(find(index), []).append(ordered[index][0])

    return tuple(
        tuple(sorted(members))
        for _root, members in sorted(groups.items(), key=lambda kv: sorted(kv[1])[0])
    )


def cluster_extent(points: Sequence[tuple[float, float]]) -> float:
    """Longest distance between any two points in a cluster.

    The cluster's span, not its bounding-box diagonal: a queue curving around a
    corner has a span longer than any axis-aligned box would suggest.
    """
    if len(points) < 2:
        return 0.0
    return max(
        distance(points[i], points[j])
        for i in range(len(points))
        for j in range(i + 1, len(points))
    )
