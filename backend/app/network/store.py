"""The loaded road network, held for the lifetime of the process.

One network at a time, deliberately. Serving several would mean every incident,
every emergency and every route request had to name which map it referred to,
and a request that forgot would silently pick one. A single loaded network makes
"which map produced this decision?" answerable by the run manifest alone.

The index is built once at load rather than per request. Route relaxation runs
in the inner loop of every emergency request, so successor lookup must be O(1);
:class:`~app.domain.network.NetworkIndex` is a derived acceleration structure
with a different lifetime from the immutable value it accelerates, which is why
it is held here rather than memoised onto the frozen model.
"""

from __future__ import annotations

from app.core.errors import NetworkNotLoadedError
from app.domain.network import NetworkIndex, RoadNetwork
from app.network.loader import NetworkSource


class NetworkStore:
    """Holds the currently loaded network, its index, and its provenance."""

    __slots__ = ("_index", "_network", "_source")

    def __init__(self) -> None:
        self._network: RoadNetwork | None = None
        self._index: NetworkIndex | None = None
        self._source: NetworkSource | None = None

    def load(self, network: RoadNetwork, source: NetworkSource) -> None:
        """Replace the loaded network, rebuilding the index."""
        self._network = network
        self._index = network.build_index()
        self._source = source

    def clear(self) -> None:
        """Unload the current network."""
        self._network = None
        self._index = None
        self._source = None

    @property
    def is_loaded(self) -> bool:
        """Whether a network is currently loaded."""
        return self._network is not None

    @property
    def source(self) -> NetworkSource | None:
        """Provenance of the loaded network, or ``None`` when none is loaded."""
        return self._source

    @property
    def network(self) -> RoadNetwork | None:
        """The loaded network, or ``None``."""
        return self._network

    def require(self) -> RoadNetwork:
        """Return the loaded network, or explain how to load one.

        A 404 rather than an empty network. An empty ``RoadNetwork`` cannot even
        be constructed without inventing a ``network_id`` and a version, and a
        caller receiving one would have no way to tell "no map is loaded" from
        "the map is genuinely empty".
        """
        if self._network is None:
            msg = (
                "No road network is loaded. Load one with POST /api/network/load, "
                "naming a file relative to the networks directory, or set "
                "NETWORK_FILE to load it at startup."
            )
            raise NetworkNotLoadedError(msg)
        return self._network

    def require_index(self) -> NetworkIndex:
        """Return the index for the loaded network, or raise as :meth:`require`."""
        index = self._index
        if index is None:
            self.require()
            msg = "network is loaded but its index is missing"
            raise RuntimeError(msg)
        return index
