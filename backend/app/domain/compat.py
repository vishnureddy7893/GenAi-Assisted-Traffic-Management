"""Deprecated compatibility alias for ``TrafficState``.

The specification allows ``TrafficState`` to survive as an alias for
:class:`~app.domain.observation.JunctionObservation`. It is provided here, and
deliberately quarantined in a module of its own, for one reason.

In a greenfield repository there is nothing to be backwards compatible *with*.
Introducing a second name for a concept on day one guarantees that six months
from now half the codebase says ``TrafficState`` and half says
``JunctionObservation``, that a reader has to check whether they are the same
thing, and that the name which lost the distinction -- "the state of traffic",
ambiguous between one camera's view and the whole network -- is the one that
survives in conversation. That ambiguity is the exact confusion the three-way
split into observation, network state and emergency state exists to remove.

So: the alias exists for external callers and for continuity with the original
specification. Nothing inside ``app`` imports it, and a test enforces that.
Accessing it emits a :class:`DeprecationWarning` naming the replacement.
"""

from __future__ import annotations

import warnings
from typing import Any

from app.domain.observation import JunctionObservation

# Resolved lazily by ``__getattr__`` below so that access emits a deprecation
# warning; there is deliberately no module-level binding to import silently.
__all__ = ["TrafficState"]  # noqa: F822

_REPLACEMENT = (
    "TrafficState is deprecated; use app.domain.observation.JunctionObservation "
    "for a single source's view of one junction, or app.domain.state.NetworkState "
    "for the fused network-wide snapshot. The two are not interchangeable, which "
    "is precisely why the single name was retired."
)


def __getattr__(name: str) -> Any:
    """Emit a deprecation warning when the alias is accessed."""
    if name == "TrafficState":
        warnings.warn(_REPLACEMENT, DeprecationWarning, stacklevel=2)
        return JunctionObservation
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
