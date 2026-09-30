"""Traffic data ingestion.

Mode A (recorded) and Mode B (live) are genuinely different kinds of input --
bounded vs unbounded, replayable vs not, virtual clock vs wall clock -- and this
package says so explicitly through SourceCapabilities rather than pretending
they are interchangeable. What converges is the output: both emit SourceSample
values on a clock they supply, and the entire decision pipeline downstream
depends only on those two things.
"""

from app.ingestion.base import (
    LIVE_KINDS,
    RECORDED_KINDS,
    PayloadKind,
    SourceCapabilities,
    SourceDescriptor,
    SourceHealth,
    SourceKind,
    SourceSample,
    SourceStatus,
    TrafficDataSource,
)
from app.ingestion.live import (
    BackpressurePolicy,
    LiveSourceConfig,
    LiveTrafficDataSource,
)
from app.ingestion.recorded import RecordedTrafficDataSource
from app.ingestion.registry import (
    SOURCE_KIND_PROFILES,
    SourceKindProfile,
    SourceRegistry,
    live_profiles,
    profile_for,
    recorded_profiles,
    source_registry,
)

__all__ = [
    "LIVE_KINDS",
    "RECORDED_KINDS",
    "SOURCE_KIND_PROFILES",
    "BackpressurePolicy",
    "LiveSourceConfig",
    "LiveTrafficDataSource",
    "PayloadKind",
    "RecordedTrafficDataSource",
    "SourceCapabilities",
    "SourceDescriptor",
    "SourceHealth",
    "SourceKind",
    "SourceKindProfile",
    "SourceRegistry",
    "SourceSample",
    "SourceStatus",
    "TrafficDataSource",
    "live_profiles",
    "profile_for",
    "recorded_profiles",
    "source_registry",
]
