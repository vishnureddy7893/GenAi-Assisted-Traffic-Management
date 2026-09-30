"""Data source contracts: capability coherence, convergence, and honest 501s.

The test doubles here live in the test suite by design. A stub source shipped in
``app`` would be one import away from being mistaken for a working pipeline, and
a source that silently yields nothing is worse than one that refuses to start.
"""

from __future__ import annotations

import datetime as dt

import pytest
from domain_fixtures import T0

from app.core.clock import ClockKind, VirtualClock, WallClock
from app.core.errors import NotAvailableYetError, SourceCapabilityError
from app.ingestion.base import (
    LIVE_KINDS,
    RECORDED_KINDS,
    PayloadKind,
    SourceCapabilities,
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
    SourceRegistry,
    live_profiles,
    profile_for,
    recorded_profiles,
    source_registry,
)

RECORDED_CAPS = SourceCapabilities(
    seekable=True,
    bounded=True,
    replayable=True,
    realtime=False,
    supports_variable_rate=True,
    nominal_fps=25.0,
    duration_s=120.0,
    sample_count=3000,
    payload_kind=PayloadKind.VIDEO_FRAME,
)

LIVE_CAPS = SourceCapabilities(
    seekable=False,
    bounded=False,
    replayable=False,
    realtime=True,
    requires_backpressure=True,
    may_drop_samples=True,
    nominal_fps=25.0,
    payload_kind=PayloadKind.VIDEO_FRAME,
)


class FakeRecordedSource(RecordedTrafficDataSource):
    """Minimal Mode A source used to exercise the contract."""

    def __init__(self, *, sample_times: tuple[float, ...] = (0.0, 1.0, 2.0)) -> None:
        super().__init__(source_id="rec-1", epoch=T0, junction_id="J11")
        self._sample_times = sample_times
        self._cursor = 0

    @property
    def kind(self) -> SourceKind:
        return SourceKind.RECORDED_VIDEO

    @property
    def capabilities(self) -> SourceCapabilities:
        return RECORDED_CAPS

    @property
    def duration_s(self) -> float | None:
        return self._sample_times[-1]

    def open(self) -> None:
        self._status = SourceStatus.READY

    def close(self) -> None:
        self._status = SourceStatus.CLOSED

    def read(self) -> SourceSample | None:
        if self._cursor >= len(self._sample_times):
            self._status = SourceStatus.EXHAUSTED
            return None
        media_time = self._sample_times[self._cursor]
        self._cursor += 1
        self.clock.advance_to(media_time)
        return SourceSample(
            source_id=self.source_id,
            sequence=self._next_sequence(),
            payload_kind=PayloadKind.VIDEO_FRAME,
            payload=None,
            captured_at=self.clock.now(),
            media_time_s=media_time,
        )

    def seek(self, media_time_s: float) -> None:
        self.capabilities.require_seekable()
        self.clock.seek_to(media_time_s)
        self._cursor = sum(1 for t in self._sample_times if t < media_time_s)

    def reset(self) -> None:
        self._cursor = 0
        self.clock.seek_to(0.0)

    def health(self) -> SourceHealth:
        return SourceHealth(status=self.status, samples_read=self._cursor)


class FakeLiveSource(LiveTrafficDataSource):
    """Minimal Mode B source used to exercise the contract."""

    def __init__(self) -> None:
        super().__init__(
            source_id="live-1",
            config=LiveSourceConfig(uri="rtsp://user:pw@camera.test/stream"),
            junction_id="J11",
        )
        self._dropped = 0

    @property
    def kind(self) -> SourceKind:
        return SourceKind.LIVE_RTSP

    @property
    def capabilities(self) -> SourceCapabilities:
        return LIVE_CAPS

    def open(self) -> None:
        self.connect()

    def connect(self) -> None:
        self._status = SourceStatus.READY

    def reconnect(self) -> None:
        self._status = SourceStatus.DEGRADED

    def close(self) -> None:
        self._status = SourceStatus.CLOSED

    def read(self) -> SourceSample | None:
        return SourceSample(
            source_id=self.source_id,
            sequence=self._next_sequence(),
            payload_kind=PayloadKind.VIDEO_FRAME,
            payload=None,
            captured_at=self.clock.now(),
            dropped_before=self._dropped,
        )

    @property
    def queue_depth(self) -> int:
        return 0

    @property
    def dropped_count(self) -> int:
        return self._dropped

    def health(self) -> SourceHealth:
        return SourceHealth(status=self.status, samples_dropped=self._dropped)


class TestConvergence:
    def test_both_modes_satisfy_the_same_base_interface(self) -> None:
        """What converges is the output, not the mechanism."""
        assert isinstance(FakeRecordedSource(), TrafficDataSource)
        assert isinstance(FakeLiveSource(), TrafficDataSource)

    def test_both_modes_emit_the_same_sample_type(self) -> None:
        with FakeRecordedSource() as recorded:
            recorded_sample = recorded.read()
        live = FakeLiveSource()
        live.open()
        live_sample = live.read()
        assert isinstance(recorded_sample, SourceSample)
        assert isinstance(live_sample, SourceSample)

    def test_the_source_supplies_the_clock(self) -> None:
        """Downstream code must never read the wall clock directly."""
        assert FakeRecordedSource().clock.kind is ClockKind.VIRTUAL
        assert FakeLiveSource().clock.kind is ClockKind.WALL
        assert isinstance(FakeRecordedSource().clock, VirtualClock)
        assert isinstance(FakeLiveSource().clock, WallClock)


class TestRecordedSource:
    def test_timestamps_come_from_the_media_timeline(self) -> None:
        """A replay of old video must not masquerade as current traffic."""
        with FakeRecordedSource(sample_times=(0.0, 30.0)) as source:
            samples = list(source)
        assert samples[0].captured_at == T0
        assert samples[1].captured_at == T0 + dt.timedelta(seconds=30)

    def test_a_bounded_source_terminates(self) -> None:
        with FakeRecordedSource() as source:
            assert len(list(source)) == 3
            assert source.status is SourceStatus.EXHAUSTED

    def test_replay_is_deterministic(self) -> None:
        """The precondition for attributing a decision change to its cause."""
        source = FakeRecordedSource()
        source.open()
        first = [s.media_time_s for s in source]
        source.reset()
        second = [s.media_time_s for s in source]
        assert first == second

    def test_seeking_repositions_the_clock(self) -> None:
        source = FakeRecordedSource(sample_times=(0.0, 1.0, 2.0, 3.0))
        source.open()
        source.seek(2.0)
        assert source.position_s == 2.0
        assert source.read().media_time_s == 2.0  # type: ignore[union-attr]

    def test_variable_rate_is_rejected_when_undeclared(self) -> None:
        """A silently ignored rate change makes throughput conclusions wrong."""

        class FixedRate(FakeRecordedSource):
            @property
            def capabilities(self) -> SourceCapabilities:
                return RECORDED_CAPS.model_copy(update={"supports_variable_rate": False})

        with pytest.raises(SourceCapabilityError, match="variable-rate"):
            FixedRate().set_rate(4.0)

    def test_an_epoch_is_required_and_must_be_aware(self) -> None:
        """Defaulting the epoch to 'now' would make stale evidence look fresh."""
        with pytest.raises(ValueError, match="timezone-aware"):

            class NaiveEpoch(FakeRecordedSource):
                def __init__(self) -> None:
                    RecordedTrafficDataSource.__init__(
                        self,
                        source_id="x",
                        epoch=dt.datetime(2024, 3, 14),
                    )

            NaiveEpoch()

    def test_capability_invariants_are_enforced(self) -> None:
        for bad, message in (
            ({"bounded": False}, "must be bounded"),
            ({"replayable": False}, "must be replayable"),
            ({"realtime": True}, "must not declare realtime"),
            ({"requires_backpressure": True}, "must not require backpressure"),
        ):
            with pytest.raises(ValueError, match=message):
                RecordedTrafficDataSource.validate_capabilities(
                    SourceKind.RECORDED_VIDEO, RECORDED_CAPS.model_copy(update=bad)
                )

    def test_a_live_kind_cannot_be_declared_recorded(self) -> None:
        with pytest.raises(ValueError, match="not a recorded"):
            RecordedTrafficDataSource.validate_capabilities(
                SourceKind.LIVE_RTSP, RECORDED_CAPS
            )

    def test_valid_recorded_capabilities_pass(self) -> None:
        RecordedTrafficDataSource.validate_capabilities(
            SourceKind.RECORDED_VIDEO, RECORDED_CAPS
        )


class TestLiveSource:
    def test_seeking_a_live_source_is_refused_loudly(self) -> None:
        """A silently ignored seek analyses the wrong point in time."""
        with pytest.raises(SourceCapabilityError, match="no seekable timeline"):
            FakeLiveSource().seek(30.0)

    def test_require_seekable_refuses_on_live_capabilities(self) -> None:
        with pytest.raises(SourceCapabilityError):
            LIVE_CAPS.require_seekable()

    def test_backpressure_defaults_to_dropping_the_oldest(self) -> None:
        """A decision must reflect the present; stale frames have no value."""
        assert FakeLiveSource().config.backpressure is BackpressurePolicy.DROP_OLDEST

    def test_connection_uri_is_not_exposed_in_the_descriptor(self) -> None:
        """RTSP URIs routinely embed credentials."""
        descriptor = FakeLiveSource().describe()
        assert "pw" not in descriptor.model_dump_json()

    def test_unbounded_sources_declare_no_duration(self) -> None:
        assert LIVE_CAPS.duration_s is None

    def test_capability_invariants_are_enforced(self) -> None:
        for bad, message in (
            ({"bounded": True}, "cannot be bounded"),
            ({"seekable": True}, "cannot be seekable"),
            ({"replayable": True}, "cannot be replayable"),
            ({"realtime": False}, "must declare realtime"),
            ({"requires_backpressure": False}, "must declare requires_backpressure"),
        ):
            with pytest.raises(ValueError, match=message):
                LiveTrafficDataSource.validate_capabilities(
                    SourceKind.LIVE_RTSP, LIVE_CAPS.model_copy(update=bad)
                )

    def test_reconnect_marks_the_source_degraded_not_failed(self) -> None:
        """A source dropping some frames is usable with reduced confidence."""
        source = FakeLiveSource()
        source.open()
        source.reconnect()
        assert source.status is SourceStatus.DEGRADED


class TestSourceHealth:
    def test_drop_rate_is_none_before_any_read(self) -> None:
        assert SourceHealth(status=SourceStatus.READY).drop_rate is None

    def test_drop_rate_is_reported(self) -> None:
        health = SourceHealth(status=SourceStatus.DEGRADED, samples_read=90, samples_dropped=10)
        assert health.drop_rate == pytest.approx(0.1)


class TestRegistry:
    def test_every_kind_belongs_to_exactly_one_mode(self) -> None:
        assert RECORDED_KINDS.isdisjoint(LIVE_KINDS)
        assert set(SourceKind) == RECORDED_KINDS | LIVE_KINDS

    def test_every_kind_has_a_catalogue_profile(self) -> None:
        for kind in SourceKind:
            assert profile_for(kind).kind is kind

    def test_profiles_are_split_by_mode(self) -> None:
        assert all(p.mode == "A" for p in recorded_profiles())
        assert all(p.mode == "B" for p in live_profiles())
        assert len(recorded_profiles()) + len(live_profiles()) == len(SOURCE_KIND_PROFILES)

    def test_only_recorded_media_sources_are_implemented(self) -> None:
        """Phase 2 adds recorded image and video ingestion, and nothing more."""
        from app.ingestion.registry import register_default_sources

        register_default_sources()
        assert source_registry.implemented_kinds() == {
            SourceKind.RECORDED_IMAGE,
            SourceKind.RECORDED_VIDEO,
        }

    def test_creating_an_unimplemented_source_raises_a_501_error(self) -> None:
        """Not an empty success, which would look like 'no traffic detected'."""
        with pytest.raises(NotAvailableYetError) as excinfo:
            source_registry.factory(SourceKind.RECORDED_COUNT_DATASET)
        assert excinfo.value.status_code == 501
        assert excinfo.value.details["phase"] == 2

    def test_registration_reconciles_the_catalogue(self) -> None:
        registry = SourceRegistry()
        registry.register(SourceKind.RECORDED_VIDEO, FakeRecordedSource)
        profiles = {p.kind: p for p in registry.profiles()}
        assert profiles[SourceKind.RECORDED_VIDEO].implemented is True
        assert profiles[SourceKind.LIVE_RTSP].implemented is False

    def test_double_registration_is_rejected(self) -> None:
        registry = SourceRegistry()
        registry.register(SourceKind.RECORDED_VIDEO, FakeRecordedSource)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(SourceKind.RECORDED_VIDEO, FakeRecordedSource)

    def test_recorded_kinds_are_listed_before_live_ones(self) -> None:
        """The recorded pipeline is the primary capstone input."""
        modes = [p.mode for p in SOURCE_KIND_PROFILES]
        assert modes == sorted(modes)
