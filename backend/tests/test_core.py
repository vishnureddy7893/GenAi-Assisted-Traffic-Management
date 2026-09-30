"""Configuration, logging, clock and error-taxonomy tests."""

from __future__ import annotations

import datetime as dt
import json
import logging

import pytest
from pydantic import ValidationError

from app.core.clock import VirtualClock, WallClock
from app.core.config import Settings
from app.core.context import bind_context, get_request_id
from app.core.errors import (
    NoAdmissibleRouteError,
    NotAvailableYetError,
    SourceCapabilityError,
    TrafficSystemError,
)
from app.core.logging import ConsoleFormatter, JsonFormatter

T0 = dt.datetime(2024, 3, 14, 8, 30, tzinfo=dt.UTC)


class TestSettings:
    def test_secrets_are_redacted_in_repr(self) -> None:
        """A secret that prints itself will eventually print itself into a log."""
        settings = Settings(openrouter_api_key="sk-super-secret-value")
        assert "sk-super-secret-value" not in repr(settings)
        assert "sk-super-secret-value" not in str(settings.openrouter_api_key)
        assert settings.openrouter_api_key is not None
        assert settings.openrouter_api_key.get_secret_value() == "sk-super-secret-value"

    def test_live_source_uri_is_a_secret(self) -> None:
        """RTSP URIs routinely embed credentials, so the whole URI is sensitive."""
        settings = Settings(live_source_uri="rtsp://user:pw@10.0.0.5:554/stream")
        assert "pw@" not in repr(settings)

    def test_cors_origins_accept_a_comma_separated_string(self) -> None:
        settings = Settings(api_cors_origins="http://a.test, http://b.test")
        assert settings.api_cors_origins == ["http://a.test", "http://b.test"]

    def test_drop_threshold_must_not_precede_staleness_threshold(self) -> None:
        """An observation cannot be dropped before it is even considered stale."""
        with pytest.raises(ValidationError, match="cannot be dropped before"):
            Settings(max_observation_staleness_s=30.0, drop_observation_after_s=5.0)

    def test_genai_is_not_configured_without_both_key_and_model(self) -> None:
        assert Settings(openrouter_api_key="k").genai_configured is False
        assert Settings(openrouter_model="m").genai_configured is False
        assert Settings(openrouter_api_key="k", openrouter_model="m").genai_configured is True

    def test_port_is_range_checked(self) -> None:
        with pytest.raises(ValidationError):
            Settings(api_port=70000)


class TestVirtualClock:
    """The clock that keeps recorded replay dimensionally correct."""

    def test_now_is_epoch_plus_media_time(self) -> None:
        clock = VirtualClock(epoch=T0)
        clock.advance_to(90.0)
        assert clock.now() == T0 + dt.timedelta(seconds=90)

    def test_replay_rate_does_not_affect_timestamps(self) -> None:
        """Replaying at 10x must not multiply every derived speed by 10."""
        slow = VirtualClock(epoch=T0, rate=1.0)
        fast = VirtualClock(epoch=T0, rate=10.0)
        slow.advance_to(30.0)
        fast.advance_to(30.0)
        assert slow.now() == fast.now()
        assert slow.monotonic() == fast.monotonic()

    def test_advance_cannot_move_backwards(self) -> None:
        """An out-of-order sample is a fault, not a silent timeline reorder."""
        clock = VirtualClock(epoch=T0)
        clock.advance_to(10.0)
        with pytest.raises(ValueError, match="cannot move backwards"):
            clock.advance_to(5.0)

    def test_seek_may_move_backwards(self) -> None:
        clock = VirtualClock(epoch=T0)
        clock.advance_to(10.0)
        clock.seek_to(2.0)
        assert clock.media_time_s == 2.0

    def test_naive_epoch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            VirtualClock(epoch=dt.datetime(2024, 3, 14, 8, 30))

    def test_set_rate_preserves_position(self) -> None:
        clock = VirtualClock(epoch=T0)
        clock.advance_to(42.0)
        clock.set_rate(4.0)
        assert clock.media_time_s == 42.0
        assert clock.rate == 4.0

    def test_elapsed_since_reports_negative_on_clock_skew(self) -> None:
        """Future-dated evidence means camera/host skew: surfaced, not clamped."""
        clock = VirtualClock(epoch=T0)
        assert clock.elapsed_since(T0 + dt.timedelta(seconds=5)) == -5.0

    def test_elapsed_since_rejects_naive_timestamps(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            VirtualClock(epoch=T0).elapsed_since(dt.datetime(2024, 3, 14))


class TestWallClock:
    def test_now_is_timezone_aware(self) -> None:
        assert WallClock().now().tzinfo is not None

    def test_monotonic_is_non_decreasing(self) -> None:
        clock = WallClock()
        assert clock.monotonic() <= clock.monotonic()


class TestErrors:
    def test_not_available_yet_maps_to_501_and_names_its_phase(self) -> None:
        error = NotAvailableYetError("later", phase=7, phase_name="Routing")
        assert error.status_code == 501
        payload = error.to_payload()["error"]
        assert payload["code"] == "not_available_yet"
        assert payload["details"]["phase"] == 7

    def test_no_admissible_route_is_a_first_class_outcome(self) -> None:
        """Never a trigger for relaxing a hard constraint to produce some answer."""
        assert NoAdmissibleRouteError("all pruned").status_code == 409

    def test_source_capability_error_carries_the_capability(self) -> None:
        error = SourceCapabilityError("cannot seek", capability="seekable")
        assert error.to_payload()["error"]["details"]["capability"] == "seekable"

    def test_every_error_renders_the_same_envelope(self) -> None:
        payload = TrafficSystemError("boom", extra=1).to_payload()
        assert set(payload["error"]) == {"code", "message", "details"}


class TestLogging:
    def test_json_formatter_emits_one_parseable_object(self) -> None:
        record = logging.LogRecord(
            name="app.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="queue estimated",
            args=(),
            exc_info=None,
        )
        record.junction_id = "J11"
        payload = json.loads(JsonFormatter().format(record))
        assert payload["message"] == "queue estimated"
        assert payload["junction_id"] == "J11"
        assert payload["level"] == "INFO"

    def test_correlation_ids_are_attached_automatically(self) -> None:
        record = logging.LogRecord(
            name="app.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )
        with bind_context(request_id="req-1", scenario_id="sc-1"):
            payload = json.loads(JsonFormatter().format(record))
        assert payload["request_id"] == "req-1"
        assert payload["scenario_id"] == "sc-1"

    def test_logging_never_raises_on_an_unserialisable_value(self) -> None:
        """A logging call must not be able to break a request."""
        record = logging.LogRecord(
            name="app.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="odd",
            args=(),
            exc_info=None,
        )
        record.weird = object()
        assert json.loads(JsonFormatter().format(record))["weird"]

    def test_console_formatter_includes_message_and_extras(self) -> None:
        record = logging.LogRecord(
            name="app.test",
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg="stale observation",
            args=(),
            exc_info=None,
        )
        record.age_s = 42.0
        line = ConsoleFormatter().format(record)
        assert "stale observation" in line
        assert "age_s=42.0" in line


class TestContext:
    def test_context_is_restored_on_exit(self) -> None:
        assert get_request_id() is None
        with bind_context(request_id="outer"):
            assert get_request_id() == "outer"
            with bind_context(request_id="inner"):
                assert get_request_id() == "inner"
            assert get_request_id() == "outer"
        assert get_request_id() is None
