"""Shared test fixtures.

Note on test doubles: the minimal detector/tracker/extractor implementations used
to exercise the CV contracts live in the test suite, never in ``app``. A stub
detector shipped in production code is one import away from being mistaken for a
working one, and a system that silently produces empty detections is worse than
one that fails to start.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from domain_fixtures import T0, measure_float, measure_int
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.domain.provenance import (
    Provenance,
    ValueSource,
)
from app.main import create_app

__all__ = ["T0", "measure_float", "measure_int"]



@pytest.fixture
def settings() -> Settings:
    """Deterministic settings, independent of any local .env file."""
    return Settings(
        app_env="local",
        log_level="WARNING",
        log_format="console",
        api_cors_origins=["http://localhost:5173"],
        max_observation_staleness_s=5.0,
        drop_observation_after_s=30.0,
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """HTTP client bound to an app built from the test settings."""
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def observed_provenance() -> Provenance:
    """Provenance for a directly measured value."""
    return Provenance(
        source=ValueSource.OBSERVED,
        producer="test.fixture",
        producer_version="1.0",
        observed_at=T0,
        computed_at=T0,
    )


