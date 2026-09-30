"""Shared domain-object builders for the test suite.

Deliberately **not** imported as ``tests.conftest``.

Ultralytics 8.4.130 installs a top-level ``tests`` package into site-packages
(a packaging bug on their side: their own test suite ships as an importable
distribution). That shadows this directory whenever site-packages precedes the
rootdir on ``sys.path``, and ``from tests.conftest import ...`` then resolves to
*their* conftest and fails with a confusing ImportError.

A uniquely named sibling module cannot collide, and pytest puts this directory
on the path for us. The fixtures in ``conftest.py`` re-export from here.
"""

from __future__ import annotations

import datetime as dt

from app.domain.provenance import (
    FloatMeasurement,
    IntMeasurement,
    Measurement,
    Provenance,
    Unit,
    ValueSource,
)

T0 = dt.datetime(2024, 3, 14, 8, 30, tzinfo=dt.UTC)


def measure_float(
    value: float,
    unit: Unit,
    *,
    at: dt.datetime = T0,
    source: ValueSource = ValueSource.OBSERVED,
    **kwargs: object,
) -> FloatMeasurement:
    """Build a float measurement with minimal ceremony."""
    return Measurement[float](
        value=value,
        unit=unit,
        provenance=Provenance(
            source=source,
            producer="test.fixture",
            producer_version="1.0",
            observed_at=at,
            computed_at=at,
            **kwargs,  # type: ignore[arg-type]
        ),
    )


def measure_int(
    value: int,
    unit: Unit,
    *,
    at: dt.datetime = T0,
    source: ValueSource = ValueSource.OBSERVED,
) -> IntMeasurement:
    """Build an integer measurement with minimal ceremony."""
    return Measurement[int](
        value=value,
        unit=unit,
        provenance=Provenance(
            source=source,
            producer="test.fixture",
            producer_version="1.0",
            observed_at=at,
            computed_at=at,
        ),
    )
