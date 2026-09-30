"""Architectural guardrails.

These tests do not exercise behaviour; they enforce constraints that are easy to
state and easy to violate silently as the codebase grows. Each one encodes a
decision that would otherwise erode.
"""

from __future__ import annotations

import ast
import warnings
from pathlib import Path

import pytest

from app.core.versions import IMPLEMENTED_PHASE
from app.domain.observation import JunctionObservation

APP_ROOT = Path(__file__).resolve().parents[1] / "app"


def _python_files() -> list[Path]:
    return sorted(APP_ROOT.rglob("*.py"))


class TestNoRealWorldControl:
    """The capstone must not acquire an actuation path to real infrastructure."""

    FORBIDDEN_CALLS = (
        "set_signal",
        "set_phase",
        "actuate",
        "send_command",
        "write_controller",
        "ntcip",
        "snmp_set",
    )

    @pytest.mark.safety
    def test_no_actuation_style_entry_points_exist(self) -> None:
        offenders: list[str] = []
        for path in _python_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    lowered = node.name.lower()
                    if any(token in lowered for token in self.FORBIDDEN_CALLS):
                        offenders.append(f"{path.name}:{node.name}")
        assert offenders == [], (
            "actuation-style entry points found; this system is advisory only and "
            f"must not gain a control path: {offenders}"
        )

    @pytest.mark.safety
    def test_signal_advisories_cannot_disable_their_advisory_flag(self) -> None:
        from app.domain.decision import SignalAdvisory
        from app.domain.enums import SignalAdvisoryAction

        advisory = SignalAdvisory(
            junction_id="J11",
            action=SignalAdvisoryAction.NO_CHANGE,
            rationale="steady state",
        )
        with pytest.raises(Exception, match="advisory"):
            advisory.model_copy(update={"advisory_only": False}).model_validate(
                advisory.model_dump() | {"advisory_only": False}
            )


class TestNoFabricatedDefaults:
    """A traffic quantity must never arrive with a plausible default."""

    #: Field-name fragments that denote a fabricable traffic quantity.
    QUANTITY_TOKENS = ("length", "speed", "count", "flow", "density", "time", "ratio")

    def _assert_no_defaulted_quantities(self, model: object, path: str) -> None:
        """Recurse a freshly constructed model asserting every quantity is absent.

        A nested *empty container* -- a ``TravelTimeEstimate`` whose own fields are
        all ``None`` -- is absence, not a default, so the check descends into it
        rather than rejecting it. What must never appear is a bare number.
        """
        from app.domain.base import DomainModel

        for name in type(model).model_fields:  # type: ignore[attr-defined]
            value = getattr(model, name)
            if isinstance(value, DomainModel):
                self._assert_no_defaulted_quantities(value, f"{path}.{name}")
                continue
            if not any(token in name for token in self.QUANTITY_TOKENS):
                continue
            if isinstance(value, dict | tuple | list | frozenset | set):
                # An empty collection is absence ("no classes were measured"),
                # not a default. A populated one would be pre-seeded data.
                assert not value, (
                    f"{path}.{name} is pre-populated with {value!r}; a traffic "
                    "quantity must not arrive with seeded contents"
                )
                continue
            assert value is None, (
                f"{path}.{name} defaults to {value!r}; a traffic quantity must be "
                "explicitly absent rather than defaulted"
            )

    @pytest.mark.safety
    def test_numeric_traffic_fields_default_to_none_not_zero(self) -> None:
        """Zero is a legitimate reading (an empty approach at 04:00).

        Using it to mean "unknown" is how a routing engine concludes that a
        gridlocked road is empty.
        """
        from app.domain.observation import ApproachObservation
        from app.domain.state import ApproachState, RoadSegmentState

        self._assert_no_defaulted_quantities(
            RoadSegmentState(segment_id="S1"), "RoadSegmentState"
        )
        self._assert_no_defaulted_quantities(
            ApproachState(approach_id="A1", incoming_segment_id="S1"), "ApproachState"
        )
        self._assert_no_defaulted_quantities(
            ApproachObservation(approach_id="A1", incoming_segment_id="S1"),
            "ApproachObservation",
        )

    @pytest.mark.safety
    def test_measurements_cannot_be_constructed_without_provenance(self) -> None:
        from app.domain.provenance import Measurement, Unit

        with pytest.raises(Exception, match="provenance"):
            Measurement[float](value=42.0, unit=Unit.KILOMETER_PER_HOUR)  # type: ignore[call-arg]


class TestCompatAliasIsQuarantined:
    def test_nothing_inside_app_imports_the_deprecated_alias(self) -> None:
        """Two names for one concept on day one become permanent by month six."""
        offenders = [
            path.relative_to(APP_ROOT).as_posix()
            for path in _python_files()
            if path.name != "compat.py" and "domain.compat" in path.read_text(encoding="utf-8")
        ]
        assert offenders == [], f"app modules must not import the compat alias: {offenders}"

    def test_the_alias_warns_and_resolves_to_the_replacement(self) -> None:
        import app.domain.compat as compat

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            alias = compat.TrafficState
        assert alias is JunctionObservation
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    def test_unknown_attributes_still_raise(self) -> None:
        import app.domain.compat as compat

        with pytest.raises(AttributeError):
            _ = compat.NotAThing  # type: ignore[attr-defined]


class TestLayering:
    def test_the_domain_layer_does_not_depend_on_the_api_layer(self) -> None:
        """Domain schemas describe traffic; they must not know about HTTP."""
        offenders: list[str] = []
        for path in (APP_ROOT / "domain").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "from app.api" in text or "import fastapi" in text:
                offenders.append(path.name)
        assert offenders == []

    def test_the_domain_layer_does_not_depend_on_ingestion_or_vision(self) -> None:
        offenders: list[str] = []
        for path in (APP_ROOT / "domain").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "from app.ingestion" in text or "from app.vision" in text:
                offenders.append(path.name)
        assert offenders == []

    def test_vision_depends_on_the_domain_not_the_reverse(self) -> None:
        text = (APP_ROOT / "vision" / "extractor.py").read_text(encoding="utf-8")
        assert "from app.domain.observation import" in text


class TestNoHardCodedSecrets:
    @pytest.mark.safety
    def test_no_credential_literals_in_source(self) -> None:
        suspicious = ("sk-", "api_key=", "password=", "secret=")
        offenders: list[str] = []
        for path in _python_files():
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#") or "example" in stripped.lower():
                    continue
                lowered = stripped.lower()
                for token in suspicious:
                    if token in lowered and "=" in lowered and '"' in lowered:
                        offenders.append(f"{path.name}:{lineno}")
        assert offenders == [], f"possible hard-coded credential: {offenders}"


class TestPhaseHonesty:
    def test_the_declared_phase_matches_the_shipped_capabilities(self) -> None:
        """Phases 1-8: foundation through traffic simulation."""
        assert IMPLEMENTED_PHASE == 8

    def test_only_raw_ingestion_sources_are_registered(self) -> None:
        """Phase 2 ships raw media ingestion. No live source, no CV pipeline."""
        from app.ingestion.base import LIVE_KINDS, SourceKind
        from app.ingestion.registry import register_default_sources, source_registry

        register_default_sources()
        implemented = source_registry.implemented_kinds()
        assert implemented == {SourceKind.RECORDED_IMAGE, SourceKind.RECORDED_VIDEO}
        assert not (implemented & LIVE_KINDS), "live ingestion must remain unimplemented"

    def test_no_model_artifact_is_referenced(self) -> None:
        from app.api.routes.health import versions

        assert versions().model_versions == {}
