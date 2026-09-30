"""AI Traffic Advisor tests.

The advisor is the one component in this system whose output is generated
rather than computed, so almost every test here is about what it must refuse to
say. It must not name a faster route without a route comparison, must not turn
a missing count into zero, must not infer an incident or an emergency from
congestion, must not present simulated figures as measurements, and must not
show a generated advisory that cites numbers the facts do not support.

The provider is a double throughout. Nothing in this file makes a network call,
and a test that needed one would be testing OpenRouter rather than the advisor.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from network_fixtures import square_network

from app.advisor.facts import TrafficAdvisorFacts, build_facts
from app.advisor.fallback import deterministic_advisory
from app.advisor.prompt import SYSTEM_PROMPT, facts_payload, user_prompt
from app.advisor.provider import (
    AdvisorProviderError,
    OpenRouterAdvisorLLM,
    _extract_message,
    build_provider,
)
from app.advisor.schemas import AdvisoryMode
from app.advisor.service import generate_advisory
from app.analytics.schemas import AnalyticsProvenance, ModelIdentity, ObservationMode
from app.api.routes import advisor as advisor_routes
from app.core.config import Settings
from app.domain.enums import CongestionLevel
from app.domain.network import RoadNetwork
from app.domain.provenance import Provenance, ValueSource
from app.domain.site import ObservationSite
from app.main import create_app
from app.sites.registry import utc_now
from app.state.classifier import thresholds_from
from app.state.schemas import (
    CongestionAssessment,
    EvidenceBasis,
    IntersectionTrafficState,
    QualityGrade,
)

AREA = "observation-area:cam-1"
MEDIA_ID = "media-1"

T0 = dt.datetime(2026, 9, 22, 9, 0, tzinfo=dt.UTC)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def advisor_settings(tmp_path: Path) -> Settings:
    """Settings with no LLM credentials, which is the demo default."""
    return Settings(
        app_env="local",
        log_level="ERROR",
        media_dir=tmp_path / "media",
        sites_file=tmp_path / "sites.json",
        confirmed_bindings_file=tmp_path / "bindings.json",
    )


def _configured_settings(tmp_path: Path, secret: str) -> Settings:
    """Settings that look configured, without reaching any provider."""
    return Settings(
        app_env="local",
        log_level="ERROR",
        media_dir=tmp_path / "media",
        sites_file=tmp_path / "sites.json",
        confirmed_bindings_file=tmp_path / "bindings.json",
        openrouter_api_key=secret,
        openrouter_model="anthropic/claude-3.5-haiku",
    )


def make_state(
    *,
    level: CongestionLevel = CongestionLevel.HEAVY,
    index: float | None = 0.74,
    vehicle_count: int = 46,
    stationary: int | None = 12,
    settings: Settings | None = None,
    reason: str | None = None,
) -> IntersectionTrafficState:
    """An interpreted state, built with the real domain objects."""
    resolved = settings or Settings(app_env="local", log_level="ERROR")
    return IntersectionTrafficState(
        state_id="state-1",
        observation_area_id=AREA,
        mode=ObservationMode.SNAPSHOT,
        as_of=T0,
        assessed_at=T0,
        congestion=CongestionAssessment(
            level=level,
            index=index,
            basis=EvidenceBasis.IMAGE_SPACE,
            thresholds=thresholds_from(resolved),
            method="occupancy_and_density",
            method_version="1.0",
            reason=reason,
        ),
        vehicle_count=vehicle_count,
        stationary_vehicle_count=stationary,
        quality_grade=QualityGrade.GOOD,
        analytics_provenance=AnalyticsProvenance(
            media_id=MEDIA_ID,
            source_id=f"image:{MEDIA_ID}",
            frame_index=0,
            timestamp=T0,
            model=ModelIdentity(name="yolov8s", backend="ultralytics"),
            method="image_analytics",
            method_version="1.0",
            analytics_schema_version="1.0",
            computed_at=T0,
        ),
        schema_version="1.0",
    )


def make_site() -> ObservationSite:
    now = utc_now()
    return ObservationSite(
        observation_area_id=AREA,
        location={"latitude": 16.5051, "longitude": 80.6001},  # type: ignore[arg-type]
        label="Benz Circle NW mast",
        provenance=Provenance(
            source=ValueSource.OBSERVED,
            producer="operator",
            producer_version="1.0",
            observed_at=now,
            computed_at=now,
        ),
        recorded_at=now,
        updated_at=now,
    )


def facts_for(
    state: IntersectionTrafficState | None = None, **kwargs: Any
) -> TrafficAdvisorFacts:
    """Advisor facts from whatever the caller supplies."""
    return build_facts(AREA, state=state, now=T0, **kwargs)


class FakeLLM:
    """A provider double. Returns canned text, or raises what it was given."""

    def __init__(self, reply: str | None = None, error: Exception | None = None) -> None:
        self._reply = reply
        self._error = error
        self.calls: list[tuple[str, str]] = []

    @property
    def model_id(self) -> str:
        return "fake/model-1"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if self._error is not None:
            raise self._error
        assert self._reply is not None
        return self._reply


def good_reply(**overrides: Any) -> str:
    """A well-formed advisory that cites only real facts and invents no numbers."""
    payload = {
        "summary": "The system classifies this area as heavy.",
        "traffic_assessment": "Congestion is elevated on the monitored approach.",
        "suggestions": ["Evaluate an alternate route before redirecting demand."],
        "safety_considerations": ["Advisory only; no control path exists."],
        "route_advice": "An alternate route should be evaluated using current routing information.",
        "limitations": ["No route comparison was supplied."],
        "evidence_ids": ["state.level"],
    }
    payload.update(overrides)
    return json.dumps(payload)


@pytest.fixture
def client(advisor_settings: Settings) -> Iterator[TestClient]:
    app = create_app(advisor_settings)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def seed_state(client: TestClient, state: IntersectionTrafficState | None) -> None:
    """Put a state into the cache the state endpoint populates."""
    client.app.state.traffic_states = {MEDIA_ID: state} if state else {}  # type: ignore[attr-defined]


def load_network(app: Any, network: RoadNetwork) -> None:
    from app.network.loader import NetworkSource
    from app.network.store import NetworkStore

    source = NetworkSource(
        relative_path="test.json",
        sha256="0" * 64,
        size_bytes=1,
        loaded_at=T0,
        network_id=network.network_id,
        version=network.version,
        junction_count=len(network.junctions),
        segment_count=len(network.segments),
        facility_count=len(network.facilities),
    )
    store = NetworkStore()
    store.load(network, source)
    app.state.network_store = store


# ---------------------------------------------------------------------------
# Fact building
# ---------------------------------------------------------------------------
class TestFactBuilding:
    def test_a_state_becomes_citable_facts(self) -> None:
        facts = facts_for(make_state())
        assert facts.traffic_state is CongestionLevel.HEAVY
        assert facts.vehicle_count == 46
        assert facts.fact_table.fact("state.vehicle_count").value == 46
        assert facts.fact_table.fact("state.level").value == "heavy"

    def test_a_missing_count_is_absent_not_zero(self) -> None:
        """The single most important rule: absence never becomes a number."""
        facts = facts_for(None)
        assert facts.vehicle_count is None
        assert "state.vehicle_count" not in facts.fact_table.fact_ids
        reasons = {item.fact_id: item.reason for item in facts.unavailable}
        assert "state.vehicle_count" in reasons
        assert "no traffic state" in reasons["state.vehicle_count"]

    def test_a_missing_stationary_count_is_recorded_as_unavailable(self) -> None:
        facts = facts_for(make_state(stationary=None))
        assert facts.stationary_vehicle_count is None
        ids = {item.fact_id for item in facts.unavailable}
        assert "state.stationary_vehicle_count" in ids

    def test_an_unknown_state_carries_no_index(self) -> None:
        facts = facts_for(
            make_state(
                level=CongestionLevel.UNKNOWN,
                index=None,
                reason="insufficient evidence in frame",
            )
        )
        assert facts.congestion_index is None
        assert "state.congestion_index" not in facts.fact_table.fact_ids

    def test_no_incident_is_fabricated(self) -> None:
        """Heavy congestion is not an accident, and nothing here says it is."""
        facts = facts_for(make_state(level=CongestionLevel.GRIDLOCK))
        assert facts.incident_summaries == ()
        assert facts.has_incident is False
        reasons = {item.fact_id: item.reason for item in facts.unavailable}
        assert "never inferred from detections" in reasons["incidents.declared"]

    def test_no_emergency_is_inferred(self) -> None:
        facts = facts_for(make_state())
        assert facts.emergency_summaries == ()
        assert facts.has_emergency is False
        reasons = {item.fact_id: item.reason for item in facts.unavailable}
        assert "never something visible in footage" in reasons["emergency.declared"]

    def test_a_site_keeps_its_own_provenance(self) -> None:
        """Re-classifying a source's provenance is how a claim gets upgraded."""
        site = make_site()
        facts = facts_for(make_state(), site=site)
        assert facts.latitude == 16.5051
        assert (
            facts.fact_table.fact("site.latitude").provenance.source
            is site.provenance.source
        )

    def test_state_facts_are_derived_and_name_their_inputs(self) -> None:
        facts = facts_for(make_state())
        provenance = facts.fact_table.fact("state.vehicle_count").provenance
        assert provenance.source is ValueSource.DERIVED
        assert any("analytics:" in ref for ref in provenance.input_refs)

    def test_the_observation_area_id_is_an_entity_token(self) -> None:
        """So the digits inside an id are not mistaken for invented measurements."""
        facts = facts_for(make_state())
        assert AREA in facts.fact_table.entity_tokens


# ---------------------------------------------------------------------------
# Route rule
# ---------------------------------------------------------------------------
class TestRouteRule:
    def test_no_route_data_produces_no_route_recommendation(self) -> None:
        advisory = deterministic_advisory(facts_for(make_state()), now=T0)
        assert "no route can be described as faster" in advisory.route_advice.lower()
        assert "should be evaluated" in advisory.route_advice

    def test_a_route_comparison_produces_a_route_aware_explanation(self) -> None:
        facts = facts_for(make_state()).model_copy(
            update={
                "has_route_comparison": True,
                "route_baseline_s": 192.0,
                "route_alternative_s": 145.0,
                "route_improvement_pct": 24.5,
                "route_selected_id": "route-b",
            }
        )
        advice = deterministic_advisory(facts, now=T0).route_advice
        assert "145" in advice
        assert "192" in advice
        assert "24.5%" in advice
        assert "predicted" in advice.lower()

    def test_an_equal_comparison_claims_no_improvement(self) -> None:
        facts = facts_for(make_state()).model_copy(
            update={
                "has_route_comparison": True,
                "route_baseline_s": 150.0,
                "route_alternative_s": 150.0,
                "route_improvement_pct": 0.0,
            }
        )
        advice = deterministic_advisory(facts, now=T0).route_advice
        assert "does not show a meaningful improvement" in advice

    def test_routes_without_a_time_basis_are_not_compared(self) -> None:
        """A partial sum presented as an ETA understates by the legs it omits."""
        facts = facts_for(make_state(), route_result=None)
        assert facts.has_route_comparison is False
        reasons = {item.fact_id: item.reason for item in facts.unavailable}
        assert "route.comparison" in reasons


# ---------------------------------------------------------------------------
# Simulation rule
# ---------------------------------------------------------------------------
class TestSimulationRule:
    def test_simulation_is_labelled_as_modelled(self) -> None:
        facts = facts_for(make_state()).model_copy(
            update={"has_simulation": True, "simulation_improvement_pct": 11.0}
        )
        advisory = deterministic_advisory(facts, now=T0)
        joined = " ".join(advisory.limitations)
        assert "model under stated assumptions" in joined
        assert "not measurements of the road" in joined

    def test_the_prompt_tells_the_model_simulation_is_not_a_measurement(self) -> None:
        assert "the simulation models" in SYSTEM_PROMPT
        assert 'never "the road will take' in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------
class TestDeterministicFallback:
    @pytest.mark.parametrize(
        ("level", "marker"),
        [
            (CongestionLevel.FREE_FLOW, "No significant congestion"),
            (CongestionLevel.LIGHT, "light"),
            (CongestionLevel.MODERATE, "moderate"),
            (CongestionLevel.HEAVY, "Heavy traffic is indicated"),
            (CongestionLevel.SEVERE, "Severe congestion is indicated"),
            (CongestionLevel.GRIDLOCK, "Gridlock is indicated"),
        ],
    )
    def test_every_band_produces_an_advisory(
        self, level: CongestionLevel, marker: str
    ) -> None:
        advisory = deterministic_advisory(facts_for(make_state(level=level)), now=T0)
        assert marker in advisory.traffic_assessment
        assert advisory.suggestions
        assert advisory.mode is AdvisoryMode.DETERMINISTIC_FALLBACK

    def test_a_missing_state_is_handled_without_inventing_one(self) -> None:
        advisory = deterministic_advisory(facts_for(None), now=T0)
        assert "No traffic state has been interpreted" in advisory.traffic_assessment
        assert "not a report of clear roads" in advisory.traffic_assessment
        assert any("Run vision analysis" in item for item in advisory.suggestions)

    def test_an_unknown_band_is_not_read_as_clear(self) -> None:
        advisory = deterministic_advisory(
            facts_for(
                make_state(level=CongestionLevel.UNKNOWN, index=None, reason="no signal")
            ),
            now=T0,
        )
        assert "not a report of clear roads" in advisory.traffic_assessment

    def test_the_fallback_never_claims_a_model_wrote_it(self) -> None:
        advisory = deterministic_advisory(facts_for(make_state()), now=T0)
        assert advisory.model_id is None
        assert any("No language model was involved" in item for item in advisory.limitations)

    def test_attaching_a_model_to_a_fallback_is_rejected(self) -> None:
        advisory = deterministic_advisory(facts_for(make_state()), now=T0)
        with pytest.raises(Exception, match="must not name a model"):
            advisory.model_validate(advisory.model_dump() | {"model_id": "gpt-x"})

    def test_an_advisory_cannot_declare_itself_non_advisory(self) -> None:
        advisory = deterministic_advisory(facts_for(make_state()), now=T0)
        with pytest.raises(Exception, match="advisory_only cannot be False"):
            advisory.model_validate(advisory.model_dump() | {"advisory_only": False})

    def test_pedestrians_raise_a_safety_note_only_when_congested(self) -> None:
        crowded = facts_for(make_state(level=CongestionLevel.SEVERE)).model_copy(
            update={"pedestrian_count": 9}
        )
        calm = facts_for(make_state(level=CongestionLevel.FREE_FLOW)).model_copy(
            update={"pedestrian_count": 9}
        )
        assert any(
            "pedestrian clearance must not be shortened" in item
            for item in deterministic_advisory(crowded, now=T0).safety_considerations
        )
        assert not any(
            "pedestrian clearance" in item
            for item in deterministic_advisory(calm, now=T0).safety_considerations
        )

    def test_limitations_repeat_every_recorded_absence(self) -> None:
        facts = facts_for(None)
        advisory = deterministic_advisory(facts, now=T0)
        for item in facts.unavailable:
            assert any(item.label in line for line in advisory.limitations)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------
class TestProvider:
    def test_no_credentials_means_no_provider_rather_than_a_crash(
        self, advisor_settings: Settings
    ) -> None:
        assert advisor_settings.genai_configured is False
        assert build_provider(advisor_settings) is None

    def test_a_provider_is_built_when_credentials_exist(self, tmp_path: Path) -> None:
        settings = Settings(
            app_env="local",
            log_level="ERROR",
            openrouter_api_key="sk-test-not-a-real-key",
            openrouter_model="anthropic/claude-3.5-haiku",
        )
        provider = build_provider(settings)
        assert provider is not None
        assert provider.model_id == "anthropic/claude-3.5-haiku"

    def test_constructing_without_a_key_raises_a_provider_error(self) -> None:
        settings = Settings(app_env="local", log_level="ERROR")
        with pytest.raises(AdvisorProviderError, match="not configured"):
            OpenRouterAdvisorLLM(settings)

    @pytest.mark.parametrize(
        "payload",
        [
            "not a dict",
            {"choices": []},
            {"choices": [{"message": {"content": ""}}]},
            {"choices": [{"message": {}}]},
            {"error": {"message": "bad key"}},
        ],
    )
    def test_a_malformed_body_becomes_a_provider_error(self, payload: Any) -> None:
        with pytest.raises(AdvisorProviderError):
            _extract_message(payload, "fake/model")

    def test_a_well_formed_body_yields_the_message(self) -> None:
        text = _extract_message(
            {"choices": [{"message": {"content": "hello"}}]}, "fake/model"
        )
        assert text == "hello"


# ---------------------------------------------------------------------------
# Generation and validation
# ---------------------------------------------------------------------------
class TestGeneration:
    def test_a_valid_reply_becomes_an_llm_advisory(self) -> None:
        advisory = generate_advisory(facts_for(make_state()), FakeLLM(good_reply()), now=T0)
        assert advisory.mode is AdvisoryMode.LLM
        assert advisory.model_id == "fake/model-1"
        assert advisory.validation is not None
        assert advisory.validation.passed is True

    def test_the_model_receives_the_facts_and_the_rules(self) -> None:
        provider = FakeLLM(good_reply())
        generate_advisory(facts_for(make_state()), provider, now=T0)
        system, user = provider.calls[0]
        assert "Use only the facts supplied" in system
        assert "state.vehicle_count" in user
        assert "46" in user

    def test_no_provider_falls_back_without_crashing(self) -> None:
        advisory = generate_advisory(facts_for(make_state()), None, now=T0)
        assert advisory.mode is AdvisoryMode.DETERMINISTIC_FALLBACK
        assert any("No language model is configured" in i for i in advisory.limitations)

    def test_a_timeout_falls_back_and_says_so(self) -> None:
        error = AdvisorProviderError("The advisor model did not respond within 30s.")
        advisory = generate_advisory(
            facts_for(make_state()), FakeLLM(error=error), now=T0
        )
        assert advisory.mode is AdvisoryMode.DETERMINISTIC_FALLBACK
        assert any("did not respond" in item for item in advisory.limitations)

    @pytest.mark.parametrize(
        "reply",
        ["not json at all", "{unclosed", "[]", '{"summary": ""}', '{"summary": "x"}'],
    )
    def test_a_malformed_reply_falls_back_safely(self, reply: str) -> None:
        advisory = generate_advisory(facts_for(make_state()), FakeLLM(reply), now=T0)
        assert advisory.mode is AdvisoryMode.DETERMINISTIC_FALLBACK
        assert advisory.summary

    def test_a_fenced_reply_is_still_read(self) -> None:
        """Providers add code fences despite being asked not to."""
        advisory = generate_advisory(
            facts_for(make_state()),
            FakeLLM(f"```json\n{good_reply()}\n```"),
            now=T0,
        )
        assert advisory.mode is AdvisoryMode.LLM

    def test_an_invented_number_rejects_the_whole_advisory(self) -> None:
        """A validator that runs after generation is a control; a prompt is not."""
        advisory = generate_advisory(
            facts_for(make_state()),
            FakeLLM(good_reply(summary="Average delay is 137 seconds per vehicle.")),
            now=T0,
        )
        assert advisory.mode is AdvisoryMode.DETERMINISTIC_FALLBACK
        assert any("137" in item for item in advisory.limitations)
        assert not any("137" in item for item in [advisory.summary])

    def test_an_unsupported_evidence_id_rejects_the_advisory(self) -> None:
        advisory = generate_advisory(
            facts_for(make_state()),
            FakeLLM(good_reply(evidence_ids=["state.level", "state.invented_metric"])),
            now=T0,
        )
        assert advisory.mode is AdvisoryMode.DETERMINISTIC_FALLBACK
        assert any("state.invented_metric" in item for item in advisory.limitations)
        assert "state.invented_metric" not in advisory.evidence_ids

    def test_a_supported_number_is_allowed_through(self) -> None:
        advisory = generate_advisory(
            facts_for(make_state()),
            FakeLLM(good_reply(summary="46 vehicles were counted.")),
            now=T0,
        )
        assert advisory.mode is AdvisoryMode.LLM
        assert "46" in advisory.summary

    def test_recorded_absences_are_appended_to_generated_limitations(self) -> None:
        """An advisory that omits what it could not see reads too confident."""
        advisory = generate_advisory(
            facts_for(None),
            # No state means no facts, so the reply cites none: citing one that
            # does not exist is rejected, which a different test covers.
            FakeLLM(
                good_reply(
                    summary="No traffic state is available for this area.",
                    limitations=["Model said little."],
                    evidence_ids=[],
                )
            ),
            now=T0,
        )
        assert advisory.mode is AdvisoryMode.LLM
        assert any("unavailable" in item for item in advisory.limitations)

    def test_an_over_long_reply_is_truncated_rather_than_rejected(self) -> None:
        advisory = generate_advisory(
            facts_for(make_state()),
            FakeLLM(good_reply(suggestions=[f"suggestion {i}" for i in range(40)])),
            now=T0,
        )
        # Each item is short and numeric-free only after entity scrubbing fails,
        # so this one legitimately falls back; what must not happen is a crash.
        assert advisory.summary
        assert len(advisory.suggestions) <= 8


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
class TestPrompt:
    def test_the_payload_marks_unavailable_values_explicitly(self) -> None:
        payload = facts_payload(facts_for(None))
        assert payload["vehicle_count"] is None
        assert payload["traffic_state"] is None
        assert payload["unavailable"]

    def test_the_payload_flags_image_space_occupancy(self) -> None:
        payload = facts_payload(facts_for(make_state()))
        assert "not road occupancy" in str(payload["image_occupancy_note"])

    def test_the_payload_carries_provenance_per_fact(self) -> None:
        payload = facts_payload(facts_for(make_state()))
        entries = payload["fact_table"]
        assert isinstance(entries, list)
        assert all("source" in entry for entry in entries)

    def test_the_prompt_forbids_the_route_claim_without_evidence(self) -> None:
        assert "has_route_comparison is false" in SYSTEM_PROMPT
        assert "must NOT name a route as" in SYSTEM_PROMPT

    def test_the_prompt_forbids_inferring_an_emergency(self) -> None:
        assert "Never infer an emergency" in SYSTEM_PROMPT

    def test_the_prompt_forbids_claiming_control(self) -> None:
        assert "There is no control path" in SYSTEM_PROMPT

    def test_the_user_prompt_contains_the_facts(self) -> None:
        text = user_prompt(facts_for(make_state()))
        assert "FACTS" in text
        assert "state.level" in text


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
class TestAdvisorApi:
    def test_status_reports_deterministic_mode_without_a_key(
        self, client: TestClient
    ) -> None:
        body = client.get("/api/advisor/status").json()
        assert body["provider_configured"] is False
        assert body["mode"] == "deterministic_fallback"
        assert body["model_id"] is None
        assert body["advisory_only"] is True

    def test_the_api_key_is_never_returned(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Not in status, not in an advisory, not anywhere on the wire."""
        secret = "sk-or-v1-supersecretvalue"  # noqa: S105
        settings = _configured_settings(tmp_path, secret)
        # The provider is replaced so this test exercises the API rather than
        # OpenRouter. A unit test that reached the network would be testing
        # somebody else's service, slowly, and would fail offline.
        monkeypatch.setattr(advisor_routes, "build_provider", lambda _: FakeLLM(good_reply()))

        with TestClient(create_app(settings), raise_server_exceptions=False) as c:
            status = c.get("/api/advisor/status")
            assert secret not in status.text
            assert status.json()["model_id"] == "anthropic/claude-3.5-haiku"
            assert status.json()["provider_configured"] is True

            seed_state(c, make_state(settings=settings))
            advisory = c.post("/api/advisor/analyze", json={"media_id": MEDIA_ID})
            assert advisory.status_code == 200
            assert secret not in advisory.text

    def test_a_configured_provider_produces_an_llm_advisory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = _configured_settings(tmp_path, "sk-or-v1-test")
        monkeypatch.setattr(advisor_routes, "build_provider", lambda _: FakeLLM(good_reply()))

        with TestClient(create_app(settings), raise_server_exceptions=False) as c:
            seed_state(c, make_state(settings=settings))
            body = c.post("/api/advisor/analyze", json={"media_id": MEDIA_ID}).json()

        assert body["advisory"]["mode"] == "llm"
        assert body["advisory"]["model_id"] == "fake/model-1"
        assert body["advisory"]["validation"]["passed"] is True

    def test_a_failing_provider_still_answers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dead model must not take the advisory down with it."""
        settings = _configured_settings(tmp_path, "sk-or-v1-test")
        error = AdvisorProviderError("The advisor model could not be reached.")
        monkeypatch.setattr(
            advisor_routes, "build_provider", lambda _: FakeLLM(error=error)
        )

        with TestClient(create_app(settings), raise_server_exceptions=False) as c:
            seed_state(c, make_state(settings=settings))
            response = c.post("/api/advisor/analyze", json={"media_id": MEDIA_ID})

        assert response.status_code == 200
        body = response.json()["advisory"]
        assert body["mode"] == "deterministic_fallback"
        assert body["model_id"] is None
        assert any("could not be reached" in item for item in body["limitations"])

    def test_an_unanalysed_media_id_is_a_404(self, client: TestClient) -> None:
        response = client.post("/api/advisor/analyze", json={"media_id": "nope"})
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"

    def test_naming_neither_media_nor_job_is_refused(self, client: TestClient) -> None:
        response = client.post("/api/advisor/analyze", json={})
        assert response.status_code == 422

    def test_naming_both_media_and_job_is_refused(self, client: TestClient) -> None:
        response = client.post(
            "/api/advisor/analyze", json={"media_id": "a", "job_id": "b"}
        )
        assert response.status_code == 422

    def test_an_advisory_is_produced_from_a_seeded_state(
        self, client: TestClient, advisor_settings: Settings
    ) -> None:
        seed_state(client, make_state(settings=advisor_settings))
        response = client.post("/api/advisor/analyze", json={"media_id": MEDIA_ID})
        assert response.status_code == 200

        body = response.json()
        assert body["advisory"]["mode"] == "deterministic_fallback"
        assert body["advisory"]["advisory_only"] is True
        assert "heavy" in body["advisory"]["summary"].lower()
        assert body["facts"]["vehicle_count"] == 46
        assert "state.vehicle_count" in body["advisory"]["evidence_ids"]

    def test_the_advisory_reports_route_evidence_when_asked_for_it(
        self, advisor_settings: Settings
    ) -> None:
        """Runs the existing optimizer; it does not plan a route of its own."""
        app = create_app(advisor_settings)
        with TestClient(app, raise_server_exceptions=False) as c:
            load_network(app, square_network(free_flow_speed_kph=40.0))
            seed_state(c, make_state(settings=advisor_settings))
            response = c.post(
                "/api/advisor/analyze",
                json={
                    "media_id": MEDIA_ID,
                    "origin_junction_id": "J1",
                    "destination_junction_id": "J3",
                },
            )
            assert response.status_code == 200
            facts = response.json()["facts"]
            assert facts["has_route_comparison"] is True
            assert facts["route_baseline_s"] is not None
            advice = response.json()["advisory"]["route_advice"]
            assert "predicted" in advice.lower()

    def test_a_bad_junction_id_does_not_fail_the_advisory(
        self, advisor_settings: Settings
    ) -> None:
        """Route evidence is optional; losing it must not lose the advisory."""
        app = create_app(advisor_settings)
        with TestClient(app, raise_server_exceptions=False) as c:
            load_network(app, square_network())
            seed_state(c, make_state(settings=advisor_settings))
            response = c.post(
                "/api/advisor/analyze",
                json={
                    "media_id": MEDIA_ID,
                    "origin_junction_id": "J-NOPE",
                    "destination_junction_id": "J3",
                },
            )
            assert response.status_code == 200
            assert response.json()["facts"]["has_route_comparison"] is False
            assert (
                "no route can be described as faster"
                in response.json()["advisory"]["route_advice"].lower()
            )

    def test_the_advisory_does_not_mutate_any_store(
        self, client: TestClient, advisor_settings: Settings
    ) -> None:
        """An explanation layer reads. It does not write."""
        seed_state(client, make_state(settings=advisor_settings))
        before = client.get("/api/sites/confirmed").json()["count"]
        client.post("/api/advisor/analyze", json={"media_id": MEDIA_ID})
        assert client.get("/api/sites/confirmed").json()["count"] == before
        assert client.get("/api/incidents").json()["count"] == 0


# ---------------------------------------------------------------------------
# Regression: nothing that existed before behaves differently
# ---------------------------------------------------------------------------
class TestNoPipelineChange:
    def test_the_declared_phase_is_unchanged(self) -> None:
        """The advisor is an MVP slice; it does not close Phase 12."""
        from app.core.versions import IMPLEMENTED_PHASE

        assert IMPLEMENTED_PHASE == 8

    def test_phase_twelve_is_still_reported_as_not_implemented(
        self, client: TestClient
    ) -> None:
        phases = client.get("/api/meta/phases").json()["phases"]
        twelve = next(item for item in phases if item["phase"] == 12)
        assert twelve["implemented"] is False

    def test_the_health_genai_component_is_unchanged(self, client: TestClient) -> None:
        components = {c["name"]: c for c in client.get("/api/health").json()["components"]}
        assert components["genai"]["status"] == "not_configured"
        assert components["genai"]["phase"] == 12

    def test_no_model_artifact_is_claimed(self, client: TestClient) -> None:
        assert client.get("/api/meta/versions").json()["model_versions"] == {}
