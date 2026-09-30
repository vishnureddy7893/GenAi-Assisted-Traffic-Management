# ADR 0017 — The LLM is an advisory layer over deterministic facts

**Status:** accepted (MVP slice of Phase 12, landed ahead of the plan)

## Context

The phase plan puts GenAI explanation at Phase 12, behind ML models and the
decision engine. A project review needed a working advisor before those exist.

The risk of pulling it forward is specific. Every other number this system shows
was computed by code that can be read, tested and argued with. An LLM produces
numbers that look identical and are not, and it does so most confidently in
exactly the places a reader is least likely to check — a plausible round figure
in the middle of a fluent paragraph.

Phase 1 anticipated this. `app/domain/decision.py` already defined `Fact`,
`FactTable`, `GenAIRecommendation`, `NarrativeValidation` and
`scan_for_unsupported_numbers`, and ADR-0009 set out why: *"A validator that runs
after generation is a control; a sentence in a prompt is not."* This step uses
that machinery rather than building beside it.

## Decision

The advisor is a **read-only explanation layer**. It runs no detection, computes
no analytics, classifies no congestion and plans no route. It reads results the
pipeline already produced, renders them as facts, asks a model to explain them,
and validates the reply before showing it.

`facts → prompt → model → parse → validate → advisory`, with a deterministic
advisory available at every step after the first.

**What the LLM receives.** A `TrafficAdvisorFacts` payload: the interpreted
traffic state, vehicle and class counts, image-space occupancy, the surveyed
site, the confirmed junction and segment, declared incidents and emergencies,
and — only where they exist — a route comparison and a simulation comparison.
Every number is pre-rendered in the fact table including unit conversions, so
the model never performs arithmetic.

**What it may recommend.** Monitoring, re-analysis, evaluating alternate routes,
checking declared incidents, escalation. Advisory actions for a human operator.

**What it cannot decide.** Anything. It has no control path. `TrafficAdvisory`
carries `advisory_only: bool = True` with a validator that refuses `False`, the
same device that stops a `SignalAdvisory` declaring itself non-advisory.

**Provenance is preserved end to end.** Each fact carries the existing
`Provenance`, so `observed`, `derived`, `predicted`, `simulated` and `assumed`
mean here what they mean everywhere else. Where a source object has its own
provenance — a surveyed site, a simulation run — it is reused verbatim. The
prompt requires the model to phrase each value according to its source: a
simulated figure is "the simulation models …", never "the road will take …".

**Absence is a fact.** A missing value is never defaulted to zero and never
dropped. It becomes an entry in `unavailable` with the reason it is missing, and
that list reaches both the model and the operator. "Vehicle count is unavailable
because no analysis has been run" is actionable; a vehicle count of 0 is a lie.

**The route rule.** With no route comparison in the facts, no route may be
called faster. The advisory says an alternate route should be evaluated — a
request for evidence rather than a claim about it. Where a comparison exists,
the durations are quoted and labelled predicted.

## The validation gate

A generated advisory is scanned before it is shown:

* **numbers** — `scan_for_unsupported_numbers` finds numeric literals in the
  prose that no fact supports;
* **evidence ids** — any cited id absent from the fact table is a claim to have
  used evidence that does not exist.

Either one **rejects the whole advisory**. Not repairs it: stripping an invented
number leaves a sentence built around it, and dropping a bad citation leaves
prose that still rests on it. The deterministic advisory is shown instead, with
the precise reason for the rejection in its limitations.

## Fallback

Three situations produce a deterministic advisory, and it is the same code path
for all three: no provider configured, the provider failed, or the reply was
rejected. It is composed from the facts by fixed rules and **never claims to be
a model's work** — `mode` is `deterministic_fallback`, `model_id` is `None`, and
the schema refuses a fallback advisory that names a model.

This makes the offline demo a first-class path rather than a degraded one. With
no API key the system still explains its facts; it simply says who wrote the
words.

## Configuration

OpenRouter, reused rather than newly chosen: `OPENROUTER_API_KEY`,
`OPENROUTER_MODEL`, `OPENROUTER_BASE_URL`, `OPENROUTER_TIMEOUT_S` and
`OPENROUTER_MAX_TOKENS` have been in `Settings` since Phase 1 in anticipation of
Phase 12, and `/api/health` already reported whether credentials were present.

The provider sits behind the `TrafficAdvisorLLM` protocol, so Anthropic, Gemini,
Groq or a local model is a new class in `app/advisor/provider.py` and no change
anywhere else. The key is read server-side, unwrapped into one HTTP header, and
appears in no response model; a test asserts it is absent from both the status
and the advisory payloads.

## Human in the loop

The advisory is text on a panel. Nothing acts on it. Every suggestion is phrased
for an operator to accept or ignore, the panel carries an "Advisory only" badge
and the sentence *"This system has no control path to any traffic signal,
vehicle or infrastructure"*, and the backend states the same thing in the
schema rather than only in prose.

## Consequences

`IMPLEMENTED_PHASE` stays at **8**, and `/api/meta/phases` continues to report
Phase 12 as not implemented. This is an MVP slice: the advisor exists, the
decision record, model artifacts and the full narrative pipeline that Phase 12
also covers do not. Claiming the phase because part of it works would be the
drift the phase table exists to prevent. Tests pin both facts.

The advisor adds **no runtime dependency**. One JSON POST does not justify one,
so `urllib.request` makes the call with an explicit timeout from a synchronous
handler that FastAPI runs in a worker thread — the same pattern as every other
handler here.

Route evidence, when requested, comes from running the **existing** optimizer
through `prepare_routing` and `optimize`. The advisor is a caller; nothing about
routing changes, and the advisory describes the same ranking the routing panel
would show.
