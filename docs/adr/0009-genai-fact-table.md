# ADR 0009 — LLM output is validated against a fact table

**Status:** accepted (Phase 1 — schemas); wired in Phase 12

## Context

Phase 12 hands structured state to a pretrained LLM via OpenRouter and asks for
a narrative recommendation.

## Decision

Build a `FactTable` containing every quantity the narrative is permitted to
mention, each with an id, a value and provenance. Require generated text to cite
fact ids, and run `scan_for_unsupported_numbers()` after generation to verify
mechanically that no numeric literal in the prose lacks backing. `DecisionRecord`
refuses to carry generated narrative without its validation result.

## Rationale

"Do not invent numbers" in a prompt is a request, not a control. Models comply
unevenly and fail most often by producing plausible round numbers exactly where
a reader is least likely to check them. A validator that runs after generation is
a control; a sentence in a prompt is not.

`Fact.display_variants` carries pre-rendered forms ("187 s", "3.1 min") so the
model never performs a unit conversion. That is an arithmetic step executed by a
component with no arithmetic guarantees, and it is unnecessary — we already know
the answer.

## Consequences

The scan is a lint, not a proof: it catches invented quantities, not invented
causal claims. Those are constrained separately, by supplying only structured
state and requiring the stated reason to reference the cost breakdown that
routing actually produced.

`entity_tokens` excludes identifiers such as `J11` and `Lane 1` from the numeric
scan, so an identifier is not mistaken for a measurement. The scan is
deliberately over-eager on small integers: a noisy lint is cheap, an invented ETA
is not.
