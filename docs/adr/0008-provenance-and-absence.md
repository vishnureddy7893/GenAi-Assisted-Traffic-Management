# ADR 0008 — Provenance is mandatory; absence is None, never zero

**Status:** accepted (Phase 1)

## Context

The system must not fabricate values. Discipline alone does not survive a
deadline, a language model, or a plausible-looking default.

## Decision

Every quantity that could be invented is a `Measurement` requiring a
`Provenance`. `ValueSource` is a closed set — `observed`, `derived`, `predicted`,
`simulated`, `static`, `assumed` — with deliberately no `mock`, `default` or
`placeholder` member. `assumed` requires `notes`; `derived` requires
`input_refs`. A quantity that is not known is `None`.

## Rationale

Zero is a legitimate traffic reading — an empty approach at 04:00 — so using it
to mean "unknown" is how a routing engine concludes that a gridlocked road is
empty and sends an ambulance into it.

Distinguishing observed from derived from predicted from simulated determines
what the system is permitted to *say*. A measured count may be stated as fact; a
prediction must be hedged; a simulated outcome must be labelled as modelled.
Phase 12 relies on exactly this field to keep generated narrative honest.

An unjustified assumption is indistinguishable from a fabrication, and an
underived derivation breaks the audit chain — hence the two extra obligations.

## Consequences

Construction is more verbose. The line is drawn pragmatically: identifiers,
enums and structural map attributes stay plain, continuous quantities carry
measurements, and topology-level provenance sits on `RoadNetwork` itself.
