# ADR 0015 — The nearest junction proposes; a human binds

**Status:** accepted (Phase 8+, step 2 of real-world routing)

## Context

Step 1 gave every observation area a surveyed coordinate (ADR-0014). The loaded
road network already carries junction positions. Computing which junction is
nearest a camera is therefore two lines of arithmetic over data we already have.

The temptation is to stop there and call the result a binding. ADR-0008 and
`app/routing/mapping.py` both say why that is exactly wrong: traffic evidence
reaches a road only through an explicit survey, because a congestion band
attached to the wrong road is a specific false claim that a route would then be
planned around.

Proximity is real evidence. It is also routinely wrong in ways only a person
catches: a camera on a mast between two junctions is nearest to one and pointed
at the other, and a coordinate typed one digit out is nearest to a junction on a
different road entirely.

## Decision

Nearest-junction evaluation and binding creation are **separate operations with
separate HTTP verbs**.

`GET /api/sites/{id}/candidate` returns a `SiteBindingCandidate`: the nearest
positioned junction, its distance, the runners-up, and every reason the ranking
might not mean anything. It carries `ValueSource.DERIVED` provenance naming the
site and the network it was computed from. It writes nothing, and there is no
code path from it to the binding registry.

`POST /api/sites/{id}/binding` is the only operation that creates a binding. It
requires `confirm: true` **and** an explicit `junction_id`. The result is a
`ConfirmedSiteBinding` carrying `ValueSource.OBSERVED` provenance naming the
person, plus a `BindingConfirmation` recording the method, the distance they
were shown, the site accuracy at the time, and whether they took the nearest
junction or overrode it.

`SiteBindingCandidate.requires_human_confirmation` is fixed `True` and a model
validator refuses any attempt to construct one where it is not — the same
device that stops a `SignalAdvisory` declaring itself non-advisory.

## Rationale

**Why two verbs rather than one endpoint with a flag.** A flag can be defaulted
wrongly, and a future caller can set it by accident. A GET that cannot write is
a structural guarantee, not a convention someone has to remember.

**Why the confirmation must name the junction.** A bare "yes" confirms whatever
sorted first at the moment the request landed. Naming the junction makes the
decision about a specific road, and if the map changed underneath the operator
the ids no longer match and the confirmation is refused rather than silently
applied to a different junction.

**Why overriding the nearest junction requires a reason.** It is the case a
reviewer is most likely to question, and an unexplained override is
indistinguishable from a misclick.

**Why uncertainty is carried rather than collapsed.** Two situations make a
nearest-junction ranking meaningless: the distance is inside the site's own
`accuracy_m`, or the runner-up is within that accuracy of the nearest. Either
clears `is_discriminating`, and the candidate says so. The proposal is still
offered — an operator who knows the junction can confirm it regardless — but it
is offered as a question.

**Why `haversine_m` is imported from `app/emergency/zones.py`** rather than
copied or moved. A second distance implementation would eventually disagree with
the first; moving it would edit the exclusion-zone module for no behavioural
reason. The import direction is admittedly odd — a site does not conceptually
depend on emergencies — and it is accepted deliberately in exchange for one
implementation of one formula.

## Consequences

A confirmed binding renders to an ordinary `ObservationBinding` naming
`junction_id`, which has always been a supported form. By existing mapping
semantics `bindings_by_segment()` groups only bindings that name a *segment*, so
a junction binding prices and prunes nothing, and confirming one **cannot change
a route**. That is correct at this step: knowing which intersection a camera
watches is not yet knowing which directed segment its queue belongs to, and
binding a queue to the wrong direction is the failure `app/routing/mapping.py`
exists to prevent. A test pins this.

`ObservationBinding` gains an optional `confirmation` field. Every mapping file
written before this still validates unchanged, and a binding without it means
what it always did: a surveyor wrote it down. The value of the field is that
"surveyed" and "an operator accepted a proximity proposal" are now
distinguishable, since they are not the same strength of evidence.

Confirmed bindings persist to their own file, separate from the sites file, and
are keyed by `(network_id, observation_area_id)`. A junction id means something
different on a different map, so bindings for other networks are retained but
reported only while their map is loaded. A confirmation made against an older
version of the loaded network is reported as stale, never auto-invalidated: a
re-surveyed map usually keeps its junction ids, and whether that holds is the
operator's call.

`GET /api/routes/mapping` reports confirmed bindings alongside the loaded
mapping file rather than merging them. Merging would make a human confirmation
indistinguishable from a surveyed binding in every downstream reader.

Still no mapping provider, no geocoding and no external routing. This step
measures a distance over data already in the process.
