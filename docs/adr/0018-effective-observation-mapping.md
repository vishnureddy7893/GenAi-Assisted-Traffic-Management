# ADR 0018 — The effective observation mapping is composed, not merged

**Status:** accepted (Phase 8+, step 4 of real-world routing)

## Context

There are now two ways to establish which road a camera watches, and both are
legitimate human survey work.

A **mapping file** (ADR-0009, `app/routing/mapping.py`) is a versioned document:
somebody walked the network, decided each binding, and wrote it down under an id
and a version that every route result pins for replay.

A **confirmed site binding** (ADR-0015, ADR-0016) is an operator being shown the
nearest junction and the directed segments meeting it, and confirming one. It is
persisted, per network, and carries its own provenance.

They can disagree. A file written months ago against a camera that has since
been remounted will disagree with the operator who just looked at the footage.
Until now the two simply sat side by side: ADR-0016 recorded that confirmed
bindings were reported *beside* the loaded mapping and deliberately not merged
into it, because merging would need a precedence rule nobody had decided and
would break the replay pin.

This decides that rule.

## Decision

An **`EffectiveObservationMapping`** is composed from both sources on request,
by a fixed rule, and carries a deterministic content hash.

**Precedence.** A human-confirmed binding supersedes a file binding for the same
`observation_area_id`. Where only one source has an opinion, that opinion
stands. Nothing is combined: a binding comes from exactly one source.

**Agreement is recorded as agreement.** Where both name the same target the
result carries the confirmed provenance and resolves as `sources_agree`.
Collapsing it to "confirmed" would discard the only corroboration this system
ever gets — two independent surveys concurring is stronger evidence than either
alone.

Agreement is compared at the level each source speaks at, not field by field. A
file binds a camera to a segment and usually records no junction; a confirmation
records both. Requiring every field to match would report a disagreement every
time a confirmation merely filled in the junction the file omitted. A segment
binding and a junction-only binding never agree: one says which road and the
other does not, and letting the vaguer claim corroborate the specific one would
be exactly backwards.

**Conflicts are reported, never hidden.** Every area where the sources were both
present appears in `overrides` with what each said, what won, and why. A
junction-only confirmation superseding a file's segment binding additionally
raises a warning, because it *withdraws* road-level attribution and somebody
should notice that rather than discover a segment binding had vanished.

**At most one binding per area** — enforced by building the result through the
existing `ObservationMapping`, whose `_unique_areas` validator has always
enforced it. A new check could disagree with the old one; the old one cannot
disagree with itself.

## Identity

`effective_mapping_hash` is SHA-256 over a canonical JSON rendering — sorted
keys, compact separators — of:

- the canonical-form version;
- the mapping file's id, version and content digest;
- the confirmed set's own content fingerprint;
- the network id and version;
- every effective binding, sorted by observation area, with its source and
  resolution;
- every override and every exclusion, sorted.

`effective_mapping_version` is its first 16 hex characters.

No timestamp, uuid or wall-clock value enters it. Two logically identical
compositions hash identically regardless of dictionary ordering, insertion order
or when they were computed.

The precedence *decisions* are hashed alongside the bindings deliberately: two
compositions that bind the same areas to the same roads for different reasons
are not the same mapping, and a reviewer replaying one needs to know which it
was.

The confirmed-set fingerprint hashes what the confirmations **say**, not when
they were said. `confirmed_at` is excluded while `confirmed_by` and `method` are
included, because the latter are evidence — a binding accepted from a bearing
alignment and one surveyed on site are different claims about the same road —
whereas a mapping identity that changed because somebody re-confirmed an
unchanged binding would be useless for the replay question it exists to answer.

The rendering follows the same convention as
`app.simulation.scenario.compute_fingerprint` and is written out separately
rather than imported: routing must not acquire a dependency on the simulator,
and the simulator is out of scope for this change.

## Stale networks

A confirmation made against a different network, a different *version* of the
loaded network, or naming an element the network does not have, is **excluded**
and reported with its reason. Nothing is migrated and nothing is assumed —
segment ids, directions and which arms meet which node are exactly what a
re-survey redraws, so such a binding requires reconfirmation.

The property that matters: an excluded confirmation does **not** supersede the
file binding for its area. The file binding survives, resolved as `file_only`.

## Derived, never stored

There is no effective-mapping registry and no file. It is recomputed from the
loaded mapping and the confirmed-binding registry on every request. It is cheap,
it cannot drift from its inputs, and it cannot become a third source of truth
that somebody edits. Its reproducibility comes from its inputs being persisted,
not from the result being remembered — a test asserts the hash survives a
restart.

## Why it is not connected to route costing

`used_by_route_costing` is fixed `False` with a validator that refuses `True`,
and the API reports `effective_mapping_in_use: false` beside the
`mapping_id`/`mapping_version` the optimizer actually pins.

The optimizer is still handed the mapping file alone. Until it consumes the
effective mapping, recording this hash against a route result would assert the
route was computed from bindings it never saw — the precise dishonesty this
composition exists to make impossible. A regression test pins that composing
changes neither route costs nor ranking.

The identity was built first, on purpose. Connecting bindings to costing without
a way to say *which* bindings were in force would produce recommendations that
cannot be replayed, and a recommendation that cannot be replayed is an anecdote.

## Consequences

`GET /api/sites/effective-mapping` is additive and read-only. `GET
/api/routes/mapping` is unchanged and existing clients keep working.

The composition requires a loaded network and answers `404 network_not_loaded`
without one, because which confirmations are in force — and which are stale — is
decided against the loaded map. An unreadable binding registry answers 502
rather than composing an empty mapping that would hide bindings still on file.

`ObservationBinding` is **not** extended again. `EffectiveBinding` carries the
whole `ConfirmedSiteBinding` alongside the rendered binding, so the network
version, site coordinate, direction, method and confirmer all survive; flattening
them into the older shape would lose provenance for no benefit.

The dashboard gains a read-only section under the site binding panel. It offers
no control for changing precedence: that is an architectural rule, and a UI
preference for it would let two operators derive two different mappings from the
same data, at which point the hash means nothing.


## Addendum — evidence attribution (2026-09-22)

The composition now has its first consumer: `app/routing/attribution.py`
resolves each interpreted observation area to its effective segment and
`/api/state/network` reports the result beside the fused snapshot.

This is attribution only. It answers "does the right camera's evidence land on
the right road?", which a person who knows the junction can check by looking at
it. It does not answer "does that evidence change a route cost", and the two are
kept apart deliberately: wiring both at once would mean a wrong binding and a
wrong penalty arriving together, with no way to tell which produced the
surprising route.

Four outcomes rather than mapped/unmapped, because each needs a different fix:
`attributed`, `junction_only` (intersection confirmed, arm not), `excluded`
(stale or other-network confirmation) and `unmapped` (nothing binds it). Only
the first carries a `segment_id`; a model validator refuses a `segment_id` on
any of the others, so a null attribution cannot read as a binding.

`EvidenceAttributionReport.used_by_route_costing` is fixed `False` with the same
validator pattern, and the route optimizer is unchanged — it still reads the
loaded mapping file alone and still reports a confirmed-but-not-file-mapped area
as unmapped. A test asserts route ranking and costs are identical before and
after attribution, and another asserts the optimizer and the attribution report
disagree about that area on purpose.
