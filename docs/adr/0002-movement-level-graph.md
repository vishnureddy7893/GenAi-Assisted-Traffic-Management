# ADR 0002 — Approaches and turn movements are first-class

**Status:** accepted (Phase 1)

## Context

The simplest network model is junction-as-node, road-as-edge.

## Decision

Model `Approach` and `TurnMovement` as first-class types, with
`conflicting_movement_ids` reserved for signal conflict checking.

## Rationale

A bare graph node cannot express either fact this system exists to reason about:
"approach 1 at J11 is congested" is an approach-level fact, and "you may not turn
right here" is a movement-level fact. Both are central to emergency routing.
Movement-expanded graphs are the standard formulation for turn-aware and
turn-penalised routing.

Deferring this would not have been a refactor. Every schema downstream —
observation, state, routing, safety scope — keys off these identifiers, so
retrofitting them at Phase 7 would be a rewrite of the whole model layer.

## Consequences

Network construction is more verbose, and referential integrity between
approaches, movements and segments is validated at load rather than trusted.

Banned turns are recorded explicitly (`is_permitted=False`) rather than omitted,
because omission is indistinguishable from an incomplete map import — and
"unknown" must never be read as "allowed".
