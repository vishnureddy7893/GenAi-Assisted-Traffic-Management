/**
 * Effective observation mapping — read-only, and deliberately so.
 *
 * Two surveys can bind the same camera to different roads: a versioned mapping
 * file and an operator confirmation. The backend composes them under a fixed
 * precedence rule and this section reports the outcome. It renders no control,
 * because precedence is an architectural decision rather than a per-user
 * preference — a UI that let one operator prefer the file and another prefer
 * confirmations would produce two different mappings from the same data, and
 * the content hash below would stop meaning anything.
 *
 * Three things must stay on screen, and each is easy to lose:
 *
 * 1. **The hash.** It is the answer to "which mapping was this?", and it is the
 *    whole point of composing rather than merging.
 * 2. **Every override.** A confirmation silently replacing a surveyed binding
 *    is exactly what this step exists to make visible.
 * 3. **That costs are unaffected.** The optimizer still reads the file alone.
 *    Rendered from the backend's own flag, never assumed.
 */

import { useCallback, useEffect, useState } from "react";

import { ApiError, apiGet, endpoints } from "../api/client";
import type {
  EffectiveMappingResponse,
  ExcludedConfirmedBinding,
  MappingOverride,
} from "../api/types";
import { StatusPill } from "./StatusPill";

/** Long enough to identify a composition, short enough to read aloud. */
function shortHash(hash: string): string {
  return hash.slice(0, 16);
}

/**
 * Whether a 200 actually carried a composition.
 *
 * This section is additive and embedded inside a panel that works without it.
 * A response of an unexpected shape must therefore render nothing rather than
 * throw, or one surprising payload from a read-only summary would blank the
 * binding controls beside it — the failure mode the whole dashboard is built to
 * avoid.
 */
function isEffectiveMapping(value: unknown): value is EffectiveMappingResponse {
  if (typeof value !== "object" || value === null) return false;
  const effective = (value as { effective?: unknown }).effective;
  if (typeof effective !== "object" || effective === null) return false;
  const candidate = effective as Partial<EffectiveMappingResponse["effective"]>;
  return (
    typeof candidate.effective_mapping_hash === "string" &&
    Array.isArray(candidate.bindings) &&
    Array.isArray(candidate.overrides) &&
    Array.isArray(candidate.excluded)
  );
}

const RESOLUTION_LABEL: Record<MappingOverride["resolution"], string> = {
  file_only: "Mapping file only",
  confirmed_only: "Confirmed binding only",
  sources_agree: "Both sources agree",
  confirmed_binding_precedence: "Human-confirmed binding precedence",
};

const EXCLUSION_LABEL: Record<ExcludedConfirmedBinding["reason"], string> = {
  different_network: "Confirmed against a different network",
  stale_network_version: "Stale network version — requires reconfirmation",
  unknown_network_element: "Names an element this network does not have",
};

function OverrideRow({ entry }: { entry: MappingOverride }): JSX.Element {
  return (
    <li className="effmap__override" data-testid={`override-${entry.observation_area_id}`}>
      <span className="effmap__area">{entry.observation_area_id}</span>
      <span className="effmap__pair">
        File: <code>{entry.file_binding ?? "—"}</code>
      </span>
      <span className="effmap__pair">
        Confirmed: <code>{entry.confirmed_binding ?? "—"}</code>
      </span>
      <span className="effmap__pair effmap__pair--effective">
        Effective: <code>{entry.effective_binding ?? "—"}</code>
      </span>
      <span className="effmap__reason">
        Reason: {RESOLUTION_LABEL[entry.resolution]}
      </span>
    </li>
  );
}

export function EffectiveMappingSummary(): JSX.Element | null {
  const [result, setResult] = useState<EffectiveMappingResponse | null>(null);
  const [error, setError] = useState<ApiError | null>(null);

  const load = useCallback(async (signal: AbortSignal) => {
    try {
      const body = await apiGet<unknown>(endpoints.effectiveMapping, signal);
      setResult(isEffectiveMapping(body) ? body : null);
      setError(null);
    } catch (caught) {
      if (caught instanceof ApiError && caught.code === "timeout") return;
      setResult(null);
      setError(caught instanceof ApiError ? caught : null);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void load(controller.signal);
    return () => controller.abort();
  }, [load]);

  // No network means no composition to report. The road network panel above
  // already says so; repeating it here would be noise.
  if (error !== null && error.code === "network_not_loaded") return null;

  if (error !== null) {
    return (
      <div className="effmap" data-testid="effective-mapping">
        <h3 className="effmap__title">Effective Observation Mapping</h3>
        <p className="bind__error" role="alert" data-testid="effective-mapping-error">
          {error.message}
        </p>
      </div>
    );
  }

  if (result === null) return null;

  const { effective } = result;
  return (
    <div className="effmap" data-testid="effective-mapping">
      <h3 className="effmap__title">
        Effective Observation Mapping
        <StatusPill tone={result.effective_mapping_in_use ? "ok" : "pending"}>
          {result.effective_mapping_in_use
            ? "in use for routing"
            : "not used for route costing"}
        </StatusPill>
      </h3>

      <dl className="effmap__facts">
        <div>
          <dt>Mapping file</dt>
          <dd data-testid="effective-mapping-file">
            {effective.mapping_id === null
              ? "none loaded"
              : `${effective.mapping_id} @ ${effective.mapping_version}`}
          </dd>
        </div>
        <div>
          <dt>Confirmed bindings</dt>
          <dd data-testid="effective-confirmed-count">
            {effective.confirmed_binding_count}
          </dd>
        </div>
        <div>
          <dt>Effective bindings</dt>
          <dd data-testid="effective-binding-count">{effective.bindings.length}</dd>
        </div>
        <div>
          <dt>Effective mapping version</dt>
          <dd data-testid="effective-mapping-hash">
            <code title={effective.effective_mapping_hash}>
              {shortHash(effective.effective_mapping_hash)}
            </code>
          </dd>
        </div>
        <div>
          <dt>Overrides</dt>
          <dd data-testid="effective-override-count">{effective.overrides.length}</dd>
        </div>
      </dl>

      {effective.overrides.length > 0 && (
        <ul className="effmap__list" data-testid="effective-overrides">
          {effective.overrides.map((entry) => (
            <OverrideRow key={entry.observation_area_id} entry={entry} />
          ))}
        </ul>
      )}

      {effective.excluded.length > 0 && (
        <ul className="effmap__list effmap__list--muted" data-testid="effective-excluded">
          {effective.excluded.map((entry) => (
            <li key={`${entry.observation_area_id}-${entry.reason}`}>
              <span className="effmap__area">{entry.observation_area_id}</span>{" "}
              excluded — {EXCLUSION_LABEL[entry.reason]}. {entry.detail}
            </li>
          ))}
        </ul>
      )}

      <p className="effmap__note">{result.note}</p>
    </div>
  );
}
