/**
 * Traffic evidence — which road each camera's evidence reaches.
 *
 * Answers one question and only one: does the right camera's evidence land on
 * the right road? A person who knows the junction can check every row here by
 * looking at it, which is exactly why this is separated from whether that
 * evidence changes a route cost. The panel says so rather than letting a reader
 * assume, because a chain that renders `Camera-A → SEG-14 → Heavy` looks very
 * much like something that is already steering traffic.
 *
 * Unmapped is shown as four distinct outcomes, not one:
 *
 * - **attributed** — a directed segment is in force;
 * - **junction only** — the intersection is confirmed, the arm is not;
 * - **excluded** — a confirmation exists and is stale or for another map;
 * - **unmapped** — nothing binds this area at all.
 *
 * Each needs a different fix, so collapsing them would tell an operator that
 * four problems are one problem. Every row carries the backend's own `reason`
 * verbatim.
 *
 * There is no control here that creates or edits a mapping. Confirmation
 * belongs to the site binding panel, where a person decides; this panel only
 * reports what that decision produced.
 */

import { useCallback, useEffect, useState } from "react";

import { ApiError, apiGet, endpoints } from "../api/client";
import type {
  AttributionStatus,
  EvidenceAttribution,
  NetworkTrafficStateResponse,
} from "../api/types";
import { Panel } from "./Panel";
import { EmptyState } from "./ResourceState";
import { StatusPill, type Tone } from "./StatusPill";

const STATUS_TONE: Record<AttributionStatus, Tone> = {
  attributed: "ok",
  junction_only: "pending",
  excluded: "danger",
  unmapped: "pending",
};

const STATUS_LABEL: Record<AttributionStatus, string> = {
  attributed: "attributed",
  junction_only: "junction only",
  excluded: "excluded",
  unmapped: "unmapped",
};

const SOURCE_LABEL: Record<string, string> = {
  mapping_file: "Mapping file",
  confirmed_binding: "Human-confirmed binding",
};

const RESOLUTION_LABEL: Record<string, string> = {
  file_only: "mapping file only",
  confirmed_only: "confirmed binding only",
  sources_agree: "both sources agree",
  confirmed_binding_precedence: "confirmed binding superseded the file",
};

/**
 * One camera's chain: area → road → what was observed there.
 *
 * The road step renders "Unmapped" rather than a dash when there is none. A
 * dash reads as "nothing to report"; this is a specific, reportable state.
 */
function AttributionRow({ entry }: { entry: EvidenceAttribution }): JSX.Element {
  const attributed = entry.status === "attributed";
  return (
    <li className="attr" data-testid={`attribution-${entry.observation_area_id}`}>
      <div className="attr__chain">
        <span className="attr__area">{entry.observation_area_id}</span>
        <span className="attr__arrow" aria-hidden="true">
          ↓
        </span>
        <span className={attributed ? "attr__road" : "attr__road attr__road--none"}>
          {attributed ? entry.segment_id : "Segment: Unmapped"}
          {attributed && entry.segment_direction !== null && (
            <span className="attr__direction"> ({entry.segment_direction})</span>
          )}
        </span>
        {entry.congestion_level !== null && (
          <>
            <span className="attr__arrow" aria-hidden="true">
              ↓
            </span>
            <span className="attr__level">
              {entry.congestion_level.replace(/_/g, " ")}
            </span>
          </>
        )}
      </div>

      <div className="attr__meta">
        <StatusPill tone={STATUS_TONE[entry.status]} dot>
          {STATUS_LABEL[entry.status]}
        </StatusPill>
        {entry.junction_id !== null && (
          <span className="attr__fact">junction {entry.junction_id}</span>
        )}
        {entry.approach_id !== null && (
          <span className="attr__fact">approach {entry.approach_id}</span>
        )}
        {entry.source !== null && (
          <span className="attr__fact">
            Source: {SOURCE_LABEL[entry.source] ?? entry.source}
            {entry.resolution !== null &&
              ` — ${RESOLUTION_LABEL[entry.resolution] ?? entry.resolution}`}
          </span>
        )}
      </div>

      <p className="attr__reason">{entry.reason}</p>
    </li>
  );
}

export function EvidenceAttributionPanel(): JSX.Element {
  const [body, setBody] = useState<NetworkTrafficStateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (signal: AbortSignal) => {
    try {
      const next = await apiGet<NetworkTrafficStateResponse>(
        endpoints.networkTrafficState,
        signal,
      );
      setBody(next);
      setError(null);
    } catch (caught) {
      if (caught instanceof ApiError && caught.code === "timeout") return;
      setBody(null);
      setError(
        caught instanceof ApiError ? caught.message : "Could not read traffic state.",
      );
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void load(controller.signal);
    return () => controller.abort();
  }, [load]);

  const attribution = body?.attribution ?? null;

  return (
    <Panel
      title="Traffic evidence"
      subtitle="Which road each camera's evidence reaches, through the effective mapping"
      aside={
        attribution !== null ? (
          <StatusPill tone="pending">not used for route costing</StatusPill>
        ) : undefined
      }
    >
      {error !== null ? (
        <p className="bind__error" role="alert" data-testid="attribution-error">
          {error}
        </p>
      ) : body === null ? null : attribution === null ? (
        <EmptyState
          title="Evidence cannot be placed on the network"
          message={
            body.attribution_unavailable_reason ??
            "No attribution is available for this snapshot."
          }
        />
      ) : attribution.attributions.length === 0 ? (
        <EmptyState
          title="No observation area has been interpreted"
          message="Run vision analysis and traffic-state interpretation on some media; there is no evidence to place yet."
        />
      ) : (
        <>
          <dl className="attr__summary">
            <div>
              <dt>Attributed</dt>
              <dd data-testid="attribution-attributed-count">
                {attribution.attributed_area_ids.length}
              </dd>
            </div>
            <div>
              <dt>Unmapped</dt>
              <dd data-testid="attribution-unmapped-count">
                {attribution.unmapped_observation_area_ids.length}
              </dd>
            </div>
            <div>
              <dt>Segments receiving evidence</dt>
              <dd data-testid="attribution-segment-count">
                {attribution.attributed_segment_ids.length}
              </dd>
            </div>
            <div>
              <dt>Effective mapping</dt>
              <dd data-testid="attribution-mapping-version">
                <code title={attribution.effective_mapping_hash}>
                  {attribution.effective_mapping_version}
                </code>
              </dd>
            </div>
          </dl>

          <ul className="attr__list" data-testid="attribution-list">
            {attribution.attributions.map((entry) => (
              <AttributionRow key={entry.observation_area_id} entry={entry} />
            ))}
          </ul>

          <p className="attr__note">{attribution.note}</p>
        </>
      )}
    </Panel>
  );
}
