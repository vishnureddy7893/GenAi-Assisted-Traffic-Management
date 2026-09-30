/**
 * AI Traffic Advisor — an explanation of facts, not a source of them.
 *
 * Everything this panel shows was computed by the deterministic pipeline before
 * the advisor ran. The advisory rephrases and prioritises it; it never supplies
 * a number of its own, and the backend rejects any generated text that tries to.
 *
 * Two things must stay visible on screen, because both are easy to lose in a
 * pleasant paragraph:
 *
 * 1. **Which mode wrote this.** An LLM advisory and a deterministic one look
 *    alike and are not alike. The mode pill is rendered from the backend's own
 *    `mode` field and is never inferred from whether text looks fluent.
 * 2. **What it could not see.** Limitations and the evidence list are rendered
 *    as prominently as the suggestions. An advisory that reads confidently
 *    while omitting that it had no route data is the failure mode here.
 *
 * The panel is advisory-only and says so, in the same words the backend uses.
 */

import { useCallback, useEffect, useState } from "react";

import { ApiError, apiGet, apiPost, endpoints } from "../api/client";
import type {
  AdvisorResponse,
  AdvisorStatusResponse,
  TrafficAdvisorFacts,
  TrafficAdvisory,
} from "../api/types";
import { Panel } from "./Panel";
import { EmptyState } from "./ResourceState";
import { StatusPill, type Tone } from "./StatusPill";

function ModePill({ mode }: { mode: TrafficAdvisory["mode"] }): JSX.Element {
  const llm = mode === "llm";
  return (
    <StatusPill tone={llm ? "accent" : "pending"} dot>
      {llm ? "LLM generated" : "Deterministic advisory mode"}
    </StatusPill>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}): JSX.Element {
  return (
    <div className="adv__section">
      <h3 className="adv__heading">{title}</h3>
      {children}
    </div>
  );
}

function Bullets({
  items,
  testId,
}: {
  items: string[];
  testId: string;
}): JSX.Element {
  return (
    <ul className="adv__list" data-testid={testId}>
      {items.map((item) => (
        <li key={item}>{item}</li>
      ))}
    </ul>
  );
}

/**
 * The facts behind the advisory, shown so a reader can check every sentence.
 *
 * A missing value renders as "unavailable" with its reason, never as a dash or
 * a zero — the whole point of the fact contract is that absence stays legible.
 */
function Evidence({ facts }: { facts: TrafficAdvisorFacts }): JSX.Element {
  return (
    <div className="adv__evidence" data-testid="advisor-evidence">
      <dl className="adv__facts">
        {facts.fact_table.facts.map((fact) => (
          <div key={fact.fact_id} className="adv__fact">
            <dt>
              {fact.label}
              <StatusPill tone={fact.provenance.source === "observed" ? "ok" : "neutral"}>
                {fact.provenance.source}
              </StatusPill>
            </dt>
            <dd>
              {String(fact.value)}
              {fact.unit !== null && ` ${fact.unit}`}
            </dd>
          </div>
        ))}
      </dl>
      {facts.unavailable.length > 0 && (
        <ul className="adv__list adv__list--muted" data-testid="advisor-unavailable">
          {facts.unavailable.map((item) => (
            <li key={item.fact_id}>
              <strong>{item.label}:</strong> unavailable — {item.reason}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function AdvisorPanel({ mediaId }: { mediaId: string | null }): JSX.Element {
  const [status, setStatus] = useState<AdvisorStatusResponse | null>(null);
  const [result, setResult] = useState<AdvisorResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const body = await apiGet<AdvisorStatusResponse>(endpoints.advisorStatus);
        if (!cancelled) setStatus(body);
      } catch {
        // A failed status read is not worth an error banner: the advisory
        // request reports its own failure, and the mode is shown from the
        // advisory itself once one exists.
        if (!cancelled) setStatus(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // A new upload invalidates the advisory: showing last image's assessment
  // beside this image's frame would be the worst kind of stale.
  useEffect(() => {
    setResult(null);
    setError(null);
  }, [mediaId]);

  const generate = useCallback(async () => {
    if (mediaId === null) return;
    setBusy(true);
    setError(null);
    try {
      const body = await apiPost<AdvisorResponse>(endpoints.advisorAnalyze, {
        media_id: mediaId,
      });
      setResult(body);
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "Advisory failed.");
    } finally {
      setBusy(false);
    }
  }, [mediaId]);

  const advisory = result?.advisory ?? null;
  const statusTone: Tone = status?.provider_configured ? "accent" : "pending";

  return (
    <Panel
      title="AI Traffic Advisor"
      subtitle="Explanation and suggestions over the facts this system already computed"
      aside={
        advisory !== null ? (
          <ModePill mode={advisory.mode} />
        ) : status !== null ? (
          <StatusPill tone={statusTone} dot>
            {status.provider_configured
              ? "LLM configured"
              : "Deterministic advisory mode"}
          </StatusPill>
        ) : undefined
      }
    >
      {mediaId === null ? (
        <EmptyState
          title="No media selected"
          message="Upload a traffic image and run analysis, then generate an advisory over the resulting facts."
        />
      ) : (
        <>
          <div className="adv__actions">
            <button
              type="button"
              className="btn"
              disabled={busy}
              onClick={() => void generate()}
            >
              {busy ? "Generating…" : "Generate advisory"}
            </button>
            {status !== null && (
              <span className="adv__mode-note">{status.note}</span>
            )}
          </div>

          {busy && (
            <p className="adv__loading" role="status" data-testid="advisor-loading">
              Generating an advisory from the recorded facts…
            </p>
          )}

          {error !== null && (
            <p className="bind__error" role="alert" data-testid="advisor-error">
              {error}
            </p>
          )}

          {advisory !== null && result !== null && (
            <div className="adv" data-testid="advisor-result">
              <Section title="Traffic Summary">
                <p className="adv__text">{advisory.summary}</p>
                <p className="adv__text">{advisory.traffic_assessment}</p>
              </Section>

              {advisory.suggestions.length > 0 && (
                <Section title="Suggestions">
                  <Bullets items={advisory.suggestions} testId="advisor-suggestions" />
                </Section>
              )}

              {advisory.safety_considerations.length > 0 && (
                <Section title="Safety Considerations">
                  <Bullets items={advisory.safety_considerations} testId="advisor-safety" />
                </Section>
              )}

              <Section title="Route Advice">
                <p className="adv__text" data-testid="advisor-route">
                  {advisory.route_advice}
                </p>
              </Section>

              {advisory.limitations.length > 0 && (
                <Section title="Limitations">
                  <Bullets items={advisory.limitations} testId="advisor-limitations" />
                </Section>
              )}

              <Section title="Evidence">
                <Evidence facts={result.facts} />
              </Section>

              <p className="adv__footer">
                <StatusPill tone="pending">Advisory only</StatusPill>
                <span>
                  {advisory.mode === "llm"
                    ? `Written by ${advisory.model_id ?? "a language model"} and validated against the facts above.`
                    : "Composed from the facts above by fixed rules. No language model was involved."}{" "}
                  This system has no control path to any traffic signal, vehicle
                  or infrastructure.
                </span>
              </p>
            </div>
          )}
        </>
      )}
    </Panel>
  );
}
