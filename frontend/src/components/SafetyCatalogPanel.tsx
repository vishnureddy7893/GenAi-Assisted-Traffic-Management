/**
 * Safety rule catalog — genuinely implemented in Phase 1.
 *
 * The rules are *policy*: what is unsafe and at which tier. That is a decision,
 * not a measurement, so it can be authored now without fabricating anything.
 *
 * The two tiers behave differently and the panel says so explicitly:
 *
 * - **Hard** constraints prune segments from the search graph before routing.
 *   They are never priced and never traded off against a faster route.
 * - **Soft** constraints add a penalty in equivalent seconds and surface as
 *   warnings.
 *
 * Soft penalty weights are shown as unset, because they are empirical tuning
 * parameters calibrated against simulation in Phase 11. Displaying an invented
 * weight would put a fabricated number at the centre of route selection.
 */

import type { SafetyCatalogResponse, SafetyRule } from "../api/types";
import type { Resource } from "../hooks/useApiResource";
import { Panel } from "./Panel";
import { EmptyState, ResourceView } from "./ResourceState";
import { StatusPill, type Tone } from "./StatusPill";

/**
 * Presentation copy for each tier. UI text only — the rules, their tiers and
 * their membership all come from the API. A tier with no entry here still
 * renders, under its raw name.
 */
const TIER_COPY: Record<string, { heading: string; tone: Tone }> = {
  hard: { heading: "Hard — removed from the search graph", tone: "danger" },
  soft: { heading: "Soft — priced in equivalent seconds", tone: "pending" },
};

function RuleRow({ rule }: { rule: SafetyRule }): JSX.Element {
  return (
    <div className="rule">
      <div className="source__head">
        <span className="rule__name">{rule.name}</span>
        <span className="rule__id">{rule.rule_id}</span>
        <span className="chip">{rule.scope}</span>
        {rule.tier === "soft" && (
          <StatusPill
            tone="pending"
            title="Penalty weights are calibrated against simulation in phase 11"
          >
            penalty uncalibrated
          </StatusPill>
        )}
      </div>
      <p className="rule__rationale">{rule.rationale}</p>
    </div>
  );
}

export function SafetyCatalogPanel({
  catalog,
  onRetry,
}: {
  catalog: Resource<SafetyCatalogResponse>;
  onRetry: () => void;
}): JSX.Element {
  return (
    <Panel
      title="Safety rule catalog"
      subtitle="Hard constraints prune routes; soft constraints price them"
      flush
      aside={
        catalog.state === "success" ? (
          <StatusPill tone="ok" dot>
            v{catalog.data.version}
          </StatusPill>
        ) : undefined
      }
    >
      <div style={{ padding: catalog.state === "success" ? 0 : 16 }}>
        <ResourceView resource={catalog} onRetry={onRetry} loadingRows={4}>
          {(data) => {
            if (data.rules.length === 0) {
              return (
                <EmptyState
                  title="No safety rules defined"
                  message={`Catalog version ${data.version} contains no rules.`}
                />
              );
            }

            // Groups come from the tiers actually present in the response, in
            // the order the backend listed them. A rule at a tier this UI has
            // no copy for still renders, rather than vanishing silently.
            const tiers: string[] = [];
            for (const rule of data.rules) {
              if (!tiers.includes(rule.tier)) tiers.push(rule.tier);
            }

            return (
              <>
                {tiers.map((tier) => {
                  const rules = data.rules.filter((rule) => rule.tier === tier);
                  const copy = TIER_COPY[tier];
                  return (
                    <div key={tier}>
                      <div className="rule-group__title">
                        {copy?.heading ?? tier}
                        <StatusPill tone={copy?.tone ?? "neutral"}>{rules.length}</StatusPill>
                      </div>
                      {rules.map((rule) => (
                        <RuleRow key={rule.rule_id} rule={rule} />
                      ))}
                    </div>
                  );
                })}
              </>
            );
          }}
        </ResourceView>
      </div>
    </Panel>
  );
}
