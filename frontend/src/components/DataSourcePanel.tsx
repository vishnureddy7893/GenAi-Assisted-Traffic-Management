/**
 * Traffic data input panel.
 *
 * Every value shown comes from `/api/sources/catalog`. This file contains no
 * list of source kinds, no capability table and no mode letters used as
 * meaning:
 *
 * - The mode tabs are the groups the backend returned.
 * - The default tab is whichever group the backend marks `is_primary`, not a
 *   hard-coded "A".
 * - Capability chips are derived from the boolean fields actually present on
 *   each profile, so a capability added to the backend schema appears here
 *   without a frontend change.
 * - The "nothing available" callout is driven by the profiles' own
 *   `implemented` flags, not by an assumption about which mode is live.
 *
 * No source is selectable while none is implemented: the controls are disabled
 * and labelled with the phase that will deliver them, rather than enabled
 * against a stub that would silently produce nothing.
 */

import { useMemo, useState } from "react";
import type { SourceCatalogResponse, SourceKindProfile, SourceModeGroup } from "../api/types";
import type { Resource } from "../hooks/useApiResource";
import { Panel } from "./Panel";
import { EmptyState, ResourceView } from "./ResourceState";
import { StatusPill } from "./StatusPill";

/** Profile fields that are not capabilities, so must not become chips. */
const NON_CAPABILITY_BOOLEANS = new Set(["implemented"]);

function humanize(key: string): string {
  return key.replace(/_/g, " ");
}

/**
 * Capability chips, derived from the profile object rather than a fixed list.
 *
 * Reading the booleans off the response means the chips describe what the
 * backend actually declared. A hard-coded array would silently omit any
 * capability added later, and would keep rendering one that was removed.
 */
function CapabilityChips({ profile }: { profile: SourceKindProfile }): JSX.Element {
  const capabilities = Object.entries(profile).filter(
    (entry): entry is [string, boolean] =>
      typeof entry[1] === "boolean" && !NON_CAPABILITY_BOOLEANS.has(entry[0]),
  );

  return (
    <div className="chips">
      {capabilities.map(([name, enabled]) => (
        <span
          key={name}
          className={enabled ? "chip chip--on" : "chip"}
          title={`${humanize(name)}: ${String(enabled)}`}
        >
          {enabled ? humanize(name) : `no ${humanize(name)}`}
        </span>
      ))}
      <span className="chip" title="Payload carried by each sample">
        {profile.payload_kind}
      </span>
    </div>
  );
}

function SourceRow({ profile }: { profile: SourceKindProfile }): JSX.Element {
  return (
    <div className="source" data-testid={`source-${profile.kind}`}>
      <div className="source__head">
        <span className="source__name">{profile.display_name}</span>
        {profile.implemented ? (
          <StatusPill tone="ok" dot>
            Available
          </StatusPill>
        ) : (
          <StatusPill
            tone="pending"
            title={`Implemented in phase ${profile.available_from_phase}`}
          >
            Phase {profile.available_from_phase}
          </StatusPill>
        )}
        {profile.required_dependencies.length > 0 && (
          <StatusPill title="External libraries this input path requires">
            needs {profile.required_dependencies.join(", ")}
          </StatusPill>
        )}
      </div>

      <p className="source__desc">{profile.description}</p>
      <CapabilityChips profile={profile} />

      <div className="source__actions">
        <button
          type="button"
          className="button"
          disabled={!profile.implemented}
          title={
            profile.implemented
              ? undefined
              : `Not implemented yet — phase ${profile.available_from_phase}`
          }
        >
          {profile.realtime ? "Connect stream" : "Load recording"}
        </button>
        {!profile.implemented && (
          <span className="source__note">
            Disabled: no implementation is registered for this source kind.
          </span>
        )}
      </div>
    </div>
  );
}

/**
 * Callout shown when nothing in a group can currently be opened.
 *
 * Driven by the group's own profiles rather than by an assumption that one
 * particular mode is the unavailable one.
 */
function GroupUnavailableBanner({ group }: { group: SourceModeGroup }): JSX.Element {
  const phases = group.profiles.map((p) => p.available_from_phase);
  const earliest = phases.length > 0 ? Math.min(...phases) : null;

  return (
    <div style={{ padding: "14px 16px", borderBottom: "1px solid var(--border)" }}>
      <div className="banner">
        <span className="banner__icon" aria-hidden="true">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M3 3l18 18" strokeLinecap="round" />
            <path d="M8.5 12.5a5 5 0 0 1 1.6-1.1M5 9a9 9 0 0 1 2.6-1.8M12 16.5h.01" strokeLinecap="round" />
          </svg>
        </span>
        <span>
          <strong>No {group.title.toLowerCase()} source is connected.</strong> None of
          the {group.profiles.length} source kinds in this group has a registered
          implementation
          {earliest !== null ? `, the earliest arriving in phase ${earliest}` : ""}. No
          data is being received through this input mode.
        </span>
      </div>
    </div>
  );
}

function ModeContent({ group }: { group: SourceModeGroup }): JSX.Element {
  const noneAvailable = group.profiles.every((profile) => !profile.implemented);

  return (
    <>
      <div className="mode-intro">{group.description}</div>
      {noneAvailable && group.profiles.length > 0 && <GroupUnavailableBanner group={group} />}
      <div className="source-list">
        {group.profiles.map((profile) => (
          <SourceRow key={profile.kind} profile={profile} />
        ))}
      </div>
    </>
  );
}

export function DataSourcePanel({
  catalog,
  onRetry,
}: {
  catalog: Resource<SourceCatalogResponse>;
  onRetry: () => void;
}): JSX.Element {
  // Null until the operator picks a tab; the effective default comes from the
  // backend's own `is_primary` flag rather than a hard-coded mode letter.
  const [selectedMode, setSelectedMode] = useState<string | null>(null);

  const data = catalog.state === "success" ? catalog.data : null;

  const availability = useMemo(() => {
    if (data === null) return null;
    return {
      any: data.any_source_available,
      implemented: data.implemented_kinds.length,
      total: data.implemented_kinds.length + data.unavailable_kinds.length,
    };
  }, [data]);

  return (
    <Panel
      title="Traffic data input"
      subtitle="Recorded and live sources converge on the same decision pipeline"
      flush
      aside={
        availability !== null ? (
          <StatusPill tone={availability.any ? "ok" : "pending"} dot={availability.any}>
            {availability.implemented}/{availability.total} source kinds available
          </StatusPill>
        ) : undefined
      }
    >
      <ResourceView resource={catalog} onRetry={onRetry} loadingRows={4}>
        {(response) => {
          if (response.modes.length === 0) {
            return (
              <EmptyState
                title="No input modes defined"
                message="The backend returned no source mode groups, so there is nothing to configure."
              />
            );
          }

          const primary = response.modes.find((group) => group.is_primary);
          const fallback = primary ?? response.modes[0];
          const active =
            response.modes.find((group) => group.mode === selectedMode) ?? fallback;

          return (
            <>
              <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--border)" }}>
                <div className="tabs" role="tablist" aria-label="Traffic data input mode">
                  {response.modes.map((group) => (
                    <button
                      key={group.mode}
                      type="button"
                      role="tab"
                      id={`mode-tab-${group.mode}`}
                      aria-selected={active !== undefined && group.mode === active.mode}
                      aria-controls={`mode-panel-${group.mode}`}
                      className="tab"
                      onClick={() => setSelectedMode(group.mode)}
                    >
                      {group.title}
                      {group.is_primary && <StatusPill tone="accent">Primary</StatusPill>}
                    </button>
                  ))}
                </div>
              </div>

              {response.note !== "" && <div className="mode-intro">{response.note}</div>}

              {active !== undefined && (
                <div
                  role="tabpanel"
                  id={`mode-panel-${active.mode}`}
                  aria-labelledby={`mode-tab-${active.mode}`}
                >
                  <ModeContent group={active} />
                </div>
              )}
            </>
          );
        }}
      </ResourceView>
    </Panel>
  );
}
