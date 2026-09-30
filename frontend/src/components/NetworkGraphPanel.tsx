/**
 * Multi-junction network visualization.
 *
 * Renders whatever `/api/network` returns and nothing else. There is no table
 * of junction ids in this file, no fixed coordinates and no example topology:
 * every node and edge drawn exists because the backend sent it, and positions
 * come from {@link layoutNetwork}, which derives them from the response.
 *
 * When no network is loaded the panel says so plainly. It does not fall back to
 * an illustrative diagram — a drawing of a network that is not loaded is
 * indistinguishable, at a glance, from a network that is.
 *
 * Edges are drawn without congestion colouring because no per-segment traffic
 * state exists. Phase 5 interprets congestion for an *observation area* — one
 * camera's field of view — and Phase 6 loaded the graph, but nothing binds a
 * camera to a segment: that is a survey fact this system has not recorded.
 * Colour here would have to come from somewhere, and the only available
 * somewhere is invention — a band on the wrong road is worse than no band.
 */

import type { RoadNetwork } from "../api/types";
import type { LayoutEdge, LayoutNode, NetworkLayout } from "../lib/networkLayout";
import { layoutNetwork } from "../lib/networkLayout";
import type { Resource } from "../hooks/useApiResource";
import { Panel } from "./Panel";
import { EmptyState, ResourceView } from "./ResourceState";
import { StatusPill } from "./StatusPill";

const JUNCTION_RADIUS = 17;

function EdgeLine({ edge }: { edge: LayoutEdge }): JSX.Element {
  return (
    <line
      className={
        edge.isFacilityAccess ? "netgraph__edge netgraph__edge--access" : "netgraph__edge"
      }
      x1={edge.x1}
      y1={edge.y1}
      x2={edge.x2}
      y2={edge.y2}
    >
      <title>
        {edge.label ?? edge.id} ({edge.fromId} → {edge.toId})
      </title>
    </line>
  );
}

function NodeShape({ node }: { node: LayoutNode }): JSX.Element {
  if (node.kind === "facility") {
    const halfWidth = Math.max(42, node.label.length * 3.6);
    return (
      <g data-testid={`network-node-${node.id}`}>
        <rect
          className="netgraph__facility"
          x={node.x - halfWidth}
          y={node.y - 15}
          width={halfWidth * 2}
          height={30}
          rx={7}
        />
        <text className="netgraph__node-label" x={node.x} y={node.y}>
          {node.label}
        </text>
        <title>
          {node.label} ({node.category})
        </title>
      </g>
    );
  }

  return (
    <g data-testid={`network-node-${node.id}`}>
      <circle className="netgraph__node" cx={node.x} cy={node.y} r={JUNCTION_RADIUS} />
      <text className="netgraph__node-label" x={node.x} y={node.y}>
        {node.label}
      </text>
      <title>
        {node.label} ({node.category})
      </title>
    </g>
  );
}

function NetworkSvg({ layout }: { layout: NetworkLayout }): JSX.Element {
  const junctionCount = layout.nodes.filter((n) => n.kind === "junction").length;
  const facilityCount = layout.nodes.length - junctionCount;

  return (
    <div className="netgraph">
      <svg
        viewBox={`0 0 ${layout.width} ${layout.height}`}
        role="img"
        aria-label={
          `Road network: ${junctionCount} junction(s), ${layout.edges.length} link(s)` +
          (facilityCount > 0 ? `, ${facilityCount} facility(ies)` : "") +
          `. Layout strategy: ${layout.strategy}.`
        }
      >
        <g>
          {layout.edges.map((edge) => (
            <EdgeLine key={edge.id} edge={edge} />
          ))}
        </g>
        <g>
          {layout.nodes.map((node) => (
            <NodeShape key={node.id} node={node} />
          ))}
        </g>
      </svg>
    </div>
  );
}

export function NetworkGraphPanel({
  network,
  onRetry,
}: {
  network: Resource<RoadNetwork>;
  onRetry: () => void;
}): JSX.Element {
  const loaded = network.state === "success" ? network.data : null;

  return (
    <Panel
      title="Multi-junction network"
      subtitle="Junctions as nodes, directed road segments as edges"
      flush
      aside={
        loaded !== null ? (
          <StatusPill tone="ok" dot title={`Network ${loaded.network_id}`}>
            {loaded.junctions.length} junctions · {loaded.segments.length} segments
          </StatusPill>
        ) : (
          <StatusPill tone="pending">No network loaded</StatusPill>
        )
      }
    >
      <div style={{ padding: loaded !== null ? 0 : 16 }}>
        <ResourceView
          resource={network}
          onRetry={onRetry}
          loadingRows={3}
          plannedCapabilities={
            "Once a network is loaded, this panel renders its real topology. " +
            "Incident markers and blockages come from phase 6; per-segment " +
            "congestion needs a camera bound to a segment, and staged signal " +
            "advisories arrive with the decision engine. They are layered on as those " +
            "phases land — every value traced to an observation, a model " +
            "prediction or a simulation run."
          }
        >
          {(data) => {
            if (data.junctions.length === 0) {
              return (
                <EmptyState
                  title="Network contains no junctions"
                  message={
                    `Network ${data.network_id} (version ${data.version}) was loaded ` +
                    "but declares no junctions, so there is nothing to draw."
                  }
                />
              );
            }

            const layout = layoutNetwork(data);
            return (
              <>
                <NetworkSvg layout={layout} />
                <p className="netgraph__caption">
                  {data.name ?? data.network_id} · version {data.version} ·{" "}
                  {data.driving_side}-hand traffic · positions derived from{" "}
                  {layout.strategy === "geographic"
                    ? "surveyed coordinates"
                    : "a deterministic layout (junctions carry no coordinates)"}
                  . Topology source: {data.provenance.source}.
                </p>
                <div className="legend" aria-label="Visual encodings">
                  <span className="legend__item">
                    <span className="legend__swatch" />
                    Road segment — no congestion colouring: no camera is bound to a segment
                  </span>
                  <span className="legend__item">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                      <circle cx="12" cy="12" r="8" />
                    </svg>
                    Junction
                  </span>
                  {data.facilities.length > 0 && (
                    <span className="legend__item">
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeDasharray="3 3" aria-hidden="true">
                        <rect x="3" y="7" width="18" height="10" rx="2" />
                      </svg>
                      Facility — emergency destination
                    </span>
                  )}
                </div>
              </>
            );
          }}
        </ResourceView>
      </div>
    </Panel>
  );
}
