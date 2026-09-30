/**
 * Deterministic layout for a road network returned by `/api/network`.
 *
 * Pure function, no React, no DOM — so the geometry can be tested directly and
 * the rendering component stays a thin projection of the API response.
 *
 * The layout contains **no domain values**. There is no table of junction ids
 * and no hard-coded coordinate anywhere: every node drawn exists because the
 * backend returned it, and its position is computed from what the backend said
 * about it.
 *
 * Two strategies, chosen from the data rather than configured:
 *
 * - **geographic** — used when every junction carries a `position`. Real
 *   coordinates are projected so the drawing matches the ground truth.
 * - **circular** — the fallback when any junction lacks a position. Ordered by
 *   junction id so the same network always draws identically; an unstable
 *   layout would make two screenshots of one network look like two networks.
 *
 * An equirectangular projection is used deliberately: at the scale of a traffic
 * network (a few km) its distortion is far below one pixel, and it keeps the
 * function dependency-free and trivially testable. It is not suitable for
 * city-wide or larger extents, which is noted here rather than discovered later.
 */

import type { GeoPoint, Junction, RoadNetwork, RoadSegment } from "../api/types";

export type LayoutStrategy = "geographic" | "circular" | "empty";

export interface LayoutNode {
  id: string;
  label: string;
  x: number;
  y: number;
  kind: "junction" | "facility";
  /** Junction control regime, or facility type. Straight from the API. */
  category: string;
}

export interface LayoutEdge {
  id: string;
  fromId: string;
  toId: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  label: string | null;
  /** True for the synthetic link joining a facility to its access junction. */
  isFacilityAccess: boolean;
}

export interface NetworkLayout {
  nodes: LayoutNode[];
  edges: LayoutEdge[];
  width: number;
  height: number;
  strategy: LayoutStrategy;
}

export interface LayoutOptions {
  width?: number;
  height?: number;
  padding?: number;
}

const DEFAULT_WIDTH = 640;
const DEFAULT_HEIGHT = 380;
const DEFAULT_PADDING = 46;
/** Distance a facility is drawn from the junction it is entered through. */
const FACILITY_OFFSET = 58;

function hasPosition(junction: Junction): junction is Junction & { position: GeoPoint } {
  return junction.position !== null;
}

/**
 * Scale a value from one range to another, collapsing a zero-width range.
 *
 * The guard compares the *magnitude* of the span. Latitude is deliberately
 * passed inverted (max first) so that north renders upwards, which makes the
 * raw difference negative — a signed comparison here silently collapsed every
 * geographic layout onto a single horizontal line.
 */
function scale(value: number, min: number, max: number, out0: number, out1: number): number {
  if (Math.abs(max - min) < 1e-12) return (out0 + out1) / 2;
  return out0 + ((value - min) / (max - min)) * (out1 - out0);
}

function geographicPositions(
  junctions: Junction[],
  width: number,
  height: number,
  padding: number,
): Map<string, { x: number; y: number }> {
  const lats = junctions.map((j) => (j.position as GeoPoint).latitude);
  const lons = junctions.map((j) => (j.position as GeoPoint).longitude);
  const minLat = Math.min(...lats);
  const maxLat = Math.max(...lats);
  const minLon = Math.min(...lons);
  const maxLon = Math.max(...lons);

  const positions = new Map<string, { x: number; y: number }>();
  for (const junction of junctions) {
    const point = junction.position as GeoPoint;
    positions.set(junction.junction_id, {
      x: scale(point.longitude, minLon, maxLon, padding, width - padding),
      // Latitude increases northwards; SVG y increases downwards.
      y: scale(point.latitude, maxLat, minLat, padding, height - padding),
    });
  }
  return positions;
}

function circularPositions(
  junctions: Junction[],
  width: number,
  height: number,
  padding: number,
): Map<string, { x: number; y: number }> {
  const positions = new Map<string, { x: number; y: number }>();
  const centreX = width / 2;
  const centreY = height / 2;

  if (junctions.length === 1) {
    const only = junctions[0];
    if (only !== undefined) positions.set(only.junction_id, { x: centreX, y: centreY });
    return positions;
  }

  const radius = Math.max(0, Math.min(width, height) / 2 - padding);
  // Sorted by id so the same network always renders identically.
  const ordered = [...junctions].sort((a, b) => a.junction_id.localeCompare(b.junction_id));

  ordered.forEach((junction, index) => {
    // Start at the top and proceed clockwise.
    const angle = (2 * Math.PI * index) / ordered.length - Math.PI / 2;
    positions.set(junction.junction_id, {
      x: centreX + radius * Math.cos(angle),
      y: centreY + radius * Math.sin(angle),
    });
  });
  return positions;
}

function placeFacility(
  anchor: { x: number; y: number },
  centreX: number,
  centreY: number,
): { x: number; y: number } {
  // Push the facility outward, away from the middle of the drawing, so it does
  // not land on top of the network it hangs off.
  const dx = anchor.x - centreX;
  const dy = anchor.y - centreY;
  const magnitude = Math.hypot(dx, dy);
  if (magnitude < 1e-9) {
    return { x: anchor.x, y: anchor.y + FACILITY_OFFSET };
  }
  return {
    x: anchor.x + (dx / magnitude) * FACILITY_OFFSET,
    y: anchor.y + (dy / magnitude) * FACILITY_OFFSET,
  };
}

function segmentLabel(segment: RoadSegment): string | null {
  return segment.name;
}

/**
 * Project a network onto a 2D canvas.
 *
 * Every returned node and edge corresponds to something the backend sent. A
 * network with no junctions yields an empty layout — never a placeholder shape.
 */
export function layoutNetwork(
  network: RoadNetwork,
  options: LayoutOptions = {},
): NetworkLayout {
  const width = options.width ?? DEFAULT_WIDTH;
  const height = options.height ?? DEFAULT_HEIGHT;
  const padding = options.padding ?? DEFAULT_PADDING;

  const junctions = network.junctions;
  if (junctions.length === 0) {
    return { nodes: [], edges: [], width, height, strategy: "empty" };
  }

  const allPositioned = junctions.every(hasPosition);
  const strategy: LayoutStrategy = allPositioned ? "geographic" : "circular";
  const positions = allPositioned
    ? geographicPositions(junctions, width, height, padding)
    : circularPositions(junctions, width, height, padding);

  const nodes: LayoutNode[] = [];
  for (const junction of junctions) {
    const point = positions.get(junction.junction_id);
    if (point === undefined) continue;
    nodes.push({
      id: junction.junction_id,
      label: junction.name ?? junction.junction_id,
      x: point.x,
      y: point.y,
      kind: "junction",
      category: junction.kind,
    });
  }

  const centreX = width / 2;
  const centreY = height / 2;

  for (const facility of network.facilities) {
    const anchor = positions.get(facility.access_junction_id);
    // A facility whose access junction is absent cannot be placed. Skipping is
    // correct: guessing a position would draw a hospital that is not there.
    if (anchor === undefined) continue;
    const point = placeFacility(anchor, centreX, centreY);
    positions.set(facility.facility_id, point);
    nodes.push({
      id: facility.facility_id,
      label: facility.name,
      x: point.x,
      y: point.y,
      kind: "facility",
      category: facility.facility_type,
    });
  }

  const edges: LayoutEdge[] = [];
  for (const segment of network.segments) {
    const from = positions.get(segment.from_junction_id);
    const to = positions.get(segment.to_junction_id);
    if (from === undefined || to === undefined) continue;
    edges.push({
      id: segment.segment_id,
      fromId: segment.from_junction_id,
      toId: segment.to_junction_id,
      x1: from.x,
      y1: from.y,
      x2: to.x,
      y2: to.y,
      label: segmentLabel(segment),
      isFacilityAccess: false,
    });
  }

  for (const facility of network.facilities) {
    const anchor = positions.get(facility.access_junction_id);
    const point = positions.get(facility.facility_id);
    if (anchor === undefined || point === undefined) continue;
    edges.push({
      id: `access:${facility.facility_id}`,
      fromId: facility.access_junction_id,
      toId: facility.facility_id,
      x1: anchor.x,
      y1: anchor.y,
      x2: point.x,
      y2: point.y,
      label: null,
      isFacilityAccess: true,
    });
  }

  return { nodes, edges, width, height, strategy };
}
