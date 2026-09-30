/**
 * Layout tests.
 *
 * Fixtures use deliberately unusual identifiers ("XQ-77", "ZZ9-NORTH") so that
 * a hard-coded topology could not accidentally satisfy these assertions.
 */

import { describe, expect, it } from "vitest";
import { layoutNetwork } from "./networkLayout";
import { makeNetwork, makeJunction, makeSegment, makeFacility } from "../test/fixtures";

describe("layoutNetwork", () => {
  it("produces one node per junction returned by the API", () => {
    const layout = layoutNetwork(
      makeNetwork({
        junctions: [makeJunction("XQ-77"), makeJunction("ZZ9-NORTH"), makeJunction("K3")],
      }),
    );
    expect(layout.nodes.map((n) => n.id).sort()).toEqual(["K3", "XQ-77", "ZZ9-NORTH"]);
  });

  it("produces one edge per segment returned by the API", () => {
    const layout = layoutNetwork(
      makeNetwork({
        junctions: [makeJunction("A1"), makeJunction("B2")],
        segments: [makeSegment("SEG-X", "A1", "B2"), makeSegment("SEG-Y", "B2", "A1")],
      }),
    );
    const roadEdges = layout.edges.filter((e) => !e.isFacilityAccess);
    expect(roadEdges.map((e) => e.id).sort()).toEqual(["SEG-X", "SEG-Y"]);
  });

  it("uses surveyed coordinates when every junction has one", () => {
    const layout = layoutNetwork(
      makeNetwork({
        junctions: [
          makeJunction("SOUTH", { position: { latitude: 12.9, longitude: 77.6 } }),
          makeJunction("NORTH", { position: { latitude: 13.1, longitude: 77.6 } }),
        ],
      }),
    );
    expect(layout.strategy).toBe("geographic");

    const north = layout.nodes.find((n) => n.id === "NORTH");
    const south = layout.nodes.find((n) => n.id === "SOUTH");
    // Higher latitude must render further up the screen (smaller y).
    expect(north?.y).toBeLessThan(south?.y ?? 0);
  });

  it("falls back to a deterministic layout when coordinates are missing", () => {
    const network = makeNetwork({
      junctions: [makeJunction("A"), makeJunction("B"), makeJunction("C")],
    });
    const first = layoutNetwork(network);
    const second = layoutNetwork(network);
    expect(first.strategy).toBe("circular");
    // Same network must always draw identically, or two screenshots of one
    // network look like two different networks.
    expect(first.nodes).toEqual(second.nodes);
  });

  it("keeps every node inside the canvas", () => {
    const layout = layoutNetwork(
      makeNetwork({
        junctions: ["A", "B", "C", "D", "E", "F"].map((id) => makeJunction(id)),
      }),
    );
    for (const node of layout.nodes) {
      expect(node.x).toBeGreaterThanOrEqual(0);
      expect(node.x).toBeLessThanOrEqual(layout.width);
      expect(node.y).toBeGreaterThanOrEqual(0);
      expect(node.y).toBeLessThanOrEqual(layout.height);
    }
  });

  it("returns an empty layout for a network with no junctions", () => {
    const layout = layoutNetwork(makeNetwork({ junctions: [], segments: [] }));
    expect(layout.nodes).toEqual([]);
    expect(layout.edges).toEqual([]);
    expect(layout.strategy).toBe("empty");
  });

  it("attaches a facility to its access junction", () => {
    const layout = layoutNetwork(
      makeNetwork({
        junctions: [makeJunction("GATE"), makeJunction("OTHER")],
        facilities: [makeFacility("MED-1", "St Jude", "GATE")],
      }),
    );
    const facility = layout.nodes.find((n) => n.id === "MED-1");
    expect(facility?.kind).toBe("facility");
    expect(facility?.label).toBe("St Jude");
    expect(
      layout.edges.some((e) => e.isFacilityAccess && e.fromId === "GATE" && e.toId === "MED-1"),
    ).toBe(true);
  });

  it("skips a facility whose access junction is absent rather than guessing", () => {
    const layout = layoutNetwork(
      makeNetwork({
        junctions: [makeJunction("ONLY")],
        facilities: [makeFacility("ORPHAN", "Nowhere", "MISSING-JUNCTION")],
      }),
    );
    expect(layout.nodes.find((n) => n.id === "ORPHAN")).toBeUndefined();
  });

  it("skips a segment whose endpoints are absent rather than drawing to origin", () => {
    const layout = layoutNetwork(
      makeNetwork({
        junctions: [makeJunction("A")],
        segments: [makeSegment("DANGLING", "A", "NOT-PRESENT")],
      }),
    );
    expect(layout.edges).toEqual([]);
  });
});
