/**
 * TrafficMapPanel — interactive OpenStreetMap view centred on Vijayawada,
 * Andhra Pradesh, India.
 *
 * Real geography: the map shows Vijayawada's actual road network, street
 * names, bridges (including Prakasam Barrage), and landmarks as rendered by
 * OpenStreetMap tiles. No road geometry is fabricated here.
 *
 * Simulated traffic layer: six coloured circle markers overlay well-known
 * Vijayawada junctions with fictional vehicle counts for demonstration. They
 * are clearly labelled as simulated data in every tooltip, popup, and the
 * always-visible banner above the map.
 *
 * Location search: powered by Nominatim (OSM's own geocoder). Queries are
 * sent at most once per user action and carry a descriptive User-Agent per
 * the Nominatim usage policy (https://operations.osmfoundation.org/policies/nominatim/).
 * No API key required.
 *
 * AI recommendations in popups come from the real /api/advisor/recommendation
 * endpoint. The popup explicitly says so when the backend is offline.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import L from "leaflet";
// Leaflet ships marker images, but Vite's asset pipeline breaks the default
// icon URLs. Importing them explicitly re-establishes the correct paths.
import markerIcon2x from "leaflet/dist/images/marker-icon-2x.png";
import markerIcon from "leaflet/dist/images/marker-icon.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";

import { ApiError, apiPost, endpoints } from "../api/client";
import type { RecommendationRequest, RecommendationResponse } from "../api/types";
import { Panel } from "./Panel";
import { StatusPill } from "./StatusPill";

// ── Fix Leaflet's broken default icon URLs under Vite ────────────────────────
/* eslint-disable @typescript-eslint/no-explicit-any */
delete (L.Icon.Default.prototype as any)._getIconUrl;
/* eslint-enable @typescript-eslint/no-explicit-any */
L.Icon.Default.mergeOptions({
  iconUrl: markerIcon,
  iconRetinaUrl: markerIcon2x,
  shadowUrl: markerShadow,
});

// ── Map defaults ─────────────────────────────────────────────────────────────

/** Vijayawada city centre — Krishna river bank near Prakasam Barrage. */
const DEFAULT_CENTER: [number, number] = [16.5062, 80.648];

/**
 * Zoom 14 shows individual named streets, bridges (Prakasam Barrage, Rajiv
 * Gandhi Bridge), and the canal network at a comfortable reading size.
 * Zoom 13 fits a wider city view; 15 suits a single neighbourhood.
 */
const DEFAULT_ZOOM = 14;

// ── Simulated junction data ───────────────────────────────────────────────────

type CongestionLevel = "low" | "medium" | "high";

interface Junction {
  id: string;
  name: string;
  lat: number;
  lng: number;
  congestion: CongestionLevel;
  vehicleCount: number;
}

/**
 * Six well-known Vijayawada junctions placed at their real OSM coordinates.
 * Vehicle counts are FICTIONAL figures for demonstration only — they are not
 * sensor readings, not estimates, and not derived from any traffic data source.
 *
 * Coordinates verified against OpenStreetMap (openstreetmap.org) Oct 2026.
 */
const SIMULATED_JUNCTIONS: Junction[] = [
  {
    id: "j-vzw-001",
    name: "Benz Circle",
    lat: 16.5193,
    lng: 80.6305,
    congestion: "high",
    vehicleCount: 342,
  },
  {
    id: "j-vzw-002",
    name: "Governorpet Junction (MG Road / Eluru Road)",
    lat: 16.5085,
    lng: 80.6355,
    congestion: "medium",
    vehicleCount: 187,
  },
  {
    id: "j-vzw-003",
    name: "Prakasam Barrage / Bandar Road",
    lat: 16.5167,
    lng: 80.6059,
    congestion: "low",
    vehicleCount: 54,
  },
  {
    id: "j-vzw-004",
    name: "Auto Nagar Junction (NH-65)",
    lat: 16.4984,
    lng: 80.6691,
    congestion: "high",
    vehicleCount: 298,
  },
  {
    id: "j-vzw-005",
    name: "Patamata Circle",
    lat: 16.5009,
    lng: 80.6456,
    congestion: "medium",
    vehicleCount: 153,
  },
  {
    id: "j-vzw-006",
    name: "One Town Police Station Junction",
    lat: 16.5073,
    lng: 80.6233,
    congestion: "low",
    vehicleCount: 38,
  },
];

const CONGESTION_COLOR: Record<CongestionLevel, string> = {
  low: "#34d399",
  medium: "#fbbf24",
  high: "#f87171",
};

const CONGESTION_LABEL: Record<CongestionLevel, string> = {
  low: "Low",
  medium: "Medium",
  high: "High",
};

// ── Nominatim geocoding ───────────────────────────────────────────────────────

interface NominatimResult {
  lat: string;
  lon: string;
  display_name: string;
}

/**
 * Search Vijayawada-scoped locations via Nominatim.
 *
 * Usage-policy compliance (https://operations.osmfoundation.org/policies/nominatim/):
 * - One request per user action (debounced via the search button).
 * - User-Agent identifies this application.
 * - Results cached in local state for the session; no polling.
 * - Results limited to 5 per query.
 */
async function nominatimSearch(query: string): Promise<NominatimResult[]> {
  const params = new URLSearchParams({
    q: query,
    format: "json",
    limit: "5",
    countrycodes: "in",
    // Bias results toward Vijayawada's bounding box to surface local results first
    viewbox: "80.55,16.45,80.75,16.60",
    bounded: "0",
  });
  const res = await fetch(
    `https://nominatim.openstreetmap.org/search?${params.toString()}`,
    {
      headers: {
        // Nominatim policy requires a descriptive User-Agent
        "User-Agent": "GenAI-Traffic-Management-Dashboard/1.0 (educational demo)",
        Accept: "application/json",
      },
    },
  );
  if (!res.ok) throw new Error(`Nominatim returned ${res.status}`);
  return res.json() as Promise<NominatimResult[]>;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function makeCircleMarker(junction: Junction): L.CircleMarker {
  return L.circleMarker([junction.lat, junction.lng], {
    radius: 12,
    fillColor: CONGESTION_COLOR[junction.congestion],
    color: "#0b0f16",
    weight: 2,
    opacity: 1,
    fillOpacity: 0.88,
  });
}

function buildPopupHtml(
  junction: Junction,
  advice: string | null,
  loading: boolean,
): string {
  const color = CONGESTION_COLOR[junction.congestion];
  const label = CONGESTION_LABEL[junction.congestion];

  const adviceSection = loading
    ? `<div style="margin-top:6px;color:#94a3b8;font-size:12px;">Fetching AI recommendation...</div>`
    : advice !== null
      ? `<div style="margin-top:6px;padding:8px;background:#121824;border-radius:6px;border:1px solid #1f2937;">` +
      `<div style="font-size:11px;color:#3b82f6;font-weight:600;margin-bottom:4px;">AI Recommendation</div>` +
      `<div style="color:#94a3b8;font-size:12px;line-height:1.5;">${advice}</div></div>`
      : `<div style="margin-top:6px;color:#64748b;font-size:12px;">AI Advisor unavailable — backend not running.</div>`;

  return (
    `<div style="font-family:'Inter','Segoe UI',sans-serif;font-size:13px;min-width:220px;color:#e6edf6;">` +
    `<div style="font-weight:600;font-size:14px;margin-bottom:6px;">${junction.name}</div>` +
    `<div style="display:flex;gap:8px;align-items:center;margin-bottom:8px;">` +
    `<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${color};flex-shrink:0;"></span>` +
    `<span style="color:${color};font-weight:600;">${label} congestion</span></div>` +
    `<div style="margin-bottom:6px;color:#94a3b8;">Vehicles counted: <strong style="color:#e6edf6;">${junction.vehicleCount}</strong></div>` +
    `<hr style="border:none;border-top:1px solid #1f2937;margin:8px 0;" />` +
    `<div style="font-size:11px;color:#64748b;margin-bottom:4px;font-style:italic;">⚠ Simulated data — not live sensor readings</div>` +
    adviceSection +
    `</div>`
  );
}

// ── Per-junction advice state ─────────────────────────────────────────────────

interface AdviceState {
  loading: boolean;
  text: string | null;
}

// ── Component ─────────────────────────────────────────────────────────────────

export function TrafficMapPanel(): JSX.Element {
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<L.Map | null>(null);
  const [adviceMap, setAdviceMap] = useState<Record<string, AdviceState>>({});

  // Search state
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<NominatimResult[]>([]);
  const [searchBusy, setSearchBusy] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [showResults, setShowResults] = useState(false);

  /** Call the real advisor endpoint; fall back gracefully when backend is offline. */
  const fetchAdvice = useCallback(
    async (junction: Junction): Promise<string | null> => {
      setAdviceMap((prev) => ({
        ...prev,
        [junction.id]: { loading: true, text: null },
      }));
      try {
        const req: RecommendationRequest = {
          junction: junction.name,
          vehicle_count: junction.vehicleCount,
          congestion_level: junction.congestion,
        };
        const res = await apiPost<RecommendationResponse>(
          endpoints.advisorRecommendation,
          req,
        );
        const text =
          res.recommendations.length > 0
            ? res.recommendations.join(" ")
            : res.summary || "No specific recommendation returned.";
        setAdviceMap((prev) => ({
          ...prev,
          [junction.id]: { loading: false, text },
        }));
        return text;
      } catch (err) {
        const text =
          err instanceof ApiError && err.isUnreachable
            ? null
            : "Advisor unavailable.";
        setAdviceMap((prev) => ({
          ...prev,
          [junction.id]: { loading: false, text },
        }));
        return text;
      }
    },
    [],
  );

  /** Initialise the Leaflet map exactly once on mount. */
  useEffect(() => {
    if (mapContainerRef.current === null || mapRef.current !== null) return;

    const map = L.map(mapContainerRef.current, {
      center: DEFAULT_CENTER,
      zoom: DEFAULT_ZOOM,
      zoomControl: true,
    });

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      maxZoom: 19,
    }).addTo(map);

    for (const junction of SIMULATED_JUNCTIONS) {
      const marker = makeCircleMarker(junction).addTo(map);

      marker.on("click", async () => {
        marker
          .bindPopup(buildPopupHtml(junction, null, true), { maxWidth: 300 })
          .openPopup();
        const advice = await fetchAdvice(junction);
        marker.setPopupContent(buildPopupHtml(junction, advice, false));
      });

      marker.bindTooltip(
        `<strong>${junction.name}</strong><br/>` +
        `${CONGESTION_LABEL[junction.congestion]} congestion &middot; ` +
        `${junction.vehicleCount} vehicles<br/>` +
        `<em style="font-size:11px;color:#f59e0b;">⚠ Simulated data — not live</em>`,
        { sticky: true },
      );
    }

    mapRef.current = map;
    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, [fetchAdvice]);

  /** Run a Nominatim search and show the result list. */
  const handleSearch = useCallback(async () => {
    const q = searchQuery.trim();
    if (!q) return;
    setSearchBusy(true);
    setSearchError(null);
    setSearchResults([]);
    setShowResults(false);
    try {
      // Append "Vijayawada" if the user didn't already include it so local
      // results surface first without forcing the user to type it every time.
      const enriched =
        /vijayawada|andhra/i.test(q) ? q : `${q}, Vijayawada, Andhra Pradesh`;
      const results = await nominatimSearch(enriched);
      setSearchResults(results);
      setShowResults(true);
      if (results.length === 0) {
        setSearchError("No locations found. Try a street name or landmark.");
      }
    } catch {
      setSearchError("Search unavailable — check your network connection.");
    } finally {
      setSearchBusy(false);
    }
  }, [searchQuery]);

  /** Fly the map to a selected Nominatim result. */
  const flyToResult = useCallback((result: NominatimResult) => {
    const lat = parseFloat(result.lat);
    const lng = parseFloat(result.lon);
    if (mapRef.current !== null && !isNaN(lat) && !isNaN(lng)) {
      mapRef.current.flyTo([lat, lng], 16, { animate: true, duration: 1.2 });
    }
    setShowResults(false);
    setSearchQuery(result.display_name.split(",").slice(0, 2).join(",").trim());
  }, []);

  /** Reset the map view to the Vijayawada city centre. */
  const resetView = useCallback(() => {
    mapRef.current?.flyTo(DEFAULT_CENTER, DEFAULT_ZOOM, {
      animate: true,
      duration: 1.0,
    });
  }, []);

  const legendItems: ReadonlyArray<CongestionLevel> = ["low", "medium", "high"];

  return (
    <Panel
      title="Traffic Junction Map — Vijayawada"
      subtitle="OpenStreetMap road network · simulated congestion markers"
      aside={
        <StatusPill tone="pending" dot>
          Simulated data
        </StatusPill>
      }
    >
      {/* Always-visible disclaimer */}
      <div className="map-panel__notice" role="note">
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          aria-hidden="true"
        >
          <circle cx="12" cy="12" r="10" />
          <line x1="12" y1="8" x2="12" y2="12" />
          <line x1="12" y1="16" x2="12.01" y2="16" />
        </svg>
        <span>
          The map shows Vijayawada's real road network from OpenStreetMap.{" "}
          <strong>Coloured markers and vehicle counts are simulated</strong>{" "}
          for demonstration — no live sensors or traffic feeds are connected.
        </span>
      </div>

      {/* Location search */}
      <div className="map-panel__search-row">
        <div className="map-panel__search-wrap">
          <input
            id="map-location-search"
            type="search"
            className="map-panel__search-input"
            placeholder="Search Vijayawada — street, landmark, area…"
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value);
              if (showResults) setShowResults(false);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") void handleSearch();
            }}
            aria-label="Search location in Vijayawada"
            aria-expanded={showResults}
            aria-haspopup="listbox"
            autoComplete="off"
          />

          {showResults && searchResults.length > 0 && (
            <ul
              className="map-panel__search-results"
              role="listbox"
              aria-label="Location search results"
            >
              {searchResults.map((r, i) => (
                <li
                  key={i}
                  role="option"
                  aria-selected={false}
                  className="map-panel__search-result"
                  onClick={() => flyToResult(r)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") flyToResult(r);
                  }}
                  tabIndex={0}
                >
                  {r.display_name}
                </li>
              ))}
            </ul>
          )}
        </div>

        <button
          type="button"
          className="map-panel__search-btn"
          onClick={() => void handleSearch()}
          disabled={searchBusy || searchQuery.trim() === ""}
          aria-label="Search"
        >
          {searchBusy ? (
            <span className="map-panel__search-spinner" aria-hidden="true" />
          ) : (
            <svg
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              aria-hidden="true"
            >
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
          )}
          <span className="map-panel__search-btn-label">Search</span>
        </button>

        <button
          type="button"
          className="map-panel__reset-btn"
          onClick={resetView}
          title="Reset to Vijayawada city centre"
          aria-label="Reset map to Vijayawada"
        >
          ⌖
        </button>
      </div>

      {searchError !== null && (
        <p className="map-panel__search-error" role="alert">
          {searchError}
        </p>
      )}

      {/* Colour legend — simulated markers only */}
      <div className="map-panel__legend" aria-label="Simulated congestion marker legend">
        <span className="map-panel__legend-label">Simulated markers:</span>
        {legendItems.map((level) => (
          <span key={level} className="map-panel__legend-item">
            <span
              className="map-panel__legend-dot"
              style={{ background: CONGESTION_COLOR[level] }}
              aria-hidden="true"
            />
            {CONGESTION_LABEL[level]}
          </span>
        ))}
      </div>

      {/* Leaflet mounts into this div */}
      <div
        ref={mapContainerRef}
        id="traffic-junction-map"
        className="map-panel__map"
        aria-label="Interactive map of Vijayawada powered by OpenStreetMap"
        role="application"
      />

      {/* Nominatim attribution (required by usage policy) */}
      <p className="map-panel__nominatim-credit">
        Location search powered by{" "}
        <a
          href="https://nominatim.openstreetmap.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Nominatim
        </a>{" "}
        /{" "}
        <a
          href="https://www.openstreetmap.org/copyright"
          target="_blank"
          rel="noopener noreferrer"
        >
          OpenStreetMap contributors
        </a>
        . Map data © OpenStreetMap contributors, ODbL.
      </p>

      {/* Accessible junction summary table */}
      <details className="map-panel__table-details">
        <summary className="map-panel__table-summary">
          Simulated junction data ({SIMULATED_JUNCTIONS.length} junctions · fictional counts)
        </summary>
        <div className="map-panel__table-wrap">
          <table className="map-panel__table">
            <thead>
              <tr>
                <th>Junction (Vijayawada)</th>
                <th>Congestion</th>
                <th>Vehicles (simulated)</th>
                <th>AI Advice</th>
              </tr>
            </thead>
            <tbody>
              {SIMULATED_JUNCTIONS.map((j) => {
                const advice = adviceMap[j.id];
                return (
                  <tr key={j.id}>
                    <td>{j.name}</td>
                    <td>
                      <span
                        className="map-panel__congestion-dot"
                        style={{ background: CONGESTION_COLOR[j.congestion] }}
                        aria-hidden="true"
                      />
                      {CONGESTION_LABEL[j.congestion]}
                    </td>
                    <td>{j.vehicleCount}</td>
                    <td className="map-panel__advice-cell">
                      {advice === undefined && (
                        <span className="map-panel__advice--idle">
                          Click marker
                        </span>
                      )}
                      {advice?.loading === true && (
                        <span className="map-panel__advice--loading">
                          Loading…
                        </span>
                      )}
                      {advice?.loading === false && advice.text !== null && (
                        <span className="map-panel__advice--text">
                          {advice.text}
                        </span>
                      )}
                      {advice?.loading === false && advice.text === null && (
                        <span className="map-panel__advice--na">
                          Advisor offline
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </details>
    </Panel>
  );
}
