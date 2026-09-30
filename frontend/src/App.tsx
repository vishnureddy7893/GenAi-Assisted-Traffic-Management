/**
 * Dashboard shell, organised as the operator workflow.
 *
 * The page is a sequence rather than a collection of panels:
 *
 *   Traffic input → computer vision → traffic analysis → traffic state
 *   → road network → incidents and emergency → route optimization
 *   → traffic simulation
 *
 * That order is the actual dependency order in the backend, and making it
 * visible matters: nearly every empty state on this page is empty because an
 * earlier stage has not been completed, not because something failed. A flat
 * grid of panels hides that, and an operator staring at an empty route panel
 * cannot tell whether the optimizer is broken or whether no network is loaded.
 *
 * Stages belonging to later phases are rendered visibly apart, marked as not
 * implemented, and carry no controls. A capability that does not exist must
 * never sit in the flow looking like one that failed.
 *
 * Every panel fetches its own resource and renders its own loading, success,
 * error and unavailable states. There is no shared "if anything failed, show
 * nothing" path — one endpoint being unimplemented must not blank a panel that
 * is working.
 */

import { useState } from "react";

import { AdvisorPanel } from "./components/AdvisorPanel";
import { AppHeader } from "./components/AppHeader";
import { DataSourcePanel } from "./components/DataSourcePanel";
import { EmergencyPanel } from "./components/EmergencyPanel";
import { EvidenceAttributionPanel } from "./components/EvidenceAttributionPanel";
import { IncidentsPanel } from "./components/IncidentsPanel";
import { NetworkGraphPanel } from "./components/NetworkGraphPanel";
import { NetworkPanel } from "./components/NetworkPanel";
import { PhasePanel } from "./components/PhasePanel";
import {
  RouteOptimizationPanel,
  type OptimizedRoute,
} from "./components/RouteOptimizationPanel";
import { SafetyCatalogPanel } from "./components/SafetyCatalogPanel";
import { SiteBindingPanel } from "./components/SiteBindingPanel";
import { SimulationPanel } from "./components/SimulationPanel";
import { SystemStatusPanel } from "./components/SystemStatusPanel";
import { TrafficInputPanel } from "./components/TrafficInputPanel";
import { VersionsPanel } from "./components/VersionsPanel";
import { WorkflowStage } from "./components/WorkflowStage";
import { endpoints } from "./api/client";
import { useApiResource } from "./hooks/useApiResource";
import type {
  EmergencyScenariosResponse,
  HealthResponse,
  PhasesResponse,
  RoadNetwork,
  SafetyCatalogResponse,
  SourceCatalogResponse,
  VersionsResponse,
} from "./api/types";

/** Health is polled for liveness only; no traffic data is polled. */
const HEALTH_POLL_MS = 15_000;

export function App(): JSX.Element {
  // Lifted so the simulation stage can evaluate the route the optimizer just
  // produced. It holds the *request*, not the routes: the simulator asks the
  // backend for candidates itself, so the two stages cannot end up comparing
  // routes drawn from different snapshots.
  const [routed, setRouted] = useState<OptimizedRoute | null>(null);
  // The media the advisor explains. Lifted so the advisory always describes
  // the image on screen rather than whichever one was analysed last.
  const [mediaId, setMediaId] = useState<string | null>(null);
  const health = useApiResource<HealthResponse>(endpoints.health, HEALTH_POLL_MS);
  const versions = useApiResource<VersionsResponse>(endpoints.versions);
  const phases = useApiResource<PhasesResponse>(endpoints.phases);
  const sources = useApiResource<SourceCatalogResponse>(endpoints.sourceCatalog);
  const safety = useApiResource<SafetyCatalogResponse>(endpoints.safetyCatalog);
  const network = useApiResource<RoadNetwork>(endpoints.network);
  const emergency = useApiResource<EmergencyScenariosResponse>(
    endpoints.emergencyScenarios,
  );

  return (
    <div className="app">
      <AppHeader health={health.resource} />

      <main className="app__main">
        <div className="banner banner--advisory">
          <span className="banner__icon" aria-hidden="true">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 3.2 3.4 7v5.4c0 4.6 3.5 8.2 8.6 9.4 5.1-1.2 8.6-4.8 8.6-9.4V7Z" strokeLinejoin="round" />
            </svg>
          </span>
          <span>
            <strong>Advisory decision support — simulation and recorded data only.</strong>{" "}
            This system analyses recorded and simulated traffic and produces
            recommendations for human operators. It has no control path to any real
            traffic signal, vehicle or infrastructure.
          </span>
        </div>

        <WorkflowStage
          step={1}
          title="Traffic input, vision and measurement"
          summary="Ingest recorded media, detect and track road users, derive measurements, then interpret them as a traffic state."
        >
          <TrafficInputPanel onMediaChanged={setMediaId} />
        </WorkflowStage>

        <WorkflowStage
          step={2}
          title="Road network"
          summary="Load the surveyed graph that every incident, dispatch and route is expressed over."
        >
          <div className="grid">
            <div className="grid__col--7">
              <NetworkPanel
                network={network.resource}
                onNetworkChanged={network.refresh}
              />
            </div>
            <div className="grid__col--5">
              <NetworkGraphPanel network={network.resource} onRetry={network.refresh} />
            </div>
          </div>
          <SiteBindingPanel network={network.resource} />
          <EvidenceAttributionPanel />
        </WorkflowStage>

        <WorkflowStage
          step={3}
          title="Incidents and emergency response"
          summary="Declare what is disrupting the network, and the response that has to travel through it."
        >
          <div className="grid">
            <div className="grid__col--6">
              <IncidentsPanel
                network={network.resource}
                onIncidentsChanged={network.refresh}
              />
            </div>
            <div className="grid__col--6">
              <EmergencyPanel
                scenarios={emergency.resource}
                onRetry={emergency.refresh}
                network={network.resource}
                onEmergencyChanged={emergency.refresh}
              />
            </div>
          </div>
        </WorkflowStage>

        <WorkflowStage
          step={4}
          title="Route optimization"
          summary="Rank admissible routes by risk-adjusted arrival time, and say why the winner won."
        >
          <RouteOptimizationPanel
            network={network.resource}
            scenarios={emergency.resource}
            onOptimized={setRouted}
          />
        </WorkflowStage>

        <WorkflowStage
          step={5}
          title="Traffic simulation"
          summary="Run the recommended route and the baseline through the same modelled conditions, and report the difference."
        >
          <SimulationPanel network={network.resource} context={routed} />
        </WorkflowStage>

        <WorkflowStage
          step={6}
          title="AI Traffic Advisor"
          summary="Explain the computed facts in operator language and suggest actions. The advisory cites the facts it used and names what it could not see."
        >
          <AdvisorPanel mediaId={mediaId} />
        </WorkflowStage>

        <WorkflowStage
          step={7}
          title="Decision record and explanation"
          summary="Human approval of a recommendation, and a generated narrative over a fact table."
          future
          futureNote={
            "Not implemented. The decision engine and human approval are phase 9, " +
            "and generated explanation is phase 12. Simulation results appear in " +
            "stage 5 above, labelled as modelled outcomes; no panel here shows a " +
            "placeholder narrative or an approval control, because a recommendation " +
            "that can be accepted before the step built to check it is exactly what " +
            "this ordering exists to prevent."
          }
        />

        <section className="stage stage--reference">
          <header className="stage__head">
            <span className="stage__step stage__step--muted" aria-hidden="true">
              ·
            </span>
            <div className="stage__headings">
              <h2 className="stage__title">Reference and system status</h2>
              <p className="stage__summary">
                Policy, capability and version information behind the stages above.
              </p>
            </div>
          </header>

          <div className="stage__body">
            <div className="grid">
              <div className="grid__col--12">
                <SystemStatusPanel health={health.resource} onRetry={health.refresh} />
              </div>
              <div className="grid__col--12">
                <DataSourcePanel catalog={sources.resource} onRetry={sources.refresh} />
              </div>
              <div className="grid__col--6">
                <SafetyCatalogPanel catalog={safety.resource} onRetry={safety.refresh} />
              </div>
              <div className="grid__col--6">
                <PhasePanel phases={phases.resource} onRetry={phases.refresh} />
              </div>
              <div className="grid__col--12">
                <VersionsPanel versions={versions.resource} onRetry={versions.refresh} />
              </div>
            </div>
          </div>
        </section>
      </main>

      <footer className="footer">
        GenAI-Assisted Traffic Management System — capstone prototype. Recommendations
        are advisory and must be reviewed by a qualified operator.
      </footer>
    </div>
  );
}
