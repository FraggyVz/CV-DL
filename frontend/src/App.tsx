import { useCallback, useEffect, useState } from "react";

type ServiceState = "checking" | "connected" | "unavailable";

type HealthResponse = {
  status: "ok";
  application: "StratScout";
};

function isHealthResponse(value: unknown): value is HealthResponse {
  if (typeof value !== "object" || value === null) {
    return false;
  }

  const response = value as Record<string, unknown>;
  return response.status === "ok" && response.application === "StratScout";
}

export default function App() {
  const [serviceState, setServiceState] = useState<ServiceState>("checking");
  const [serviceMessage, setServiceMessage] = useState("Checking the local service...");

  const checkService = useCallback(async () => {
    setServiceState("checking");
    setServiceMessage("Checking the local service...");

    try {
      const response = await fetch("/api/health");
      if (!response.ok) {
        throw new Error(`The service returned HTTP ${response.status}.`);
      }

      const body: unknown = await response.json();
      if (!isHealthResponse(body)) {
        throw new Error("The service returned an unexpected health response.");
      }

      setServiceState("connected");
      setServiceMessage("The Python service is ready.");
    } catch (error) {
      setServiceState("unavailable");
      setServiceMessage(
        error instanceof Error ? error.message : "The local service could not be reached.",
      );
    }
  }, []);

  useEffect(() => {
    void checkService();
  }, [checkService]);

  return (
    <main className="page-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="StratScout home">
          <span className="brand-mark" aria-hidden="true">
            S
          </span>
          <span>StratScout</span>
        </a>
        <span className="local-badge">
          <span className="local-dot" />
          LOCAL WORKSPACE
        </span>
      </header>

      <section className="hero" aria-labelledby="page-title">
        <p className="eyebrow">CS2 · MIRAGE · VIDEO ANALYSIS</p>
        <h1 id="page-title">
          Make sense of
          <br />
          every round.
        </h1>
        <p className="hero-copy">
          Prepare minimap recordings, inspect player detections and review tactical segments
          from one local workspace.
        </p>
      </section>

      <section className="service-card" aria-labelledby="service-title">
        <div className="service-copy">
          <div className="service-heading">
            <span className={`service-indicator ${serviceState}`} aria-hidden="true" />
            <h2 id="service-title">Application service</h2>
          </div>
          <p aria-live="polite">{serviceMessage}</p>
        </div>
        {serviceState === "unavailable" && (
          <button className="secondary-button" onClick={() => void checkService()} type="button">
            Retry connection
          </button>
        )}
      </section>

      <section className="workspace-grid" aria-label="Planned workspace areas">
        <article className="workspace-card">
          <span className="card-number">01</span>
          <h2>Match library</h2>
          <p>Browse prepared recordings and their processing details.</p>
          <span className="card-status">COMING NEXT</span>
        </article>
        <article className="workspace-card">
          <span className="card-number">02</span>
          <h2>Detection review</h2>
          <p>Inspect observed player positions alongside the minimap video.</p>
          <span className="card-status">COMING NEXT</span>
        </article>
        <article className="workspace-card">
          <span className="card-number">03</span>
          <h2>Tactical segments</h2>
          <p>Navigate, label and review segments using the agreed five classes.</p>
          <span className="card-status">COMING NEXT</span>
        </article>
      </section>

      <aside className="notice" aria-label="Current project limitation">
        <span className="notice-mark" aria-hidden="true">
          i
        </span>
        <p>
          This is the application scaffold. The tactical classifier is not implemented or
          validated, so this page does not display tactical predictions.
        </p>
      </aside>

      <footer className="page-footer">
        <span>Self-hosted · Your recordings stay on this machine</span>
        <span>StratScout prototype</span>
      </footer>
    </main>
  );
}
