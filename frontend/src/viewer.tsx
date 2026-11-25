// The local app's "All round classifications" section as a Streamlit component, for the
// cloud demo (deploy/streamlit). Streamlit runs the same analysis code as the local app
// and passes every round's results and pipeline records in; this renders them with the
// same components and styles, so the two deployments look and read the same.
//
// It speaks Streamlit's component messages directly (no extra package): it announces
// itself, receives "streamlit:render" with the data, and reports its height.
import React, { useEffect, useMemo, useState } from "react";
import ReactDOM from "react-dom/client";
import { ModelWalkthrough, PredictionCard, RESULT_TABS } from "./App";
import type { AnalysisResult, ResultTab } from "./App";
import ModelPipeline from "./ModelPipeline";
import type { PipelineRecord } from "./ModelPipeline";
import "./app.css";

type ViewerRound = {
  id: string;
  round_number: number;
  filename: string;
  status: "complete" | "failed" | "idle";
  error?: string | null;
  analysis_id?: string;
  results: AnalysisResult[];
  pipelines: Record<string, PipelineRecord>;
};

function send(type: string, extra: Record<string, unknown> = {}) {
  window.parent.postMessage({ isStreamlitMessage: true, type, ...extra }, "*");
}

function ResultsViewer() {
  const [rounds, setRounds] = useState<ViewerRound[]>([]);
  const [activeTab, setActiveTab] = useState<ResultTab>("comparison");
  const [resultRoundId, setResultRoundId] = useState("");

  useEffect(() => {
    function onMessage(event: MessageEvent) {
      if (event.data?.type === "streamlit:render") {
        setRounds((event.data.args?.rounds as ViewerRound[]) ?? []);
      }
    }
    window.addEventListener("message", onMessage);
    send("streamlit:componentReady", { apiVersion: 1 });
    const observer = new ResizeObserver(() =>
      send("streamlit:setFrameHeight", { height: document.documentElement.scrollHeight }),
    );
    observer.observe(document.body);
    return () => {
      window.removeEventListener("message", onMessage);
      observer.disconnect();
    };
  }, []);

  const shownRoundId = useMemo(
    () =>
      rounds.some((round) => round.id === resultRoundId)
        ? resultRoundId
        : (rounds.find((round) => round.status === "complete")?.id ?? rounds[0]?.id ?? ""),
    [rounds, resultRoundId],
  );
  const complete = rounds.filter((round) => round.status === "complete").length;
  const shown =
    activeTab === "comparison" ? rounds : rounds.filter((round) => round.id === shownRoundId);

  return (
    <section className="workspace-section results-section viewer-embed" aria-labelledby="results-title">
      <div className="section-heading">
        <div>
          <p className="eyebrow">05 / CLASSIFICATION</p>
          <h2 id="results-title">All round classifications</h2>
        </div>
        <span className="quiet-label">
          {complete} / {rounds.length} COMPLETE
        </span>
      </div>
      <div aria-label="Model result views" className="result-tabs" role="tablist">
        {RESULT_TABS.map((tab) => (
          <button
            aria-selected={activeTab === tab.id}
            className={activeTab === tab.id ? "active" : ""}
            id={`tab-${tab.id}`}
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            role="tab"
            type="button"
          >
            {tab.label}
          </button>
        ))}
      </div>
      {activeTab !== "comparison" && rounds.length > 0 && (
        <div aria-label="Choose the round to inspect" className="round-picker" role="group">
          <span className="round-picker-label">ROUND</span>
          {rounds.map((round) => {
            const selected = round.id === shownRoundId;
            return (
              <button
                aria-label={`Round ${round.round_number}${round.status === "complete" ? "" : ", not analysed"}`}
                aria-pressed={selected}
                className={`round-chip ${selected ? "active" : ""} ${round.status === "complete" ? "complete" : ""}`}
                key={round.id}
                onClick={() => setResultRoundId(round.id)}
                title={round.filename}
                type="button"
              >
                {round.round_number}
              </button>
            );
          })}
        </div>
      )}
      <div aria-labelledby={`tab-${activeTab}`} className="match-classifications" role="tabpanel">
        {rounds.length === 0 ? (
          <p className="empty-state">Upload a round clip to see its TCN and GRU output here.</p>
        ) : (
          shown.map((round) => (
            <article className="match-classification" key={round.id}>
              <div className="match-classification-heading">
                <div>
                  <p className="round-card-number">Round {round.round_number}</p>
                  <h3>{round.filename}</h3>
                </div>
                <span className="round-card-status">
                  {round.status === "complete"
                    ? "Analysis complete"
                    : round.status === "failed"
                      ? "Needs attention"
                      : "Not analysed"}
                </span>
              </div>
              {round.status === "complete" && round.results.length > 0 ? (
                round.results.map((result) => {
                  const visibleWarnings = result.warnings.filter(
                    (warning) => !warning.startsWith("background from only "),
                  );
                  return (
                    <article className="result-segment" key={result.segment_id}>
                      <div className="result-segment-heading">
                        <span className={`phase-pill ${result.phase}`}>{result.phase}</span>
                        <span>
                          {result.detection.player_icon_observations} player-icon observations ·{" "}
                          {result.detection.bomb_visible_frames}/{result.detection.frames} frames
                          with bomb visible
                        </span>
                      </div>
                      <div className="prediction-grid">
                        {(activeTab === "comparison" || activeTab === "tcn") && (
                          <PredictionCard model="TCN" prediction={result.predictions.tcn} />
                        )}
                        {(activeTab === "comparison" || activeTab === "gru") && (
                          <PredictionCard model="Compact pooled GRU" prediction={result.predictions.gru} />
                        )}
                        {activeTab === "tcn" && (
                          <ModelWalkthrough
                            detection={result.detection}
                            model="TCN"
                            prediction={result.predictions.tcn}
                          />
                        )}
                        {activeTab === "gru" && (
                          <ModelWalkthrough
                            detection={result.detection}
                            model="Compact pooled GRU"
                            prediction={result.predictions.gru}
                          />
                        )}
                      </div>
                      {(activeTab === "tcn" || activeTab === "gru") && (
                        <ModelPipeline
                          available={Boolean(round.pipelines[result.segment_id])}
                          key={`${activeTab}-${round.analysis_id ?? ""}-${result.segment_id}`}
                          model={activeTab}
                          record={round.pipelines[result.segment_id]}
                          unavailableReason={result.pipeline_error}
                        />
                      )}
                      {visibleWarnings.length > 0 && (
                        <ul className="result-warnings">
                          {visibleWarnings.map((warning) => (
                            <li key={warning}>{warning}</li>
                          ))}
                        </ul>
                      )}
                    </article>
                  );
                })
              ) : round.status === "failed" ? (
                <p className="error-message" role="alert">
                  {round.error ?? "Analysis failed. Correct the input and run it again."}
                </p>
              ) : null}
            </article>
          ))
        )}
      </div>
    </section>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ResultsViewer />
  </React.StrictMode>,
);
