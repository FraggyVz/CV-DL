import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent } from "react";

type ServiceState = "checking" | "connected" | "unavailable";
type ResultTab = "comparison" | "tcn" | "gru";
type SegmentPhase = "attack" | "retake";
type TacticClass = "A_DIRECT" | "B_DIRECT" | "A_SPLIT" | "B_SPLIT" | "RETAKE";

type ModelPrediction = {
  tactic: TacticClass;
  confidence: number;
  uncertain: boolean;
  runner_up: TacticClass;
  probabilities: Record<TacticClass, number>;
};

type AnalysisResult = {
  segment_id: string;
  round_number: number;
  phase: SegmentPhase;
  start_s: number;
  end_s: number;
  detection: {
    frames: number;
    player_icon_observations: number;
    bomb_visible_frames: number;
  };
  warnings: string[];
  predictions: {
    tcn: ModelPrediction;
    gru: ModelPrediction;
  };
};

type AnalysisState = {
  analysis_id?: string;
  video_id: string;
  status: string;
  stage: string;
  message: string;
  progress: number;
  error: string | null;
  results: AnalysisResult[];
};

type UploadedVideo = {
  id: string;
  filename: string;
  size_bytes: number;
  uploaded_at: string;
  content_url: string;
  duration_seconds?: number;
  match_id?: string;
  match_name?: string;
  round_number?: number;
  phase?: SegmentPhase;
  prepared_clip?: boolean;
};

type SegmentDraft = {
  id: string;
  start_s: number;
  end_s: number;
  round_number: number;
  phase: SegmentPhase;
};

type MatchWorkspace = {
  id: string;
  name: string;
};

type RoundAnalysis = {
  state?: AnalysisState;
  error?: string;
};

type HealthResponse = {
  status: "ok";
  application: "StratScout";
};

const RESULT_TABS: { id: ResultTab; label: string }[] = [
  { id: "comparison", label: "TCN vs GRU" },
  { id: "tcn", label: "TCN only" },
  { id: "gru", label: "GRU only" },
];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isHealthResponse(value: unknown): value is HealthResponse {
  return (
    isRecord(value) && value.status === "ok" && value.application === "StratScout"
  );
}

const ACTIVE_MATCH_STORAGE_KEY = "stratscout.active-match";
const MATCHES_STORAGE_KEY = "stratscout.matches";

function readActiveMatch(): MatchWorkspace {
  const saved = window.localStorage.getItem(ACTIVE_MATCH_STORAGE_KEY);
  if (saved) {
    try {
      const value: unknown = JSON.parse(saved);
      if (isRecord(value) && typeof value.id === "string" && typeof value.name === "string") {
        return { id: value.id, name: value.name };
      }
    } catch (error) {
      if (!(error instanceof SyntaxError)) {
        throw error;
      }
    }
    window.localStorage.removeItem(ACTIVE_MATCH_STORAGE_KEY);
  }
  return { id: crypto.randomUUID(), name: "Match 1" };
}

function readSavedMatches(): MatchWorkspace[] {
  const saved = window.localStorage.getItem(MATCHES_STORAGE_KEY);
  if (!saved) {
    return [];
  }
  try {
    const value: unknown = JSON.parse(saved);
    if (Array.isArray(value)) {
      return value.filter(
        (match): match is MatchWorkspace =>
          isRecord(match) &&
          typeof match.id === "string" &&
          typeof match.name === "string",
      );
    }
  } catch (error) {
    if (!(error instanceof SyntaxError)) {
      throw error;
    }
  }
  window.localStorage.removeItem(MATCHES_STORAGE_KEY);
  return [];
}

function matchNumber(name: string): number | null {
  const match = /^Match (\d+)$/.exec(name);
  return match ? Number(match[1]) : null;
}

function isUploadedVideo(value: unknown): value is UploadedVideo {
  return (
    isRecord(value) &&
    typeof value.id === "string" &&
    typeof value.filename === "string" &&
    typeof value.size_bytes === "number" &&
    typeof value.uploaded_at === "string" &&
    typeof value.content_url === "string" &&
    (value.duration_seconds === undefined || typeof value.duration_seconds === "number") &&
    (value.match_id === undefined || typeof value.match_id === "string") &&
    (value.match_name === undefined || typeof value.match_name === "string") &&
    (value.round_number === undefined || typeof value.round_number === "number") &&
    (value.prepared_clip === undefined || typeof value.prepared_clip === "boolean") &&
    (value.phase === undefined || value.phase === "attack" || value.phase === "retake")
  );
}

function isVideoList(value: unknown): value is { videos: UploadedVideo[] } {
  return (
    isRecord(value) &&
    Array.isArray(value.videos) &&
    value.videos.every(isUploadedVideo)
  );
}

function isSegmentDraft(value: unknown): value is SegmentDraft {
  return (
    isRecord(value) &&
    typeof value.id === "string" &&
    typeof value.start_s === "number" &&
    typeof value.end_s === "number" &&
    typeof value.round_number === "number" &&
    (value.phase === "attack" || value.phase === "retake")
  );
}

function isSegmentList(value: unknown): value is { segments: SegmentDraft[] } {
  return (
    isRecord(value) &&
    Array.isArray(value.segments) &&
    value.segments.every(isSegmentDraft)
  );
}

function isTacticClass(value: unknown): value is TacticClass {
  return (
    value === "A_DIRECT" ||
    value === "B_DIRECT" ||
    value === "A_SPLIT" ||
    value === "B_SPLIT" ||
    value === "RETAKE"
  );
}

function isModelPrediction(value: unknown): value is ModelPrediction {
  if (!isRecord(value) || !isRecord(value.probabilities)) {
    return false;
  }
  const probabilities = value.probabilities;
  const classes: TacticClass[] = ["A_DIRECT", "B_DIRECT", "A_SPLIT", "B_SPLIT", "RETAKE"];
  return (
    isTacticClass(value.tactic) &&
    typeof value.confidence === "number" &&
    typeof value.uncertain === "boolean" &&
    isTacticClass(value.runner_up) &&
    classes.every((name) => typeof probabilities[name] === "number")
  );
}

function isAnalysisResult(value: unknown): value is AnalysisResult {
  if (!isRecord(value) || !isRecord(value.detection) || !isRecord(value.predictions)) {
    return false;
  }
  return (
    typeof value.segment_id === "string" &&
    typeof value.round_number === "number" &&
    typeof value.start_s === "number" &&
    typeof value.end_s === "number" &&
    (value.phase === "attack" || value.phase === "retake") &&
    typeof value.detection.frames === "number" &&
    typeof value.detection.player_icon_observations === "number" &&
    typeof value.detection.bomb_visible_frames === "number" &&
    Array.isArray(value.warnings) &&
    value.warnings.every((warning) => typeof warning === "string") &&
    isModelPrediction(value.predictions.tcn) &&
    isModelPrediction(value.predictions.gru)
  );
}

function isAnalysisState(value: unknown): value is AnalysisState {
  return (
    isRecord(value) &&
    typeof value.video_id === "string" &&
    typeof value.status === "string" &&
    typeof value.stage === "string" &&
    typeof value.message === "string" &&
    typeof value.progress === "number" &&
    (typeof value.error === "string" || value.error === null) &&
    Array.isArray(value.results) &&
    value.results.every(isAnalysisResult)
  );
}

async function responseError(response: Response): Promise<string> {
  const body: unknown = await response.json().catch(() => null);
  if (isRecord(body) && typeof body.detail === "string") {
    return body.detail;
  }
  return `Request failed with HTTP ${response.status}.`;
}

function formatClock(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.floor(seconds % 60);
  return `${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024 * 1024) {
    return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function PredictionCard({
  model,
  prediction,
}: {
  model: string;
  prediction: ModelPrediction;
}) {
  const classes: TacticClass[] = ["A_DIRECT", "B_DIRECT", "A_SPLIT", "B_SPLIT", "RETAKE"];
  return (
    <div className="prediction-card">
      <div className="prediction-heading">
        <span>{model}</span>
        {prediction.uncertain && <span className="uncertain-badge">UNCERTAIN</span>}
      </div>
      <strong className="predicted-tactic">{prediction.tactic}</strong>
      <span className="prediction-confidence">
        {Math.round(prediction.confidence * 100)}% confidence · next {prediction.runner_up}
      </span>
      <div className="probability-list">
        {classes.map((name) => (
          <div className="probability-row" key={name}>
            <span>{name}</span>
            <span className="probability-track">
              <span style={{ width: `${Math.min(prediction.probabilities[name] * 100, 100)}%` }} />
            </span>
            <span>{Math.round(prediction.probabilities[name] * 100)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [serviceState, setServiceState] = useState<ServiceState>("checking");
  const [serviceMessage, setServiceMessage] = useState("Checking the local service...");
  const [videos, setVideos] = useState<UploadedVideo[]>([]);
  const [activeMatch, setActiveMatch] = useState<MatchWorkspace>(() => readActiveMatch());
  const [savedMatches, setSavedMatches] = useState<MatchWorkspace[]>(() => readSavedMatches());
  const activeMatchStored = useRef(
    Boolean(window.localStorage.getItem(ACTIVE_MATCH_STORAGE_KEY)),
  );
  const savedMatchesStored = useRef(
    Boolean(window.localStorage.getItem(MATCHES_STORAGE_KEY)),
  );
  const [selectedVideoId, setSelectedVideoId] = useState("");
  const [videoState, setVideoState] = useState("Loading recordings...");
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");
  const [uploadPhase, setUploadPhase] = useState<SegmentPhase>("attack");
  const [errorMessage, setErrorMessage] = useState("");
  const [segments, setSegments] = useState<SegmentDraft[]>([]);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [videoError, setVideoError] = useState("");
  const [activeTab, setActiveTab] = useState<ResultTab>("comparison");
  const [analysis, setAnalysis] = useState<AnalysisState | null>(null);
  const [roundAnalyses, setRoundAnalyses] = useState<Record<string, RoundAnalysis>>({});
  const [startingAllAnalysis, setStartingAllAnalysis] = useState(false);
  const [removingVideoId, setRemovingVideoId] = useState("");
  const [resettingWorkspace, setResettingWorkspace] = useState(false);
  const [savingPreparedClipId, setSavingPreparedClipId] = useState("");
  const [playbackSpeed, setPlaybackSpeed] = useState<"1" | "2">("2");
  const [startingAnalysis, setStartingAnalysis] = useState(false);

  const rememberMatches = useCallback((matches: MatchWorkspace[]) => {
    const uniqueMatches = [...new Map(matches.map((match) => [match.id, match])).values()];
    window.localStorage.setItem(MATCHES_STORAGE_KEY, JSON.stringify(uniqueMatches));
    savedMatchesStored.current = true;
    setSavedMatches(uniqueMatches);
  }, []);

  const selectedVideo = videos.find((video) => video.id === selectedVideoId);
  const preparedClip = selectedVideo?.prepared_clip ?? false;
  const isCurrentRound =
    selectedVideo?.match_id === activeMatch.id &&
    typeof selectedVideo.round_number === "number";
  const matchOptions = useMemo(() => {
    const matches = new Map<string, MatchWorkspace>(
      savedMatches.map((match) => [match.id, match]),
    );
    matches.set(activeMatch.id, activeMatch);
    for (const video of videos) {
      if (video.match_id && video.match_name) {
        matches.set(video.match_id, { id: video.match_id, name: video.match_name });
      }
    }
    return [...matches.values()].sort((left, right) => left.name.localeCompare(right.name));
  }, [activeMatch, savedMatches, videos]);
  const currentMatchVideos = useMemo(
    () =>
      videos
        .filter(
          (video) =>
            video.match_id === activeMatch.id && typeof video.round_number === "number",
        )
        .sort((left, right) => (left.round_number ?? 0) - (right.round_number ?? 0)),
    [activeMatch.id, videos],
  );
  const legacyVideos = useMemo(
    () => videos.filter((video) => !video.match_id),
    [videos],
  );
  const analysisRunning =
    analysis !== null &&
    ["queued", "preparing", "detecting", "classifying"].includes(analysis.status);
  const roundsNeedingAnalysis = currentMatchVideos.filter((video) => {
    const status = roundAnalyses[video.id]?.state?.status;
    return (
      status !== "complete" &&
      !["queued", "preparing", "detecting", "classifying"].includes(status ?? "")
    );
  });
  const allRoundsComplete =
    currentMatchVideos.length > 0 &&
    currentMatchVideos.every((video) => roundAnalyses[video.id]?.state?.status === "complete");
  const matchHasRunningAnalysis = currentMatchVideos.some((video) => {
    const status = roundAnalyses[video.id]?.state?.status;
    return ["queued", "preparing", "detecting", "classifying"].includes(status ?? "");
  });

  useEffect(() => {
    if (activeMatchStored.current) {
      window.localStorage.setItem(ACTIVE_MATCH_STORAGE_KEY, JSON.stringify(activeMatch));
    }
  }, [activeMatch]);

  const checkService = useCallback(async () => {
    setServiceState("checking");
    setServiceMessage("Checking the local service...");
    try {
      const response = await fetch("/api/health");
      if (!response.ok) {
        throw new Error(await responseError(response));
      }
      const body: unknown = await response.json();
      if (!isHealthResponse(body)) {
        throw new Error("The service returned an unexpected health response.");
      }
      setServiceState("connected");
      setServiceMessage("Local service connected");
    } catch (error) {
      setServiceState("unavailable");
      setServiceMessage(
        error instanceof Error ? error.message : "The local service could not be reached.",
      );
    }
  }, []);

  const loadVideos = useCallback(async () => {
    setVideoState("Loading recordings...");
    try {
      const response = await fetch("/api/videos");
      if (!response.ok) {
        throw new Error(await responseError(response));
      }
      const body: unknown = await response.json();
      if (!isVideoList(body)) {
        throw new Error("The service returned an unexpected recording list.");
      }
      let matchToLoad = activeMatch;
      if (!activeMatchStored.current) {
        const latestMatchVideo = body.videos.find(
          (video) => video.match_id && video.match_name,
        );
        if (latestMatchVideo?.match_id && latestMatchVideo.match_name) {
          matchToLoad = {
            id: latestMatchVideo.match_id,
            name: latestMatchVideo.match_name,
          };
          setActiveMatch(matchToLoad);
        }
        window.localStorage.setItem(ACTIVE_MATCH_STORAGE_KEY, JSON.stringify(matchToLoad));
        activeMatchStored.current = true;
      }
      if (!savedMatchesStored.current) {
        const matches = new Map<string, MatchWorkspace>();
        for (const video of body.videos) {
          if (video.match_id && video.match_name) {
            matches.set(video.match_id, {
              id: video.match_id,
              name: video.match_name,
            });
          }
        }
        matches.set(matchToLoad.id, matchToLoad);
        const activeNumber = matchNumber(matchToLoad.name);
        if (activeNumber !== null) {
          const knownNames = new Set([...matches.values()].map((match) => match.name));
          for (let number = 1; number < activeNumber; number += 1) {
            const name = `Match ${number}`;
            if (!knownNames.has(name)) {
              const id = crypto.randomUUID();
              matches.set(id, { id, name });
            }
          }
        }
        rememberMatches([...matches.values()]);
      }
      setVideos(body.videos);
      const activeVideos = body.videos
        .filter(
          (video) =>
            video.match_id === matchToLoad.id && typeof video.round_number === "number",
        )
        .sort((left, right) => (left.round_number ?? 0) - (right.round_number ?? 0));
      setSelectedVideoId((current) => {
        const currentVideo = body.videos.find((video) => video.id === current);
        if (currentVideo?.match_id === matchToLoad.id) {
          return current;
        }
        return activeVideos.at(-1)?.id ?? "";
      });
      setVideoState(body.videos.length ? "" : "No recordings uploaded yet.");
    } catch (error) {
      setVideoState(
        error instanceof Error ? error.message : "Could not load recordings from this app.",
      );
    }
  }, [activeMatch, rememberMatches]);

  useEffect(() => {
    void checkService();
    void loadVideos();
  }, [checkService, loadVideos]);

  useEffect(() => {
    setCurrentTime(0);
    setDuration(0);
    setVideoError("");
    setErrorMessage("");
    setAnalysis(null);
    if (!selectedVideo) {
      setSegments([]);
      return;
    }

    let active = true;
    setSegments([]);
    fetch(`/api/videos/${selectedVideo.id}/segments`)
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(await responseError(response));
        }
        const body: unknown = await response.json();
        if (!isSegmentList(body)) {
          throw new Error("The service returned invalid segment boundaries.");
        }
        if (active) {
          setSegments(body.segments);
        }
      })
      .catch((error: unknown) => {
        if (active) {
          setErrorMessage(
            error instanceof Error ? error.message : "Could not load segment boundaries.",
          );
        }
      });
    fetch(`/api/videos/${selectedVideo.id}/analysis`)
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(await responseError(response));
        }
        const body: unknown = await response.json();
        if (!isAnalysisState(body)) {
          throw new Error("The service returned invalid analysis status.");
        }
        if (active) {
          setAnalysis(body.status === "idle" ? null : body);
        }
      })
      .catch((error: unknown) => {
        if (active) {
          setErrorMessage(
            error instanceof Error ? error.message : "Could not load analysis status.",
          );
        }
      });

    return () => {
      active = false;
    };
  }, [selectedVideo]);

  useEffect(() => {
    let active = true;
    let timer = 0;
    async function pollRoundAnalyses() {
      const entries = await Promise.all(
        currentMatchVideos.map(async (video): Promise<[string, RoundAnalysis]> => {
          try {
            const response = await fetch(`/api/videos/${video.id}/analysis`);
            if (!response.ok) {
              throw new Error(await responseError(response));
            }
            const body: unknown = await response.json();
            if (!isAnalysisState(body)) {
              throw new Error("The service returned invalid analysis status.");
            }
            return [video.id, { state: body }];
          } catch (error) {
            return [
              video.id,
              {
                error:
                  error instanceof Error
                    ? error.message
                    : "Could not load this round's analysis status.",
              },
            ];
          }
        }),
      );
      if (!active) {
        return;
      }
      setRoundAnalyses((current) => ({ ...current, ...Object.fromEntries(entries) }));
      if (
        entries.some(([, round]) =>
          ["queued", "preparing", "detecting", "classifying"].includes(
            round.state?.status ?? "",
          ),
        )
      ) {
        timer = window.setTimeout(() => void pollRoundAnalyses(), 1200);
      }
    }
    void pollRoundAnalyses();
    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [currentMatchVideos, matchHasRunningAnalysis]);

  useEffect(() => {
    if (selectedVideo && selectedVideo.match_id === activeMatch.id && analysis) {
      setRoundAnalyses((current) => ({
        ...current,
        [selectedVideo.id]: { state: analysis },
      }));
    }
  }, [activeMatch.id, analysis, selectedVideo]);

  useEffect(() => {
    if (!selectedVideo || !analysis?.analysis_id || !analysisRunning) {
      return;
    }
    let active = true;
    let timer = 0;
    const poll = async () => {
      try {
        const response = await fetch(`/api/videos/${selectedVideo.id}/analysis`);
        if (!response.ok) {
          throw new Error(await responseError(response));
        }
        const body: unknown = await response.json();
        if (!isAnalysisState(body)) {
          throw new Error("The service returned invalid analysis status.");
        }
        if (active) {
          setAnalysis(body);
        }
      } catch (error) {
        if (active) {
          setErrorMessage(
            error instanceof Error ? error.message : "Could not refresh analysis status.",
          );
        }
      } finally {
        if (active) {
          timer = window.setTimeout(poll, 1000);
        }
      }
    };
    timer = window.setTimeout(poll, 1000);
    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [analysis?.analysis_id, analysisRunning, selectedVideo]);

  function selectMatch(matchId: string) {
    const match = matchOptions.find((option) => option.id === matchId);
    if (!match) {
      return;
    }
    rememberMatches([...matchOptions]);
    activeMatchStored.current = true;
    setActiveMatch(match);
    window.localStorage.setItem(ACTIVE_MATCH_STORAGE_KEY, JSON.stringify(match));
    const matchVideos = videos
      .filter(
        (video) => video.match_id === match.id && typeof video.round_number === "number",
      )
      .sort((left, right) => (left.round_number ?? 0) - (right.round_number ?? 0));
    setSelectedVideoId(matchVideos.at(-1)?.id ?? "");
  }

  function startNewMatch() {
    const nextNumber =
      Math.max(0, ...matchOptions.map((match) => matchNumber(match.name) ?? 0)) + 1;
    const match = {
      id: crypto.randomUUID(),
      name: `Match ${nextNumber}`,
    };
    rememberMatches([...matchOptions, match]);
    activeMatchStored.current = true;
    setActiveMatch(match);
    window.localStorage.setItem(ACTIVE_MATCH_STORAGE_KEY, JSON.stringify(match));
    setSelectedVideoId("");
    setErrorMessage("");
  }

  async function handleUpload(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.currentTarget.files ?? []);
    event.currentTarget.value = "";
    if (files.length === 0) {
      return;
    }

    const mp4Files = files.filter((file) => file.name.toLowerCase().endsWith(".mp4"));
    const issues = files
      .filter((file) => !file.name.toLowerCase().endsWith(".mp4"))
      .map((file) => `${file.name}: choose an MP4 recording.`);
    if (mp4Files.length === 0) {
      setErrorMessage(issues.join(" "));
      return;
    }

    setUploading(true);
    setErrorMessage("");
    setUploadStatus("");
    const uploaded: UploadedVideo[] = [];
    let nextRoundNumber =
      currentMatchVideos.reduce(
        (highest, video) => Math.max(highest, video.round_number ?? 0),
        0,
      ) + 1;
    try {
      for (const [index, file] of mp4Files.entries()) {
        setUploadStatus(`Uploading ${index + 1} of ${mp4Files.length}…`);
        try {
          const form = new FormData();
          form.append("file", file);
          form.append("match_id", activeMatch.id);
          form.append("match_name", activeMatch.name);
          form.append("round_number", String(nextRoundNumber));
          form.append("phase", uploadPhase);
          const response = await fetch("/api/videos", { method: "POST", body: form });
          if (!response.ok) {
            throw new Error(await responseError(response));
          }
          const body: unknown = await response.json();
          if (!isUploadedVideo(body)) {
            throw new Error("The service returned invalid recording details.");
          }
          uploaded.push(body);
          setVideos((current) => [body, ...current.filter((video) => video.id !== body.id)]);
          nextRoundNumber += 1;
        } catch (error) {
          const detail =
            error instanceof Error ? error.message : "The recording could not be uploaded.";
          issues.push(`${file.name}: ${detail}`);
        }
      }

      if (uploaded.length > 0) {
        setSelectedVideoId(uploaded[0].id);
        setVideoState("");
      }
      if (issues.length > 0) {
        const outcome =
          uploaded.length > 0
            ? `Uploaded ${uploaded.length} of ${files.length} selected files. `
            : "";
        setErrorMessage(`${outcome}${issues.join(" ")}`);
      }
    } finally {
      setUploading(false);
      setUploadStatus("");
    }
  }

  async function savePreparedClipSetting(videoId: string, value: boolean) {
    setSavingPreparedClipId(videoId);
    setErrorMessage("");
    try {
      const response = await fetch(`/api/videos/${videoId}/analysis-settings`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prepared_clip: value }),
      });
      if (!response.ok) {
        throw new Error(await responseError(response));
      }
      const body: unknown = await response.json();
      if (!isUploadedVideo(body)) {
        throw new Error("The service returned invalid recording settings.");
      }
      setVideos((current) =>
        current.map((video) => (video.id === videoId ? body : video)),
      );
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "Could not save the prepared-clip setting.",
      );
    } finally {
      setSavingPreparedClipId("");
    }
  }

  async function startAnalysis() {
    if (
      !selectedVideo ||
      !segments.length ||
      startingAnalysis ||
      analysisRunning ||
      savingPreparedClipId === selectedVideo.id
    ) {
      return;
    }
    setStartingAnalysis(true);
    setErrorMessage("");
    try {
      const response = await fetch(`/api/videos/${selectedVideo.id}/analysis`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          playback_speed: Number(playbackSpeed),
          prepared_clip: selectedVideo.prepared_clip ?? false,
          min_confidence: 0.6,
        }),
      });
      if (!response.ok) {
        throw new Error(await responseError(response));
      }
      const body: unknown = await response.json();
      if (!isAnalysisState(body)) {
        throw new Error("The service returned invalid analysis status.");
      }
      setAnalysis(body);
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "Could not start round analysis.",
      );
    } finally {
      setStartingAnalysis(false);
    }
  }

  async function analyzeAllRounds() {
    const videosToAnalyze = currentMatchVideos.filter((video) =>
      roundsNeedingAnalysis.some((round) => round.id === video.id),
    );
    if (
      videosToAnalyze.length === 0 ||
      startingAllAnalysis ||
      savingPreparedClipId !== ""
    ) {
      return;
    }

    setStartingAllAnalysis(true);
    setErrorMessage("");
    const entries = await Promise.all(
      videosToAnalyze.map(async (video): Promise<[string, RoundAnalysis]> => {
        try {
          const response = await fetch(`/api/videos/${video.id}/analysis`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              playback_speed: Number(playbackSpeed),
              prepared_clip: video.prepared_clip ?? false,
              min_confidence: 0.6,
            }),
          });
          if (response.status === 409) {
            const statusResponse = await fetch(`/api/videos/${video.id}/analysis`);
            if (!statusResponse.ok) {
              throw new Error(await responseError(statusResponse));
            }
            const statusBody: unknown = await statusResponse.json();
            if (!isAnalysisState(statusBody)) {
              throw new Error("The service returned invalid analysis status.");
            }
            return [video.id, { state: statusBody }];
          }
          if (!response.ok) {
            throw new Error(await responseError(response));
          }
          const body: unknown = await response.json();
          if (!isAnalysisState(body)) {
            throw new Error("The service returned invalid analysis status.");
          }
          return [video.id, { state: body }];
        } catch (error) {
          return [
            video.id,
            {
              error:
                error instanceof Error
                  ? error.message
                  : "Could not start this round's analysis.",
            },
          ];
        }
      }),
    );
    setRoundAnalyses((current) => ({ ...current, ...Object.fromEntries(entries) }));
    setStartingAllAnalysis(false);
  }

  async function removeRecording(video: UploadedVideo) {
    const label =
      typeof video.round_number === "number"
        ? `Round ${video.round_number}`
        : "this previous recording";
    if (
      !window.confirm(
        `Remove ${label} (${video.filename}) and its saved analysis? Later rounds in this match will be renumbered.`,
      )
    ) {
      return;
    }
    setRemovingVideoId(video.id);
    setErrorMessage("");
    try {
      const response = await fetch(`/api/videos/${video.id}`, { method: "DELETE" });
      if (!response.ok) {
        throw new Error(await responseError(response));
      }
      setRoundAnalyses((current) => {
        const next = { ...current };
        delete next[video.id];
        return next;
      });
      await loadVideos();
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "The recording could not be removed.",
      );
    } finally {
      setRemovingVideoId("");
    }
  }

  async function resetWorkspace() {
    if (
      !window.confirm(
        "Reset the entire local workspace? This permanently removes every uploaded recording and saved analysis from every match, including previous recordings.",
      )
    ) {
      return;
    }
    setResettingWorkspace(true);
    setErrorMessage("");
    try {
      const response = await fetch("/api/videos", { method: "DELETE" });
      if (!response.ok) {
        throw new Error(await responseError(response));
      }
      const body: unknown = await response.json();
      if (!isRecord(body) || typeof body.deleted_count !== "number") {
        throw new Error("The service returned an unexpected reset response.");
      }
      const freshMatch = { id: crypto.randomUUID(), name: "Match 1" };
      activeMatchStored.current = true;
      window.localStorage.setItem(ACTIVE_MATCH_STORAGE_KEY, JSON.stringify(freshMatch));
      rememberMatches([freshMatch]);
      setActiveMatch(freshMatch);
      setVideos([]);
      setSelectedVideoId("");
      setVideoState("No recordings uploaded yet.");
      setRoundAnalyses({});
      setAnalysis(null);
      setSegments([]);
      setCurrentTime(0);
      setDuration(0);
      setPlaybackSpeed("2");
      setActiveTab("comparison");
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "The workspace could not be reset.",
      );
    } finally {
      setResettingWorkspace(false);
    }
  }

  function seekTo(seconds: number) {
    if (videoRef.current) {
      videoRef.current.currentTime = seconds;
      setCurrentTime(seconds);
    }
  }

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <main className="page-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="StratScout home">
          <span className="brand-mark" aria-hidden="true">
            S
          </span>
          <span>StratScout</span>
        </a>
        <div className="header-status">
          <span className={`service-indicator ${serviceState}`} aria-hidden="true" />
          <span>{serviceMessage}</span>
          <span className="local-badge">LOCAL WORKSPACE</span>
        </div>
      </header>

      <section className="hero" aria-labelledby="page-title">
        <p className="eyebrow">CS2 · MIRAGE · MATCH WORKSPACE</p>
        <h1 id="page-title">
          Review the round.
          <br />
          Read the strategy.
        </h1>
        <p className="hero-copy">
          Upload one round clip at a time, keep each analysis in your match list, and
          compare the TCN and GRU predictions.
        </p>
      </section>

      <section className="workspace-section upload-section" aria-labelledby="upload-title">
        <div className="section-heading">
          <div>
            <p className="eyebrow">01 / RECORDING</p>
            <h2 id="upload-title">Build your match rounds</h2>
          </div>
          <span className="quiet-label">MP4 · STORED LOCALLY</span>
        </div>

        <div className="upload-row">
          <label className={`upload-button ${uploading ? "busy" : ""}`}>
            <input
              accept=".mp4,video/mp4"
              disabled={uploading}
              multiple
              onChange={(event) => void handleUpload(event)}
              type="file"
            />
            <span className="upload-icon" aria-hidden="true">
              ↑
            </span>
            <span aria-live="polite">
              {uploading ? uploadStatus : "Upload MP4 recordings"}
            </span>
          </label>
          <div className="recording-picker">
            <label htmlFor="match-select">Current match</label>
            <select
              disabled={uploading}
              id="match-select"
              onChange={(event) => selectMatch(event.target.value)}
              value={activeMatch.id}
            >
              {matchOptions.map((match) => (
                <option key={match.id} value={match.id}>
                  {match.name}
                </option>
              ))}
            </select>
          </div>
          <button
            className="secondary-button"
            disabled={uploading}
            onClick={startNewMatch}
            type="button"
          >
            New match
          </button>
          <button
            className="danger-button"
            disabled={uploading || resettingWorkspace}
            onClick={() => void resetWorkspace()}
            type="button"
          >
            {resettingWorkspace ? "Resetting…" : "Reset workspace"}
          </button>
          <div className="recording-picker">
            <label htmlFor="upload-phase">Clip type</label>
            <select
              disabled={uploading}
              id="upload-phase"
              onChange={(event) => setUploadPhase(event.target.value as SegmentPhase)}
              value={uploadPhase}
            >
              <option value="attack">Attack</option>
              <option value="retake">Retake</option>
            </select>
          </div>
          {legacyVideos.length > 0 && (
            <div className="recording-picker">
              <label htmlFor="legacy-recording">Previous recordings</label>
              <select
                id="legacy-recording"
                onChange={(event) => setSelectedVideoId(event.target.value)}
                value={selectedVideo?.match_id ? "" : selectedVideoId}
              >
                <option value="">Not assigned to a match</option>
                {legacyVideos.map((video) => (
                  <option key={video.id} value={video.id}>
                    {video.filename}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>
        {videoState && <p className="inline-message">{videoState}</p>}
        {errorMessage && (
          <p className="error-message" role="alert">
            {errorMessage}
          </p>
        )}
        <p className="storage-note">
          Each uploaded clip becomes the next numbered round in {activeMatch.name}. The
          entire clip is analysed; select Attack or Retake before uploading. Previous
          recordings remain available and are not changed.
        </p>
      </section>

      <section className="workflow-section" aria-labelledby="workflow-title">
        <div className="section-heading compact-heading">
          <div>
            <p className="eyebrow">02 / ANALYSIS FLOW</p>
            <h2 id="workflow-title">From round clip to model output</h2>
          </div>
          <span className="quiet-label">PIPELINE STATUS</span>
        </div>
        <ol className="pipeline">
          <li className={selectedVideo ? "complete" : "current"}>
            <span className="pipeline-number">{selectedVideo ? "✓" : "1"}</span>
            <span className="pipeline-title">Upload</span>
            <span className="pipeline-state">{selectedVideo ? "Ready" : "Waiting"}</span>
          </li>
          <li
            className={
              analysisRunning && analysis?.status !== "classifying"
                ? "current"
                : analysis?.status === "classifying" || analysis?.status === "complete"
                  ? "complete"
                  : ""
            }
          >
            <span className="pipeline-number">
              {analysis?.status === "classifying" || analysis?.status === "complete" ? "✓" : "2"}
            </span>
            <span className="pipeline-title">Prepare & detect</span>
            <span className="pipeline-state">
              {analysis?.status === "preparing" || analysis?.status === "detecting"
                ? "In progress"
                : analysis?.status === "classifying" || analysis?.status === "complete"
                  ? "Complete"
                  : analysis?.status === "failed"
                    ? "Failed"
                    : selectedVideo
                      ? "Ready to run"
                      : "Waiting"}
            </span>
          </li>
          <li className={isCurrentRound ? "complete" : selectedVideo ? "current" : ""}>
            <span className="pipeline-number">{isCurrentRound ? "✓" : "3"}</span>
            <span className="pipeline-title">Assign round</span>
            <span className="pipeline-state">
              {isCurrentRound
                ? `Round ${selectedVideo?.round_number}`
                : selectedVideo
                  ? "Legacy clip"
                  : "Automatic"}
            </span>
          </li>
          <li className={analysis?.status === "classifying" ? "current" : analysis?.status === "complete" ? "complete" : ""}>
            <span className="pipeline-number">
              {analysis?.status === "complete" ? "✓" : "4"}
            </span>
            <span className="pipeline-title">Classify</span>
            <span className="pipeline-state">
              {analysis?.status === "classifying"
                ? "In progress"
                : analysis?.status === "complete"
                  ? "Complete"
                  : analysis?.status === "failed"
                    ? "Failed"
                    : "Waiting"}
            </span>
          </li>
        </ol>
        <p className="workflow-note">
          Upload one MP4 per round, then analyse it. Round numbers advance automatically
          within the selected match; full-match round detection is not supported.
        </p>
        {analysisRunning && analysis && (
          <div className="analysis-progress" aria-live="polite">
            <div className="analysis-progress-heading">
              <span>{analysis.message}</span>
              <span>{analysis.progress}%</span>
            </div>
            <div
              className="analysis-progress-track"
              role="progressbar"
              aria-label="Analysis progress"
              aria-valuenow={analysis.progress}
              aria-valuemin={0}
              aria-valuemax={100}
            >
              <span style={{ width: `${analysis.progress}%` }} />
            </div>
          </div>
        )}
        {analysis?.status === "failed" && analysis.error && (
          <p className="error-message" role="alert">
            {analysis.error}
          </p>
        )}
      </section>

      <section className="workspace-section rounds-section" aria-labelledby="rounds-title">
        <div className="section-heading">
          <div>
            <p className="eyebrow">03 / MATCH ROUNDS</p>
            <h2 id="rounds-title">{activeMatch.name} rounds</h2>
          </div>
          <div className="round-actions">
            <button
              className="secondary-button"
              disabled={
                currentMatchVideos.length === 0 ||
                startingAllAnalysis ||
                savingPreparedClipId !== "" ||
                roundsNeedingAnalysis.length === 0
              }
              onClick={() => void analyzeAllRounds()}
              type="button"
            >
              {startingAllAnalysis
                ? "Starting all rounds…"
                : matchHasRunningAnalysis
                  ? "Analyse remaining rounds"
                  : allRoundsComplete
                    ? "All rounds analysed"
                    : "Analyse all rounds"}
            </button>
            <span className="quiet-label">{currentMatchVideos.length} ROUNDS</span>
          </div>
        </div>
        {currentMatchVideos.length === 0 ? (
          <p className="empty-state">
            No rounds yet. Upload a clip to create Round 1, then analyse it before moving on.
          </p>
        ) : (
          <>
            <div className="match-round-list">
              {currentMatchVideos.map((video) => {
                const round = roundAnalyses[video.id];
                const state = round?.state;
                const result = state?.status === "complete" ? state.results[0] : undefined;
                const status = round?.error
                  ? "Status unavailable"
                  : state?.status === "complete"
                    ? "Analysis complete"
                    : state?.status === "failed"
                      ? "Needs attention"
                      : state?.status === "stale"
                        ? "Needs re-analysis"
                        : state &&
                            ["queued", "preparing", "detecting", "classifying"].includes(
                              state.status,
                            )
                          ? `Analysing · ${state.progress}%`
                          : round
                            ? "Ready to analyse"
                            : "Loading saved result…";
                return (
                  <div className="round-card-row" key={video.id}>
                    <button
                      aria-pressed={selectedVideoId === video.id}
                      className={`round-card ${selectedVideoId === video.id ? "selected" : ""}`}
                      onClick={() => setSelectedVideoId(video.id)}
                      type="button"
                    >
                      <span className="round-card-number">Round {video.round_number}</span>
                      <span className="round-card-details">
                        <span className="round-card-filename">{video.filename}</span>
                        <span className="round-card-phase">
                          {(video.phase ?? "attack") === "attack" ? "Attack" : "Retake"}
                          {result &&
                            ` · TCN ${result.predictions.tcn.tactic} · GRU ${result.predictions.gru.tactic}`}
                          {round?.error && ` · ${round.error}`}
                        </span>
                      </span>
                      <span className="round-card-status">{status}</span>
                    </button>
                    <button
                      aria-label={`Remove Round ${video.round_number}`}
                      className="danger-button round-remove"
                      disabled={removingVideoId === video.id || resettingWorkspace}
                      onClick={() => void removeRecording(video)}
                      type="button"
                    >
                      {removingVideoId === video.id ? "Removing…" : "Remove"}
                    </button>
                  </div>
                );
              })}
            </div>
            <p className="inline-message">
              Analyse all uses the current playback speed and each round’s saved prepared-clip setting.
            </p>
          </>
        )}
      </section>

      {selectedVideo && (
        <>
          <section className="workspace-section video-section" aria-labelledby="video-title">
            <div className="section-heading">
              <div>
                <p className="eyebrow">04 / VIDEO REVIEW</p>
                <h2 id="video-title">
                  {isCurrentRound
                    ? `Round ${selectedVideo.round_number}: ${selectedVideo.filename}`
                    : selectedVideo.filename}
                </h2>
              </div>
              <span className="quiet-label">
                {isCurrentRound ? activeMatch.name.toUpperCase() : "LEGACY RECORDING"}
              </span>
              {!isCurrentRound && (
                <button
                className="danger-button"
                disabled={removingVideoId === selectedVideo.id || resettingWorkspace}
                onClick={() => void removeRecording(selectedVideo)}
                type="button"
                >
                {removingVideoId === selectedVideo.id ? "Removing…" : "Remove recording"}
                </button>
              )}
            </div>

            <div className="review-grid">
              <div className="player-column">
                <video
                  ref={videoRef}
                  className="video-player"
                  controls
                  onError={() =>
                    setVideoError(
                      "This MP4 could not be played. It may be damaged or its codec could not be converted for browser playback.",
                    )
                  }
                  onLoadedMetadata={(event) => {
                    setDuration(event.currentTarget.duration || 0);
                    setCurrentTime(event.currentTarget.currentTime);
                  }}
                  onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
                  preload="metadata"
                  src={selectedVideo.content_url}
                >
                  Your browser does not support MP4 video playback.
                </video>
                {videoError && (
                  <p className="error-message" role="alert">
                    {videoError}
                  </p>
                )}
                <div className="timeline">
                  <div className="timeline-track" aria-hidden="true">
                    <span className="timeline-progress" style={{ width: `${progress}%` }} />
                  </div>
                  <input
                    aria-label="Seek through recording"
                    className="seek-slider"
                    disabled={!duration}
                    max={duration || 0}
                    min={0}
                    onChange={(event) => seekTo(Number(event.target.value))}
                    step={0.1}
                    type="range"
                    value={Math.min(currentTime, duration)}
                  />
                  <div className="time-readout">
                    <span>{formatClock(currentTime)}</span>
                    <span>{formatClock(duration)}</span>
                  </div>
                </div>
              </div>

              <aside className="boundary-panel" aria-labelledby="boundary-title">
                <p className="eyebrow">
                  {isCurrentRound ? `ROUND ${selectedVideo.round_number}` : "PREVIOUS RECORDING"}
                </p>
                <h3 id="boundary-title">Clip boundaries are automatic</h3>
                <p className="boundary-help">
                  New round clips are analysed from the first frame to the last. Previous
                  recordings keep their saved segments; if none exist, the full clip is used.
                </p>
                <label className="field-label" htmlFor="playback-speed">
                  Demo replay speed
                </label>
                <select
                  id="playback-speed"
                  disabled={preparedClip}
                  onChange={(event) => setPlaybackSpeed(event.target.value as "1" | "2")}
                  value={playbackSpeed}
                >
                  <option value="2">2× (recording guide default)</option>
                  <option value="1">1× (normal speed)</option>
                </select>
                <label className="prepared-option">
                  <input
                    checked={preparedClip}
                    disabled={savingPreparedClipId === selectedVideo.id}
                    onChange={(event) =>
                      void savePreparedClipSetting(selectedVideo.id, event.target.checked)
                    }
                    type="checkbox"
                  />
                  <span>
                    {savingPreparedClipId === selectedVideo.id
                      ? "Saving prepared-clip setting…"
                      : "Already a prepared minimap clip (494×500, 10 fps)"}
                  </span>
                </label>
                <p className="boundary-help">
                  For a full-screen recording, add a matching local crop profile at
                  configs/recording-video.json. Check the prepared-clip option only for
                  already-cropped 494×500, 10 fps footage.
                </p>
                <button
                  className="primary-button analyze-button"
                  disabled={
                    !segments.length ||
                    startingAnalysis ||
                    analysisRunning ||
                    savingPreparedClipId === selectedVideo.id
                  }
                  onClick={() => void startAnalysis()}
                  type="button"
                >
                  {startingAnalysis
                    ? "Starting analysis..."
                    : analysisRunning
                      ? "Analysis running..."
                      : analysis?.status === "complete"
                        ? isCurrentRound
                          ? "Analyse this round again"
                          : "Run analysis again"
                        : isCurrentRound
                          ? `Analyse Round ${selectedVideo.round_number}`
                          : "Run detection & classify"}
                </button>
              </aside>
            </div>
          </section>
        </>
      )}

      <section className="workspace-section results-section" aria-labelledby="results-title">
        <div className="section-heading">
          <div>
            <p className="eyebrow">05 / CLASSIFICATION</p>
            <h2 id="results-title">All round classifications</h2>
          </div>
          <span className="quiet-label">
            {currentMatchVideos.filter(
              (video) => roundAnalyses[video.id]?.state?.status === "complete",
            ).length}{" "}
            / {currentMatchVideos.length} COMPLETE
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
        <div aria-labelledby={`tab-${activeTab}`} className="match-classifications" role="tabpanel">
          {currentMatchVideos.length === 0 ? (
            <p className="empty-state">Upload a round clip to see its TCN and GRU output here.</p>
          ) : (
            currentMatchVideos.map((video) => {
              const round = roundAnalyses[video.id];
              const state = round?.state;
              return (
                <article className="match-classification" key={video.id}>
                  <div className="match-classification-heading">
                    <div>
                      <p className="round-card-number">Round {video.round_number}</p>
                      <h3>{video.filename}</h3>
                    </div>
                    <span className="round-card-status">
                      {round?.error
                        ? "Status unavailable"
                        : state?.status === "complete"
                          ? "Analysis complete"
                          : state?.status === "failed"
                            ? "Needs attention"
                            : state?.status === "stale"
                              ? "Needs re-analysis"
                              : state &&
                                  ["queued", "preparing", "detecting", "classifying"].includes(
                                    state.status,
                                  )
                                ? `${state.message} · ${state.progress}%`
                                : "Not analysed"}
                    </span>
                  </div>
                  {state?.status === "complete" && state.results.length > 0 ? (
                    state.results.map((result) => {
                      const visibleWarnings = result.warnings.filter(
                        (warning) => !warning.startsWith("background from only "),
                      );
                      return (
                        <article className="result-segment" key={result.segment_id}>
                          <div className="result-segment-heading">
                            <span className={`phase-pill ${result.phase}`}>{result.phase}</span>
                            <span>
                              {result.detection.player_icon_observations} player-icon observations
                              · {result.detection.bomb_visible_frames}/{result.detection.frames}{" "}
                              frames with bomb visible
                            </span>
                          </div>
                          <div className="prediction-grid">
                            {(activeTab === "comparison" || activeTab === "tcn") && (
                              <PredictionCard model="TCN" prediction={result.predictions.tcn} />
                            )}
                            {(activeTab === "comparison" || activeTab === "gru") && (
                              <PredictionCard
                                model="Compact pooled GRU"
                                prediction={result.predictions.gru}
                              />
                            )}
                          </div>
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
                  ) : state?.status === "failed" ? (
                    <p className="error-message" role="alert">
                      {state.error ?? "Analysis failed. Correct the input and run it again."}
                    </p>
                  ) : round?.error ? (
                    <p className="error-message" role="alert">
                      {round.error}
                    </p>
                  ) : null}
                </article>
              );
            })
          )}
        </div>
      </section>

      <aside className="notice" aria-label="Analysis limitation">
        <span className="notice-mark" aria-hidden="true">
          i
        </span>
        <p>
          Each new clip is treated as one round, so use one round/tactic clip per upload.
          Automatic round detection inside a full match is not available. Model confidence
          and detector counts are review aids, not accuracy guarantees.
        </p>
      </aside>

      <footer className="page-footer">
        <span>Self-hosted · Recordings stay on this machine</span>
        <span>TCN and GRU output will be grouped by round</span>
      </footer>
    </main>
  );
}
