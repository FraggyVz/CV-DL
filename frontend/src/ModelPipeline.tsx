import { useEffect, useMemo, useRef, useState } from "react";
import type { MouseEvent, ReactNode } from "react";

// --------------------------------------------------------------------------- data

type Heatmap = { labels: string[]; values: number[][]; min: number; max: number };

type ModelStage = {
  input: Heatmap;
  input_width: number;
  pooled: { mean: number[]; max: number[] };
  peak_columns: number[];
  seeds: Record<string, number>[];
  ensemble: Record<string, number>;
};

type TcnStage = ModelStage & {
  resnet_pcs: Heatmap | null;
  blocks: { dilation: number; activations: Heatmap }[];
  architecture: {
    channels: number;
    kernel: number;
    dilations: number[];
    receptive_field_steps: number;
    receptive_field_s: number;
    step_s: number;
  };
};

type GruStage = ModelStage & {
  projection: Heatmap;
  hidden: Heatmap;
  architecture: { projection: number; hidden: number; readout: string; parameters: number };
};

export type PipelineRecord = {
  format: string;
  classes: string[];
  steps: number;
  columns: number;
  duration_s: number;
  column_times_s: number[];
  detection: {
    radar_image: string;
    positions: [number, number, number, number][];
    regions: { name: string; rects: number[][] }[];
    t_regions: Heatmap;
    ct_regions: Heatmap;
    t_reached: Heatmap;
    counts: { t_seen: number[]; ct_seen: number[]; bomb_planted_est: number[] };
    channels: number;
  };
  tcn: TcnStage;
  gru: GruStage;
};

const cache = new Map<string, Promise<PipelineRecord>>();

function loadPipeline(url: string): Promise<PipelineRecord> {
  let request = cache.get(url);
  if (!request) {
    request = fetch(url).then(async (response) => {
      if (!response.ok) {
        const body: unknown = await response.json().catch(() => null);
        const detail =
          body && typeof body === "object" && "detail" in body ? String(body.detail) : "";
        throw new Error(detail || `Request failed with HTTP ${response.status}.`);
      }
      return (await response.json()) as PipelineRecord;
    });
    request.catch(() => cache.delete(url));
    cache.set(url, request);
  }
  return request;
}

// --------------------------------------------------------------------------- helpers

const nice = (name: string) => name.replace(/_/g, " ");
const seconds = (s: number | undefined) => `${(s ?? 0).toFixed(1)} s`;

function argmax(probabilities: Record<string, number>, classes: string[]): string {
  return classes.reduce((a, b) => (probabilities[a] >= probabilities[b] ? a : b));
}

/** Rows that are non-zero somewhere, so empty regions don't take up space. */
function activeRows(data: Heatmap): Heatmap {
  const keep = data.values.map((row) => row.some((v) => v > 0.05));
  if (!keep.some(Boolean)) return data;
  return {
    ...data,
    labels: data.labels.filter((_, i) => keep[i]),
    values: data.values.filter((_, i) => keep[i]),
  };
}

/** The order the Ts first reached each region: [{region, time}] sorted by time. */
function route(reached: Heatmap, times: number[]) {
  return reached.labels
    .map((region, i) => {
      const column = reached.values[i].findIndex((v) => v >= 0.5);
      return { region, time: column >= 0 ? times[column] : null };
    })
    .filter((step): step is { region: string; time: number } => step.time !== null)
    .filter((step) => step.region !== "t_spawn")
    .sort((a, b) => a.time - b.time);
}

function firstTime(values: number[], times: number[], threshold = 0.5): number | null {
  const column = values.findIndex((v) => v >= threshold);
  return column >= 0 ? times[column] : null;
}

// --------------------------------------------------------------------------- colour

type Scale = "lime" | "t" | "ct" | "diverging";
const RGB = {
  ground: [20, 24, 21],
  lime: [200, 241, 105],
  t: [236, 166, 92],
  ct: [120, 176, 236],
  negative: [110, 160, 225],
};

function mix(a: number[], b: number[], f: number): [number, number, number] {
  const t = Math.max(0, Math.min(1, f));
  return [0, 1, 2].map((i) => Math.round(a[i] + (b[i] - a[i]) * t)) as [number, number, number];
}

function colour(scale: Scale, value: number, lo: number, hi: number): [number, number, number] {
  if (scale === "diverging") {
    const limit = 2.5;
    return value >= 0
      ? mix(RGB.ground, RGB.lime, value / limit)
      : mix(RGB.ground, RGB.negative, -value / limit);
  }
  const span = hi - lo || 1;
  return mix(RGB.ground, RGB[scale], (value - lo) / span);
}

// --------------------------------------------------------------------------- explanation

function Explain({ read, takeaway }: { read: string; takeaway?: ReactNode }) {
  return (
    <div className="viz-explain">
      <p>
        <span className="viz-explain-label">How to read</span>
        {read}
      </p>
      {takeaway && (
        <p className="viz-takeaway">
          <span className="viz-explain-label">This round</span>
          {takeaway}
        </p>
      )}
    </div>
  );
}

// --------------------------------------------------------------------------- charts

type Band = { label: string; rows: number };

function HeatmapChart({
  title,
  data,
  times,
  scale,
  showLabels = false,
  bands,
  height: fixedHeight,
}: {
  title: string;
  data: Heatmap;
  times: number[];
  scale: Scale;
  showLabels?: boolean;
  bands?: Band[];
  height?: number;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<{ row: number; col: number } | null>(null);
  const rows = data.values.length;
  const cols = rows ? data.values[0].length : 0;
  const lo = Math.min(0, data.min);

  useEffect(() => {
    const element = canvas.current;
    const context = element?.getContext("2d");
    if (!element || !context || !rows || !cols) return;
    element.width = cols;
    element.height = rows;
    const image = context.createImageData(cols, rows);
    data.values.forEach((row, r) =>
      row.forEach((value, c) => {
        const [red, green, blue] = colour(scale, value, lo, data.max);
        image.data.set([red, green, blue, 255], (r * cols + c) * 4);
      }),
    );
    context.putImageData(image, 0, 0);
  }, [data, rows, cols, scale, lo]);

  function track(event: MouseEvent<HTMLCanvasElement>) {
    const box = event.currentTarget.getBoundingClientRect();
    const col = Math.floor(((event.clientX - box.left) / box.width) * cols);
    const row = Math.floor(((event.clientY - box.top) / box.height) * rows);
    if (row >= 0 && row < rows && col >= 0 && col < cols) setHover({ row, col });
  }

  const labelled = showLabels || Boolean(bands);
  const height = fixedHeight ?? Math.max(46, Math.min(rows * (showLabels ? 14 : 3), 220));
  return (
    <figure className="viz-figure">
      <figcaption className="viz-title">
        <span>{title}</span>
        <span className="viz-readout">
          {hover
            ? `${nice(data.labels[hover.row])} · ${seconds(times[hover.col])} · ${data.values[hover.row][hover.col]}`
            : "hover for values"}
        </span>
      </figcaption>
      <div className={`heatmap ${labelled ? "with-labels" : ""}`}>
        {showLabels && (
          <ol className="heatmap-labels" style={{ height }}>
            {data.labels.map((label) => (
              <li key={label}>{nice(label)}</li>
            ))}
          </ol>
        )}
        {bands && (
          <ol className="heatmap-bands" style={{ height }}>
            {bands.map((band) => (
              <li key={band.label} style={{ flexGrow: band.rows }}>
                {band.label}
              </li>
            ))}
          </ol>
        )}
        <canvas
          aria-label={`${title} heatmap`}
          className="heatmap-canvas"
          onMouseLeave={() => setHover(null)}
          onMouseMove={track}
          ref={canvas}
          role="img"
          style={{ height }}
        />
      </div>
      <TimeAxis duration={times[times.length - 1] ?? 0} indent={labelled} />
    </figure>
  );
}

function TimeAxis({ duration, indent }: { duration: number; indent?: boolean }) {
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  return (
    <div className={`time-axis ${indent ? "indent" : ""}`} aria-hidden="true">
      {ticks.map((t) => (
        <span key={t} style={{ left: `${t * 100}%` }}>
          {(t * duration).toFixed(0)}s
        </span>
      ))}
    </div>
  );
}

function LineChart({
  series,
  times,
  max,
  marker,
}: {
  series: { name: string; values: number[]; className: string }[];
  times: number[];
  max: number;
  marker?: { time: number; label: string } | null;
}) {
  const width = 320;
  const height = 120;
  const n = times.length;
  const duration = times[n - 1] || 1;
  const x = (i: number) => (n > 1 ? (i / (n - 1)) * width : 0);
  const y = (v: number) => height - (v / (max || 1)) * (height - 8) - 4;
  return (
    <figure className="viz-figure">
      <figcaption className="viz-title">
        <span>Players visible over time</span>
        <span className="viz-legend">
          {series.map((s) => (
            <span className={`legend-key ${s.className}`} key={s.name}>
              {s.name}
            </span>
          ))}
          {marker && <span className="legend-key bomb">bomb planted</span>}
        </span>
      </figcaption>
      <div className="line-wrap">
        <svg className="line-chart" preserveAspectRatio="none" viewBox={`0 0 ${width} ${height}`}>
          {[0, 0.5, 1].map((f) => (
            <line className="grid-line" key={f} x1={0} x2={width} y1={y(f * max)} y2={y(f * max)} />
          ))}
          {marker && (
            <line
              className="plant-line"
              x1={(marker.time / duration) * width}
              x2={(marker.time / duration) * width}
              y1={0}
              y2={height}
            />
          )}
          {series.map((s) => (
            <polyline
              className={`line ${s.className}`}
              key={s.name}
              points={s.values.map((v, i) => `${x(i)},${y(v)}`).join(" ")}
            />
          ))}
        </svg>
        <span className="line-max">{max}</span>
      </div>
      <TimeAxis duration={duration} />
    </figure>
  );
}

function RadarOverlay({ detection }: { detection: PipelineRecord["detection"] }) {
  const [showRegions, setShowRegions] = useState(true);
  return (
    <figure className="viz-figure radar-figure">
      <figcaption className="viz-title">
        <span>Every detected player icon</span>
        <label className="viz-toggle">
          <input
            checked={showRegions}
            onChange={(event) => setShowRegions(event.target.checked)}
            type="checkbox"
          />
          Map regions
        </label>
      </figcaption>
      <div className="radar-frame">
        {detection.radar_image && <img alt="" src={detection.radar_image} />}
        <svg aria-label="Player icon positions" role="img" viewBox="0 0 1 1" preserveAspectRatio="none">
          {showRegions &&
            detection.regions.map((region) =>
              region.rects.map(([x0, y0, x1, y1], i) => (
                <rect
                  className="region-rect"
                  height={y1 - y0}
                  key={`${region.name}-${i}`}
                  width={x1 - x0}
                  x={x0}
                  y={y0}
                />
              )),
            )}
          {detection.positions.map(([t, x, y, team], i) => (
            <circle
              className={team === 0 ? "dot-t" : "dot-ct"}
              cx={x}
              cy={y}
              key={i}
              r={0.006}
              style={{ opacity: 0.25 + 0.75 * t }}
            />
          ))}
        </svg>
        {showRegions && (
          <div className="region-names" aria-hidden="true">
            {detection.regions.map((region) => {
              const [x0, y0] = region.rects[0];
              return (
                <span key={region.name} style={{ left: `${x0 * 100}%`, top: `${y0 * 100}%` }}>
                  {nice(region.name)}
                </span>
              );
            })}
          </div>
        )}
      </div>
      <p className="viz-caption">
        <span className="legend-key t">T</span> <span className="legend-key ct">CT</span> faint =
        early in the clip, bright = late
      </p>
    </figure>
  );
}

function RouteTimeline({ steps }: { steps: { region: string; time: number }[] }) {
  if (!steps.length) return <p className="viz-empty">The Ts did not clearly reach any region.</p>;
  return (
    <figure className="viz-figure">
      <figcaption className="viz-title">
        <span>T route: when each region was first reached</span>
      </figcaption>
      <ol className="route">
        {steps.map((step, i) => (
          <li key={step.region}>
            {i > 0 && <span className="route-arrow" aria-hidden="true">→</span>}
            <span className="route-stop">
              <strong>{nice(step.region)}</strong>
              <span>{seconds(step.time)}</span>
            </span>
          </li>
        ))}
      </ol>
    </figure>
  );
}

function PeakHistogram({ peaks, times, units }: { peaks: number[]; times: number[]; units: string }) {
  const top = Math.max(...peaks, 1);
  return (
    <figure className="viz-figure">
      <figcaption className="viz-title">
        <span>When each {units} hit its maximum</span>
        <span className="viz-readout">
          {peaks.reduce((a, b) => a + b, 0)} {units}s
        </span>
      </figcaption>
      <div className="bar-strip" role="img" aria-label="Max-pooling peaks over time">
        {peaks.map((count, i) => (
          <span
            key={i}
            style={{ height: `${(count / top) * 100}%` }}
            title={`${seconds(times[i])}: ${count}`}
          />
        ))}
      </div>
      <TimeAxis duration={times[times.length - 1] ?? 0} />
    </figure>
  );
}

function SeedBars({ stage, classes }: { stage: ModelStage; classes: string[] }) {
  const winner = argmax(stage.ensemble, classes);
  return (
    <figure className="viz-figure">
      <figcaption className="viz-title">
        <span>Probability per class</span>
        <span className="viz-legend">
          <span className="legend-key seeds">each seed</span>
          <span className="legend-key ensemble">average (final answer)</span>
        </span>
      </figcaption>
      <div className="seed-chart">
        {classes.map((c) => (
          <div className={`seed-row ${c === winner ? "winner" : ""}`} key={c}>
            <span className="seed-class">{c}</span>
            <div className="seed-track">
              <span className="seed-fill" style={{ width: `${stage.ensemble[c] * 100}%` }} />
              {stage.seeds.map((seed, i) => (
                <span
                  className="seed-tick"
                  key={i}
                  style={{ left: `${seed[c] * 100}%` }}
                  title={`seed ${i}: ${(seed[c] * 100).toFixed(1)}%`}
                />
              ))}
            </div>
            <span className="seed-value">{Math.round(stage.ensemble[c] * 100)}%</span>
          </div>
        ))}
      </div>
    </figure>
  );
}

function DilationDiagram({ dilations, kernel, stepS }: { dilations: number[]; kernel: number; stepS: number }) {
  // One output step traced down the stack, one convolution per block.
  const span = (kernel - 1) / 2;
  const layers: number[][] = [[0]];
  for (let d = dilations.length - 1; d >= 0; d--) {
    const below = new Set<number>();
    layers[0].forEach((p) => {
      for (let k = -span; k <= span; k++) below.add(p + k * dilations[d]);
    });
    layers.unshift([...below]);
  }
  const reach = Math.max(...layers[0].map(Math.abs));
  const width = 360;
  const rowGap = 24;
  const left = 46;
  const height = rowGap * dilations.length + 30;
  const x = (p: number) => left + (width - left) / 2 + (p / (reach || 1)) * ((width - left) / 2 - 8);
  const y = (layer: number) => height - 16 - layer * rowGap;
  // Receptive field after each block, both convolutions counted.
  let total = 1;
  const reachAfter = dilations.map((d) => (total += 2 * (kernel - 1) * d) * stepS);
  return (
    <figure className="viz-figure">
      <figcaption className="viz-title">
        <span>How far each block can see</span>
      </figcaption>
      <svg className="diagram" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Dilation diagram">
        {layers.slice(1).map((nodes, li) =>
          nodes.map((p) =>
            [-span, 0, span].map((k) => (
              <line
                className="diagram-edge"
                key={`${li}-${p}-${k}`}
                x1={x(p)}
                x2={x(p + k * dilations[li])}
                y1={y(li + 1)}
                y2={y(li)}
              />
            )),
          ),
        )}
        {layers.map((nodes, li) =>
          nodes.map((p) => (
            <circle className="diagram-node" cx={x(p)} cy={y(li)} key={`${li}-${p}`} r={li === layers.length - 1 ? 4 : 2.4} />
          )),
        )}
        <text className="diagram-label" x={0} y={y(0) + 3}>
          input
        </text>
        {dilations.map((d, i) => (
          <text className="diagram-label" key={d} x={0} y={y(i + 1) + 3}>
            block {i + 1}
          </text>
        ))}
        <text className="diagram-label" textAnchor="middle" x={x(0)} y={height - 2}>
          ← rows of the clip (0.2 s each) →
        </text>
      </svg>
      <ol className="reach-list">
        {dilations.map((d, i) => (
          <li key={d}>
            <strong>Block {i + 1}</strong> skips {d} row{d > 1 ? "s" : ""} · sees {reachAfter[i].toFixed(1)} s
          </li>
        ))}
      </ol>
    </figure>
  );
}

function GruDiagram({ stage }: { stage: GruStage }) {
  const boxes = [
    { label: `${stage.input_width}`, sub: "numbers in" },
    { label: `${stage.architecture.projection}`, sub: "projection" },
    { label: `${stage.architecture.hidden}`, sub: "GRU memory" },
    { label: `${stage.architecture.hidden * 2}`, sub: "mean + max" },
    { label: "5", sub: "classes" },
  ];
  const width = 360;
  const box = 54;
  const gap = (width - boxes.length * box) / (boxes.length - 1);
  const memory = 2 * (box + gap);
  return (
    <figure className="viz-figure">
      <figcaption className="viz-title">
        <span>What happens at every row</span>
        <span className="viz-readout">{stage.architecture.parameters.toLocaleString()} parameters</span>
      </figcaption>
      <svg className="diagram" viewBox={`0 0 ${width} 96`} role="img" aria-label="GRU diagram">
        {boxes.map((b, i) => {
          const x0 = i * (box + gap);
          return (
            <g key={b.sub}>
              {i > 0 && <line className="diagram-arrow" x1={x0 - gap + 2} x2={x0 - 3} y1={40} y2={40} />}
              <rect className={`diagram-box ${i === 2 ? "accent" : ""}`} height={36} rx={5} width={box} x={x0} y={22} />
              <text className="diagram-box-label" textAnchor="middle" x={x0 + box / 2} y={44}>
                {b.label}
              </text>
              <text className="diagram-label" textAnchor="middle" x={x0 + box / 2} y={74}>
                {b.sub}
              </text>
            </g>
          );
        })}
        <path className="diagram-loop" d={`M ${memory + 12} 22 C ${memory + 12} 2, ${memory + box - 12} 2, ${memory + box - 12} 22`} />
        <text className="diagram-label" textAnchor="middle" x={memory + box / 2} y={92}>
          memory passes to the next row
        </text>
      </svg>
    </figure>
  );
}

function Flow({ model, record }: { model: "tcn" | "gru"; record: PipelineRecord }) {
  const stage = model === "tcn" ? record.tcn : record.gru;
  const steps =
    model === "tcn"
      ? ["Radar clip", "Detector", `${stage.input_width} numbers / row`, "5 dilated blocks", "Mean + max", "3 seeds", "Prediction"]
      : ["Radar clip", "Detector", `${stage.input_width} numbers / row`, "GRU memory", "Mean + max", "3 seeds", "Prediction"];
  return (
    <ol className="viz-flow" aria-label="Pipeline overview">
      {steps.map((step, i) => (
        <li key={step} className={i === 3 ? "model" : ""}>
          {step}
        </li>
      ))}
    </ol>
  );
}

// --------------------------------------------------------------------------- layout

function Stage({
  number,
  title,
  text,
  children,
}: {
  number: string;
  title: string;
  text: string;
  children: ReactNode;
}) {
  return (
    <section className="viz-stage">
      <header className="viz-stage-heading">
        <span className="walkthrough-step-number">{number}</span>
        <div>
          <h5>{title}</h5>
          <p>{text}</p>
        </div>
      </header>
      <div className="viz-stage-body">{children}</div>
    </section>
  );
}

export default function ModelPipeline({
  model,
  url,
  available,
  unavailableReason,
}: {
  model: "tcn" | "gru";
  url: string;
  available: boolean;
  unavailableReason?: string;
}) {
  const [record, setRecord] = useState<PipelineRecord | null>(null);
  const [error, setError] = useState("");
  const [allBlocks, setAllBlocks] = useState(false);

  useEffect(() => {
    if (!available) return;
    let active = true;
    setError("");
    loadPipeline(url)
      .then((data) => active && setRecord(data))
      .catch((reason: Error) => active && setError(reason.message));
    return () => {
      active = false;
    };
  }, [url, available]);

  const summary = useMemo(() => {
    if (!record) return null;
    const times = record.column_times_s;
    const { t_seen, ct_seen, bomb_planted_est } = record.detection.counts;
    const stage = model === "tcn" ? record.tcn : record.gru;
    const peakColumn = stage.peak_columns.indexOf(Math.max(...stage.peak_columns));
    const seedAnswers = stage.seeds.map((seed) => argmax(seed, record.classes));
    return {
      times,
      maxT: Math.round(Math.max(...t_seen)),
      maxCt: Math.round(Math.max(...ct_seen)),
      countMax: Math.max(5, Math.ceil(Math.max(...t_seen, ...ct_seen))),
      plant: firstTime(bomb_planted_est, times),
      route: route(record.detection.t_reached, times),
      peakTime: times[peakColumn],
      seedAnswers,
      winner: argmax(stage.ensemble, record.classes),
    };
  }, [record, model]);

  if (!available) {
    return (
      <p className="viz-empty">
        {unavailableReason
          ? `The pipeline view could not be built for this round (${unavailableReason}).`
          : "This round was analysed before the pipeline view existed. Analyse it again to see each stage."}
      </p>
    );
  }
  if (error) return <p className="error-message">{error}</p>;
  if (!record || !summary) return <p className="viz-empty">Loading the pipeline record…</p>;

  const d = record.detection;
  const times = summary.times;
  const stage = model === "tcn" ? record.tcn : record.gru;
  const name = model === "tcn" ? "TCN" : "Compact pooled GRU";
  const tcn = record.tcn;
  const gru = record.gru;
  const resnetWidth = stage.input_width - d.channels;
  const agree = summary.seedAnswers.every((a) => a === summary.winner);
  const shownBlocks = allBlocks ? tcn.blocks : [tcn.blocks[0], tcn.blocks[tcn.blocks.length - 1]];

  return (
    <div className="model-pipeline" aria-label={`${name} pipeline visualisation`}>
      <div className="walkthrough-heading">
        <p className="eyebrow">INSIDE THIS PREDICTION</p>
        <h4>
          How the {name} reached {summary.winner} · {record.duration_s} s clip, {record.steps} rows
        </h4>
      </div>
      <Flow model={model} record={record} />

      <Stage
        number="01"
        text="A rule-based detector (colour and shape rules, no training) finds every T icon, CT icon and the bomb in each frame of the radar."
        title="Find the players"
      >
        <div className="viz-pair">
          <RadarOverlay detection={d} />
          <LineChart
            marker={summary.plant !== null ? { time: summary.plant, label: "plant" } : null}
            max={summary.countMax}
            series={[
              { name: "T", values: d.counts.t_seen, className: "t" },
              { name: "CT", values: d.counts.ct_seen, className: "ct" },
            ]}
            times={times}
          />
        </div>
        <Explain
          read="Left: each dot is one detected icon, drawn on the radar with the map regions the model knows. Right: how many players of each side were visible at each moment."
          takeaway={
            <>
              Up to {summary.maxT} T and {summary.maxCt} CT visible at once.{" "}
              {summary.plant !== null
                ? `Bomb plant estimated at ${seconds(summary.plant)}.`
                : "No bomb plant detected."}
            </>
          }
        />
      </Stage>

      <Stage
        number="02"
        text={`The positions become ${d.channels} map signals every 0.2 s, such as how many Ts are in each region and which regions they have reached. Regions nobody entered are hidden.`}
        title="Turn positions into map signals"
      >
        <RouteTimeline steps={summary.route} />
        <div className="viz-pair">
          <HeatmapChart data={activeRows(d.t_regions)} scale="t" showLabels times={times} title="Ts in each region" />
          <HeatmapChart data={activeRows(d.ct_regions)} scale="ct" showLabels times={times} title="CTs in each region" />
        </div>
        <Explain
          read="Each heatmap row is a map region and time runs left to right; brighter means more players there. The route above lists the order the Ts first got to each region, which is what separates a direct attack from a split."
          takeaway={
            summary.route.length
              ? `Route: ${summary.route.map((s) => `${nice(s.region)} (${seconds(s.time)})`).join(" → ")}.`
              : "The Ts did not clearly reach a region in this clip."
          }
        />
      </Stage>

      <Stage
        number="03"
        text={`Each 0.2 s row becomes ${stage.input_width} numbers: the ${d.channels} map signals plus ${resnetWidth} image features from a frozen ResNet-18, all scaled to the training data.`}
        title="Build the input sequence"
      >
        <HeatmapChart
          bands={[
            { label: `${d.channels} map signals`, rows: d.channels },
            { label: `${resnetWidth} ResNet image features`, rows: resnetWidth },
          ]}
          data={stage.input}
          height={170}
          scale="diverging"
          times={times}
          title={`What the ${name} reads`}
        />
        <Explain
          read="One column per moment, one row per number. Lime means higher than usual in training, blue lower, dark about average. This whole picture is the model's only input."
        />
      </Stage>

      {model === "tcn" ? (
        <Stage
          number="04"
          text={`The TCN slides small filters along time. Each of its ${tcn.blocks.length} blocks skips further between the rows it compares, so later blocks see longer stretches of play: about ${tcn.architecture.receptive_field_s} s by the end.`}
          title="Look for patterns over time"
        >
          <div className="viz-pair">
            <DilationDiagram
              dilations={tcn.architecture.dilations}
              kernel={tcn.architecture.kernel}
              stepS={tcn.architecture.step_s}
            />
            <div className="viz-stack">
              {shownBlocks.map((block) => {
                const index = tcn.blocks.indexOf(block);
                return (
                  <HeatmapChart
                    data={block.activations}
                    height={80}
                    key={block.dilation}
                    scale="lime"
                    times={times}
                    title={`Block ${index + 1} output (${tcn.architecture.channels} pattern detectors)`}
                  />
                );
              })}
              <button className="viz-more" onClick={() => setAllBlocks(!allBlocks)} type="button">
                {allBlocks ? "Show first and last block only" : `Show all ${tcn.blocks.length} blocks`}
              </button>
            </div>
          </div>
          <Explain
            read="Left: one output (top dot) traced back to the input rows it depends on; each block doubles the gap. Right: each row is one learned pattern detector, bright where its pattern appears. Block 1 reacts to moments, the last block to whole sequences of movement."
          />
        </Stage>
      ) : (
        <Stage
          number="04"
          text={`The GRU reads the rows in order, one at a time. Each row is squeezed to ${gru.architecture.projection} numbers, then a ${gru.architecture.hidden}-number memory is updated, keeping what matters from earlier in the clip.`}
          title="Read the clip in order, with memory"
        >
          <GruDiagram stage={gru} />
          <HeatmapChart
            data={gru.hidden}
            scale="diverging"
            showLabels
            times={times}
            title={`The GRU's memory (${gru.architecture.hidden} units) as it reads the clip`}
          />
          <Explain
            read="Each row is one memory unit, read left to right as the GRU moves through the clip. Sudden colour changes mark moments that changed what the model remembers, such as a group entering a new region."
          />
        </Stage>
      )}

      <Stage
        number="05"
        text="To get one answer for the whole clip, every channel is summarised by its average and its highest value over time. Clips of any length then give the same-sized summary."
        title="Summarise the whole clip"
      >
        <PeakHistogram
          peaks={stage.peak_columns}
          times={times}
          units={model === "tcn" ? "pattern detector" : "memory unit"}
        />
        <Explain
          read="Each bar counts how many channels reached their highest value at that moment. Tall bars are the moments the max summary comes from. This shows when, not why."
          takeaway={`Most channels peaked around ${seconds(summary.peakTime)}.`}
        />
      </Stage>

      <Stage
        number="06"
        text={`A final layer turns the summary into five probabilities. ${stage.seeds.length} copies of the model, trained from different random starts (seeds), each vote, and their probabilities are averaged.`}
        title="Vote and average"
      >
        <SeedBars classes={record.classes} stage={stage} />
        <Explain
          read="The filled bar is the averaged probability, the final answer. The small ticks are the individual seeds; ticks close together mean the copies agree."
          takeaway={
            agree
              ? `All ${stage.seeds.length} seeds chose ${summary.winner}; the average gives it ${Math.round(stage.ensemble[summary.winner] * 100)}%.`
              : `The seeds disagreed (${summary.seedAnswers.join(", ")}); the average picks ${summary.winner} at ${Math.round(stage.ensemble[summary.winner] * 100)}%.`
          }
        />
      </Stage>

      <p className="walkthrough-note">
        Every value above is this clip's own data at each stage (model internals from seed 0).
        They show how the result was computed, not which detail caused it.
      </p>
    </div>
  );
}
