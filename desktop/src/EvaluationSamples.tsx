import {useEffect, useMemo, useRef, useState, type KeyboardEvent, type ReactNode} from "react";
import {ApiError} from "./api";
import type {Category, Dataset, Evaluation, Sample, TruthBoxes} from "./evaluationSource";
import {useMeasurementStore} from "./MeasurementJobs";
import {categoryLabels, largestIouDrop, predicates, textAlignment} from "./measurementModel";
import {paintMask} from "./evaluationMask";
import {count} from "./tensorTree";
import {shortId} from "./traceModel";

type Overlays = {mask: boolean; before: boolean};
type Art = {image?: string; error?: string; boxes?: TruthBoxes; overlayError?: string};
const number = (value: number | null | undefined) => value == null ? "—" : value.toFixed(2);
const className = (dataset: Dataset, name: number | string | null | undefined) => typeof name === "number" ? dataset.classes?.[name] ?? String(name) : name ?? "—";
const errorText = (error: unknown) => error instanceof ApiError ? "HTTP " + error.status : "decode failed";

export function sampleLabel(dataset: Dataset, row: Sample) {
  const prefix = "Sample " + shortId(row.sample_id) + ", " + categoryLabels[row.category] + ", ";
  if (dataset.task === "classification") return prefix + "truth " + className(dataset, row.label) + ", before " +
    className(dataset, row.before.pred as number) + ", after " + className(dataset, row.after.pred as number);
  if (dataset.task === "ocr") return prefix + "truth " + row.label + ", before " + row.before.text + ", after " + row.after.text;
  return prefix + "score " + number(row.score_before) + " to " + number(row.score_after);
}

function BoxOverlay({truth, row, before}: {truth: TruthBoxes; row: Sample; before: boolean}) {
  const predicted = (before ? row.before : row.after).boxes ?? [];
  return <svg className="ev-overlay ev-boxes" viewBox={`0 0 ${truth.width} ${truth.height}`} aria-hidden="true">
    {truth.boxes.map(({bbox: [x,y,width,height], category_id}, i) =>
      <rect key={i} className="ev-truth-box" {...{x,y,width,height}} vectorEffect="non-scaling-stroke"><title>Truth {category_id}</title></rect>)}
    {predicted.map(([x,y,x2,y2,score,category], i) =>
      <rect key={i} className="ev-pred-box" x={x} y={y} width={Math.max(0,x2-x)} height={Math.max(0,y2-y)} vectorEffect="non-scaling-stroke">
        <title>{before ? "Before" : "After"} {category} · {number(score)}</title></rect>)}
  </svg>;
}

function SampleImage({dataset, row, size, mask, before, onLabel, onBoxes}: {dataset: Dataset; row: Sample; size: number;
  onLabel: (label: string) => void; onBoxes?: (value: TruthBoxes) => void} & Overlays) {
  const {source} = useMeasurementStore();
  const [art, setArt] = useState<Art>({});
  const [dimensions, setDimensions] = useState<[number, number]>();
  const [visible, setVisible] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const sample = row.sample_id;
  const datasetId = dataset.dataset_id;
  const task = dataset.task;
  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {setVisible(true); observer.disconnect();}
    }, {rootMargin: "120px"});
    observer.observe(root.current!);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (!visible) return;
    const controller = new AbortController();
    let url: string | undefined;
    const {signal} = controller;
    void source.sample(datasetId, sample, "image", signal).then(response => response.blob()).then(blob => {
      if (signal.aborted) return;
      url = URL.createObjectURL(blob);
      setArt(old => ({...old, image: url}));
    }).catch(error => {if (!signal.aborted) setArt(old => ({...old, error: errorText(error)}));});
    return () => {controller.abort(); if (url) URL.revokeObjectURL(url);};
  }, [datasetId, sample, source, visible]);
  useEffect(() => {
    if (!visible || art.error || task === "classification" || (task === "segmentation" && !mask)) return;
    const controller = new AbortController(), {signal} = controller;
    void source.sample(datasetId, sample, "label", signal).then(async response => {
      if (task === "segmentation") {
        const bytes = await response.arrayBuffer();
        if (!signal.aborted && canvas.current) await paintMask(canvas.current, bytes, response.headers.get("Content-Type") ?? "image/png", dataset.ignore_index ?? 255, signal);
      } else if (task === "detection") {
        const boxes: TruthBoxes = await response.json();
        if (!signal.aborted) {setArt(old => ({...old, boxes})); onBoxes?.(boxes);}
      } else {
        const label = await response.text();
        if (!signal.aborted) onLabel(label);
      }
    }).catch(error => {if (!signal.aborted) setArt(old => ({...old, overlayError: errorText(error)}));});
    return () => controller.abort();
  }, [datasetId, sample, task, source, visible, mask, dataset.ignore_index, art.error, onLabel, onBoxes]);
  const ratio = dimensions ? size / Math.max(...dimensions) : 1;
  const pixelated = !!dimensions && Math.max(...dimensions) <= 64;
  const scale = pixelated && ratio >= 1 ? Math.floor(ratio) : ratio;
  return <div ref={root} className="ev-image-container" style={{"--ev-image-size": size + "px"} as React.CSSProperties}>
    {art.error ? <div className="ev-image-error">Image unavailable · {art.error}</div> :
      <div className="ev-image-stack" style={dimensions ? {width: dimensions[0] * scale, height: dimensions[1] * scale} : undefined}>
        {art.image && <img src={art.image} alt={sampleLabel(dataset, row)} data-pixelated={pixelated}
          onLoad={e => setDimensions([e.currentTarget.naturalWidth, e.currentTarget.naturalHeight])}
          onError={() => setArt(old => ({...old, error: "decode failed"}))}/>}
        {task === "segmentation" && <canvas ref={canvas} className="ev-overlay" hidden={!mask || !dimensions || !!art.overlayError} aria-hidden="true"/>}
        {dimensions && art.boxes && <BoxOverlay truth={art.boxes} row={row} before={before}/>}
      </div>}
    {art.overlayError && !art.error && <span className="ev-overlay-error">Truth overlay unavailable · {art.overlayError}</span>}
  </div>;
}

function Fields({dataset, row, full, truth, boxes}: {dataset: Dataset; row: Sample; full?: boolean; truth?: string; boxes?: TruthBoxes}) {
  const [before, after] = [row.before, row.after];
  const diff = useMemo(() => dataset.task === "ocr" ? textAlignment(before.text ?? "", after.text ?? "") : undefined, [dataset.task, before.text, after.text]);
  const drop = dataset.task === "segmentation" ? largestIouDrop(row) : undefined;
  let description: ReactNode;
  switch (dataset.task) {
    case "classification": description = <>Truth {className(dataset, row.label)} · {className(dataset, before.pred as number)} → {className(dataset, after.pred as number)}</>; break;
    case "ocr": {
      const spelling = (parts: NonNullable<typeof diff>["before"]) => parts.map(({text, changed}, i) => changed ? <mark key={i}>{text}</mark> : text);
      description = <>Truth {truth ?? row.label}<br/>Before {spelling(diff!.before)}<br/>After {spelling(diff!.after)}</>;
      break;
    }
    case "detection": description = <>{before.boxes?.length ?? 0} → {after.boxes?.length ?? 0} predicted boxes
      {before.tp !== undefined && after.tp !== undefined && <> · TP {before.tp} → {after.tp}</>}
      {before.fp !== undefined && after.fp !== undefined && <> · FP {before.fp} → {after.fp}</>}</>;
      break;
    case "segmentation": description = drop ? <>{className(dataset, drop.index)} {drop.delta.toFixed(2)}</> : null;
  }
  const metric = {classification: "Accuracy", segmentation: "mIoU", detection: "F1", ocr: "CER"}[dataset.task];
  return <div className="ev-sample-fields">{description && <div>{description}</div>}
    {(full || dataset.task !== "classification") && <div>{metric} {number(row.score_before)} → {number(row.score_after)}</div>}
    {full && row.condition && <div>Condition {row.condition}</div>}
    {full && dataset.task === "detection" && <details><summary>Truth and predicted boxes</summary>
      <pre>{JSON.stringify({truth: boxes?.boxes, before: before.boxes, after: after.boxes}, null, 2)}</pre></details>}
    {full && dataset.task === "segmentation" && <details><summary>Per-class pixel counts</summary>
      <pre>{JSON.stringify({before, after}, null, 2)}</pre></details>}
  </div>;
}

function SampleContents({dataset, row, size, full, ...overlays}: {dataset: Dataset; row: Sample; size: number; full?: boolean} & Overlays) {
  const [truth, setTruth] = useState<string>();
  const [boxes, setBoxes] = useState<TruthBoxes>();
  return <><SampleImage {...{dataset,row,size}} {...overlays} onLabel={setTruth} onBoxes={full ? setBoxes : undefined}/>
    <Fields {...{dataset,row,truth,full,boxes}}/></>;
}

export function SampleInspector({dataset, row, overlays, setOverlays}: {dataset: Dataset; row: Sample; overlays: Overlays; setOverlays: (value: Overlays) => void}) {
  const [copied, setCopied] = useState("");
  return <aside className="tv-inspector" aria-label="Sample inspector"><section className="tv-section">
    <h2>Sample {shortId(row.sample_id)}</h2><p>Shown to you only.</p>
    <SampleContents dataset={dataset} row={row} size={368} {...overlays} full/>
    <OverlayControls {...{dataset, overlays, setOverlays}}/>
    <code>{row.sample_id}</code><button type="button" onClick={() => void navigator.clipboard.writeText(row.sample_id)
      .then(() => setCopied("Sample ID copied.")).catch(() => setCopied("Clipboard unavailable."))}>Copy sample ID</button>
    <span className="visually-hidden" role="status">{copied}</span>

  </section></aside>;
}

function OverlayControls({dataset, overlays, setOverlays}: {dataset: Dataset; overlays: Overlays; setOverlays: (value: Overlays) => void}) {
  if (dataset.task === "segmentation") return <button type="button" aria-pressed={overlays.mask} onClick={() => setOverlays({...overlays, mask: !overlays.mask})}>Truth mask</button>;
  if (dataset.task === "detection") return <button type="button" aria-pressed={overlays.before} onClick={() => setOverlays({...overlays, before: !overlays.before})}>Boxes: {overlays.before ? "Before" : "After"}</button>;
  return null;
}
export const categories: Category[] = ["broken", "fixed", "changed"];

export function EvaluationSamples({job, dataset, category, setCategory, selected, select, overlays, setOverlays}:
  {job: Evaluation; dataset: Dataset; category: Category; setCategory: (value: Category) => void; selected?: Sample;
    select: (row?: Sample) => void; overlays: Overlays; setOverlays: (value: Overlays) => void}) {
  const {source} = useMeasurementStore();
  const [rows, setRows] = useState<Sample[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [total, setTotal] = useState<number>();
  const [sort, setSort] = useState("score_change");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [reissue, setReissue] = useState(0);
  const [large, setLarge] = useState(false);
  const [truth, setTruth] = useState("");
  const [condition, setCondition] = useState("");
  const [distance, setDistance] = useState(0);
  const request = useRef<AbortController | null>(null);
  const grid = useRef<HTMLDivElement>(null);
  function load(next?: string) {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setLoading(true); setError("");
    void source.changes(job.evaluation_id, {category, sort, limit: 60, cursor: next}, controller.signal).then(page => {
      if (controller.signal.aborted) return;
      setRows(old => next ? [...old, ...page.changes] : page.changes);
      setCursor(page.cursor ?? null); setTotal(page.total);
    }).catch(err => {if (!controller.signal.aborted) setError(err instanceof Error ? err.message : "Sample results unavailable.");})
      .finally(() => {if (!controller.signal.aborted) setLoading(false);});
  }
  useEffect(() => {
    setRows([]); setCursor(null); setTotal(undefined); select(); load();
    return () => request.current?.abort();
  }, [category, sort, job.evaluation_id, source, reissue]);
  const conditions = [...new Set([...rows.flatMap(row => row.condition ? [row.condition] : []), ...(condition ? [condition] : [])])].sort();
  const filtered = rows.filter(row => (!truth || String(row.label) === truth) && (!condition || row.condition === condition) &&
    (!distance || Math.round(row.score_after * Math.max(1, Array.from(String(row.label ?? "")).length)) >= distance));
  const focusId = filtered.find(row => row.sample_id === selected?.sample_id)?.sample_id ?? filtered[0]?.sample_id;
  const size = large ? 160 : 96;
  function moveFocus(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    const buttons = Array.from(grid.current?.querySelectorAll("button") ?? []);
    const y = buttons[0]?.offsetTop;
    const perRow = buttons.filter(button => button.offsetTop === y).length;
    const step = {ArrowDown: perRow, ArrowUp: -perRow, ArrowLeft: -1, ArrowRight: 1}[event.key];
    if (step === undefined) return;
    event.preventDefault();
    buttons[Math.max(0, Math.min(buttons.length - 1, index + step))]?.focus();
  }
  if (job.baseline_id === null) return <p className="tv-empty">No baseline: absolute metrics only.</p>;
  return <div className="ev-samples">
    <p className="ev-pixel-notice">Images are shown to you only. The agent saw sample IDs, labels and conditions, never pixels.</p>
    <div className="ev-actions" aria-label="Sample categories">
      {categories.map(kind => <button key={kind} type="button" aria-pressed={kind === category} onClick={() => setCategory(kind)}>
        {categoryLabels[kind]} {count(job.counts?.[kind] ?? 0)}</button>)}
      <details><summary>Correctness</summary><p>{predicates[dataset.task]}</p>
        <p>Changed means the record changed and its correctness stayed the same.</p></details>
    </div>
    <div className="ev-actions" aria-label="Sample filters">
      {!!dataset.classes?.length && dataset.task === "classification" && <select aria-label="Filter by truth class" value={truth} onChange={e => setTruth(e.target.value)}>
        <option value="">All truth classes</option>{dataset.classes.map(label => <option key={label}>{label}</option>)}</select>}
      <select aria-label="Filter by condition" value={condition} onChange={e => setCondition(e.target.value)}><option value="">All conditions</option>
        {conditions.map(value => <option key={value}>{value}</option>)}</select>
      {dataset.task === "ocr" && <label>Minimum edit distance <input type="number" min="0" value={distance} onChange={e => setDistance(Math.max(0, Number(e.target.value)))}/></label>}
      <select aria-label="Sort samples" value={sort} onChange={e => setSort(e.target.value)}><option value="score_change">Severity</option><option value="sample_id">Sample ID</option></select>
      <button type="button" aria-pressed={large} onClick={() => setLarge(!large)}>Large</button>
      <OverlayControls {...{dataset, overlays, setOverlays}}/>
    </div>
    {error && <div role="alert" className="tv-failure">{error}<button type="button" onClick={() => setReissue(n => n + 1)}>Retry</button></div>}
    {!loading && !rows.length && !error && <p className="tv-empty">No {category === "changed" ? "changed" : category} samples.
      {job.counts && <> {count(job.counts.fixed)} fixed, {count(job.counts.broken)} broken, {count(job.counts.changed)} changed.</>}</p>}
    {rows.length > 0 && !filtered.length && <p className="tv-empty">No matching samples in the loaded pages.</p>}
    <div className="ev-sample-grid" role="grid" aria-label="Changed samples" ref={grid} style={{"--ev-tile": size + "px"} as React.CSSProperties}>
      {filtered.map((row, i) => <div key={row.sample_id} role="row"><div role="gridcell" aria-selected={row.sample_id === selected?.sample_id}>
        <button type="button" className="ev-sample-tile" data-category={row.category} tabIndex={row.sample_id === focusId ? 0 : -1}
          aria-label={sampleLabel(dataset, row)} onKeyDown={e => moveFocus(e, i)} onFocus={() => select(row)} onClick={() => select(row)}>
          <SampleContents dataset={dataset} row={row} size={size} {...overlays}/>
        </button></div></div>)}
    </div>
    {loading && <p role="status" className="tv-empty">Loading samples…</p>}
    {total !== undefined && <p className="tv-empty">Showing {count(filtered.length)} of {count(total)} {category}{filtered.length !== rows.length ? " · filters on loaded pages" : ""}
      {cursor && <button type="button" disabled={loading} onClick={() => load(cursor)}>Load more</button>}</p>}
  </div>;
}
