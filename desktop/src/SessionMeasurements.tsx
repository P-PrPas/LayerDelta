import {useEffect, useRef, useState, type ReactNode} from "react";
import type {Category, Dataset, Evaluation, Intervention, Metric, Probe, ProbeNode, Sample} from "./evaluationSource";
import {JobChip, useMeasurementStore} from "./MeasurementJobs";
import {categories, EvaluationSamples} from "./EvaluationSamples";
import {categoryLabels, intervalDomain, metricNumber, originOf, reading} from "./measurementModel";
import {shortId} from "./traceModel";
import {count} from "./tensorTree";
import type {TraceEvent} from "./traceSource";
import type {TraceActions} from "./TraceInspector";
import {timeText} from "./TraceTimeline";

export type DetailView = "metrics" | "samples";
export type EvalLocation = {id: string; view: DetailView; category: Category};
export type SelectedSample = {dataset: Dataset; row: Sample};
export type EvalOpen = (id: string, view?: DetailView, category?: Category) => void;

export function useMeasurementDatasets(events: TraceEvent[]) {
  const {source} = useMeasurementStore();
  const [data, setData] = useState(new Map<string, Dataset>());
  const attempted = useRef(new Set<string>());
  const request = useRef<AbortController>(null);
  useEffect(() => {
    const controller = new AbortController();
    request.current = controller;
    return () => {controller.abort(); attempted.current.clear();};
  }, [source]);
  useEffect(() => {
    const signal = request.current?.signal;
    if (!signal) return;
    for (const event of events) {
      const id = event.dataset_id;
      if (!id || attempted.current.has(id)) continue;
      attempted.current.add(id);
      void source.dataset(id, signal).then(record => {
        if (!signal.aborted) setData(previous => new Map(previous).set(id, record));
      }).catch(() => {});
    }
  }, [events, source]);
  return data;
}

function useNode(checkpoint?: string, node?: string) {
  const {source} = useMeasurementStore();
  const [value, setValue] = useState<{key: string; node: ProbeNode}>();
  const key = checkpoint + "/" + node;
  useEffect(() => {
    if (!checkpoint || !node) return;
    const controller = new AbortController();
    void source.node(checkpoint, node, controller.signal).then(result => {
      if (!controller.signal.aborted) setValue({key, node: result});
    }).catch(() => {});
    return () => controller.abort();
  }, [checkpoint, node, key, source]);
  return value?.key === key ? value.node : undefined;
}

export function InterventionText({job, event}: {job?: Evaluation; event: TraceEvent}) {
  const {source} = useMeasurementStore();
  const [resolved, setResolved] = useState<{id: string; value: Intervention}>();
  const id = event.intervention_id;
  useEffect(() => {
    if (!id || job?.intervention) return;
    const controller = new AbortController();
    void source.intervention(id, controller.signal).then(result => {
      if (result && !controller.signal.aborted) setResolved({id, value: result});
    }).catch(() => {});
    return () => controller.abort();
  }, [id, job?.intervention, source]);
  const intervention = job?.intervention ?? (resolved?.id === id ? resolved?.value : undefined);
  const node = useNode(intervention ? job?.checkpoint_id ?? event.checkpoint_id ?? undefined : undefined, intervention?.node_id);
  return <>{intervention ? <span>{intervention.group ?? node?.module_path ?? intervention.node_id} · units {intervention.units.slice(0, 12).join(", ")}
    {intervention.units.length > 12 && " … (" + count(intervention.units.length) + ")"} · {intervention.op}{intervention.factor != null && " " + intervention.factor}</span> :
    id ? <span title={id}>{shortId(id)}{event.units != null && " · " + count(event.units) + " units"}</span> : "none"}</>;
}

function useDataset(id: string | undefined, supplied?: Dataset) {
  const {source} = useMeasurementStore();
  const [fallback, setFallback] = useState<Dataset>();
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    if (!id || supplied) return;
    const controller = new AbortController();
    setFailed(false);
    void source.dataset(id, controller.signal).then(d => {if (!controller.signal.aborted) setFallback(d);})
      .catch(() => {if (!controller.signal.aborted) setFailed(true);});
    return () => controller.abort();
  }, [id, supplied, source]);
  return {dataset: supplied ?? (fallback?.dataset_id === id ? fallback : undefined), failed};
}

function IntervalPlot({metric, extent}: {metric: Metric; extent: number}) {
  const pos = (x: number) => 60 + (metric.better === "lower" ? -1 : 1) * (x / (extent || 1)) * 54;
  const color = reading(metric);
  return <svg className="ev-interval" viewBox="0 0 120 14" width="120" height="14" aria-hidden="true" data-reading={color}>
    <path className="ev-guide" d="M4 7h112 M60 1v12"/>
    {metric.ci && <line stroke="currentColor" strokeWidth="2" x1={pos(metric.ci[0])} x2={pos(metric.ci[1])} y1="7" y2="7"/>}
    {metric.delta !== null && <circle fill="currentColor" cx={pos(metric.delta)} cy="7" r="3"/>}
  </svg>;
}
function MetricTable({metrics, dataset, absolute = false, extent}: {metrics: Metric[]; dataset?: Dataset; absolute?: boolean;
  extent?: ReturnType<typeof intervalDomain>}) {
  const sorted = [...metrics].sort((a,b) => Number(b.primary) - Number(a.primary));
  const domain = extent ?? intervalDomain(sorted);
  return <div className="ev-metric-scroll"><table className="ev-metrics">
    <thead><tr><th>Metric</th>{!absolute && <th>Baseline</th>}<th>Checkpoint</th>{!absolute && <>
      <th>Delta</th><th>95% interval</th><th>Better →</th><th>Reading</th></>}</tr></thead>
    <tbody>{sorted.map(m => {
      const row = m as Metric & {class?: number; n?: number};
      const label = (index: number) => dataset?.classes?.[index] ?? index;
      const confusion = /^confusion:(\d+)->(\d+)(?::(.*))?$/.exec(m.name);
      const name = row.class != null ? label(row.class) : confusion ?
        "Confusion " + label(Number(confusion[1])) + " → " + label(Number(confusion[2])) + (confusion[3] ? " · " + confusion[3] : "") :
        m.name.replaceAll("_", " ");
      return <tr key={m.name + ":" + (row.class ?? "")} data-primary={m.primary}>
        <th scope="row" title={String(name)}>{name}</th>
        {!absolute && <td>{metricNumber(m.baseline, m.unit)}</td>}<td>{metricNumber(m.value, m.unit)}</td>
        {!absolute && <><td>{metricNumber(m.delta, m.unit, true)}</td>
          <td>{m.ci ? "[" + m.ci.map(n => metricNumber(n, m.unit, true)).join(", ") + "]" : row.n == null ? "No interval" : "No interval (n = " + count(row.n) + ")"}</td>
          <td><IntervalPlot metric={m} extent={domain[m.unit]}/></td><td className="ev-reading" data-reading={reading(m)}>{reading(m)}</td></>}
      </tr>;
    })}</tbody>
  </table></div>;
}
function PrimaryResult({metric, absolute}: {metric: Metric; absolute: boolean}) {
  const value = metricNumber(metric.value, metric.unit);
  const word = reading(metric);
  return <div className="ev-primary">
    <strong>{metric.name.replaceAll("_", " ")}</strong>
    {absolute ? <span>{value}</span> : <>
      <span>{metricNumber(metric.baseline, metric.unit)} → {value}</span>
      <span>{metricNumber(metric.delta, metric.unit, true)}</span>
      <span>{metric.ci ? "95% CI [" + metric.ci.map(n => metricNumber(n, metric.unit, true)).join(", ") + "]" : "No interval"}</span>
      <IntervalPlot metric={metric} extent={intervalDomain([metric])[metric.unit]}/>
      {word && <span className="ev-reading" data-reading={word}>{word}</span>}
    </>}
  </div>;
}
function CopyId({id, label}: {id: string; label: string}) {
  const [copied, setCopied] = useState("");
  return <span className="ev-copy"><code title={id}>{shortId(id)}</code>
    <button aria-label={"Copy " + label} type="button" className="tv-copy" onClick={() => {
      void navigator.clipboard.writeText(id).then(() => setCopied(label + " copied.")).catch(() => setCopied("Clipboard unavailable."));
    }}>⧉</button><span className="visually-hidden" role="status">{copied}</span></span>;
}
const checkpointName = (names: Map<string, string>, id?: string | null) => id ? names.get(id) ?? shortId(id) : "—";
const side = (dataset: Dataset | undefined, job: Evaluation) => (dataset?.name ?? shortId(job.dataset_id)) + " / " + job.split;
function EvaluationHeader({event, job, dataset, names, now, events, jump}: {event: TraceEvent; job?: Evaluation; dataset?: Dataset;
  names: Map<string, string>; now: number; events: TraceEvent[]; jump: (seq: number) => void}) {
  const id = event.evaluation_id!;
  const apply = events.find(row => row.seq === event.links?.apply_seq);
  const parent = job ? job.baseline_id : event.baseline_id;
  const task = job?.evaluator?.task ?? dataset?.task;
  const version = job?.evaluator?.version;
  const hash = job?.fingerprint ?? event.fingerprint;
  const recipe = apply?.recipe_hash ?? job?.recipe_hash;
  return <>
    <header><strong>#{event.seq} · evaluate</strong><time dateTime={event.at}>{timeText(event.at)}</time><JobChip id={id} now={now}/></header>
    <p className="ev-measured">
      <strong>{checkpointName(names, job?.checkpoint_id ?? event.checkpoint_id)}</strong>
      {parent ? " vs " + checkpointName(names, parent) : parent === null ? " · absolute" : ""}
      {job ? " · " + side(dataset, job) : event.dataset_id ? " · " + (dataset?.name ?? shortId(event.dataset_id)) : ""}
      {typeof task === "string" && " · " + task}{(typeof version === "string" || typeof version === "number") && " v" + version}
      {job && " · " + count(job.samples) + " samples"}
    </p>
    <p className="tv-meta ev-measured">{event.links?.apply_seq != null && <button type="button" className="tv-text-button" onClick={() => jump(event.links!.apply_seq!)}>From apply #{event.links.apply_seq}</button>}
      {recipe && <> Recipe <CopyId id={recipe} label="recipe"/></>}
      {hash && <> Fingerprint <CopyId id={hash} label="fingerprint"/></>}
      {(job?.baseline_cached || event.baseline_cached) && <span className="tv-flag">cached baseline</span>}</p>
    <p className="tv-meta">Intervention <InterventionText {...{job,event}}/></p>
    {(job?.intervention || event.intervention_id) && <p className="ev-measure-only">Measurement only. No checkpoint was created.</p>}
  </>;
}
function Counts({job, open}: {job: Evaluation; open: EvalOpen}) {
  return job.baseline_id !== null && job.counts ? <div className="ev-actions">
    {(["fixed", "broken", "changed"] as Category[]).map(kind => <button key={kind} type="button" className="tv-text-button" onClick={() => open(job.evaluation_id, "samples", kind)}>
      {categoryLabels[kind]} {count(job.counts![kind])}</button>)}
  </div> : null;
}
function compareAction(job?: Evaluation, actions?: TraceActions) {
  return job?.baseline_id && job.baseline_id !== job.checkpoint_id && actions ? <button type="button" onClick={() => actions.tensor(job.checkpoint_id, undefined, job.baseline_id!)}>
    Compare with baseline ↗</button> : null;
}
function ResultNotice({job, unavailable}: {job?: Evaluation; unavailable?: boolean}) {
  if (unavailable) return <p className="tv-meta">Results are not available from this server.</p>;
  if (job?.status === "failed") return <p className="tv-failure">The evaluation job failed{job.error_code ? " · " + job.error_code : ""}.{job.message && " " + job.message}</p>;
  if (job?.status === "cancelled") return <p className="tv-meta">The evaluation was cancelled.</p>;
  if (job?.status !== "succeeded") return <p className="tv-meta">Results appear when the job finishes.</p>;
  if (job.baseline_id === null) return <p className="tv-meta">No baseline: absolute metrics only.</p>;
  return null;
}
export function EvaluationsList({events, names, datasets, now, actions, jump, open}: {
  events: TraceEvent[]; names: Map<string, string>; datasets: Map<string, Dataset>; now: number;
  actions: TraceActions; jump: (seq: number) => void; open: EvalOpen;
}) {
  const {read} = useMeasurementStore();
  const seen = new Set<string>();
  const evaluations = events.filter(e => {
    if (e.tool !== "evaluate" || !e.evaluation_id || seen.has(e.evaluation_id)) return false;
    seen.add(e.evaluation_id); return true;
  });
  return <div className="tv-cards">{!evaluations.length && <p className="tv-empty">This session has not run an evaluation.</p>}
    {evaluations.map(event => {
      const state = read(event.evaluation_id!);
      const job = state?.record as Evaluation | undefined;
      const dataset = datasets.get(job?.dataset_id ?? event.dataset_id ?? "");
      const ready = job?.status === "succeeded";
      const metrics = [...(job?.metrics ?? [])].sort((a,b) => Number(b.primary) - Number(a.primary)).slice(0, 3);
      return <article key={event.evaluation_id} className="tv-card ev-card">
        <EvaluationHeader {...{event,job,dataset,names,now,events,jump}}/>
        <ResultNotice job={job} unavailable={state?.failed}/>
        {!!metrics.length && <MetricTable metrics={metrics} dataset={dataset} absolute={job?.baseline_id === null}/>}
        {ready && <Counts job={job} open={open}/>}
        <div className="ev-actions">
          <button type="button" disabled={!ready} onClick={() => open(event.evaluation_id!)}>Metrics</button>
          <button type="button" disabled={!ready || job.baseline_id === null} onClick={() => open(event.evaluation_id!, "samples")}>
            Samples{job?.counts ? ": " + count(job.counts.broken) + " broken" : ""}</button>
          {compareAction(job, actions)}<button type="button" onClick={() => jump(event.seq)}>Show in timeline</button>
        </div>
      </article>;
    })}</div>;
}

function Facts({values}: {values: Record<string, unknown>}) {
  return <dl className="tv-fields ev-facts">{Object.entries(values).map(([key, value]) => <div key={key}><dt>{key.replaceAll("_", " ")}</dt>
    <dd><code>{typeof value === "string" ? value : JSON.stringify(value)}</code></dd></div>)}</dl>;
}
export function EvaluationDetail({detail, event, events, now, names, datasets, actions, jump, open, back, selected, setSelected, overlays, setOverlays}: {
  detail: EvalLocation; event: TraceEvent; events: TraceEvent[]; now: number; names: Map<string, string>; datasets: Map<string, Dataset>;
  actions: TraceActions; jump: (seq: number) => void; open: EvalOpen; back: () => void; selected?: SelectedSample;
  setSelected: (record?: SelectedSample) => void; overlays: {mask: boolean; before: boolean}; setOverlays: (overlays: {mask: boolean; before: boolean}) => void;
}) {
  const state = useMeasurementStore().read(detail.id);
  const job = state?.record as Evaluation | undefined;
  const id = job?.dataset_id ?? event.dataset_id;
  const {dataset, failed} = useDataset(id, datasets.get(id ?? ""));
  const ready = job?.status === "succeeded";
  const absolute = job?.baseline_id === null;
  const metrics = job?.metrics ?? [];
  const extent = intervalDomain(metrics.flatMap(metric => [metric, ...(metric.per_class ?? [])]));
  const splitHash = job?.split_sha256 ?? event.split_sha256 ?? dataset?.split_sha256?.[job?.split ?? ""];
  return <div className="ev-detail">
    <div className="ev-actions ev-detail-nav">
      <button type="button" onClick={back}>← Evaluations</button>
      <button type="button" aria-pressed={detail.view === "metrics"} onClick={() => open(detail.id)}>Metrics</button>
      <button type="button" aria-pressed={detail.view === "samples"} disabled={!ready || absolute} onClick={() => open(detail.id, "samples", detail.category)}>Samples</button>
    </div>
    <article className="tv-card ev-card">
      <EvaluationHeader {...{event,job,dataset,names,now,events,jump}}/>
      <ResultNotice job={job} unavailable={state?.failed}/>
      {detail.view === "metrics" && ready && <>
        <div className="ev-actions">{compareAction(job, actions)}<button type="button" onClick={() => jump(event.seq)}>Show in timeline</button></div>
        {job.evaluator && <details open><summary>Evaluator</summary><Facts values={job.evaluator}/></details>}
        {dataset && <details open><summary>Dataset preprocessing</summary><Facts values={dataset.preprocessing}/></details>}
        {splitHash && <p className="tv-meta">Dataset split hash <CopyId id={splitHash} label="split hash"/></p>}
        {!!job.metrics?.length && <MetricTable metrics={job.metrics} dataset={dataset} absolute={absolute} extent={extent}/>}
        {job.metrics?.map(metric => !!metric.per_class?.length && <details key={metric.name}>
          <summary>{metric.name.replaceAll("_", " ")} per class ({metric.per_class.length})</summary>
          <MetricTable metrics={[...metric.per_class].sort((a,b) => Math.abs(b.delta ?? 0) - Math.abs(a.delta ?? 0)).map(row => ({...row, primary: false}))} dataset={dataset} absolute={absolute} extent={extent}/>
        </details>)}
        <p className="tv-meta">Deltas are paired on the same samples. Intervals are 95% paired bootstrap, 2,000 resamples, fixed seed.</p>
      </>}
    </article>
    {detail.view === "samples" && ready && !absolute && (dataset ?
      <EvaluationSamples {...{job,dataset,overlays,setOverlays}} category={detail.category} setCategory={category => open(detail.id, "samples", category)}
        selected={selected?.row} select={row => setSelected(row ? {dataset, row} : undefined)}/> :
      <p className="tv-empty" role="status">{failed ? "Dataset details unavailable." : "Loading dataset…"}</p>)}
  </div>;
}

function ProbeResults({report, actions}: {report: Probe; actions: TraceActions}) {
  return <>
    <p className="tv-meta">{count(report.samples)} samples{report.split_by && " · split by " + report.split_by}{report.min_aggregate != null && " · K = " + count(report.min_aggregate)}</p>
    {report.raw && <p className="tv-failure">Raw activation values were released to this session (egress class 2).</p>}
    {report.nodes?.map(node => <ProbeAggregates key={node.node_id} {...{report,node,actions}}/>)}
    {report.skipped_small && Object.keys(report.skipped_small as object).length > 0 && <div>
      <h3>Skipped groups below K</h3><pre>{JSON.stringify(report.skipped_small, null, 2)}</pre></div>}
  </>;
}
function ProbeAggregates({report, node, actions}: {report: Probe; node: NonNullable<Probe["nodes"]>[number]; actions: TraceActions}) {
  const metadata = useNode(report.checkpoint_id, node.node_id);
  const weight = metadata?.parameters?.find(name => name.endsWith(".weight")) ?? metadata?.parameters?.[0];
  const group = node.group ?? report.group;
  const [shown, setShown] = useState(32);
  return <div className="ev-probe-node">
    <h3 title={node.node_id}>{metadata?.module_path ?? node.node_id}{group && " · " + group}</h3>
    <div className="ev-actions">
      <button type="button" onClick={() => actions.architecture(report.checkpoint_id, group ?? node.node_id, undefined, group ? "group" : "node")}>Open in Architecture ↗</button>
      {weight && <button type="button" onClick={() => actions.tensor(report.checkpoint_id, {tensor: weight})}>Open in Tensors ↗</button>}
    </div>
    {Object.entries(node.groups).map(([name, group]) => <details key={name} open={Object.keys(node.groups).length === 1}>
      <summary>{name} · {count(group.n)} values per unit</summary>
      <table className="ev-probe-table"><thead><tr><th>Unit</th><th>Mean</th><th>Std</th><th>L2</th><th>Positive</th>{group.histogram && <th>Histogram</th>}</tr></thead>
        <tbody>{group.mean.slice(0, shown).map((n, i) => <tr key={i}>
          <th scope="row">{i}</th><td>{n?.toPrecision(3)}</td><td>{group.std[i]?.toPrecision(3)}</td><td>{group.l2[i]?.toPrecision(3)}</td><td>{metricNumber(group.frac_positive[i], "rate")}</td>
          {group.histogram && <td><Histogram counts={group.histogram.counts[i] ?? []}/></td>}</tr>)}</tbody></table>
    </details>)}
    {node.units > shown && <button type="button" onClick={() => setShown(n => n + 32)}>Show more units</button>}
  </div>;
}
function Histogram({counts}: {counts: number[]}) {
  const max = Math.max(1, ...counts);
  return <svg role="img" aria-label={counts.join(", ") + " counts"} viewBox={`0 0 ${Math.max(1, counts.length)} 12`} width="54" height="16" preserveAspectRatio="none">
    {counts.map((n,i) => <rect key={i} x={i} y={12 - n / max * 12} width=".8" height={n / max * 12} fill="currentColor"/>)}
  </svg>;
}

export function MeasurementEvidence({event, events, datasets, now, actions, open, jump}: {event: TraceEvent; events: TraceEvent[];
  datasets: Map<string, Dataset>; now: number; actions: TraceActions; open: EvalOpen; jump: (seq: number) => void}) {
  const {read} = useMeasurementStore();
  const id = event.evaluation_id ?? event.probe_id;
  if (!id) return null;
  const origin = originOf(events, event);
  const record = read(id)?.record;
  const isRead = ["get_evaluation", "evaluation_changes", "get_probe"].includes(event.tool ?? "");
  const raw = (origin ?? event).capabilities.includes("weights:values") && event.session !== "human";
  const sections: ReactNode[] = [];
  if (event.dataset_id) {
    const name = datasets.get(event.dataset_id)?.name;
    if (name) sections.push(<p key="dataset">Dataset {name}</p>);
  }
  if (isRead && origin) sections.push(<p key="origin">The agent read the results of <button type="button" className="tv-text-button" onClick={() => jump(origin.seq)}>#{origin.seq}</button>.</p>);
  if (event.offset != null) {
    const limit = event.page_limit;
    sections.push(<p key="page">{limit ? "Page " + (Math.floor(event.offset / limit) + 1) + " · requested rows " + (event.offset + 1) + "–" + (event.offset + limit) :
      "Rows starting at " + (event.offset + 1)}</p>);
  }
  if (record && "evaluation_id" in record) {
    const evalEvent = origin ?? event;
    const job = record;
    const dataset = datasets.get(job.dataset_id);
    const primary = job.metrics?.filter(row => row.primary);
    const final = job.split === "holdout";
    sections.push(<p key="context">{final ? "Human final evaluation" : side(dataset, job)} · {count(job.samples)} samples</p>);
    if (event.intervention_id || job.intervention) sections.push(<p key="intervention">Intervention <InterventionText job={job} event={evalEvent}/></p>);
    if (!final && primary?.length) sections.push(<PrimaryResult key="delta" metric={primary[0]} absolute={job.baseline_id === null}/>);
    if (job.status === "failed") sections.push(<ResultNotice key="error" job={job}/>);
    if (!final && evalEvent.tool === "evaluate") sections.push(<div key="actions" className="ev-actions">
      <button type="button" onClick={() => open(id)}>Open evaluation</button>
      {job.status === "succeeded" && job.baseline_id !== null && <button type="button" onClick={() => open(id, "samples")}>Samples</button>}
      {compareAction(job, actions)}</div>);
  }
  if (record && "probe_id" in record) sections.push(<ProbeResults key="probe" report={record} actions={actions}/>);
  return <section className="tv-section ev-evidence">
    <h3>{event.probe_id ? "Probe" : "Evaluation"}</h3><JobChip id={id} raw={raw} now={now}/>
    {read(id)?.failed && <p>Results are not available from this server.</p>}
    {sections}
  </section>;
}

export function ApplyMeasurements({events, seq, open}: {events: TraceEvent[]; seq: number; open: EvalOpen}) {
  const {read} = useMeasurementStore();
  return <>{events.filter(e => e.tool === "evaluate" && e.evaluation_id && e.links?.apply_seq === seq && e.replayed !== true).map(event => {
    const id = event.evaluation_id!;
    const job = read(id)?.record as Evaluation | undefined;
    const primary = job?.metrics?.find(m => m.primary);
    return <div className="ev-actions" key={id}><button type="button" className="tv-text-button" onClick={() => open(id)}>
      Measured #{event.seq}{primary && <> · {primary.name.replaceAll("_", " ")} {metricNumber(primary.delta ?? primary.value, primary.unit, primary.delta !== null)}
        {primary.ci && " [" + primary.ci.map(n => metricNumber(n, primary.unit, true)).join(", ") + "]"}
        {reading(primary) && " · " + reading(primary)}</>}</button><JobChip id={id} now={Date.now()}/></div>;
  })}</>;
}
