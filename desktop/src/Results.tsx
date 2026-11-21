import {useEffect, useRef, useState} from "react";
import {Change, Comparison, get, Page, Slice} from "./api";
import "./results.css";

const heatStops = [
  [8, 18, 19], [27, 75, 78], [45, 132, 123], [128, 202, 158], [247, 224, 142],
];

function activationColor(value: number) {
  const position = value * (heatStops.length - 1);
  const index = Math.min(Math.floor(position), heatStops.length - 2);
  const fraction = position - index;
  return heatStops[index].map((low, axis) => Math.round(low + fraction * (heatStops[index + 1][axis] - low)));
}

export function Pixels({matrix, label, normalize = false}: {matrix: number[][]; label: string; normalize?: boolean}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const maximum = normalize ? Math.max(0, ...matrix.flat()) : 1;
  useEffect(() => {
    const canvas = ref.current!;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const pixels = ctx.createImageData(canvas.width, canvas.height);
    matrix.flat().forEach((value, i) => {
      const level = maximum ? Math.max(0, Math.min(1, value / maximum)) : 0;
      const gray = Math.round(level * 255);
      const rgb = normalize ? activationColor(level) : [gray, gray, gray];
      pixels.data.set([...rgb, 255], i * 4);
    });
    ctx.putImageData(pixels, 0, 0);
  }, [matrix, normalize, maximum]);
  const range = normalize ? "Post-ReLU activation, local range 0 to " + maximum.toPrecision(4)
    : "Input intensity, fixed range 0 to 1";
  return <canvas ref={ref} role="img" className={normalize ? "tensor-map" : "tensor-image"}
    title={range} aria-label={label + ". " + range} data-maximum={maximum}
    width={matrix[0].length} height={matrix.length} />;
}

export const percent = (n: number|null) => n === null ? "—" : (100*n).toFixed(2) + "%";

function sliceTitle(id: string) {
  const label = id.match(/^(normal|blurred)_truth_(\d)$/);
  if (label) return (label[1] === "blurred" ? "Blurred" : "Normal") + " · true digit " + label[2];
  const confusion = id.match(/^baseline_(\d)_as_(\d)_blurred$/);
  if (confusion) return "Blurred · baseline " + confusion[1] + " → " + confusion[2];
  if (id === "normal") return "All normal images";
  if (id === "blurred") return "All blurred images";
  if (id === "overall") return "All images";
  return id;
}

function polarity(value: number|null) {
  if (value === null || value === 0) return "neutral";
  return value > 0 ? "positive" : "negative";
}

function SliceSummary({slice, name}: {slice: Slice; name:string}) {
  const delta = slice.delta_pp;
  return <div className="cmp-summary">
    <div className="cmp-summary-heading"><h3>{name}</h3><span>N {slice.n.toLocaleString("en-US")}</span></div>
    <div className="cmp-summary-detail">{sliceTitle(slice.slice_id)}</div>
    <div className="cmp-outcome"><strong>{percent(slice.accuracy_after)}</strong>
      <span className="cmp-delta" data-tone={polarity(delta)}>{delta === null ? "—" : (delta > 0 ? "+" : "") + delta.toFixed(2) + " pp"}</span></div>
    <div className="cmp-baseline">Baseline {percent(slice.accuracy_before)}</div>
  </div>;
}

function ConfusionMatrix({name, matrix, max}: {name:string; matrix:number[][]; max:number}) {
  return <div className="cmp-matrix">
    <h4>{name}</h4><table aria-label={name + " confusion counts by true and predicted digit"}>
      <thead><tr><th scope="col" aria-label="True digit">y</th>{matrix.map((_,i) => <th key={i} scope="col">{i}</th>)}</tr></thead>
      <tbody>{matrix.map((row,i) => <tr key={i}><th scope="row">{i}</th>{row.map((n,j) => {
        const intensity = max ? Math.log1p(n)/Math.log1p(max) : 0;
        const rgb = i===j ? "108,198,159" : "222,151,100";
        return <td key={j} style={{backgroundColor:n ? "rgba("+rgb+","+(0.09 + .31*intensity)+")" : undefined}}
          data-empty={!n} title={"Truth "+i+" · predicted "+j+" · "+n+" samples"}>{n}</td>;
      })}</tr>)}</tbody>
    </table>
  </div>;
}

export function Results({comparison, runId, inspect}: {
  comparison: Comparison; runId: string; inspect: (row: Change) => void;
}) {
  const [selected, setSelected] = useState(comparison.primary_slice_id);
  const [changes, setChanges] = useState<Change[]>([]);
  const [cursor, setCursor] = useState<string|null>(null);
  const [pending, setPending] = useState(true);
  const [error, setError] = useState("");
  const slice = comparison.slices.find(x => x.slice_id === selected) ?? comparison.slices[0];
  const primary = comparison.slices.find(x => x.slice_id === comparison.primary_slice_id);
  const guardrail = comparison.slices.find(x => x.slice_id === comparison.guardrail_slice_id);
  const overall = comparison.slices.find(x => x.slice_id === "overall");
  const totalChanges = overall ? overall.fixed + overall.regressed + overall.changed_wrong : null;
  const max = Math.max(...slice.confusion_before.flat(), ...slice.confusion_after.flat(), 0);
  useEffect(() => {
    let alive = true;
    setPending(true); setError("");
    get<Page<Change>>("/runs/" + runId + "/changes?limit=25").then(page => {
      if (alive) { setChanges(page.items); setCursor(page.next_cursor); }
    }).catch(e => {if (alive) setError(e.message);}).finally(() => {if (alive) setPending(false);});
    return () => {alive = false;};
  }, [runId]);
  async function more() {
    setPending(true); setError("");
    try {
      const page = await get<Page<Change>>("/runs/" + runId + "/changes?limit=25&cursor=" + encodeURIComponent(cursor!));
      setChanges(previous => [...previous, ...page.items]); setCursor(page.next_cursor);
    } catch (e) { setError((e as Error).message); }
    finally {setPending(false);}
  }
  return <div className="comparison-view">
    <div className="cmp-title"><h2>Paired predictions</h2>
      <span>{comparison.runtime?.actual_device?.toUpperCase()}</span></div>
    <div className="cmp-summaries">{primary && <SliceSummary name="Primary slice" slice={primary}/>}
      {guardrail && <SliceSummary name="Normal guardrail" slice={guardrail}/>}</div>
    <details className="cmp-record"><summary>Run provenance</summary>
      <dl className="facts"><dt>Run</dt><dd>{runId}</dd><dt>Baseline</dt><dd>{comparison.baseline_run_id || "—"}</dd>
        <dt>Actual device</dt><dd>{comparison.runtime?.actual_device || "See run report"}</dd>
        {comparison.recipe_hash && <><dt>Recipe</dt><dd>{comparison.recipe_hash}</dd></>}</dl>
    </details>
    <div className="cmp-slice-label"><h3>Slice metrics</h3><label>Evaluation slice<select value={selected} onChange={e => setSelected(e.target.value)}>
      {comparison.slices.map(s => <option key={s.slice_id} value={s.slice_id}>{sliceTitle(s.slice_id)}</option>)}</select></label></div>
    <div className="cmp-scroll" tabIndex={0} role="region" aria-label="Slice metrics table">
      <table className="cmp-metrics"><caption>Accuracy on {sliceTitle(slice.slice_id)} · N = {slice.n}</caption>
        <thead><tr>{["Baseline","Intervention","Δ pp","Fixed","Regressed","Changed wrong"].map(x => <th key={x} scope="col">{x}</th>)}</tr></thead>
        <tbody><tr><td data-col="Baseline">{percent(slice.accuracy_before)}</td><td data-col="Intervention">{percent(slice.accuracy_after)}</td>
          <td data-col="Δ pp" data-tone={polarity(slice.delta_pp)}>{slice.delta_pp?.toFixed(2) ?? "—"}</td><td data-col="Fixed">{slice.fixed}</td>
          <td data-col="Regressed">{slice.regressed}</td><td data-col="Changed wrong">{slice.changed_wrong}</td></tr></tbody></table>
    </div>
    <details className="cmp-confusion"><summary>Confusion matrices</summary>
      <p>Rows show true digits. Columns show predictions. Both grids share a log count scale from 0 to {max}.</p>
      <div className="cmp-matrices"><ConfusionMatrix name="Baseline" matrix={slice.confusion_before} max={max}/>
        <ConfusionMatrix name="Intervention" matrix={slice.confusion_after} max={max}/></div></details>
    <div className="cmp-changes-title"><h3>Changed predictions</h3><span>{totalChanges === null ? "All slices" : totalChanges.toLocaleString("en-US")+" across all slices"}</span></div>
    {error && <p role="alert">{error}</p>}
    {changes.length ? <table className="cmp-changes"><thead><tr>{["Sample","Truth","Before","After","Change"].map(s => <th key={s} scope="col">{s}</th>)}</tr></thead>
      <tbody>{changes.map(row => <tr key={row.sample_id}><td><button className="link" onClick={() => inspect(row)}>{row.sample_id}</button></td>
        <td data-col="Truth">{row.truth}</td><td data-col="Before">{row.before}</td><td data-col="After">{row.after}</td>
        <td data-col="Change"><span className="cmp-category" data-category={row.category}>{row.category.replaceAll("_"," ")}</span></td></tr>)}</tbody></table>
      : !error && <p role="status" className="cmp-empty">{pending ? "Loading changes…" : "No prediction changes."}</p>}
    {cursor && <button type="button" className="cmp-more" disabled={pending} onClick={more}>{pending ? "Loading changes…" : "Load more changes"}</button>}
  </div>;
}
