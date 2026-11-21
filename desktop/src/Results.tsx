import {useEffect, useRef, useState} from "react";
import {Change, Comparison, get, Page} from "./api";

export function Pixels({matrix, label, normalize = false}: {matrix: number[][]; label: string; normalize?: boolean}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current!;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const pixels = ctx.createImageData(canvas.width, canvas.height);
    const max = normalize ? Math.max(...matrix.flat(), 1e-12) : 1;
    matrix.flat().forEach((value, i) => {
      const count = Math.round(255 * value / max);
      pixels.data.set([count, count, count, 255], 4 * i);
    });
    ctx.putImageData(pixels, 0, 0);
  }, [matrix, normalize]);
  return <canvas ref={ref} role="img" aria-label={label} width={matrix[0].length} height={matrix.length} />;
}

export const percent = (n: number|null) => n === null ? "—" : (100*n).toFixed(2) + "%";

export function Results({comparison, runId, inspect}: {
  comparison: Comparison; runId: string; inspect: (row: Change) => void;
}) {
  const [selected, setSelected] = useState(comparison.primary_slice_id);
  const [changes, setChanges] = useState<Change[]>([]);
  const [cursor, setCursor] = useState<string|null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const slice = comparison.slices.find(x => x.slice_id === selected) ?? comparison.slices[0];
  useEffect(() => {
    let alive = true;
    setPending(true);
    get<Page<Change>>("/runs/" + runId + "/changes?limit=25").then(page => {
      if (alive) {setChanges(page.items); setCursor(page.next_cursor);}
    }).catch(e => {if (alive) setError(e.message);}).finally(() => {if (alive) setPending(false);});
    return () => {alive = false;};
  }, [runId]);
  async function more() {
    setPending(true);
    try {
      const page = await get<Page<Change>>("/runs/" + runId + "/changes?limit=25&cursor=" + encodeURIComponent(cursor!));
      setChanges([...changes, ...page.items]); setCursor(page.next_cursor);
    } catch (e) { setError((e as Error).message); }
    finally {setPending(false);}
  }
  return <>
    <h2>Paired predictions</h2>
    <dl className="facts"><dt>Run</dt><dd>{runId}</dd><dt>Baseline</dt><dd>{comparison.baseline_run_id || "—"}</dd>
      <dt>Actual device</dt><dd>{comparison.runtime?.actual_device || "See run report"}</dd></dl>
    <label>Evaluation slice<select value={selected} onChange={e => setSelected(e.target.value)}>
      {comparison.slices.map(s => <option key={s.slice_id}>{s.slice_id}</option>)}</select></label>
    <table><caption>Accuracy on {slice.slice_id} · N = {slice.n}</caption><thead><tr>
      {["Baseline","Intervention","Δ pp","Fixed","Regressed","Changed wrong"].map(x => <th key={x}>{x}</th>)}
    </tr></thead><tbody><tr><td>{percent(slice.accuracy_before)}</td><td>{percent(slice.accuracy_after)}</td>
      <td>{slice.delta_pp?.toFixed(2) ?? "—"}</td><td>{slice.fixed}</td><td>{slice.regressed}</td><td>{slice.changed_wrong}</td></tr></tbody></table>
    <details><summary>Confusion matrices</summary><div className="pair">
      {[["Baseline",slice.confusion_before],["Intervention",slice.confusion_after]].map(([name, mat]) =>
        <table key={String(name)}><caption>{String(name)} · truth by prediction</caption><thead><tr><th>Digit</th>
          {Array.from({length:10}, (_,i) => <th key={i}>{i}</th>)}</tr></thead><tbody>
          {(mat as number[][]).map((row,i) => <tr key={i}><th>{i}</th>{row.map((n,j) => <td key={j}>{n}</td>)}</tr>)}
        </tbody></table>)}</div></details>
    <h3>Changed predictions · all slices</h3>
    {error && <p role="alert">{error}</p>}
    {changes.length ? <table><thead><tr><th>Sample</th><th>Truth</th><th>Before</th><th>After</th><th>Change</th></tr></thead>
      <tbody>{changes.map(row => <tr key={row.sample_id}><td><button className="link" onClick={() => inspect(row)}>{row.sample_id}</button></td>
        <td>{row.truth}</td><td>{row.before}</td><td>{row.after}</td><td>{row.category}</td></tr>)}</tbody></table> :
      <p>{pending ? "Loading changes…" : "No prediction changes."}</p>}
    {cursor && <button disabled={pending} onClick={more}>Load more changes</button>}
  </>;
}
