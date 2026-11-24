import {useEffect, useLayoutEffect, useRef, useState} from "react";
import {useTheme} from "./theme";
import {DiffUnit, inspectErrorText, Tensor, TensorDataSource, Unit} from "./tensorSource";
import {count, useWindow} from "./tensorTree";
import {metric, useWidth} from "./tensorValues";

type Row = {position: number; stats?: Unit; diff?: DiffUnit; histogram?: number[];
  parentL2?: number | null; childL2?: number | null};
type Column = {label: string; get: (row: Row) => number | null | undefined; format?: (n: number) => string; missing?: (row: Row) => string};
const inspectColumns: Column[] = [
  {label: "Position", get: r => r.position}, {label: "L2", get: r => r.stats?.l2},
  {label: "Mean", get: r => r.stats?.mean}, {label: "Std", get: r => r.stats?.std},
  {label: "Min", get: r => r.stats?.min}, {label: "Max", get: r => r.stats?.max},
  {label: "Zero %", get: r => r.stats?.zero_frac, format: n => metric(n * 100) + "%"},
  {label: "NaN/Inf", get: r => r.stats?.nonfinite_count},
  {label: "Constant", get: r => Number(r.stats?.constant), format: n => n ? "Yes" : "No"},
];
function missingRelL2(row: Row) {
  if (row.diff?.max_abs === null) return "No values are finite on both sides; relative L2 is undefined.";
  if (row.parentL2 === 0) return "Parent is all zero; relative L2 is undefined.";
  return "Relative L2 is undefined for the paired finite values.";
}

function missingCosine(row: Row) {
  if (row.diff?.max_abs === null) return "No values are finite on both sides; cosine is undefined.";
  if (row.parentL2 === 0 && row.childL2 === 0) return "Parent and child are all zero; cosine is undefined.";
  if (row.parentL2 === 0) return "Parent is all zero; cosine is undefined.";
  if (row.childL2 === 0) return "Child is all zero; cosine is undefined.";
  return "Cosine is undefined for the paired finite values.";
}

const compareColumns: Column[] = [
  inspectColumns[0], {label: "Changed values", get: r => r.diff?.changed_elems, format: count},
  {label: "Max |Δ|", get: r => r.diff?.max_abs},
  {label: "Rel L2", get: r => r.diff?.rel_l2, missing: missingRelL2},
  {label: "Cosine", get: r => r.diff?.cosine, missing: missingCosine},
];
const profiles: Column[] = [
  inspectColumns[1], inspectColumns[2], inspectColumns[3], inspectColumns[6],
  {label: "Max |x|", get: r => r.stats?.min == null || r.stats.max == null ? null : Math.max(Math.abs(r.stats.min), Math.abs(r.stats.max))},
];

function Profile({rows, total, get, onPosition, selected, axis}: {rows: Row[]; total: number; get: Column["get"];
  onPosition: (n: number) => void; selected: number; axis: number}) {
  const {ref: box, width} = useWidth<HTMLDivElement>();
  const canvas = useRef<HTMLCanvasElement>(null);
  const loaded = new Map(rows.map(r => [r.position, r]));
  useTheme();
  const [hover, setHover] = useState<number | null>(null);
  const info = hover === null ? null : loaded.get(hover);
  useLayoutEffect(() => {
    const ctx = canvas.current?.getContext("2d");
    if (!ctx || !width || !total) return;
    const values = rows.map(get).filter((n): n is number => n != null);
    const low = Math.min(0, ...values);
    const high = Math.max(0, ...values);
    const y = (v: number) => Math.round(58 - 53 * (v - low) / (high - low || 1));
    const envelope = new Map<number, {min: number; max: number; loaded: number}>();
    for (const row of rows) {
      const x = Math.floor(row.position * width / total);
      const value = get(row);
      const bucket = envelope.get(x) ?? {min: Infinity, max: -Infinity, loaded: 0};
      bucket.loaded++;
      if (value != null) {bucket.min = Math.min(bucket.min, value); bucket.max = Math.max(bucket.max, value);}
      envelope.set(x, bucket);
    }
    const style = getComputedStyle(document.documentElement);
    const accent = style.getPropertyValue("--accent").trim();
    const line = style.getPropertyValue("--line-strong").trim();
    ctx.fillStyle = style.getPropertyValue("--bg").trim();
    ctx.fillRect(0, 0, width, 64);
    ctx.strokeStyle = line;
    ctx.beginPath(); ctx.moveTo(0, y(0) + .5); ctx.lineTo(width, y(0) + .5); ctx.stroke();
    const bars = Math.max(1, Math.floor(width / total));
    for (let x = 0; x < width; x += bars) {
      const bucket = envelope.get(x);
      const start = Math.ceil(x * total / width);
      const end = Math.ceil((x + 1) * total / width);
      if (total > width && (bucket?.loaded ?? 0) < end - start ||
          total <= width && !loaded.has(Math.min(total - 1, Math.floor(x * total / width)))) {
        ctx.fillStyle = line;
        for (let py = (x % 7); py < 64; py += 7) ctx.fillRect(x, py, 1, 1);
      }
    }
    for (const [x, bucket] of envelope) {
      if (!Number.isFinite(bucket.max)) continue;
      const barWidth = Math.max(1, bars - 1);
      const zero = y(0), min = y(bucket.min), max = y(bucket.max);
      const near = y(bucket.min > 0 ? bucket.min : bucket.max < 0 ? bucket.max : 0);
      if (max < min) {
        ctx.fillStyle = "color-mix(in srgb, " + accent + " 45%, transparent)";
        ctx.fillRect(x, max, barWidth, min - max);
      }
      if (near !== zero) {
        ctx.fillStyle = accent;
        ctx.fillRect(x, Math.min(zero, near), barWidth, Math.abs(zero - near));
      }
      ctx.fillStyle = accent;
      ctx.fillRect(x, near, barWidth, 1);
    }
    ctx.fillStyle = style.getPropertyValue("--ink").trim();
    if (selected >= 0) ctx.fillRect(Math.floor(selected * width / total), 2, 1, 60);
  });

  return <div ref={box} className="tb-profile">
    <canvas ref={canvas} width={width} height={64} role="img" tabIndex={0}
      aria-label={"Profile of " + total + " positions along axis " + axis + ", " + rows.length + " loaded"}
      onKeyDown={e => {
        if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
          e.preventDefault();
          const position = Math.min(total - 1, Math.max(0, selected + (e.key === "ArrowLeft" ? -1 : 1)));
          setHover(position); onPosition(position);
        }
      }} onPointerMove={e => setHover(Math.min(total - 1, Math.max(0, Math.floor((e.clientX - e.currentTarget.getBoundingClientRect().left) / width * total))))}
      onPointerLeave={() => setHover(null)} onClick={() => {if (hover !== null) onPosition(hover);}}/>
    <span className="tb-meta" aria-live="polite">
      {hover === null ? count(rows.length) + " of " + count(total) + " positions loaded"
        : "Position " + hover + " · " + (info ? info.stats?.l2 === 0 ? "all zero" : metric(get(info)) : "not loaded")}
    </span>
  </div>;
}

export function Units({source, checkpoint, other, tensor, axis, setAxis, onPosition, selected, comparable, open, setOpen}:
  {source: TensorDataSource; checkpoint: string; other: string; tensor: Tensor; axis: number; setAxis: (axis: number) => void;
    onPosition: (axis: number, position: number) => void; selected: number; comparable: boolean;
    open: boolean; setOpen: (open: boolean) => void}) {
  const [result, setResult] = useState<{key: string; rows: Row[]; cursor: string | null; precision: string; edges?: number[]}>();
  const [error, setError] = useState("");
  const [metricIndex, setMetricIndex] = useState(0);
  const [sort, setSort] = useState<{column: number; desc: boolean}>();
  const [pending, setPending] = useState(false);
  const rank = tensor.shape.length;
  const profileOnly = rank === 1;
  const total = tensor.shape[axis] ?? 0;
  const reduce = tensor.shape.flatMap((_, i) => i === axis ? [] : [i]);
  const key = JSON.stringify([checkpoint, other, tensor.name, axis]);
  const data = result?.key === key ? result : undefined;
  const canRead = open && total > 0 && (!other || comparable);
  const columns = other ? compareColumns : inspectColumns;
  const sortedBy = sort ?? {column: other ? 1 : 0, desc: !!other};
  const get = columns[sortedBy.column]?.get ?? columns[0].get;
  const rows = data?.rows ?? [];
  const sorted = [...rows].sort((a, b) => {
    const x = get(a), y = get(b);
    return x == null ? y == null ? a.position - b.position : 1 : y == null ? -1 :
      (sortedBy.desc ? y - x : x - y) || a.position - b.position;
  });
  const windowed = useWindow(sorted.length);
  const generation = useRef(0);

  async function load(cursor: string | null, previous: Row[] = []) {
    const request = ++generation.current;
    setPending(true);
    setError("");
    try {
      const selector = {tensor: tensor.name};
      if (other) {
        const page = await source.diff(other, checkpoint, {selector, unit_axis: axis, limit: 1000, cursor});
        if (request !== generation.current) return;
        const differences = page.result.units ?? [];
        const unresolved = profileOnly ? [] : differences.filter(row =>
          row.max_abs !== null && (row.rel_l2 === null || row.cosine === null)).map(row => row.index[axis]);
        const norms = async (id: string) => {
          const result = await source.stats(id, {selector: {...selector, index: {[axis]: unresolved}}, reduce_over_axes: reduce, limit: 1000});
          return new Map(result.result.units.map(row => [row.index[axis], row.l2]));
        };
        const [parent, child] = unresolved.length ? await Promise.allSettled([norms(other), norms(checkpoint)]) : [];
        if (request === generation.current) setResult({key, rows: [...previous, ...differences.map(diff => ({
          position: diff.index[axis], diff,
          parentL2: parent?.status === "fulfilled" ? parent.value.get(diff.index[axis]) : undefined,
          childL2: child?.status === "fulfilled" ? child.value.get(diff.index[axis]) : undefined,
        }))], cursor: page.cursor, precision: page.result.accumulate_dtype});
      } else {
        const page = await source.stats(checkpoint, {selector, reduce_over_axes: reduce,
          bins: profileOnly ? 0 : 16, limit: 2048, cursor});
        if (request === generation.current) setResult({key, rows: [...previous, ...page.result.units.map((stats, i) =>
          ({position: stats.index[axis], stats, histogram: page.result.histogram?.counts[i]}))],
          cursor: page.cursor, precision: page.result.accumulate_dtype, edges: page.result.histogram?.edges});
      }
    } catch (e) {
      if (request === generation.current) setError(inspectErrorText(e));
    } finally {
      if (request === generation.current) setPending(false);
    }
  }

  useEffect(() => {
    if (canRead) void load(null);
    return () => {generation.current++;};
  }, [source, key, canRead]);
  useEffect(() => setSort(undefined), [other]);

  return <section className="tb-units" aria-label="Position statistics">
    <header className="tb-tools">
      <button aria-expanded={open} aria-controls="tb-unit-panel" onClick={() => setOpen(!open)}>{open ? "▾" : "▸"} {rank ? "Positions along axis " + axis : "Position statistics"}</button>
      {open && rank > 1 && <label>Axis<select aria-label="Position axis" value={axis}
        onChange={e => setAxis(+e.target.value)}>{tensor.shape.map((_, i) => <option key={i} value={i}>axis {i}</option>)}</select></label>}
      {open && data && <span className="tb-meta">accumulated in {data.precision}</span>}
      {open && !profileOnly && !other && total > 0 && <label>Profile<select value={metricIndex} onChange={e => setMetricIndex(+e.target.value)}>
        {profiles.map((column, i) => <option key={column.label} value={i}>{column.label}</option>)}
      </select></label>}
      {open && !profileOnly && !other && rows.length > 0 && <button onClick={() => setSort({column: 1, desc: false})}>Lowest L2 first</button>}
    </header>
    {rank <= 1 && <p className="tb-meta">Each position is a single value. See Heatmap or Values.</p>}
    <div id="tb-unit-panel" hidden={!open}>
      {open && pending && canRead && <p role="status">Loading position statistics…</p>}
      {open && error && canRead && <p role="alert">{error}</p>}
      {open && !canRead && <p className="tb-meta">{!rank ? "See the tensor facts for summary statistics." : !total ? "No positions along this axis." : "Position diff is unavailable for these tensors."}</p>}
      {open && canRead && <Profile rows={rows} total={total} get={other ? compareColumns[2].get : profiles[metricIndex].get}
        onPosition={n => onPosition(axis, n)} selected={selected} axis={axis}/>}
      {open && data?.cursor && <button disabled={pending} onClick={() => void load(data.cursor, rows)}>Load more positions</button>}
      {rank > 1 && <div className="tb-unit-table" hidden={!open || !rows.length} ref={windowed.ref} onScroll={windowed.onScroll}
        role="table" aria-label="Position statistics" aria-rowcount={sorted.length + 1} aria-colcount={columns.length + (other ? 0 : 1)}>
        <div role="row" className="tb-unit-row tb-unit-header" data-compare={!!other}>
          {columns.map((column, i) => <div key={column.label} role="columnheader"
            aria-sort={sortedBy.column === i ? sortedBy.desc ? "descending" : "ascending" : "none"}>
            <button onClick={() => setSort({column: i, desc: sortedBy.column === i ? !sortedBy.desc : false})}>
              {column.label}{sortedBy.column === i ? sortedBy.desc ? " ▾" : " ▴" : ""}
            </button>
          </div>)}
          {!other && <div role="columnheader">Histogram</div>}
        </div>
        <div role="rowgroup" style={{height: sorted.length * 28, position: "relative"}}>
          {sorted.slice(windowed.start, windowed.end).map((row, i) =>
            <div key={row.position} role="row" aria-rowindex={windowed.start + i + 2}
              className="tb-unit-row" data-compare={!!other} data-selected={row.position === selected} style={{top: (windowed.start + i) * 28}}>
              {columns.map((column, c) => {
                const value = column.get(row);
                return <div key={column.label} role="cell" title={value == null ? column.missing?.(row) : undefined}>
                  {c === 0 ? <button onClick={() => onPosition(axis, row.position)}>{row.position}</button>
                    : c === 1 && !other && value === 0 ? <span title="L2 = 0">all zero</span>
                    : value == null ? "—" : column.format?.(value) ?? metric(value)}
                </div>;
              })}
              {!other && <div role="cell"><span className="tb-hist" role="img" aria-label={"16-bin histogram: " + row.histogram?.join(", ")}
                title={data?.edges ? "Shared edges: " + data.edges.map(metric).join(", ") : undefined}>
                {row.histogram?.map((value, j) => <i key={j} style={{height: 12 * value / Math.max(1, ...row.histogram!)}}/>)}
              </span></div>}
            </div>)}
        </div>
      </div>}
    </div>
  </section>;
}
