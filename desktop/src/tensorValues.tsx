import {KeyboardEvent, useLayoutEffect, useRef, useState} from "react";
import {Range, Value} from "./tensorSource";
import {useWindow} from "./tensorTree";

export const printed = (value: Value | undefined) =>
  value === undefined ? "—" : Object.is(value, -0) ? "-0.0" : value === "Inf" ? "+Inf" : String(value);
export const metric = (value: number | null | undefined) => value == null ? "—" : Number(value.toPrecision(6)).toString();
export const numeric = (v: Value) => typeof v === "number" || typeof v === "boolean";
export function displayUlp(value: Value, dtype: string) {
  const format = ({BF16:[7, -126], F16:[10, -14], F32:[23, -126], F64:[52, -1022]} as Record<string, number[]>)[dtype];
  return typeof value === "number" && format ? " · ulp " + metric(2 ** (Math.max(format[1], Math.floor(Math.log2(Math.abs(value)))) - format[0])) : "";
}

const ramps = {
  diverging: ["#7fb3ff", "#4f7fc9", "#2c4366", "#161c1a", "#6b4425", "#c47a35", "#ffc27a"],
  sequential: ["#161c1a", "#24443a", "#4f8a74", "#c6f2e0"],
  reversed: ["#c6f2e0", "#4f8a74", "#24443a", "#161c1a"],
};
type Ramp = keyof typeof ramps;
export type Scale = {kind: Ramp | "constant" | "bool" | "empty"; low: number; high: number; centre: number;
  note: string; label: string; color: (value: Value) => string};
export type ScaleMode = "p99" | "max" | "mean";

function rampColor(ramp: Ramp, t: number) {
  const stops = ramps[ramp];
  const position = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const index = Math.min(stops.length - 2, Math.floor(position));
  return "color-mix(in oklab, " + stops[index] + " " + (100 * (1 - (position - index))) + "%, " + stops[index + 1] + ")";
}

export function scaleFor(dtype: string, range: Range | null, mode: ScaleMode, mean: number | null,
  values: Value[], windowOnly = false, delta = false): Scale {
  const note = windowOnly ? "Scale from this window only" : "";
  if (windowOnly) {
    const finite = values.filter((v): v is number => typeof v === "number");
    if (finite.length) {
      const abs = finite.map(Math.abs).sort((a, b) => a - b);
      range = {min: Math.min(...finite), max: Math.max(...finite), abs_max: abs.at(-1)!, abs_p99: abs[Math.ceil(abs.length * .99) - 1]};
    }
  }
  function flat(kind: "constant" | "bool" | "empty", label: string): Scale {
    return {kind, low: 0, high: 0, centre: 0, note, label,
      color: value => numeric(value) ? kind === "bool" ? value ? "#9fc7b8" : "#3a4640" : Number(value) === 0 ? "#161c1a" : "#4f8a74" : "#101512"};
  }
  if (dtype === "BOOL") return flat("bool", "False / True");
  if (!range) return flat("empty", values.length ? "No finite values" : "No values");
  if (!delta && range.min === range.max && values.every(numeric))
    return flat("constant", range.min === 0 ? "All values are 0" : "All values = " + printed(range.min));
  const centred = mode === "mean" && mean !== null && !delta;
  const centre = centred ? mean : 0;
  const magnitude = centred ? Math.max(Math.abs(range.min - centre), Math.abs(range.max - centre))
    : (mode === "max" ? range.abs_max : range.abs_p99) || range.abs_max;
  const diverging = delta || centred || range.min < 0 && range.max > 0;
  const kind = diverging ? "diverging" : range.min >= 0 ? "sequential" : "reversed";
  const low = kind === "sequential" ? 0 : centre - magnitude;
  const high = kind === "reversed" ? 0 : centre + magnitude;
  return {kind, low, high, centre, note,
    label: kind + (centred ? " centred at " + metric(centre) : " " + (diverging ? "±" : "") + metric(magnitude) + " (" + mode + ")"),
    color(value) {
      if (typeof value !== "number") return "#101512";
      return rampColor(kind, high === low ? .5 : (value - low) / (high - low));
    }};
}

export function ScaleLegend({scale, mode, setMode, mean}: {scale: Scale; mode: ScaleMode;
  setMode: (mode: ScaleMode) => void; mean: number | null}) {
  const ramp = scale.kind in ramps ? ramps[scale.kind as Ramp] : null;
  return <div className="tb-legend" aria-label={"Colour scale · " + scale.label}>
    {ramp ? <>
      <span>{metric(scale.low)}</span><i className="tb-ramp" style={{background: "linear-gradient(90deg in oklab, " + ramp.join(",") + ")"}}/>
      <span>{metric(scale.high)}</span>
      <select aria-label="Scale" value={mode} onChange={e => setMode(e.target.value as ScaleMode)}>
        <option value="p99">{scale.kind} {scale.kind === "diverging" ? "±" : ""}p99</option>
        <option value="max">{scale.kind} {scale.kind === "diverging" ? "±" : ""}max</option>
        <option value="mean" disabled={mean === null}>Centre at mean</option>
      </select>
    </> : scale.kind === "bool" ? <><i className="tb-bool-false"/> False <i className="tb-bool-true"/> True</> : <span>{scale.label}</span>}
    {scale.note && <span>{scale.note}</span>}
  </div>;
}

export function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(0);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const update = () => setWidth(el.clientWidth);
    const observer = new ResizeObserver(update);
    observer.observe(el);
    update();
    return () => observer.disconnect();
  }, []);
  return {ref, width};
}

export type Cells = {
  rows: number; cols: number; tileRows: number; tileCols: number;
  values: (Value | undefined)[]; changed?: boolean[]; scale: Scale;
};
export function moveCell(key: string, index: number, rows: number, cols: number) {
  const row = Math.floor(index / cols), col = index % cols;
  const step = ({ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1]} as Record<string, number[]>)[key];
  if (!step) return null;
  return Math.max(0, Math.min(rows - 1, row + step[0])) * cols + Math.max(0, Math.min(cols - 1, col + step[1]));
}

export function Heatmap({cells, cursor, onCursor, onPin, label, zoom, width}: {cells: Cells; cursor: number;
  onCursor: (i: number) => void; onPin: (i: number) => void; label: string; zoom: string; width: number}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const raster = useRef<HTMLCanvasElement>(null);
  const {rows, cols, tileRows, tileCols, values, changed, scale} = cells;
  const cell = zoom === "fit" ? Math.max(2, Math.min(24, Math.floor(Math.min(width / cols, 208 / rows)))) : Number(zoom);
  const gap = tileCols > 1 || tileRows > 1 ? 1 : 0;
  const pixel = (index: number, tile: number) => index * cell + Math.floor(index / tile) * gap;
  const w = pixel(cols - 1, tileCols) + cell, h = pixel(rows - 1, tileRows) + cell;

  useLayoutEffect(() => {
    const image = raster.current ?? document.createElement("canvas");
    image.width = w; image.height = h;
    raster.current = image;
    const ctx = image.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "#101512";
    ctx.fillRect(0, 0, w, h);
    values.forEach((value, index) => {
      if (value === undefined) return;
      const x = pixel(index % cols, tileCols), y = pixel(Math.floor(index / cols), tileRows);
      ctx.fillStyle = changed && !changed[index] ? "#161c1a" : scale.color(value);
      ctx.fillRect(x, y, cell, cell);
      if (typeof value === "string") {
        ctx.fillStyle = "#ff5fa2";
        ctx.fillRect(x + Math.floor(cell / 3), y + Math.floor(cell / 3), Math.max(1, Math.floor(cell / 3)), Math.max(1, Math.floor(cell / 3)));
      }
      if ((!changed || changed[index]) && typeof value === "number" && scale.kind !== "constant" &&
          (value < scale.low || value > scale.high)) {
        ctx.fillStyle = "#e3e2da";
        ctx.fillRect(x + Math.max(0, cell - 2), y, Math.min(2, cell), 1);
      }
      if (changed?.[index]) {
        ctx.strokeStyle = "#ffe2b8";
        if (value === 0 || value === "NaN") {
          ctx.beginPath(); ctx.moveTo(x + .5, y + cell - .5); ctx.lineTo(x + cell - .5, y + .5); ctx.stroke();
        }
        ctx.strokeRect(x + .5, y + .5, Math.max(0, cell - 1), Math.max(0, cell - 1));
      }
    });
  }, [w, h, cols, tileCols, tileRows, cell, gap, values, changed, scale]);

  useLayoutEffect(() => {
    const ctx = ref.current?.getContext("2d");
    if (!ctx || !raster.current) return;
    ctx.drawImage(raster.current, 0, 0);
    if (cursor >= 0 && cursor < values.length) {
      const x = pixel(cursor % cols, tileCols), y = pixel(Math.floor(cursor / cols), tileRows);
      ctx.fillStyle = "#c6f2e04d"; ctx.fillRect(x, 0, cell, h); ctx.fillRect(0, y, w, cell);
      ctx.strokeStyle = "#c6f2e0"; ctx.strokeRect(x + .5, y + .5, Math.max(0, cell - 1), Math.max(0, cell - 1));
    }
  });

  function address(event: {clientX: number; clientY: number; currentTarget: HTMLCanvasElement}) {
    const rect = event.currentTarget.getBoundingClientRect();
    const position = (p: number, tile: number) => {
      const group = Math.floor(p / (tile * cell + gap));
      return group * tile + Math.min(tile - 1, Math.floor((p - group * (tile * cell + gap)) / cell));
    };
    return Math.min(rows - 1, Math.max(0, position(event.clientY - rect.top, tileRows))) * cols +
      Math.min(cols - 1, Math.max(0, position(event.clientX - rect.left, tileCols)));
  }
  return <canvas className="tb-heatmap" ref={ref} width={w} height={h} role="img" aria-label={label}
    tabIndex={0} onPointerMove={e => onCursor(address(e))} onPointerLeave={() => onCursor(-1)}
    onClick={e => onPin(address(e))} onKeyDown={e => {
      const next = moveCell(e.key, Math.max(cursor, 0), rows, cols);
      if (next !== null) {e.preventDefault(); onCursor(next);}
      if (e.key === "Enter" || e.key === " ") {e.preventDefault(); onPin(Math.max(0, cursor));}
    }}/>;
}

export function ValueGrid({cells, cursor, onCursor, onPin, dense, tint, dtype, describe, syncScroll}: {cells: Cells;
  cursor: number; onCursor: (i: number) => void; onPin: (i: number) => void; dense: boolean; tint: boolean;
  dtype: string; describe: (i: number) => string; syncScroll?: (grid: HTMLDivElement) => void}) {
  const {rows, cols, values, changed, scale} = cells;
  const windowed = useWindow(rows, 28);
  const [scroll, setScroll] = useState(0);
  const colWidth = dense ? 84 : 116;
  const start = Math.max(0, Math.floor(scroll / colWidth) - 3);
  const end = Math.min(cols, start + Math.ceil((windowed.ref.current?.clientWidth ?? 500) / colWidth) + 7);
  const gridWidth = cols * colWidth;
  function key(event: KeyboardEvent) {
    const next = moveCell(event.key, Math.max(cursor, 0), rows, cols);
    if (next === null) {
      if (event.key === "Enter") onPin(Math.max(0, cursor));
      return;
    }
    event.preventDefault();
    onCursor(next);
    windowed.scrollTo(Math.floor(next / cols));
    const el = windowed.ref.current;
    const x = (next % cols) * colWidth;
    if (el && (x < el.scrollLeft || x + colWidth > el.scrollLeft + el.clientWidth))
      el.scrollLeft = x < el.scrollLeft ? x : x + colWidth - el.clientWidth;
  }
  return <div className="tb-value-scroll" ref={windowed.ref} onScroll={e => {windowed.onScroll(); setScroll(e.currentTarget.scrollLeft); syncScroll?.(e.currentTarget);}}
    role="grid" aria-label="Tensor values" aria-rowcount={rows} aria-colcount={cols} tabIndex={0}
    onKeyDown={key}>
    <div role="rowgroup" style={{height: rows * 28, width: gridWidth, position: "relative"}}>
      {Array.from({length: windowed.end - windowed.start}, (_, offset) => {
        const row = offset + windowed.start;
        return <div key={row} role="row" aria-rowindex={row + 1} style={{top: row * 28, position: "absolute"}}>
          {Array.from({length: end - start}, (_, i) => {
            const col = i + start, index = row * cols + col, value = values[index];
            return <div key={col} role="gridcell" aria-colindex={col + 1} aria-selected={cursor === index}
              title={describe(index) + " = " + printed(value) + (value === undefined ? "" : displayUlp(value, dtype))}
              className="tb-value" data-nonfinite={typeof value === "string"} data-changed={changed?.[index]}
              data-hatch={changed?.[index] && (value === 0 || value === "NaN")}
              style={{left: col * colWidth, width: colWidth,
                backgroundColor: tint && value !== undefined && numeric(value) && (!changed || changed[index]) ?
                  "color-mix(in oklab, " + scale.color(value) + " 35%, var(--p-bg))" : undefined}}
              onPointerMove={() => onCursor(index)} onClick={() => onPin(index)}
              onCopy={e => {if (value !== undefined) {e.clipboardData.setData("text/plain", printed(value)); e.preventDefault(); e.stopPropagation();}}}>
              {dense && typeof value === "number" && !Object.is(value, -0) ? metric(Number(value.toPrecision(4))) : printed(value)}
            </div>;
          })}
        </div>;
      })}
    </div>
  </div>;
}
