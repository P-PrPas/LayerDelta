import {useEffect, useMemo, useRef, useState} from "react";
import {AxisRange, coordinates, DiffReply, DiffStatus, elements, Envelope, indicesFor, InspectError, Range, Read,
  Resolved, resolveSelector, Selector, Tensor, TensorDataSource, Unit, Value, ValueDiff} from "./tensorSource";
import {count, statuses} from "./tensorTree";
import {Cells, displayUlp, fitCell, Heatmap, heatmapHeight, heatmapSize, metric, minFitCell, printed, ScaleLegend, ScaleMode, scaleFor, useWidth, ValueGrid, valueGridWidths} from "./tensorValues";
import {Units} from "./tensorUnits";

type View = "Heatmap" | "Values" | "Mosaic";
type ReadPage = Envelope<Read, {selector: Resolved}>;
type DeltaPage = DiffReply<ValueDiff>;
type Cursors = {child: string | null; parent?: string | null; delta?: string | null; pair?: string};
type LoadedPage = {child: ReadPage; parent?: ReadPage; delta?: DeltaPage};
type Layout = {rows: number; cols: number; spatial: number[]; view: View; index: Record<string, number[] | AxisRange>; shape?: number[]};
type Display = Pick<Layout, "rows" | "cols" | "spatial"> & {flats: boolean; flatCols: number; selectorText: string};
type DisplayedPage = LoadedPage & {key: string; dataset: string; range: LoadedPage & {key: string}; display: Display};
const pageCap = 10000;
const smallAxes = (tensor: Tensor) => tensor.shape.flatMap((n, i) => n > 0 && n <= 7 ? [i] : []);

function defaultLayout(t: Tensor, view?: View): Layout {
  const small = smallAxes(t);
  const spatial = t.shape.length === 4 && small.length === 2 && (!view || view === "Mosaic") ? small : [];
  const axes = t.shape.map((_, i) => i).filter(i => !spatial.includes(i));
  const cols = axes.at(-1) ?? -1;
  const rows = axes.at(-2) ?? -1;
  const index: Layout["index"] = {};
  for (const axis of axes) if (axis !== rows && axis !== cols && t.shape[axis]) index[axis] = [0];
  return {rows, cols, spatial, index, view: view ?? (spatial.length ? "Mosaic" : "Heatmap")};
}

function syntax(t: Tensor, index: Layout["index"], shape?: number[]) {
  return t.name + " [" + t.shape.map((_, axis) => {
    const part = index[axis];
    return !part ? ":" : Array.isArray(part) ? part.length === 1 ? String(part[0]) : "[" + part.join(", ") + "]"
      : part.start + ":" + part.stop;
  }).join(", ") + "]" + (shape ? " (shape guard [" + shape.join(", ") + "])" : "");
}

function parseSyntax(text: string, t: Tensor): Pick<Layout, "index" | "shape"> {
  const match = /^\s*(.*?)\s*\[(.*)\](?:\s*\(shape guard \[(\d+(?:\s*,\s*\d+)*|)\]\))?\s*$/.exec(text);
  if (!match || match[1] !== t.name)
    throw new InspectError("unknown_tensor", "Enter the open tensor name followed by its slice.", "selector.tensor");
  const parts = match[2].trim() ? match[2].split(/,(?![^\[]*\])/).map(s => s.trim()) : [];
  if (parts.length !== t.shape.length)
    throw new InspectError("invalid_axis", "Enter one slice per axis.", "selector.index");
  const index: Layout["index"] = {};
  parts.forEach((part, axis) => {
    const dim = t.shape[axis];
    if (part === ":") return;
    if (/^\d+$/.test(part)) index[axis] = [+part];
    else if (/^\d*:\d*$/.test(part)) {
      const [start, stop] = part.split(":");
      index[axis] = {start: start ? +start : 0, stop: stop ? +stop : dim};
    } else if (/^\[\d+(\s*,\s*\d+)*\]$/.test(part)) index[axis] = part.slice(1, -1).split(",").map(Number);
    else throw new InspectError("invalid_index", "Use an index, start:stop, or a list of indices.", "selector.index." + axis);
  });
  const selector = {tensor: t.name, index};
  resolveSelector(selector, t);
  const shape = match[3] === undefined ? undefined : match[3] ? match[3].split(",").map(Number) : [];
  return {index, shape};
}

function combineRanges(a: Range | null | undefined, b: Range | null | undefined) {
  if (!a || !b) return a || b || (a === null || b === null ? null : undefined);
  return {min: Math.min(a.min, b.min), max: Math.max(a.max, b.max),
    abs_max: Math.max(a.abs_max, b.abs_max), abs_p99: Math.max(a.abs_p99, b.abs_p99)};
}

type InspectorProps = {source: TensorDataSource; checkpoint: string; other: string; tensor: Tensor;
  status?: DiffStatus; coverage: number; shortcuts: boolean; copy: (value: unknown) => void; focusRequest: number};

export function TensorInspector({source, checkpoint, other, tensor: t, status, coverage, shortcuts, copy, focusRequest}: InspectorProps) {
  const [layout, setLayout] = useState(() => defaultLayout(t));
  const [page, setPage] = useState(0);
  const [cursorPageSize, setCursorPageSize] = useState<number>();
  const [flatCursors, setFlatCursors] = useState<Cursors[]>([{child: null}]);
  const [fixedInput, setFixedInput] = useState<Record<number, string>>({});
  const [unitAxis, setUnitAxis] = useState(0);
  const [unitsOpen, setUnitsOpen] = useState(t.shape.length > 1);
  const canReadStats = t.shape.length > 1 || unitsOpen;
  const [zoom, setZoom] = useState("fit");
  const [scaleMode, setScaleMode] = useState<ScaleMode>("p99");
  const [dense, setDense] = useState(false);
  const [tint, setTint] = useState(false);
  const [cursor, setCursor] = useState(-1);
  const [pinned, setPinned] = useState(-1);
  const [pinnedPane, setPinnedPane] = useState(0);
  const [draft, setDraft] = useState<string>();
  const [error, setError] = useState<InspectError>();
  const [readError, setReadError] = useState<InspectError>();
  const [retry, setRetry] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [whole, setWhole] = useState<{key: string; unit?: Unit}>();
  const [stats, setStats] = useState<{key: string; mean: number | null}>();
  const [global, setGlobal] = useState<{key: string; child: ReadPage; parent?: ReadPage; delta?: DeltaPage}>();
  const [loaded, setLoaded] = useState<DisplayedPage>();
  const {ref: stage, width} = useWidth<HTMLDivElement>();
  const expr = useRef<HTMLInputElement>(null);
  const {rows, cols, spatial, view, index, shape} = layout;
  const comparable = !status || status === "changed" || status === "same";
  const canRead = comparable || status === "only_a" || status === "only_b";
  const comparing = Boolean(other && comparable);
  const baseSelector = useMemo(() => ({tensor: t.name, shape: shape ?? t.shape, index}), [t.name, t.shape, index, shape]);
  const baseKey = JSON.stringify(baseSelector);
  const layoutSelector = useMemo(() => ({tensor: t.name, index}), [t.name, index]);
  const ids = useMemo(() => indicesFor(resolveSelector(layoutSelector, t)), [t, layoutSelector]);
  const fixed = t.shape.map((_, i) => i).filter(i => i !== rows && i !== cols && !spatial.includes(i));
  const stretch = fixed.filter(axis => ids[axis].length > 1);
  const flats = stretch.length > 0 || t.shape.length === 1;
  const fullRows = rows < 0 ? 1 : ids[rows].length, fullCols = cols < 0 ? 1 : ids[cols].length;
  const tileRows = spatial[0] === undefined ? 1 : ids[spatial[0]].length;
  const tileCols = spatial[1] === undefined ? 1 : ids[spatial[1]].length;
  const tileArea = Math.max(1, tileRows * tileCols);
  const fullCount = elements(ids.map(axis => axis.length));
  const regionCap = Math.floor(pageCap / tileArea);
  const fit = view !== "Values" && zoom === "fit";
  const paneCount = comparing ? 3 : 1;
  const paneWidth = Math.max(1, ((width || 800) - 40 - (paneCount - 1) * 14) / paneCount);
  const gap = tileRows > 1 || tileCols > 1 ? 1 : 0;
  const rowLimit = Math.max(1, Math.floor((heatmapHeight + gap) / Math.max(1, tileRows * minFitCell + gap)));
  const fitsRows = fullRows <= rowLimit && fullRows <= regionCap;
  const balancedRows = Math.max(1, Math.min(rowLimit, Math.ceil(Math.sqrt(regionCap * fullRows / Math.max(1, fullCols)))));
  const windowRows = !fit && fullCount <= pageCap ? fullRows : Math.min(fullRows,
    fitsRows ? fullRows : spatial.length ? Math.min(16, balancedRows) : balancedRows);
  const matrixCell = fitCell(windowRows * tileRows + gap * Math.max(0, windowRows - 1));
  const colLimit = fit ? Math.floor((paneWidth + gap) / Math.max(1, tileCols * matrixCell + gap)) :
    Math.max(!comparing && fitsRows ? 1000 : 1, Math.floor(paneWidth / Math.max(1, tileCols * minFitCell)));
  const windowCols = !fit && fullCount <= pageCap ? fullCols :
    Math.max(1, Math.min(fullCols, !fit && spatial.length ? Math.min(32, colLimit) : colLimit, Math.floor(regionCap / Math.max(1, windowRows))));
  const vector = t.shape.length === 1;
  const flatColsCap = vector ? 128 : 80;
  const flatCell = fitCell(vector ? 1 : Math.ceil(Math.min(fullCount, pageCap) / flatColsCap));
  const flatCols = fit ? Math.max(1, Math.min(flatColsCap, Math.floor(paneWidth / flatCell))) : flatColsCap;
  const flatPageSize = Math.min(vector ? 128 : pageCap, fit ? flatCols * (vector ? 1 : Math.floor(heatmapHeight / flatCell)) : pageCap);
  const cell = fit ? flats ? flatCell : matrixCell : view === "Values" ? minFitCell : Number(zoom);
  const colsPages = Math.max(1, Math.ceil(fullCols / (windowCols || 1)));
  const rectangularPages = Math.max(1, Math.ceil(fullRows / (windowRows || 1)) * colsPages);
  const windowed = !flats && rectangularPages > 1;
  // Flat cursors mark offsets for the page size that produced them.
  const windowPage = windowed ? Math.min(page, rectangularPages - 1) : flats && flatPageSize === cursorPageSize ? page : 0;
  const rowStart = Math.floor(windowPage / colsPages) * windowRows, colStart = (windowPage % colsPages) * windowCols;
  const rowPart = ids[rows]?.slice(rowStart, rowStart + windowRows);
  const colPart = ids[cols]?.slice(colStart, colStart + windowCols);
  const selection = useMemo(() => {
    if (!windowed) return baseSelector;
    const index = {...baseSelector.index};
    for (const [axis, part] of [[rows, rowPart], [cols, colPart]] as const) {
      if (axis < 0 || !part?.length) continue;
      index[axis] = part.at(-1)! - part[0] + 1 === part.length ? {start: part[0], stop: part.at(-1)! + 1} : part;
    }
    return {...baseSelector, index};
  }, [baseSelector, rows, cols, windowed, rowStart, colStart, windowRows, windowCols]);
  const dataset = JSON.stringify([checkpoint, comparing ? other : "", t.name]);
  const selectorText = syntax(t, selection.index ?? {}, shape);
  const viewKey = JSON.stringify([dataset, baseKey, windowPage, selection, rows, cols, spatial, flats ? [flatPageSize, flatCols] : null, retry]);
  const globalKey = JSON.stringify([dataset, baseKey]);
  const activeId = status === "only_a" ? other : checkpoint;
  const wholeKey = activeId + t.name;
  const current = loaded?.key === viewKey ? loaded : undefined;
  const previous = loaded?.dataset === dataset ? loaded : undefined;
  const shown = current ?? previous;
  const failure = error ?? readError;
  const stale = !!failure && !!shown;
  const pending = canRead && !current && !failure;
  const flatCount = Math.max(0, Math.min(flatPageSize, fullCount - windowPage * flatPageSize));
  const colsReading = flats ? Math.min(flatCols, flatCount) :
    (windowed ? colPart?.length ?? 1 : fullCols) * tileCols;
  const rowsReading = flats ? Math.ceil(flatCount / Math.max(1, colsReading)) :
    (windowed ? rowPart?.length ?? 1 : fullRows) * tileRows;
  const computeRange = global?.key !== globalKey && fullCount > 0 && fullCount <= 2 ** 24;
  const skeleton = pending && !shown && heatmapSize(Math.max(1, rowsReading), Math.max(1, colsReading), tileRows, tileCols, cell);
  // The backend rejects explicit zero-length ranges, including the echoed empty-axis range.
  const copySelection = t.shape.includes(0) ? selection : shown?.child.resolved.selector ?? resolveSelector({...selection, shape: t.shape}, t);

  useEffect(() => {if (focusRequest) expr.current?.focus();}, [focusRequest]);

  useEffect(() => {
    setElapsed(0);
    if (!pending) return;
    const started = performance.now();
    const timer = window.setInterval(() => setElapsed(Math.floor((performance.now() - started) / 1000)), 1000);
    return () => window.clearInterval(timer);
  }, [viewKey, pending]);

  useEffect(() => {
    if (!canReadStats) return;
    let live = true;
    source.stats(activeId, {selector: {tensor: t.name}, reduce_over_axes: null}).then(result => {
      if (live) setWhole({key: wholeKey, unit: result.result.units[0]});
    }).catch(() => {if (live) setWhole({key: wholeKey});});
    return () => {live = false;};
  }, [source, activeId, wholeKey, t.name, canReadStats]);

  useEffect(() => {
    if (!canReadStats || !fullCount || fullCount > 2 ** 24 || scaleMode !== "mean") return;
    let live = true;
    Promise.all([
      source.stats(activeId, {selector: baseSelector, reduce_over_axes: null}),
      comparing ? source.stats(other, {selector: baseSelector, reduce_over_axes: null}) : null,
    ]).then(([a, b]) => {
      if (!live) return;
      // Use the checkpoint’s returned centre for a common comparison scale.
      setStats({key: globalKey, mean: a.result.units[0]?.mean ?? b?.result.units[0]?.mean ?? null});
    }).catch(() => {if (live) setStats({key: globalKey, mean: null});});
    return () => {live = false;};
  }, [source, globalKey, activeId, baseSelector, comparing, fullCount, other, scaleMode, canReadStats]);

  useEffect(() => {
    let live = true;
    setReadError(undefined);
    if (!canRead) return;
    const pageSize = flats ? flatPageSize : pageCap;
    async function fetchPage(selector: Selector, include_range: boolean): Promise<LoadedPage> {
      const readChild = source.read_tensor(activeId, {selector, max_elems: pageSize, include_range});
      if (!comparing) return {child: await readChild};
      const [child, parent, delta] = await Promise.all([
        readChild,
        source.read_tensor(other, {selector, max_elems: pageSize, include_range}),
        source.diff(other, checkpoint, {selector, values: true, unit_axis: null, max_elems: pageSize, include_range}),
      ]);
      return {child, parent, delta};
    }
    async function read() {
      try {
        const saved = global?.key === globalKey ? global : undefined;
        const firstPage = saved ?? fetchPage(baseSelector, true);
        const changedPageSize = flats && windowPage === 0 && saved && saved.child.result.returned_elems !== flatCount;
        const [first, window] = await Promise.all([firstPage, windowed || changedPageSize ? fetchPage(selection, false) : undefined]);
        if (!live) return;
        const range = saved ?? {key: globalKey, ...first};
        if (!saved) setGlobal(range);
        let data = window ?? range;
        if (flats && windowPage > 0) {
          const child = await source.read_tensor(activeId, {selector: selection, max_elems: pageSize, cursor: flatCursors[windowPage]?.child});
          let parent: ReadPage | undefined, delta: DeltaPage | undefined;
          if (comparing) {
            const {offset, returned_elems} = child.result;
            const max_elems = pageSize;
            const cursorAt = async <T extends {cursor: string | null}>(initial: T, next: (cursor: string) => Promise<T>) => {
              let value = initial;
              for (let step = 0; step < windowPage && value.cursor; step++) value = await next(value.cursor);
              return value;
            };
            const saved = flatCursors[windowPage];
            const parentPage = (cursor: string) => source.read_tensor(other, {selector: selection, max_elems, cursor});
            const deltaPage = (cursor: string) => source.diff(other, checkpoint, {selector: selection, values: true, unit_axis: null, max_elems, cursor});
            [parent, delta] = saved?.pair === other && saved.parent && saved.delta ? await Promise.all([
              parentPage(saved.parent), deltaPage(saved.delta),
            ]) : await Promise.all([cursorAt(range.parent!, parentPage), cursorAt(range.delta!, deltaPage)]);
            if (parent.result.offset !== offset || delta.result.offset !== offset || delta.result.returned_elems !== returned_elems)
              throw new Error("The comparison windows are not aligned.");
          }
          data = {child, parent, delta};
        }
        if (live) setLoaded({...data, key: viewKey, dataset, range, display: {rows, cols, spatial, flats, flatCols, selectorText}});
      } catch (e) {
        if (live) setReadError(e instanceof InspectError ? e : new InspectError("read_failed", (e as Error).message));
      }
    }
    void read();
    return () => {live = false;};
  }, [source, viewKey, activeId, status]);

  const canNext = windowed ? windowPage + 1 < rectangularPages : !!current?.child.cursor;
  const truncated = windowed || !!shown?.child.truncated || windowPage > 0;
  function stepWindow(step: number) {
    if (step < 0 ? windowPage === 0 : !canNext) return;
    if (step > 0 && flats && current?.child.cursor) {
      setCursorPageSize(flatPageSize);
      setFlatCursors(previous => [...previous.slice(0, windowPage + 1), {
        child: current.child.cursor, parent: current.parent?.cursor, delta: current.delta?.cursor, pair: other,
      }]);
    }
    setPage(windowPage + step); setCursor(-1); setPinned(-1); setDraft(undefined);
  }

  function changeLayout(next: Layout) {
    setLayout(next); setPage(0); setFlatCursors([{child: null}]); setCursor(-1); setPinned(-1);
    setDraft(undefined); setError(undefined); setFixedInput({});
  }

  function changeView(next: View) {
    if (next === view) return;
    if (next === "Mosaic" && !(t.shape.length === 4 && smallAxes(t).length === 2)) return;
    if (next === "Values" || view === "Values" && (next === "Mosaic") === !!spatial.length) {
      setLayout({...layout, view: next});
      return;
    }
    const defaults = defaultLayout(t, next);
    changeLayout(defaults);
  }

  function changeAxis(type: "rows" | "cols", next: number) {
    const counterpart = type === "rows" ? "cols" : "rows";
    const newLayout = {...layout, [type]: next, [counterpart]: next === layout[counterpart] ? layout[type] : layout[counterpart]};
    const newIndex = {...index};
    t.shape.forEach((dim, axis) => {
      if (dim && axis !== newLayout.rows && axis !== newLayout.cols && !spatial.includes(axis))
        newIndex[axis] ??= [0];
      else if (axis === next && Array.isArray(newIndex[axis]) && newIndex[axis].length === 1) delete newIndex[axis];
    });
    changeLayout({...newLayout, index: newIndex});
  }

  function fixedAt(axis: number, position: number, fromProfile = false) {
    if (!Number.isInteger(position) || position < 0 || position >= t.shape[axis]) {
      setError(new InspectError("index_out_of_range", "Enter an index from 0 to " + (t.shape[axis] - 1) + ".", "selector.index." + axis));
      return;
    }
    let next = layout;
    if (fromProfile && (axis === rows || axis === cols || spatial.includes(axis))) {
      const remaining = t.shape.map((_, i) => i).filter(i => i !== axis);
      next = {...layout, rows: remaining.at(-2) ?? -1, cols: remaining.at(-1) ?? -1, spatial: [],
        view: view === "Values" ? "Values" : "Heatmap"};
    }
    const nextIndex: Layout["index"] = {...next.index, [axis]: [position]};
    if (fromProfile) t.shape.forEach((dim, i) => {
      if (dim && i !== next.rows && i !== next.cols && !next.spatial.includes(i)) nextIndex[i] ??= [0];
    });
    changeLayout({...next, index: nextIndex});
  }

  function submit() {
    try {
      const slice = parseSyntax(draft ?? selectorText, t);
      changeLayout({...layout, ...slice});
      setRetry(n => n + 1);
    } catch (e) {setError(e as InspectError);}
  }

  const scoped = shown?.range;
  const mean = stats?.key === scoped?.key ? stats?.mean ?? null : null;
  const shownDisplay = shown?.display;
  const modes = scaleMode === "mean" && mean === null ? "p99" : scaleMode;
  const childRange = scoped?.child.result.range;
  const parentRange = scoped?.parent?.result.range;
  const scale = useMemo(() => scaleFor(t.dtype, combineRanges(childRange, parentRange), modes, mean,
    [...(shown?.child.result.values ?? []), ...(shown?.parent?.result.values ?? [])],
    !!scoped?.child.result.range_skipped || !!scoped?.parent?.result.range_skipped), [t.dtype, childRange, parentRange, modes, mean, shown, scoped]);
  const deltaScale = useMemo(() => scaleFor("float64", scoped?.delta?.result.range,
    modes === "max" ? "max" : "p99", null, shown?.delta?.result.values ?? [], !!scoped?.delta?.result.range_skipped,
    {nonfiniteCount: scoped?.delta?.result.nonfinite_count}),
    [scoped, modes, shown]);
  const mapped = useMemo(() => {
    if (!shown) return null;
    const {rows, cols, spatial, flats, flatCols} = shown.display;
    const positionAxes = indicesFor(shown.child.resolved.selector);
    const result = shown.child.result;
    if (!result.returned_elems) return null;
    if (flats) {
      const width = Math.min(flatCols, result.returned_elems);
      return {height: Math.ceil(result.returned_elems / width), width, tileHeight: 1, tileWidth: 1,
        positions: result.values.map((_, i) => coordinates(i + result.offset, result.selection_shape).map((p, axis) => positionAxes[axis][p])),
        order: result.values.map((_, i) => i)};
    }
    const tr = Math.max(1, positionAxes[spatial[0]]?.length ?? 1), tc = Math.max(1, positionAxes[spatial[1]]?.length ?? 1);
    const height = (positionAxes[rows]?.length ?? 1) * tr, width = (positionAxes[cols]?.length ?? 1) * tc;
    const order = Array<number>(result.values.length);
    const positions = Array<number[]>(result.values.length);
    result.values.forEach((_, i) => {
      const p = coordinates(i, result.selection_shape);
      const y = (p[rows] ?? 0) * tr + (p[spatial[0]] ?? 0);
      const x = (p[cols] ?? 0) * tc + (p[spatial[1]] ?? 0);
      order[y * width + x] = i;
      positions[y * width + x] = p.map((n, axis) => positionAxes[axis][n]);
    });
    return {height, width, tileHeight: tr, tileWidth: tc, order, positions};
  }, [shown, t.shape.length]);
  const focusCell = pinned >= 0 ? pinned : cursor;
  const labels = (i: number) => "[" + (mapped?.positions[i]?.join(", ") ?? "—") + "]";
  const unit = whole?.key === wholeKey ? whole.unit : undefined;
  const mapperAxes = t.shape.map((_, i) => i).filter(i => !spatial.includes(i));
  const panes = useMemo<{title: string; values: Value[]; scale: ReturnType<typeof scaleFor>; dtype: string; changed?: (0 | 1)[]; zero?: boolean}[]>(() => shown ? [
    ...(shown.parent ? [{title: "Parent", values: shown.parent.result.values, scale, dtype: t.dtype,
      zero: scoped?.parent?.result.range?.abs_max === 0 && scoped.parent.result.nonfinite_count === 0}] : []),
    {title: shown.parent ? "Child" : "Values", values: shown.child.result.values, scale, dtype: t.dtype,
      zero: !!shown.parent && childRange?.abs_max === 0 && scoped?.child.result.nonfinite_count === 0},
    ...(shown.delta ? [{title: "Δ (child − parent)", values: shown.delta.result.values, scale: deltaScale,
      dtype: "float64", changed: shown.delta.result.changed}] : []),
  ] : [], [shown, scale, deltaScale, t.dtype, scoped, childRange]);
  const grids = useMemo(() => mapped ? panes.map((p): Cells => ({
    rows: mapped.height, cols: mapped.width, tileRows: mapped.tileHeight, tileCols: mapped.tileWidth,
    values: mapped.order.map(i => p.values[i]), changed: p.changed && mapped.order.map(i => !!p.changed?.[i]), scale: p.scale,
  })) : [], [panes, mapped]);
  const gridHeaders = useMemo(() => {
    if (!mapped || !shownDisplay || view !== "Values") return null;
    const {rows, cols, spatial, flats} = shownDisplay;
    const linear = flats && t.shape.length > 1;
    const rowAxes = [rows, spatial[0] ?? -1].filter(axis => axis >= 0);
    const colAxes = [cols, spatial[1] ?? -1].filter(axis => axis >= 0);
    const positions = (index: number, axes: number[]) => axes.length ?
      axes.map(axis => mapped.positions[index]?.[axis] ?? "—").join("·") : "—";
    const start = shown?.child.result.offset ?? 0;
    return {
      rows: Array.from({length: mapped.height}, (_, i) => linear ? String(start + i * mapped.width) : positions(i * mapped.width, rowAxes)),
      cols: Array.from({length: mapped.width}, (_, i) => linear ? String(i) : positions(i, colAxes)),
      corner: linear ? "Flat row start and column offset" : [
        rowAxes.length ? "Rows axis " + rowAxes.join(" · ") : "",
        colAxes.length ? "Columns axis " + colAxes.join(" · ") : "",
      ].filter(Boolean).join("; ") || "Scalar",
    };
  }, [mapped, shownDisplay, shown?.child.result.offset, t.shape.length, view]);
  const gridWidths = useMemo(() => gridHeaders ? valueGridWidths(grids.map(grid => grid.values), gridHeaders.cols, dense) : [],
    [gridHeaders, grids, dense]);
  const readout = focusCell >= 0 && mapped && panes.length ? labels(focusCell) + " = " + panes.map(p => {
    const v = p.values[mapped.order[focusCell]];
    return (panes.length > 1 ? p.title + " " : "") + printed(v) + " · " + p.dtype.toLowerCase() +
      (v === undefined ? "" : displayUlp(v, p.dtype));
  }).join(" / ") : "";
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (!shortcuts || e.ctrlKey || e.altKey || e.metaKey || !(e.target instanceof HTMLElement) ||
        e.target.closest("input, select, textarea, dialog, [contenteditable]")) return;
    const viewKey = ({h: "Heatmap", v: "Values", m: "Mosaic"} as Record<string, View>)[e.key];
    if (viewKey) {e.preventDefault(); changeView(viewKey);}
    if (e.code === "BracketLeft" || e.code === "BracketRight") {
      e.preventDefault();
      const step = e.code === "BracketLeft" ? -1 : 1;
      if (e.shiftKey) stepWindow(step);
      else if (fixed[0] !== undefined) {
        const a = fixed[0];
        const next = (ids[a][0] ?? 0) + step;
        if (next >= 0 && next < t.shape[a]) fixedAt(a, next);
      }
    }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  return <section className="tb-inspector" data-stale={stale} aria-label="Inspector" onKeyDown={e => {
    if (e.ctrlKey && e.key.toLowerCase() === "c" && !(window.getSelection()?.toString()) &&
      (!(e.target instanceof HTMLInputElement) || e.target.selectionStart === e.target.selectionEnd)) {
      e.preventDefault(); copy(copySelection);
    }
  }}>
    <div className="tb-inspect-body">
      <form className="tb-selector" onSubmit={e => {e.preventDefault(); submit();}}>
        <label htmlFor="tb-selector">selector</label>
        <input id="tb-selector" ref={expr} value={draft ?? selectorText} spellCheck={false}
          title={t.name} aria-invalid={/(?:^|\.)selector(?:\.|$)/.test(failure?.field ?? "")} aria-describedby={failure ? "tb-selector-error" : undefined}
          onChange={e => setDraft(e.target.value)}/>
        {draft !== undefined && <button type="submit">Apply slice</button>}
        <button type="button" onClick={() => copy(copySelection)}>Copy selector</button>
        {current?.child.fingerprints.selection && <span className="tb-meta tb-fingerprint" title={current.child.fingerprints.selection}>
          {current.child.fingerprints.selection.slice(0, 12)}…{current.child.fingerprints.selection.slice(-4)}</span>}
      </form>
      {failure && <p id="tb-selector-error" role="alert" className="tb-alert">{failure.message} · <code>{failure.code}</code>{failure.field && <> <code>{failure.field}</code></>}</p>}
      <dl className="tb-facts">
        <div><dt>dtype</dt><dd>{t.dtype} · {t.editable ? "editable" : <span title={t.read_only_reason ?? ""}>read-only dtype</span>}</dd></div>
        <div><dt>shape</dt><dd>{t.shape.join(" × ") || "scalar"}</dd></div>
        <div><dt>numel</dt><dd>{count(t.numel)}</dd></div>
        <div><dt>bytes</dt><dd>{count(t.nbytes)}</dd></div>
        <div><dt>constant</dt><dd>{unit ? unit.constant ? "yes · " + printed(unit.min ?? undefined) : "no" : "—"}</dd></div>
        <div><dt>sha256</dt><dd>{t.tensor_sha256 ?? "not computed"} <button disabled title="Not checked yet">Compute</button></dd></div>
        {t.shard && <div><dt>shard</dt><dd>{t.shard}</dd></div>}
        <div><dt>axis roles</dt><dd>0 of {count(coverage)} axes known</dd></div>
      </dl>
      <div className="tb-axes"><span>Axes</span>{t.shape.map((dim, i) =>
        <span key={i} className="tb-axis" title={"Axis " + i + " · size " + dim}>axis {i} · unknown</span>)}
        <span className="tb-provenance">Axis meaning is unknown without a graph (v0.3). Axes are shown by index.</span></div>
      {status && status !== "changed" && (other || status !== "same") && <p className="tb-inspect-note"
        data-diff={status}>{status === "same" ? "No change (bitwise equal)" :
          status === "shape_mismatch" || status === "dtype_mismatch" ? statuses[status][1] + ". These tensors can’t be compared value by value."
            : statuses[status][1] + ". This tensor is available on one side."}</p>}
      <div className="tb-tools tb-view-tools">
        <div className="tb-segments" role="group" aria-label="View">
          {(t.shape.length === 4 && smallAxes(t).length === 2 ? ["Mosaic", "Heatmap", "Values"] as const : ["Heatmap", "Values"] as const)
            .map(name => <button type="button" key={name} aria-pressed={name === view} onClick={() => changeView(name)}>
              {name} <kbd>{name[0].toLowerCase()}</kbd>
            </button>)}
        </div>
        {rows >= 0 && <label>{spatial.length ? "Mosaic rows" : "Rows"} <select aria-label="Rows axis" value={rows} onChange={e => changeAxis("rows", +e.target.value)}>
          {mapperAxes.map(i => <option key={i} value={i} disabled={i === cols}>axis {i}</option>)}</select></label>}
        {cols >= 0 && <label>{spatial.length ? "Mosaic cols" : "Cols"} <select aria-label="Columns axis" value={cols} onChange={e => changeAxis("cols", +e.target.value)}>
          {mapperAxes.map(i => <option key={i} value={i} disabled={i === rows}>axis {i}</option>)}</select></label>}
        {fixed.map(axis => <div className="tb-step" key={axis}>
          <label htmlFor={"tb-axis-" + axis}>axis {axis}</label>
          <button aria-label={"Previous index on axis " + axis} disabled={!ids[axis][0]} onClick={() => fixedAt(axis, ids[axis][0] - 1)}>−</button>
          <input id={"tb-axis-" + axis} type="number" min={0} max={t.shape[axis] - 1}
            value={fixedInput[axis] ?? (ids[axis][0] ?? "")} onChange={e => setFixedInput({...fixedInput, [axis]: e.target.value})}
            onBlur={() => {if (fixedInput[axis] !== undefined) fixedAt(axis, fixedInput[axis] === "" ? NaN : +fixedInput[axis]);}}
            onKeyDown={e => {if (e.key === "Enter") e.currentTarget.blur();}}/>
          <button aria-label={"Next index on axis " + axis} disabled={(ids[axis][0] ?? 0) >= t.shape[axis] - 1}
            onClick={() => fixedAt(axis, ids[axis][0] + 1)}>+</button>
        </div>)}
        {view !== "Values" ? <label>Zoom <select value={zoom} onChange={e => setZoom(e.target.value)}>
          <option value="fit">Fit</option>{[1, 2, 4, 8].map(n => <option key={n} value={n}>{n}×</option>)}</select></label>
          : <><label><input type="checkbox" checked={dense} onChange={e => setDense(e.target.checked)}/>Dense</label>
            <label><input type="checkbox" checked={tint} onChange={e => setTint(e.target.checked)}/>Tint</label></>}
        {shown && !(scale.kind === "constant" && panes[0]?.zero && panes[1]?.zero) &&
          <ScaleLegend scale={scale} mode={scaleMode} setMode={setScaleMode} mean={canReadStats && fullCount <= 2 ** 24 ? mean ?? 0 : null}/>}
      </div>
      {pending && <div role="status" className="tb-read-status">
        {flats ? "Reading " + count(flatCount) + " values" :
          "Reading " + count(rowsReading) + " × " + count(colsReading) + " window"}
        {computeRange && (" · computing range over " + count(fullCount) + " values")}
        {elapsed > 0 && <span aria-hidden="true" className="tb-read-elapsed"> · {elapsed} s elapsed</span>}
      </div>}
      {stale && <p role="status" className="tb-read-status">Showing the last valid selection</p>}
      <div ref={stage} className="tb-stage" data-fit={fit} data-pending={pending && !!shown} data-compare={panes.length > 1}>
        {skeleton && Array.from({length: paneCount}, (_, i) => <div key={i} className="tb-window-skeleton" aria-hidden="true"
          style={{width: Math.min(view === "Values" ? (colsReading + 1) * 36 : skeleton.w, paneWidth),
            height: view === "Values" ? Math.min(255, (rowsReading + 1) * 28) : skeleton.h}}/>)}
        {mapped && panes.map((p, i) => {
          const cells = grids[i];
          const pin = (position: number) => {setPinned(position); setPinnedPane(i);};
          return <div className="tb-pane" key={p.title}>
            {(panes.length > 1 || view === "Mosaic") && <h3>{panes.length > 1 ? p.title : "Mosaic"} <span className="tb-meta" title={p.changed ? p.scale.fullLabel : undefined}>
              {p.changed ? p.scale.kind === "constant" ? "" : p.scale.label : view === "Mosaic" ? mapped.tileHeight + "×" + mapped.tileWidth + " cells" : ""}</span></h3>}
            {gridHeaders ? <ValueGrid cells={cells} cursor={focusCell} onCursor={setCursor} onPin={pin}
              dense={dense} tint={tint} dtype={p.dtype} describe={labels} headers={gridHeaders} widths={gridWidths} syncScroll={panes.length > 1 ? scroller => {
                stage.current?.querySelectorAll<HTMLDivElement>(".tb-value-scroll").forEach(peer => {
                  if (peer !== scroller) {peer.scrollTop = scroller.scrollTop; peer.scrollLeft = scroller.scrollLeft;}
                });
              } : undefined}/> :
              <Heatmap cells={cells} cursor={focusCell} onCursor={setCursor} onPin={pin} cell={cell}
                label={view + " of " + shownDisplay?.selectorText + ", " + mapped.height + " by " + mapped.width + ", " + p.scale.label +
                  (p.scale.note ? ", " + p.scale.note : "") +
                  (p.scale.kind === "empty" || p.scale.kind === "constant" || p.scale.kind === "bool" ? "" :
                    ", range " + metric(p.scale.low) + " to " + metric(p.scale.high))}/>}
            {p.zero && <p className="tb-pane-zero tb-meta">All values are 0</p>}
            {p.changed && p.scale.kind === "constant" && <p className="tb-delta-zero tb-meta">{p.scale.label}</p>}
            {p.changed && <p className="tb-delta-key tb-meta">
              <span><span aria-hidden="true">▢</span> changed (bitwise)</span>
              <span><span aria-hidden="true">▨</span> changed, Δ = 0</span>
            </p>}
          </div>;
        })}
        {current && !mapped && <p className="tb-meta">This selection contains no values.</p>}
      </div>
      <div className="tb-readout">
        <span className="tb-meta" aria-live="polite">{readout || (scoped?.child.result.nonfinite_count ? count(scoped.child.result.nonfinite_count) + " non-finite values" : "")}</span>
        {pinned >= 0 && <><button onClick={() => setPinned(-1)}>Unpin</button><button onClick={() => {
          const v = panes[pinnedPane]?.values[mapped?.order[focusCell] ?? -1];
          if (v !== undefined) copy(printed(v));
        }}>Copy value</button></>}
      </div>
      {truncated && !stale && <div className="tb-trunc">
        <span>{windowed ?
          (pending ? "Requested rows " : "Showing rows ") + (rowPart?.[0] ?? 0) + "–" + (rowPart?.at(-1) ?? 0) + " of " + count(fullRows) +
          " · cols " + (colPart?.[0] ?? 0) + "–" + (colPart?.at(-1) ?? 0) + " of " + count(fullCols) +
          " (" + count((rowPart?.length ?? 1) * (colPart?.length ?? 1) * tileArea) + " of " + count(fullCount) + " values)"
          : shown ? "Showing " + count(shown.child.result.offset) + "–" +
            count(shown.child.result.offset + shown.child.result.returned_elems - 1) + " of " + count(fullCount) + " values" : ""}</span>
        <span className="tb-minimap" aria-hidden="true"><i style={{
          left: (windowed ? colStart / fullCols : (shown?.child.result.offset ?? 0) / fullCount) * 100 + "%",
          top: (windowed ? rowStart / fullRows : 0) * 100 + "%",
          width: (windowed ? (colPart?.length ?? 1) / fullCols : (shown?.child.result.returned_elems ?? 0) / fullCount) * 100 + "%",
          height: (windowed ? (rowPart?.length ?? 1) / fullRows : 1) * 100 + "%",
        }}/></span>
        <button disabled={!windowPage} onClick={() => stepWindow(-1)}>Previous window</button>
        <button disabled={!canNext} onClick={() => stepWindow(1)}>Next window</button>
      </div>}
    </div>
    <Units source={source} checkpoint={activeId} other={other} tensor={t} axis={unitAxis} setAxis={setUnitAxis}
      onPosition={(a, p) => fixedAt(a, p, true)} selected={ids[unitAxis]?.length === 1 ? ids[unitAxis][0] : -1}
      comparable={comparable} open={unitsOpen} setOpen={setUnitsOpen}/>
  </section>;
}
