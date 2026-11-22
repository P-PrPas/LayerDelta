import {KeyboardEvent, useLayoutEffect, useMemo, useRef, useState} from "react";
import {DiffStatus, DiffTensor, globPattern, Query, Tensor} from "./tensorSource";

export const count = (n: number) => n.toLocaleString("en-US");
export const compact = (n: number, byte = false) => {
  const exponent = n >= 1e9 ? 3 : n >= 1e6 ? 2 : n >= 1e3 ? 1 : 0;
  const value = exponent ? Number((n / 1000 ** exponent).toFixed(2)) : n;
  return count(value) + (byte ? [" B", " KB", " MB", " GB"][exponent] : ["", " K", " M", " B"][exponent]);
};

export function useWindow(length: number, rowHeight = 28) {
  const ref = useRef<HTMLDivElement>(null);
  const [top, setTop] = useState(0);
  const [height, setHeight] = useState(0);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new ResizeObserver(() => setHeight(el.clientHeight));
    observer.observe(el);
    setHeight(el.clientHeight);
    return () => observer.disconnect();
  }, []);
  const visible = Math.ceil(height / rowHeight);
  const first = Math.min(Math.floor(top / rowHeight), Math.max(0, length - visible));
  const start = Math.max(0, first - 20);
  const end = Math.min(length, first + visible + 20);
  return {ref, start, end, visible, onScroll: () => setTop(ref.current?.scrollTop ?? 0),
    scrollTo(index: number) {
      const el = ref.current;
      if (!el) return;
      const y = index * rowHeight;
      if (y < el.scrollTop) el.scrollTop = y;
      else if (y + rowHeight > el.scrollTop + height) el.scrollTop = y + rowHeight - height;
      setTop(el.scrollTop);
    }};
}

const natural = new Intl.Collator("en", {numeric: true});
export const statuses: Record<DiffStatus, [string, string]> = {
  same: ["=", "Bitwise equal"], changed: ["Δ", "Changed"], only_a: ["−", "Only in parent"],
  only_b: ["+", "Only in child"], shape_mismatch: ["≠", "Shape changed"], dtype_mismatch: ["≠", "Dtype changed"],
};
export const tokens = (text: string) => text.trim().split(/\s+/).filter(Boolean)
  .map(token => /^(dtype|rank|editable|size|changed|identical|constant):/i.test(token) ? token.toLowerCase() : token);
const sizeToken = /^size:([<>]=?|=)?(\d+(?:\.\d+)?)(b|kb|mb|gb)?$/;
const deferred = /^(identical|constant):/;
const facets = /^(dtype|rank|editable|size|changed):/;

function sizeBound(token: string) {
  const match = sizeToken.exec(token);
  if (!match) return null;
  const [, op = "=", number, unit = "b"] = match;
  return {op, bytes: Number(number) * ({b:1, kb:1000, mb:1e6, gb:1e9}[unit]!)};
}

export function predicate(text: string): Query {
  const result: Query = {};
  const parts = tokens(text);
  const name = parts.filter(token => !facets.test(token) && !deferred.test(token)).join(" ");
  if (name) result.name_glob = /[*?]/.test(name) ? name : "*" + name + "*";
  for (const token of parts) {
    if (deferred.test(token) || token.startsWith("changed:")) continue;
    const [key, value] = token.split(":");
    if (key === "dtype") (result.dtype ??= []).push(value.toUpperCase());
    else if (key === "rank" && /^\d+$/.test(value)) result.min_ndim = result.max_ndim = +value;
    else if (key === "editable" && (value === "yes" || value === "no")) result.editable = value === "yes";
    else if (key === "size") {
      const size = sizeBound(token);
      if (!size) continue;
      if (size.op === "=") result.min_bytes = result.max_bytes = size.bytes;
      if (size.op === ">") result.min_bytes = Math.floor(size.bytes) + 1;
      if (size.op === ">=") result.min_bytes = Math.ceil(size.bytes);
      if (size.op === "<") result.max_bytes = Math.ceil(size.bytes) - 1;
      if (size.op === "<=") result.max_bytes = Math.floor(size.bytes);
    }
  }
  return result;
}

export function filterTensors(tensors: Tensor[], text: string, differences: Map<string, DiffTensor> | null,
  statsMatches: Set<string> | null) {
  const parts = tokens(text);
  const query = predicate(text);
  const name = query.name_glob ? globPattern(query.name_glob) : null;
  return tensors.filter(t => {
    const changed = differences?.get(t.name)?.status;
    return (!statsMatches || statsMatches.has(t.name)) &&
      (!query.dtype || query.dtype.includes(t.dtype)) &&
      (query.min_ndim === undefined || t.shape.length === query.min_ndim) &&
      (query.editable === undefined || t.editable === query.editable) &&
      t.nbytes >= (query.min_bytes ?? 0) && t.nbytes <= (query.max_bytes ?? Infinity) &&
      (!differences || !parts.includes("changed:yes") || (changed !== undefined && changed !== "same")) &&
      (!differences || !parts.includes("changed:no") || changed === undefined || changed === "same") &&
      (!name || name.test(t.name));
  });
}

type Node = {name: string; key: string; children: Map<string, Node>; tensor?: Tensor;
  count: number; size: number; changed: number};
type Row = Node & {level: number; parent: string | null; open: boolean; group: boolean};

function buildTree(tensors: Tensor[], diffs: Map<string, DiffTensor> | null) {
  const root = new Map<string, Node>();
  for (const tensor of tensors) {
    let children = root, key = "";
    const segments = tensor.name.split(".");
    segments.forEach((name, i) => {
      key += (i ? "." : "") + name;
      const leaf = i === segments.length - 1;
      const lookup = (leaf ? "leaf:" : "group:") + name;
      let node = children.get(lookup);
      if (!node) {
        node = {name, key: (leaf ? "leaf:" : "group:") + key, children: new Map(), count: 0, size: 0, changed: 0};
        children.set(lookup, node);
      }
      node.count++;
      node.size += tensor.nbytes;
      const status = diffs?.get(tensor.name)?.status;
      if (status && status !== "same") node.changed++;
      if (leaf) node.tensor = tensor;
      children = node.children;
    });
  }
  return root;
}

function flatten(nodes: Map<string, Node>, expanded: Record<string, boolean>, selected: string,
  filter: boolean, level = 1, parent: string | null = null): Row[] {
  const rows: Row[] = [];
  for (const entry of [...nodes.values()].sort((a, b) => natural.compare(a.name, b.name))) {
    let node = entry;
    let name = node.name;
    while (node.children.size === 1) {
      const child = node.children.values().next().value!;
      if (child.tensor) break;
      node = child;
      name += "." + node.name;
    }
    const group = !node.tensor;
    const open = group && (expanded[node.key] ?? (filter || level === 1 || selected.startsWith(node.key.slice(6) + ".")));
    rows.push({...node, name, level, parent, open, group});
    if (open) rows.push(...flatten(node.children, expanded, selected, filter, level + 1, node.key));
  }
  return rows;
}

function MiddleName({name}: {name: string}) {
  const tail = Math.min(12, Math.ceil(name.length / 2));
  return <span className="tb-name-text"><span>{name.slice(0, -tail)}</span><span>{name.slice(-tail)}</span></span>;
}

export function TensorTree({tensors, selected, onSelect, filter, diffs, maxBytes, focusInspector, focusRequest}:
  {tensors: Tensor[]; selected: string; onSelect: (name: string) => void; filter: boolean;
    diffs: Map<string, DiffTensor> | null; maxBytes: number; focusInspector: () => void; focusRequest: number}) {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [activeKey, setActiveKey] = useState("");
  const [focused, setFocused] = useState(false);
  const restoringFocus = useRef(false);
  const scrollByMode = useRef({browse: 0, compare: 0});
  const mode = diffs ? "compare" : "browse";
  const tree = useMemo(() => buildTree(tensors, diffs), [tensors, diffs]);
  const rows = useMemo(() => flatten(tree, expanded, selected, filter), [tree, expanded, selected, filter]);
  const active = rows.findIndex(row => row.key === activeKey);
  const current = active < 0 ? Math.max(0, rows.findIndex(row => row.tensor?.name === selected)) : active;
  const windowed = useWindow(rows.length);
  const first = Math.min(windowed.start, current);
  const activeRendered = current >= windowed.start && current < windowed.end;
  const visible = rows.slice(windowed.start, windowed.end).map((row, i) => ({row, i: i + windowed.start}));
  if (!activeRendered && rows[current]) visible.splice(first === current ? 0 : visible.length, 0, {row: rows[current], i: current});

  useLayoutEffect(() => {
    if (!windowed.ref.current) return;
    windowed.ref.current.scrollTop = scrollByMode.current[mode];
    windowed.onScroll();
  }, [mode]);

  const focusRow = () => windowed.ref.current?.querySelector<HTMLElement>("#tb-tree-row-" + current)?.focus({preventScroll: true});
  useLayoutEffect(() => {
    if (focusRequest) (rows.length ? focusRow() : windowed.ref.current?.focus());
  }, [focusRequest]);
  useLayoutEffect(() => {
    if (!restoringFocus.current) return;
    restoringFocus.current = false;
    focusRow();
  });

  function navigate(event: KeyboardEvent<HTMLDivElement>) {
    const row = rows[current];
    if (!row) return;
    let next = current;
    const toggle = (open: boolean) => setExpanded(prev => ({...prev, [row.key]: open}));
    switch (event.key) {
      case "ArrowDown": next++; break;
      case "ArrowUp": next--; break;
      case "Home": next = 0; break;
      case "End": next = rows.length - 1; break;
      case "PageDown": next += windowed.visible; break;
      case "PageUp": next -= windowed.visible; break;
      case "ArrowRight":
        if (row.tensor) {onSelect(row.tensor.name); focusInspector();}
        else if (!row.open) toggle(true);
        else next++;
        break;
      case "ArrowLeft":
        if (row.open) toggle(false);
        else if (row.parent) next = rows.findIndex(r => r.key === row.parent);
        break;
      case "Enter":
        if (row.tensor) onSelect(row.tensor.name);
        else toggle(!row.open);
        break;
      case "*":
        setExpanded(prev => ({...prev, ...Object.fromEntries(rows.filter(r =>
          r.group && r.parent === row.parent).map(r => [r.key, true]))}));
        break;
      default: return;
    }
    event.preventDefault();
    next = Math.max(0, Math.min(rows.length - 1, next));
    restoringFocus.current = next !== current;
    setActiveKey(rows[next].key);
    windowed.scrollTo(next);
  }

  return <div className="tb-tree-scroll" ref={windowed.ref} onScroll={e => {
    scrollByMode.current[mode] = e.currentTarget.scrollTop;
    windowed.onScroll();
  }}
    role="treegrid" aria-label="Tensors" aria-rowcount={rows.length} aria-colcount={diffs ? 6 : 5}
    tabIndex={rows.length ? -1 : 0}
    onFocus={() => setFocused(true)} onBlur={e => {if (!e.currentTarget.contains(e.relatedTarget)) setFocused(false);}}
    onKeyDown={navigate}>
    <div role="rowgroup" style={{height: rows.length * 28, position: "relative"}}>
      {visible.map(({row, i}) => {
        const t = row.tensor;
        const diff = t && diffs?.get(t.name);
        const status = t && diffs ? diff?.status ?? "same" : undefined;
        const logSize = 36 * Math.log10(row.size + 1) / Math.log10(Math.max(maxBytes, 1) + 1);
        return <div key={row.key} id={"tb-tree-row-" + i} role="row" aria-rowindex={i + 1} tabIndex={i === current ? 0 : -1}
          aria-level={row.level} aria-expanded={row.group ? row.open : undefined}
          aria-selected={t ? t.name === selected : undefined}
          className={"tb-row" + (row.group ? " tb-group" : "") + (diffs ? " tb-row-diff" : "")}
          data-focused={focused && i === current} style={{top: i * 28}}
          onClick={e => {
            setActiveKey(row.key); e.currentTarget.focus({preventScroll: true});
            if (t) onSelect(t.name);
            else setExpanded(prev => ({...prev, [row.key]: !row.open}));
          }}>
          <span role="gridcell" className="tb-row-name" title={t?.name ?? row.key.slice(6)} style={{paddingLeft: 10 + (row.level - 1) * 14}}>
            {Array.from({length: row.level - 1}, (_, level) => <i key={level} className="tb-guide" style={{left: 15 + level * 14}}/>)}
            <span aria-hidden="true" className="tb-chevron">{row.group ? row.open ? "▾" : "▸" : ""}</span>
            <MiddleName name={row.name}/>
            {row.group && <small>{count(row.count)}</small>}
          </span>
          <span role="gridcell">{t && <span className="tb-dtype" data-editable={t.editable}>{t.dtype}</span>}</span>
          <span role="gridcell" className="tb-shape">{t ? t.shape.join("×") || "scalar" : ""}</span>
          <span role="gridcell" className="tb-size">{compact(row.size, true)}<i style={{width: Math.min(36, logSize)}}/></span>
          {diffs && <span role="gridcell" className="tb-size tb-delta" title="Changed values · relative L2">
            {row.group ? row.changed || "" : diff?.changed_elems ?? ""}
            {!!diff?.rel_l2 && <i style={{width: 36 * Math.min(diff.rel_l2, 1)}}/>}
          </span>}
          <span role="gridcell" className="tb-flags">
            {t && !t.editable && <span role="img" title="Read-only dtype" aria-label="Read-only dtype">🔒︎</span>}
            {status && <span role="img" data-diff={status} title={statuses[status][1]} aria-label={statuses[status][1]}>{statuses[status][0]}</span>}
          </span>
        </div>;
      })}
    </div>
  </div>;
}
