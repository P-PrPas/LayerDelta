import {useEffect, useMemo, useRef, useState} from "react";
import type {Group, NavigatorData, NodeDetail} from "./navigatorSource";
import {axisTotal, layoutGraph, parentPath, roleNode, shortOp, tail, under, valueRefs} from "./navigatorModel";

export type Overlay = "none" | "coverage" | "group" | "binding";
export const overlays: Overlay[] = ["none", "coverage", "group", "binding"];

export function NavigatorCanvas({data, details, scope, expanded, selection, group, members, overlay, matching, searching, unit, showDetails, connectionsReady,
  onSelect, onScope, onExpand, onOverlay}: {
  data: NavigatorData; details: Record<string, NodeDetail>; scope: string; expanded: Set<string>; selection: string;
  group?: Group; members: Set<string>; overlay: Overlay; matching: Set<string>; searching: boolean; unit: number; showDetails: boolean; connectionsReady: boolean;
  onSelect: (id: string) => void; onScope: (path: string) => void; onExpand: (path: string) => void; onOverlay: (value: Overlay) => void;
}) {
  const layout = useMemo(() => layoutGraph(data.nodes, details, scope, expanded), [data.nodes, details, scope, expanded]);
  const viewport = useRef<HTMLDivElement>(null);
  const [area, setArea] = useState({x: 0, y: 0, width: 0, height: 0});
  const currentBox = layout.boxes.find(box => box.id === selection);
  const [focused, setFocused] = useState(false);
  const [zoom, setZoom] = useState(1);
  const modules = useMemo(() => new Map(data.modules.map(m => [m.path, m])), [data.modules]);
  const tensors = useMemo(() => new Map(data.tensors.map(t => [t.name, t])), [data.tensors]);
  const values = useMemo(() => new Map(data.nodes.flatMap(n => n.outputs.map(out => [out.value, out] as const))), [data.nodes]);
  const groupActive = overlay === "group" && !!group;
  const spans = useMemo(() => {
    const map = new Map<string, Group["spans"][number]>();
    for (const item of data.groups) for (const span of item.spans) if (!map.has(span.value)) map.set(span.value, span);
    return map;
  }, [data.groups]);
  const activeSpans = new Map(groupActive ? group.spans.map(span => [span.value, span]) : []);
  const outputOwners = useMemo(() => new Map(data.nodes.flatMap(node => node.outputs.map(out => [out.value, node.id] as const))), [data.nodes]);
  const nodeMap = useMemo(() => new Map(data.nodes.map(node => [node.id, node])), [data.nodes]);
  const portCounts = useMemo(() => {
    const counts = new Map<string, {in: Set<string>; out: Set<string>}>();
    function record(boxId: string, direction: "in" | "out", value: string) {
      let info = counts.get(boxId);
      if (!info) {info = {in: new Set(), out: new Set()}; counts.set(boxId, info);}
      info[direction].add(value);
    }
    for (const [node, detail] of Object.entries(details)) {
      const target = layout.owners.get(node);
      for (const value of valueRefs([detail.args, Object.values(detail.kwargs)])) {
        const producer = outputOwners.get(value);
        const source = producer ? layout.owners.get(producer) : undefined;
        if (source === target) continue;
        if (source) record(source.id, "out", value);
        if (target) record(target.id, "in", value);
      }
    }
    return counts;
  }, [details, layout.owners, outputOwners]);
  const markedOutputs = new Set(groupActive ? group.boundary.filter(b => b.reason === "graph_output").map(b => b.output) : []);
  const opaque = new Set(data.nodes.filter(node => node.opaque_reason).map(node => node.id));
  const ports = data.summary?.outputs.flatMap((value, index) => {
    const producer = outputOwners.get(value);
    const box = producer && layout.owners.get(producer);
    const span = spans.get(value);
    const width = span && values.get(value)?.shape?.[span.dim];
    return box ? [{index, box, width}] : [];
  }) ?? [];
  const portOffset = ports.map(({box}, index) => ports.slice(0, index).filter(port => port.box === box).length);
  const miniWidth = Math.max(layout.width, area.width / zoom);
  const miniHeight = Math.max(layout.height, area.height / zoom);
  function measure() {
    const el = viewport.current;
    if (el) setArea({x: el.scrollLeft, y: el.scrollTop, width: el.clientWidth, height: el.clientHeight});
  }
  useEffect(() => {
    const el = viewport.current;
    if (!el) return;
    const observer = new ResizeObserver(measure);
    observer.observe(el); measure();
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    const el = viewport.current;
    const tensorNode = selection.startsWith("tensor:") ? tensors.get(selection.slice(7))?.roles?.map(roleNode).find(Boolean) : undefined;
    const owner = layout.owners.get(tensorNode ?? selection.replace(/^node:/, ""));
    const ancestor = selection.startsWith("module:") ? layout.boxes.filter(item => item.scope && under(selection.slice(7), item.scope))
      .sort((a, b) => b.scope!.length - a.scope!.length)[0] : undefined;
    const highlighted = group && layout.boxes.find(item => !item.expanded && item.ids.some(member => members.has(member)));
    const box = layout.boxes.find(item => item.id === selection) ?? owner ?? ancestor ?? highlighted;
    if (el && box) el.scrollTo({left: (box.x + box.w / 2) * zoom - el.clientWidth / 2,
      top: (box.y + box.h / 2) * zoom - el.clientHeight / 2});
  }, [selection, scope, layout.boxes, layout.owners, tensors, zoom, group, members]);
  const crumbs = scope ? scope.split(".") : [];
  const boxLabel = (box: typeof layout.boxes[number]) => box.node ? shortOp(box.node.op) : tail(box.scope ?? "");
  const tooMany = layout.boxes.length > 600;
  return <section className="an-canvas" aria-label="Graph canvas">
    <div className="an-canvas-bar">
      <nav aria-label="Graph scope" className="an-breadcrumb">
        <button onClick={() => onScope("")} aria-current={!scope ? "location" : undefined}>root</button>
        {crumbs.map((part, i) => <span key={i}> / <button aria-current={i === crumbs.length - 1 ? "location" : undefined}
          onClick={() => onScope(crumbs.slice(0, i + 1).join("."))}>{part}</button></span>)}
      </nav>
      <label>Overlay <select value={overlay} onChange={e => onOverlay(e.target.value as Overlay)}>
        {overlays.map(value => <option key={value}>{value}</option>)}</select></label>
      <label className="an-zoom">Zoom <select value={zoom} onChange={e => setZoom(+e.target.value)}>
        {[.5, .75, 1, 1.25].map(value => <option key={value} value={value}>{value * 100}%</option>)}</select></label>
    </div>
    <div className="an-map-wrap">
      <div ref={viewport} className="an-viewport" onScroll={measure} onFocus={() => setFocused(true)}
        onBlur={e => {if (!e.currentTarget.contains(e.relatedTarget)) setFocused(false);}}
        role={focused ? "application" : undefined} aria-label="Dataflow layout" tabIndex={0} onKeyDown={e => {
          if (e.key === "Backspace") {e.preventDefault(); onScope(parentPath(scope)); return;}
          if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key) || tooMany) return;
          const step = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : -1;
          const next = layout.boxes[Math.max(0, Math.min(layout.boxes.length - 1, layout.boxes.indexOf(currentBox!) + step))];
          if (next) {
            e.preventDefault(); onSelect(next.id);
            viewport.current?.querySelectorAll<HTMLButtonElement>("[data-box]").forEach(button => {
              if (button.dataset.box === next.id) button.focus({preventScroll: true});
            });
          }
        }}>
        {tooMany ? <p className="an-empty">This scope exceeds 600 visible nodes. Open a narrower module in the outline.</p> :
          <div className="an-map-space" style={{width: layout.width * zoom, height: layout.height * zoom}}>
            <div className="an-map" data-small={zoom < .6} style={{width: layout.width, height: layout.height, transform: "scale(" + zoom + ")"}}>
              <svg className="an-edges" width={layout.width} height={layout.height} aria-hidden="true">
                {layout.edges.map(({from, to, value, source, target}) => {
                  const x = from.x + from.w, y = from.y + 55, tx = to.x, ty = to.y + 55;
                  const mid = x + (tx - x) / 2;
                  const detour = tx < x || tx - x > 130;
                  const lane = Math.min(from.y, to.y) - 14;
                  const link = detour ?
                    ["M", x, y, "H", x + 16, "V", lane, "H", tx - 16, "V", ty, "H", tx].join(" ") :
                    ["M", x, y, "C", mid, y, mid, ty, tx, ty].join(" ");
                  const cut = opaque.has(source) || opaque.has(target);
                  const active = activeSpans.has(value);
                  const span = activeSpans.get(value) ?? spans.get(value);
                  const dim = span && values.get(value)?.shape?.[span.dim];
                  const targetNode = nodeMap.get(target);
                  const concatSpan = active && targetNode?.op.includes(".cat.") ?
                    targetNode.outputs.map(out => activeSpans.get(out.value)).find(Boolean) : undefined;
                  const label = dim === undefined ? "" : String(dim) + (concatSpan ? " @" + concatSpan.offset : "");
                  return <g key={[from.id, to.id, value].join("/")} className="an-edge" data-active={active}
                    data-cut={cut} data-dim={groupActive && !active}>
                    {cut ? <><path d={"M " + x + " " + y + " H " + (mid - 5)}/><path d={"M " + (mid + 5) + " " + y + " V " + ty + " H " + tx}/>
                      <text x={mid} y={y - 5} textAnchor="middle">⊘</text></> :
                      <path d={link}/>}
                    {label && <text x={mid} y={(detour ? lane : (y + ty) / 2) - 7} textAnchor="middle">{label}</text>}
                  </g>;
                })}
              </svg>
              {layout.boxes.map(box => {
                const info = modules.get(box.scope ?? "");
                const memberCount = box.ids.filter(id => members.has(id)).length;
                const hits = searching && (matching.has(box.id) || box.ids.some(id => matching.has("node:" + id)));
                const params = box.node ? details[box.node.id]?.parameters ?? [] : [];
                const isOpaque = !!box.node?.opaque_reason;
                const coverage = info ? info.axes.derived / Math.max(1, axisTotal(info.axes)) : 0;
                const bound = params.filter(name => tensors.get(name)?.binding === "bound").length;
                const portsIn = portCounts.get(box.id)?.in.size ?? 0;
                const portsOut = (portCounts.get(box.id)?.out.size ?? 0) + ports.filter(port => port.box === box).length;
                return <div key={box.id} className={"an-box" + (box.expanded ? " an-frame-expanded" : "")} data-opaque={isOpaque}
                  data-selected={selection === box.id} data-member={groupActive && !!memberCount}
                  data-dim={groupActive && !memberCount && !box.expanded} data-match={hits}
                  data-shaded={overlay === "coverage" && !!info}
                  style={{left: box.x, top: box.y, width: box.w, height: box.h, "--an-coverage": coverage * 30 + "%"} as React.CSSProperties}>
                  <div className="an-box-head">
                    <button className="an-box-select" data-box={box.id} onClick={() => onSelect(box.id)}
                      onDoubleClick={() => {if (box.scope !== undefined) onScope(box.scope);}}
                      onKeyDown={e => {if (e.key === "Enter" && box.scope !== undefined) {e.preventDefault(); onScope(box.scope);}}}>
                      {isOpaque && "⊘ opaque · "}{boxLabel(box)}
                    </button>
                    {box.scope !== undefined && <button className="an-box-expand" aria-label={(box.expanded ? "Collapse " : "Expand ") + box.scope}
                      aria-expanded={box.expanded} onClick={() => onExpand(box.scope!)}>{box.expanded ? "−" : "+"}</button>}
                  </div>
                  <div className="an-box-meta">
                    {box.node ? <><span>{box.node.id}{overlay === "binding" ? " · " + bound + "/" + params.length + " bound" : ""}</span>
                      {!!params.length && <button className="an-tensor-chip" title={params.join(", ")} onClick={() => onSelect("tensor:" + params[0])}>{tail(params[0])}{params.length > 1 && " +" + (params.length - 1)}</button>}</> :
                      <span>{info?.type && info.type + " · "}{box.ids.length} ops{!box.expanded && connectionsReady && " · in " + portsIn + " / out " + portsOut}</span>}
                    {!!info?.opaque_count && <span title="Contains opaque operations" className="an-opaque-label">⊘ {info.opaque_count}</span>}
                    {groupActive && !!memberCount && <span className="an-member-badge">{memberCount} {memberCount === 1 ? "member" : "members"}</span>}
                  </div>
                  {info && !box.expanded && <div className="an-box-coverage" data-opaque={!!info.opaque_count}><i style={{width: Math.min(info.opaque_count ? 93 : 100, coverage * 100) + "%"}}/></div>}
                </div>;
              })}
              {ports.map(({index, box, width}, i) => <span className="an-output" key={index}
                data-member={markedOutputs.has(index)} style={{left: box.x + box.w - 10, top: box.y + 65 + portOffset[i] * 17}}>
                ▷ out {index}{width === undefined ? "" : " · " + width}
              </span>)}
            </div>
          </div>}
      </div>
      {!tooMany && layout.boxes.length > 0 && <svg className="an-minimap" aria-label="Graph minimap" role="img" tabIndex={0}
        viewBox={"0 0 " + miniWidth + " " + miniHeight} preserveAspectRatio="none"
        onKeyDown={e => {
          const step = {ArrowLeft: [-60, 0], ArrowRight: [60, 0], ArrowUp: [0, -60], ArrowDown: [0, 60]}[e.key];
          if (step) {e.preventDefault(); viewport.current?.scrollBy(step[0], step[1]);}
        }}
        onPointerDown={e => {e.currentTarget.setPointerCapture(e.pointerId); pan(e);}}
        onPointerMove={e => {if (e.currentTarget.hasPointerCapture(e.pointerId)) pan(e);}}>
        {layout.boxes.filter(box => !box.expanded).map(box => <rect key={box.id} x={box.x} y={box.y} width={box.w} height={box.h}
          fill={box.ids.some(id => members.has(id)) && groupActive ? "#c6f2e0" : box.ids.some(id => opaque.has(id)) ? "#ff8f7a" : "#46534c"}/>)}
        <rect x={area.x / zoom} y={area.y / zoom} width={area.width / zoom} height={area.height / zoom}
          className="an-mini-viewport" vectorEffect="non-scaling-stroke"/>
      </svg>}
      <div className="an-map-legend"><span>— dataflow</span><span>⊘ opaque</span><span>▷ output</span>{groupActive && <span>■ group</span>}
        {!showDetails && <span>Open a narrower scope to load edges</span>}</div>
      <span className="visually-hidden" aria-live="polite">{focused && currentBox ?
        boxLabel(currentBox) + (groupActive && currentBox.ids.some(id => members.has(id)) ? ", member of " + group.label + ", unit " + unit : "") : ""}</span>
    </div>
  </section>;

  function pan(e: React.PointerEvent<SVGSVGElement>) {
    const bounds = e.currentTarget.getBoundingClientRect();
    viewport.current?.scrollTo({
      left: (e.clientX - bounds.left) / bounds.width * miniWidth * zoom - area.width / 2,
      top: (e.clientY - bounds.top) / bounds.height * miniHeight * zoom - area.height / 2,
    });
  }
}
