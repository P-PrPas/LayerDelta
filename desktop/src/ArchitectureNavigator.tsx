import {useEffect, useMemo, useRef, useState} from "react";
import {liveCheckpointSource} from "./checkpointSource";
import {inspectErrorText, type Checkpoint} from "./tensorSource";
import {count, useWindow} from "./tensorTree";
import {attachGraph, getGraphNode, loadNavigator, type NavigatorData, type NodeDetail} from "./navigatorSource";
import {axisTotal, groupNodes, outline, outlineMatches, shortOp, tail, under, type Row} from "./navigatorModel";
import {NavigatorCanvas, overlays, type Overlay} from "./NavigatorCanvas";
import {NavigatorInspector, type OpenTensor, type Tab} from "./NavigatorInspector";
import "./architectureNavigator.css";

const toggle = (items: Set<string>, item: string) => {
  const next = new Set(items);
  if (next.has(item)) next.delete(item); else next.add(item);
  return next;
};
const query = () => new URLSearchParams(location.search);
function saveLocation(id: string, selection = "", unit = 0, scope = "") {
  const url = new URL(location.href);
  url.search = new URLSearchParams({screen: "architecture", ...(id ? {ckpt: id} : {}), ...(selection ? {sel: selection} : {}),
    ...(unit ? {u: String(unit)} : {}), ...(scope ? {scope} : {})}).toString();
  history.replaceState(null, "", url);
}
const percentage = (derived: number, total: number) => total ? (derived === total ? 100 : Math.min(99.9, 100 * derived / total)).toFixed(1) + "%" : "—";

function ModuleOutline({rows, shown, selected, expanded, searching, matched, fallback, onSelect, onOpen, onExpand}: {
  rows: Map<string, Row>; shown: Set<string>; selected: string; expanded: Set<string>; searching: boolean; matched: number; fallback: boolean;
  onSelect: (id: string) => void; onOpen: (path: string) => void; onExpand: (id: string) => void;
}) {
  const [active, setActive] = useState("");
  const children = new Map<string | null, Row[]>();
  for (const row of rows.values()) {
    const siblings = children.get(row.parent);
    if (siblings) siblings.push(row); else children.set(row.parent, [row]);
  }
  const flat: {row: Row; depth: number; expandable: boolean; open: boolean}[] = [];
  function visit(parent: string | null, depth: number) {
    for (const row of children.get(parent) ?? []) {
      if (!shown.has(row.id)) continue;
      const expandable = !!children.get(row.id)?.some(r => shown.has(r.id));
      const open = expandable && (searching || expanded.has(row.id));
      flat.push({row, depth, expandable, open});
      if (open) visit(row.id, depth + 1);
    }
  }
  const roots = children.get(null);
  if (roots) roots.sort((a, b) => Number(!!a.id.startsWith("binding:")) - Number(!!b.id.startsWith("binding:")));
  visit(null, 0);
  const visible = useWindow(flat.length);
  const index = Math.max(0, flat.findIndex(item => item.row.id === (active || selected)));
  function focus(next: number) {
    if (!flat.length) return;
    const position = Math.max(0, Math.min(flat.length - 1, next));
    const id = flat[position].row.id;
    setActive(id); visible.scrollTo(position);
    requestAnimationFrame(() => visible.ref.current?.querySelectorAll<HTMLDivElement>("[data-outline-row]").forEach(el => {
      if (el.dataset.outlineRow === id) el.focus({preventScroll: true});
    }));
  }
  return <>
    <p className="an-outline-summary" role="status">{fallback ? "From tensor names, no graph" : "Module outline"} · {count(searching ? matched : flat.length)} {searching ? "matches" : "rows"}</p>
    <div role="treegrid" aria-label="Module outline" aria-rowcount={flat.length} aria-colcount={4} ref={visible.ref}
      className="an-outline-scroll" onScroll={visible.onScroll} onKeyDown={e => {
        const item = flat[index];
        if (!item) return;
        switch (e.key) {
          case "ArrowDown": focus(index + 1); break;
          case "ArrowUp": focus(index - 1); break;
          case "Home": focus(0); break;
          case "End": focus(flat.length - 1); break;
          case "ArrowRight":
            if (item.expandable && !item.open) onExpand(item.row.id);
            else if (item.open) focus(index + 1);
            break;
          case "ArrowLeft":
            if (item.open && !searching) onExpand(item.row.id);
            else if (item.row.parent) focus(flat.findIndex(r => r.row.id === item.row.parent));
            break;
          case "Enter": case " ":
            if (e.altKey && item.row.path && item.row.id.startsWith("module:")) onOpen(item.row.path);
            else onSelect(item.row.id);
            break;
          default: return;
        }
        e.preventDefault();
      }}>
      {!flat.length && <p className="an-empty">No matching rows.</p>}
      <div role="rowgroup" style={{height: flat.length * 28}}>
        {flat.slice(visible.start, visible.end).map(({row, depth, expandable, open}, i) => {
          const rowIndex = i + visible.start;
          const axes = row.summary?.axes;
          const coverage = axes ? axes.derived / Math.max(1, axisTotal(axes)) : 0;
          return <div role="row" key={row.id} data-outline-row={row.id} className="an-outline-row" aria-level={depth + 1}
            aria-selected={selected === row.id} aria-expanded={expandable ? open : undefined} aria-rowindex={rowIndex + 1}
            tabIndex={rowIndex === index ? 0 : -1} onFocus={() => setActive(row.id)}
            onDoubleClick={() => {if (row.path && row.id.startsWith("module:")) onOpen(row.path);}}
            onClick={() => {setActive(row.id); onSelect(row.id);}} style={{top: rowIndex * 28, paddingLeft: 7 + depth * 14}}>
            <div role="gridcell" className="an-row-name" title={row.tensor ?? row.path ?? row.label}>
              {expandable ? <button tabIndex={-1} aria-label={(open ? "Collapse " : "Expand ") + row.label} onClick={e => {
                e.stopPropagation(); onExpand(row.id);
              }}>{open ? "▾" : "▸"}</button> : <span aria-hidden="true"/>}
              <strong>{row.label}</strong>
            </div>
            <div role="gridcell" className="an-meta" title={row.type ?? undefined}>{row.type ?? ""}</div>
            <div role="gridcell" className="an-row-count an-meta" title={row.summary ? count(row.summary.tensor_count) + " tensors" : undefined}>
              {row.summary ? count(row.summary.node_count) : row.id.startsWith("binding:") ? count(children.get(row.id)?.length ?? 0) : ""}
            </div>
            <div role="gridcell" className="an-row-coverage" title={axes ? count(axes.derived) + " of " + count(axisTotal(axes)) + " derived" : undefined}>
              {axes && <span className="an-meter"><i style={{width: coverage * 100 + "%"}}/></span>}
              {row.flags && <span className="an-flag" aria-label={row.flags.includes("⊘") ? "Opaque operation" : row.flags.includes("∅") ? "Unbound tensor" : "Open group"}>{row.flags}</span>}
            </div>
          </div>;
        })}
      </div>
    </div>
  </>;
}

function GraphWorkspace({id, data, openTensor, reference, onAttached}: {id: string; data: NavigatorData; openTensor: OpenTensor; reference: boolean; onAttached: () => void}) {
  const [selection, setSelection] = useState(() => query().get("sel") ?? "");
  const [unit, setUnit] = useState(() => Math.max(0, Number(query().get("u")) || 0));
  const [scope, setScope] = useState(() => {
    const path = query().get("scope") ?? "";
    return data.modules.some(module => module.path === path) ? path : "";
  });
  const [tab, setTab] = useState<Tab>(() => selection.startsWith("group:") ? "Group" : selection.startsWith("tensor:") ? "Tensor" : "Node");
  const [overlay, setOverlay] = useState<Overlay>(tab === "Group" ? "group" : "none");
  const [expansion, setExpansion] = useState(new Set<string>());
  const [outlineOpen, setOutlineOpen] = useState(new Set<string>());
  const [search, setSearch] = useState("");
  const [details, setDetails] = useState<Record<string, NodeDetail>>({});
  const [nodeError, setNodeError] = useState<{id: string; text: string}>();
  const [edgeError, setEdgeError] = useState("");
  const [readyScope, setReadyScope] = useState<string>();
  const searchBox = useRef<HTMLInputElement>(null);
  const shortcuts = useRef<HTMLDialogElement>(null);
  const requested = useRef(new Map<string, Promise<NodeDetail>>());
  const node = selection.startsWith("node:") ? selection.slice(5) : "";
  const tensor = selection.startsWith("tensor:") ? selection.slice(7) : "";
  const group = data.groups.find(g => selection === "group:" + g.group_id);
  const currentUnit = group ? Math.min(group.size - 1, Math.max(0, Math.floor(unit))) : 0;
  const tensorMap = useMemo(() => new Map(data.tensors.map(t => [t.name, t])), [data.tensors]);
  const nodes = useMemo(() => new Map(data.nodes.map(n => [n.id, n])), [data.nodes]);
  const scoped = useMemo(() => data.nodes.filter(n => under(n.module_path ?? "", scope)), [data.nodes, scope]);
  const rows = useMemo(() => outline(data, details), [data, details]);
  const searchResult = useMemo(() => outlineMatches(rows, search), [rows, search]);
  const members = useMemo(() => groupNodes(group, data.nodes, details), [group, data.nodes, details]);
  const coverage = data.summary?.coverage;
  const coverageText = coverage ? count(coverage.derived) + " of " + count(axisTotal(coverage)) + " axes derived (" +
    percentage(coverage.derived, axisTotal(coverage)) + ")" : "";
  const [coverageOpen, setCoverageOpen] = useState(false);
  const detailsForScope = scoped.length <= 600;

  useEffect(() => {saveLocation(id, selection, currentUnit, scope);}, [id, selection, currentUnit, scope]);
  useEffect(() => {
    let live = true;
    const ids = new Set(detailsForScope ? scoped.map(n => n.id) : []);
    if (node && nodes.has(node)) ids.add(node);
    const queue = [...ids].filter(key => !details[key]);
    const errors: string[] = [];
    const batch: Record<string, NodeDetail> = {};
    setEdgeError("");
    function request(idToRead: string) {
      const existing = requested.current.get(idToRead);
      if (existing) return existing;
      const promise = getGraphNode(id, idToRead).catch(error => {requested.current.delete(idToRead); throw error;});
      requested.current.set(idToRead, promise);
      return promise;
    }
    if (node && nodes.has(node)) request(node).then(result => {
      if (live) {setDetails(old => ({...old, [node]: result})); setNodeError(undefined);}
    }).catch(error => {if (live) setNodeError({id: node, text: inspectErrorText(error)});});
    const chunks = Array.from({length: Math.min(8, queue.length)}, () => [] as string[]);
    queue.splice(0).forEach((entry, index) => chunks[index % chunks.length].push(entry));
    Promise.all(chunks.map(async chunk => {
      for (const next of chunk) {
        if (!live) return;
        try {batch[next] = await request(next);}
        catch (error) {errors.push(inspectErrorText(error));}
      }
    })).then(() => {
      if (live) {
        if (Object.keys(batch).length) setDetails(old => ({...old, ...batch}));
        setReadyScope(scope);
        if (errors.length) setEdgeError("Some graph connections could not be loaded. " + errors[0]);
      }
    });
    return () => {live = false;};
  }, [id, detailsForScope, scoped, scope, node, nodes]);

  function select(next: string) {
    setSelection(next);
    if (next.startsWith("group:")) {
      if (next !== selection) setUnit(0);
      setTab("Group"); setOverlay("group"); setScope("");
    } else if (next.startsWith("tensor:")) setTab("Tensor");
    else if (next) setTab("Node");
    else {setTab("Group"); setOverlay("none");}
    if (!next.startsWith("group:") && overlay === "group") setOverlay("none");
    const row = rows.get(next);
    if (row) setOutlineOpen(previous => {
      const expanded = new Set(previous);
      for (let parent = row.parent; parent; parent = rows.get(parent)?.parent ?? null) expanded.add(parent);
      return expanded;
    });
  }
  function scopeTo(path: string) {
    setScope(path);
    if (path) {select("module:" + path); setOutlineOpen(prev => new Set(prev).add("module:" + path));}
  }
  function showNode(nodeId: string) {
    const found = nodes.get(nodeId);
    if (!found) return;
    const path = found.module_path ?? "";
    setScope(path); select("node:" + nodeId);
    if (path) setOutlineOpen(prev => new Set(prev).add("module:" + path));
  }
  function changeUnit(next: number) {
    if (group && Number.isInteger(next)) setUnit(Math.max(0, Math.min(group.size - 1, next)));
  }
  const shortcutsButton = <button type="button" aria-label="Architecture shortcuts" onClick={() => shortcuts.current?.showModal()}>?</button>;

  useEffect(() => {
    function keyboard(e: KeyboardEvent) {
      if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey || e.target instanceof Element && e.target.closest("input, select, textarea, dialog, [contenteditable]")) return;
      if (e.key === "/") {e.preventDefault(); searchBox.current?.focus();}
      if (e.key === "?") shortcuts.current?.showModal();
      if (e.key === "Escape" && search) {setSearch(""); searchBox.current?.focus();}
      if (e.key === "[" || e.key === "]") changeUnit(currentUnit + (e.key === "[" ? -1 : 1));
      if (e.key === "o" && data.summary) setOverlay(overlays[(overlays.indexOf(overlay) + 1) % overlays.length]);
      if (e.key === "t" && tensor) openTensor(id, tensor);
      if (e.key === "g") {
        const groupId = tensorMap.get(tensor)?.groups?.[0]?.group_id ?? details[node]?.groups[0];
        if (groupId) select("group:" + groupId);
      }
    }
    window.addEventListener("keydown", keyboard);
    return () => window.removeEventListener("keydown", keyboard);
  });

  return <div className="an-workspace">
    <div className="an-graph-status">
      {coverage ? <button type="button" aria-expanded={coverageOpen} onClick={() => setCoverageOpen(!coverageOpen)}>
        <strong>Graph attached</strong> · {data.summary?.graph.provider} · {coverageText} · {count(coverage.opaque_ops.reduce((n, op) => n + op.count, 0))} opaque · {count(coverage.groups.open)} open groups
      </button> : <span className="an-meta">No graph attached</span>}
      {shortcutsButton}
    </div>
    {coverage && coverageOpen && <section className="an-coverage" aria-label="Graph coverage">
            <div><h3>Axes</h3>
        <div className="an-coverage-bar" aria-hidden="true">{(["derived", "declared", "inferred", "unknown"] as const).map(item =>
          <i key={item} data-quality={item} style={{flex: coverage[item] ?? 0}}/>)}</div>
        {(["derived", "declared", "inferred", "unknown"] as const).map(item => <button type="button" key={item} onClick={() => setSearch("prov:" + item)}>
          {item} {count(coverage[item] ?? 0)}</button>)}</div>
      <div><h3>Binding</h3>{Object.entries(coverage.tensors).map(([item, n]) => <button type="button" key={item} onClick={() => setSearch(item)}>{item} {count(n)}</button>)}</div>
      <div><h3>Groups and boundaries</h3>{Object.entries(coverage.groups).map(([item, n]) =>
        <button type="button" key={item} onClick={() => {setSearch(item); setTab("Group");}}>{item} {count(n)}</button>)}
        {coverage.opaque_ops.map(({op, count: n}) => <button type="button" key={op} onClick={() => {
          setSearch("opaque op:" + shortOp(op));
          const found = data.nodes.find(node => node.op === op && node.opaque_reason);
          if (found) showNode(found.id);
        }}>{shortOp(op)} {count(n)}</button>)}</div>
    </section>}
    <div className="an-panes">
      <aside className="an-outline" aria-label="Outline">
        <div className="an-search"><input ref={searchBox} aria-label="Search modules, operations and tensors"
          placeholder="Search modules, tensors, op:…" value={search} onChange={e => setSearch(e.target.value)}
          onKeyDown={e => {
            if (e.key === "Escape") {
              e.preventDefault(); setSearch("");
              requestAnimationFrame(() => document.querySelector<HTMLElement>(".an-outline-row[tabindex='0']")?.focus());
            }
            if (e.key === "Enter") {
              const token = /(?:^|\s)group:([^\s]+)/.exec(search)?.[1]?.replace(/…/g, "");
              const match = token && data.groups.filter(g => g.group_id.startsWith(token));
              if (match && match.length === 1) select("group:" + match[0].group_id);
            }
          }}/><kbd>/</kbd></div>
        {data.summary && <div className="an-tokens">
          {[data.nodes[0] ? "op:" + shortOp(data.nodes[0].op) : "", "opaque", "open", "unbound", "unused"].filter(Boolean).map(token =>
            <button type="button" key={token} aria-pressed={search === token} onClick={() => setSearch(search === token ? "" : token)}>{token}</button>)}
        </div>}
        <ModuleOutline rows={rows} shown={searchResult.shown} selected={selection} expanded={outlineOpen}
          searching={!!search.trim()} matched={searchResult.matching.size} fallback={!data.summary}
          onSelect={next => {
            select(next);
            const item = rows.get(next);
            if (item?.path !== undefined && !under(item.path, scope)) setScope(item.path);
          }}
          onOpen={scopeTo} onExpand={key => setOutlineOpen(previous => toggle(previous, key))}/>
      </aside>
      {data.summary ? <div className="an-graph-pane">
        {edgeError && <p className="an-inline-alert" role="alert">{edgeError}</p>}
        {detailsForScope && readyScope !== scope && <p className="an-edge-status" role="status">Loading graph connections…</p>}
        <NavigatorCanvas data={data} details={details} scope={scope} expanded={expansion} selection={selection}
          group={group} members={members} overlay={overlay} matching={searchResult.matching} searching={!!search.trim()} unit={currentUnit}
          showDetails={detailsForScope} connectionsReady={readyScope === scope && !edgeError} onSelect={select} onScope={scopeTo}
          onExpand={path => setExpansion(prev => toggle(prev, path))} onOverlay={setOverlay}/>
      </div> : <AttachCard id={id} offers={data.offers} reference={reference} onAttached={onAttached}/>}
      <NavigatorInspector checkpoint={id} tab={tab} setTab={setTab} node={details[node]} tensor={tensorMap.get(tensor)}
        module={selection.startsWith("module:") ? selection.slice(7) : undefined}
        group={group} groups={search === "open" ? data.groups.filter(g => !g.closed) : search === "closed" ? data.groups.filter(g => g.closed) : data.groups}
        tensors={tensorMap} unit={currentUnit} setUnit={changeUnit} nodeError={nodeError?.id === node ? nodeError.text : undefined}
        onSelect={select} onNode={showNode} openTensor={openTensor} graph={!!data.summary}/>
    </div>
    <dialog ref={shortcuts} className="an-dialog"><h2>Architecture shortcuts</h2>
      <dl className="an-facts"><dt>/</dt><dd>Search the outline</dd><dt>↑ ↓ ← →</dt><dd>Move, expand or collapse</dd>
        <dt>Enter</dt><dd>Select an outline row or enter a canvas frame</dd><dt>Backspace</dt><dd>Up one canvas scope</dd>
        <dt>g · t · o</dt><dd>Group · Tensors · cycle overlay</dd><dt>[ · ]</dt><dd>Previous or next unit</dd></dl>
      <form method="dialog"><button>Close</button></form>
    </dialog>
  </div>;
}

function AttachCard({id, offers, reference, onAttached}: {id: string; offers: NavigatorData["offers"]; reference: boolean; onAttached: () => void}) {
  const builtin = offers.providers.find(p => p.name === "builtin");
  const [model, setModel] = useState(reference && builtin?.models?.includes("reference_cnn") ? "reference_cnn" : "");
  const [job, setJob] = useState<{stage: string; done: number; total: number}>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const active = useRef<AbortController>(null);
  useEffect(() => () => active.current?.abort(), []);
  async function submit(spec: {provider: "builtin"; model: string} | {reuse: string}) {
    if (busy) return;
    const controller = new AbortController();
    active.current = controller; setError(""); setBusy(true); setJob(undefined);
    try {
      await attachGraph(id, spec, setJob, controller.signal);
      onAttached();
    } catch (error) {if (!controller.signal.aborted) setError(inspectErrorText(error));}
    finally {if (!controller.signal.aborted) setBusy(false);}
  }
  return <section className="an-attach" aria-labelledby="attach-heading">
    <div className="an-attach-card">
      <h1 id="attach-heading">Attach a graph</h1>
      <p>A graph tells LayerDelta what each axis means and which slices move together.</p>
      <form onSubmit={e => {e.preventDefault(); if (model) void submit({provider: "builtin", model});}}>
        <fieldset disabled={busy}><legend>Graph source</legend>
          <div className="an-provider"><strong>Built-in model</strong><p>Runs only LayerDelta’s code.</p>
            <label>Model <select value={model} onChange={e => setModel(e.target.value)}>
              <option value="">Choose a model</option>{builtin?.models?.map(value => <option key={value}>{value}</option>)}</select></label>
          </div>
          {offers.providers.filter(p => p.name !== "builtin").map(provider => <div className="an-provider" key={provider.name}>
            <strong>{provider.name === "onnx" ? "ONNX" : "Torch export"}</strong><p>{provider.reason ?? "Unavailable in this workspace."}</p></div>)}
          <button className="primary" disabled={!model || !builtin?.enabled}>Attach graph</button>
        </fieldset>
      </form>
      {offers.reusable.map(graph => <div className="an-provider" key={graph.ir_sha256}>
        <p>Reuse a {graph.provider} graph from {count(graph.checkpoints)} compatible checkpoint{graph.checkpoints === 1 ? "" : "s"}.</p>
        <button disabled={busy} onClick={() => void submit({reuse: graph.ir_sha256})}>Attach existing graph</button>
      </div>)}
      {busy && <ol className="an-stages" role="status" aria-label="Graph attachment progress">
        {["Building", "Binding", "Deriving"].map((stage, i) => <li key={stage}>
          {(job?.done ?? 0) > i ? "✓ " : ""}{stage}{job?.stage === stage.toLowerCase() ? "… " : ""}</li>)}
      </ol>}
      {error && <p role="alert" className="an-inline-alert">{error}</p>}
    </div>
  </section>;
}

export function ArchitectureNavigator({openTensor}: {openTensor: OpenTensor}) {
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>();
  const [id, setId] = useState(() => query().get("ckpt") ?? "");
  const [loaded, setLoaded] = useState<{id: string; data: NavigatorData}>();
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    let live = true;
    liveCheckpointSource.listCheckpoints().then(entries => {
      if (!live) return;
      setCheckpoints(entries);
      setId(prev => entries.some(c => c.checkpoint_id === prev) ? prev :
        entries.find(c => c.origin === "fixture")?.checkpoint_id ?? entries[0]?.checkpoint_id ?? "");
    }).catch(error => {if (live) setError(inspectErrorText(error));});
    return () => {live = false;};
  }, [retry]);
  useEffect(() => {
    if (!id || !checkpoints?.some(checkpoint => checkpoint.checkpoint_id === id)) return;
    const controller = new AbortController();
    setLoaded(undefined); setError("");
    loadNavigator(id, controller.signal).then(data => {
      if (!controller.signal.aborted) setLoaded({id, data});
    }).catch(error => {if (!controller.signal.aborted) setError(inspectErrorText(error));});
    return () => controller.abort();
  }, [id, retry, checkpoints]);
  const active = checkpoints?.find(c => c.checkpoint_id === id);
  const data = loaded?.id === id ? loaded.data : undefined;
  return <div className="arch-nav">
    <header className="context-bar an-context">
      <div className="context-title"><span>Workspace</span><b>/</b><strong title={active?.display_name}>{active?.display_name ?? "Architecture"}</strong></div>
      <div className="an-context-tools">
        {checkpoints && checkpoints.length > 0 && <label>Checkpoint <select aria-label="Architecture checkpoint" value={id}
          onChange={e => {saveLocation(e.target.value); setId(e.target.value);}}>
          {checkpoints.map(c => <option key={c.checkpoint_id} value={c.checkpoint_id}>{c.display_name}</option>)}</select></label>}
        <button type="button" onClick={() => openTensor(id, undefined, true)}>Import…</button>
      </div>
    </header>
    <main id="work" tabIndex={-1}>
      {error ? <p className="an-inline-alert" role="alert">{error} <button onClick={() => {setError(""); setRetry(r => r + 1);}}>Retry</button></p> :
        data ? <GraphWorkspace key={id + (data.summary?.graph.ir_sha256 ?? "")} id={id} data={data} openTensor={openTensor}
          reference={active?.origin === "fixture"} onAttached={() => setRetry(r => r + 1)}/> :
        checkpoints?.length === 0 ? <p className="an-empty">Import a checkpoint to inspect its architecture.</p> :
          <p className="an-empty" role="status">Loading architecture…</p>}
    </main>
  </div>;
}
