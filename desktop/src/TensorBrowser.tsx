import {CSSProperties, useEffect, useMemo, useRef, useState} from "react";
import {TensorInspector} from "./TensorInspector";
import {createFixtureSource, fixtureTensor} from "./tensorFixture";
import {Checkpoint, Description, DiffStatus, DiffTensor, Envelope, globPattern, httpTensorSource, ImportProgress, InspectError, ModelDiff, Tensor,
  TensorDataSource} from "./tensorSource";
import {compact, count, filterTensors, predicate, TensorTree, tokens} from "./tensorTree";
import {metric} from "./tensorValues";
import "./tensorBrowser.css";

const parameters = new URLSearchParams(location.search);
const defaultSource = parameters.get("tensor-source") === "http" ? httpTensorSource :
  createFixtureSource(parameters.has("empty"));
const shortId = (id: string) => id.replace(/^sha256:/, "").slice(0, 12);
const label = (c: Checkpoint) => c.display_name ?? c.filename;
const importStages = {header: "Reading header", extents: "Validating extents", table: "Building tensor table"} as const;

function failureTitle(code: string) {
  if (["file_size", "header_size"].includes(code)) return "File is larger than the workspace limit";
  if (["header_json", "duplicate_field", "tensor_record", "metadata", "dtype_unknown", "shape"].includes(code))
    return "Header is not valid safetensors";
  if (["extent", "extent_mismatch", "extent_overlap_or_gap", "data_size"].includes(code)) return "Tensor data does not match the header";
  if (code === "duplicate_tensor") return "Two tensors share a name";
  if (["index_invalid", "index_mismatch", "shard_missing"].includes(code)) return "Sharded checkpoint is incomplete";
  if (code === "file_changed") return "File changed on disk while reading";
  if (code === "pickle_format" || code === "unsupported_format") return "Unsupported checkpoint format";
  return "Checkpoint could not be opened";
}

function ImportCard({source, checkpoints, descriptions, onOpen, onImported, copy}:
  {source: TensorDataSource; checkpoints: Checkpoint[]; descriptions: Map<string, Description>;
    onOpen: (id: string) => void; onImported: (c: Checkpoint) => void; copy: (value: unknown) => void}) {
  const [path, setPath] = useState("");
  const [file, setFile] = useState<File>();
  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState<ImportProgress | null>(null);
  const [error, setError] = useState<InspectError>();
  const picker = useRef<HTMLInputElement>(null);
  const field = useRef<HTMLInputElement>(null);

  function choose(file?: File) {
    if (!file || busy) return;
    setPath(file.name); setFile(file); setError(undefined);
  }
  async function submit() {
    if (busy) return;
    const chosen = path.trim();
    if (/\.(pt|pth|bin)$/i.test(chosen)) {
      setError(new InspectError("pickle_format", "Pickle files can run code when loaded. Convert with layerdelta convert yourself, then import the .safetensors."));
      return;
    }
    if (!/\.safetensors(?:\.index\.json)?$/i.test(chosen)) {
      setError(new InspectError("unsupported_format", "Choose a .safetensors file or a .safetensors.index.json shard index."));
      return;
    }
    setBusy(true); setError(undefined); setStage(null);
    try {onImported(await source.importCheckpoint({path: chosen, file}, setStage));}
    catch (e) {setError(e instanceof InspectError ? e : new InspectError("import_failed", (e as Error).message));}
    finally {setBusy(false);}
  }

  useEffect(() => {
    function drop(event: DragEvent) {
      if (!event.dataTransfer?.types.includes("Files")) return;
      event.preventDefault();
      if (event.type === "drop") choose(event.dataTransfer.files[0]);
    }
    window.addEventListener("dragover", drop);
    window.addEventListener("drop", drop);
    return () => {window.removeEventListener("dragover", drop); window.removeEventListener("drop", drop);};
  }, [busy]);

  const activeStage = stage ? Object.keys(importStages).indexOf(stage.stage) : -1;
  return <div className="tb-import-wrap">
    <section className="tb-import">
      <h1>Open a checkpoint</h1>
      <form onSubmit={e => {e.preventDefault(); void submit();}}>
        <label htmlFor="tb-import-path">Path to a safetensors checkpoint</label>
        <div className="tb-import-field">
          <input id="tb-import-path" ref={field} autoComplete="off" spellCheck={false} value={path} disabled={busy}
            placeholder="D:\models\checkpoint.safetensors" onChange={e => {setPath(e.target.value); setFile(undefined); setError(undefined);}}/>
          <input ref={picker} type="file" hidden tabIndex={-1} accept=".safetensors,.safetensors.index.json,.json"
            onChange={e => {choose(e.target.files?.[0]); e.target.value = "";}}/>
          <button type="button" disabled={busy} onClick={() => picker.current?.click()}>Choose file…</button>
          <button className="primary" disabled={busy || !path.trim()}>Import</button>
        </div>
        <p>LayerDelta reads only the header and the bytes you inspect. Nothing in the file is executed.</p>
        {source.fixture && <p className="tb-meta">Fixture mode uses sample tensors for imported files.</p>}
      </form>
      {busy && !stage && <p role="status">Importing checkpoint…</p>}
      {busy && stage && <ol role="status" aria-label="Import progress" className="tb-stages">
        {Object.entries(importStages).map(([key, name], i) => {
          const current = stage?.stage === key ? stage : null;
          return <li key={key}><span aria-hidden="true">{i < activeStage || current?.done === current?.total && current ? "✓" : i === activeStage ? "◌" : "○"}</span>
            <span>{name}</span><span className="tb-meta">{current ? count(current.done) + " / " + count(current.total) : i < activeStage ? "Done" : "Pending"}</span>
            {current && <progress max={current.total} value={current.done} aria-label={name}/>}</li>;
        })}
      </ol>}
      {error && <div role="alert" className="tb-import-error">
        <strong>{failureTitle(error.code)}</strong>
        <p>{error.message}</p>
        {error.missing?.length ? <ul>{error.missing.map(name => <li key={name}><code>{name}</code></li>)}</ul> : null}
        <code>{error.code}</code>
        {error.code === "file_changed" ? <button onClick={() => void submit()}>Re-import</button>
          : <button onClick={() => copy({code: error.code, message: error.message, field: error.field, missing: error.missing})}>Copy details</button>}
      </div>}
    </section>
    {checkpoints.length > 0 && <section className="tb-recent"><h2>Workspace checkpoints</h2>
      {checkpoints.map(c => {
        const meta = descriptions.get(c.checkpoint_id)?.checkpoint;
        return <button key={c.checkpoint_id} onClick={() => onOpen(c.checkpoint_id)}>
          <span>{label(c)}</span><span className="tb-meta">{shortId(c.checkpoint_id)}
            {meta && " · " + count(meta.tensor_count) + " tensors · " + compact(meta.file_bytes, true)}
            {c.created_at && " · " + c.created_at}</span>
        </button>;
      })}
    </section>}
  </div>;
}

export function TensorBrowser({source = defaultSource}: {source?: TensorDataSource}) {
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>();
  const [id, setId] = useState("");
  const [other, setOther] = useState("");
  const lastOther = useRef("");
  const [importing, setImporting] = useState(false);
  const [descriptions, setDescriptions] = useState(new Map<string, Description>());
  const [diff, setDiff] = useState<{key: string; result: ModelDiff}>();
  const [diffError, setDiffError] = useState("");
  const [loadError, setLoadError] = useState("");
  const [retry, setRetry] = useState(0);
  const [filter, setFilter] = useState("");
  const [debounced, setDebounced] = useState("");
  const [statsMatches, setStatsMatches] = useState<Set<string> | null>(null);
  const [statBusy, setStatBusy] = useState(false);
  const [statError, setStatError] = useState("");
  const statGeneration = useRef(0);
  const [selected, setSelected] = useState(fixtureTensor);
  const [shortcuts, setShortcuts] = useState(() => localStorage.getItem("layerdelta.tensor-shortcuts") !== "off");
  const [sheet, setSheet] = useState(false);
  const [treeFocus, setTreeFocus] = useState(0);
  const [inspectorFocus, setInspectorFocus] = useState(0);
  const [treeWidth, setTreeWidth] = useState(430);
  const [drawer, setDrawer] = useState(false);
  const [message, setMessage] = useState("");
  const search = useRef<HTMLInputElement>(null);
  const dialog = useRef<HTMLDialogElement>(null);
  const active = checkpoints?.find(c => c.checkpoint_id === id);
  const description = descriptions.get(id);
  const parent = descriptions.get(other);
  const wholeDiffKey = JSON.stringify([other, id, null]);
  const nameGlob = diff?.key === wholeDiffKey ? undefined : predicate(debounced).name_glob;
  const diffKey = JSON.stringify([other, id, nameGlob ?? null]);
  const compared = diff?.key === diffKey ? diff.result : undefined;
  const all = useMemo(() => {
    const tensors = new Map(description?.tensors.map(t => [t.name, t]));
    if (other) parent?.tensors.forEach(t => {if (!tensors.has(t.name)) tensors.set(t.name, t);});
    return tensors;
  }, [description, other, parent]);
  const diffs = useMemo(() => other ? new Map(compared?.tensors.map(t => [t.name, t])) : null, [compared, other]);
  const filtered = useMemo(() => filterTensors([...all.values()], debounced,
    compared ? diffs : null, statsMatches), [all, debounced, compared, diffs, statsMatches]);
  const maxBytes = useMemo(() => [...all.values()].reduce((max, t) => Math.max(max, t.nbytes), 0), [all]);
  const selectedTensor = all.get(selected);
  const currentTensor = description?.tensors.find(t => t.name === selected);
  const otherTensor = parent?.tensors.find(t => t.name === selected);
  const alignment: DiffStatus | undefined = !other || !parent || !selectedTensor ? undefined :
    !currentTensor ? "only_a" : !otherTensor ? "only_b" : currentTensor.dtype !== otherTensor.dtype ?
      "dtype_mismatch" : currentTensor.shape.join() !== otherTensor.shape.join() ? "shape_mismatch" : undefined;
  const status = alignment ?? (compared && (!nameGlob || globPattern(nameGlob).test(selected)) ?
    diffs?.get(selected)?.status ?? "same" : undefined);
  const summary = compared?.summary;

  function compareWith(next: string) {
    setOther(next);
    if (next) lastOther.current = next;
    const retained = tokens(filter).filter(t => !t.startsWith("changed:"));
    if (next) retained.push("changed:yes");
    setFilter(retained.join(" ")); setDebounced(retained.join(" "));
  }

  function openCheckpoint(next: string, list = checkpoints ?? []) {
    setId(next); setImporting(false); setLoadError("");
    setSelected(source.fixture ? fixtureTensor : "");
    compareWith(list.find(c => c.checkpoint_id === next)?.parent_id ?? "");
  }

  useEffect(() => {
    let live = true;
    source.listCheckpoints().then(items => {
      if (!live) return;
      setCheckpoints(items);
      if (items.length) openCheckpoint(items[0].checkpoint_id, items);
    }).catch(e => {if (live) {setCheckpoints([]); setLoadError((e as Error).message);}});
    return () => {live = false;};
  }, [source, retry]);

  useEffect(() => {
    let live = true;
    for (const active of new Set([id, other].filter(Boolean))) {
      if (descriptions.has(active)) continue;
      void (async () => {
        try {
          let cursor: string | null = null;
          let first: Description | undefined;
          const tensors: Tensor[] = [];
          do {
            const response = await source.describe(active, {limit: 20000, cursor});
            if (!live) return;
            first ??= response.result;
            tensors.push(...response.result.tensors);
            cursor = response.cursor;
          } while (cursor);
          if (live && first) setDescriptions(prev => new Map(prev).set(active, {...first, tensors}));
        } catch (e) {if (live) setLoadError((e as Error).message);}
      })();
    }
    return () => {live = false;};
  }, [id, other, source, retry]);

  useEffect(() => {
    let live = true;
    setDiffError("");
    if (!other || !id) return;
    void (async () => {
      try {
        let cursor: string | null = null;
        let first: ModelDiff | undefined;
        const tensors: DiffTensor[] = [];
        do {
          const response: Envelope<ModelDiff> = await source.diff(other, id, {selector: null, name_glob: nameGlob, limit: 1000, cursor});
          if (!live) return;
          first ??= response.result;
          tensors.push(...response.result.tensors);
          cursor = response.cursor;
        } while (cursor);
        if (live && first) setDiff({key: diffKey, result: {...first, tensors}});
      } catch (e) {if (live) setDiffError((e as Error).message);}
    })();
    return () => {live = false;};
  }, [source, diffKey, other, id, retry]);

  useEffect(() => {
    const timer = setTimeout(() => setDebounced(filter), 120);
    return () => clearTimeout(timer);
  }, [filter]);
  useEffect(() => {
    if (description && !selectedTensor) setSelected(description.tensors[0]?.name ?? "");
  }, [description, selectedTensor]);
  useEffect(() => {
    statGeneration.current++;
    setStatsMatches(null); setStatBusy(false); setStatError("");
  }, [id, filter]);
  useEffect(() => {
    if (sheet) dialog.current?.showModal();
    else dialog.current?.close();
  }, [sheet]);

  async function copy(value: unknown) {
    try {
      await navigator.clipboard.writeText(typeof value === "string" ? value : JSON.stringify(value, null, 2));
      setMessage("Copied to clipboard");
    } catch {setMessage("Clipboard access is unavailable.");}
  }
  async function searchStats() {
    const query = predicate(filter);
    if (!query.name_glob && selected) query.name_glob = selected.includes(".") ? selected.slice(0, selected.lastIndexOf(".") + 1) + "*" : selected;
    const generation = ++statGeneration.current;
    setStatBusy(true); setStatError("");
    try {
      let cursor: string | null = null;
      const matches = new Set<string>();
      do {
        const page = await source.query(id, {...query, where: {stat: "l2", axis: 0, op: "lt", value: .01}, max_scan_elems: 2 ** 20, limit: 500, cursor});
        if (generation !== statGeneration.current) return;
        page.result.matches.forEach(m => matches.add(m.name));
        cursor = page.cursor;
      } while (cursor);
      setStatsMatches(matches);
    } catch (e) {if (generation === statGeneration.current) setStatError((e as Error).message);}
    finally {if (generation === statGeneration.current) setStatBusy(false);}
  }

  function toggleToken(token: string) {
    const parts = tokens(filter);
    setFilter((parts.includes(token) ? parts.filter(t => t !== token) : [...parts, token]).join(" "));
  }

  const alternatives = checkpoints?.filter(c => c.checkpoint_id !== id) ?? [];
  function toggleCompare() {
    compareWith(other ? "" : alternatives.find(c => c.checkpoint_id === lastOther.current)?.checkpoint_id ??
      alternatives[0]?.checkpoint_id ?? "");
  }
  const clear = () => {setFilter(""); setDebounced(""); setStatsMatches(null); setTreeFocus(x => x + 1);};
  const hasFilters = tokens(debounced).some(t => !/^(identical|constant):/.test(t)) || !!statsMatches;
  const showImport = importing || !id && !!checkpoints;
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape" && e.target === search.current) {e.preventDefault(); clear(); return;}
      if (!shortcuts || sheet || e.ctrlKey || e.altKey || e.metaKey ||
        (e.target instanceof HTMLElement && e.target.closest("input, select, textarea, [contenteditable]"))) return;
      if (e.key === "/" && !showImport) {e.preventDefault(); setDrawer(true); requestAnimationFrame(() => search.current?.focus());}
      if (e.key === "d" && alternatives.length && !showImport) {e.preventDefault(); toggleCompare();}
      if (e.key === "?") {e.preventDefault(); setSheet(true);}
      if (e.key === "Escape" && filter) {e.preventDefault(); clear();}
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });
  return <div className="tensor-browser">
    <header className="context-bar tb-context">
      <div className="context-title">Workspace <b aria-hidden="true">/</b> <strong title={active && label(active)}>{active ? label(active) : "Tensors"}</strong></div>
      <div className="tb-tools">
        {source.fixture && <span className="tb-meta">Fixture data</span>}
        {checkpoints?.length ? <>
          <label>Checkpoint <select aria-label="Checkpoint" value={id} onChange={e => openCheckpoint(e.target.value)}>
            {checkpoints.map(c => <option key={c.checkpoint_id} value={c.checkpoint_id}>{shortId(c.checkpoint_id)} · {label(c)}</option>)}
          </select></label>
          <label>Compare with <select aria-label="Compare with" value={other} disabled={!alternatives.length || showImport}
            onChange={e => compareWith(e.target.value)}>
            <option value="">none</option>{alternatives.map(c => <option key={c.checkpoint_id} value={c.checkpoint_id}>
              {shortId(c.checkpoint_id)} · {active?.parent_id === c.checkpoint_id ? "parent" : label(c)}</option>)}
          </select></label>
        </> : null}
        {id && <button onClick={() => setImporting(!importing)}>{importing ? "Back to tensors" : "Import…"}</button>}
        <button onClick={() => setSheet(true)} aria-label="Keyboard shortcuts">?</button>
      </div>
    </header>
    <main id="work" tabIndex={-1} className="tb-work">
      <h1 className="visually-hidden">Tensors</h1>
      <span className="visually-hidden" role="status">{message}</span>
      {loadError && <p role="alert" className="tb-alert">{loadError} <button onClick={() => {
        setLoadError(""); setRetry(n => n + 1);
      }}>Retry</button></p>}
      {showImport ? <ImportCard source={source} checkpoints={checkpoints ?? []} descriptions={descriptions} copy={copy}
        onOpen={next => openCheckpoint(next)} onImported={c => {
          const next = [c, ...(checkpoints ?? []).filter(x => x.checkpoint_id !== c.checkpoint_id)];
          setCheckpoints(next); setDescriptions(prev => {const updated = new Map(prev); updated.delete(c.checkpoint_id); return updated;});
          openCheckpoint(c.checkpoint_id, next); setRetry(n => n + 1);
        }}/> :
        !description ? <div className="tb-loading" role="status"><p>{checkpoints ? "Loading tensor table…" : "Opening workspace checkpoints…"}</p>
          {Array.from({length: 12}, (_, i) => <i key={i}/>)}</div> :
        <div className="tb-panes" style={{"--tb-tree-width": treeWidth + "px"} as CSSProperties}>
          <button className="tb-drawer" aria-controls="tb-tree-pane" aria-expanded={drawer} onClick={() => setDrawer(!drawer)}>Tensor tree</button>
          <section className="tb-tree" id="tb-tree-pane" data-drawer={drawer} aria-label="Tensor tree">
            <div className="tb-filter">
              <span aria-hidden="true">⌕</span><input ref={search} aria-label="Filter tensors" spellCheck={false}
                placeholder="Filter by name, glob or dtype:bf16 rank:4" value={filter} onChange={e => setFilter(e.target.value)}/><kbd>/</kbd>
            </div>
            <div className="tb-chips" role="group" aria-label="Facets">
              {["rank:4", "dtype:bf16", "editable:no", "size:>10mb", "identical:yes", "constant:yes", ...(other ? ["changed:yes"] : [])].map(token =>
                <button key={token} className="tb-chip" disabled={/^(identical|constant):/.test(token)}
                  title={/^(identical|constant):/.test(token) ? "Not checked yet" : token === "dtype:bf16" ?
                    count(description.dtypes.BF16 ?? 0) + " BF16 tensors" : undefined}
                  aria-pressed={tokens(filter).includes(token)} onClick={() => toggleToken(token)}>{token}</button>)}
            </div>
            <div className="tb-summary" role="status">
              {hasFilters ? count(filtered.length) + " of " + count(all.size) + " tensors match" :
                count(description.checkpoint.tensor_count) + " tensors · " + compact(description.checkpoint.element_count) +
                " elements · " + compact(description.checkpoint.file_bytes, true)}
              <span title="Identical-bytes check is not available yet">identical bytes: not checked</span>
            </div>
            {other && <div className="tb-compare-banner" role={diffError ? "alert" : "status"}>
              {diffError || (summary ? <>
                <strong title="Changed means bitwise inequality of stored bytes">{count(summary.changed)} of {count(summary.tensors_total)} changed</strong>
                {" · " + count(summary.changed_elems_total) + " values (bitwise) · max |Δ| " + metric(summary.max_abs)}
                {nameGlob && <div className="tb-meta">Name scope {nameGlob}</div>}
                <div className="tb-alignment">
                  <span data-diff="only_a">− {count(summary.only_a_count)} only in parent</span>
                  <span data-diff="only_b">+ {count(summary.only_b_count)} only in child</span>
                  {!!summary.shape_mismatch_count && <span data-diff="shape_mismatch">≠ {count(summary.shape_mismatch_count)} shape changed</span>}
                  {!!summary.dtype_mismatch_count && <span data-diff="dtype_mismatch">≠ {count(summary.dtype_mismatch_count)} dtype changed</span>}
                </div>
                <span>Matched by name only. A renamed tensor shows as one only-in-parent and one only-in-child.</span>
              </> : "Comparing checkpoint values…")}
            </div>}
            <div className="tb-tree-tools">
              <button onClick={() => copy(predicate(filter))} title="Copies the name, dtype, rank, editability and size predicate">Copy as query</button>
              <button disabled={statBusy} onClick={() => statsMatches ? setStatsMatches(null) : void searchStats()}
                title="Search axis 0 for positions with L2 below 0.01. Uses the current filter or selected tensor’s group.">
                {statsMatches ? "Clear L2 search" : "Find low L2"}</button>
            </div>
            {statBusy && <p className="tb-meta" role="status">Searching position statistics…</p>}
            {statError && <p className="tb-alert" role="alert">{statError}</p>}
            {!filtered.length && <div className="tb-tree-empty"><p>No tensors match <code>{debounced}</code>.</p><button onClick={clear}>Clear filter</button></div>}
            <TensorTree tensors={filtered} selected={selected} onSelect={name => {setSelected(name); setDrawer(false);}}
              filter={hasFilters} diffs={diffs} maxBytes={maxBytes} focusInspector={() => setInspectorFocus(x => x + 1)} focusRequest={treeFocus}/>
          </section>
          <div className="tb-splitter" role="separator" aria-label="Tensor tree width" aria-orientation="vertical"
            aria-valuemin={340} aria-valuemax={480} aria-valuenow={treeWidth} tabIndex={0}
            onKeyDown={e => {
              if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
              e.preventDefault(); setTreeWidth(w => Math.min(480, Math.max(340, w + (e.key === "ArrowRight" ? 10 : -10))));
            }} onPointerDown={e => e.currentTarget.setPointerCapture(e.pointerId)}
            onPointerMove={e => {
              if (e.currentTarget.hasPointerCapture(e.pointerId))
                setTreeWidth(Math.min(480, Math.max(340, e.clientX - e.currentTarget.parentElement!.getBoundingClientRect().left)));
            }}/>
          {selectedTensor ? <TensorInspector key={selectedTensor.name + selectedTensor.shape.join(",")} source={source}
            checkpoint={id} other={compared ? other : ""} tensor={selectedTensor} status={status}
            coverage={description.coverage.unknown} shortcuts={shortcuts} copy={copy} focusRequest={inspectorFocus}/>
            : <div className="tb-inspect-empty">Select a tensor to inspect its values.</div>}
        </div>}
    </main>
    <dialog className="tb-shortcuts" ref={dialog} onCancel={() => setSheet(false)} onClose={() => setSheet(false)}>
      <h2>Keyboard shortcuts</h2>
      <dl>{[
        ["/ · Esc", "Focus or clear the filter"], ["↑ ↓ · Home End · PgUp PgDn", "Move through the tensor tree"],
        ["← → · Enter · *", "Collapse, expand, open, or expand siblings"], ["h · v · m", "Heatmap, Values, or Mosaic"],
        ["[ ] · Shift+[ ]", "Change the fixed index or window"], ["Arrow keys · Enter", "Move and pin the value cursor"],
        ["d", "Toggle comparison"], ["Ctrl+C", "Copy selector in the inspector"], ["?", "Open this sheet"],
      ].map(([key, description]) => <div key={key}><dt><kbd>{key}</kbd></dt><dd>{description}</dd></div>)}</dl>
      <label><input type="checkbox" checked={shortcuts} onChange={e => {
        setShortcuts(e.target.checked); localStorage.setItem("layerdelta.tensor-shortcuts", e.target.checked ? "on" : "off");
      }}/> Enable single-key shortcuts</label>
      <button autoFocus onClick={() => setSheet(false)}>Done</button>
    </dialog>
  </div>;
}
