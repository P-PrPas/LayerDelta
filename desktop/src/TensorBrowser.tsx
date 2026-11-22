import {CSSProperties, useEffect, useMemo, useRef, useState} from "react";
import {ApiError} from "./api";
import {liveCheckpointSource} from "./checkpointSource";
import {TensorInspector} from "./TensorInspector";
import {createFixtureSource, fixtureIds, fixtureTensor} from "./tensorFixture";
import {Checkpoint, CheckpointImport, CheckpointSource, Description, DiffStatus, DiffTensor, globPattern, ImportProgress,
  InspectError, ModelDiff, ModelDiffReply, Tensor, TensorDataSource} from "./tensorSource";
import {compact, count, filterTensors, predicate, TensorTree, tokens} from "./tensorTree";
import {metric} from "./tensorValues";
import "./tensorBrowser.css";

const parameters = new URLSearchParams(location.search);
const fixtures = createFixtureSource(parameters.has("empty"));
const defaultCatalog = parameters.get("tensor-source") === "fixture" || parameters.has("empty") ? fixtures : liveCheckpointSource;
const shortId = (id: string) => id.replace(/^sha256:/, "").slice(0, 12);
const importStages = {validating: "Validating", copying: "Copying", verifying: "Verifying", publishing: "Publishing"} as const;

function checkpointFacts(checkpoint: Checkpoint, fallback?: Description["checkpoint"]) {
  const bytes = checkpoint.file_bytes ?? fallback?.file_bytes;
  const tensors = checkpoint.tensor_count ?? fallback?.tensor_count;
  return [checkpoint.origin, bytes === undefined ? null : compact(bytes, true),
    tensors === undefined ? null : count(tensors) + (tensors === 1 ? " tensor" : " tensors")].filter(Boolean).join(" · ");
}

function failureTitle(code: string) {
  if (["file_size", "header_size", "total_size", "payload_too_large"].includes(code)) return "File is larger than the workspace limit";
  if (["header_json", "duplicate_field", "tensor_record", "metadata", "dtype_unknown", "shape"].includes(code))
    return "Header is not valid safetensors";
  if (code.startsWith("extent") || code === "data_size") return "Tensor data does not match the header";
  if (code === "duplicate_tensor") return "Two tensors share a name";
  if (code.startsWith("index_") || ["shard_missing", "shard_outside_directory"].includes(code))
    return "Sharded checkpoint is incomplete";
  if (code === "file_changed") return "File changed on disk while reading";
  if (code === "source_not_found") return "Checkpoint file was not found";
  if (code === "invalid_path") return "Enter an absolute file path";
  if (["length_required", "length_mismatch", "invalid_filename"].includes(code)) return "Checkpoint upload could not be read";
  if (["unauthenticated", "forbidden"].includes(code)) return "Open a human desktop session to import";
  if (code === "unsupported_format") return "Unsupported checkpoint format";
  return "Checkpoint could not be opened";
}

function ImportCard({catalog, fixture, checkpoints, descriptions, onOpen, onImported, copy, busy, setBusy}:
  {catalog: CheckpointSource; fixture: boolean; checkpoints: Checkpoint[]; descriptions: Map<string, Description>;
    onOpen: (id: string) => void; onImported: (result: CheckpointImport) => void; copy: (value: unknown) => void;
    busy: boolean; setBusy: (busy: boolean) => void}) {
  const [path, setPath] = useState("");
  const [file, setFile] = useState<File>();
  const [stage, setStage] = useState<ImportProgress | null>(null);
  const [error, setError] = useState<InspectError | ApiError>();
  const picker = useRef<HTMLInputElement>(null);
  const field = useRef<HTMLInputElement>(null);

  function choose(file?: File) {
    if (!file || busy) return;
    setPath(file.name); setFile(file); setError(undefined);
  }
  async function submit() {
    if (busy) return;
    const chosen = file?.name ?? path.trim();
    if (/\.(pt|pth|bin|ckpt|pkl)$/i.test(chosen)) {
      setError(new InspectError("unsupported_format", "Pickle files can run code when loaded. Convert with layerdelta convert yourself, then import the .safetensors."));
      return;
    }
    if (file && chosen.endsWith(".safetensors.index.json")) {
      setFile(undefined);
      setPath("");
      setError(new InspectError("unsupported_format", "Paste the absolute path to the shard index so LayerDelta can read its adjacent shards."));
      return;
    }
    if (!/\.safetensors(?:\.index\.json)?$/.test(chosen)) {
      setError(new InspectError("unsupported_format", "Choose a .safetensors file or paste the path to a .safetensors.index.json shard index."));
      return;
    }
    setBusy(true); setError(undefined); setStage(null);
    try {onImported(await catalog.importCheckpoint({path: chosen, file}, setStage));}
    catch (e) {setError(e instanceof InspectError || e instanceof ApiError ? e : new InspectError("import_failed", (e as Error).message));}
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
  const title = error && failureTitle(error.code);
  const overLimit = title === "File is larger than the workspace limit";
  const needsPath = error && ["invalid_path", "source_not_found", "shard_missing", "shard_outside_directory"].includes(error.code);
  const reExport = title === "Header is not valid safetensors" || title === "Tensor data does not match the header" || error?.code === "duplicate_tensor";
  return <div className="tb-import-wrap">
    <section className="tb-import">
      <h1>Open a checkpoint</h1>
      <form onSubmit={e => {e.preventDefault(); void submit();}}>
        <label htmlFor="tb-import-path">{file ? "File to upload" : "Absolute path to a safetensors checkpoint"}</label>
        <div className="tb-import-field">
          <input id="tb-import-path" ref={field} autoComplete="off" spellCheck={false} value={path} disabled={busy}
            placeholder="D:\models\checkpoint.safetensors" onChange={e => {setPath(e.target.value); setFile(undefined); setError(undefined);}}/>
          <input ref={picker} type="file" hidden tabIndex={-1} accept=".safetensors" aria-label="Checkpoint file"
            onChange={e => {choose(e.target.files?.[0]); e.target.value = "";}}/>
          <button type="button" disabled={busy} onClick={() => picker.current?.click()}>Choose file…</button>
          <button className="primary" disabled={busy || !path.trim()}>{file ? "Upload" : "Import"}</button>
        </div>
        <p>Paste an absolute path or choose or drop a single file to upload. For sharded checkpoints, paste the index path.</p>
        <p>LayerDelta validates the checkpoint without executing code from it.</p>
        {fixture && <p className="tb-meta">Fixture mode uses sample tensors for imported files.</p>}
      </form>
      {busy && !stage && <p className="tb-meta" role="status">Submitting checkpoint…</p>}
      {busy && <ol role="status" aria-label="Import progress" className="tb-stages">
        {Object.entries(importStages).map(([key, name], i) => {
          const current = stage?.stage === key ? stage : null;
          const done = i < activeStage || !!current && current.total > 0 && current.done >= current.total;
          return <li key={key}><span aria-hidden="true">{done ? "✓" : current ? "◌" : "○"}</span>
            <span>{name}</span><span className="tb-meta">{current && current.total > 0 ?
              count(current.done) + " / " + count(current.total) + (current.unit ? " " + current.unit : "") :
              done ? "Done" : current ? "In progress" : "Pending"}</span>
            {current && current.total > 0 && <progress max={current.total} value={current.done} aria-label={name}/>}</li>;
        })}
      </ol>}
      {error && <div role="alert" className="tb-import-error">
        <strong>{title}</strong>
        <p>{error.message}</p>
        {reExport && <p>Re-export the checkpoint as safetensors and try again.</p>}
        {overLimit && <p>{file && <>Selected file is {compact(file.size, true)}. </>}The workspace file cap is set with <code>--max-import-bytes</code>.</p>}
        {error instanceof InspectError && !!error.missing?.length && <ul>{error.missing.map(name => <li key={name}><code>{name}</code></li>)}</ul>}
        <code>{error.code}</code>
        {error.code === "file_changed" ? <button onClick={() => void submit()}>Re-import</button> :
          overLimit && file ? <button onClick={() => picker.current?.click()}>Choose another file…</button> :
          needsPath || overLimit ? <button onClick={() => field.current?.focus()}>Edit path</button> :
          <button onClick={() => copy({code: error.code, message: error.message})}>Copy details</button>}
      </div>}
    </section>
    {!!checkpoints.length && <section className="tb-recent"><h2>Workspace checkpoints</h2>
      {checkpoints.map(entry => <button key={entry.checkpoint_id} disabled={busy} onClick={() => onOpen(entry.checkpoint_id)}>
        <span>{entry.display_name}</span>
        <span className="tb-meta">{shortId(entry.checkpoint_id)} · {checkpointFacts(entry, descriptions.get(entry.checkpoint_id)?.checkpoint)}
          {entry.imported_at && " · " + new Date(entry.imported_at).toLocaleString()}</span>
      </button>)}
    </section>}
  </div>;
}

export function TensorBrowser({source = fixtures, catalog = defaultCatalog}:
  {source?: TensorDataSource; catalog?: CheckpointSource}) {
  const liveCatalog = catalog !== fixtures;
  const sampleInspection = liveCatalog && source.fixture;
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>();
  const [id, setId] = useState("");
  const [other, setOther] = useState("");
  const lastOther = useRef("");
  const [importing, setImporting] = useState(false);
  const [importBusy, setImportBusy] = useState(false);
  const [importNotice, setImportNotice] = useState("");
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
  const inspectId = id && (sampleInspection ? fixtureIds.child : id);
  const inspectOther = other && (sampleInspection ? fixtureIds.parent : other);
  const description = descriptions.get(inspectId);
  const parent = descriptions.get(inspectOther);
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
    setId(next); setImporting(false); setLoadError(""); setImportNotice("");
    setSelected(source.fixture ? fixtureTensor : "");
    const parent = list.find(c => c.checkpoint_id === next)?.parent_id;
    compareWith(parent && list.some(c => c.checkpoint_id === parent) ? parent : "");
  }

  useEffect(() => {
    let live = true;
    catalog.listCheckpoints().then(items => {
      if (!live) return;
      setCheckpoints(items);
      if (items.length) openCheckpoint((items.find(c => c.checkpoint_id === id) ??
        items.find(c => c.origin === "fixture") ?? items[0]).checkpoint_id, items);
    }).catch(e => {if (live) setLoadError((e as Error).message);});
    return () => {live = false;};
  }, [catalog, retry]);

  useEffect(() => {
    if (!id) return;
    let live = true;
    catalog.getCheckpoint(id).then(entry => {
      if (live) setCheckpoints(previous => previous?.map(c => c.checkpoint_id === id ? entry : c));
    }).catch(e => {if (live) setLoadError((e as Error).message);});
    return () => {live = false;};
  }, [catalog, id, retry]);

  useEffect(() => {
    let live = true;
    for (const active of new Set([inspectId, inspectOther].filter(Boolean))) {
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
  }, [inspectId, inspectOther, source, retry]);

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
          const response: ModelDiffReply = await source.diff(inspectOther, inspectId, {selector: null, name_glob: nameGlob, limit: 1000, cursor});
          if (!live) return;
          first ??= response.result;
          tensors.push(...response.result.tensors);
          cursor = response.cursor;
        } while (cursor);
        if (live && first) setDiff({key: diffKey, result: {...first, tensors}});
      } catch (e) {if (live) setDiffError((e as Error).message);}
    })();
    return () => {live = false;};
  }, [source, diffKey, inspectOther, inspectId, retry]);

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
        const page = await source.query(inspectId, {...query, where: {stat: "l2", axis: 0, op: "lt", value: .01}, max_scan_elems: 2 ** 20, limit: 500, cursor});
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
      <div className="tb-catalog-context">
        <div className="context-title">Workspace <b aria-hidden="true">/</b>
          <strong title={active?.display_name}>{active?.display_name ?? "Tensors"}</strong>
        </div>
        {liveCatalog && active && <span className="tb-meta tb-catalog-facts">{checkpointFacts(active)}</span>}
      </div>
      <div className="tb-tools">
        {source.fixture && <span className="tb-meta">{sampleInspection ? "Inspection uses sample tensors" : "Fixture data"}</span>}
        {checkpoints?.length ? <>
          <label>Checkpoint <select aria-label="Checkpoint" value={id} disabled={importBusy} onChange={e => openCheckpoint(e.target.value)}>
            {checkpoints.map(c => <option key={c.checkpoint_id} value={c.checkpoint_id}>{c.display_name} · {checkpointFacts(c)} · {shortId(c.checkpoint_id)}</option>)}
          </select></label>
          <label>Compare with <select aria-label="Compare with" value={other} disabled={!alternatives.length || showImport}
            onChange={e => compareWith(e.target.value)}>
            <option value="">none</option>{alternatives.map(c => <option key={c.checkpoint_id} value={c.checkpoint_id}>
              {shortId(c.checkpoint_id)} · {active?.parent_id === c.checkpoint_id ? "parent · " + c.display_name : c.display_name}</option>)}
          </select></label>
        </> : null}
        {id && <button disabled={importBusy} onClick={() => {setImporting(!importing); setImportNotice("");}}>{importing ? "Back to tensors" : "Import…"}</button>}
        <button onClick={() => setSheet(true)} aria-label="Keyboard shortcuts">?</button>
      </div>
    </header>
    <main id="work" tabIndex={-1} className="tb-work">
      <h1 className="visually-hidden">Tensors</h1>
      <span className="visually-hidden" role="status">{message}</span>
      {importNotice && <p className="tb-import-notice" role="status">{importNotice}</p>}
      {loadError && <p role="alert" className="tb-alert">{loadError} <button onClick={() => {
        setLoadError(""); setRetry(n => n + 1);
      }}>Retry</button></p>}
      {showImport ? <ImportCard catalog={catalog} fixture={!liveCatalog} checkpoints={checkpoints ?? []} descriptions={descriptions}
        copy={copy} busy={importBusy} setBusy={setImportBusy}
        onOpen={next => openCheckpoint(next)} onImported={({checkpoint, existing}) => {
          const entries = checkpoints ?? [];
          const next = entries.some(c => c.checkpoint_id === checkpoint.checkpoint_id) ?
            entries.map(c => c.checkpoint_id === checkpoint.checkpoint_id ? checkpoint : c) : [checkpoint, ...entries];
          setCheckpoints(next);
          openCheckpoint(checkpoint.checkpoint_id, next);
          setImportNotice((existing ? "Already in this workspace · " : "Checkpoint imported · ") + checkpoint.display_name);
        }}/> :
        !description ? !loadError && <div className="tb-loading" role="status"><p>{checkpoints ? "Loading tensor table…" : "Opening workspace checkpoints…"}</p>
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
          {selectedTensor ? <TensorInspector key={JSON.stringify([id, status === "only_a" ? other : null, selectedTensor.name, selectedTensor.shape])} source={source}
            checkpoint={inspectId} other={compared ? inspectOther : ""} tensor={selectedTensor} status={status}
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
