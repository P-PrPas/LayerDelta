import {lazy, Suspense, useCallback, useEffect, useRef, useState} from "react";
import {createRoot} from "react-dom/client";
import {Catalog, Change, Comparison, ApiError, connect, Device, displayError, get, Job, Page, post, Recipe, Sample, saveJson, waitFor} from "./api";
import {Results} from "./Results";
import {ThemeProvider} from "./theme";
import {WorkspaceFrame, type Place} from "./WorkspaceFrame";
import {liveCheckpointSource} from "./checkpointSource";
import type {Checkpoint, Envelope} from "./tensorSource";
import {dimensions, type Budget, type Dimension} from "./traceSource";
import {ChannelFields, Explorer} from "./Explorer";
import "./tokens.css";
import "./data-ramps.css";
import "./style.css";
import "./controls.css";
import "./workspace.css";
const TensorBrowser = lazy(() => import("./TensorBrowser").then(module => ({default: module.TensorBrowser})));

const ArchitectureNavigator = lazy(() => import("./ArchitectureNavigator").then(module => ({default: module.ArchitectureNavigator})));

const SessionTraceViewer = lazy(() => import("./SessionTraceViewer").then(module => ({default: module.SessionTraceViewer})));

const runViews = {
  explore: "Model explorer", edit: "Experiment editor", compare: "Run comparison", history: "History & artifacts",
} as const;
type RunView = keyof typeof runViews;
type Screen = Place | typeof runViews[RunView];
const otherScreens = {architecture: "Architecture", tensors: "Tensors", sessions: "Sessions", final: "Final evaluation"} as const;
const legacy = {
  "model-explorer": "explore", "experiment-editor": "edit", "run-comparison": "compare", "history-artifacts": "history",
} as const;
function runView(screen: Screen) {
  return (Object.keys(runViews) as RunView[]).find(view => runViews[view] === screen);
}
function locationScreen(): Screen {
  const query = new URLSearchParams(location.search);
  const screen = query.get("screen");
  const view = screen === "runs" ? query.get("view") : Object.entries(legacy).find(([key]) => key === screen)?.[1];
  return Object.entries(otherScreens).find(([key]) => key === screen)?.[1] ??
    Object.entries(runViews).find(([key]) => key === view)?.[1] ?? runViews.explore;
}
type View = {run: Job; comparison?: Comparison; report?: unknown};

function Workbench() {
  const [screen, setScreen] = useState<Screen>(locationScreen);
  useEffect(() => {
    const sync = () => {
      setScreen(locationScreen());
      setCheckpoint(new URLSearchParams(location.search).get("ckpt") ?? "");
    };
    window.addEventListener("popstate", sync);
    return () => window.removeEventListener("popstate", sync);
  }, []);
  const [traceLiveUntil, setTraceLiveUntil] = useState(0);
  const [tracePending, setTracePending] = useState(0);
  const traceActivity = useCallback((remaining: number) => {
    const next = remaining > 0 ? Date.now() + remaining : 0;
    setTraceLiveUntil(previous => !!previous === !!next && Math.abs(previous - next) < 500 ? previous : next);
  }, []);
  useEffect(() => {
    if (!traceLiveUntil) return;
    const timer = window.setTimeout(() => setTraceLiveUntil(0), Math.max(0, traceLiveUntil - Date.now()));
    return () => window.clearTimeout(timer);
  }, [traceLiveUntil]);
  const [catalog, setCatalog] = useState<Catalog>();
  const [targetId, setTarget] = useState("");
  const [channel, setChannel] = useState(0);
  const [device, setDevice] = useState<Device>("cpu");
  const [checkpoint, setCheckpoint] = useState(() => new URLSearchParams(location.search).get("ckpt") ?? "");
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>();
  const [connected, setConnected] = useState(false);
  const [sessionState, setSessionState] = useState<"connecting" | "ready" | "unavailable">("connecting");
  const [sessionRequestId, setSessionRequestId] = useState<string | null>(null);
  const checkpointRequest = useRef(0);
  const [checkpointError, setCheckpointError] = useState("");
  const [importBusy, setImportBusy] = useState(false);
  const [attached, setAttached] = useState("");
  const [traceBudget, setTraceBudget] = useState<{session: string; budget: Partial<Record<Dimension, Budget>>} | null>(null);
  const [source, setSource] = useState("");
  const [browser, setBrowser] = useState<Page<Sample>>();
  const [condition, setCondition] = useState("blurred");
  const [samples, setSamples] = useState<Sample[]>([]);
  const [operation, setOperation] = useState<"zero"|"scale">("zero");
  const [scale, setScale] = useState("0.5");
  const [hypothesis, setHypothesis] = useState("");
  const [recipe, setRecipe] = useState<Recipe>();
  const [view, setView] = useState<View>();
  const [jobs, setJobs] = useState<Job[]>([]);
  const [cursor, setCursor] = useState<string|null>(null);
  const [previewCancel, setPreviewCancel] = useState<(() => void) | null>(null);
  const observePreview = useCallback((cancel: (() => void) | null) => setPreviewCancel(() => cancel), []);
  const [busy, setBusy] = useState("");
  const [job, setJob] = useState<Job>();
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [holdoutConfirm, confirmHoldout] = useState(false);
  const [exportConfirm, confirmExport] = useState(false);
  const [download, setDownload] = useState("");
  const [audit, setAudit] = useState<{run_id:string;created_at:string;recipe_hash:string}[]>([]);

  const target = catalog?.targets.find(t => t.target_id === targetId);
  const sample = samples.find(s => s.condition === condition && s.source_sample_id === source);
  const validChannel = !!target && Number.isInteger(channel) && channel >= 0 && channel < target.channel_count;
  const validScale = operation === "zero" || (scale.trim() !== "" && Number(scale) >= 0 && Number(scale) <= 2);
  const spec = target && {target_id: targetId, mode:target.mode, channel_index:channel,
    operation, ...(operation === "scale" ? {scale:Number(scale)} : {})};

  async function task(label: string, action: () => Promise<void>) {
    setError(""); setMessage(""); setBusy(label);
    try {await action();}
    catch (e) {setError(displayError(e, "Could not complete this action."));}
    finally {setBusy(""); setJob(undefined);}
  }
  async function refresh() {
    const page = await get<Page<Job>>("/runs?limit=25"); setJobs(page.items.filter(run => run.kind !== "final_evaluation")); setCursor(page.next_cursor);
  }
  async function loadCatalog() {
    const data = await get<Catalog>("/catalog");
    setCatalog(data);
    return data;
  }
  async function browse(cursor?: string|null) {
    const page = await get<Page<Sample>>("/samples?condition=normal&limit=25"
      + (cursor ? "&cursor=" + encodeURIComponent(cursor) : ""));
    setBrowser(page);
  }
  async function loadCheckpoints() {
    const request = ++checkpointRequest.current;
    setCheckpointError("");
    try {
      const entries = await liveCheckpointSource.listCheckpoints();
      if (request !== checkpointRequest.current) return;
      setCheckpoints(entries);
      const requested = new URLSearchParams(location.search).get("ckpt") ?? "";
      if (!entries.some(item => item.checkpoint_id === requested))
        chooseCheckpoint((entries.find(item => item.origin === "fixture") ?? entries[0])?.checkpoint_id ?? "");
      setConnected(true);
    } catch (error) {
      if (request !== checkpointRequest.current) return;
      setConnected(false);
      if (error instanceof ApiError && error.status === 401) {
        setSessionRequestId(error.requestId);
        setSessionState("unavailable");
      } else {
        setCheckpointError("Could not load workspace checkpoints.");
      }
    }
  }

  const chooseCheckpoint = useCallback((id: string) => {
    const url = new URL(location.href);
    if ((url.searchParams.get("ckpt") ?? "") !== id) {
      if (id) url.searchParams.set("ckpt", id);
      else url.searchParams.delete("ckpt");
      for (const key of ["tensor", "slice", "compare", "sel", "scope", "u"]) url.searchParams.delete(key);
      history.replaceState(null, "", url);
    }
    setCheckpoint(id);
  }, []);
  async function openWorkspace() {
    setSessionState("connecting");
    try {
      await connect();
    } catch (error) {
      setSessionRequestId(error instanceof ApiError ? error.requestId : null);
      setSessionState("unavailable");
      return;
    }
    setSessionState("ready");
    void loadCheckpoints();
    void task("Opening workspace", async () => {
      const data = await loadCatalog();
      setTarget(data.targets[0]?.target_id ?? "");
      setSource(data.demo.example_pairs[0]?.source_sample_id ?? "");
      await refresh();
    });
  }
  useEffect(() => {void openWorkspace();}, []);
  useEffect(() => {
    if (!connected || !checkpoint) return;
    const controller = new AbortController();
    void get<Envelope<{graph: unknown}>>("/checkpoints/" + encodeURIComponent(checkpoint) + "/graph?limit=1", controller.signal)
      .then(data => {if (!controller.signal.aborted) setAttached(data.result.graph ? checkpoint : "");})
      .catch(() => {if (!controller.signal.aborted) setAttached("");});
    return () => controller.abort();
  }, [connected, checkpoint]);
  useEffect(() => {
    let alive = true;
    setSamples([]);
    if (source) get<Page<Sample>>("/samples?source_sample_id=" + encodeURIComponent(source))
      .then(p => {if (alive) setSamples(p.items);}).catch(e => {if (alive) setError(displayError(e, "Could not load the source images."));});
    return () => {alive = false;};
  }, [source]);
  useEffect(() => {confirmExport(false); setDownload("");}, [view?.run.run_id]);
  useEffect(() => {confirmHoldout(false);}, [recipe?.recipe_hash, device, screen, checkpoint]);

  useEffect(() => {
    if (screen !== "Final evaluation" || sessionState !== "ready") return;
    const controller = new AbortController();
    void get<{items: typeof audit}>("/holdout-audit", controller.signal)
      .then(events => {if (!controller.signal.aborted) setAudit(events.items);})
      .catch(error => {if (!controller.signal.aborted) setError(displayError(error, "Could not load the holdout audit."));});
    return () => controller.abort();
  }, [screen, sessionState, view?.run.run_id]);

  function showScreen(next: Screen) {
    const url = new URL(location.href);
    const view = runView(next);
    const value = view ? "runs" : next === "Final evaluation" ? "final" : next.toLowerCase();
    if (url.searchParams.get("screen") !== value || url.searchParams.get("view") !== (view ?? null)) {
      for (const field of ["sel", "scope", "u", "tensor", "import", "slice", "session", "seq", "compare", "evaluation", "view"])
        url.searchParams.delete(field);
      url.searchParams.set("screen", value);
      if (view) url.searchParams.set("view", view);
      history.pushState(null, "", url);
    }
    setScreen(next);
  }
  function navigate(next: Screen) {
    showScreen(next); setError(""); setMessage("");
    if (next === "Architecture" || next === "Tensors") void loadCheckpoints();
    if (next === "History & artifacts") task("Loading history", refresh);

  }
  function openTensor(id: string, tensor?: string, importing = false, slice?: string, compare?: string) {
    const url = new URL(location.href);
    url.search = new URLSearchParams({screen: "tensors", ...(id ? {ckpt: id} : {}), ...(tensor ? {tensor} : {}), ...(importing ? {import: "1"} : {}), ...(slice ? {slice} : {}), ...(compare ? {compare} : {})}).toString();
    history.pushState(null, "", url);
    if (id) setCheckpoint(id);
    navigate("Tensors");
  }
  function chooseTarget(id: string) {
    if (id !== targetId) {setTarget(id); setChannel(0);}
  }
  async function seeRun(original: Pick<Job, "run_id">) {
    const run = await get<Job>("/runs/" + original.run_id);
    let comparison: Comparison|undefined;
    let report: unknown;
    if (run.recipe_id) setRecipe(await get<Recipe>("/recipes/" + run.recipe_id));
    else setRecipe(undefined);
    if (run.status === "succeeded") {
      if (run.kind === "baseline") report = await get("/runs/" + run.run_id + "/report");
      else comparison = await get<Comparison>("/runs/" + run.run_id + "/comparison");
    }
    setView({run, comparison, report}); showScreen(run.kind === "final_evaluation" ? "Final evaluation" : "Run comparison");
  }
  async function calculate(saved?: Recipe, baseline = false) {
    const chosen = saved ?? (baseline ? undefined : await post<Recipe>("/recipes", {hypothesis,intervention:spec,checkpoint_id:checkpoint}));
    if (chosen) setRecipe(chosen);
    const job = await post<Job>("/runs", chosen ? {kind:"intervention",recipe_id:chosen.recipe_id,device}
      : {kind:"baseline",checkpoint_id:checkpoint,device}, true);
    await waitFor(job, "/runs", setJob); await seeRun(job); await refresh();
  }
  async function inspect(row: Change) {
    if (view?.run.split === "holdout") {
      setMessage("The explorer displays exploration samples. Holdout predictions are available in this report."); return;
    }
    setSource(row.source_sample_id); setCondition(row.condition); showScreen("Model explorer");
  }
  async function finalRun() {
    if (!recipe || !holdoutConfirm) return;
    const job = await post<Job>("/final-evaluations", {recipe_id:recipe.recipe_id,recipe_hash:recipe.recipe_hash,
      device,confirmed:true}, true);
    await waitFor(job,"/runs",setJob); await seeRun(job); await refresh();
  }
  async function exportChild() {
    if (!view || !exportConfirm) return;
    const job = await post<Job>("/checkpoint-exports",{source_run_id:view.run.run_id,confirmed:true},true);
    const done = await waitFor<{download_url:string}>(job,"/checkpoint-exports",setJob);
    setDownload(done.result.download_url);
  }
  async function importFile(file: File, kind: "checkpoint"|"recipe") {
    if (kind === "checkpoint") {
      const form = new FormData(); form.append("file", file);
      const added = await post<{checkpoint_id:string}>("/checkpoints",form);
      await loadCatalog(); await loadCheckpoints(); chooseCheckpoint(added.checkpoint_id); showScreen("Model explorer");
      setMessage("Checkpoint loaded.");
    } else {
      const frozen = await post<Recipe>("/recipes/import", JSON.parse(await file.text()));
      setRecipe(frozen); setView(undefined); setMessage("Recipe verified and ready to replay.");
    }
  }
  const runId = view?.run.run_id;
  const active = !!busy;
  const channels = <ChannelFields targets={catalog?.targets ?? []} target={target} channel={channel}
    onTarget={chooseTarget} onChannel={setChannel}/>;
  const isExplorer = screen === "Model explorer";
  const resultPanel = view && <section className="result-panel">
            {view.comparison ? <Results comparison={view.comparison} runId={view.run.run_id} key={view.run.run_id} inspect={inspect}/> :
              <><h2>{view.run.kind} · {view.run.status}</h2><p className="run-identifier">{view.run.run_id}</p>
                {view.run.status === "failed" && <p className="ui-banner" data-tone="neg" role="alert">The run failed.</p>}
                {view.report !== undefined && <details><summary>Baseline report</summary><pre>{JSON.stringify(view.report,null,2)}</pre></details>}</>}
            <div className="actions">{view.run.status==="succeeded" && <button onClick={() => task("Exporting report",async () =>
              saveJson(runId+".json",await get("/runs/"+runId+"/report")))}>Save run report</button>}
              {recipe && view.run.kind !== "final_evaluation" && <button onClick={() => navigate("Final evaluation")}>Review frozen recipe</button>}</div>
          </section>;
  const viewName = runView(screen);
  const place: Place = viewName ? "Runs" : screen as Place;
  const activeCheckpoint = checkpoints?.find(item => item.checkpoint_id === checkpoint);
  const supportsRuns = !!checkpoint && activeCheckpoint?.adapter_compatible !== false;
  const traceFixture = place === "Sessions" && (import.meta.env.MODE === "trace-fixture" || import.meta.env.DEV) &&
    new URLSearchParams(location.search).get("trace") !== "live";
  const finiteBudget = dimensions.flatMap(dimension => {
    const usage = traceBudget?.budget[dimension];
    return usage?.limit == null ? [] : [dimension.replaceAll("_", " ") + " " + usage.used.toLocaleString() + "/" + usage.limit.toLocaleString()];
  });
  const goToBudget = () => {
    if (!traceBudget) return;
    const url = new URL(location.href);
    url.search = new URLSearchParams({screen: "sessions", ...(checkpoint ? {ckpt: checkpoint} : {}), session: traceBudget.session}).toString();
    if (place === "Sessions") document.getElementById("trace-budgets")?.scrollIntoView({block: "nearest"});
    else {
      history.pushState(null, "", url);
      navigate("Sessions");
    }
  };
  return <WorkspaceFrame place={place} navigate={next => navigate(next === "Runs" ? "Model explorer" : next)}
    checkpoints={checkpoints} checkpoint={checkpoint} pickCheckpoint={chooseCheckpoint} checkpointLocked={active || importBusy}
    graphAttached={!!checkpoint && attached === checkpoint} connected={connected} dataset={catalog?.dataset_revision_id}
    live={traceLiveUntil > Date.now()} pending={tracePending} navLocked={active}
    device={<label className="frame-device"><span aria-hidden="true">Device</span>
      <select aria-label="Requested device" className="frame-device-select" value={device} disabled={active || !catalog}
        onChange={event => setDevice(event.target.value as Device)}>
        {(catalog?.devices ?? [{id: "cpu", available: true}]).map(d =>
          <option key={d.id} value={d.id} disabled={!d.available}>{d.id.toUpperCase()}{!d.available && " · unavailable"}</option>)}
      </select></label>}
    job={<><span className="frame-job-text">{busy ? busy + "…" : importBusy ? "Importing checkpoint…" : previewCancel ? "Computing preview…" : "No job running"}{job && " " + job.status}</span>
      {previewCancel && !busy && <button onClick={previewCancel}>Cancel preview</button>}
      {job && ["queued","running"].includes(job.status) && <button onClick={() => post("/runs/"+job.run_id+"/cancel",{})
        .catch(e => setError(displayError(e, "Could not cancel the job.")))}>Cancel job</button>}</>}
    budget={finiteBudget.length > 0 && <button className="frame-budget" onClick={goToBudget}
      title={finiteBudget.join(" · ")}>Agent budget {finiteBudget.join(" · ")}</button>}>
      {sessionState !== "ready" && !traceFixture ? <main id="work" tabIndex={-1}>
        {sessionState === "connecting" ? <p className="ui-banner" role="status">Opening workspace…</p> :
          <div className="ui-banner" data-tone="neg" role="alert">
            <span>Open LayerDelta from its desktop launcher to start a session.</span>
            <button onClick={() => void openWorkspace()}>Retry</button>
            {sessionRequestId && <small>Request <code>{sessionRequestId}</code></small>}
          </div>}
      </main> : sessionState === "ready" && checkpointError && !traceFixture ? <main id="work" tabIndex={-1}>
        <div className="ui-banner" data-tone="neg" role="alert">{checkpointError} <button onClick={loadCheckpoints}>Retry</button></div>
      </main> : screen === "Tensors" ? <Suspense fallback={<main id="work" className="work-area" tabIndex={-1} role="status">Opening tensor browser…</main>}>
        <TensorBrowser checkpoints={checkpoints} checkpoint={checkpoint} onCheckpoint={chooseCheckpoint}
          importBusy={importBusy} onImportBusy={setImportBusy} onImported={record => {
            checkpointRequest.current++;
            setCheckpointError("");
            setCheckpoints(entries => entries?.some(entry => entry.checkpoint_id === record.checkpoint_id) ?
              entries.map(entry => entry.checkpoint_id === record.checkpoint_id ? record : entry) :
              [record, ...(entries ?? [])]);
          }}/>
      </Suspense> : screen === "Architecture" ? <Suspense fallback={<main id="work" className="work-area" tabIndex={-1} role="status">Opening architecture…</main>}>
        <ArchitectureNavigator openTensor={openTensor} checkpoint={checkpoint} checkpoints={checkpoints}
          onAttached={() => setAttached(checkpoint)}/>
      </Suspense> : screen === "Sessions" ? <Suspense fallback={<main id="work" className="work-area" tabIndex={-1} role="status">Opening sessions…</main>}>
        <SessionTraceViewer activity={traceActivity} pending={setTracePending} observeBudget={setTraceBudget} actions={{
          tensor: (id, selector, compare) => openTensor(id, selector?.tensor, false, selector ? JSON.stringify(selector) : undefined, compare),
          architecture: (id, group, unit, kind = "group") => {
            const url = new URL(location.href);
            url.search = new URLSearchParams({screen: "architecture", ckpt: id, sel: kind + ":" + group, ...(unit !== undefined ? {u: String(unit)} : {})}).toString();
            history.pushState(null, "", url);
            setCheckpoint(id);
            navigate("Architecture");
          },
          async run(id) {
            await connect();
            await seeRun({run_id: id});
          },
        }}/>
      </Suspense> : <>
      <header className="runs-toolbar">
        <h1>{place}</h1>
        {viewName && <nav className="ui-segments" aria-label="Runs views">{(Object.keys(runViews) as RunView[]).map(view =>
          <button type="button" key={view} disabled={active} aria-current={view === viewName ? "page" : undefined}
            onClick={() => navigate(runViews[view])}>{view[0].toUpperCase() + view.slice(1)}</button>)}</nav>}
        {isExplorer && catalog && <span className="page-context">Exploration split</span>}
        {catalog && isExplorer && supportsRuns && <button className="primary" disabled={active || !sample || !validChannel}
          onClick={() => task("Computing baseline", () => calculate(undefined, true))}>Run baseline</button>}
        {catalog && supportsRuns && screen === "Experiment editor" && <button className="primary" data-test="intervene"
          disabled={active || !validChannel || !validScale || !hypothesis.trim()}
          onClick={() => task("Evaluating intervention", () => calculate())}>Run intervention</button>}
        {catalog && screen === "Final evaluation" && recipe && <button className="primary" disabled={active || !holdoutConfirm}
          onClick={() => task("Evaluating holdout", finalRun)}>Run final evaluation</button>}
      </header>
      <main id="work" className="work-area" tabIndex={-1}>
        <div role="status" className="status">{message}</div>
        {error && <p className="ui-banner" data-tone="neg" role="alert">{error}</p>}
        {catalog && viewName && !supportsRuns && checkpoints &&
          <p>Choose a checkpoint compatible with the reference CNN to use Runs.</p>}
        {catalog && (!viewName || supportsRuns || viewName === "history" || viewName === "compare") && <fieldset disabled={active} className="screen-body"><legend className="visually-hidden">{screen} controls</legend>
          {isExplorer && <Explorer catalog={catalog} target={target} channel={channel} choose={chooseTarget}
            device={device} checkpoint={checkpoint} sample={sample} observePreview={observePreview}
            controls={<>
              <label>Source image<select value={source} onChange={e => setSource(e.target.value)}>
                {[...new Set([source,...catalog.demo.example_pairs.map(x => x.source_sample_id)])].filter(Boolean).map(s => <option key={s}>{s}</option>)}</select></label>
              <label>Condition<select value={condition} onChange={e => setCondition(e.target.value)}>
                <option value="normal">Normal</option><option value="blurred">Blurred</option></select></label>
              <details className="source-browser"><summary>Browse exploration sources</summary>
                <button onClick={() => task("Loading samples", () => browse())}>Load first 25 sources</button>
                {browser && <><label>Available sources<select value={browser.items.some(s => s.source_sample_id===source) ? source : ""}
                  onChange={e => setSource(e.target.value)}><option value="" disabled>Select a source</option>
                  {browser.items.map(s => <option value={s.source_sample_id} key={s.sample_id}>{s.source_sample_id} · truth {s.label}</option>)}</select></label>
                  {browser.next_cursor && <button onClick={() => task("Loading samples",() => browse(browser.next_cursor))}>Next 25 sources</button>}</>}
              </details>
              {channels}
            </>}/>}
          {screen==="Experiment editor" && <div className="editor-grid">
            <section className="editor-panel"><h2>Intervention</h2>{channels}
              <div className="field-row"><label>Operation<select value={operation} onChange={e => setOperation(e.target.value as typeof operation)}>
                <option value="zero">Zero channel</option><option value="scale">Scale channel</option></select></label>
                {operation==="scale" && <label>Scale factor (0–2)<input type="number" min="0" max="2" step="any" value={scale} onChange={e=>setScale(e.target.value)}/></label>}</div>
              <label>Hypothesis<textarea maxLength={4096} placeholder="What change do you expect on blurred images?"
                value={hypothesis} onChange={e=>setHypothesis(e.target.value)}/></label>
            </section>
            <section className="edit-summary"><h2>Execution plan</h2><dl className="record-list">
              <dt>Target</dt><dd>{target?.target_id}</dd><dt>Operation</dt><dd>{operation==="zero" ? "Set to zero" : "Multiply by " + (scale || "—")}</dd>
              <dt>Slice</dt><dd>{target?.mode==="weight" ? "Kernel output" : "Post-ReLU output"} · channel {validChannel ? channel : "—"}</dd>
              <dt>Runtime</dt><dd>{device.toUpperCase()} · exploration</dd></dl>
              <p>{target?.mode==="weight" ? "Only this output kernel changes. Its bias and other channels retain their parent values." :
                "The hook changes this feature map during inference. The parent weights stay fixed."}</p>
            </section></div>}
          {screen==="Run comparison" && (view && view.run.kind !== "final_evaluation" ? resultPanel : <section className="empty-panel ui-empty"><h2>No run selected</h2><p>Run an intervention or open a saved result to compare predictions.</p>
            <div className="actions"><button className="primary" onClick={() => navigate("Experiment editor")}>Open editor</button>
            <button onClick={() => navigate("History & artifacts")}>View run history</button></div></section>)}
          {screen==="History & artifacts" && <div className="history-grid"><section className="history-panel"><h2>Run history</h2>
            {jobs.length ? <div className="table-scroll"><table><thead><tr><th>Run</th><th>Kind</th><th>Split</th><th>Status</th></tr></thead>
              <tbody>{jobs.map(j => <tr key={j.run_id}><td><button className="link run-link" title={j.run_id} aria-label={"Open run " + j.run_id}
                onClick={() => task("Opening run",() => seeRun(j))}>{j.run_id}</button></td><td>{j.kind}</td><td>{j.split}</td><td>{j.status}</td></tr>)}</tbody></table></div>
              : <p>No runs in this workspace.</p>}
            {cursor && <button onClick={() => task("Loading history",async () => {const p=await get<Page<Job>>("/runs?limit=25&cursor="+encodeURIComponent(cursor));
              setJobs([...jobs,...p.items.filter(run => run.kind !== "final_evaluation")]); setCursor(p.next_cursor);})}>Load more runs</button>}</section>
            <section className="artifact-panel"><h2>Artifacts</h2><div className="field-row">
              {(["recipe","checkpoint"] as const).map(kind => <label key={kind}>Import {kind}<input type="file" accept={kind==="recipe"?".json":".safetensors"}
                onChange={e => {const file=e.target.files?.[0];e.target.value="";if(file) task("Importing "+kind,()=>importFile(file,kind));}}/></label>)}</div>
              {recipe && <div className="selected-recipe"><h3>Selected recipe</h3><code>{recipe.recipe_hash}</code><div className="actions"><button onClick={() => {
                const {recipe_id,...frozen}=recipe; saveJson(recipe_id+".json",frozen);
              }}>Save recipe</button><button className="primary" onClick={() => task("Replaying recipe",() => calculate(recipe))}>Replay recipe</button></div></div>}
              {view?.run.status==="succeeded" && view.run.recipe_id===recipe?.recipe_id && recipe?.intervention.mode==="weight" &&
                <div className="selected-recipe"><h3>Child checkpoint</h3><label className="check"><input type="checkbox" checked={exportConfirm} onChange={e=>confirmExport(e.target.checked)}/>
                  Export the approved weight slice from {view.run.run_id}.</label>
                  <button disabled={!exportConfirm} onClick={() => task("Exporting checkpoint",exportChild)}>Create child checkpoint</button>
                  {download && <a download href={download}>Download Safetensors</a>}</div>}</section></div>}
          {screen==="Final evaluation" && <>{view?.run.kind === "final_evaluation" && resultPanel}<div className="final-grid"><section className="final-panel"><h2>Holdout confirmation</h2>
            {recipe ? <><p>{recipe.hypothesis}</p><dl className="record-list"><dt>Frozen recipe</dt><dd>{recipe.recipe_hash}</dd>
              <dt>Checkpoint SHA-256</dt><dd>{recipe.dependencies.checkpoint_sha256}</dd><dt>Intervention</dt>
              <dd>{recipe.intervention.target_id} · channel {recipe.intervention.channel_index} · {recipe.intervention.operation} {recipe.intervention.scale}</dd></dl>
              <label className="check"><input type="checkbox" checked={holdoutConfirm} onChange={e=>confirmHoldout(e.target.checked)}/>
                Confirm this frozen recipe for an audited holdout evaluation on {device.toUpperCase()}.</label>
            </> :
              <><p>Select a completed intervention or import a frozen recipe first.</p><button onClick={() => navigate("History & artifacts")}>View run history</button></>}
          </section><section className="audit-panel"><h2>Holdout audit</h2>
            {audit.length ? <ul>{audit.map(event => <li key={event.run_id}><time dateTime={event.created_at}>{new Date(event.created_at).toLocaleString()}</time>
              <button className="link" title={event.run_id} onClick={() => task("Opening holdout run",async () => seeRun(await get<Job>("/runs/"+event.run_id)))}>{event.run_id}</button></li>)}</ul> :
              <p>No holdout evaluations recorded.</p>}</section></div></>}
        </fieldset>}
      </main></>}
    </WorkspaceFrame>;
}

createRoot(document.getElementById("root")!).render(<ThemeProvider><Workbench/></ThemeProvider>);
