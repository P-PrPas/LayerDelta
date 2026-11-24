import {lazy, Suspense, useCallback, useEffect, useState} from "react";
import {createRoot} from "react-dom/client";
import {Catalog, Change, Comparison, connect, Device, get, Job, Page, post, Recipe, Sample, saveJson, waitFor} from "./api";
import {Results} from "./Results";
import {ChannelFields, Explorer} from "./Explorer";
import "./style.css";
import "./workspace.css";
const TensorBrowser = lazy(() => import("./TensorBrowser").then(module => ({default: module.TensorBrowser})));

const ArchitectureNavigator = lazy(() => import("./ArchitectureNavigator").then(module => ({default: module.ArchitectureNavigator})));

const SessionTraceViewer = lazy(() => import("./SessionTraceViewer").then(module => ({default: module.SessionTraceViewer})));

const screens = ["Tensors", "Architecture", "Sessions", "Model explorer", "Experiment editor", "Run comparison", "History & artifacts", "Final evaluation"] as const;
type Screen = typeof screens[number];
type View = {run: Job; comparison?: Comparison; report?: unknown};

function Workbench() {
  function locationScreen(): Screen {
    const value = new URLSearchParams(location.search).get("screen");
    return value === "architecture" ? "Architecture" : value === "tensors" ? "Tensors" : value === "sessions" ? "Sessions" : "Model explorer";
  }
  const [screen, setScreen] = useState<Screen>(locationScreen);
  useEffect(() => {
    const sync = () => setScreen(locationScreen());
    window.addEventListener("popstate", sync);
    return () => window.removeEventListener("popstate", sync);
  }, []);
  const [menuOpen, setMenuOpen] = useState(false);
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
  const [checkpoint, setCheckpoint] = useState("");
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
    catch (e) {setError((e as Error).message);}
    finally {setBusy(""); setJob(undefined);}
  }
  async function refresh() {
    const page = await get<Page<Job>>("/runs?limit=25"); setJobs(page.items); setCursor(page.next_cursor);
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
  useEffect(() => {
    task("Opening workspace", async () => {
      await connect();
      const data = await loadCatalog();
      setTarget(data.targets[0].target_id); setCheckpoint(data.checkpoints[0].checkpoint_id);
      setSource(data.demo.example_pairs[0].source_sample_id); await refresh();
    });
  }, []);
  useEffect(() => {
    let alive = true;
    setSamples([]);
    if (source) get<Page<Sample>>("/samples?source_sample_id=" + encodeURIComponent(source))
      .then(p => {if (alive) setSamples(p.items);}).catch(e => {if (alive) setError(e.message);});
    return () => {alive = false;};
  }, [source]);
  useEffect(() => {confirmExport(false); setDownload("");}, [view?.run.run_id]);
  useEffect(() => {confirmHoldout(false);}, [recipe?.recipe_hash, device, screen]);

  function navigate(next: Screen) {
    if (menuOpen) requestAnimationFrame(() => document.getElementById("work")?.focus());
    setMenuOpen(false);
    const url = new URL(location.href);
    if (next === "Architecture" || next === "Tensors" || next === "Sessions") {
      const value = next.toLowerCase();
      if (url.searchParams.get("screen") !== value) {
        for (const field of ["sel", "scope", "u", "tensor", "import", "slice", "session", "seq", "compare"]) url.searchParams.delete(field);
        url.searchParams.set("screen", value);
        history.pushState(null, "", url);
      }
    } else if (url.searchParams.has("screen")) {
      url.search = "";
      history.pushState(null, "", url);
    }
    setScreen(next); setError(""); setMessage("");
    if (next === "History & artifacts") task("Loading history", refresh);
    if (next === "Final evaluation") task("Loading holdout audit", async () => {
      const events = await get<{items:typeof audit}>("/holdout-audit");
      setAudit(events.items);
    });
  }
  function openTensor(id: string, tensor?: string, importing = false, slice?: string, compare?: string) {
    const url = new URL(location.href);
    url.search = new URLSearchParams({screen: "tensors", ...(id ? {ckpt: id} : {}), ...(tensor ? {tensor} : {}), ...(importing ? {import: "1"} : {}), ...(slice ? {slice} : {}), ...(compare ? {compare} : {})}).toString();
    history.pushState(null, "", url);
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
    setView({run, comparison, report}); setScreen("Run comparison");
  }
  async function calculate(saved?: Recipe, baseline = false) {
    const chosen = saved ?? (baseline ? undefined : await post<Recipe>("/recipes", {hypothesis,intervention:spec,checkpoint_id:checkpoint}));
    if (chosen) setRecipe(chosen);
    const job = await post<Job>("/runs", chosen ? {kind:"intervention",recipe_id:chosen.recipe_id,device}
      : {kind:"baseline",checkpoint_id:checkpoint,device}, true);
    await waitFor(job, "/runs", setJob); await seeRun(job); await refresh();
  }
  async function inspect(row: Change) {
    if (view?.run.kind === "final" || view?.run.split === "holdout") {
      setMessage("The explorer displays exploration samples. Holdout predictions are available in this report."); return;
    }
    setSource(row.source_sample_id); setCondition(row.condition); setScreen("Model explorer");
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
      await loadCatalog(); setCheckpoint(added.checkpoint_id); setScreen("Model explorer");
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
  const names = ["tensors", "architecture", "sessions", "model", "edit", "compare", "history", "final"];
  return <div className={"workbench-shell" + (screen === "Tensors" || screen === "Architecture" || screen === "Sessions" ? " workbench-tensors" : "")}><a className="skip" href="#work">Skip to workbench</a>
    <aside className="workspace-rail" onKeyDown={event => {
      if (event.key === "Escape" && menuOpen) {
        setMenuOpen(false);
        document.getElementById("workspace-menu-toggle")?.focus();
      }
    }}>
      <div className="wordmark"><i className="stack-mark" aria-hidden="true"><b/><b/><b/></i><strong>LayerDelta</strong></div>
      <button type="button" className="nav-toggle" id="workspace-menu-toggle" aria-controls="workspace-nav"
        aria-expanded={menuOpen} onClick={() => setMenuOpen(!menuOpen)}>{menuOpen ? "Close menu" : "Menu"}</button>
      <nav id="workspace-nav" data-open={menuOpen} aria-label="Workbench"><ol>{screens.map((s,i) => <li key={s}><button aria-current={screen === s ? "page" : undefined}
        disabled={active && s !== "Tensors" && screen !== "Tensors"} onClick={() => navigate(s)}><i className={"nav-sign " + names[i]} aria-hidden="true"/>{s}
          {s === "Sessions" && tracePending > 0 && <span className="rail-trace-count" aria-label={`${tracePending} pending proposals`}>{tracePending}</span>}
          {s === "Sessions" && traceLiveUntil > Date.now() && <i className="rail-trace-live" role="img" aria-label="Agent active in the last minute"/>}</button></li>)}</ol></nav>
      <div className="rail-identity"><i aria-hidden="true"/><div><strong>Local workspace</strong>
        <small>{screen === "Tensors" ? "Tensor browser" : screen === "Architecture" ? "Architecture navigator" : screen === "Sessions" ? "Session trace viewer" : catalog?.dataset_revision_id ?? "Connecting…"}</small></div></div>
    </aside>
    <div className="workspace-canvas">
      {screen === "Tensors" ? <Suspense fallback={<main id="work" className="work-area" tabIndex={-1} role="status">Opening tensor browser…</main>}>
        <TensorBrowser/>
      </Suspense> : screen === "Architecture" ? <Suspense fallback={<main id="work" className="work-area" tabIndex={-1} role="status">Opening architecture…</main>}>
        <ArchitectureNavigator openTensor={openTensor}/>
      </Suspense> : screen === "Sessions" ? <Suspense fallback={<main id="work" className="work-area" tabIndex={-1} role="status">Opening sessions…</main>}>
        <SessionTraceViewer activity={traceActivity} pending={setTracePending} actions={{
          tensor: (id, selector, compare) => openTensor(id, selector?.tensor, false, selector ? JSON.stringify(selector) : undefined, compare),
          architecture: (id, group, unit, kind = "group") => {
            const url = new URL(location.href);
            url.search = new URLSearchParams({screen: "architecture", ckpt: id, sel: kind + ":" + group, ...(unit !== undefined ? {u: String(unit)} : {})}).toString();
            history.pushState(null, "", url);
            navigate("Architecture");
          },
          async run(id) {
            await connect();
            await seeRun({run_id: id});
            const url = new URL(location.href); url.search = "";
            history.pushState(null, "", url);
          },
        }}/>
      </Suspense> : <>
      <header className="context-bar">
        <div className="context-title"><details className="workspace-info"><summary>Workspace</summary><dl className="record-list">
          <dt>Dataset revision</dt><dd>{catalog?.dataset_revision_id ?? "Connecting…"}</dd>
          <dt>Checkpoint SHA-256</dt><dd>{checkpoint || "Loading…"}</dd></dl></details>
          <b aria-hidden="true">/</b><strong>{catalog?.dataset_revision_id ?? "Connecting…"}</strong></div>
        {catalog && <fieldset className="context-inputs" disabled={active}><legend className="visually-hidden">Workspace settings</legend>
          <label>Checkpoint<select value={checkpoint} title={checkpoint} onChange={e => setCheckpoint(e.target.value)}>
            {catalog.checkpoints.map(x => <option key={x.checkpoint_id} value={x.checkpoint_id} title={x.sha256}>{x.sha256.slice(0,12)}</option>)}</select></label>
          <label>Requested device<select value={device} onChange={e => setDevice(e.target.value as Device)}>
            {catalog.devices.map(d => <option key={d.id} value={d.id} disabled={!d.available}>{d.id.toUpperCase()}{!d.available && " · unavailable"}</option>)}</select></label>
        </fieldset>}
      </header>
      <main id="work" className="work-area" tabIndex={-1}>
        <div className="page-title"><h1>{screen}</h1>
          {isExplorer && catalog && <span className="page-context">Exploration split</span>}</div>
        <div role="status" className="status">{busy && <>{busy}… {job && <span>{job.run_id} · {job.status}</span>}</>}{message}</div>
        {error && <p role="alert">{error}</p>}
        {job && ["queued","running"].includes(job.status) && <button onClick={() => post("/runs/"+job.run_id+"/cancel",{})
          .catch(e => setError(e.message))}>Cancel job</button>}
        {catalog && <fieldset disabled={active} className="screen-body"><legend className="visually-hidden">{screen} controls</legend>
          {isExplorer && <Explorer catalog={catalog} target={target} channel={channel} choose={chooseTarget}
            device={device} checkpoint={checkpoint} sample={sample} baseline={() => task("Computing baseline", () => calculate(undefined,true))}
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
              <button className="primary" data-test="intervene" disabled={!validChannel||!validScale||!hypothesis.trim()}
                onClick={() => task("Evaluating intervention",() => calculate())}>Run intervention</button>
            </section>
            <section className="edit-summary"><h2>Execution plan</h2><dl className="record-list">
              <dt>Target</dt><dd>{target?.target_id}</dd><dt>Operation</dt><dd>{operation==="zero" ? "Set to zero" : "Multiply by " + (scale || "—")}</dd>
              <dt>Slice</dt><dd>{target?.mode==="weight" ? "Kernel output" : "Post-ReLU output"} · channel {validChannel ? channel : "—"}</dd>
              <dt>Runtime</dt><dd>{device.toUpperCase()} · exploration</dd></dl>
              <p>{target?.mode==="weight" ? "Only this output kernel changes. Its bias and other channels retain their parent values." :
                "The hook changes this feature map during inference. The parent weights stay fixed."}</p>
            </section></div>}
          {screen==="Run comparison" && (view ? <section className="result-panel">
            {view.comparison ? <Results comparison={view.comparison} runId={view.run.run_id} key={view.run.run_id} inspect={inspect}/> :
              <><h2>{view.run.kind} · {view.run.status}</h2><p className="run-identifier">{view.run.run_id}</p>
                {view.run.error && <p role="alert">{typeof view.run.error === "string" ? view.run.error : view.run.error.message}</p>}
                {view.report !== undefined && <details><summary>Baseline report</summary><pre>{JSON.stringify(view.report,null,2)}</pre></details>}</>}
            <div className="actions">{view.run.status==="succeeded" && <button onClick={() => task("Exporting report",async () =>
              saveJson(runId+".json",await get("/runs/"+runId+"/report")))}>Save run report</button>}
              {recipe && <button onClick={() => navigate("Final evaluation")}>Review frozen recipe</button>}</div>
          </section> : <section className="empty-panel"><h2>No run selected</h2><p>Run an intervention or open a saved result to compare predictions.</p>
            <div className="actions"><button className="primary" onClick={() => navigate("Experiment editor")}>Open editor</button>
            <button onClick={() => navigate("History & artifacts")}>View run history</button></div></section>)}
          {screen==="History & artifacts" && <div className="history-grid"><section className="history-panel"><h2>Run history</h2>
            {jobs.length ? <div className="table-scroll"><table><thead><tr><th>Run</th><th>Kind</th><th>Split</th><th>Status</th></tr></thead>
              <tbody>{jobs.map(j => <tr key={j.run_id}><td><button className="link run-link" title={j.run_id} aria-label={"Open run " + j.run_id}
                onClick={() => task("Opening run",() => seeRun(j))}>{j.run_id}</button></td><td>{j.kind}</td><td>{j.split}</td><td>{j.status}</td></tr>)}</tbody></table></div>
              : <p>No runs in this workspace.</p>}
            {cursor && <button onClick={() => task("Loading history",async () => {const p=await get<Page<Job>>("/runs?limit=25&cursor="+encodeURIComponent(cursor));
              setJobs([...jobs,...p.items]); setCursor(p.next_cursor);})}>Load more runs</button>}</section>
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
          {screen==="Final evaluation" && <div className="final-grid"><section className="final-panel"><h2>Holdout confirmation</h2>
            {recipe ? <><p>{recipe.hypothesis}</p><dl className="record-list"><dt>Frozen recipe</dt><dd>{recipe.recipe_hash}</dd>
              <dt>Checkpoint SHA-256</dt><dd>{recipe.dependencies.checkpoint_sha256}</dd><dt>Intervention</dt>
              <dd>{recipe.intervention.target_id} · channel {recipe.intervention.channel_index} · {recipe.intervention.operation} {recipe.intervention.scale}</dd></dl>
              <label className="check"><input type="checkbox" checked={holdoutConfirm} onChange={e=>confirmHoldout(e.target.checked)}/>
                Confirm this frozen recipe for an audited holdout evaluation on {device.toUpperCase()}.</label>
              <button className="primary" disabled={!holdoutConfirm} onClick={() => task("Evaluating holdout",finalRun)}>Run final evaluation</button></> :
              <><p>Select a completed intervention or import a frozen recipe first.</p><button onClick={() => navigate("History & artifacts")}>View run history</button></>}
          </section><section className="audit-panel"><h2>Holdout audit</h2>
            {audit.length ? <ul>{audit.map(event => <li key={event.run_id}><time dateTime={event.created_at}>{new Date(event.created_at).toLocaleString()}</time>
              <button className="link" title={event.run_id} onClick={() => task("Opening holdout run",async () => seeRun(await get<Job>("/runs/"+event.run_id)))}>{event.run_id}</button></li>)}</ul> :
              <p>No holdout evaluations recorded.</p>}</section></div>}
        </fieldset>}
      </main></>}
    </div>
  </div>;
}

createRoot(document.getElementById("root")!).render(<Workbench/>);
