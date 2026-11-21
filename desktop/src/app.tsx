import {useEffect, useState} from "react";
import {createRoot} from "react-dom/client";
import {Catalog, Change, Comparison, connect, Device, get, Job, Page, post, Preview, Recipe, Sample, saveJson, waitFor} from "./api";
import {percent, Pixels, Results} from "./Results";
import "./style.css";

const screens = ["Model explorer", "Experiment editor", "Run comparison", "History & artifacts", "Final evaluation"] as const;
type Screen = typeof screens[number];
type View = {run: Job; comparison?: Comparison; report?: unknown};

function Workbench() {
  const [screen, setScreen] = useState<Screen>(screens[0]);
  const [catalog, setCatalog] = useState<Catalog>();
  const [targetId, setTarget] = useState("");
  const [channel, setChannel] = useState(0);
  const [device, setDevice] = useState<Device>("cpu");
  const [checkpoint, setCheckpoint] = useState("");
  const [source, setSource] = useState("");
  const [browser, setBrowser] = useState<Page<Sample>>();
  const [condition, setCondition] = useState("blurred");
  const [samples, setSamples] = useState<Sample[]>([]);
  const [preview, setPreview] = useState<Preview>();
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

  const target = catalog?.targets.find(t => t.target_id === targetId);
  const sample = samples.find(s => s.condition === condition);
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
    setSamples([]); setPreview(undefined);
    if (source) get<Page<Sample>>("/samples?source_sample_id=" + encodeURIComponent(source))
      .then(p => {if (alive) setSamples(p.items);}).catch(e => {if (alive) setError(e.message);});
    return () => {alive = false;};
  }, [source]);
  useEffect(() => {setPreview(undefined);}, [targetId, channel, condition, device, checkpoint]);
  useEffect(() => {confirmExport(false); setDownload("");}, [view?.run.run_id]);
  useEffect(() => {confirmHoldout(false);}, [recipe?.recipe_hash]);

  function navigate(next: Screen) {
    setScreen(next); setError(""); setMessage("");
    if (next === "History & artifacts") task("Loading history", refresh);
  }
  function chooseTarget(id: string) {setTarget(id); setChannel(0);}
  async function seeRun(original: Job) {
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
  return <><a className="skip" href="#work">Skip to workbench</a>
    <header><strong>LayerDelta</strong><span>Experiment workbench</span></header>
    <nav aria-label="Workbench">{screens.map(s => <button key={s} aria-current={screen===s ? "page":undefined}
      disabled={active} onClick={() => navigate(s)}>{s}</button>)}</nav>
    <main id="work" tabIndex={-1}>
    <h1>{screen}</h1>
    <div role="status" className="status">{busy && <>{busy}… {job && <span>{job.run_id} · {job.status}</span>}</>}
      {message}</div>{error && <p role="alert">{error}</p>}
    {job && ["queued","running"].includes(job.status) && <button onClick={() => post("/runs/"+job.run_id+"/cancel",{})
      .catch(e => setError(e.message))}>Cancel job</button>}
    {catalog && <fieldset disabled={active} className="workspace">
    <legend>Workspace settings</legend><label>Checkpoint<select value={checkpoint} onChange={e => setCheckpoint(e.target.value)}>
      {catalog.checkpoints.map(x => <option key={x.checkpoint_id} value={x.checkpoint_id}>{x.checkpoint_id} · {x.sha256.slice(0,12)}</option>)}</select></label>
    <label>Requested device<select value={device} onChange={e => setDevice(e.target.value as Device)}>
      {catalog.devices.map(d => <option key={d.id} value={d.id} disabled={!d.available}>{d.id.toUpperCase()}{!d.available && " · unavailable"}</option>)}</select></label>
    <span>Dataset · {catalog.dataset_revision_id}</span></fieldset>}
    {catalog && <fieldset disabled={active} className="body">
    {(screen==="Model explorer" || screen==="Experiment editor") && <section><h2>Channel selection</h2>
      <div className="fields"><label>Layer and mode<select value={targetId} onChange={e => chooseTarget(e.target.value)}>{catalog.targets.map(t =>
        <option key={t.target_id} value={t.target_id}>{t.target_id} · {t.mode}</option>)}</select></label>
      <label>Output channel<input type="number" min="0" max={(target?.channel_count ?? 1)-1} step="1"
        value={Number.isNaN(channel) ? "" : channel} onChange={e => setChannel(e.target.valueAsNumber)}/></label></div>
      <p>{target?.channel_count} channels · {target?.parameter_name ?? target?.module_path + " · post-ReLU"}</p>
      {!validChannel && <p role="alert">Choose an integer channel from 0 to {(target?.channel_count ?? 1)-1}.</p>}</section>}
    {screen==="Model explorer" && <section><h2>Exploration samples</h2>
      <p>Blurred {catalog.demo.truth}s predicted as {catalog.demo.baseline_prediction} in the reference baseline.</p>
      <div className="fields"><label>Source image<select value={source} onChange={e => setSource(e.target.value)}>
        {[...new Set([source,...catalog.demo.example_pairs.map(x => x.source_sample_id)])].filter(Boolean).map(s => <option key={s}>{s}</option>)}</select></label>
      <label>Condition<select value={condition} onChange={e => setCondition(e.target.value)}><option>normal</option><option>blurred</option></select></label></div>
      <details><summary>Browse all exploration sources</summary>
        <button onClick={() => task("Loading samples", () => browse())}>Load first 25 sources</button>
        {browser && <><label>Available sources<select value={browser.items.some(s => s.source_sample_id===source) ? source : ""}
          onChange={e => setSource(e.target.value)}><option value="" disabled>Select a source</option>
          {browser.items.map(s => <option value={s.source_sample_id} key={s.sample_id}>{s.source_sample_id} · truth {s.label}</option>)}</select></label>
          {browser.next_cursor && <button onClick={() => task("Loading samples",() => browse(browser.next_cursor))}>Next 25 sources</button>}</>}
      </details>
      <div className="actions"><button data-test="preview" disabled={!sample||!validChannel} onClick={() => task("Computing preview",async () => {
        const job = await post<Job>("/previews",{sample_id:sample!.sample_id,target_id:targetId,channel_index:channel,device,checkpoint_id:checkpoint},true);
        const done = await waitFor<Preview>(job,"/previews",setJob); setPreview(done.result);
      })}>Preview channel</button><button onClick={() => task("Computing baseline", () => calculate(undefined,true))}>Run baseline</button></div>
      {preview && <div data-test="preview-result"><p>{sample?.sample_id} · truth {sample?.label} · {preview.actual_device}</p><div className="pair">
        <div><h3>Input image</h3><Pixels matrix={preview.image} label={"MNIST "+condition+" digit "+sample?.label}/></div>
        <div><h3>Post-ReLU activation</h3><Pixels matrix={preview.activation.values} normalize label={"Activation map for "+preview.hook}/></div></div>
        <table><caption>Channel statistics</caption><thead><tr><th>Tensor</th><th>Shape</th><th>Mean</th><th>Max</th><th>Zeros</th></tr></thead>
          <tbody>{(["activation","kernel"] as const).map(k => <tr key={k}><th>{k}</th><td>{preview[k].shape.join(" × ")}</td>
            <td>{preview[k].mean.toPrecision(5)}</td><td>{preview[k].max.toPrecision(5)}</td><td>{percent(preview[k].zero_fraction)}</td></tr>)}</tbody></table></div>}
      <details><summary>Model architecture</summary><table><thead><tr><th>Layer</th><th>Output shape</th><th>Channel access</th></tr></thead>
      <tbody>{catalog.model.layers.map(layer => <tr key={layer.layer_id}><td>{layer.layer_id}</td>
        <td>{layer.output_shape.join(" × ")}</td><td>{layer.read_only ? "Read only" : "Editable"}</td></tr>)}</tbody></table></details>
    </section>}
    {screen==="Experiment editor" && <section><h2>Intervention</h2><label>Hypothesis<textarea maxLength={4096}
      placeholder="What do you expect this channel edit to change?" value={hypothesis} onChange={e => setHypothesis(e.target.value)}/></label>
      <div className="fields"><label>Operation<select value={operation} onChange={e => setOperation(e.target.value as typeof operation)}>
        <option value="zero">Zero channel</option><option value="scale">Scale channel</option></select></label>
      {operation==="scale" && <label>Scale factor (0–2)<input type="number" min="0" max="2" step="any" value={scale} onChange={e => setScale(e.target.value)}/></label>}</div>
      <p>{target?.mode==="weight" ? "Edit one output kernel and preserve its bias." : "Modify one feature map after ReLU during inference."}</p>
      <button data-test="intervene" disabled={!validChannel||!validScale||!hypothesis.trim()} onClick={() => task("Evaluating intervention",() => calculate())}>Run intervention</button></section>}
    {screen==="Run comparison" && (!view ? <p>Select a run from History or run an intervention.</p> : <section>
      {view.comparison ? <Results comparison={view.comparison} runId={view.run.run_id} key={view.run.run_id} inspect={inspect}/> :
        <><h2>{view.run.kind} · {view.run.status}</h2><p>{view.run.run_id}</p>
        {view.report !== undefined && <details><summary>Baseline report</summary><pre>{JSON.stringify(view.report,null,2)}</pre></details>}</>}
      <div className="actions">{view.run.status==="succeeded" && <button onClick={() => task("Exporting report",async () =>
        saveJson(runId+".json",await get("/runs/"+runId+"/report")))}>Save run report</button>}
      {recipe && <button onClick={() => navigate("Final evaluation")}>Review frozen recipe</button>}</div></section>)}
    {screen==="History & artifacts" && <section><h2>Run history</h2>
      {jobs.length ? <table><thead><tr><th>Run</th><th>Kind</th><th>Status</th></tr></thead><tbody>{jobs.map(j => <tr key={j.run_id}>
        <td><button className="link" onClick={() => task("Opening run",() => seeRun(j))}>{j.run_id}</button></td><td>{j.kind}</td><td>{j.status}</td></tr>)}</tbody></table>
        : <p>No runs in this workspace.</p>}
      {cursor && <button onClick={() => task("Loading history",async () => {const p=await get<Page<Job>>("/runs?limit=25&cursor="+encodeURIComponent(cursor));
        setJobs([...jobs,...p.items]); setCursor(p.next_cursor);})}>Load more runs</button>}
      <h2>Artifacts</h2><div className="fields">
      {(["recipe","checkpoint"] as const).map(kind => <label key={kind}>Import {kind}<input type="file" accept={kind==="recipe"?".json":".safetensors"}
        onChange={e => {const file=e.target.files?.[0]; e.target.value=""; if(file) task("Importing "+kind,() => importFile(file,kind));}}/></label>)}</div>
      {recipe && <><p>Selected recipe · <code>{recipe.recipe_hash}</code></p><div className="actions"><button onClick={() => {
        const {recipe_id,...frozen}=recipe; saveJson(recipe_id+".json",frozen);
      }}>Save recipe</button><button onClick={() => task("Replaying recipe",() => calculate(recipe))}>Replay recipe</button></div></>}
      {view?.run.status==="succeeded" && view.run.recipe_id===recipe?.recipe_id && recipe?.intervention.mode==="weight" &&
        <><h3>Child checkpoint</h3><label className="check"><input type="checkbox" checked={exportConfirm} onChange={e => confirmExport(e.target.checked)}/>
        Export the approved weight slice from {view.run.run_id}.</label><button disabled={!exportConfirm} onClick={() => task("Exporting checkpoint",exportChild)}>Create child checkpoint</button>
        {download && <a download href={download}>Download Safetensors</a>}</>}</section>}
    {screen==="Final evaluation" && <section><h2>Holdout confirmation</h2>{recipe ? <>
      <p>{recipe.hypothesis}</p><dl className="facts"><dt>Frozen recipe</dt><dd>{recipe.recipe_hash}</dd><dt>Checkpoint SHA-256</dt>
      <dd>{recipe.dependencies.checkpoint_sha256}</dd><dt>Intervention</dt><dd>{recipe.intervention.target_id} · channel {recipe.intervention.channel_index} · {recipe.intervention.operation} {recipe.intervention.scale}</dd></dl>
      <label className="check"><input type="checkbox" checked={holdoutConfirm} onChange={e => confirmHoldout(e.target.checked)}/>
      Confirm this frozen recipe for an audited holdout evaluation on {device.toUpperCase()}.</label>
      <button disabled={!holdoutConfirm} onClick={() => task("Evaluating holdout",finalRun)}>Run final evaluation</button>
      </> : <p>Select a completed intervention or import a frozen recipe first.</p>}</section>}
    </fieldset>}</main></>;
}

createRoot(document.getElementById("root")!).render(<Workbench/>);
