import {useState, type ReactNode} from "react";
import {count} from "./tensorTree";
import {failed, isApply, selectionKey, shortId, statusMeaning, toolOf} from "./traceModel";
import type {ResolvedSelection, SessionRow, TraceEvent, TraceRecipe} from "./traceSource";
import {timeText} from "./TraceTimeline";

export type TraceActions = {
  tensor: (id: string, selector?: Extract<ResolvedSelection["selector"], {tensor: string}>, compare?: string) => void;
  architecture: (id: string, group: string, unit?: number, kind?: "node" | "group") => void;
  run: (id: string) => Promise<void>;
};
type Evidence = {
  names: Map<string, string>; resolutions: Map<string, ResolvedSelection>; recipes: Map<string, TraceRecipe>;
  actions: TraceActions; jump: (seq: number) => void;
};
const idFields = new Set(["session", "request_id", "checkpoint_id", "other_checkpoint_id", "selection_id", "recipe_hash", "child_id", "run_id", "idempotency_key", "target_session", "proposal_id",
  "evaluation_id", "probe_id", "dataset_id", "split_hash", "fingerprint", "baseline_id", "intervention_id"]);
const fields = [
  "event_id", "session", "seq", "at", "route", "tool", "via", "request_id", "status", "error_code", "field", "checkpoint_id",
  "other_checkpoint_id", "selection_id", "selections", "ops", "recipe_hash", "child_id", "run_id", "idempotency_key", "replayed", "replay_of",
  "existing", "charged", "budget_charged", "capabilities", "links", "duration_ms", "control", "evaluation_id", "probe_id", "dataset_id", "split", "split_hash",
  "fingerprint", "baseline_id", "baseline_cached", "eval_samples", "intervention_id", "units", "page_offset", "page_limit", "principal", "target_session", "budget_change", "proposal_id",
] as const;

function CopyField({label, value, copied}: {label: string; value: string; copied: (message: string) => void}) {
  return <button type="button" className="tv-copy" aria-label={"Copy " + label} title={"Copy " + label} onClick={() => {
    navigator.clipboard.writeText(value).then(() => copied(label + " copied.")).catch(() => copied("Clipboard unavailable."));
  }}>⧉</button>;
}
export function Links({event, jump}: {event: TraceEvent; jump: Evidence["jump"]}) {
  const read = event.links?.selection_seqs ?? [];
  const plan = event.links?.plan_seqs ?? [];
  return <div className="tv-links">
    {!!read.length && <div><span>Reads</span>{read.map(seq => <button type="button" key={seq} onClick={() => jump(seq)}>#{seq}</button>)}</div>}
    {!!plan.length && <div><span>Plans</span>{plan.map(seq => <button type="button" key={seq} onClick={() => jump(seq)}>#{seq}</button>)}</div>}
    {event.links?.proposal_seq !== undefined && <div><span>Proposal</span><button type="button" onClick={() => jump(event.links!.proposal_seq!)}>#{event.links.proposal_seq}</button></div>}
    {event.replay_of !== null && <div><span>Original event</span><button type="button" onClick={() => jump(event.replay_of!)}>#{event.replay_of}</button></div>}
  </div>;
}
function Hypothesis({recipe}: {recipe?: TraceRecipe}) {
  return recipe?.hypothesis ? <div className="tv-hypothesis"><strong>Agent’s stated hypothesis</strong><p>{recipe.hypothesis}</p></div> : null;
}
const titleFor = (id: string, names: Map<string, string>) => names.get(id) ?? shortId(id);

export function TraceInspector({event, origin, decision, review, session, requested, more, load, measurement, ...evidence}: Evidence & {
  event?: TraceEvent; origin?: TraceEvent; decision?: TraceEvent; measurement?: ReactNode; review?: () => void; session: SessionRow; requested?: number; more: boolean; load: () => void;
}) {
  const [copied, setCopied] = useState("");
  const [actionError, setActionError] = useState("");
  const summary = event && statusMeaning(event, session.egress_class);
  const checkpoint = event?.checkpoint_id ?? origin?.checkpoint_id;
  const child = event?.child_id ?? event?.control?.child_id;
  const firstSelection = event?.selection_id ?? event?.selections[0] ?? origin?.selections[0];
  const selector = checkpoint && firstSelection ? evidence.resolutions.get(selectionKey(checkpoint, firstSelection))?.selector : undefined;
  const hash = event?.recipe_hash ?? origin?.recipe_hash;
  const recipe = hash ? evidence.recipes.get(hash) : undefined;
  return <aside className="tv-inspector" aria-label="Event inspector">
    <header className="tv-section"><h2>{event ? "Event #" + event.seq : "Event inspector"}</h2>
      {event && <span className="tv-meta">{toolOf(event)} · {timeText(event.at)} · HTTP {event.status}</span>}</header>
    {event ? <>
      <section className="tv-section"><h3>Event fields</h3><dl className="tv-fields">
        {fields.filter(key => event[key] !== null && event[key] !== undefined).map(key => {
          const field = event[key];
          const value = typeof field === "string" ? field : JSON.stringify(field);
          return <div key={key}><dt>{key === "idempotency_key" ? "idempotency digest" : key}</dt>
            <dd><code title={value}>{idFields.has(key) ? shortId(value) : value}</code>{idFields.has(key) && <CopyField label={key} value={value} copied={setCopied}/>}</dd></div>;
        })}
      </dl><span className="visually-hidden" role="status">{copied}</span></section>
      {selector && <section className="tv-section"><h3>Resolved selector</h3><code>{JSON.stringify(selector)}</code></section>}
      {summary && <section className="tv-section"><h3>What this means</h3><p className="tv-meaning" data-error={failed(event)}>{summary}</p>
        {decision && <button className="tv-text-button" type="button" onClick={() => evidence.jump(decision.seq)}>
          {decision.control?.action === "approve" ? "Approved" : "Rejected"} at #{decision.seq}</button>}</section>}
      {measurement}
      {(event.links || event.replay_of !== null) && <section className="tv-section"><h3>Provenance</h3><Links event={event} jump={evidence.jump}/></section>}
      {recipe?.hypothesis && <section className="tv-section"><Hypothesis recipe={recipe}/></section>}
      <section className="tv-section">
        <h3>Evidence</h3>
        <div className="tv-actions">
          {checkpoint && <button type="button" onClick={() => evidence.actions.tensor(checkpoint, selector && "tensor" in selector ? selector : undefined)}>Open in Tensors ↗</button>}
          {checkpoint && child && event.status < 400 &&
            <button type="button" onClick={() => evidence.actions.tensor(child, undefined, checkpoint)}>Compare child with parent ↗</button>}
          {checkpoint && selector && "group" in selector && <button type="button" onClick={() =>
            evidence.actions.architecture(checkpoint, selector.group, Array.isArray(selector.units) ? selector.units[0] : selector.units?.start)}>Open in Architecture ↗</button>}
          {review && <button type="button" onClick={review}>Review proposal</button>}
          {event.run_id && <button type="button" onClick={() => {
            setActionError(""); void evidence.actions.run(event.run_id!).catch(error => setActionError(error instanceof Error ? error.message : String(error)));
          }}>Open run ↗</button>}
          {event.request_id && <button type="button" onClick={() => {
            void navigator.clipboard.writeText(event.request_id!).then(() => setCopied("Request ID copied.")).catch(() => setCopied("Clipboard unavailable."));
          }}>Copy request ID</button>}
        </div>
        {actionError && <p role="alert" className="tv-alert-text">{actionError}</p>}
      </section>
    </> : <section className="tv-section"><p>{requested ? "Event #" + requested + " is not loaded." : "Select an event to inspect its recorded fields."}</p>
      {requested !== undefined && more && <button type="button" onClick={load}>Load next page</button>}</section>}
    <p className="tv-privacy">The trace records IDs. Opening a tensor reads it as you (human session).</p>
  </aside>;
}

export function TraceApplies({events, measurement, ...evidence}: Evidence & {events: TraceEvent[]; measurement?: (seq: number) => ReactNode}) {
  const applies = events.filter(isApply);
  const [openRecipe, setOpenRecipe] = useState<number>();
  return <div className="tv-cards">
    {!applies.length && <p className="tv-empty">No successful applies in loaded events.</p>}
    {applies.map(event => {
      const origin = events.find(e => e.seq === event.links?.proposal_seq);
      const entry = origin ?? event;
      const parent = entry.checkpoint_id;
      const child = event.child_id ?? event.control?.child_id;
      const replays = events.filter(e => e.replay_of === entry.seq && e.child_id === child);
      const recipe = entry.recipe_hash ? evidence.recipes.get(entry.recipe_hash) : undefined;
      return <article className="tv-card" key={event.event_id}>
        <header><strong>#{event.seq} · {event.control?.action === "approve" ? "approved proposal" : "apply"}</strong><time dateTime={event.at}>{timeText(event.at)}</time>
          {event.charged > 0 && <span className="tv-flag tv-charge">charged</span>}
          {event.existing === true && <span className="tv-flag">existing</span>}</header>
        <div className="tv-lineage">
          {parent && <span title={parent}>Parent <strong>{titleFor(parent, evidence.names)}</strong></span>}
          {child && <><span aria-hidden="true">→</span><span title={child}>Child <strong>{titleFor(child, evidence.names)}</strong></span></>}
        </div>
        {event.existing !== null && <p className="tv-meta">{event.existing ? "Child already in the catalog · nothing written" : "checkpoint_created"}</p>}
        {entry.recipe_hash && <p className="tv-meta" title={entry.recipe_hash}>Recipe {shortId(entry.recipe_hash)}
          {typeof entry.ops === "number" && " · " + count(entry.ops) + " ops"} · {count(entry.selections.length)} selections</p>}
        <Hypothesis recipe={recipe}/>
        {measurement?.(event.seq)}
        <Links event={event} jump={evidence.jump}/>
        {!!replays.length && <p className="tv-meta">Replayed {replays.map(e => <button type="button" className="tv-text-button" key={e.seq} onClick={() => evidence.jump(e.seq)}>#{e.seq}</button>)} · no new charge</p>}
        <div className="tv-actions">
          {parent && child &&
            <button type="button" onClick={() => evidence.actions.tensor(child, undefined, parent)}>Compare child with parent ↗</button>}
          {recipe && <button type="button" aria-expanded={openRecipe === event.seq} onClick={() => setOpenRecipe(openRecipe === event.seq ? undefined : event.seq)}>Open recipe</button>}
          <button type="button" onClick={() => evidence.jump(event.seq)}>Show in timeline</button>
        </div>
        {recipe && openRecipe === event.seq && <div className="tv-recipe tv-meta"><code title={recipe.recipe_hash}>{recipe.recipe_hash}</code>
          {recipe.ops !== null && <p>{count(recipe.ops)} ops</p>}</div>}
      </article>;
    })}
  </div>;
}

export function TraceDialog({title, cancel, children, action}: {title: string; cancel: () => void; children: ReactNode; action: ReactNode}) {
  return <dialog className="tv-dialog" aria-label={title} ref={dialog => {
    if (dialog && !dialog.open) {dialog.showModal(); dialog.querySelector<HTMLButtonElement>("[data-cancel]")?.focus();}
  }} onCancel={e => {e.preventDefault(); cancel();}}>
    <h2>{title}</h2>{children}<button type="button" data-cancel onClick={cancel}>Cancel</button>{action}
  </dialog>;
}
