import {useState, type ReactNode} from "react";
import {count} from "./tensorTree";
import {failed, isReplay, selectionKey, shortId, statusMeaning, toolOf} from "./traceModel";
import type {ResolvedSelection, SessionRow, TraceEvent, TraceRecipe} from "./traceSource";
import {timeText} from "./TraceTimeline";

export type TraceActions = {
  tensor: (id: string, selector?: Extract<ResolvedSelection["selector"], {tensor: string}>, compare?: string) => void;
  architecture: (id: string, group: string, unit?: number) => void;
  run: (id: string) => Promise<void>;
};
type Evidence = {
  names: Map<string, string>; resolutions: Map<string, ResolvedSelection>; recipes: Map<string, TraceRecipe>;
  actions: TraceActions; jump: (seq: number) => void;
};
const idFields = new Set(["session", "request_id", "checkpoint_id", "other_checkpoint_id", "selection_id", "recipe_hash", "child_id", "run_id", "idempotency_key", "target_session", "proposal_id"]);
const fields = [
  "event_id", "session", "seq", "at", "route", "tool", "via", "request_id", "status", "error_code", "field", "checkpoint_id",
  "other_checkpoint_id", "selection_id", "selections", "ops", "recipe_hash", "child_id", "run_id", "idempotency_key", "replayed", "replay_of",
  "existing", "charged", "budget_charged", "capabilities", "links", "duration_ms", "principal", "target_session", "budget_change", "proposal_id",
] as const;

function CopyField({label, value, copied}: {label: string; value: string; copied: (message: string) => void}) {
  return <button type="button" className="tv-copy" aria-label={"Copy " + label} title={"Copy " + label} onClick={() => {
    navigator.clipboard.writeText(value).then(() => copied(label + " copied.")).catch(() => copied("Clipboard unavailable."));
  }}>⧉</button>;
}
function Links({event, jump}: {event: TraceEvent; jump: Evidence["jump"]}) {
  const read = event.links?.selection_seqs ?? [];
  const plan = event.links?.plan_seqs ?? [];
  return <div className="tv-links">
    {!!read.length && <div><span>Reads</span>{read.map(seq => <button type="button" key={seq} onClick={() => jump(seq)}>#{seq}</button>)}</div>}
    {!!plan.length && <div><span>Plans</span>{plan.map(seq => <button type="button" key={seq} onClick={() => jump(seq)}>#{seq}</button>)}</div>}
    {event.replay_of !== null && <div><span>Original event</span><button type="button" onClick={() => jump(event.replay_of!)}>#{event.replay_of}</button></div>}
  </div>;
}
function Hypothesis({recipe}: {recipe?: TraceRecipe}) {
  return recipe?.hypothesis ? <div className="tv-hypothesis"><strong>Agent’s stated hypothesis</strong><p>{recipe.hypothesis}</p></div> : null;
}
const titleFor = (id: string, names: Map<string, string>) => names.get(id) ?? shortId(id);

export function TraceInspector({event, session, requested, more, load, ...evidence}: Evidence & {
  event?: TraceEvent; session: SessionRow; requested?: number; more: boolean; load: () => void;
}) {
  const [copied, setCopied] = useState("");
  const [actionError, setActionError] = useState("");
  const summary = event && statusMeaning(event, session.egress_class);
  const checkpoint = event?.checkpoint_id;
  const firstSelection = event?.selection_id ?? event?.selections[0];
  const selector = checkpoint && firstSelection ? evidence.resolutions.get(selectionKey(checkpoint, firstSelection))?.selector : undefined;
  const recipe = event?.recipe_hash ? evidence.recipes.get(event.recipe_hash) : undefined;
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
      {summary && <section className="tv-section"><h3>What this means</h3><p className="tv-meaning" data-error={failed(event)}>{summary}</p></section>}
      {(event.links || event.replay_of !== null) && <section className="tv-section"><h3>Provenance</h3><Links event={event} jump={evidence.jump}/></section>}
      {recipe?.hypothesis && <section className="tv-section"><Hypothesis recipe={recipe}/></section>}
      <section className="tv-section">
        <h3>Evidence</h3>
        <div className="tv-actions">
          {checkpoint && <button type="button" onClick={() => evidence.actions.tensor(checkpoint, selector && "tensor" in selector ? selector : undefined)}>Open in Tensors ↗</button>}
          {checkpoint && event.child_id && event.status < 400 &&
            <button type="button" onClick={() => evidence.actions.tensor(event.child_id!, undefined, checkpoint)}>Compare child with parent ↗</button>}
          {checkpoint && selector && "group" in selector && <button type="button" onClick={() =>
            evidence.actions.architecture(checkpoint, selector.group, Array.isArray(selector.units) ? selector.units[0] : selector.units?.start)}>Open in Architecture ↗</button>}
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

export function TraceApplies({events, ...evidence}: Evidence & {events: TraceEvent[]}) {
  const applies = events.filter(event => toolOf(event) === "apply" && event.status >= 200 && event.status < 300 && event.status !== 202 && !isReplay(event));
  const [openRecipe, setOpenRecipe] = useState<number>();
  return <div className="tv-cards">
    {!applies.length && <p className="tv-empty">No successful applies in loaded events.</p>}
    {applies.map(event => {
      const replays = events.filter(e => e.replay_of === event.seq);
      const recipe = event.recipe_hash ? evidence.recipes.get(event.recipe_hash) : undefined;
      return <article className="tv-card" key={event.event_id}>
        <header><strong>#{event.seq} · apply</strong><time dateTime={event.at}>{timeText(event.at)}</time>
          {event.charged > 0 && <span className="tv-flag tv-charge">charged</span>}
          {event.existing === true && <span className="tv-flag">existing</span>}</header>
        <div className="tv-lineage">
          {event.checkpoint_id && <span title={event.checkpoint_id}>Parent <strong>{titleFor(event.checkpoint_id, evidence.names)}</strong></span>}
          {event.child_id && <><span aria-hidden="true">→</span><span title={event.child_id}>Child <strong>{titleFor(event.child_id, evidence.names)}</strong></span></>}
        </div>
        {event.existing !== null && <p className="tv-meta">{event.existing ? "Child already in the catalog · nothing written" : "checkpoint_created"}</p>}
        {event.recipe_hash && <p className="tv-meta" title={event.recipe_hash}>Recipe {shortId(event.recipe_hash)}
          {typeof event.ops === "number" && " · " + count(event.ops) + " ops"} · {count(event.selections.length)} selections</p>}
        <Hypothesis recipe={recipe}/>
        <Links event={event} jump={evidence.jump}/>
        {!!replays.length && <p className="tv-meta">Replayed {replays.map(e => <button type="button" className="tv-text-button" key={e.seq} onClick={() => evidence.jump(e.seq)}>#{e.seq}</button>)} · no new charge</p>}
        <div className="tv-actions">
          {event.checkpoint_id && event.child_id &&
            <button type="button" onClick={() => evidence.actions.tensor(event.child_id!, undefined, event.checkpoint_id!)}>Compare child with parent ↗</button>}
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
