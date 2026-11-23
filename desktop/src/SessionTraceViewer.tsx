/// <reference types="vite/client" />
import {useEffect, useMemo, useRef, useState} from "react";
import {compact, count} from "./tensorTree";
import {dimensions, liveTraceSource, type Budget, type Dimension, type Proposal, type ProposalPreview, type ResolvedSelection, type SessionRow, type TraceEvent, type TraceRecipe, type TraceSource} from "./traceSource";
import {failed, liveSession, selectionKey, sessionsFromEvents, shortId} from "./traceModel";
import {TraceTimeline, timeText} from "./TraceTimeline";
import {Links, TraceApplies, TraceDialog, TraceInspector, type TraceActions} from "./TraceInspector";
import {traceFailure, useTraceFeed, type TraceFailure} from "./useTraceFeed";
import "./sessionTraceViewer.css";

const dimensionLabel: Record<Dimension, string> = {values: "Values", scan: "Scan", applies: "Applies", blob_bytes: "Blob"};
const blankSession = (id: string): SessionRow => ({session: id, kind: id === "human" ? "human" : "agent",
  created_at: null, last_at: null, last_status: null, capabilities: [], events: 0, applies: 0, charged_runs: 0,
  egress_class: null, revoked: null, via_mix: null});
type View = "Timeline" | "Applies" | "Proposals";
const isFixture = (import.meta.env.MODE === "trace-fixture" || import.meta.env.DEV) && new URLSearchParams(location.search).get("trace") !== "live";

function patchLocation(session: string, seq?: number) {
  const url = new URL(location.href);
  if (session) url.searchParams.set("session", session); else url.searchParams.delete("session");
  if (seq === undefined) url.searchParams.delete("seq"); else url.searchParams.set("seq", String(seq));
  history.replaceState(null, "", url);
}

function Failure({error, retry, stale}: {error: TraceFailure; retry: () => void; stale?: boolean}) {
  return <div className="tv-failure" role="alert">
    <span>{stale ? "Showing stale events. " : ""}{error.message}</span>
    {error.code && <code>{error.code}</code>}{error.requestId && <code title={error.requestId}>Request {error.requestId}</code>}
    <button type="button" onClick={retry}>Retry</button>
  </div>;
}
const egressLabels = ["Class 0 · metadata only", "Class 1 · stats only", "Class 2 · values allowed"];

function SessionList({rows, active, select, discovery, loadMore, now}: {rows: SessionRow[]; active: string; select: (id: string) => void;
  discovery: boolean; loadMore?: () => void; now: number}) {
  function row(session: SessionRow) {
    return <button type="button" className="tv-session" aria-current={session.session === active ? "true" : undefined}
      onClick={() => select(session.session)} key={session.session}>
      <span className="tv-session-head"><span title={session.session}>{shortId(session.session)}</span>
        {liveSession(session, now) && <i className="tv-live-dot" aria-label="Active in the last minute"/>}
        {session.revoked && <span className="tv-flag tv-revoked">Revoked</span>}</span>
      {session.created_at && <span className="tv-meta" title={session.created_at}>Started {timeText(session.created_at)}
        {session.kind === "agent" && " · server process"}</span>}
      {session.last_at && <span className="tv-meta" title={session.last_at}>Last event {timeText(session.last_at)}</span>}
      {!!session.capabilities.length && <span className="tv-caps">{session.capabilities.map(cap => <span className="tv-flag" key={cap}>{cap}</span>)}</span>}
      <span className="tv-meta">{count(session.events)} events · {count(session.applies)} applies · {count(session.charged_runs)} charged</span>
      {session.egress_class !== null && <span className="tv-meta">{egressLabels[session.egress_class]}</span>}
      {session.via_mix && <span className="tv-meta">{Object.entries(session.via_mix).map(([via, n]) => via + " " + count(n)).join(" · ")}</span>}
      {session.last_status === 429 && <span className="tv-alert-text">Budget used up</span>}
      {session.revoked && <span className="tv-meta">Revoked {timeText(session.revoked)}</span>}
    </button>;
  }
  return <aside className="tv-sessions" aria-label="Sessions">
    <h2>Agent sessions</h2>
    {discovery && <p className="tv-meta tv-list-note">From loaded events</p>}
    {rows.filter(item => item.kind === "agent").map(row)}
    {loadMore && <button type="button" className="tv-list-more" onClick={loadMore}>Load more sessions</button>}
    <h2>Human</h2>
    <p className="tv-meta tv-list-note">Hidden by default</p>
    {rows.filter(item => item.kind === "human").map(row)}
  </aside>;
}

function BudgetMeter({dimension, budget, edit, jump}: {dimension: Dimension; budget: Budget; edit?: () => void; jump?: () => void}) {
  const ratio = budget.limit === null ? null : budget.limit === 0 ? 1 : budget.used / budget.limit;
  const format = (n: number) => compact(n, dimension === "blob_bytes");
  return <div className="tv-meter" data-level={ratio === null ? "normal" : ratio >= 1 ? "exhausted" : ratio >= .8 ? "near" : "normal"}>
    <span>{dimensionLabel[dimension]} <strong>{format(budget.used)}{budget.limit !== null && " / " + format(budget.limit)}</strong></span>
    {ratio !== null && <progress aria-label={dimensionLabel[dimension] + " budget used"} max={1} value={Math.min(1, ratio)}/>}
    {jump && ratio !== null && ratio >= 1 && <button type="button" className="tv-text-button" onClick={jump}>429</button>}
    {edit && <button type="button" className="tv-text-button" aria-label={"Edit " + dimensionLabel[dimension] + " budget"} onClick={edit}>Edit</button>}
  </div>;
}
function Proposals({items, recipes, names, events, jump, canApprove, preview, decide}: {items: Proposal[]; recipes: Map<string, TraceRecipe>;
  names: Map<string, string>; events: TraceEvent[]; jump: (seq: number) => void; canApprove: boolean;
  preview?: (proposal: Proposal) => Promise<ProposalPreview>; decide?: (id: string, action: "approve" | "reject") => void}) {
  const [plans, setPlans] = useState<Record<string, ProposalPreview>>();
  const [busy, setBusy] = useState<string>();
  const [error, setError] = useState("");
  return <div className="tv-cards">{items.length === 0 && <p className="tv-empty">No pending proposals.</p>}{items.map(item => {
    const recipe = recipes.get(item.recipe_hash);
    const plan = plans?.[item.proposal_id];
    const event = events.find(event => event.seq === item.seq);
    return <article className="tv-card" key={item.proposal_id}>
      <header><strong title={item.proposal_id}>{shortId(item.proposal_id)}</strong>{item.seq !== null && <button type="button" className="tv-text-button" onClick={() => jump(item.seq!)}>From #{item.seq} apply</button>}
        <time dateTime={item.created_at}>{timeText(item.created_at)}</time></header>
      <p className="tv-meta">Parent <strong title={item.checkpoint_id}>{names.get(item.checkpoint_id) ?? shortId(item.checkpoint_id)}</strong></p>
      <p className="tv-meta" title={item.recipe_hash}>Recipe {shortId(item.recipe_hash)} · {count(item.selections.length)} selections</p>
      {recipe?.hypothesis && <div className="tv-hypothesis"><strong>Agent’s stated hypothesis</strong><p>{recipe.hypothesis}</p></div>}
      {event?.links && <Links event={event} jump={jump}/>}
      {plan && <div className="tv-plan"><h3>Plan preview</h3>
        {!!plan.ops.length && <table><thead><tr><th>Operation</th><th>Tensor</th><th>Selected values</th></tr></thead>
          <tbody>{plan.ops.map((op, i) => <tr key={i}><td>{op.op}</td><td>{op.tensor}</td><td>{count(op.selected_elems)}</td></tr>)}</tbody></table>}
        {[...plan.refusals, ...plan.warnings].map((message, i) => <p className="tv-meaning" key={i}>{message}</p>)}
        {plan.side_effects.map((effect, i) => <p key={i}>{effect.kind === "checkpoint_may_be_created" ? "Checkpoint may be created" : String(effect.kind)}{typeof effect.bytes === "number" && " · " + compact(effect.bytes, true)}
          {typeof effect.changed_values === "number" && " · " + count(effect.changed_values) + " changed values"}</p>)}</div>}
      {error && <p role="alert">{error}</p>}
      {preview && decide && <div className="tv-actions">
        <button type="button" disabled={busy !== undefined} onClick={() => {
          setBusy(item.proposal_id); setError("");
          setPlans(old => {const next = {...old}; delete next[item.proposal_id]; return next;});
          void preview(item).then(reply => setPlans(old => ({...old, [item.proposal_id]: reply})))
            .catch(err => setError(traceFailure(err).message)).finally(() => setBusy(undefined));
        }}>Preview plan</button>
        <button type="button" disabled={!canApprove || !plan || !!plan.refusals.length || busy === item.proposal_id} onClick={() => decide(item.proposal_id, "approve")}>Approve</button>
        <button type="button" onClick={() => decide(item.proposal_id, "reject")}>Reject</button>
      </div>}
    </article>;
  })}</div>;
}
type Confirm = {kind: "revoke"} | {kind: "budget"; dimension: Dimension; budget: Budget} |
  {kind: "proposal"; id: string; decision: "approve" | "reject"};

function SessionBody({source, session, actions, following, setFollowing, now, refreshSessions, pending}: {source: TraceSource; session: SessionRow;
  actions: TraceActions; following: boolean; setFollowing: (value: boolean) => void; now: number; refreshSessions: () => void; pending: (value: number) => void}) {
  const feed = useTraceFeed(source, session.session, following);
  const [selected, setSelected] = useState<number | undefined>(() => {
    const n = Number(new URLSearchParams(location.search).get("seq"));
    return Number.isSafeInteger(n) && n > 0 ? n : undefined;
  });
  const [view, setView] = useState<View>("Timeline");
  const [names, setNames] = useState(new Map<string, string>());
  const [resolutions, setResolutions] = useState(new Map<string, ResolvedSelection>());
  const [recipes, setRecipes] = useState(new Map<string, TraceRecipe>());
  const seenNames = useRef(new Set<string>());
  const seenSelections = useRef(new Set<string>());
  const seenRecipes = useRef(new Set<string>());
  const [queue, setQueue] = useState<Proposal[] | null>(null);
  const pendingCount = queue?.length ?? 0;
  useEffect(() => pending(pendingCount), [pending, pendingCount]);
  const [queueError, setQueueError] = useState<TraceFailure>();
  const [queueRevision, refreshQueue] = useState(0);
  const [confirm, setConfirm] = useState<Confirm>();
  const [limit, setLimit] = useState("");
  const [lower, setLower] = useState(false);
  const [controlError, setControlError] = useState("");
  const [controlBusy, setControlBusy] = useState(false);
  const sessionId = session.session;
  const canSupervise = !!source.controls && session.current === true && !session.revoked;
  const unlimited = confirm?.kind === "budget" && confirm.dimension !== "applies" && limit.trim() === "";
  const events = feed.events;
  const summary = feed.summary;
  const selectedEvent = events.find(event => event.seq === selected);
  const scopedSignal = useRef<AbortController>(null);

  useEffect(() => {
    const controller = new AbortController(); scopedSignal.current = controller;
    return () => controller.abort();
  }, []);
  useEffect(() => {
    const signal = scopedSignal.current?.signal;
    if (!signal) return;
    const ids = [...new Set(events.flatMap(e => [e.checkpoint_id, e.other_checkpoint_id, e.child_id, e.control?.child_id].filter((id): id is string => !!id)))].filter(id => !seenNames.current.has(id));
    ids.forEach(id => seenNames.current.add(id));
    const normalized = new Map<string, string>();
    void (async () => {
      // Each ID is looked up at most once per loaded session, without blocking the timeline.
      for (let i = 0; i < ids.length && !signal.aborted; i += 8) await Promise.all(ids.slice(i, i + 8).map(async id => {
        try {const name = await source.checkpointName(id, signal); if (name) normalized.set(id, name);}
        catch { /* The short checkpoint ID is the fallback. */ }
      }));
      if (!signal.aborted && normalized.size) setNames(old => new Map([...old, ...normalized]));
    })();
    if (source.resolve) {
      const refs = events.flatMap(event => event.checkpoint_id ? [event.selection_id, ...event.selections].flatMap(id =>
        id ? [{checkpoint_id: event.checkpoint_id!, selection_id: id}] : []) : []);
      const batch = refs.filter(ref => {
        const key = selectionKey(ref.checkpoint_id, ref.selection_id);
        if (seenSelections.current.has(key)) return false;
        seenSelections.current.add(key); return true;
      });
      for (let i = 0; i < batch.length; i += 500) void source.resolve(batch.slice(i, i + 500), signal).then(rows => {
        if (!signal.aborted) setResolutions(old => new Map([...old, ...rows.map(row => [selectionKey(row.checkpoint_id, row.selection_id), row] as const)]));
      }).catch(() => {});
    }
  }, [events, source]);
  useEffect(() => {
    const signal = scopedSignal.current?.signal;
    if (!signal || !source.recipe) return;
    const hashes = [...new Set([...events.map(e => e.recipe_hash), ...(queue?.map(item => item.recipe_hash) ?? [])])];
    for (const hash of hashes) {
      if (!hash || seenRecipes.current.has(hash)) continue;
      seenRecipes.current.add(hash);
      void source.recipe(hash, signal).then(item => {
        if (item && !signal.aborted) setRecipes(old => new Map(old).set(hash, item));
      }).catch(() => {});
    }
  }, [events, queue, source]);
  useEffect(() => {
    if (!source.approvals || session.kind !== "agent") return;
    const controller = new AbortController();
    let timer: number | undefined, busy = false;
    async function refresh() {
      if (controller.signal.aborted || busy) return;
      busy = true;
      try {
        const list = await source.approvals!(sessionId, controller.signal);
        if (!controller.signal.aborted) {setQueue(list); setQueueError(undefined);}
      } catch (error) {
        if (!controller.signal.aborted) setQueueError(traceFailure(error));
      } finally {
        busy = false;
        if (following && !controller.signal.aborted && !document.hidden) timer = window.setTimeout(refresh, 2000);
      }
    }
    function visibility() {window.clearTimeout(timer); if (!document.hidden) void refresh();}
    void refresh();
    document.addEventListener("visibilitychange", visibility);
    return () => {controller.abort(); window.clearTimeout(timer); document.removeEventListener("visibilitychange", visibility);};
  }, [source, sessionId, session.kind, queueRevision, following]);

  function select(seq: number) {setSelected(seq); patchLocation(sessionId, seq);}
  function jump(seq: number) {
    setView("Timeline"); select(seq);
    if (!events.some(event => event.seq === seq) && feed.more) feed.load();
  }
  async function confirmControl() {
    const controls = source.controls;
    if (!controls || !confirm || controlBusy) return;
    setControlError("");
    const number = unlimited ? null : Number(limit);
    if (confirm.kind === "budget") {
      if (number !== null && (!/^\d+$/.test(limit.trim()) || !Number.isSafeInteger(number))) {setControlError("Enter a non-negative whole number."); return;}
      if (number !== null && (confirm.budget.limit === null || number < confirm.budget.limit) && !lower) {setLower(true); return;}
    }
    setControlBusy(true);
    try {
      if (confirm.kind === "revoke") await controls.revoke(sessionId);
      if (confirm.kind === "budget") await controls.budget(sessionId, confirm.dimension, number);
      if (confirm.kind === "proposal") await controls.decide(confirm.id, confirm.decision);
      setConfirm(undefined); feed.load(); refreshSessions(); refreshQueue(n => n + 1);
      if (confirm.kind === "proposal" && confirm.decision === "approve") setView("Applies");
    } catch (error) {setControlError(traceFailure(error).message);}
    finally {setControlBusy(false);}
  }
  const latestCaps = session.capabilities;
  const loadedErrors = useMemo(() => events.filter(failed), [events]);
  const blocked = loadedErrors.filter(e => e.status === 403).length;
  const budgetEvent = (dimension: Dimension) => events.find(event => event.status === 429 &&
    (event.field === dimension || event.field === "budget." + dimension))?.seq;
  const evidence = {names, resolutions, recipes, actions, jump};
  return <>
    <header className="tv-context context-bar"><div className="context-title">Sessions <b>/</b><strong title={sessionId}>{shortId(sessionId)}</strong></div>
      {liveSession(session, now) && <span className="tv-live-pill"><i className="tv-live-dot"/>Live</span>}
      <div className="tv-context-meters">
        {session.egress_class !== null && <span className="tv-flag">
          {egressLabels[session.egress_class]}</span>}
        {session.kind === "agent" && (summary?.budget ? dimensions.map(dimension => {
          const budget = summary.budget?.[dimension], seq = budgetEvent(dimension);
          return budget && <BudgetMeter key={dimension} dimension={dimension} budget={budget} jump={seq !== undefined ? () => jump(seq) : undefined}
            edit={canSupervise ? () => {
              setConfirm({kind: "budget", dimension, budget}); setLimit(budget.limit?.toString() ?? ""); setLower(false); setControlError("");
            } : undefined}/>;
        }) : summary && <div className="tv-meter"><span>Applies <strong>{count(summary.charged_runs)} charged</strong></span>
          <small>Limit not visible to the human session</small></div>)}
        {queue !== null && <button type="button" onClick={() => setView("Proposals")}>Approvals ({count(queue.length)})</button>}
        {session.revoked ? <span className="tv-revoked">Revoked {timeText(session.revoked)}</span> :
          canSupervise && <button type="button" className="tv-revoke" onClick={() => {
            setConfirm({kind: "revoke"}); setControlError("");
          }}>Revoke…</button>}
        <TraceHelp/>
      </div>
    </header>
    <section className="tv-center" aria-label="Session events">
      <div className="tv-tabs" role="group" aria-label="Trace view">
        {(["Timeline", "Applies", ...(queue === null ? [] : ["Proposals"])] as View[]).map(tab =>
          <button type="button" key={tab} aria-pressed={view === tab} onClick={() => setView(tab)}>{tab}{tab === "Proposals" && " (" + count(queue?.length ?? 0) + ")"}</button>)}
      </div>
      <div className="tv-counts">
        {summary && <span>{count(summary.events)} events · {count(summary.applies)} applies · {count(summary.replays)} replays ·
          {" "}{count(summary.errors ?? loadedErrors.length)} errors{summary.errors == null && " in loaded events"}
          {!!blocked && " (" + count(blocked) + " blocked in loaded events)"}</span>}
        {!!latestCaps.length && <span className="tv-caps" aria-label="Latest capabilities">{latestCaps.map(cap =>
          <span key={cap} className="tv-flag">{cap}</span>)}</span>}
      </div>
      {feed.error && <Failure error={feed.error} retry={feed.load} stale={!!events.length}/>}
      {queueError && <Failure error={queueError} retry={() => refreshQueue(n => n + 1)}/>}
      {view === "Timeline" && <TraceTimeline events={events} selected={selectedEvent} session={session} names={names} resolved={resolutions}
        following={following} setFollowing={setFollowing} more={feed.more} loading={feed.loading} stale={!!feed.error} load={feed.load} select={select} now={now}/>}
      {view === "Applies" && <><TraceApplies events={events} {...evidence}/>{feed.more && <button type="button" onClick={feed.load}>Load next page</button>}</>}
      {view === "Proposals" && <Proposals items={queue ?? []} recipes={recipes} names={names} events={events} jump={jump}
        canApprove={canSupervise} preview={source.controls?.preview}
        decide={source.controls ? (id, decision) => {setConfirm({kind: "proposal", id, decision}); setControlError("");} : undefined}/>}
    </section>
    <TraceInspector key={selected ?? "empty"} event={selectedEvent} origin={events.find(event => event.seq === selectedEvent?.links?.proposal_seq)}
      review={selectedEvent?.proposal_id && queue?.some(item => item.proposal_id === selectedEvent.proposal_id) ? () => setView("Proposals") : undefined}
      session={session} requested={selected} more={feed.more} load={feed.load} {...evidence}/>
    {confirm && <TraceDialog title={confirm.kind === "revoke" ? "Revoke " + shortId(sessionId) + "?" :
      confirm.kind === "budget" ? "Edit " + DimensionTitle(confirm.dimension) + " budget" : (confirm.decision === "approve" ? "Approve" : "Reject") + " proposal?"}
      cancel={() => {if (!controlBusy) setConfirm(undefined);}}
      action={<button className={confirm.kind === "revoke" ? "tv-revoke" : "primary"} type="button" disabled={controlBusy}
        onClick={() => void confirmControl()}>{confirm.kind === "revoke" ? "Revoke session" :
          confirm.kind === "budget" ? lower ? "Confirm lower limit" : "Save limit" : confirm.decision === "approve" ? "Approve proposal" : "Reject proposal"}</button>}>
      {confirm.kind === "revoke" && <p>Future requests from this session will be refused. Children it created stay in the catalog. Revoking this session is permanent.</p>}
      {confirm.kind === "budget" && <>
        <p>{count(confirm.budget.used)} used / {confirm.budget.limit === null ? "unlimited" : count(confirm.budget.limit)} current limit</p>
        <label>New limit <input inputMode="numeric" placeholder={confirm.dimension !== "applies" ? "Unlimited" : undefined} value={limit} disabled={controlBusy} onChange={e => {setLimit(e.target.value); setLower(false);}}/></label>
        {confirm.dimension !== "applies" && <p>Leave blank for no limit.</p>}
        {/^\d+$/.test(limit.trim()) && Number.isSafeInteger(Number(limit)) && <p>{count(Math.max(0, Number(limit) - confirm.budget.used))} remaining under this limit.</p>}
        {lower && <p className="tv-meaning">Confirm the lower limit. Requests that would exceed it will be refused.</p>}
      </>}
      {confirm.kind === "proposal" && <p>{confirm.decision === "approve" ? "Approve the previewed recipe and allow it to create a child." : "Reject this proposal without applying it."}</p>}
      {controlError && <p role="alert">{controlError}</p>}
    </TraceDialog>}
  </>;
}
const DimensionTitle = (dimension: Dimension) => dimensionLabel[dimension].toLowerCase();

function TraceHelp() {
  const dialog = useRef<HTMLDialogElement>(null);
  return <>
    <button type="button" aria-label="Session viewer shortcuts" onClick={() => dialog.current?.showModal()}>?</button>
    <dialog className="tv-dialog" ref={dialog} aria-label="Session viewer shortcuts"><h2>Session viewer shortcuts</h2>
      <p><kbd>j</kbd> / <kbd>k</kbd> move between events · <kbd>Enter</kbd> inspect</p><p><kbd>[</kbd> / <kbd>]</kbd> previous / next apply · <kbd>e</kbd> next error</p>
      <p><kbd>l</kbd> live tail · <kbd>/</kbd> filter</p><form method="dialog"><button>Close</button></form></dialog>
  </>;
}

function TraceWorkspace({source, actions, activity, pending}: {source: TraceSource; actions: TraceActions; activity: (milliseconds: number) => void; pending: (value: number) => void}) {
  const [session, setSession] = useState(() => new URLSearchParams(location.search).get("session") ?? "");
  const [following, setFollowing] = useState(true);
  const [catalog, setCatalog] = useState<SessionRow[]>();
  const [catalogError, setCatalogError] = useState<TraceFailure>();
  const [catalogRevision, setCatalogRevision] = useState(0);
  const [now, setNow] = useState(source.now());
  const discovery = useTraceFeed(source, undefined, following, source.sessions === null);
  const rows = useMemo(() => catalog ?? sessionsFromEvents(discovery.events), [catalog, discovery.events]);
  const selected = rows.find(row => row.session === session);
  useEffect(() => {
    const timer = window.setInterval(() => {if (!document.hidden) setNow(source.now());}, 1000);
    return () => window.clearInterval(timer);
  }, [source]);
  useEffect(() => {
    if (!source.sessions) return;
    const controller = new AbortController();
    let busy = false, timer: number | undefined;
    async function load() {
      if (busy || controller.signal.aborted) return;
      busy = true;
      try {
        const result = await source.sessions!(controller.signal);
        if (!controller.signal.aborted) {setCatalog(result.sort((a, b) =>
          (b.created_at ?? b.last_at ?? "").localeCompare(a.created_at ?? a.last_at ?? ""))); setCatalogError(undefined);}
      } catch (error) {if (!controller.signal.aborted) setCatalogError(traceFailure(error));}
      finally {
        busy = false;
        if (following && !controller.signal.aborted && !document.hidden) timer = window.setTimeout(load, 2000);
      }
    }
    function visible() {window.clearTimeout(timer); if (!document.hidden) void load();}
    void load();
    document.addEventListener("visibilitychange", visible);
    return () => {controller.abort(); window.clearTimeout(timer); document.removeEventListener("visibilitychange", visible);};
  }, [source, following, catalogRevision]);
  useEffect(() => {
    if (!session) {
      const first = rows.find(row => row.kind === "agent");
      if (first) {setSession(first.session); patchLocation(first.session);}
    }
    const remaining = rows.filter(row => row.kind === "agent" && !row.revoked && row.last_at)
      .reduce((n, row) => Math.max(n, Date.parse(row.last_at!) + 60_000 - now), 0);
    activity(remaining);
  }, [rows, now, session, activity]);
  const error = catalogError ?? discovery.error;
  const waiting = source.sessions ? catalog === undefined : discovery.loading;
  function refreshSessions() {if (source.sessions) setCatalogRevision(n => n + 1); else discovery.load();}
  return <main className="trace-viewer" id="work" tabIndex={-1} data-fixture={isFixture}>
    {isFixture && <div className="tv-fixture">Fixture data · simulation</div>}
    {!session && <header className="tv-context context-bar"><div className="context-title">Sessions</div>
      <div className="tv-context-meters"><TraceHelp/></div></header>}
    <SessionList rows={rows} active={session} select={id => {if (session !== id) {setSession(id); patchLocation(id);}}}
      discovery={!source.sessions} loadMore={discovery.more ? discovery.load : undefined} now={now}/>
    {error && (!session || catalogError) && <div className="tv-catalog-error"><Failure error={error} retry={refreshSessions} stale={!!rows.length}/></div>}
    {session ? <SessionBody key={session} source={source} session={selected ?? blankSession(session)} actions={actions}
      following={following} setFollowing={setFollowing} now={now} refreshSessions={refreshSessions} pending={pending}/> :
      <section className="tv-center tv-unselected">
        {waiting && !error ? <><h2 role="status">Loading sessions…</h2><div className="tv-skeleton">{Array.from({length: 7}, (_, i) => <i key={i}/>)}</div></> :
          !error && <><h2>No agent sessions {discovery.more ? "in loaded events" : "yet"}</h2>
            <p>Start the service with an agent token. The agent’s requests will appear here.</p>
            <a href="https://github.com/WeightForge/LayerDelta/blob/main/docs/guides/agent-integration.md" target="_blank" rel="noreferrer">Agent integration guide ↗</a></>}
      </section>}
    {!session && <aside className="tv-inspector tv-blank"><h2>Event inspector</h2><p>Select a session and an event.</p></aside>}
  </main>;
}

export function SessionTraceViewer({actions, activity, pending}: {actions: TraceActions; activity: (milliseconds: number) => void; pending: (value: number) => void}) {
  const [source, setSource] = useState<TraceSource | undefined>(isFixture ? undefined : liveTraceSource);
  useEffect(() => {
    if ((import.meta.env.MODE === "trace-fixture" || import.meta.env.DEV) && isFixture) {
      let live = true;
      void import("./traceFixture").then(({createTraceFixture}) => {if (live) setSource(createTraceFixture());});
      return () => {live = false;};
    }
  }, []);
  return source ? <TraceWorkspace source={source} actions={actions} activity={activity} pending={pending}/> :
    <main id="work" tabIndex={-1} role="status">Loading session fixture…</main>;
}
