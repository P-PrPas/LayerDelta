import {useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState} from "react";
import {count, useWindow} from "./tensorTree";
import type {ResolvedSelection, SessionRow, TraceEvent} from "./traceSource";
import {eventTarget, failed, isApply, isCharged, isReplay, laneOf, shortId, timelineRows, toolOf, type Lane} from "./traceModel";

const rowHeight = 31;
const icons: Record<Lane, string> = {Read: "◇", Plan: "▧", Write: "■", Run: "●", Meta: "·", Control: "◆"};
const date = new Intl.DateTimeFormat(undefined, {hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false});
export const timeText = (value: string) => Number.isFinite(Date.parse(value)) ? date.format(new Date(value)) : value;
function relativeTime(value: string, now: number) {
  const seconds = Math.floor((now - Date.parse(value)) / 1000);
  if (!Number.isFinite(seconds) || seconds < 0) return value;
  return seconds < 60 ? seconds + "s ago" : seconds < 3600 ? Math.floor(seconds / 60) + "m ago" : Math.floor(seconds / 3600) + "h ago";
}
const selectableLanes: Lane[] = ["Read", "Plan", "Write", "Run"];

export function TraceTimeline({events, selected, session, names, resolved, following, setFollowing, more, loading, stale, load, select, now}:
  {events: TraceEvent[]; selected?: TraceEvent; session: SessionRow; names: Map<string, string>; resolved: Map<string, ResolvedSelection>;
   following: boolean; setFollowing: (next: boolean) => void; more: boolean; loading: boolean; stale: boolean;
   load: () => void; select: (seq: number) => void; now: number}) {
  const [lanes, setLanes] = useState(new Set<Lane>());
  const [errors, setErrors] = useState(false);
  const [charged, setCharged] = useState(false);
  const [hide, setHide] = useState(false);
  const [checkpoint, setCheckpoint] = useState("");
  const [query, setQuery] = useState("");
  const [opened, setOpened] = useState(new Set<number>());
  const [cursor, setCursor] = useState<number>();
  const [scrolled, setScrolled] = useState(false);
  const [unseen, setUnseen] = useState(0);
  const [announcement, setAnnouncement] = useState("");
  const search = useRef<HTMLInputElement>(null);
  const atEnd = useRef(true);
  const previous = useRef(0);
  const announced = useRef(0);
  const selectionKey = selected?.seq;
  const links = useMemo(() => new Set(isApplySelected() ? [...(selected?.links?.plan_seqs ?? []), ...(selected?.links?.selection_seqs ?? [])] : []), [selected]);
  function isApplySelected() {return !!selected && (isApply(selected) || toolOf(selected) === "apply" && !!selected.links);}
  const keep = useCallback((event: TraceEvent) =>
    (!lanes.size || lanes.has(laneOf(event))) && (!errors || failed(event)) && (!charged || isCharged(event)) &&
    (!checkpoint || event.checkpoint_id === checkpoint || event.other_checkpoint_id === checkpoint || event.child_id === checkpoint) &&
    (!query || (toolOf(event) + " " + eventTarget(event, names, resolved) + " " + event.status + " " + (event.error_code ?? "")).toLowerCase().includes(query.toLowerCase())),
  [lanes, errors, charged, checkpoint, query, names, resolved]);
  const rows = useMemo(() => {
    const exposed = new Set(links);
    if (selectionKey !== undefined) exposed.add(selectionKey);
    if (cursor !== undefined) exposed.add(cursor);
    return timelineRows(events, links, opened, keep, hide, exposed);
  }, [events, links, opened, keep, hide, selectionKey, cursor]);
  const win = useWindow(rows.length, rowHeight);
  const checkpointIds = useMemo(() => [...new Set(events.flatMap(e =>
    [e.checkpoint_id, e.other_checkpoint_id, e.child_id].filter((id): id is string => !!id)))], [events]);
  const budgetRow = rows.findIndex(row => "event" in row && row.event.status === 429);
  const mix = Object.entries(session.via_mix ?? {}).filter(([, n]) => n > 0).length > 1;
  const focused = cursor ?? selectionKey;
  const focusIndex = Math.max(0, rows.findIndex(row => "event" in row && row.event.seq === focused));

  function updateScroll() {
    const el = win.ref.current;
    if (!el) return;
    atEnd.current = el.scrollHeight - el.scrollTop - el.clientHeight <= rowHeight * 1.25;
    setScrolled(!atEnd.current);
    if (atEnd.current) setUnseen(0);
  }
  function tail() {
    const el = win.ref.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    atEnd.current = true;
    setScrolled(false); setUnseen(0); win.onScroll();
  }
  useLayoutEffect(() => {
    const newer = Math.max(0, events.length - previous.current);
    const first = !previous.current && !!events.length;
    previous.current = events.length;
    if (!newer) return;
    if (first && selectionKey !== undefined) return;
    if (following && atEnd.current) tail();
    else setUnseen(n => n + newer);
  }, [events.length, rows.length, following]);
  useLayoutEffect(() => {
    if (selectionKey === undefined) return;
    const i = rows.findIndex(row => "event" in row && row.event.seq === selectionKey);
    if (i >= 0) win.scrollTo(i);
  }, [selectionKey, events.some(event => event.seq === selectionKey)]);
  useEffect(() => {
    if (!unseen) {setAnnouncement(""); return;}
    const timer = window.setTimeout(() => {
      announced.current = Date.now();
      setAnnouncement(count(unseen) + " new events");
    }, Math.max(0, 5000 - (Date.now() - announced.current)));
    return () => window.clearTimeout(timer);
  }, [unseen]);

  function move(key: string) {
    const candidates = events.filter(e => !hide || keep(e) || links.has(e.seq));
    const seq = cursor ?? selected?.seq;
    const i = candidates.findIndex(event => event.seq === seq);
    let next: TraceEvent | undefined;
    if (key === "j" || key === "ArrowDown") next = candidates[Math.min(candidates.length - 1, i + 1)];
    if (key === "k" || key === "ArrowUp") next = candidates[Math.max(0, i < 0 ? candidates.length - 1 : i - 1)];
    if (key === "e") next = candidates.find(event => event.seq > (seq ?? 0) && failed(event));
    if (key === "]") next = candidates.find(event => event.seq > (seq ?? 0) && isApply(event));
    if (key === "[") next = candidates.filter(event => event.seq < (seq ?? Infinity) && isApply(event)).at(-1);
    if (!next) return;
    if (key === "[" || key === "]" || key === "e") select(next.seq);
    setCursor(next.seq);
    requestAnimationFrame(() => {
      const row = win.ref.current?.querySelector<HTMLElement>("[data-seq='" + next.seq + "']");
      if (row) {row.scrollIntoView({block: "nearest"}); row.focus({preventScroll: true});}
    });
  }
  useLayoutEffect(() => {
    if (cursor === undefined) return;
    const i = rows.findIndex(row => "event" in row && row.event.seq === cursor);
    if (i >= 0) win.scrollTo(i);
  }, [cursor]);
  useEffect(() => {
    function keydown(event: KeyboardEvent) {
      if (event.defaultPrevented || event.altKey || event.metaKey || event.ctrlKey ||
        event.target instanceof Element && event.target.closest("input, select, textarea, dialog, [contenteditable]")) return;
      if (event.key === "/") {event.preventDefault(); search.current?.focus();}
      if (event.key === "l") {event.preventDefault(); setFollowing(!following);}
      if (["j", "k", "[", "]", "e"].includes(event.key)) {event.preventDefault(); move(event.key);}
    }
    window.addEventListener("keydown", keydown);
    return () => window.removeEventListener("keydown", keydown);
  });

  return <>
    <div className="tv-filters" aria-label="Timeline filters">
      {selectableLanes.map(lane => <button className="tv-chip" key={lane} aria-pressed={lanes.has(lane)} type="button" onClick={() => {
        const next = new Set(lanes); if (next.has(lane)) next.delete(lane); else next.add(lane); setLanes(next);
      }}>{lane}s</button>)}
      <button className="tv-chip" type="button" aria-pressed={errors} onClick={() => setErrors(!errors)}>Errors only</button>
      <button className="tv-chip" type="button" aria-pressed={charged} onClick={() => setCharged(!charged)}>Charged</button>
      <label><input type="checkbox" checked={hide} onChange={e => setHide(e.target.checked)}/> Hide filtered</label>
      <select aria-label="Filter by checkpoint" value={checkpoint} onChange={e => setCheckpoint(e.target.value)}>
        <option value="">All checkpoints</option>{checkpointIds.map(id => <option key={id} value={id}>{names.get(id) ?? shortId(id)}</option>)}
      </select>
      <input ref={search} aria-label="Filter trace events" value={query} onChange={e => setQuery(e.target.value)} placeholder="Filter events…"/>
    </div>
    <div className="tv-timeline-meta"><span>{stale ? "Disconnected · retrying" : !following ? "Live tail off" : scrolled ? "Paused (scrolled up)" : "Live"}</span>
      <button type="button" aria-pressed={following} onClick={() => {if (!following) tail(); setFollowing(!following);}}>Live tail</button>
      {more && <button type="button" onClick={load}>Load next page</button>}
    </div>
    <div className="tv-grid-wrap">
      <div className="tv-grid-head tv-event" aria-hidden="true"><span>Seq</span><span>Time</span><span/>
        <span>Tool</span><span>Target</span><span>Status</span><span>Flags</span></div>
      <div className="tv-event-scroll" role="grid" aria-label="Trace timeline" aria-rowcount={rows.length} aria-colcount={7}
        aria-busy={loading} ref={win.ref} onScroll={() => {win.onScroll(); updateScroll();}}
        data-stale={stale} data-highlight={links.size > 0} onKeyDown={e => {
          if (e.key === "ArrowUp" || e.key === "ArrowDown") {e.preventDefault(); move(e.key);}
          if (e.key === "Enter" && cursor !== undefined && e.target instanceof HTMLElement && e.target.dataset.seq) {
            e.preventDefault(); select(cursor);
          }
        }}>
        {loading && <p className="tv-empty" role="status">Loading events…</p>}
        {loading && <div aria-hidden="true" className="tv-skeleton">{Array.from({length: 7}, (_, i) => <i key={i}/>)}</div>}
        {!loading && !stale && !events.length && <p className="tv-empty">This session has made no requests.</p>}
        {!loading && !!events.length && !rows.length && <p className="tv-empty">No matching events.</p>}
        <div role="rowgroup" className="tv-event-list" style={{height: rows.length * rowHeight}}>
          {rows.slice(win.start, win.end).map((row, offset) => {
            const i = win.start + offset;
            const style = {top: i * rowHeight};
            if ("missing" in row) return <div role="row" className="tv-event tv-linked" key={row.key} style={style} aria-rowindex={i + 1}>
              <span role="rowheader">#{row.missing}</span><span role="gridcell" className="tv-full">Not loaded <button type="button" onClick={load}>Load</button></span>
            </div>;
            if ("burst" in row) {
              const first = row.burst[0], last = row.burst.at(-1)!;
              const targets = [...new Set(row.burst.map(event => eventTarget(event, names, resolved)))];
              return <div role="row" key={row.key} style={style} aria-rowindex={i + 1} className="tv-event tv-burst" data-dim={!keep(first)}>
                <span role="rowheader">#{first.seq}</span><span role="gridcell" className="tv-full">
                  <button type="button" aria-expanded={row.expanded} onClick={() => setOpened(previous => {
                    const next = new Set(previous); if (next.has(first.seq)) next.delete(first.seq); else next.add(first.seq); return next;
                  })}>{row.expanded ? "▾" : "▸"} {count(row.burst.length)} reads</button>
                  <span title={targets.join(" · ")}>{targets[0]}{targets.length > 1 && " +" + (targets.length - 1)} · #{first.seq}–#{last.seq}</span>
                </span>
              </div>;
            }
            const event = row.event, lane = laneOf(event), linked = links.has(event.seq);
            const budget = Object.entries(event.budget_charged ?? {}).filter(([, n]) => n > 0);
            const target = eventTarget(event, names, resolved);
            return <div className={"tv-event" + (linked ? " tv-linked" : "")} role="row" aria-rowindex={i + 1}
              key={row.key} style={style} aria-selected={selectionKey === event.seq} data-seq={event.seq} data-status={event.status}
              data-error={failed(event)} data-lane={lane} data-dim={!keep(event) && !linked} tabIndex={i === focusIndex ? 0 : -1}
              onFocus={() => setCursor(event.seq)} onClick={() => {setCursor(event.seq); select(event.seq);}}>
              <span role="rowheader">#{event.seq}</span>
              <time role="gridcell" dateTime={event.at} title={event.at + " · " + relativeTime(event.at, now)}>{timeText(event.at)}</time>
              <span role="gridcell" className="tv-lane" aria-label={lane} title={lane}>{icons[lane]}</span>
              <span role="gridcell" title={toolOf(event)}>{toolOf(event)}</span>
              <span role="gridcell" title={target}>{lane === "Control" && event.budget_change ?
                event.budget_change.dimension + " " + (event.budget_change.from ?? "unlimited") + " → " + (event.budget_change.to ?? "unlimited") : target}</span>
              <span role="gridcell" className="tv-event-status" title={event.error_code ?? undefined}>{event.status}{event.error_code && " " + event.error_code}</span>
              <span role="gridcell" className="tv-event-flags" title={[
                ...budget.map(([dim, n]) => dim + " " + count(n)), row.capChanged ? "Capabilities changed" : "",
                event.proposal_id ? shortId(event.proposal_id) : "",
              ].filter(Boolean).join(" · ")}>
                {row.capChanged && <span className="tv-flag">caps changed</span>}
                {isReplay(event) ? <span className="tv-flag tv-replay">↩ {event.replay_of === null ? "replayed" : "#" + event.replay_of}</span> :
                  isCharged(event) && <span className="tv-flag tv-charge">{budget.length ? budget[0][0] + " " + count(budget[0][1]) : "charged"}</span>}
                {event.existing && <span className="tv-flag">existing</span>}
                {mix && event.via && <span className="tv-flag">{event.via}</span>}
              </span>
            </div>;
          })}
        </div>
      </div>
      {unseen > 0 && <button className="tv-new" type="button" onClick={() => {setFollowing(true); tail();}}>{count(unseen)} new events ↓</button>}
      {budgetRow >= 0 && <button className="tv-budget-marker" type="button" title="Jump to first budget event"
        style={{top: Math.min(96, 4 + budgetRow / rows.length * 92) + "%"}}
        aria-label="Jump to first budget event" onClick={() => select(events.find(e => e.status === 429)!.seq)}/>}
      <span className="visually-hidden" aria-live="polite">{announcement}</span>
    </div>
  </>;
}
