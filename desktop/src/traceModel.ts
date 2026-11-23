import type {ResolvedSelection, SessionRow, TraceEvent} from "./traceSource";

export type Lane = "Read" | "Plan" | "Write" | "Run" | "Meta" | "Control";
const lanes: Record<string, Lane> = {
  list_checkpoints: "Read", get_checkpoint: "Read", describe: "Read", graph: "Read", graph_node: "Read", groups: "Read",
  stats: "Read", query: "Read", diff: "Read", read_tensor: "Read", plan: "Plan", validate: "Plan",
  apply: "Write", upload_blob: "Write", propose: "Run", run: "Run", status: "Run", compare: "Run", changes: "Run",
  cancel: "Run", evaluate: "Run", revoke: "Control", budget: "Control", approve: "Control", reject: "Control",
};
export function toolOf(event: TraceEvent) {
  const route = event.route.split(" ").at(-1) ?? "";
  if (route === "/checkpoints") return event.route.startsWith("GET ") ? "list_checkpoints" : event.tool ?? route;
  if (route === "/checkpoints/{checkpoint_id}") return "get_checkpoint";
  if (route.endsWith("/graph/nodes/{node_id}")) return "graph_node";
  if (route === "/blobs" && event.route.startsWith("POST ")) return "upload_blob";
  if (route === "/runs") return event.route.startsWith("POST ") ? "run" : "status";
  if (route.startsWith("/runs/{run_id}")) {
    if (route.endsWith("/comparison")) return "compare";
    if (route.endsWith("/changes")) return "changes";
    if (route.endsWith("/cancel")) return "cancel";
    if (route === "/runs/{run_id}") return "status";
  }
  if (event.tool === "agent_session" || route === "/agent-session") return "session_info";
  return event.tool ?? (route.split("/").at(-1)?.replaceAll("-", "_") || "unknown");
}
export const laneOf = (event: TraceEvent): Lane => lanes[toolOf(event)] ?? "Meta";
export const failed = (event: TraceEvent) => event.status >= 400;
export const isReplay = (event: TraceEvent) => event.replayed === true || event.replay_of !== null;
export const isCharged = (event: TraceEvent) => event.charged > 0 || Object.values(event.budget_charged ?? {}).some(n => n > 0);
export const isApply = (event: TraceEvent) => toolOf(event) === "apply" && event.status >= 200 && event.status < 300 && event.status !== 202 && !isReplay(event);
export const liveSession = (session: SessionRow, now: number) => !!session.last_at && !session.revoked && now - Date.parse(session.last_at) < 60_000;
export const selectionKey = (checkpoint: string, selection: string) => JSON.stringify([checkpoint, selection]);
export function shortId(id: string) {
  if (id.length <= 12) return id;
  if (id.startsWith("agent-")) return id.slice(0, 10) + "…";
  if (id.startsWith("sel_")) return id.slice(0, 8) + "…";
  return id.slice(0, 8) + "…";
}

export function eventTarget(event: TraceEvent, names: Map<string, string>, resolved: Map<string, ResolvedSelection>) {
  const name = (id: string) => names.get(id) ?? shortId(id);
  const selection = event.checkpoint_id && event.selection_id ? resolved.get(selectionKey(event.checkpoint_id, event.selection_id))?.selector : undefined;
  const detail = selection ? "tensor" in selection ? selection.tensor + (selection.index && Object.keys(selection.index).length ?
    " " + Object.entries(selection.index).map(([axis, index]) => "axis " + axis + " " + JSON.stringify(index)).join(", ") : "") :
    shortId(selection.group) : event.selection_id ? shortId(event.selection_id) : "";
  return [event.checkpoint_id && name(event.checkpoint_id), detail,
    event.other_checkpoint_id && "→ " + name(event.other_checkpoint_id)].filter(Boolean).join(" · ") || "—";
}

export function sessionsFromEvents(events: TraceEvent[]): SessionRow[] {
  const sessions = new Map<string, SessionRow>();
  for (const event of events) {
    let row = sessions.get(event.session);
    if (!row) {
      row = {session: event.session, kind: event.session === "human" ? "human" : "agent",
        created_at: event.seq === 1 ? event.at : null, last_at: null, last_status: null, capabilities: [],
        events: 0, applies: 0, charged_runs: 0, egress_class: null, revoked: null, via_mix: null};
      sessions.set(event.session, row);
    }
    row.last_at = event.at; row.last_status = event.status; row.capabilities = event.capabilities;
    row.events++; row.applies += Number(isApply(event)); row.charged_runs += event.charged;
    if (event.via) {row.via_mix ??= {}; row.via_mix[event.via] = (row.via_mix[event.via] ?? 0) + 1;}
  }
  return [...sessions.values()].sort((a, b) => (b.last_at ?? "").localeCompare(a.last_at ?? ""));
}

export type TimelineRow = {key: string; event: TraceEvent; capChanged?: boolean} |
  {key: string; burst: TraceEvent[]; expanded: boolean} | {key: string; missing: number};
export function timelineRows(events: TraceEvent[], links: Set<number>, open: Set<number>, keep: (event: TraceEvent) => boolean, hide: boolean, exposed = links): TimelineRow[] {
  const rows: TimelineRow[] = [];
  let burst: TraceEvent[] = [];
  let caps: string | undefined;
  const loaded = new Set(events.map(e => e.seq));
  const missing = [...links].filter(seq => !loaded.has(seq)).sort((a, b) => a - b);
  function flush() {
    if (burst.length >= 3) {
      const expanded = open.has(burst[0].seq);
      rows.push({key: "burst:" + burst[0].seq, burst, expanded});
      if (expanded) for (const event of burst) rows.push({key: String(event.seq), event});
    } else for (const event of burst) rows.push({key: String(event.seq), event});
    burst = [];
  }
  for (const event of events) {
    while (missing[0] < event.seq) {
      flush(); const seq = missing.shift()!; rows.push({key: "missing:" + seq, missing: seq});
    }
    const current = [...event.capabilities].sort().join(" ");
    const capChanged = caps !== undefined && caps !== current;
    caps = current;
    if (hide && !keep(event) && !exposed.has(event.seq)) {flush(); continue;}
    const read = laneOf(event) === "Read" && event.status >= 200 && event.status < 300 && !!event.checkpoint_id;
    if (!read || exposed.has(event.seq) || capChanged) {
      flush(); rows.push({key: String(event.seq), event, capChanged}); continue;
    }
    if (burst.length && (burst[0].checkpoint_id !== event.checkpoint_id || keep(burst[0]) !== keep(event))) flush();
    burst.push(event);
  }
  flush();
  for (const seq of missing) rows.push({key: "missing:" + seq, missing: seq});
  return rows;
}

export function statusMeaning(event: TraceEvent, egress: SessionRow["egress_class"]) {
  if (event.status === 403) {
    const valued = toolOf(event) === "read_tensor" || event.error_code === "egress_denied" || event.error_code === "egress_forbidden";
    if (egress === 1 && valued) return "Blocked. Values are not released to this session (egress class 1).";
    const lane = laneOf(event);
    const needs = lane === "Write" ? ["weights:write", "weights:read"] :
      ["read_tensor", "stats", "diff", "query", "plan", "validate"].includes(toolOf(event)) ? ["weights:read"] :
        lane === "Read" ? ["model:read"] : [];
    const missing = needs.find(cap => !event.capabilities.includes(cap));
    return missing ? "Blocked. The token lacks " + missing + "." : "Blocked. The server refused this request.";
  }
  if (event.status === 429) {
    const dimension = ["values", "scan", "applies", "blob_bytes"].find(key => event.field === key || event.field === "budget." + key);
    return dimension ? "The " + dimension + " budget is used up." : "Budget used up.";
  }
  if (event.status === 401 && event.error_code === "session_revoked") return "The session was revoked. The server refused this request.";
  if (event.error_code === "idempotency_conflict") return "The same key was reused for a different request.";
  if (event.status === 422) return "Request error.";
  if (event.status === 404) return "The requested item was not found.";
  if (event.status === 202 && event.proposal_id) return "This apply is awaiting approval.";
  if (isReplay(event)) return "The earlier result was returned with no new charge.";
  return null;
}
