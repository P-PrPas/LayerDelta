import type {Index} from "./tensorSource";
import type {ResolvedSelection, SessionRow, TraceEvent} from "./traceSource";

export type Lane = "Read" | "Plan" | "Write" | "Measure" | "Meta" | "Control";
const lanes: Record<string, Lane> = {
  list_checkpoints: "Read", get_checkpoint: "Read", describe: "Read", graph: "Read", graph_node: "Read", groups: "Read",
  stats: "Read", query: "Read", diff: "Read", read_tensor: "Read", plan: "Plan", validate: "Plan",
  apply: "Write", upload_blob: "Write", propose: "Measure", run: "Measure", status: "Measure", compare: "Measure", changes: "Measure",
  cancel: "Measure", evaluate: "Measure", get_evaluation: "Measure", evaluation_changes: "Measure",
  probe: "Measure", get_probe: "Measure", final_evaluate: "Measure", revoke: "Control", budget: "Control", approve: "Control", reject: "Control",
  revoke_session: "Control", change_budget: "Control", approve_proposal: "Control", reject_proposal: "Control",
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
export const isApply = (event: TraceEvent) => event.control?.action === "approve" ||
  toolOf(event) === "apply" && event.status >= 200 && event.status < 300 && event.status !== 202 && !isReplay(event);
export const liveSession = (session: SessionRow, now: number) => !!session.last_at && !session.revoked && now - Date.parse(session.last_at) < 60_000;
export const selectionKey = (checkpoint: string, selection: string) => JSON.stringify([checkpoint, selection]);
export function shortId(id: string) {
  if (id.length <= 12) return id;
  if (id.startsWith("agent-")) return id.slice(0, 10) + "…";
  if (id.startsWith("sel_")) return id.slice(0, 8) + "…";
  return id.slice(0, 8) + "…";
}

function compactRange(start: number, stop: number) {
  return stop - start === 1 ? String(start) : start + "–" + (stop - 1);
}
function axisText(index: Index | undefined) {
  if (!index) return ":";
  if (!Array.isArray(index)) return index.start + ":" + index.stop;
  if (index.length === 1) {
    const part = index[0];
    return typeof part === "number" ? String(part) : part.start + ":" + part.stop;
  }
  const pieces: string[] = [];
  let start: number | undefined;
  let end: number | undefined;
  for (const part of index) {
    const next = typeof part === "number" ? part : part.start;
    if (end !== undefined && next !== end) {
      pieces.push(compactRange(start!, end));
      start = undefined;
    }
    start ??= next;
    end = typeof part === "number" ? part + 1 : part.stop;
  }
  if (end !== undefined) pieces.push(compactRange(start!, end));
  return "[" + pieces.join(", ") + "]";
}
export function tensorTarget(selector: Extract<ResolvedSelection["selector"], {tensor: string}>) {
  const {tensor, index = {}, shape} = selector;
  const axes = Object.keys(index);
  if (!shape && !axes.length) return tensor;
  const rank = shape?.length ?? Math.max(...axes.map(Number)) + 1;
  const parts = Array.from({length: rank}, (_, axis) => axisText(index[axis]));
  if (!shape) parts.push("…");
  return tensor + " [" + parts.join(", ") + "]";
}

export function eventTarget(event: TraceEvent, names: Map<string, string>, resolved: Map<string, ResolvedSelection>) {
  const name = (id: string) => names.get(id) ?? shortId(id);
  const selectionId = event.selection_id ?? event.selections[0];
  const selection = event.checkpoint_id && selectionId ? resolved.get(selectionKey(event.checkpoint_id, selectionId))?.selector : undefined;
  const detail = selection ? "tensor" in selection ? tensorTarget(selection) :
    shortId(selection.group) : selectionId ? shortId(selectionId) : "";
  return [event.checkpoint_id && name(event.checkpoint_id), detail,
    event.control?.child_id ? "→ " + name(event.control.child_id) : event.other_checkpoint_id && "→ " + name(event.other_checkpoint_id)].filter(Boolean).join(" · ") || "—";
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

export function measureResultKey(event: TraceEvent) {
  const tool = toolOf(event);
  if (tool === "get_probe") return event.probe_id;
  if (tool === "get_evaluation" || tool === "evaluation_changes") return event.evaluation_id;
  return undefined;
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
    const capChanged = !event.control && caps !== undefined && caps !== current;
    if (!event.control) caps = current;
    if (hide && !keep(event) && !exposed.has(event.seq)) {flush(); continue;}
    const result = measureResultKey(event);
    const read = event.status >= 200 && event.status < 300 && (laneOf(event) === "Read" && !!event.checkpoint_id || result !== undefined);
    if (!read || exposed.has(event.seq) || capChanged) {
      flush(); rows.push({key: String(event.seq), event, capChanged}); continue;
    }
    if (burst.length && (measureResultKey(burst[0]) !== result || burst[0].checkpoint_id !== event.checkpoint_id || keep(burst[0]) !== keep(event))) flush();
    burst.push(event);
  }
  flush();
  for (const seq of missing) rows.push({key: "missing:" + seq, missing: seq});
  return rows;
}

export function statusMeaning(event: TraceEvent, egress: SessionRow["egress_class"]) {
  const tool = toolOf(event);
  if (event.status === 403 && event.split === "holdout") return "Blocked. The holdout split is reserved for human final evaluation. Nothing was run.";
  if (event.status === 403 && (tool === "evaluate" || tool === "probe") && !event.capabilities.includes("eval:run")) return "Blocked. This token lacks eval:run. Nothing was run.";
  if (event.status === 403 && tool === "probe" && !event.capabilities.includes("weights:read")) return "Blocked. Probes need weights:read. Nothing was run.";
  if (event.status === 429 && /^(budget\.)?evals$/.test(event.field ?? "")) return "The evaluation budget is used up. Nothing was run.";
  if (event.status === 429 && /^(budget\.)?eval_samples$/.test(event.field ?? "")) return "The sample budget would be exceeded. Nothing was run.";
  if (event.error_code === "executable_mismatch") return "The checkpoint does not fit its executable. Nothing was run.";
  if (event.error_code === "probe_unsupported") return "This node cannot be probed with this model’s provider.";
  if (event.error_code === "no_baseline") return "This checkpoint has no parent and no baseline was given.";
  if (event.error_code === "aggregate_too_small" && tool === "probe") return "Too few samples for this probe under egress class 1. Nothing was released.";
  if (tool === "evaluate" && isReplay(event)) return "The earlier evaluation was returned. No new charge.";
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
  if (event.status === 202 && event.proposal_id) return "This apply entered the approval queue.";
  if (event.control?.action === "approve") return "A human approved the proposal.";
  if (event.control?.action === "reject") return "A human rejected the proposal.";
  if (event.control?.action === "budget_change") return "A human changed the session budget.";
  if (event.control?.action === "revoke") return "A human revoked the session.";
  if (isReplay(event)) return "The earlier result was returned with no new charge.";
  return null;
}
