import {connect, get} from "./api";
import type {Envelope, Selector} from "./tensorSource";

// Session viewer fields follow docs/design/session-trace-viewer.md §8.
export type Dimension = "values" | "scan" | "applies" | "blob_bytes";
export type Budget = {limit: number | null; used: number; remaining: number | null};
export const dimensions: Dimension[] = ["values", "scan", "applies", "blob_bytes"];
export type TraceEvent = {
  event_id: number; session: string; seq: number; at: string; route: string; status: number;
  tool: string | null; via: "mcp" | "http" | null; request_id: string | null; error_code: string | null; field: string | null;
  checkpoint_id: string | null; other_checkpoint_id: string | null; selection_id: string | null;
  selections: string[]; ops: number | string[] | null; recipe_hash: string | null; child_id: string | null; run_id: string | null;
  idempotency_key: string | null; replayed: boolean | null; replay_of: number | null; existing: boolean | null;
  charged: number; budget_charged: Partial<Record<Dimension, number>> | null; capabilities: string[];
  links: {plan_seqs: number[]; selection_seqs: number[]; child_id: string | null} | null;
  duration_ms: number | null; principal: string | null; target_session: string | null;
  budget_change: {dimension: Dimension; from: number | null; to: number | null} | null; proposal_id: string | null;
};
export type TraceSummary = {events: number; applies: number; replays: number; charged_runs: number;
  errors: number | null; by_status: Record<string, number> | null; budget: Partial<Record<Dimension, Budget>> | null};
export type TracePage = {events: TraceEvent[]; summary: TraceSummary; next_after: number | null};
export type SessionRow = {
  session: string; kind: "agent" | "human"; created_at: string | null; last_at: string | null; last_status: number | null;
  capabilities: string[]; events: number; applies: number; charged_runs: number;
  egress_class: 1 | 2 | null; revoked: string | null; via_mix: Partial<Record<"http" | "mcp", number>> | null;
};
export type SelectionRef = {checkpoint_id: string; selection_id: string};
export type ResolvedSelection = SelectionRef & {selector: Selector | {group: string; units?: number[] | {start: number; stop: number}}};
export type TraceRecipe = {recipe_hash: string; hypothesis: string | null; ops: number | null};
export type Proposal = {proposal_id: string; seq: number; checkpoint_id: string; recipe_hash: string; selections: string[]; created_at: string};
export type ProposalPreview = {warnings: string[]; refusals: string[]; side_effects: Record<string, string | number>[]};
export interface TraceSource {
  page(input: {session?: string; after: number; limit: number}, signal: AbortSignal): Promise<TracePage>;
  sessions: ((signal: AbortSignal) => Promise<SessionRow[]>) | null;
  checkpointName(id: string, signal: AbortSignal): Promise<string | null>;
  resolve: ((ids: SelectionRef[], signal: AbortSignal) => Promise<ResolvedSelection[]>) | null;
  recipe: ((hash: string, signal: AbortSignal) => Promise<TraceRecipe | null>) | null;
  approvals: ((session: string, signal: AbortSignal) => Promise<Proposal[] | null>) | null;
  controls: {
    revoke(session: string): Promise<void>;
    budget(session: string, dimension: Dimension, limit: number): Promise<void>;
    preview(id: string): Promise<ProposalPreview>;
    decide(id: string, decision: "approve" | "reject"): Promise<void>;
  } | null;
  now(): number;
}
export const pageSize = 500;

// Keep the projection on the ID-only trace schema, including when future routes add fields.
export function traceEvent(raw: Partial<TraceEvent> & Pick<TraceEvent, "event_id" | "session" | "seq" | "at" | "route" | "status">): TraceEvent {
  return {
    event_id: raw.event_id, session: raw.session, seq: raw.seq, at: raw.at, route: raw.route, status: raw.status,
    tool: raw.tool ?? null, via: raw.via ?? null, request_id: raw.request_id ?? null, error_code: raw.error_code ?? null, field: raw.field ?? null,
    checkpoint_id: raw.checkpoint_id ?? null, other_checkpoint_id: raw.other_checkpoint_id ?? null,
    selection_id: raw.selection_id ?? null, selections: raw.selections ?? [], ops: raw.ops ?? null, recipe_hash: raw.recipe_hash ?? null,
    child_id: raw.child_id ?? null, run_id: raw.run_id ?? null, idempotency_key: raw.idempotency_key ?? null, replayed: raw.replayed ?? null,
    replay_of: raw.replay_of ?? null, existing: raw.existing ?? null, charged: raw.charged ?? 0,
    budget_charged: raw.budget_charged ? Object.fromEntries(dimensions.flatMap(dim =>
      typeof raw.budget_charged?.[dim] === "number" ? [[dim, raw.budget_charged[dim]]] : [])) : null,
    capabilities: raw.capabilities ?? [],
    links: raw.links ? {plan_seqs: raw.links.plan_seqs ?? [], selection_seqs: raw.links.selection_seqs ?? [], child_id: raw.links.child_id ?? null} : null,
    duration_ms: raw.duration_ms ?? null, principal: raw.principal ?? null, target_session: raw.target_session ?? null,
    budget_change: raw.budget_change ? {dimension: raw.budget_change.dimension, from: raw.budget_change.from, to: raw.budget_change.to} : null,
    proposal_id: raw.proposal_id ?? null,
  };
}

export const liveTraceSource: TraceSource = {
  async page({session, after, limit}, signal) {
    await connect();
    const params = new URLSearchParams({after: String(after), limit: String(limit), ...(session ? {session} : {})});
    const {result} = await get<Envelope<TracePage>>("/trace?" + params, signal);
    return {events: result.events.map(traceEvent), next_after: result.next_after,
      summary: {...result.summary, errors: result.summary.errors ?? null, by_status: result.summary.by_status ?? null, budget: result.summary.budget ?? null}};
  },
  sessions: null,
  async checkpointName(id, signal) {
    await connect();
    return (await get<{display_name?: string}>("/checkpoints/" + encodeURIComponent(id), signal)).display_name ?? null;
  },
  resolve: null, recipe: null, approvals: null, controls: null, now: Date.now,
};
