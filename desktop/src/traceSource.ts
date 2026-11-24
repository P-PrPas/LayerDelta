import {ApiError, connect, get, post} from "./api";
import type {Envelope, Selector} from "./tensorSource";

// Session viewer fields follow docs/design/session-trace-viewer.md §8.
export type Dimension = "values" | "scan" | "applies" | "blob_bytes" | "evals" | "eval_samples";
export type Budget = {limit: number | null; used: number; remaining: number | null};
export const dimensions: Dimension[] = ["values", "scan", "applies", "blob_bytes", "evals", "eval_samples"];
export type TraceEvent = {
  event_id: number; session: string; seq: number; at: string; route: string; status: number;
  tool: string | null; via: "mcp" | "http" | null; request_id: string | null; error_code: string | null; field: string | null;
  checkpoint_id: string | null; other_checkpoint_id: string | null; selection_id: string | null;
  selections: string[]; ops: number | string[] | null; recipe_hash: string | null; child_id: string | null; run_id: string | null;
  idempotency_key: string | null; replayed: boolean | null; replay_of: number | null; existing: boolean | null;
  charged: number; budget_charged: Partial<Record<Dimension, number>> | null; capabilities: string[];
  links: {plan_seqs: number[]; selection_seqs: number[]; child_id: string | null; proposal_seq?: number; apply_seq?: number} | null;
  evaluation_id?: string; probe_id?: string; dataset_id?: string; split?: string; split_hash?: string;
  fingerprint?: string; baseline_id?: string | null; baseline_cached?: boolean; eval_samples?: number;
  intervention_id?: string; units?: number; page_offset?: number; page_limit?: number;
  control: {principal: "human"; action: "revoke" | "budget_change" | "approve" | "reject";
    dimension?: Dimension; from?: number | null; to?: number | null; proposal_id?: string; child_id?: string} | null;
  duration_ms: number | null; principal: string | null; target_session: string | null;
  budget_change: {dimension: Dimension; from: number | null; to: number | null} | null; proposal_id: string | null;
};
export type TraceSummary = {events: number; applies: number; replays: number; charged_runs: number;
  errors: number | null; by_status: Record<string, number> | null; budget: Partial<Record<Dimension, Budget>> | null};
export type TracePage = {events: TraceEvent[]; summary: TraceSummary; next_after: number | null};
export type SessionRow = {
  session: string; kind: "agent" | "human"; created_at: string | null; last_at: string | null; last_status: number | null;
  capabilities: string[]; events: number; applies: number; charged_runs: number;
  egress_class: 0 | 1 | 2 | null; revoked: string | null; via_mix: Partial<Record<"http" | "mcp", number>> | null;
  current?: boolean;
};
export type SelectionRef = {checkpoint_id: string; selection_id: string};
export type ResolvedSelection = SelectionRef & {selector: Selector | {group: string; units?: number[] | {start: number; stop: number}}};
export type TraceRecipe = {recipe_hash: string; hypothesis: string | null; ops: number | null};
export type Proposal = {proposal_id: string; seq: number | null; checkpoint_id: string; recipe_hash: string; selections: string[]; created_at: string};
type PlanOp = {op: string; tensor: string; selected_elems: number};
export type ProposalPreview = {ops: PlanOp[]; warnings: string[]; refusals: string[]; side_effects: Record<string, string | number>[]};
export interface TraceSource {
  page(input: {session?: string; after: number; limit: number}, signal: AbortSignal): Promise<TracePage>;
  sessions: ((signal: AbortSignal) => Promise<SessionRow[]>) | null;
  checkpointName(id: string, signal: AbortSignal): Promise<string | null>;
  resolve: ((ids: SelectionRef[], signal: AbortSignal) => Promise<ResolvedSelection[]>) | null;
  recipe: ((hash: string, signal: AbortSignal) => Promise<TraceRecipe | null>) | null;
  approvals: ((session: string, signal: AbortSignal) => Promise<Proposal[] | null>) | null;
  controls: {
    revoke(session: string): Promise<void>;
    budget(session: string, dimension: Dimension, limit: number | null): Promise<void>;
    preview(proposal: Proposal): Promise<ProposalPreview>;
    decide(id: string, decision: "approve" | "reject"): Promise<void>;
  } | null;
  now(): number;
}
export const pageSize = 500;

// Keep the projection on the ID-only trace schema, including when future routes add fields.
export function traceEvent(raw: Partial<TraceEvent> & Pick<TraceEvent, "event_id" | "session" | "seq" | "at" | "route" | "status">): TraceEvent {
  const control = raw.control;
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
    links: raw.links ? {plan_seqs: raw.links.plan_seqs ?? [], selection_seqs: raw.links.selection_seqs ?? [],
      child_id: raw.links.child_id ?? null, proposal_seq: raw.links.proposal_seq, apply_seq: raw.links.apply_seq} : null,
    evaluation_id: raw.evaluation_id, probe_id: raw.probe_id, dataset_id: raw.dataset_id, split: raw.split, split_hash: raw.split_hash,
    fingerprint: raw.fingerprint, baseline_id: raw.baseline_id, baseline_cached: raw.baseline_cached, eval_samples: raw.eval_samples,
    intervention_id: raw.intervention_id, units: raw.units, page_offset: raw.page_offset, page_limit: raw.page_limit,
    control: control ? {principal: control.principal, action: control.action, dimension: control.dimension, from: control.from, to: control.to,
      proposal_id: control.proposal_id, child_id: control.child_id} : null,
    duration_ms: raw.duration_ms ?? null, principal: raw.principal ?? null, target_session: raw.target_session ?? null,
    budget_change: raw.budget_change ? {dimension: raw.budget_change.dimension, from: raw.budget_change.from, to: raw.budget_change.to} : null,
    proposal_id: raw.proposal_id ?? null,
  };
}

function parseFrozenRecipe(text: string) {
  // Python includes numeric spelling in the canonical digest (2.0 differs from 2).
  return JSON.parse(text, (_key: string, value: unknown, context?: {source: string}) => {
    if (typeof value !== "number") return value;
    const rawJSON = (JSON as JSON & {rawJSON?: (source: string) => object}).rawJSON;
    if (!rawJSON || !context) throw new Error("Update WebView2 to preview frozen recipes.");
    return rawJSON(context.source);
  });
}

async function frozenRecipe(hash: string, signal?: AbortSignal, preserveNumbers = false) {
  await connect();
  const {result} = await get<{result: {recipe: {recipe_hash: string; hypothesis?: string; program: unknown[]}}}>
    ("/recipes/by-hash/" + encodeURIComponent(hash), signal, preserveNumbers ? parseFrozenRecipe : undefined);
  if (result.recipe.recipe_hash !== hash) throw new Error("The frozen recipe does not match this proposal.");
  return result.recipe;
}

export const liveTraceSource: TraceSource = {
  async page({session, after, limit}, signal) {
    await connect();
    const params = new URLSearchParams({after: String(after), limit: String(limit), ...(session ? {session} : {})});
    const {result} = await get<Envelope<TracePage>>("/trace?" + params, signal);
    return {events: result.events.map(traceEvent), next_after: result.next_after,
      summary: {...result.summary, errors: result.summary.errors ?? null, by_status: result.summary.by_status ?? null, budget: result.summary.budget ?? null}};
  },
  async sessions(signal) {
    await connect();
    type Row = Omit<SessionRow, "revoked" | "via_mix"> & {revoked_at: string | null; via: SessionRow["via_mix"]};
    const {result} = await get<{result: {sessions: Row[]; current_agent_session: string | null}}>("/trace/sessions", signal);
    return result.sessions.map(({revoked_at, via, ...row}) => ({...row, revoked: revoked_at, via_mix: via,
      current: row.kind === "agent" && row.session === result.current_agent_session}));
  },
  async checkpointName(id, signal) {
    await connect();
    return (await get<{display_name?: string}>("/checkpoints/" + encodeURIComponent(id), signal)).display_name ?? null;
  },
  async resolve(refs, signal) {
    await connect();
    const ids = [...new Set(refs.map(ref => ref.selection_id))];
    const {result} = await post<{result: {selections: Record<string, Omit<ResolvedSelection, "selection_id">[]>}}>
      ("/selections/resolve", {ids}, false, signal);
    return refs.flatMap(ref => {
      const match = result.selections[ref.selection_id]?.find(row => row.checkpoint_id === ref.checkpoint_id);
      return match ? [{...ref, selector: match.selector}] : [];
    });
  },
  async recipe(hash, signal) {
    try {
      const recipe = await frozenRecipe(hash, signal);
      return {recipe_hash: hash, hypothesis: recipe.hypothesis ?? null, ops: recipe.program.length};
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) return null;
      throw error;
    }
  },
  async approvals(session, signal) {
    await connect();
    const params = new URLSearchParams({session, status: "pending"});
    const {result} = await get<{result: {approval_gate: boolean; proposals: Proposal[]}}>("/proposals?" + params, signal);
    return result.approval_gate ? result.proposals : null;
  },
  controls: {
    async revoke(session) {
      await connect();
      await post("/agent-sessions/" + encodeURIComponent(session) + "/revoke", {});
    },
    async budget(session, dimension, limit) {
      await connect();
      await post("/agent-sessions/" + encodeURIComponent(session) + "/budget", {dimension, limit});
    },
    async preview(proposal) {
      const recipe = await frozenRecipe(proposal.recipe_hash, undefined, true);
      const plan = await post<Envelope<{ops: PlanOp[]; warnings?: {message: string}[]; noop: boolean; child_bytes: number;
        changed_elems_total: number}>>("/checkpoints/" + encodeURIComponent(proposal.checkpoint_id) + "/plan", {recipe});
      return {ops: plan.result.ops, warnings: plan.result.warnings?.map(warning => warning.message) ?? [], refusals: [], side_effects: plan.result.noop ? [] :
        [{kind: "checkpoint_may_be_created", bytes: plan.result.child_bytes, changed_values: plan.result.changed_elems_total}]};
    },
    async decide(id, decision) {
      await connect();
      await post("/proposals/" + encodeURIComponent(id) + "/" + decision, {});
    },
  },
  now: Date.now,
};
