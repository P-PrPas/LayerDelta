import {dimensions, traceEvent, type Dimension, type Proposal, type ResolvedSelection, type TraceEvent, type TraceSource} from "./traceSource";
import {failed, isApply, isReplay, sessionsFromEvents} from "./traceModel";

const epoch = Date.UTC(2026, 0, 15, 12, 0, 0);
const agent = "agent-3f9c0e21b7d4";
const archive = "agent-a71b85413db5";
const checkpoint = "84e3".padEnd(64, "a");
const child = "9fac".padEnd(64, "c");
const recipe = "7be2".padEnd(64, "f");
const selection = "sel_a563".padEnd(36, "b");
const groupSelection = "sel_04b2".padEnd(36, "d");
const proposal = "p_51d200000000";
const caps = ["model:read", "weights:read", "weights:write"];
const stamp = (ms: number) => new Date(ms).toISOString();

export function createTraceFixture(clock = Date.now): TraceSource {
  const started = clock();
  const now = () => epoch + 60_000 + clock() - started;
  const events: TraceEvent[] = [];
  const revoked = new Map<string, string>();
  const pending = new Map<string, Proposal>([[proposal, {proposal_id: proposal, seq: 46, checkpoint_id: checkpoint,
    recipe_hash: recipe, selections: [selection], created_at: stamp(epoch + 58_000)}]]);
  const limits: Record<Dimension, number | null> = {values: 100_000, scan: 10_000_000, applies: 10, blob_bytes: null};
  const used = {values: 0, scan: 8_600_000, applies: 1, blob_bytes: 0};
  function add(session: string, tool: string, at: number, fields: Partial<TraceEvent> = {}) {
    const seq = events.filter(e => e.session === session).length + 1;
    const event = traceEvent({event_id: events.length + 1, session, seq, at: stamp(at), tool,
      route: "POST /checkpoints/{checkpoint_id}/" + tool.replaceAll("_", "-"), status: 200, checkpoint_id: checkpoint,
      request_id: (events.length + 1).toString(16).padStart(32, "0"), via: seq % 9 === 0 ? "http" : "mcp",
      capabilities: session === archive ? caps.slice(0, 2) : caps, duration_ms: 4 + seq % 31, ...fields});
    events.push(event);
    return event;
  }

  for (let i = 0; i < 510; i++) add(archive, i % 17 === 0 ? "describe" : "stats", epoch - 86400_000 + i * 1000,
    {selection_id: i % 17 === 0 ? null : selection, ...(i === 509 ? {status: 429, error_code: "budget_exhausted", field: "scan"} : {})});
  add("human", "get_checkpoint", epoch - 20_000, {route: "GET /checkpoints/{checkpoint_id}", via: "http"});
  for (let seq = 1; seq <= 46; seq++) {
    const at = epoch + seq * 1250;
    if (seq === 40) {
      add(agent, "apply", at, {status: 201, recipe_hash: recipe, child_id: child, ops: 2, selections: [selection],
        links: {plan_seqs: [37, 39], selection_seqs: [9, 22, 27], child_id: child}, charged: 1, budget_charged: {applies: 1}, existing: false,
        idempotency_key: "abc".padEnd(32, "9"), replayed: false});
    } else if (seq === 44) {
      add(agent, "apply", at, {recipe_hash: recipe, child_id: child, replayed: true, replay_of: 40,
        idempotency_key: "abc".padEnd(32, "9")});
    } else if (seq === 46) {
      add(agent, "apply", at, {status: 202, recipe_hash: recipe, proposal_id: proposal, selections: [selection]});
    } else if (seq === 37 || seq === 39) {
      add(agent, "plan", at, {recipe_hash: recipe, ops: 2, selections: [selection]});
    } else if (seq === 42) {
      add(agent, "read_tensor", at, {status: 403, error_code: "forbidden", selection_id: selection});
    } else if (seq === 43) {
      add(agent, "stats", at, {status: 429, error_code: "budget_exhausted", field: "budget.scan", selection_id: selection});
    } else if (seq === 45) {
      add(agent, "validate", at, {status: 422, error_code: "invalid_argument"});
    } else if (seq === 36) {
      add(agent, "query", at, {selection_id: groupSelection});
    } else add(agent, seq % 11 === 0 ? "describe" : "stats", at, {selection_id: selection, budget_charged: seq % 3 === 0 ? {scan: 4096} : null});
  }
  const names = new Map([[checkpoint, "Reference fixture"], [child, "Agent child"]]);
  function control(session: string, tool: string, fields: Partial<TraceEvent> = {}) {
    add(session, tool, now(), {principal: "human", target_session: session, checkpoint_id: null, via: "http",
      route: "POST /agent-sessions/{session}/" + tool, ...fields});
  }
  return {
    now,
    async page({session, after, limit}, signal) {
      signal.throwIfAborted();
      const matched = session ? events.filter(e => e.session === session) : events;
      const next = matched.filter(e => e.event_id > after).slice(0, limit + 1);
      const page = next.slice(0, limit);
      const counts: Record<string, number> = {};
      for (const e of matched) counts[e.status] = (counts[e.status] ?? 0) + 1;
      return {events: page, next_after: next.length > limit ? page.at(-1)!.event_id : null,
        summary: {events: matched.length, applies: matched.filter(isApply).length, replays: matched.filter(isReplay).length,
          charged_runs: matched.reduce((n, e) => n + e.charged, 0), errors: matched.filter(failed).length, by_status: counts,
          budget: session === agent ? Object.fromEntries(dimensions.map(d => {
            const limit = limits[d];
            return [d, {limit, used: used[d], remaining: limit === null ? null : Math.max(0, limit - used[d])}];
          })) : null}};
    },
    async sessions(signal) {
      signal.throwIfAborted();
      return sessionsFromEvents(events).map(row => ({...row, egress_class: row.kind === "agent" ? 1 : null, revoked: revoked.get(row.session) ?? null}));
    },
    async checkpointName(id) {return names.get(id) ?? null;},
    async resolve(ids) {
      return ids.flatMap<ResolvedSelection>(id => id.selection_id === selection ? [{...id, selector: {tensor: "classifier.weight", shape: [10, 1176], index: {"0": [{start: 2, stop: 4}]}}}] :
        id.selection_id === groupSelection ? [{...id, selector: {group: "g_e16313d7eb141cdf2d7ad2bc", units: [1]}}] : []);
    },
    async recipe(hash) {
      return hash === recipe ? {recipe_hash: hash, hypothesis: "Compare the effect of zeroing two classifier rows. <sample> https://example.invalid", ops: 2} : null;
    },
    async approvals(session) {return session === agent ? [...pending.values()] : null;},
    controls: {
      async revoke(session) {
        if (revoked.has(session)) return;
        revoked.set(session, stamp(now())); control(session, "revoke");
      },
      async budget(session, dimension, limit) {
        if (!Number.isSafeInteger(limit) || limit < 0) throw new Error("Enter a non-negative whole number.");
        const from = limits[dimension];
        limits[dimension] = limit;
        control(session, "budget", {budget_change: {dimension, from, to: limit}});
      },
      async preview(id) {
        if (!pending.has(id)) throw new Error("This proposal is no longer pending.");
        return {warnings: ["The plan touches two rows in the classifier."], refusals: [],
          side_effects: [{kind: "checkpoint_created", bytes: 47104}]};
      },
      async decide(id, decision) {
        if (!pending.delete(id)) throw new Error("This proposal is no longer pending.");
        control(agent, decision, {proposal_id: id, route: "POST /proposals/{id}/" + decision});
        if (decision === "approve") add(agent, "apply", now(), {status: 201, proposal_id: id, recipe_hash: recipe,
          child_id: child, existing: true, selections: [selection], links: {plan_seqs: [37, 39], selection_seqs: [9, 22, 27], child_id: child}});
      },
    },
  };
}
