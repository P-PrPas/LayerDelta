import assert from "node:assert/strict";
import {after, test} from "node:test";
import {createServer} from "vite";

const vite = await createServer({server: {middlewareMode: true, ws: false}, appType: "custom", optimizeDeps: {noDiscovery: true, include: []}});
after(() => vite.close());
const {createTraceFixture} = await vite.ssrLoadModule("/src/traceFixture.ts");
const {liveTraceSource, traceEvent} = await vite.ssrLoadModule("/src/traceSource.ts");
const {isApply, laneOf, sessionsFromEvents, statusMeaning, timelineRows, tensorTarget, toolOf} = await vite.ssrLoadModule("/src/traceModel.ts");
const signal = new AbortController().signal;

test("live projection keeps ID-only fields and leaves S5 data absent", () => {
  const event = traceEvent({event_id: 6, session: "agent-example", seq: 2, at: "2026-01-01T00:00:00Z",
    route: "GET /checkpoints/{checkpoint_id}", tool: "checkpoints", status: 403, error_code: "forbidden", prompt: "untrusted"});
  assert.equal(toolOf(event), "get_checkpoint");
  assert.equal(laneOf(event), "Read");
  assert.equal(event.via, null);
  assert.equal(event.budget_charged, null);
  assert.ok(!Object.hasOwn(event, "prompt"));
  assert.match(statusMeaning(event, null), /model:read/);
});

test("live supervision uses the current agent, approval gate, and exact frozen recipe", async t => {
  t.mock.method(globalThis, "fetch", async (url, options) => {
    const route = String(url).replace("/api/v1/projects/test", "");
    requests.push({route, body: options?.body && JSON.parse(options.body), raw: options?.body, csrf: options?.headers?.["X-CSRF-Token"]});
    const result = route === "/trace/sessions" ? {current_agent_session: "agent-current", sessions: [
      {session: "agent-old", kind: "agent", revoked_at: "2026-01-01", via: {http: 2}},
      {session: "agent-current", kind: "agent", revoked_at: null, via: {http: 2, mcp: 1}},
    ]} : route.startsWith("/proposals?") ? {approval_gate: gate, proposals: [proposal]} :
      route.startsWith("/recipes/by-hash/") ? {recipe} :
        route.endsWith("/plan") ? {ops: [{op: "scale", tensor: "right", selected_elems: 4}], warnings: [{message: "Open group"}], child_bytes: 24, changed_elems_total: 4} :
          route === "/selections/resolve" ? {selections: {sel_1: [{checkpoint_id: "other", selector: {tensor: "wrong"}},
            {checkpoint_id: "parent", selector: {tensor: "right"}}]}} : {};
    if (route.startsWith("/recipes/by-hash/")) {
      return new Response(JSON.stringify({result}).replace('"k":2', '"k":2.0'));
    }
    return Response.json({result});
  });
  const requests = [];
  const proposal = {proposal_id: "p_1", recipe_hash: "abc", checkpoint_id: "parent"};
  const recipe = {recipe_hash: "abc", hypothesis: "<untrusted>", program: [{op: "scale", args: {k: 2}}]};
  let gate = false;
  Object.defineProperties(globalThis, {
    location: {value: {hash: ""}, configurable: true},
    sessionStorage: {value: {getItem: () => JSON.stringify({project_id: "test", csrf: "human-token"})}, configurable: true},
  });
  t.after(() => {delete globalThis.location; delete globalThis.sessionStorage;});
  const sessions = await liveTraceSource.sessions(signal);
  assert.deepEqual(sessions.map(row => [row.current, row.revoked, row.via_mix]),
    [[false, "2026-01-01", {http: 2}], [true, null, {http: 2, mcp: 1}]]);
  assert.equal(await liveTraceSource.approvals("agent-current", signal), null);
  gate = true;
  assert.deepEqual(await liveTraceSource.approvals("agent-current", signal), [proposal]);
  assert.deepEqual(await liveTraceSource.resolve([{checkpoint_id: "parent", selection_id: "sel_1"}], signal),
    [{checkpoint_id: "parent", selection_id: "sel_1", selector: {tensor: "right"}}]);
  assert.deepEqual((await liveTraceSource.controls.preview(proposal)).warnings, ["Open group"]);
  await liveTraceSource.controls.budget("agent-current", "scan", null);
  await liveTraceSource.controls.decide(proposal.proposal_id, "approve");
  await liveTraceSource.controls.revoke("agent-current");
  assert.match(requests.find(r => r.route.endsWith("/plan")).raw, /"k":2\.0/);
  assert.deepEqual(requests.filter(r => r.body).map(r => [r.route, r.body, r.csrf]), [
    ["/selections/resolve", {ids: ["sel_1"]}, "human-token"],
    ["/checkpoints/parent/plan", {recipe}, "human-token"],
    ["/agent-sessions/agent-current/budget", {dimension: "scan", limit: null}, "human-token"],
    ["/proposals/p_1/approve", {}, "human-token"],
    ["/agent-sessions/agent-current/revoke", {}, "human-token"],
  ]);
});

test("recorded approval keeps the proposal link and the ID-only control payload", () => {
  const event = traceEvent({event_id: 9, session: "agent-current", seq: 3, at: "2026-01-01T00:00:00Z", status: 200,
    route: "POST /proposals/{proposal_id}/approve", tool: "approve_proposal",
    control: {principal: "human", action: "approve", proposal_id: "p_1", child_id: "child", hypothesis: "untrusted"},
    links: {proposal_seq: 2, plan_seqs: [1], child_id: "child"}});
  assert.equal(laneOf(event), "Control");
  assert.equal(isApply(event), true);
  assert.equal(event.links.proposal_seq, 2);
  assert.equal(event.control.child_id, "child");
  assert.equal(Object.hasOwn(event.control, "hypothesis"), false);
});

test("timeline selector text preserves axis order and compacts disjoint index ranges", () => {
  assert.equal(tensorTarget({tensor: "classifier.weight", shape: [10, 1176], index: {0: [{start: 2, stop: 4}]}}), "classifier.weight [2:4, :]");
  assert.equal(tensorTarget({tensor: "units", shape: [16, 4], index: {1: [3], 0: [3, 9, 12, 13, 14, 15]}}), "units [[3, 9, 12–15], 3]");
});

test("fixture pages by global ID without merging agent sessions", async () => {
  const source = createTraceFixture(() => 1);
  const a = await source.page({after: 0, limit: 500}, signal);
  const b = await source.page({after: a.next_after, limit: 500}, signal);
  assert.equal(a.events.length, 500);
  assert.ok(b.events[0].event_id > a.events.at(-1).event_id);
  assert.equal(b.next_after, null);
  assert.equal(new Set([...a.events, ...b.events].map(e => e.event_id)).size, b.summary.events);
  assert.equal(sessionsFromEvents([...a.events, ...b.events]).filter(s => s.kind === "agent").length, 2);
  assert.deepEqual(a, await createTraceFixture(() => 1).page({after: 0, limit: 500}, signal));
});

test("a selected apply only highlights the recorded links, including reads from a burst", async () => {
  const source = createTraceFixture(() => 1);
  const [session] = await source.sessions(signal);
  const page = await source.page({session: session.session, after: 0, limit: 500}, signal);
  assert.equal(page.events.filter(isApply).length, 1);
  const apply = page.events.find(isApply);
  const links = new Set([...apply.links.plan_seqs, ...apply.links.selection_seqs, 900]);
  const rows = timelineRows(page.events, links, new Set(), () => true, false);
  assert.ok(rows.some(row => row.burst?.length >= 3));
  assert.equal(rows.filter(row => row.event && links.has(row.event.seq)).length, 5);
  assert.ok(rows.some(row => row.missing === 900));
  assert.ok(rows.some(row => row.event?.status === 403));
});

test("fixture proposals apply the previewed recipe without a second charge", async () => {
  const source = createTraceFixture(() => 1);
  const [session] = await source.sessions(signal);
  const [proposal] = await source.approvals(session.session, signal);
  assert.deepEqual((await source.controls.preview(proposal)).refusals, []);
  await source.controls.decide(proposal.proposal_id, "approve");
  const page = await source.page({session: session.session, after: 0, limit: 500}, signal);
  assert.equal(page.summary.applies, 2);
  assert.equal(page.summary.charged_runs, 1);
  assert.equal((await source.approvals(session.session, signal)).length, 0);
  await source.controls.revoke(session.session);
  assert.ok((await source.sessions(signal)).find(s => s.session === session.session).revoked);
});
