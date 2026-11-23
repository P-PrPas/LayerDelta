import assert from "node:assert/strict";
import {after, test} from "node:test";
import {createServer} from "vite";

const vite = await createServer({server: {middlewareMode: true, ws: false}, appType: "custom", optimizeDeps: {noDiscovery: true, include: []}});
after(() => vite.close());
const {createTraceFixture} = await vite.ssrLoadModule("/src/traceFixture.ts");
const {traceEvent} = await vite.ssrLoadModule("/src/traceSource.ts");
const {isApply, laneOf, sessionsFromEvents, statusMeaning, timelineRows, toolOf} = await vite.ssrLoadModule("/src/traceModel.ts");
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
  assert.deepEqual((await source.controls.preview(proposal.proposal_id)).refusals, []);
  await source.controls.decide(proposal.proposal_id, "approve");
  const page = await source.page({session: session.session, after: 0, limit: 500}, signal);
  assert.equal(page.summary.applies, 2);
  assert.equal(page.summary.charged_runs, 1);
  assert.equal((await source.approvals(session.session, signal)).length, 0);
  await source.controls.revoke(session.session);
  assert.ok((await source.sessions(signal)).find(s => s.session === session.session).revoked);
});
