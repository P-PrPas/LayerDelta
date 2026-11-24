import assert from "node:assert/strict";
import {after, test} from "node:test";
import {createServer} from "vite";

const vite = await createServer({server: {middlewareMode: true, ws: false}, appType: "custom", optimizeDeps: {noDiscovery: true, include: []}});
after(() => vite.close());
const {evaluationLinks, intervalDomain, metricNumber, originOf, reading, textAlignment} = await vite.ssrLoadModule("/src/measurementModel.ts");
const {traceEvent} = await vite.ssrLoadModule("/src/traceSource.ts");
const {timelineRows, statusMeaning} = await vite.ssrLoadModule("/src/traceModel.ts");
const {indexPalette} = await vite.ssrLoadModule("/src/evaluationMask.ts");

test("only a primary interval produces a reading, and lower-is-better is mirrored", () => {
  const metric = {primary: true, unit: "rate", better: "lower", delta: -.04, ci: [-.08, -.02]};
  assert.equal(reading(metric), "Improved");
  assert.equal(reading({...metric, better: "higher"}), "Worse");
  assert.equal(reading({...metric, ci: [-.08, 0]}), "No clear change");
  assert.equal(reading({...metric, ci: null}), "");
  assert.equal(reading({...metric, primary: false}), "");
  assert.equal(metricNumber(.0055, "rate", true), "+0.55 pp");
  assert.deepEqual(intervalDomain([metric, {unit: "count", delta: 20}]), {rate: .08, count: 20, score: 0});
});
const event = (seq, tool, data = {}) => traceEvent({seq, event_id: seq, at: "2026-01-01T00:00:00Z", session: "agent", route: "GET /test",
  status: 200, tool, ...data});

test("trace chains use explicit apply links only and result bursts never hide errors", () => {
  const write = event(3, "apply", {links: {plan_seqs: [2], selection_seqs: [1]}});
  const evaluation = event(4, "evaluate", {status: 202, evaluation_id: "ev_first", links: {apply_seq: 3},
    dataset_id: "ds_01", eval_samples: 600, budget_charged: {evals: 1, eval_samples: 600}, internal_evaluator: {}});
  const reads = [5,6,7].map(seq => event(seq, "get_evaluation", {evaluation_id: "ev_first"}));
  const bad = event(8, "get_evaluation", {evaluation_id: "ev_first", status: 403});
  const another = event(9, "evaluate", {evaluation_id: "ev_other", links: {apply_seq: null}, split_sha256: "digest", offset: 0, page_limit: 7});
  const events = [write, evaluation, ...reads, bad, another];
  assert.deepEqual([...evaluationLinks(events, evaluation)].sort(), [1, 2, 3]);
  assert.deepEqual([...evaluationLinks(events, write)].sort(), [1, 2, 4]);
  assert.deepEqual([...evaluationLinks(events, another)], []);
  assert.equal(another.split_sha256, "digest");
  assert.equal(another.offset, 0);
  assert.equal(another.page_limit, 7);
  assert.equal(originOf(events, bad), evaluation);
  assert.deepEqual(evaluation.budget_charged, {evals: 1, eval_samples: 600});
  assert.equal(Object.hasOwn(evaluation, "internal_evaluator"), false);
  const rows = timelineRows(events, new Set(), new Set(), () => true, false);
  assert.equal(rows.find(row => row.burst)?.burst.length, 3);
  assert.ok(rows.some(row => row.event === bad));
  assert.match(statusMeaning(event(10,"evaluate",{status:403,split:"holdout"}),1), /holdout/);
});

test("OCR alignment preserves plain text and marks insertions and substitutions", () => {
  const rows = textAlignment("4071", "4011");
  assert.deepEqual(rows.after.map(part => part.changed), [false, false, true, false]);
  assert.equal(textAlignment("<>", "<x>").after.find(part => part.changed)?.text, "x");
});

test("PNG palette replacement maps duplicate colors to distinct truth indices", () => {
  const png = Uint8Array.from(Buffer.from(
    "89504e470d0a1a0a0000000d49484452000000010000000108030000000000000000000006504c5445aabbccaabbcc000000000000000049454e44ae426082", "hex"));
  const output = indexPalette(png);
  assert.deepEqual([...output.slice(41, 47)], [0,0,0,1,1,1]);
  assert.notEqual(Buffer.from(output).readUInt32BE(47), 0);
});
