import assert from "node:assert/strict";
import {test} from "node:test";
import {bucketLit, layoutGraph, outline, outlineMatches, unitAt} from "./src/navigatorModel.ts";

test("member strips retain every packed block when compressed", () => {
  const mapping = {offset: 3, block: 2, period: 20, repeat: 3};
  assert.equal(unitAt(mapping, 47, 5), 2);
  assert.equal(unitAt(mapping, 17, 5), undefined);
  assert.equal(bucketLit(mapping, 2, 9, 27), false);
  assert.equal(bucketLit(mapping, 2, 8, 28), true);
  assert.equal(bucketLit(mapping, 2, 48, 62), true);
  assert.equal(bucketLit(mapping, 2, 49, 62), false);
});

const source = {id: "n0", op: "aten.conv2d.default", module_path: "features.first", outputs: [{value: "v1"}]};
const sink = {id: "n1", op: "aten.cumsum.default", module_path: "head", outputs: [{value: "v2"}], opaque_reason: "no_rule"};
const details = {n1: {...sink, args: [["v1", "v1"], {str: "v200"}], kwargs: {}, parameters: []}};
test("the layered layout retains real value references when a compound expands", () => {
  const collapsed = layoutGraph([source, sink], details, "", new Set());
  assert.equal(collapsed.edges.length, 1);
  assert.equal(collapsed.edges[0].from.id, "module:features");
  assert.ok(collapsed.edges[0].from.x < collapsed.edges[0].to.x);
  const expanded = layoutGraph([source, sink], details, "", new Set(["features", "features.first"]));
  assert.equal(expanded.edges[0].from.id, "node:n0");
  assert.equal(expanded.edges[0].to.id, "module:head");
});

test("outline tokens reveal ancestors and keep unbound tensors distinct", () => {
  const rows = outline({summary: {}, modules: [], nodes: [source, sink], groups: [],
    tensors: [{name: "unused.weight", dtype: "F32", binding: "unbound", roles: []}]}, details);
  const found = outlineMatches(rows, "opaque op:cumsum");
  assert.deepEqual([...found.matching], ["node:n1"]);
  assert.ok(found.shown.has("module:head"));
  assert.ok(!found.shown.has("module:features"));
  assert.ok(outlineMatches(rows, "unbound").shown.has("binding:unbound"));
});
