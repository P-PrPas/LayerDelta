import assert from "node:assert/strict";
import {after, test} from "node:test";
import {createServer} from "vite";

const vite = await createServer({server: {middlewareMode: true, ws: false}, appType: "custom"});
after(() => vite.close());
const {createFixtureSource, fixtureTensor} = await vite.ssrLoadModule("/src/tensorFixture.ts");
const {resolveSelector} = await vite.ssrLoadModule("/src/tensorSource.ts");
const source = createFixtureSource();
const [child, parent] = (await source.listCheckpoints()).map(c => c.checkpoint_id);

test("the describe fixture pages a 10k catalog and exposes deferred fields as null", async () => {
  const first = await source.describe(child, {limit: 8000});
  const second = await source.describe(child, {limit: 8000, cursor: first.cursor});
  assert.ok(first.result.checkpoint.tensor_count > 10000);
  assert.equal(first.result.tensors.length + second.result.tensors.length, first.result.checkpoint.tensor_count);
  assert.ok(first.truncated && !second.truncated);
  assert.equal(first.result.identical_bytes, null);
  assert.ok(first.result.tensors.every(t => !t.roles.length && t.tensor_sha256 === null));
});

test("selector ranges are canonical and flat pages preserve their range", async () => {
  const selector = {tensor: "diagnostics.nonfinite", index: {0: [1, 0, 1]}};
  const first = await source.read_tensor(child, {selector, max_elems: 4});
  assert.deepEqual(first.resolved.selector.index[0], [[0, 2]]);
  assert.deepEqual(first.result.values, ["NaN", "Inf", "-Inf", -0]);
  const second = await source.read_tensor(child, {selector, max_elems: 4, cursor: first.cursor});
  assert.equal(second.result.offset, 4);
  assert.equal(second.fingerprints.selection, first.fingerprints.selection);
  assert.deepEqual(second.result.range, first.result.range);
  await assert.rejects(source.read_tensor(child, {selector: {tensor: selector.tensor}, max_elems: 4, cursor: first.cursor}), {code: "invalid_cursor"});
  assert.throws(() => resolveSelector({tensor: selector.tensor, index: {0: [8]}}, {name: selector.tensor, shape: [8, 8]}),
    {code: "index_out_of_range"});
});

test("a zero delta can be changed and unpaired tensors have explicit statuses", async () => {
  const selector = {tensor: fixtureTensor, index: {0: [0], 1: [0], 2: [0], 3: [0]}};
  const diff = await source.diff(parent, child, {selector, values: true, unit_axis: null});
  assert.deepEqual(diff.result.values, [0]);
  assert.deepEqual(diff.result.changed, [1]);
  assert.equal(diff.result.rel_l2, null);
  const model = await source.diff(parent, child, {selector: null, name_glob: "manifest.*"});
  assert.deepEqual(new Set(model.result.tensors.map(t => t.status)), new Set(["only_a", "only_b", "shape_mismatch", "dtype_mismatch"]));
});

test("position pages retain shared histogram edges", async () => {
  const request = {selector: {tensor: "diagnostics.long_profile"}, reduce_over_axes: [], bins: 16, limit: 2};
  const first = await source.stats(child, request);
  const second = await source.stats(child, {...request, cursor: first.cursor});
  assert.equal(first.result.unit_count, 50000);
  assert.equal(second.result.units[0].unit, 2);
  assert.deepEqual(second.result.histogram.edges, first.result.histogram.edges);
  assert.equal(second.result.histogram.counts[0].reduce((a, b) => a + b), 1);
});
