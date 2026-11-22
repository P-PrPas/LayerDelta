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
  assert.ok(first.result.tensors.every(t => !t.roles.length && t.tensor_sha256 === null && t.identical_bytes_group === null));
});

test("resolved ranges can be replayed and range metadata is first-page by default", async () => {
  const selector = {tensor: "diagnostics.nonfinite", index: {0: [1, 0, 1]}};
  const first = await source.read_tensor(child, {selector, max_elems: 4});
  assert.deepEqual(first.resolved.selector.index[0], [{start: 0, stop: 2}]);
  assert.deepEqual(first.resolved.selector.shape, [8, 8]);
  assert.equal(first.resolved.include_range, true);
  assert.deepEqual(first.result.values, ["NaN", "Inf", "-Inf", -0]);
  const second = await source.read_tensor(child, {selector, max_elems: 4, cursor: first.cursor});
  assert.equal(second.result.offset, 4);
  assert.equal(second.fingerprints.selection, first.fingerprints.selection);
  assert.ok(!("range" in second.result) && !("nonfinite_count" in second.result));
  const replay = await source.read_tensor(child, {selector: first.resolved.selector, max_elems: 4,
    cursor: first.cursor, include_range: true});
  assert.equal(replay.fingerprints.selection, first.fingerprints.selection);
  assert.deepEqual(replay.result.range, first.result.range);
  assert.equal(replay.result.nonfinite_count, 3);
  await assert.rejects(source.read_tensor(parent, {selector, cursor: first.cursor}), {code: "invalid_cursor"});
  await assert.rejects(source.read_tensor(child, {selector: {...selector, shape: [7, 8]}}),
    {code: "invalid_argument", field: "selector.shape"});
  const nonfinite = await source.read_tensor(child, {selector: {...selector,
    index: {0: [0], 1: [{start: 1, stop: 3}, {start: 0, stop: 2}]}}});
  assert.equal(nonfinite.result.range, null);
  assert.equal(nonfinite.result.nonfinite_count, 3);
  const skipped = await source.read_tensor(child, {selector: {tensor: selector.tensor}, include_range: false});
  assert.ok(!("range" in skipped.result));
  await assert.rejects(source.read_tensor(child, {selector: {tensor: selector.tensor}, max_elems: 4, cursor: first.cursor}), {code: "invalid_cursor"});
  assert.throws(() => resolveSelector({tensor: selector.tensor, index: {0: [8]}}, {name: selector.tensor, shape: [8, 8]}),
    {code: "index_out_of_range"});
});

test("a zero delta can be changed and unpaired tensors have explicit statuses", async () => {
  const selector = {tensor: fixtureTensor, index: {0: [0], 1: [0], 2: [0], 3: [0]}};
  const diff = await source.diff(parent, child, {selector, values: true, unit_axis: null});
  assert.deepEqual(diff.result.values, [0]);
  assert.deepEqual(diff.result.changed, [1]);
  assert.deepEqual(diff.fingerprints, {checkpoint_a: parent, checkpoint_b: child, selection: diff.fingerprints.selection, op_revision: 1});
  assert.equal(diff.result.range.abs_max, 0);
  assert.equal(diff.result.tensor.rel_l2, 0);
  assert.equal(diff.result.tensor.total_elems, 1);
  assert.equal(diff.result.tensor.status, "changed");
  const window = {selector: {tensor: fixtureTensor, index: {0: [0], 1: [0]}}, values: true, max_elems: 1};
  const first = await source.diff(parent, child, window);
  const second = await source.diff(parent, child, {...window, cursor: first.cursor});
  assert.ok(!("range" in second.result));
  const ranged = await source.diff(parent, child, {...window, cursor: first.cursor, include_range: true});
  assert.deepEqual(ranged.result.range, first.result.range);
  await assert.rejects(source.diff(child, parent, {...window, cursor: first.cursor}), {code: "invalid_cursor"});
  const model = await source.diff(parent, child, {selector: null, name_glob: "manifest.*"});
  assert.deepEqual(new Set(model.result.tensors.map(t => t.status)), new Set(["only_a", "only_b", "shape_mismatch", "dtype_mismatch"]));
  assert.equal(model.result.offset, 0);
  assert.equal(model.result.summary.shape_mismatch_count, 1);
  assert.deepEqual(model.result.tensors.find(t => t.status === "shape_mismatch").b, [8]);
  assert.equal(model.result.summary.max_abs, null);
  const units = await source.diff(parent, child, {selector: {tensor: fixtureTensor}, unit_axis: 0, limit: 1});
  assert.equal(units.result.unit_count, 64);
  assert.equal(units.result.offset, 0);
  assert.equal(units.result.units[0].unit, 0);
});

test("position pages retain shared histogram edges", async () => {
  const request = {selector: {tensor: "diagnostics.long_profile"}, reduce_over_axes: [], bins: 16, limit: 2};
  const first = await source.stats(child, request);
  const second = await source.stats(child, {...request, cursor: first.cursor});
  assert.equal(first.result.unit_count, 50000);
  assert.deepEqual(second.result.units[0].index, {0: 2});
  assert.ok(!("unit" in second.result.units[0]));
  assert.equal(second.result.units[0].nonfinite_count, 0);
  assert.deepEqual(second.result.histogram.edges, first.result.histogram.edges);
  assert.equal(second.result.histogram.counts[0].reduce((a, b) => a + b), 1);
  const aggregate = await source.stats(child, {selector: {tensor: fixtureTensor}, reduce_over_axes: null});
  assert.deepEqual(aggregate.result.reduce_over_axes, [0, 1, 2, 3]);
});

test("query caps listed positions without dropping them from the selection", async () => {
  const result = await source.query(child, {name_glob: "diagnostics.long_profile", where: {stat: "l2", axis: 0, op: "le", value: 1}});
  const match = result.result.matches[0];
  assert.equal(match.units.length, 1000);
  assert.equal(match.units_total, 50000);
  assert.ok(match.units_truncated);
  const read = await source.read_tensor(child, {selector: match.selector, max_elems: 1, include_range: false});
  assert.equal(read.result.total_elems, 50000);
  assert.equal(read.fingerprints.selection, match.selection_id);
  const sparse = await source.query(child, {name_glob: "diagnostics.u64", where: {stat: "l2", axis: 0, op: "ge", value: 2 ** 53}});
  assert.deepEqual(sparse.result.matches[0].selector.index[0], [{start: 1, stop: 4}]);
});

test("I64 and U64 values preserve their exact decimal digits", async () => {
  const signed = await source.read_tensor(child, {selector: {tensor: "diagnostics.i64"}});
  const unsigned = await source.read_tensor(child, {selector: {tensor: "diagnostics.u64"}});
  assert.equal(signed.result.values[0], "-9223372036854775808");
  assert.equal(unsigned.result.values[3], "18446744073709551615");
  assert.equal(unsigned.result.nonfinite_count, 0);
});
