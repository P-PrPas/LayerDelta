import {
  Checkpoint, coordinates, Description, DiffReply, DiffTensor, elements, Envelope, globPattern, indicesFor, Index,
  InspectError, isNonfinite, Match, ModelDiff, ModelDiffRequest, ModelDiffResolved, PageRequest, Range,
  resolveSelector, Resolved, SelectionDiff, Selector, SummaryDiffRequest, Tensor, TensorDataSource, Unit, UnitDiff,
  UnitDiffRequest, Value, ValueDiff, ValueDiffRequest,
} from "./tensorSource";

const parent: Checkpoint = {checkpoint_id: "sha256:7be20d55c913" + "0".repeat(52), filename: "unet-seg-v3-base.safetensors"};
const child: Checkpoint = {checkpoint_id: "sha256:a41f09c2e7d1" + "1".repeat(52), filename: "unet-seg-v3.safetensors"};
export const fixtureTensor = "decoder.up2.conv1.weight";
const edited = new Set([fixtureTensor, "encoder.down1.conv.weight", "embeddings.token.weight"]);
const onlyA = "manifest.removed.weight";
const onlyB = "manifest.added.weight";
const mismatch = "manifest.resized.weight";
const dtypeMismatch = "manifest.retyped.weight";
const pause = (ms: number) => new Promise<void>(done => setTimeout(done, ms));

function tensor(name: string, dtype: string, shape: number[]): Tensor {
  const numel = elements(shape);
  const editable = dtype === "F32" || dtype === "BF16";
  return {name, dtype, shape, numel, nbytes: numel * (dtype === "BF16" ? 2 : dtype === "BOOL" ? 1 : 4),
    editable, read_only_reason: editable ? null : "dtype_not_editable", shard: null,
    tensor_sha256: null, identical_group: null, roles: []};
}

const shared = [
  tensor(fixtureTensor, "F32", [64, 128, 3, 3]),
  tensor("decoder.up2.conv1.bias", "F32", [64]),
  tensor("decoder.up2.conv2.weight", "BF16", [64, 64, 3, 3]),
  tensor("decoder.up2.norm.running_var", "F32", [64]),
  tensor("decoder.up2.norm.weight", "BF16", [64]),
  tensor("encoder.down1.conv.weight", "BF16", [32, 8, 3, 3]),
  tensor("encoder.down1.conv.bias", "BF16", [32]),
  tensor("embeddings.token.weight", "BF16", [512, 256]),
  tensor("diagnostics.nonfinite", "F32", [8, 8]),
  tensor("diagnostics.mask", "BOOL", [8, 8]),
  tensor("diagnostics.scalar", "F32", []),
  tensor("diagnostics.empty", "F32", [0, 8]),
  tensor("diagnostics.long_profile", "F32", [50000]),
  tensor("diagnostics.large_selection", "F32", [4097, 4096]),
  ...Array.from({length: 10394}, (_, i) => tensor(
    "encoder.layers." + Math.floor(i / 40) + ".attn." + (i % 40) + (i % 4 ? ".weight" : ".bias"),
    i % 3 ? "BF16" : "F32", i % 4 ? [256, 256] : [256])),
];
const catalog = (isChild: boolean) => [...shared,
  tensor(isChild ? onlyB : onlyA, "BF16", [32]),
  tensor(mismatch, "F32", [isChild ? 8 : 4]),
  tensor(dtypeMismatch, isChild ? "F32" : "BF16", [8])]
  .sort((a, b) => a.name < b.name ? -1 : a.name > b.name ? 1 : 0);
const aCatalog = catalog(false);
const bCatalog = catalog(true);
const aByName = new Map(aCatalog.map(t => [t.name, t]));
const bByName = new Map(bCatalog.map(t => [t.name, t]));

function seeded(t: Tensor, position: number[], id: string): Value {
  const flat = position.reduce((n, index, axis) => n * t.shape[axis] + index, 0);
  const row = position[0] ?? 0;
  const col = position[1] ?? 0;
  const isChild = id !== parent.checkpoint_id;
  if (t.dtype === "BOOL") return (flat * 17 % 31) < 10;
  if (t.name === "diagnostics.nonfinite" && flat < 4)
    return (["NaN", "Inf", "-Inf", -0] as const)[flat];
  if (t.name === "diagnostics.scalar") return 1;
  if (t.name === "diagnostics.large_selection") return (col % 16 - 8) / 64;
  if (t.name.endsWith(".bias") || t.name.endsWith(".norm.weight"))
    return t.name.endsWith(".bias") ? 0 : 1;
  if (t.name.endsWith("running_var")) return (flat % 9 + 1) / 16;
  if (edited.has(t.name) && flat === 0) return isChild ? 0 : -0;
  const phase = Math.imul(flat + 21, 2654435761) >>> 0;
  let value = Math.round((Math.sin(row * .76 + col * .43) * 32 + (phase % 61 - 30)) *
    (row === 9 ? .18 : 1)) / 512;
  if (row === 17 && t.shape.length > 1) value = isChild && edited.has(t.name) ? value / 2 : 0;
  if (isChild && row === 8 && edited.has(t.name)) value += 1 / 16;
  if (t.name === fixtureTensor && flat === 77) value = .75;
  return storedFloat(value, t.dtype);
}

const floatBits = new DataView(new ArrayBuffer(4));
function storedFloat(value: number, dtype: string) {
  if (dtype !== "BF16") return Math.fround(value);
  floatBits.setFloat32(0, value);
  const bits = floatBits.getUint32(0);
  floatBits.setUint32(0, (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000);
  return floatBits.getFloat32(0);
}

function displayValue(value: Value, dtype: string): Value {
  if (typeof value !== "number" || value === 0) return value;
  for (let digits = 1; digits < (dtype === "BF16" ? 5 : 10); digits++) {
    const rounded = Number(value.toPrecision(digits));
    if (storedFloat(rounded, dtype) === value) return rounded;
  }
  return value;
}

function limit(request: PageRequest, fallback: number, cap: number) {
  const count = request.limit ?? fallback;
  if (!Number.isInteger(count) || count < 1 || count > cap) throw new InspectError("cap_exceeded", "Invalid page size.", "limit");
  return count;
}

function pageStart(request: PageRequest, key: string, total: number) {
  if (!request.cursor) return 0;
  const split = request.cursor.indexOf(":");
  const pos = Number(request.cursor.slice(0, split));
  if (split < 1 || !Number.isInteger(pos) || pos < 1 || pos >= total || request.cursor.slice(split + 1) !== key)
    throw new InspectError("invalid_cursor", "Cursor belongs to a different request.", "cursor");
  return pos;
}

function reply<T, R>(id: string, resolved: R, result: T, offset: number, count: number,
  total: number, key: string, selection: string | null = null, other?: string): Envelope<T, R> {
  const next = offset + count;
  return {schema_version: 1, resolved, result, fingerprints: {checkpoint: id, selection, op_revision: 1,
    ...(other ? {checkpoint_b: other} : {})}, side_effects: [], truncated: next < total,
    cursor: next < total ? next + ":" + key : null};
}

async function selectionId({index, shape, tensor}: Resolved) {
  const canonical = '{"index":{' + Object.keys(index).sort().map(axis =>
    JSON.stringify(axis) + ":" + JSON.stringify(index[axis])).join(",") + '},"shape":' +
    JSON.stringify(shape) + ',"tensor":' + JSON.stringify(tensor) + "}";
  const hash = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(canonical));
  return "sel_" + Array.from(new Uint8Array(hash).slice(0, 16), b => b.toString(16).padStart(2, "0")).join("");
}

const byName = (id: string) => id === parent.checkpoint_id ? aByName : bByName;

function selection(id: string, selector: Selector) {
  const t = byName(id).get(selector.tensor);
  if (!t) throw new InspectError("unknown_tensor", "This tensor is not present in the checkpoint.", "selector.tensor");
  if (selector.checkpoint && selector.checkpoint !== id)
    throw new InspectError("invalid_argument", "Selector checkpoint differs from the open checkpoint.", "selector.checkpoint");
  const resolved = resolveSelector(selector, t);
  const axes = indicesFor(resolved);
  const shape = axes.map(axis => axis.length);
  return {t, resolved, axes, shape, total: elements(shape)};
}

function range(values: Value[]): Range | null {
  const finite = values.filter((v): v is number => typeof v === "number");
  if (!finite.length) return null;
  let min = Infinity;
  let max = -Infinity;
  for (const value of finite) {min = Math.min(min, value); max = Math.max(max, value);}
  const abs = finite.map(Math.abs).sort((a, b) => a - b);
  return {min, max, abs_max: abs.at(-1)!, abs_p99: abs[Math.ceil(abs.length * .99) - 1]};
}

const numeric = (v: Value) => v === "Inf" ? Infinity : v === "-Inf" ? -Infinity : Number(v);
const encoded = (v: number): Value => Number.isNaN(v) ? "NaN" : v === Infinity ? "Inf" : v === -Infinity ? "-Inf" : v;
const addresses = (axes: number[][], shape: number[], offset: number, count: number) =>
  Array.from({length: count}, (_, i) => coordinates(offset + i, shape).map((p, axis) => axes[axis][p]));
const samples = (id: string, t: Tensor, axes: number[][]) => {
  const shape = axes.map(axis => axis.length);
  return addresses(axes, shape, 0, elements(shape)).map(position => numeric(seeded(t, position, id)));
};

function summary(values: number[], unit: number | null, index: Record<string, number>): Unit {
  const finite = values.filter(Number.isFinite);
  const count = values.length;
  const n = finite.length;
  let min = Infinity, max = -Infinity, sum = 0, l1 = 0, square = 0, zeros = 0;
  for (const v of finite) {
    min = Math.min(min, v); max = Math.max(max, v); sum += v; l1 += Math.abs(v); square += v * v;
    if (v === 0) zeros++;
  }
  const mean = n ? sum / n : null;
  return {unit, index, count, nonfinite: count - n, min: n ? min : null, max: n ? max : null, mean,
    std: mean === null ? null : Math.sqrt(finite.reduce((s, v) => s + (v - mean) ** 2, 0) / n),
    l1: n ? l1 : null, l2: n ? Math.sqrt(square) : null, zero_frac: count ? zeros / count : null,
    constant: count > 0 && n === count && min === max};
}

function delta(a: Value, b: Value): Value {return encoded(numeric(b) - numeric(a));}
function comparison(t: Tensor, a: string, b: string, positions: number[][], changedAt?: (p: number[]) => void) {
  let changed_elems = 0, max_abs: number | null = null, aa = 0, bb = 0, dd = 0, ab = 0;
  for (const position of positions) {
    const va = seeded(t, position, a), vb = seeded(t, position, b);
    if (!Object.is(va, vb)) {changed_elems++; changedAt?.(position);}
    const x = numeric(va), y = numeric(vb);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    const d = y - x;
    max_abs = Math.max(max_abs ?? 0, Math.abs(d)); aa += x*x; bb += y*y; dd += d*d; ab += x*y;
  }
  return {changed_elems, max_abs, rel_l2: max_abs === null ? null : dd === 0 ? 0 : aa ? Math.sqrt(dd / aa) : null,
    cosine: aa && bb ? dd === 0 ? 1 : Math.max(-1, Math.min(1, ab / Math.sqrt(aa * bb))) : null};
}

function modelDiff(a: string, b: string, request: ModelDiffResolved): ModelDiff {
  const former = byName(a), latter = byName(b);
  const match = globPattern(request.name_glob ?? "*");
  const names = [...new Set([...former.keys(), ...latter.keys()])].filter(n => match.test(n)).sort();
  const summary: ModelDiff["summary"] = {tensors_total: names.length, changed: 0, same: 0, only_a: [], only_b: [],
    shape_mismatch: [], dtype_mismatch: [], changed_elems_total: 0, max_abs: null,
    only_a_count: 0, only_b_count: 0, shape_mismatch_count: 0, dtype_mismatch_count: 0};
  const tensors: DiffTensor[] = [];
  for (const name of names) {
    const ta = former.get(name), tb = latter.get(name);
    const status = !ta ? "only_b" : !tb ? "only_a" : ta.dtype !== tb.dtype ? "dtype_mismatch"
      : ta.shape.join() !== tb.shape.join() ? "shape_mismatch" : a !== b && edited.has(name) &&
        (a === parent.checkpoint_id || b === parent.checkpoint_id) ? "changed" : "same";
    const axis = request.unit_axis !== null && request.unit_axis < (tb?.shape.length ?? 0) ? request.unit_axis : undefined;
    const unitFields = (positions: number[]) => axis === undefined ? {} : {
      unit_axis: axis,
      changed_units: {total: positions.length, listed: positions.slice(0, 1000), truncated: positions.length > 1000},
    };
    if (status === "changed") {
      const t = tb!;
      const {axes, shape, total} = selection(b, {tensor: name});
      const changed = new Set<number>();
      const metrics = comparison(t, a, b, addresses(axes, shape, 0, total), axis === undefined ? undefined : p => changed.add(p[axis]));
      summary.changed++;
      summary.changed_elems_total += metrics.changed_elems;
      summary.max_abs = Math.max(summary.max_abs ?? 0, metrics.max_abs ?? 0);
      tensors.push({name, status, dtype: t.dtype, shape, numel: t.numel, basis: "values",
        ...metrics, ...unitFields([...changed].sort((a, b) => a - b))});
    } else if (status === "same") {
      summary.same++;
      const max_abs = tb!.numel ? 0 : null;
      summary.max_abs ??= max_abs;
      if (request.include_same) tensors.push({name, status, dtype: tb!.dtype, shape: tb!.shape, numel: tb!.numel,
        basis: "values", changed_elems: 0, max_abs, rel_l2: max_abs, cosine: tb!.numel && !name.endsWith(".bias") ? 1 : null, ...unitFields([])});
    } else {
      summary[`${status}_count` as const]++;
      if (summary[status].length < 1000) summary[status].push(name);
      if (status === "dtype_mismatch") tensors.push({name, status, a: ta!.dtype, b: tb!.dtype});
      else if (status === "shape_mismatch") tensors.push({name, status, a: ta!.shape, b: tb!.shape});
      else tensors.push({name, status, dtype: (tb ?? ta)!.dtype, shape: (tb ?? ta)!.shape});
    }
  }
  return {summary, tensors, offset: 0, accumulate_dtype: "float64"};
}

async function diff(a: string, b: string, request: ModelDiffRequest): Promise<Envelope<ModelDiff, ModelDiffResolved>>;
async function diff(a: string, b: string, request: UnitDiffRequest): Promise<DiffReply<UnitDiff>>;
async function diff(a: string, b: string, request: SummaryDiffRequest): Promise<DiffReply<SelectionDiff>>;
async function diff(a: string, b: string, request: ValueDiffRequest): Promise<DiffReply<ValueDiff>>;
async function diff(a: string, b: string, request: ModelDiffRequest | UnitDiffRequest | SummaryDiffRequest | ValueDiffRequest) {
  const size = limit(request, 500, 1000);
  const unitAxis = request.unit_axis === undefined ? 0 : request.unit_axis;
  if (!request.selector) {
    await pause(240);
    if (unitAxis !== null && (!Number.isInteger(unitAxis) || unitAxis < 0))
      throw new InspectError("invalid_axis", "Use a nonnegative integer axis or null.", "unit_axis");
    const resolved: ModelDiffResolved = {selector: null, name_glob: request.name_glob ?? null, include_same: request.include_same ?? false,
      unit_axis: unitAxis, limit: size};
    const result = modelDiff(a, b, resolved);
    const key = JSON.stringify(["diff", a, b, resolved]);
    const offset = pageStart(request, key, result.tensors.length);
    const tensors = result.tensors.slice(offset, offset + size);
    return reply(a, resolved, {...result, offset, tensors}, offset, tensors.length, result.tensors.length, key, null, b);
  }
  const {t, resolved, axes, shape, total} = selection(a, request.selector);
  const other = byName(b).get(t.name);
  if (!other || other.dtype !== t.dtype || other.shape.join() !== t.shape.join())
    throw new InspectError("incompatible_tensors", "These tensors cannot be compared value by value.", "selector.tensor");
  const id = await selectionId(resolved);
  const axis = request.values || !shape.length ? null : unitAxis;
  const max_elems = request.values ? limit({limit: request.max_elems}, 1000, 10000) : null;
  const args = {selector: resolved, unit_axis: axis, values: !!request.values, limit: size, max_elems};
  const key = JSON.stringify(["diff", a, b, id, axis, max_elems ?? size]);
  // shortcut: fixtures inspect small comparison candidates, replace this scan with the T4c response.
  const all = total <= 2 ** 24 ? addresses(axes, shape, 0, total) : [];
  const metrics = comparison(t, a, b, all.length || !total ? all : addresses(axes, shape, 0, 1));
  const result: SelectionDiff = {accumulate_dtype: "float64", tensor: {name: t.name, dtype: t.dtype, shape: t.shape,
    basis: "values", total_elems: total, status: metrics.changed_elems ? "changed" : "same", ...metrics}};
  if (max_elems !== null) {
    const offset = pageStart(request, key, total);
    const returned_elems = Math.min(total - offset, max_elems);
    const positions = all.length ? all.slice(offset, offset + returned_elems) : addresses(axes, shape, offset, returned_elems);
    const values = positions.map(p => delta(seeded(t, p, a), seeded(t, p, b)));
    const changed: (0 | 1)[] = positions.map(p => Object.is(seeded(t, p, a), seeded(t, p, b)) ? 0 : 1);
    const deltas = all.map(p => delta(seeded(t, p, a), seeded(t, p, b)));
    const range_summary = total <= 2 ** 24 ? {range: range(deltas), nonfinite_count: deltas.filter(isNonfinite).length}
      : {range: null, nonfinite_count: null, range_skipped: "selection_too_large" as const};
    return reply(a, args, {...result, offset, returned_elems, values, changed, delta_dtype: "float64" as const, ...range_summary},
      offset, returned_elems, total, key, id, b);
  }
  if (axis === null) return reply(a, args, result, 0, 0, 0, key, id, b);
  const positions = axes[axis];
  if (!positions) throw new InspectError("invalid_axis", "Unit axis does not exist.", "unit_axis");
  const offset = pageStart(request, key, positions.length);
  const units = positions.slice(offset, offset + size).map(unit => {
    const axesForUnit = axes.map((p, i) => i === axis ? [unit] : p);
    const unitShape = axesForUnit.map(p => p.length);
    return {unit, ...comparison(t, a, b, addresses(axesForUnit, unitShape, 0, elements(unitShape)))};
  });
  return reply(a, args, {...result, unit_axis: axis, unit_count: positions.length, offset, units},
    offset, units.length, positions.length, key, id, b);
}

export function createFixtureSource(empty = false): TensorDataSource {
  let checkpoints = empty ? [] : [child, parent];
  return {
    fixture: true,
    async listCheckpoints() {return checkpoints;},
    async importCheckpoint({path}, report) {
      const fixtureName = path.split(/[/\\]/).at(-1) ?? "";
      for (const stage of ["header", "extents", "table"] as const) {
        const total = stage === "header" ? 19200 : stage === "extents" ? bCatalog.length : 1;
        report({stage, done: 0, total});
        await pause(300);
        if (stage === "header" && /^invalid\./i.test(fixtureName))
          throw new InspectError("header_json", "Re-export the checkpoint as safetensors and try again.");
        if (stage === "extents" && /^missing\./i.test(fixtureName))
          throw new InspectError("shard_missing", "The index refers to a shard that could not be opened.", null, ["model-00002-of-00002.safetensors"]);
        report({stage, done: total, total});
        await pause(230);
      }
      const checkpoint = {...child, filename: fixtureName};
      checkpoints = [checkpoint, ...checkpoints.filter(c => c.checkpoint_id !== checkpoint.checkpoint_id)];
      return checkpoint;
    },
    async describe(id, request) {
      await pause(130);
      const isParent = id === parent.checkpoint_id;
      const all = isParent ? aCatalog : bCatalog;
      const matches = globPattern(request.name_glob ?? "*");
      const matched = all.filter(t => matches.test(t.name));
      const key = JSON.stringify(["describe", id, request.name_glob, request.limit ?? 5000]);
      const offset = pageStart(request, key, matched.length);
      const size = limit(request, 5000, 20000);
      const tensors = matched.slice(offset, offset + size);
      const dtypes: Record<string, number> = {};
      let file_bytes = 0, element_count = 0, unknown = 0;
      for (const t of all) {file_bytes += t.nbytes; element_count += t.numel; unknown += t.shape.length;}
      for (const t of matched) dtypes[t.dtype] = (dtypes[t.dtype] ?? 0) + 1;
      const sharded = checkpoints.find(c => c.checkpoint_id === id)?.filename.endsWith(".index.json") ?? false;
      const result: Description = {checkpoint: {format: "safetensors", sharded, shard_count: sharded ? 2 : 1,
        file_bytes, tensor_count: all.length, element_count, metadata: {format: "pt"}},
        coverage: {unit: "axes", derived: 0, declared: 0, inferred: 0, unknown, opaque_ops: []},
        identical_bytes: null, dtypes, matched: matched.length, offset, tensors};
      return reply(id, {name_glob: request.name_glob ?? null, limit: size}, result, offset, tensors.length, matched.length, key);
    },
    async read_tensor(id, request) {
      const {t, resolved, axes, shape, total} = selection(id, request.selector);
      const fingerprint = await selectionId(resolved);
      const key = JSON.stringify(["read_tensor", id, fingerprint, request.max_elems ?? 1000]);
      const offset = pageStart(request, key, total);
      const max_elems = limit({limit: request.max_elems}, 1000, 10000);
      const count = Math.min(total - offset, max_elems);
      const values = addresses(axes, shape, offset, count).map(p => seeded(t, p, id));
      const skip = total > 2 ** 24;
      const all = skip ? [] : count === total ? values : addresses(axes, shape, 0, total).map(p => seeded(t, p, id));
      const nonfinite_count = skip ? null : all.filter(isNonfinite).length;
      return reply(id, {selector: resolved, max_elems}, {tensor: t.name, dtype: t.dtype, shape: t.shape,
        selection_shape: shape, total_elems: total, offset, returned_elems: count, values: values.map(v => displayValue(v, t.dtype)),
        nonfinite_count, range: range(all), ...(skip ? {range_skipped: "selection_too_large" as const} : {})},
        offset, count, total, key, fingerprint);
    },
    async stats(id, request) {
      const {t, resolved, axes} = selection(id, request.selector);
      const repeatedRows = t.name === "diagnostics.large_selection";
      const sampleRows = (positions: number[][]) => samples(id, t, repeatedRows && positions[0].length ?
        [[positions[0][0]], positions[1]] : positions);
      const allAxes = axes.map((_, i) => i);
      const reduce = request.reduce_over_axes ?? allAxes;
      const unit_axes = allAxes.filter(i => !reduce.includes(i));
      const kept = unit_axes.map(i => axes[i]);
      const shape = kept.map(axis => axis.length);
      const unit_count = elements(shape);
      const fingerprint = await selectionId(resolved);
      const key = JSON.stringify(["stats", id, fingerprint, reduce, request.bins ?? 0, request.limit ?? 2048]);
      const offset = pageStart(request, key, unit_count);
      const size = limit(request, 2048, 2048);
      const count = Math.min(unit_count - offset, size);
      const bins = request.bins ?? 0;
      const whole = bins ? summary(sampleRows(axes), null, {}) : null;
      const low = whole?.min ?? 0, span = (whole?.max ?? 0) - low || 1;
      const edges = Array.from({length: bins + 1}, (_, i) => bins ? low + span * i / bins : low);
      const counts: number[][] = [];
      const units = addresses(kept, shape, offset, count).map(position => {
        const index = Object.fromEntries(unit_axes.map((axis, i) => [axis, position[i]]));
        const positions = axes.map((positions, axis) => axis in index ? [index[axis]] : positions);
        const values = sampleRows(positions);
        const repeats = repeatedRows ? positions[0].length : 1;
        if (bins) {
          const buckets = Array<number>(bins).fill(0);
          for (const v of values) if (Number.isFinite(v)) buckets[Math.min(bins - 1, Math.max(0, Math.floor((v - low) / span * bins)))] += repeats;
          counts.push(buckets);
        }
        const unit = summary(values, position.length === 1 ? position[0] : null, index);
        return {...unit, count: unit.count * repeats, nonfinite: unit.nonfinite * repeats,
          l1: unit.l1 === null ? null : unit.l1 * repeats, l2: unit.l2 === null ? null : unit.l2 * Math.sqrt(repeats)};
      });
      return reply(id, {selector: resolved, reduce_over_axes: reduce, bins, limit: size}, {tensor: t.name, dtype: t.dtype, reduce_over_axes: reduce, unit_axes,
        unit_count, offset, units, accumulate_dtype: "float64" as const, histogram: bins ? {edges, counts} : null, profile: null},
        offset, count, unit_count, key, fingerprint);
    },
    diff,
    async query(id, request) {
      const match = globPattern(request.name_glob ?? "*");
      const candidates = [...byName(id).values()].filter(t => match.test(t.name) &&
        (!request.dtype || request.dtype.includes(t.dtype)) &&
        t.shape.length >= (request.min_ndim ?? 0) && t.shape.length <= (request.max_ndim ?? Infinity) &&
        t.nbytes >= (request.min_bytes ?? 0) && t.nbytes <= (request.max_bytes ?? Infinity) &&
        (request.editable === undefined || request.editable === t.editable));
      const matches: Omit<Match, "selection_id">[] = [];
      if (request.where && candidates.reduce((sum, t) => sum + (t.shape[request.where!.axis] === undefined ? 0 : t.numel), 0) >
        (request.max_scan_elems ?? 2 ** 27)) throw new InspectError("cap_exceeded", "Narrow the filter before searching by statistics.", "max_scan_elems");
      for (const t of candidates) {
        const selector: Selector = {tensor: t.name};
        const where = request.where;
        if (!where) {matches.push({name: t.name, selector, total_elems: t.numel}); continue;}
        if (t.shape[where.axis] === undefined) continue;

        const {axes} = selection(id, selector);
        const units = axes[where.axis].flatMap(unit => {
          const stats = summary(samples(id, t, axes.map((positions, axis) => axis === where.axis ? [unit] : positions)), null, {});
          const value = where.stat === "max_abs" ? stats.min === null ? null : Math.max(Math.abs(stats.min), Math.abs(stats.max!)) : stats[where.stat];
          if (value === null) return [];
          const {op, value: cutoff} = where;
          return (op === "lt" ? value < cutoff : op === "le" ? value <= cutoff :
            op === "gt" ? value > cutoff : value >= cutoff) ? [{unit, value}] : [];
        });
        units.sort((a, b) => (where.op.startsWith("l") ? a.value - b.value : b.value - a.value) || a.unit - b.unit);
        if (units.length) matches.push({name: t.name, selector: {...selector, index: {[where.axis]: units.map(u => u.unit)}},
          total_elems: t.numel / t.shape[where.axis] * units.length, units});
      }
      const {cursor: _cursor, ...predicate} = request;
      const key = JSON.stringify([id, predicate]);
      const offset = pageStart(request, key, matches.length);
      const items = await Promise.all(matches.slice(offset, offset + limit(request, 100, 500)).map(async m =>
        ({...m, selection_id: await selectionId(selection(id, m.selector).resolved)})));
      return reply(id, {}, {matched: matches.length, offset, matches: items}, offset, items.length, matches.length, key);
    },
  };
}
