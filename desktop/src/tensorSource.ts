import {ApiError, connect, get, post} from "./api";

// Inspect wire types follow docs/reference/inspect-ops.md.
export type Value = number | boolean | "NaN" | "Inf" | "-Inf" | `${bigint}`;
export const isNonfinite = (value: Value | undefined) => value === "NaN" || value === "Inf" || value === "-Inf";
export type AxisRange = {start: number; stop: number};
export type Index = number[] | AxisRange | AxisRange[];
export type Selector = {checkpoint?: string; tensor: string; shape?: number[]; index?: Record<string, Index>};
export type Resolved = {tensor: string; shape: number[]; index: Record<string, AxisRange[]>};
type PrecisionFingerprint = {selection: string | null; op_revision: 1};
export type CheckpointFingerprint = PrecisionFingerprint & {checkpoint: string | null};
export type DiffFingerprint = PrecisionFingerprint & {checkpoint_a: string | null; checkpoint_b: string | null};
export type Envelope<T, R = Record<string, unknown>, F = CheckpointFingerprint> = {
  schema_version: 1; resolved: R; result: T;
  fingerprints: F;
  side_effects: []; truncated: boolean; cursor: string | null;
};
export type Role = {axis: number | null; role: string; provenance: string; basis: string; factors?: {size: number; name: string}[]};
export type Tensor = {
  name: string; dtype: string; shape: number[]; numel: number; nbytes: number; editable: boolean;
  read_only_reason: "dtype_not_editable" | null; shard: string | null;
  tensor_sha256: string | null; identical_bytes_group: string | null; roles: Role[];
};
export type Checkpoint = {
  checkpoint_id: string; display_name: string; origin: "fixture" | "imported" | "child";
  parent_id?: string | null; recipe_hash?: string | null; imported_at?: string | null;
  format?: "safetensors"; sharded?: boolean; shard_count?: number; file_bytes?: number;
  tensor_count?: number; element_count?: number; adapter_compatible?: boolean;
};
export type Description = {
  checkpoint: {format: "safetensors"; sharded: boolean; shard_count: number; file_bytes: number;
    tensor_count: number; element_count: number; metadata: Record<string, string>};
  coverage: {unit: "axes"; derived: number; declared: number; inferred: number; unknown: number; opaque_ops: (string | {op: string; count: number})[]};
  identical_bytes?: null; dtypes: Record<string, number>; matched: number; offset: number; tensors: Tensor[];
};
export type Range = {min: number; max: number; abs_max: number; abs_p99: number};
export type RangeSummary =
  | {range?: never; nonfinite_count?: never; range_skipped?: never}
  | {range: Range | null; nonfinite_count: number; range_skipped?: never}
  // The shipped backend can include null keys when it skips the scan.
  | {range?: null; nonfinite_count?: null; range_skipped: "selection_too_large"};
type ValuePage = {offset: number; returned_elems: number; values: Value[]} & RangeSummary;
export type Read = ValuePage & {
  tensor: string; dtype: string; shape: number[]; selection_shape: number[]; total_elems: number;
};
export type Unit = {
  index: Record<string, number>; count: number; nonfinite_count: number; min: number | null;
  max: number | null; mean: number | null; std: number | null; l1: number | null;
  l2: number | null; zero_frac: number | null; constant: boolean;
};
export type Stats = {
  tensor: string; dtype: string; reduce_over_axes: number[]; unit_axes: number[];
  unit_count: number; offset: number; accumulate_dtype: "float64"; units: Unit[];
  histogram?: {edges: number[]; counts: number[][]} | null; profile?: null;
};
export type DiffStatus = "same" | "changed" | "only_a" | "only_b" | "shape_mismatch" | "dtype_mismatch";
type Metrics = {changed_elems: number; max_abs: number | null; rel_l2: number | null; cosine: number | null};
export type DiffUnit = Metrics & {index: Record<string, number>};
type ComparedTensor = Metrics & {name: string; dtype: string; shape: number[]; status: "same" | "changed"; basis: "values"};
export type DiffTensor =
  | (ComparedTensor & {numel: number; unit_axis?: number; changed_units?: {total: number; listed: number[]; truncated: boolean}})
  | {name: string; status: "only_a" | "only_b"; dtype: string; shape: number[]}
  | {name: string; status: "shape_mismatch"; a: number[]; b: number[]}
  | {name: string; status: "dtype_mismatch"; a: string; b: string};
export type ModelDiff = {accumulate_dtype: "float64"; summary: {
  tensors_total: number; changed: number; same: number; only_a: string[]; only_b: string[];
  shape_mismatch: string[]; dtype_mismatch: string[]; changed_elems_total: number; max_abs: number | null;
  only_a_count: number; only_b_count: number; shape_mismatch_count: number; dtype_mismatch_count: number;
}; offset: number; tensors: DiffTensor[]};
export type SelectionDiff = {accumulate_dtype: "float64"; tensor: ComparedTensor & {total_elems: number}};
export type UnitDiff = SelectionDiff & ({unit_axis: number; unit_count: number; offset: number; units: DiffUnit[]}
  | {units?: never});
export type ValueDiff = SelectionDiff & ValuePage & {changed: (0 | 1)[]; delta_dtype: "float64"};
export type PageRequest = {limit?: number; cursor?: string | null};
export type DescribeRequest = PageRequest & {name_glob?: string};
export type Query = DescribeRequest & {dtype?: string[]; min_ndim?: number; max_ndim?: number; editable?: boolean;
  min_bytes?: number; max_bytes?: number; max_scan_elems?: number;
  where?: {stat: "l1" | "l2" | "max_abs" | "mean" | "std" | "min" | "max" | "zero_frac";
    axis: number; op: "lt" | "le" | "gt" | "ge"; value: number}};
export type Match = {name: string; selector: Selector | null; selector_truncated: boolean; selection_id: string; total_elems: number;
  units?: {index: Record<string, number>; value: number}[]; units_total?: number; units_truncated?: boolean};
type ReadRequest = {selector: Selector; max_elems?: number; include_range?: boolean | null; cursor?: string | null};
type StatsRequest = PageRequest & {selector: Selector; reduce_over_axes?: number[] | null; bins?: number};
export type ModelDiffRequest = DescribeRequest & {selector?: null; include_same?: boolean; unit_axis?: number | null; values?: false};
export type UnitDiffRequest = PageRequest & {selector: Selector; unit_axis?: number; values?: false};
export type SummaryDiffRequest = PageRequest & {selector: Selector; unit_axis: null; values?: false};
export type ValueDiffRequest = ReadRequest & PageRequest & {unit_axis?: number | null; values: true};
export type ImportProgress = {
  stage: "validating" | "copying" | "verifying" | "publishing";
  done: number; total: number; unit: "bytes" | "files" | null; tensor_count?: number | null;
};
export type CheckpointImport = {checkpoint: Checkpoint; existing: boolean};
export interface CheckpointSource {
  listCheckpoints(): Promise<Checkpoint[]>;
  getCheckpoint(id: string): Promise<Checkpoint>;
  importCheckpoint(input: {path: string; file?: File}, progress: (stage: ImportProgress | null) => void): Promise<CheckpointImport>;
}
export class InspectError extends Error {
  constructor(public code: string, message: string, public field: string | null = null, public missing?: string[]) {
    super(message);
  }
}
type ReadResolved = {selector: Resolved; max_elems: number; include_range: boolean};
type StatsResolved = {selector: Resolved; reduce_over_axes: number[]; bins: number; limit: number};
export type ModelDiffResolved = {selector: null; name_glob: string | null; unit_axis: number | null; include_same: boolean; limit: number};
export type SelectionDiffResolved = {selector: Resolved; unit_axis: number | null; values: boolean; limit: number;
  max_elems: number | null; include_range?: boolean};
export type ModelDiffReply = Envelope<ModelDiff, ModelDiffResolved, DiffFingerprint>;
export type DiffReply<T> = Envelope<T, SelectionDiffResolved, DiffFingerprint>;
export interface TensorDataSource {
  describe(id: string, request: DescribeRequest): Promise<Envelope<Description, {name_glob: string | null; limit: number}>>;
  read_tensor(id: string, request: ReadRequest): Promise<Envelope<Read, ReadResolved>>;
  stats(id: string, request: StatsRequest): Promise<Envelope<Stats, StatsResolved>>;
  diff(a: string, b: string, request: ModelDiffRequest): Promise<ModelDiffReply>;
  diff(a: string, b: string, request: UnitDiffRequest): Promise<DiffReply<UnitDiff>>;
  diff(a: string, b: string, request: SummaryDiffRequest): Promise<DiffReply<SelectionDiff>>;
  diff(a: string, b: string, request: ValueDiffRequest): Promise<DiffReply<ValueDiff>>;
  query(id: string, request: Query): Promise<Envelope<{matched: number; offset: number; matches: Match[]}>>;
}

export function inspectErrorText(error: unknown) {
  if (error instanceof InspectError || error instanceof ApiError)
    return error.message + " (" + [error.code, error.field].filter(Boolean).join(" · ") + ")";
  return error instanceof Error ? error.message : String(error);
}

const inspectRoute = (id: string, op: string) => "/checkpoints/" + encodeURIComponent(id) + "/" + op;

async function inspectRequest<T>(send: () => Promise<T>): Promise<T> {
  try {await connect(); return await send();}
  catch (error) {
    if (error instanceof ApiError) throw new InspectError(error.code, error.message, error.field);
    throw error;
  }
}

function inspectPost<T>(id: string, op: string, request: unknown) {
  return inspectRequest(() => post<T>(inspectRoute(id, op), request));
}

function diff(a: string, b: string, request: ModelDiffRequest): Promise<ModelDiffReply>;
function diff(a: string, b: string, request: UnitDiffRequest): Promise<DiffReply<UnitDiff>>;
function diff(a: string, b: string, request: SummaryDiffRequest): Promise<DiffReply<SelectionDiff>>;
function diff(a: string, b: string, request: ValueDiffRequest): Promise<DiffReply<ValueDiff>>;
function diff(a: string, b: string, request: ModelDiffRequest | UnitDiffRequest | SummaryDiffRequest | ValueDiffRequest) {
  return inspectPost<ModelDiffReply | DiffReply<SelectionDiff | UnitDiff | ValueDiff>>(a, "diff", {...request, other: b});
}

export const httpTensorSource: TensorDataSource = {
  describe(id, {name_glob, limit, cursor}) {
    const query = new URLSearchParams();
    if (name_glob !== undefined) query.set("name_glob", name_glob);
    if (limit !== undefined) query.set("limit", String(limit));
    if (cursor) query.set("cursor", cursor);
    return inspectRequest(() => get(inspectRoute(id, "describe") + "?" + query));
  },
  read_tensor: (id, request) => inspectPost(id, "read-tensor", request),
  stats: (id, request) => inspectPost(id, "stats", request),
  diff,
  query: (id, request) => inspectPost(id, "query", request),
};

export const elements = (shape: number[]) => shape.reduce((count, size) => count * size, 1);
export const globPattern = (glob: string) =>
  new RegExp("^" + glob.split(/(\*|\?)/).map(part =>
    part === "*" ? ".*" : part === "?" ? "." : part.replace(/[.*+?^$(){}|[\]\\]/g, "\\$&")).join("") + "$");
export const coordinates = (flat: number, shape: number[]) => {
  const out = Array<number>(shape.length);
  for (let axis = shape.length - 1; axis >= 0; axis--) {
    out[axis] = flat % shape[axis];
    flat = Math.floor(flat / shape[axis]);
  }
  return out;
};

export function resolveSelector(selector: Selector, tensor: Tensor): Resolved {
  if (selector.tensor !== tensor.name) throw new InspectError("unknown_tensor", "Unknown tensor.", "selector.tensor");
  if (selector.shape && (selector.shape.length !== tensor.shape.length || selector.shape.some((size, axis) => size !== tensor.shape[axis])))
    throw new InspectError("invalid_argument", "Selector shape differs from the tensor shape.", "selector.shape");
  const index: Resolved["index"] = {};
  for (const axis of Object.keys(selector.index ?? {})) {
    if (!/^(0|[1-9]\d*)$/.test(axis) || +axis >= tensor.shape.length)
      throw new InspectError("invalid_axis", "Axis " + axis + " does not exist.", "selector.index." + axis);
  }
  tensor.shape.forEach((dim, axis) => {
    const part = selector.index?.[axis];
    const field = "selector.index." + axis;
    if (part === undefined) {
      index[axis] = [{start: 0, stop: dim}];
      return;
    }
    const parts = Array.isArray(part) ? part : [part];
    if (parts.length > 100000) throw new InspectError("cap_exceeded", "Use at most 100,000 indices or ranges.", field);
    const rangeList = parts.every(item => typeof item === "object" && item !== null);
    if (!parts.length || !rangeList && parts.some(item => typeof item !== "number"))
      throw new InspectError("invalid_index", "Use a nonempty list of indices or ranges.", field);

    const ranges = parts.map((item): AxisRange => {
      if (typeof item === "number") {
        if (!Number.isInteger(item) || item < 0)
          throw new InspectError("invalid_index", "Use a non-negative integer index.", field);
        if (item >= dim) throw new InspectError("index_out_of_range", "Index " + item + " is outside axis " + axis + " of size " + dim + ".", field);
        return {start: item, stop: item + 1};
      }
      const {start, stop} = item;
      if (!Number.isInteger(start) || !Number.isInteger(stop) || start < 0 || start >= stop || stop > dim)
        throw new InspectError("invalid_range", "Use a nonempty range inside axis " + axis + " of size " + dim + ".", field);
      return {start, stop};
    }).sort((a, b) => a.start - b.start || a.stop - b.stop);
    const merged: AxisRange[] = [];
    for (const range of ranges) {
      const previous = merged.at(-1);
      if (previous && previous.stop >= range.start) previous.stop = Math.max(previous.stop, range.stop);
      else merged.push(range);
    }
    index[axis] = merged;
  });
  return {tensor: tensor.name, shape: tensor.shape, index};
}

export const indicesFor = (resolved: Resolved) => resolved.shape.map((_, axis) =>
  resolved.index[axis].flatMap(({start, stop}) => Array.from({length: stop - start}, (_, i) => start + i)));
