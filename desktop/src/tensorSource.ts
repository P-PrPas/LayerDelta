// Inspect wire types follow docs/inspect-ops.md rev4 (79e8209).
export type Value = number | boolean | "NaN" | "Inf" | "-Inf";
export const isNonfinite = (value: Value | undefined) => value === "NaN" || value === "Inf" || value === "-Inf";
export type Index = number[] | {start: number; stop: number};
export type Selector = {checkpoint?: string; tensor: string; index?: Record<string, Index>};
export type Resolved = {tensor: string; shape: number[]; index: Record<string, [number, number][]>};
export type Envelope<T, R = Record<string, unknown>> = {
  schema_version: 1; resolved: R; result: T;
  fingerprints: {checkpoint: string | null; selection: string | null; op_revision: 1; checkpoint_b?: string | null};
  side_effects: []; truncated: boolean; cursor: string | null;
};
export type Tensor = {
  name: string; dtype: string; shape: number[]; numel: number; nbytes: number; editable: boolean;
  read_only_reason: "dtype_not_editable" | null; shard: string | null;
  tensor_sha256: string | null; identical_group: string | null; roles: [];
};
export type Checkpoint = {
  checkpoint_id: string; filename: string; display_name?: string | null; origin?: "imported" | "applied" | null;
  parent_id?: string | null; recipe_hash?: string | null; created_at?: string | null;
};
export type Description = {
  checkpoint: {format: "safetensors"; sharded: boolean; shard_count: number; file_bytes: number;
    tensor_count: number; element_count: number; metadata: Record<string, string>};
  coverage: {unit: "axes"; derived: number; declared: number; inferred: number; unknown: number; opaque_ops: string[]};
  identical_bytes?: null; dtypes: Record<string, number>; matched: number; offset: number; tensors: Tensor[];
};
export type Range = {min: number; max: number; abs_max: number; abs_p99: number};
type RangeSummary = {range: Range | null; nonfinite_count: number | null; range_skipped?: "selection_too_large"};
type ValuePage = {offset: number; returned_elems: number; values: Value[]} & RangeSummary;
export type Read = ValuePage & {
  tensor: string; dtype: string; shape: number[]; selection_shape: number[]; total_elems: number;
};
export type Unit = {
  unit: number | null; index: Record<string, number>; count: number; nonfinite: number; min: number | null;
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
export type DiffUnit = Metrics & {unit: number};
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
export type Match = {name: string; selector: Selector; selection_id: string; total_elems: number;
  units?: {unit: number; value: number}[]};
type ReadRequest = {selector: Selector; max_elems?: number; cursor?: string | null};
type StatsRequest = PageRequest & {selector: Selector; reduce_over_axes?: number[] | null; bins?: number};
export type ModelDiffRequest = DescribeRequest & {selector?: null; include_same?: boolean; unit_axis?: number | null; values?: false};
export type UnitDiffRequest = PageRequest & {selector: Selector; unit_axis?: number; values?: false};
export type SummaryDiffRequest = PageRequest & {selector: Selector; unit_axis: null; values?: false};
export type ValueDiffRequest = ReadRequest & PageRequest & {unit_axis?: number | null; values: true};
export type ImportProgress = {stage: "header" | "extents" | "table"; done: number; total: number};
export class InspectError extends Error {
  constructor(public code: string, message: string, public field: string | null = null, public missing?: string[]) {
    super(message);
  }
}
type ReadResolved = {selector: Resolved; max_elems: number};
type StatsResolved = {selector: Resolved; reduce_over_axes: number[]; bins: number; limit: number};
export type ModelDiffResolved = {selector: null; name_glob: string | null; unit_axis: number | null; include_same: boolean; limit: number};
export type SelectionDiffResolved = {selector: Resolved; unit_axis: number | null; values: boolean; limit: number; max_elems: number | null};
export type DiffReply<T> = Envelope<T, SelectionDiffResolved>;
export interface TensorDataSource {
  readonly fixture?: boolean;
  listCheckpoints(): Promise<Checkpoint[]>;
  importCheckpoint(input: {path: string; file?: File}, progress: (stage: ImportProgress | null) => void): Promise<Checkpoint>;
  describe(id: string, request: DescribeRequest): Promise<Envelope<Description, {name_glob: string | null; limit: number}>>;
  read_tensor(id: string, request: ReadRequest): Promise<Envelope<Read, ReadResolved>>;
  stats(id: string, request: StatsRequest): Promise<Envelope<Stats, StatsResolved>>;
  diff(a: string, b: string, request: ModelDiffRequest): Promise<Envelope<ModelDiff, ModelDiffResolved>>;
  diff(a: string, b: string, request: UnitDiffRequest): Promise<DiffReply<UnitDiff>>;
  diff(a: string, b: string, request: SummaryDiffRequest): Promise<DiffReply<SelectionDiff>>;
  diff(a: string, b: string, request: ValueDiffRequest): Promise<DiffReply<ValueDiff>>;
  query(id: string, request: Query): Promise<Envelope<{matched: number; offset: number; matches: Match[]}>>;
}

async function awaitingRoutes(): Promise<never> {
  throw new InspectError("routes_unavailable", "Checkpoint operations are waiting for T4b/T4c.");
}

// Replace these method bodies with the T4c transport once the routes and auth are available.
export const httpTensorSource: TensorDataSource = {
  listCheckpoints: awaitingRoutes, importCheckpoint: awaitingRoutes, describe: awaitingRoutes,
  read_tensor: awaitingRoutes, stats: awaitingRoutes, diff: awaitingRoutes, query: awaitingRoutes,
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
  const index: Resolved["index"] = {};
  for (const axis of Object.keys(selector.index ?? {})) {
    if (!/^(0|[1-9]\d*)$/.test(axis) || +axis >= tensor.shape.length)
      throw new InspectError("invalid_axis", "Axis " + axis + " does not exist.", "selector.index." + axis);
  }
  tensor.shape.forEach((dim, axis) => {
    const part = selector.index?.[axis];
    const field = "selector.index." + axis;
    if (part === undefined) index[axis] = [[0, dim]];
    else if (Array.isArray(part)) {
      if (!part.length || part.length > 100000 || part.some(n => !Number.isInteger(n) || n < 0))
        throw new InspectError("invalid_index", "Use one or more non-negative integer indices.", field);
      const ranges: [number, number][] = [];
      for (const n of [...new Set(part)].sort((a, b) => a - b)) {
        if (n >= dim) throw new InspectError("index_out_of_range", "Index " + n + " is outside axis " + axis + " of size " + dim + ".", field);
        const previous = ranges.at(-1);
        if (previous?.[1] === n) previous[1]++;
        else ranges.push([n, n + 1]);
      }
      index[axis] = ranges;
    } else {
      if (!Number.isInteger(part.start) || !Number.isInteger(part.stop) || part.start < 0 || part.start >= part.stop || part.stop > dim)
        throw new InspectError("invalid_range", "Use a nonempty range inside axis " + axis + " of size " + dim + ".", field);
      index[axis] = [[part.start, part.stop]];
    }
  });
  return {tensor: tensor.name, shape: tensor.shape, index};
}

export const indicesFor = (resolved: Resolved) => resolved.shape.map((_, axis) =>
  resolved.index[axis].flatMap(([start, stop]) => Array.from({length: stop - start}, (_, i) => start + i)));
