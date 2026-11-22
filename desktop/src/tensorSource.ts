export type Value = number | boolean | "NaN" | "Inf" | "-Inf";
export type Index = number[] | {start: number; stop: number};
export type Selector = {checkpoint?: string; tensor: string; index?: Record<string, Index>};
export type Resolved = {tensor: string; shape: number[]; index: Record<string, [number, number][]>};
export type Envelope<T, R = Record<string, unknown>> = {
  schema_version: 1; resolved: R; result: T;
  fingerprints: {checkpoint: string | null; selection: string | null; op_revision: 1; checkpoint_b?: string};
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
export type Read = {
  tensor: string; dtype: string; shape: number[]; selection_shape: number[]; total_elems: number;
  offset: number; returned_elems: number; values: Value[]; nonfinite_count: number | null;
  range: Range | null; range_skipped?: "selection_too_large";
};
export type Unit = {
  unit: number | null; index: Record<string, number>; count: number; nonfinite: number; min: number | null;
  max: number | null; mean: number | null; std: number | null; l1: number | null;
  l2: number | null; zero_frac: number | null; constant: boolean;
};
export type Stats = {
  tensor: string; dtype: string; reduce_over_axes: number[] | null; unit_axes: number[];
  unit_count: number; offset: number; accumulate_dtype: "float64"; units: Unit[];
  histogram?: {edges: number[]; counts: number[][]} | null; profile?: null;
};
export type DiffStatus = "same" | "changed" | "only_a" | "only_b" | "shape_mismatch" | "dtype_mismatch";
type Metrics = {changed_elems: number; max_abs: number | null; rel_l2: number | null; cosine: number | null};
export type DiffUnit = Metrics & {unit: number};
export type DiffTensor = Partial<Metrics> & {
  name: string; status: DiffStatus; dtype?: string; shape?: number[]; numel?: number; basis?: "values";
  unit_axis?: number | null; changed_units?: {total: number; listed: number[]; truncated: boolean};
};
export type ModelDiff = {accumulate_dtype: "float64"; summary: {
  tensors_total: number; changed: number; same: number; only_a: string[]; only_b: string[];
  shape_mismatch: string[]; dtype_mismatch: string[]; changed_elems_total: number; max_abs: number | null;
}; tensors: DiffTensor[]};
type SelectionDiff = Metrics & {accumulate_dtype: "float64"; basis: "values"};
export type UnitDiff = SelectionDiff & {units: DiffUnit[]};
export type ValueDiff = SelectionDiff & Pick<Read, "offset" | "total_elems" | "returned_elems" | "values" | "range">
  & {changed: (0 | 1)[]; delta_dtype: "float64"};
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
export type ImportProgress = {stage: "header" | "extents" | "table"; done: number; total: number};
export class InspectError extends Error {
  constructor(public code: string, message: string, public field: string | null = null, public missing?: string[]) {
    super(message);
  }
}
type SelectionReply<T> = Envelope<T, {selector: Resolved}>;
export interface TensorDataSource {
  readonly fixture?: boolean;
  listCheckpoints(): Promise<Checkpoint[]>;
  importCheckpoint(input: {path: string; file?: File}, progress: (stage: ImportProgress | null) => void): Promise<Checkpoint>;
  describe(id: string, request: DescribeRequest): Promise<Envelope<Description>>;
  read_tensor(id: string, request: ReadRequest): Promise<SelectionReply<Read>>;
  stats(id: string, request: StatsRequest): Promise<SelectionReply<Stats>>;
  diff(a: string, b: string, request: DescribeRequest & {selector?: null; include_same?: boolean; unit_axis?: number | null}): Promise<Envelope<ModelDiff>>;
  diff(a: string, b: string, request: PageRequest & {selector: Selector; unit_axis: number; values?: false}): Promise<SelectionReply<UnitDiff>>;
  diff(a: string, b: string, request: ReadRequest & {unit_axis: null; values: true}): Promise<SelectionReply<ValueDiff>>;
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
