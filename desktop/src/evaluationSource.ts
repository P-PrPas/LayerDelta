import {connect, get, getResponse, post} from "./api";

export type Task = "classification" | "segmentation" | "detection" | "ocr";
export type Category = "broken" | "fixed" | "changed";
export type Metric = {name: string; primary: boolean; unit: "rate" | "count" | "score"; better: "higher" | "lower";
  value: number | null; baseline: number | null; delta: number | null; ci: [number, number] | null;
  per_class?: (Metric & {class: number; n?: number})[]};
export type JobState = "queued" | "running" | "succeeded" | "failed" | "cancelled";
type Job = {status: JobState; created_at: string; started_at: string | null; finished_at: string | null;
  error_code: string | null; message: string | null; checkpoint_id: string; dataset_id: string; samples: number};
export type Intervention = {node_id?: string; group?: string; units: number[]; op: string; factor?: number | null};
export type Evaluation = Job & {evaluation_id: string; split: string; split_sha256?: string; fingerprint: string; baseline_id: string | null;
  baseline_cached: boolean; intervention?: Intervention; evaluator?: Record<string, unknown> | null; metrics?: Metric[];
  counts?: Record<Category, number> & {samples: number}; resamples?: number | null; recipe_hash?: string | null};
export type Aggregate = {n: number; mean: number[]; std: number[]; l2: number[]; frac_positive: number[];
  histogram?: {edges: number[]; counts: number[][]}};
export type Probe = Job & {probe_id: string; group?: string; split_by?: string | null; limit_samples?: number; min_aggregate?: number;
  nodes?: {node_id: string; group?: string; unit_axis: number; units: number; groups: Record<string, Aggregate>}[];
  skipped_small?: unknown; raw: boolean};
export type Dataset = {dataset_id: string; name: string; task: Task; split_sha256?: Record<string, string | null>; classes?: string[]; categories?: {id: number; name: string}[];
  ignore_index?: number; preprocessing: Record<string, unknown>};
export type SampleRecord = {pred?: number | number[]; inter?: number[]; truth?: number[]; text?: string;
  boxes?: number[][]; tp?: number; fp?: number};
export type Sample = {sample_id: string; label: string | number | null; condition: string | null; before: SampleRecord; after: SampleRecord;
  category: Category; score_before: number; score_after: number};
export type ChangePage = {changes: Sample[]; total: number; offset: number; truncated: boolean; cursor?: string | null};
export type TruthBoxes = {boxes: {bbox: [number, number, number, number]; category_id: number}[]; width: number; height: number};
export type ProbeNode = {module_path: string | null; parameters?: string[]};

const enc = encodeURIComponent;
const samplePath = (dataset: string, id: string) => "/datasets/" + enc(dataset) + "/samples/" + enc(id);
export interface EvaluationSource {
  evaluation(id: string, signal: AbortSignal): Promise<Evaluation>;
  probe(id: string, signal: AbortSignal, activationsReleased: boolean): Promise<Probe>;
  dataset(id: string, signal: AbortSignal): Promise<Dataset>;
  changes(id: string, filter: {category: Category; sort: string; limit: number; cursor?: string}, signal: AbortSignal): Promise<ChangePage>;
  sample(dataset: string, id: string, part: "image" | "label", signal: AbortSignal): Promise<Response>;
  intervention(id: string, signal: AbortSignal): Promise<Intervention | undefined>;
  node(checkpoint: string, node: string, signal: AbortSignal): Promise<ProbeNode>;
}

export const liveEvaluationSource: EvaluationSource = {
  async evaluation(id, signal) {await connect(); return get("/evaluations/" + enc(id), signal);},
  async probe(id, signal, activationsReleased) {
    await connect();
    const {values, ...report} = await get<Omit<Probe, "raw"> & {values?: unknown}>("/probes/" + enc(id), signal);
    // The human can read raw outputs, but the viewer retains only whether the agent received them.
    return {...report, raw: activationsReleased && values != null};
  },
  async dataset(id, signal) {await connect(); return get("/datasets/" + enc(id), signal);},
  async changes(id, {category, sort, limit, cursor}, signal) {
    await connect();
    const query = new URLSearchParams({category, sort, limit: String(limit), ...(cursor ? {cursor} : {})});
    return get("/evaluations/" + enc(id) + "/changes?" + query, signal);
  },
  async sample(dataset, id, part, signal) {
    await connect();
    return getResponse(samplePath(dataset, id) + "/" + part, signal);
  },
  async intervention(id, signal) {
    await connect();
    const {result} = await post<{result: {interventions: Record<string, Intervention>}}>("/interventions/resolve", {ids: [id]}, false, signal);
    return result.interventions[id];
  },
  async node(checkpoint, node, signal) {
    await connect();
    const {result} = await get<{result: {node: ProbeNode}}>("/checkpoints/" + enc(checkpoint) + "/graph/nodes/" + enc(node), signal);
    return result.node;
  },
};
