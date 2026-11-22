export type Device = "cpu" | "cuda";
export interface Target {
  target_id: string; mode: "activation" | "weight"; module_path: string;
  channel_count: number; parameter_name: string | null;
}
export interface Intervention {
  target_id: string; mode: Target["mode"]; channel_index: number; operation: "zero" | "scale"; scale?: number;
}
export interface Recipe {
  recipe_id: string; recipe_hash: string; hypothesis: string; intervention: Intervention;
  dependencies: { checkpoint_sha256: string };
}
export interface Sample { sample_id: string; source_sample_id: string; condition: string; label: number }
export interface Catalog {
  targets: Target[]; devices: {id: Device; available: boolean}[];
  checkpoints: {checkpoint_id: string; sha256: string}[]; dataset_revision_id: string;
  model: {layers: {layer_id:string; type:string; output_shape:number[]; read_only:boolean}[]};
  demo: {truth: number; baseline_prediction: number; example_pairs: {source_sample_id: string}[]};
}
export interface Job {
  run_id: string; kind: string; status: string; recipe_id?: string;
  device?: string; actual_device?: string; split?: string; error?: {message?: string; code?: string} | string;
}
export interface Stats { mean: number; max: number; zero_fraction: number; shape: number[] }
export interface Preview {
  actual_device: string; hook: string; image: number[][]; kernel: Stats;
  activation: Stats & {values: number[][]};
}
export interface Slice {
  slice_id: string; n: number; accuracy_before: number|null; accuracy_after: number|null;
  delta_pp: number|null; fixed: number; regressed: number; changed_wrong: number;
  confusion_before: number[][]; confusion_after: number[][];
}
export interface Change {
  sample_id: string; source_sample_id: string; condition: string; truth: number;
  before: number; after: number; category: string;
}
export interface Comparison {
  primary_slice_id: string; guardrail_slice_id: string; slices: Slice[]; changes?: Change[];
  baseline_run_id?: string; recipe_hash?: string; runtime?: {actual_device?: string};
}
export interface Page<T> { items: T[]; next_cursor: string | null }
interface Session {project_id: string; csrf: string}
let session: Session;
let connection: Promise<void> | undefined;

export class ApiError extends Error {
  constructor(message: string, public code: string, public status: number, public field: string | null = null) {
    super(message);
  }
}

async function asJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const error = body.error ?? body.detail;
    throw new ApiError(error?.message || response.statusText, error?.code || "http_error", response.status, error?.field ?? null);
  }
  return response.json();
}

export function connect() {
  return connection ??= connectHuman().catch(error => {connection = undefined; throw error;});
}

async function connectHuman() {
  const ticket = location.hash.slice(1);
  if (ticket) {
    history.replaceState(null, "", location.pathname + location.search);
    session = await asJson<Session>(await fetch("/api/v1/session", {
      method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify({ticket}),
    }));
    sessionStorage.setItem("layerdelta-session", JSON.stringify(session));
  } else {
    const saved = sessionStorage.getItem("layerdelta-session");
    if (!saved) throw new Error("Open LayerDelta from its desktop launcher to start a session.");
    session = JSON.parse(saved);
  }
}

export function path(route: string) {
  return "/api/v1/projects/" + encodeURIComponent(session.project_id) + route;
}

export async function get<T>(route: string): Promise<T> {
  return asJson<T>(await fetch(path(route), {cache: "no-store"}));
}

export async function post<T>(route: string, data: unknown, job = false): Promise<T> {
  const headers: Record<string,string> = {"X-CSRF-Token": session.csrf};
  let body: BodyInit;
  if (data instanceof FormData) body = data;
  else if (data instanceof Blob) {
    headers["Content-Type"] = "application/octet-stream";
    body = data;
  } else {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(data);
  }
  if (job) headers["Idempotency-Key"] = crypto.randomUUID();
  const submit = () => fetch(path(route), {method: "POST", headers, body});
  // Recover a lost response with the same job key.
  return asJson<T>(await (job ? submit().catch(submit) : submit()));
}

export async function waitFor<T>(job: Job, route: string, observe: (job: Job) => void) {
  const deadline = Date.now() + 15 * 60_000;
  for (;;) {
    observe(job);
    if (job.status === "succeeded") return job as Job & {result: T};
    if (["failed", "cancelled"].includes(job.status)) {
      throw new Error(typeof job.error === "string" ? job.error : job.error?.message || job.status);
    }
    if (Date.now() >= deadline) throw new Error("Polling timed out. Check this job in History.");
    await new Promise(resolve => setTimeout(resolve, 600));
    job = await get<Job>(route + "/" + encodeURIComponent(job.run_id));
  }
}

export function saveJson(name: string, value: unknown) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], {type:"application/json"}));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 30_000);
}
