import {connect, get, post} from "./api";
import type {Envelope, Role, Tensor} from "./tensorSource";
export type {Role} from "./tensorSource";
export type GraphTensor = Omit<Tensor, "roles"> & {roles?: Role[]; binding?: "bound" | "unused" | "unbound"; groups?: {group_id: string; axis: number}[]};
export type Axes = {derived: number; declared: number; unknown: number; inferred?: number};
export type Coverage = Axes & {opaque_ops: {op: string; count: number}[]; tensors: Record<"bound" | "unused" | "unbound", number>; groups: Record<"closed" | "open", number>};
type Provider = {name: string; enabled: boolean; reason: string | null; models?: string[]};
type Reuse = {ir_sha256: string; provider: string; checkpoints: number};
type Offers = {providers: Provider[]; reusable: Reuse[]};
type Graph = Offers & {ir_sha256: string; provider: string; table_signature: string; attached_at: string};
export type GraphNode = {id: string; op: string; module_path: string | null; module_type: string | null; outputs: {value: string; kind: string; shape?: number[]}[]; rule: string | null; opaque_reason: string | null};
export type NodeDetail = GraphNode & {args: unknown[]; kwargs: Record<string, unknown>; parameters: string[]; groups: string[]};
export type Module = {path: string; parent: string | null; type: string | null; node_count: number; tensor_count: number; axes: Axes; opaque_count: number; open_groups: number};
export type Mapping = {offset: number; block: number; period: number; repeat: number};
export type Member = Mapping & {tensor: string; axis: number; role: string};
export type Group = {group_id: string; label: string; size: number; closed: boolean; kind: string; constraints: {kind: string; n?: number}; notes: string[];
  boundary: {node: string | null; op: string | null; reason: string; output?: number}[];
  members: Member[]; spans: (Mapping & {value: string; dim: number})[]; spans_truncated: boolean};
export type GraphSummary = {graph: Graph; coverage: Coverage; warnings: string[]; counts: {nodes: number; params: number};
  inputs: {value: string; shape: number[]}[]; outputs: string[]};
type GraphPage = ({graph: null} & Offers) | (GraphSummary & {nodes?: GraphNode[]; modules?: Module[]});
export type NavigatorData = {summary: GraphSummary | null; offers: Offers; modules: Module[]; nodes: GraphNode[]; groups: Group[]; tensors: GraphTensor[]};
type AttachJob = {attach_id: string; status: "queued" | "running" | "succeeded" | "failed";
  stage: string; done: number; total: number; error: {code: string; message: string; names?: string[]} | null};

function checkpointRoute(id: string, route: string) {
  return "/checkpoints/" + encodeURIComponent(id) + "/" + route;
}

async function pages<T, Row>(route: string, rows: (page: T) => Row[], signal: AbortSignal, first?: Envelope<T>) {
  const out: Row[] = [];
  let page = first;
  do {
    signal.throwIfAborted();
    page ??= await get<Envelope<T>>(route);
    for (const row of rows(page.result)) out.push(row);
    if (!page.cursor) return out;
    const cursor = page.cursor;
    signal.throwIfAborted();
    page = await get<Envelope<T>>(route + "&cursor=" + encodeURIComponent(cursor));
  } while (true);
}

export async function loadNavigator(id: string, signal: AbortSignal): Promise<NavigatorData> {
  await connect();
  const route = (name: string) => checkpointRoute(id, name);
  const tensors = pages<{tensors: GraphTensor[]}, GraphTensor>(route("describe?limit=2000"), p => p.tensors, signal);
  const summary = get<Envelope<GraphPage>>(route("graph?limit=500"));
  const [first, table] = await Promise.all([summary, tensors]);
  if (!first.result.graph) return {summary: null, offers: first.result, modules: [], groups: [], nodes: [], tensors: table};
  const [nodes, modules, groups] = await Promise.all([
    pages<GraphPage, GraphNode>(route("graph?limit=500"), p => "nodes" in p ? p.nodes ?? [] : [], signal, first),
    pages<GraphPage, Module>(route("graph?view=modules&limit=500"), p => "modules" in p ? p.modules ?? [] : [], signal),
    pages<{groups: Group[]}, Group>(route("groups?limit=200"), p => p.groups, signal),
  ]);
  return {summary: first.result, offers: first.result.graph, nodes, modules, groups, tensors: table};
}

export async function getGraphNode(id: string, node: string) {
  return (await get<Envelope<{node: NodeDetail}>>(checkpointRoute(id, "graph/nodes/" + encodeURIComponent(node)))).result.node;
}

export async function attachGraph(id: string, spec: {provider: "builtin"; model: string} | {reuse: string}, observe: (job: AttachJob) => void, signal: AbortSignal) {
  let job = await post<AttachJob>(checkpointRoute(id, "graphs"), spec);
  const stop = Date.now() + 120_000;
  while (true) {
    signal.throwIfAborted();
    observe(job);
    if (job.status === "succeeded") return;
    if (job.status === "failed") {
      const names = job.error?.names ?? [];
      throw new Error([job.error?.code, job.error?.message, names.slice(0, 20).join(", "), names.length > 20 ? "and " + (names.length - 20) + " more" : ""].filter(Boolean).join(" · "));
    }
    if (Date.now() >= stop) throw new Error("The graph is still building. Refresh to check for the attachment.");
    await new Promise(resolve => setTimeout(resolve, 300));
    signal.throwIfAborted();
    job = await get<AttachJob>("/graph-attachments/" + encodeURIComponent(job.attach_id));
  }
}

export async function groupProfile(id: string, group: Group, member: Member, signal: AbortSignal) {
  type Result = {members: {tensor: string; axis: number; units: {unit: number; l2: number | null}[]}[]};
  const rows = new Map<number, number | null>();
  const stop = Math.min(64, group.size);
  let cursor: string | null = null;
  // shortcut: show the first 64 profile positions until the inspector has a paged profile.
  do {
    signal.throwIfAborted();
    const reply: Envelope<Result> = await post(checkpointRoute(id, "stats"), {
      selector: {group: group.group_id, units: {start: 0, stop}}, limit: 1000, cursor,
    });
    for (const row of reply.result.members)
      if (row.tensor === member.tensor && row.axis === member.axis)
        for (const unit of row.units) rows.set(unit.unit, unit.l2);
    cursor = reply.cursor;
  } while (cursor && rows.size < stop);
  return rows;
}
