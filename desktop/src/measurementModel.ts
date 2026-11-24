import type {Metric, Sample, Task} from "./evaluationSource";
import type {TraceEvent} from "./traceSource";

export function reading(metric: Metric) {
  if (!metric.primary || !metric.ci) return "";
  const [lo, hi] = metric.ci;
  if (lo <= 0 && hi >= 0) return "No clear change";
  return (lo > 0) === (metric.better === "higher") ? "Improved" : "Worse";
}
export function metricNumber(n: number | null, unit: Metric["unit"], delta = false) {
  if (n == null) return "—";
  const value = unit === "rate" ? (n * 100).toFixed(2) : unit === "count" ? n.toLocaleString("en-US") : n.toFixed(3);
  return (delta && n > 0 ? "+" : "") + value + (unit === "rate" ? (delta ? " pp" : "%") : "");
}
export function intervalDomain(metrics: Metric[]) {
  const domains = {rate: 0, count: 0, score: 0};
  for (const m of metrics) {
    domains[m.unit] = Math.max(domains[m.unit], Math.abs(m.delta ?? 0), ...((m.ci ?? []).map(Math.abs)));
  }
  return domains;
}
export function evaluationLinks(events: TraceEvent[], event?: TraceEvent) {
  const links = new Set<number>();
  if (!event) return links;
  function addApply(apply: TraceEvent) {
    for (const seq of [...(apply.links?.plan_seqs ?? []), ...(apply.links?.selection_seqs ?? [])]) links.add(seq);
    if (apply.links?.proposal_seq !== undefined) links.add(apply.links.proposal_seq);
  }
  if (event.links?.apply_seq !== undefined) {
    links.add(event.links.apply_seq);
    const apply = events.find(row => row.seq === event.links?.apply_seq);
    if (apply) addApply(apply);
  }
  if (event.tool === "apply" || event.route.endsWith("/apply") || event.control?.action === "approve") {
    addApply(event);
    for (const row of events) if (row.links?.apply_seq === event.seq) links.add(row.seq);
  }
  return links;
}
export function originOf(events: TraceEvent[], event: TraceEvent) {
  return events.find(row => (row.tool === "evaluate" && row.evaluation_id && row.evaluation_id === event.evaluation_id ||
    row.tool === "probe" && row.probe_id && row.probe_id === event.probe_id) && row.replayed !== true);
}
export const categoryLabels = {broken: "Broken", fixed: "Fixed", changed: "Changed, still wrong"} as const;
export const predicates: Record<Task, string> = {
  classification: "The predicted class equals the truth.",
  segmentation: "Sample mean IoU is at least 0.5.",
  detection: "Every truth box is matched at IoU 0.5, with no false positives.",
  ocr: "The decoded string matches the truth exactly.",
};

export function textAlignment(before: string, after: string) {
  const a = Array.from(before), b = Array.from(after);
  // Keep unusually long output responsive. A changed suffix is still marked.
  if (a.length * b.length > 100_000) {
    const prefix = a.findIndex((ch, i) => ch !== b[i]);
    const same = prefix < 0 ? Math.min(a.length, b.length) : prefix;
    return {before: a.map((text, i) => ({text, changed: i >= same})), after: b.map((text, i) => ({text, changed: i >= same}))};
  }
  const n = b.length + 1;
  const cost = new Uint32Array((a.length + 1) * n);
  for (let i = a.length; i >= 0; i--) for (let j = b.length; j >= 0; j--) {
    const k = i * n + j;
    cost[k] = i === a.length ? b.length - j : j === b.length ? a.length - i :
      a[i] === b[j] ? cost[k + n + 1] : 1 + Math.min(cost[k + n + 1], cost[k + n], cost[k + 1]);
  }
  const left = [], right = [];
  let i = 0, j = 0;
  while (i < a.length || j < b.length) {
    const k = i * n + j;
    if (i < a.length && j < b.length && (a[i] === b[j] || cost[k] === 1 + cost[k + n + 1])) {
      const changed = a[i] !== b[j];
      left.push({text: a[i++], changed}); right.push({text: b[j++], changed});
    } else if (i < a.length && cost[k] === 1 + cost[k + n]) {
      left.push({text: a[i++], changed: true});
    } else {
      right.push({text: b[j++], changed: true});
    }
  }
  return {before: left, after: right};
}
export function largestIouDrop(sample: Sample) {
  let result: {index: number; delta: number} | undefined;
  sample.before.inter?.forEach((before, i) => {
    const after = sample.after.inter?.[i];
    if (after === undefined) return;
    const pre = (Array.isArray(sample.before.pred) ? sample.before.pred[i] ?? 0 : 0) + (sample.before.truth?.[i] ?? 0) - before;
    const post = (Array.isArray(sample.after.pred) ? sample.after.pred[i] ?? 0 : 0) + (sample.after.truth?.[i] ?? 0) - after;
    if (!pre || !post) return;
    const delta = after / post - before / pre;
    if (delta < 0 && (!result || delta < result.delta)) result = {index: i, delta};
  });
  return result;
}
