import {createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode} from "react";
import {liveEvaluationSource, type Evaluation, type EvaluationSource, type Probe} from "./evaluationSource";

type Entry = {record?: Evaluation | Probe; failed?: boolean};
type Store = {read: (id: string) => Entry | undefined; watch: (id: string, target: HTMLElement, raw: boolean) => () => void; source: EvaluationSource};
const Context = createContext<Store | null>(null);
export function useMeasurementStore() {return useContext(Context)!;}

export function MeasurementJobs({children, source = liveEvaluationSource}: {children: ReactNode; source?: EvaluationSource}) {
  const [entries, setEntries] = useState(new Map<string, Entry>());
  const jobs = useRef(new Map<string, {record?: Evaluation | Probe; failed?: boolean; readers: number; raw: boolean;
    request?: AbortController; timer?: number}>());
  const run = useCallback(function run(id: string) {
    const entry = jobs.current.get(id);
    if (!entry || !entry.readers || document.hidden || entry.request || entry.failed) return;
    if (entry.record && !["queued", "running"].includes(entry.record.status)) return;
    const controller = new AbortController();
    entry.request = controller;
    const result = id.startsWith("pr_") ? source.probe(id, controller.signal, entry.raw) : source.evaluation(id, controller.signal);
    const publish = () => setEntries(old => new Map(old).set(id, {record: entry.record, failed: entry.failed}));
    void result.then(record => {
      if (controller.signal.aborted) return;
      entry.record = record;
      publish();
      if (entry.readers && !document.hidden && ["queued", "running"].includes(record.status)) {
        entry.timer = window.setTimeout(() => run(id), 2000);
      }
    }).catch(() => {
      if (!controller.signal.aborted) {entry.failed = true; publish();}
    }).finally(() => {
      if (entry.request === controller) entry.request = undefined;
    });
  }, [source]);
  const watch = useCallback((id: string, target: HTMLElement, raw: boolean) => {
    const entry = jobs.current.get(id) ?? {readers: 0, raw};
    jobs.current.set(id, entry);
    if (raw && !entry.raw) {
      entry.raw = true;
      entry.record = undefined;
    }
    let watching = false;
    const observer = new IntersectionObserver(([item]) => {
      if (item.isIntersecting === watching) return;
      watching = item.isIntersecting;
      if (watching) {
        entry.readers++;
        window.clearTimeout(entry.timer);
        run(id);
      } else if (--entry.readers === 0) {
        window.clearTimeout(entry.timer);
      }
    });
    observer.observe(target);
    return () => {
      observer.disconnect();
      if (watching && --entry.readers === 0) window.clearTimeout(entry.timer);
    };
  }, [run]);
  useEffect(() => {
    const records = jobs.current;
    const visible = () => {
      for (const [id, entry] of records) {
        window.clearTimeout(entry.timer);
        if (!document.hidden) run(id);
      }
    };
    document.addEventListener("visibilitychange", visible);
    return () => {
      document.removeEventListener("visibilitychange", visible);
      for (const entry of records.values()) {entry.request?.abort(); window.clearTimeout(entry.timer);}
      records.clear();
    };
  }, [run]);
  return <Context.Provider value={{read: id => entries.get(id), watch, source}}>{children}</Context.Provider>;
}

export function JobChip({id, raw = false, now}: {id: string; raw?: boolean; now: number}) {
  const {read, watch} = useMeasurementStore();
  const target = useRef<HTMLSpanElement>(null);
  useEffect(() => watch(id, target.current!, raw), [id, raw, watch]);
  const value = read(id)?.record;
  const elapsed = value?.started_at ? Math.max(0, Math.floor((Date.parse(value.finished_at ?? "") || now) / 1000 - Date.parse(value.started_at) / 1000)) : null;
  return <span ref={target} className="ev-job-anchor">{value && <span className="tv-flag ev-job" data-job={value.status} title={value.message ?? undefined}>
    {value.status}{value.error_code ? " · " + value.error_code : elapsed === null ? "" : " · " + elapsed + " s"}</span>}
    {value && "raw" in value && value.raw && <span className="tv-flag ev-job" data-job="failed">raw</span>}</span>;
}
