import {useEffect, useRef, useState} from "react";
import {ApiError} from "./api";
import {pageSize, type TraceEvent, type TraceSource, type TraceSummary} from "./traceSource";

export type TraceFailure = {message: string; code?: string; requestId?: string | null};
export function traceFailure(error: unknown): TraceFailure {
  return error instanceof ApiError ? {message: error.message, code: error.code, requestId: error.requestId} :
    {message: error instanceof Error ? error.message : String(error)};
}
type Feed = {events: TraceEvent[]; summary?: TraceSummary; more: boolean; loading: boolean; error?: TraceFailure};

export function useTraceFeed(source: TraceSource, session: string | undefined, polling: boolean, enabled = true) {
  const [feed, setFeed] = useState<Feed>({events: [], more: false, loading: true});
  const wake = useRef<(manual?: boolean) => void>(() => {});
  const pollingRef = useRef(polling);
  useEffect(() => {
    pollingRef.current = polling;
    wake.current();
  }, [polling]);
  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    const {signal} = controller;
    let after = 0;
    let busy = false, first = true;
    let timer: number | undefined;
    async function read(manual = false) {
      window.clearTimeout(timer);
      if (signal.aborted || busy || (!manual && !first && (document.hidden || !pollingRef.current))) return;
      busy = true;
      try {
        const page = await source.page({session, after, limit: pageSize}, signal);
        if (signal.aborted) return;
        const forward = page.events.filter(event => event.event_id > after);
        after = Math.max(after, page.next_after ?? 0, forward.at(-1)?.event_id ?? 0);
        setFeed(previous => ({events: forward.length ? [...previous.events, ...forward] : previous.events, summary: page.summary,
          more: page.next_after !== null, loading: false}));
      } catch (error) {
        if (!signal.aborted) setFeed(previous => ({...previous, loading: false, error: traceFailure(error)}));
      } finally {
        busy = false; first = false;
        if (!signal.aborted && pollingRef.current && !document.hidden) timer = window.setTimeout(read, 2000);
      }
    }
    function visibility() {
      if (document.hidden) window.clearTimeout(timer);
      else void read();
    }
    wake.current = manual => {void read(manual);};
    document.addEventListener("visibilitychange", visibility);
    void read();
    return () => {
      controller.abort(); window.clearTimeout(timer);
      document.removeEventListener("visibilitychange", visibility);
      wake.current = () => {};
    };
  }, [source, session, enabled]);
  return {...feed, load: () => wake.current(true)};
}
