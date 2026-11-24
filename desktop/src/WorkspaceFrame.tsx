import {useEffect, useState, type ReactNode} from "react";
import type {Checkpoint} from "./tensorSource";
import {ThemeSetting} from "./theme";
import "./frame.css";

const icons = {
  Architecture: <><rect x="2" y="2" width="5" height="4" rx="1"/><rect x="9" y="10" width="5" height="4" rx="1"/><path d="M4.5 6v3.5c0 .8.7 1.5 1.5 1.5h3"/></>,
  Tensors: <><rect x="2.5" y="2.5" width="11" height="11" rx="1"/><path d="M2.5 6.5h11M2.5 10h11M6.5 2.5v11M10 2.5v11"/></>,
  Sessions: <path d="M3 4h10M3 8h7M3 12h9"/>,
  Runs: <path d="M5 3.5v9l7-4.5z"/>,
  "Final evaluation": <><rect x="3.5" y="7" width="9" height="6.5" rx="1"/><path d="M5.5 7V5a2.5 2.5 0 0 1 5 0v2"/></>,
  Settings: <><circle cx="8" cy="8" r="2"/><path d="M8 1.8v1.8M8 12.4v1.8M1.8 8h1.8M12.4 8h1.8M3.6 3.6l1.3 1.3M11.1 11.1l1.3 1.3M3.6 12.4l1.3-1.3M11.1 4.9l1.3-1.3"/></>,
  Rail: <><rect x="2" y="2.5" width="12" height="11" rx="1"/><path d="M6 2.5v11"/></>,
  Copy: <><rect x="5.5" y="5.5" width="8" height="8" rx="1"/><path d="M10.5 5.5v-2a1 1 0 0 0-1-1h-6a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h2"/></>,
};
function Icon({name}: {name: keyof typeof icons}) {
  return <svg className="frame-icon" aria-hidden="true" viewBox="0 0 16 16" fill="none" stroke="currentColor"
    strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">{icons[name]}</svg>;
}

const wideWindow = matchMedia("(min-width: 1200px)");
const places = [
  {label: "Model", items: ["Architecture", "Tensors"]},
  {label: "Agents", items: ["Sessions"]},
  {label: "Experiments", items: ["Runs", "Final evaluation"]},
] as const;
export type Place = typeof places[number]["items"][number];

function shortName(name: string) {
  const letters = [...name];
  return letters.length <= 36 ? name : letters.slice(0, 19).join("") + "…" + letters.slice(-16).join("");
}

export function WorkspaceFrame({
  place, navigate, checkpoints, checkpoint, pickCheckpoint, checkpointLocked, graphAttached, dataset, connected,
  live, pending, device, job, budget, navLocked, children,
}: {
  place: Place; navigate: (place: Place) => void; checkpoints?: Checkpoint[]; checkpoint: string;
  pickCheckpoint: (id: string) => void; checkpointLocked: boolean; graphAttached: boolean; dataset?: string; connected: boolean;
  live: boolean; pending: number; device: ReactNode; job: ReactNode; budget?: ReactNode; navLocked: boolean; children: ReactNode;
}) {
  const [rail, setRail] = useState<"open" | "closed">();
  const [wide, setWide] = useState(wideWindow.matches);
  useEffect(() => {
    const update = () => setWide(wideWindow.matches);
    wideWindow.addEventListener("change", update);
    return () => wideWindow.removeEventListener("change", update);
  }, []);
  const railOpen = rail === undefined ? wide : rail === "open";
  const [copied, setCopied] = useState("");
  const active = checkpoints?.find(entry => entry.checkpoint_id === checkpoint);
  return <div className="workbench-shell workspace-frame" data-rail={railOpen ? "open" : "closed"}>
    <a className="skip" href="#work">Skip to workbench</a>
    <header className="frame-top">
      <div className="frame-brand"><button type="button" className="frame-icon-button" onClick={() => setRail(railOpen ? "closed" : "open")}
        aria-label="Toggle navigation rail" title="Toggle navigation rail" aria-controls="workspace-nav" aria-expanded={railOpen}><Icon name="Rail"/></button>
        <strong>LayerDelta</strong></div>
      <details className="frame-workspace">
        <summary title="Workspace details">Local workspace</summary>
        <dl><dt>Dataset revision</dt><dd>{dataset ?? "—"}</dd></dl>
      </details>
      <span aria-hidden="true" className="frame-divider">/</span>
      <select aria-label="Checkpoint" className="frame-checkpoint" title={active?.display_name}
        value={active ? checkpoint : ""} disabled={checkpointLocked || !checkpoints?.length}
        onChange={event => {setCopied(""); pickCheckpoint(event.target.value);}}>
        {!active && <option value="">{checkpoints ? "No checkpoint" : "Loading checkpoints…"}</option>}
        {checkpoints?.map(entry => <option key={entry.checkpoint_id} value={entry.checkpoint_id} title={entry.display_name}>
          {shortName(entry.display_name)} · {entry.checkpoint_id.slice(0, 8)}
        </option>)}
      </select>
      {active && <button type="button" className="frame-icon-button" title="Copy full checkpoint SHA-256"
        aria-label="Copy full checkpoint SHA-256" onClick={() => {
          navigator.clipboard.writeText(checkpoint).then(() => setCopied("Checkpoint SHA-256 copied."),
            () => setCopied("Could not copy the checkpoint SHA-256."));
        }}><Icon name="Copy"/></button>}
      <span className="visually-hidden" role="status">{copied}</span>
      {graphAttached && <span className="frame-graph-tag">Graph attached</span>}
    </header>
    <aside className="workspace-rail frame-rail">
      <nav id="workspace-nav" aria-label="Workbench">
        {places.map(group => <section key={group.label} aria-label={group.label}>
          <h2>{group.label}</h2>
          <ul>{group.items.map(item => <li key={item}>
            <button type="button" title={item} aria-label={item} aria-current={place === item ? "page" : undefined}
              disabled={navLocked && item !== "Tensors" && place !== "Tensors"} onClick={() => navigate(item)}>
              <Icon name={item}/><span className="frame-rail-label">{item}</span>
              {item === "Sessions" && pending > 0 && <span className="frame-pending" title={pending + " pending proposals"}>{pending}</span>}
              {item === "Sessions" && live && <span className="frame-live" role="img" aria-label="Agent active in the last minute"/>}
            </button>
          </li>)}</ul>
        </section>)}
      </nav>
      <details className="frame-settings"><summary title="Settings" aria-label="Settings"><Icon name="Settings"/>
        <span className="frame-rail-label">Settings</span></summary>
        <div><ThemeSetting/></div>
      </details>
    </aside>
    <div className="workspace-canvas frame-content">{children}</div>
    <footer className="frame-status">
      <span className="frame-connection"><i data-connected={connected} aria-hidden="true"/>
        {connected ? "Connected to local server" : "Local server disconnected"}</span>
      {device}
      <div className="frame-job" role="status">{job}</div>
      {budget}
    </footer>
  </div>;
}
