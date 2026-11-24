import {useEffect, useRef, useState} from "react";
import type {ReactNode} from "react";
import {Catalog, Device, Job, post, Preview, Sample, Target, waitFor, displayError} from "./api";
import {percent, Pixels} from "./Results";

const shortLayer = (name: string) => name.replace("features.", "").replace("block1", "Block 1").replace("block2", "Block 2");

export function ChannelFields({targets, target, channel, onTarget, onChannel}: {
  targets: Target[]; target?: Target; channel: number;
  onTarget: (id: string) => void; onChannel: (n: number) => void;
}) {
  const max = (target?.channel_count ?? 1) - 1;
  const invalid = !Number.isInteger(channel) || channel < 0 || channel > max;
  return <div className="channel-fields">
    <label>Layer and mode<select value={target?.target_id ?? ""} onChange={e => onTarget(e.target.value)}>
      {targets.map(t => <option key={t.target_id} value={t.target_id} title={t.target_id}>{shortLayer(t.module_path.split(".").slice(0,-1).join("."))} · {t.mode === "weight" ? "Conv kernel" : "Post-ReLU"}</option>)}
    </select></label>
    <label>Output channel<input type="number" min={0} max={max} step={1} aria-invalid={invalid}
      aria-describedby="channel-range" value={Number.isNaN(channel) ? "" : channel}
      onChange={e => onChannel(e.target.valueAsNumber)}/></label>
    <small id="channel-range" role={invalid ? "alert" : undefined}>Channel indices 0–{max}</small>
    {target && <details className="target-mapping"><summary>Target mapping</summary><dl className="record-list">
      <dt>Target ID</dt><dd>{target.target_id}</dd>
      <dt>{target.parameter_name ? "Parameter" : "Post-ReLU hook"}</dt><dd>{target.parameter_name ?? target.module_path}</dd>
    </dl></details>}
  </div>;
}

function Architecture({catalog, target, choose}: {catalog: Catalog; target?: Target; choose: (id: string) => void}) {
  const groups: {id: string; layers: Catalog["model"]["layers"]}[] = [];
  for (const layer of catalog.model.layers) {
    const id = layer.layer_id.startsWith("features.") ? layer.layer_id.split(".").slice(0,2).join(".") : layer.layer_id;
    const last = groups.at(-1);
    if (last?.id === id) last.layers.push(layer);
    else groups.push({id, layers:[layer]});
  }
  return <section className="topology" aria-labelledby="topology-heading">
    <div className="section-title"><h2 id="topology-heading">Model topology</h2>
      <span className="meta-text">{catalog.model.layers.length} layers</span></div>
    <ol className="layer-path">{groups.map(({id, layers}) => {
      const option = catalog.targets.find(t => t.module_path.startsWith(id + ".") && t.mode === (target?.mode ?? "activation"));
      const node = <><span className={"layer-glyph " + (option ? "convolution" : "read-only")} aria-hidden="true"/>
        <span><strong>{shortLayer(id)}</strong><small>{layers.at(-1)!.output_shape.join(" × ")}</small></span></>;
      return <li key={id}>{option ? <button type="button" onClick={() => choose(option.target_id)}
        aria-pressed={target?.module_path.startsWith(id + ".")} title={"Inspect " + shortLayer(id)}>{node}</button> :
        <div className="static-layer" title="Read-only layer">{node}</div>}</li>;
    })}</ol>
    <details className="layer-details"><summary>Layer shapes and access</summary><div className="table-scroll"><table>
      <thead><tr><th>Layer</th><th>Output shape</th><th>Channel access</th></tr></thead><tbody>
        {catalog.model.layers.map(l => <tr key={l.layer_id}><td>{l.layer_id}</td><td>{l.output_shape.join(" × ")}</td><td>{l.read_only ? "Read only" : "Editable"}</td></tr>)}
      </tbody></table></div></details>
  </section>;
}

type PreviewState = {key: string; phase:"pending"|"ready"|"error"|"cancelled"; data?: Preview; message?: string};

export function Explorer({catalog, target, channel, choose, device, checkpoint, sample, controls, observePreview}: {
  catalog: Catalog; target?: Target; channel: number; choose: (id: string) => void; device: Device;
  checkpoint: string; sample?: Sample; controls: ReactNode; observePreview: (cancel: (() => void) | null) => void;
}) {
  const valid = sample && target && Number.isInteger(channel) && channel >= 0 && channel < target.channel_count;
  const query = valid ? JSON.stringify({sample_id:sample.sample_id, target_id:target.target_id,
    channel_index:channel, device, checkpoint_id:checkpoint}) : "";
  const [revision, setRevision] = useState(0);
  const [state, setState] = useState<PreviewState>();
  const cancelCurrent = useRef<(() => void) | null>(null);
  const key = query + "|" + revision;
  const current = state?.key === key ? state : undefined;
  const preview = current?.data;
  const pending = current?.phase === "pending";
  useEffect(() => {
    if (!pending) return;
    observePreview(() => cancelCurrent.current?.());
    return () => observePreview(null);
  }, [observePreview, pending]);

  useEffect(() => {
    if (!query) return;
    let cancelled = false;
    let submitted: Job | undefined;
    let stopSent = false;
    const requestStop = () => {
      if (submitted && !stopSent && (submitted.status === "running" || submitted.status === "queued")) {
        stopSent = true;
        void post("/runs/" + submitted.run_id + "/cancel", {}).catch(() => {});
      }
    };
    const cancel = () => {
      cancelled = true;
      requestStop();
    };
    cancelCurrent.current = () => {cancel(); setState({key, phase:"cancelled"});};
    const timer = window.setTimeout(async () => {
      if (cancelled) return;
      setState({key, phase:"pending"});
      try {
        submitted = await post<Job>("/previews", JSON.parse(query), true);
        if (cancelled) {requestStop(); return;}
        const done = await waitFor<Preview>(submitted, "/previews", job => {
          submitted = job;
          if (cancelled) throw new Error("Preview superseded");
        });
        if (!cancelled) setState({key, phase:"ready", data:done.result});
      } catch (e) {
        if (!cancelled) setState({key, phase:"error", message:displayError(e, "Could not compute the channel preview.")});
      }
    }, 320);
    return () => {window.clearTimeout(timer); cancel(); cancelCurrent.current = null;};
  }, [key, query]);

  const description = !query ? "Choose a source image and a valid channel." :
    current?.phase === "cancelled" ? "Preview cancelled." : "Computing channel preview…";
  return <>
    <Architecture catalog={catalog} target={target} choose={choose}/>
    <div className="inspect-grid">
      <section className="inspect-settings"><h2>Sample & channel</h2>
        {controls}
        <div className="inspect-actions"><button data-test="preview" disabled={!query || current?.phase === "pending"}
          onClick={() => setRevision(r => r+1)}>Preview channel</button></div>
      </section>
      <section className="inspection" aria-labelledby="channel-heading">
        <div className="section-title"><h2 id="channel-heading">Channel preview</h2>
          <span className="meta-text">{target ? shortLayer(target.module_path) + " / ch " + (Number.isInteger(channel) ? channel : "—") : "—"}</span></div>
        {preview && sample ? <div data-test="preview-result">
          <p className="preview-id">{sample.sample_id} · truth {sample.label} · {preview.actual_device}</p>
          <div className="tensor-pair">{(["input","activation"] as const).map(kind => {
            const input = kind === "input";
            const matrix = input ? preview.image : preview.activation.values;
            return <div className="tensor" key={kind}><div className="tensor-title">
              <h3>{input ? "Input image" : "Post-ReLU activation"}</h3><span>{matrix[0].length} × {matrix.length}</span>
            </div><div className={"tensor-scale " + (input ? "gray-scale" : "activation-scale")}><span aria-hidden="true"/>Intensity 0–{input ? "1" : preview.activation.max.toPrecision(4)}</div>
            <div className="tensor-surface"><Pixels matrix={matrix} normalize={!input}
              label={input ? "MNIST " + sample.condition + " digit " + sample.label : "Activation map for " + preview.hook}/></div></div>;
          })}</div>
          <div className="table-scroll"><table className="tensor-stats"><caption>Channel statistics</caption>
            <thead><tr><th>Tensor</th><th>Shape</th><th>Mean</th><th>Max</th><th>Zeros</th></tr></thead>
            <tbody>{(["activation","kernel"] as const).map(k => <tr key={k}><th>{k}</th>
              <td>{preview[k].shape.join(" × ")}</td><td>{preview[k].mean.toPrecision(5)}</td>
              <td>{preview[k].max.toPrecision(5)}</td><td>{percent(preview[k].zero_fraction)}</td></tr>)}</tbody></table></div>
        </div> : <div className="await-preview">
          {current?.phase === "error" ? <p role="alert">{current.message}</p> : <p role="status">{description}</p>}
        </div>}
      </section>
    </div>
  </>;
}
