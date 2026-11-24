import {useEffect, useState} from "react";
import type {GraphTensor, Group, Member, NodeDetail, Role} from "./navigatorSource";
import {groupProfile} from "./navigatorSource";
import {bucketLit, roleNode, shortGroup, shortOp, unitAt} from "./navigatorModel";
import {count, compact} from "./tensorTree";
import {inspectErrorText} from "./tensorSource";

export type Tab = "Node" | "Group" | "Tensor";
const tabs: Tab[] = ["Node", "Group", "Tensor"];
export type OpenTensor = (checkpoint: string, tensor?: string, importing?: boolean, slice?: string) => void;

function roleLabel(role: Role) {
  if (role.provenance === "unknown") return "unknown";
  return role.factors?.length ? "packed " + role.factors.map(f => f.name + " " + f.size).join(" × ") : role.role;
}

function boundaryText(reason: string) {
  if (reason === "graph_output") return "Reaches a model output. Permuting would reorder the outputs.";
  if (reason === "opaque_op") return "Passes through an opaque operation.";
  return reason.replaceAll("_", " ");
}

function MemberStrip({member, dimension, group, unit, setUnit}: {member: Member; dimension: number; group: Group; unit: number; setUnit: (n: number) => void}) {
  const pixels = Math.min(256, dimension);
  const text = "Axis " + member.axis + ", " + count(dimension) + " positions, unit " + unit + ", offset " + member.offset +
    ", block " + member.block + ", repeat " + member.repeat + (member.repeat > 1 ? ", period " + member.period : "");
  return <div className="an-index-strip" role="img" aria-label={text} title={text} onPointerMove={e => {
    const bounds = e.currentTarget.getBoundingClientRect();
    const index = Math.min(dimension - 1, Math.floor((e.clientX - bounds.left) / bounds.width * dimension));
    const next = unitAt(member, index, group.size);
    if (next !== undefined) setUnit(next);
  }}>
    {Array.from({length: pixels}, (_, i) => <i aria-hidden="true" key={i}
      data-lit={bucketLit(member, unit, Math.ceil(i * dimension / pixels), Math.ceil((i + 1) * dimension / pixels))}/>)}
  </div>;
}

export function NavigatorInspector({checkpoint, tab, setTab, node, module, tensor, group, groups, tensors, unit, setUnit, onSelect, onNode, openTensor, graph, nodeError}: {
  checkpoint: string; tab: Tab; setTab: (value: Tab) => void; node?: NodeDetail; module?: string; tensor?: GraphTensor; group?: Group;
  groups: Group[]; tensors: Map<string, GraphTensor>; unit: number; setUnit: (unit: number) => void;
  onSelect: (id: string) => void; onNode: (id: string) => void; openTensor: OpenTensor; graph: boolean; nodeError?: string;
}) {
  const profileMember = group?.members.find(m => m.role === "out_unit") ?? group?.members[0];
  const profileKey = group ? checkpoint + "/" + group.group_id : "";
  const [profile, setProfile] = useState<{key: string; values?: Map<number, number | null>; error?: string}>();
  const [copyStatus, setCopyStatus] = useState("");
  useEffect(() => {
    if (!group || !profileMember || !group.size || tab !== "Group") return;
    const controller = new AbortController();
    groupProfile(checkpoint, group, profileMember, controller.signal)
      .then(values => {if (!controller.signal.aborted) setProfile({key: profileKey, values});})
      .catch(error => {if (!controller.signal.aborted) setProfile({key: profileKey, error: inspectErrorText(error)});});
    return () => controller.abort();
  }, [checkpoint, group, profileKey, profileMember, tab]);
  const values = profile?.key === profileKey ? profile.values : undefined;
  const maximum = Math.max(1e-12, ...(values ? [...values.values()].map(n => n ?? 0) : []));
  const groupById = new Map(groups.map(g => [g.group_id, g]));
  const groupLinks = (ids: string[]) => ids.map(id => <button key={id} title={id} className="an-text-button" onClick={() => onSelect("group:" + id)}>
    {groupById.get(id)?.label ?? shortGroup(id)} <span className="an-meta">{shortGroup(id)}</span>
  </button>);
  function roles(t: GraphTensor) {
    const list = t.roles ?? (t.shape.length ? t.shape.map((_, axis) => ({axis, role: "unknown", provenance: "unknown", basis: "No graph attached"})) :
      [{axis: null, role: "unknown", provenance: "unknown", basis: "No graph attached"}]);
    return <div className="an-roles">{list.map(role => {
      const id = t.groups?.find(g => role.axis === g.axis)?.group_id;
      const basis = roleNode(role);
      const label = <>{role.axis === null ? "scalar" : "axis " + role.axis} · {roleLabel(role)} · <small>{role.provenance}</small></>;
      return <div key={role.axis ?? "scalar"} className="an-role-line">
        {id ? <button className="an-role" data-provenance={role.provenance} title={role.basis} onClick={() => onSelect("group:" + id)}>{label}</button> :
          <span className="an-role" data-provenance={role.provenance} title={role.basis}>{label}</span>}
        {basis ? <button className="an-text-button an-meta" onClick={() => onNode(basis)}>{role.basis}</button> :
          <span className="an-meta">{role.basis.replace(/^unbound$/, "Unbound · not in the graph")}</span>}
      </div>;
    })}</div>;
  }
  function openLink(name: string, member?: Member) {
    const slice = member && [member.axis, member.offset + unit * member.block, member.block, member.period, member.repeat].join(":");
    return <a className="an-open" href={"?" + new URLSearchParams({screen: "tensors", ckpt: checkpoint, tensor: name, ...(slice ? {slice} : {})})}
      onClick={e => {if (!e.metaKey && !e.ctrlKey && !e.shiftKey && !e.altKey) {e.preventDefault(); openTensor(checkpoint, name, false, slice);}}}>Open in Tensors ↗</a>;
  }
  return <aside className="an-inspector" aria-label="Architecture inspector">
    <div className="an-tabs ui-tabs" role="tablist" aria-label="Inspector view" onKeyDown={e => {
      const step = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
      if (step) {
        e.preventDefault();
        const next = (tabs.indexOf(tab) + tabs.length + step) % tabs.length;
        setTab(tabs[next]); e.currentTarget.querySelectorAll("button")[next]?.focus();
      }
    }}>{tabs.map(name => <button key={name} id={"an-tab-" + name} type="button" role="tab" aria-selected={tab === name}
      tabIndex={tab === name ? 0 : -1} aria-controls="an-inspector-panel" onClick={() => setTab(name)}>{name}</button>)}</div>
    <div id="an-inspector-panel" role="tabpanel" aria-labelledby={"an-tab-" + tab} tabIndex={0}>
      {tab === "Node" && (node ? <>
        <section className="an-section"><h2>{shortOp(node.op)}</h2>
          <p className="an-meta">{node.id} · {node.module_path || "root"}</p>
          <dl className="an-facts"><dt>Operation</dt><dd>{node.op}</dd><dt>Rule</dt><dd>{node.rule ?? "—"}</dd></dl>
          {node.opaque_reason && <p className="an-warning">⊘ Opaque · {node.opaque_reason.replaceAll("_", " ")}</p>}
          {node.outputs.map(out => <p key={out.value} className="an-meta">{out.value} · {out.kind}{out.shape && " · " + out.shape.join(" × ")}</p>)}
        </section>
        {node.parameters.map(name => <section className="an-section" key={name}>
          <button className="an-text-button" onClick={() => onSelect("tensor:" + name)}>{name}</button>
          {tensors.has(name) && roles(tensors.get(name)!)}
        </section>)}
        {!!node.groups.length && <section className="an-section"><h3>Coupling groups</h3>{groupLinks(node.groups)}</section>}
      </> : <section className="an-section">{nodeError ? <p role="alert">{nodeError}</p> :
        module !== undefined ? <><h2>{module || "root"}</h2><p>Open the module to inspect its operations.</p></> :
          <p>Select an operation from the outline or canvas.</p>}</section>)}
      {tab === "Tensor" && (tensor ? <>
        <section className="an-section"><h2>{tensor.name}</h2><dl className="an-facts">
          <dt>Dtype</dt><dd>{tensor.dtype}</dd><dt>Shape</dt><dd>{tensor.shape.join(" × ") || "scalar"}</dd>
          <dt>Size</dt><dd>{compact(tensor.nbytes, true)}</dd><dt>Binding</dt><dd>{tensor.binding ?? "No graph attached"}</dd>
          <dt>Storage</dt><dd>{tensor.shard ?? "Single file"}</dd>
        </dl>{openLink(tensor.name)}</section>
        <section className="an-section"><h3>Axis roles</h3>{roles(tensor)}</section>
        {!!tensor.groups?.length && <section className="an-section"><h3>Coupling groups</h3>{groupLinks([...new Set(tensor.groups.map(g => g.group_id))])}</section>}
      </> : <section className="an-section"><p>Select a tensor to see its binding and roles.</p></section>)}
      {tab === "Group" && (group ? <>
        <section className="an-section"><div className="an-title"><h2>{group.label}</h2><code title={group.group_id}>{shortGroup(group.group_id)}</code>
          <span className="ui-tag" data-tone={group.closed ? "pos" : "warn"}>{group.closed ? "Closed" : "Open"}</span></div>
          <button className="an-text-button" onClick={() => navigator.clipboard.writeText(group.group_id)
            .then(() => setCopyStatus("Group ID copied")).catch(() => setCopyStatus("Clipboard unavailable"))}>Copy group ID</button>
          <span className="visually-hidden" role="status">{copyStatus}</span>
          <p className="an-meta">{count(group.size)} units{group.constraints.kind === "within_blocks" && " · permute within blocks of " + group.constraints.n}</p>
          <div className="an-unit"><button aria-label="Previous unit" onClick={() => setUnit(unit - 1)} disabled={unit <= 0}>−</button>
            <label>Unit <input type="number" min={0} max={group.size - 1} value={unit}
              onChange={e => {if (e.target.value) setUnit(+e.target.value);}}/></label>
            <button aria-label="Next unit" onClick={() => setUnit(unit + 1)} disabled={unit >= group.size - 1}>+</button>
          </div>
          {profileMember && <div className="an-group-profile"><p className="an-meta">L2 of {profileMember.tensor} · axis {profileMember.axis}
            {group.size > 64 && " · first 64 units"}</p>
            {values ? <div className="an-profile-bars" aria-label="Group L2 profile">{[...values].sort(([a], [b]) => a - b).map(([i, v]) =>
              <button type="button" key={i} tabIndex={-1} aria-label={"Unit " + i + ", L2 " + (v ?? "unavailable")}
                title={"Unit " + i + " · L2 " + (v ?? "unavailable")} aria-pressed={unit === i} onClick={() => setUnit(i)}
                style={{height: Math.max(2, (v ?? 0) / maximum * 34)}}/>)}</div> :
              <p role="status" className="an-meta">{profile?.key === profileKey && profile.error ? profile.error : "Loading unit profile…"}</p>}
          </div>}
        </section>
        <section className="an-section"><h3>Members</h3>{group.members.map((member, i) => {
          const t = tensors.get(member.tensor), dimension = t?.shape[member.axis];
          const role = t?.roles?.find(r => r.axis === member.axis);
          return <div className="an-member" key={i}>
            <div className="an-member-head"><button className="an-text-button" onClick={() => onSelect("tensor:" + member.tensor)}>{member.tensor}</button>
              <span className="an-role" data-provenance={role?.provenance} title={role?.basis}>axis {member.axis} · {role ? roleLabel(role) : member.role}</span></div>
            {dimension !== undefined && <MemberStrip member={member} dimension={dimension} group={group} unit={unit} setUnit={setUnit}/>}
            <div className="an-member-footer"><span className="an-meta">@{member.offset} · block {member.block}{member.repeat > 1 && " · repeat " + member.repeat + " × period " + member.period}</span>
              {openLink(member.tensor, member)}</div>
          </div>;
        })}{group.spans_truncated && <p className="an-meta">Activation highlight is limited to the spans returned by the server.</p>}</section>
        {!!group.boundary.length && <section className="an-section"><h3>Boundaries</h3>{group.boundary.map((b, i) => <div className="an-note" key={i}>
          <p>{boundaryText(b.reason)} {b.output !== undefined ? "Output " + b.output + "." : ""}</p>
          <span className="an-meta">{[b.node, b.op].filter(Boolean).join(" · ")}</span>
          {b.node && <button className="an-text-button" onClick={() => onNode(b.node!)}>Show on canvas</button>}
        </div>)}</section>}
        {!!group.notes.length && <section className="an-section"><h3>Group notes</h3>{group.notes.map((note, i) => <p className="an-warning" key={i}>{note}</p>)}</section>}
        <section className="an-section"><button onClick={() => onSelect("")}>Browse groups</button></section>
      </> : <section className="an-section"><h2>Coupling groups</h2>{groups.length ? groupLinks(groups.map(g => g.group_id)) :
        <p>{graph ? "No coupling groups in this graph." : "Attach a graph to inspect coupling groups."}</p>}</section>)}
    </div>
  </aside>;
}
