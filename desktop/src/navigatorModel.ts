import type {GraphNode, Group, Mapping, Module, NavigatorData, NodeDetail, Role} from "./navigatorSource";

export const under = (path: string, scope: string) => !scope || path === scope || path.startsWith(scope + ".");
export const parentPath = (path: string) => path.includes(".") ? path.slice(0, path.lastIndexOf(".")) : "";
export const tail = (path: string) => path.split(".").at(-1) || "root";
export const shortOp = (op: string) => op.replace(/^aten\./, "").replace(/\.(default|Tensor)$/, "");
export const shortGroup = (id: string) => id.slice(0, 6) + "…";
export const roleNode = (role: Role) => /@(n\d+)$/.exec(role.basis)?.[1];
export const axisTotal = (axes: {derived: number; declared: number; unknown: number; inferred?: number}) =>
  axes.derived + axes.declared + axes.unknown + (axes.inferred ?? 0);

export function valueRefs(values: unknown[]): string[] {
  return values.flatMap(value => Array.isArray(value) ? valueRefs(value) :
    typeof value === "string" && /^v\d+$/.test(value) ? [value] : []);
}

export function bucketLit(mapping: Mapping, unit: number, begin: number, end: number) {
  const {offset, block, repeat, period} = mapping;
  const first = offset + unit * block;
  if (repeat <= 1 || !period) return first < end && first + block > begin;
  const k = Math.max(0, Math.ceil((begin - first - block + 1) / period));
  return k < repeat && first + k * period < end;
}

export function unitAt(mapping: Mapping, index: number, size: number) {
  const relative = index - mapping.offset;
  if (relative < 0 || !mapping.block) return;
  const period = mapping.repeat > 1 && mapping.period > 0 ? mapping.period : Infinity;
  if (Math.floor(relative / period) >= mapping.repeat) return;
  const unit = Math.floor((relative % period) / mapping.block);
  return unit < size ? unit : undefined;
}

export type Row = {id: string; parent: string | null; label: string; type?: string | null; path?: string; node?: GraphNode; tensor?: string;
  summary?: Module; flags?: string; terms: string[]};

export function outline(data: NavigatorData, details: Record<string, NodeDetail>) {
  const map = new Map<string, Row>();
  const nodes = new Map(data.nodes.map(node => [node.id, node]));
  function folder(path: string): string | null {
    if (!path) return null;
    const id = "module:" + path;
    if (!map.has(id)) map.set(id, {id, parent: folder(parentPath(path)), label: tail(path), path, terms: [path]});
    return id;
  }
  for (const row of data.modules) {
    if (!row.path) continue;
    const item = map.get(folder(row.path)!)!;
    item.summary = row; item.type = row.type;
    item.flags = (row.opaque_count ? "⊘ " : "") + (row.open_groups ? "◌" : "");
    item.terms.push(...(row.opaque_count ? ["opaque"] : []), ...(row.open_groups ? ["open"] : []));
  }
  const tensors = new Map(data.tensors.map(tensor => [tensor.name, tensor]));
  const spans = new Map<string, string[]>();
  const boundaries = new Map<string, string[]>();
  for (const group of data.groups) {
    const terms = ["group:" + group.group_id, group.closed ? "closed" : "open"];
    for (const span of group.spans) spans.set(span.value, [...(spans.get(span.value) ?? []), ...terms]);
    for (const boundary of group.boundary) if (boundary.node)
      boundaries.set(boundary.node, [...(boundaries.get(boundary.node) ?? []), ...terms]);
  }
  const tensorTerms = (name: string) => {
    const tensor = tensors.get(name)!;
    return [name, ...(tensor.binding ? [tensor.binding] : []),
      ...(tensor.roles?.flatMap(role => ["role:" + role.role, "prov:" + role.provenance]) ?? ["role:unknown", "prov:unknown"]),
      ...(tensor.groups?.map(g => "group:" + g.group_id) ?? [])];
  };
  for (const node of data.nodes) {
    const id = "node:" + node.id;
    const parent = folder(node.module_path ?? "");
    const terms = [node.op, node.id, "op:" + shortOp(node.op), ...(node.opaque_reason ? ["opaque"] : []),
      ...node.outputs.flatMap(o => spans.get(o.value) ?? []), ...(boundaries.get(node.id) ?? []),
      ...(details[node.id]?.parameters.filter(name => tensors.has(name)).flatMap(tensorTerms) ?? [])];
    map.set(id, {id, parent, label: shortOp(node.op), type: node.id, path: node.module_path ?? "", node, terms,
      flags: node.opaque_reason ? "⊘" : undefined});
  }
  for (const tensor of data.tensors) {
    const label = tensor.name;
    let parent: string | null;
    if (tensor.binding === "unbound" || tensor.binding === "unused") {
      const kind = tensor.binding;
      parent = "binding:" + kind;
      if (!map.has(parent)) map.set(parent, {id: parent, parent: null, label: kind === "unbound" ? "Not in graph (unbound)" : "Unused by forward", terms: [kind]});
    } else {
      const basis = tensor.roles?.map(roleNode).find(id => id && nodes.has(id));
      parent = folder(basis ? nodes.get(basis)?.module_path ?? "" : data.summary ? "" : parentPath(label));
    }
    const id = "tensor:" + label;
    map.set(id, {id, parent, label: parent?.startsWith("module:") ? tail(label) : label, type: tensor.dtype, tensor: label,
      flags: tensor.binding === "unbound" ? "∅" : undefined, terms: tensorTerms(label)});
  }
  return map;
}

export function outlineMatches(rows: Map<string, Row>, search: string) {
  const tokens = search.toLowerCase().replace(/…/g, "").trim().split(/\s+/).filter(Boolean);
  const matching = new Set<string>(), shown = new Set<string>();
  for (const row of rows.values()) {
    if (!tokens.every(token => row.terms.some(term => {
      const text = term.toLowerCase();
      return token.includes(":") ? text.startsWith(token) : ["opaque", "open", "closed", "unused", "unbound", "bound"].includes(token) ?
        text === token : text.includes(token);
    }))) continue;
    matching.add(row.id);
    for (let current: Row | undefined = row; current && !shown.has(current.id);
      current = current.parent ? rows.get(current.parent) : undefined) shown.add(current.id);
  }
  return {matching, shown};
}

export type Box = {id: string; x: number; y: number; w: number; h: number; scope?: string; ids: string[]; node?: GraphNode; expanded?: boolean};
export function layoutGraph(nodes: GraphNode[], details: Record<string, NodeDetail>, scope: string, expanded: Set<string>) {
  const producers = new Map(nodes.flatMap(n => n.outputs.map(output => [output.value, n.id] as const)));
  const edges = nodes.flatMap(node => [...new Set(valueRefs([details[node.id]?.args, Object.values(details[node.id]?.kwargs ?? {})]))]
    .flatMap(value => {
      const source = producers.get(value);
      return source ? [{source, target: node.id, value}] : [];
    }));
  const inbound = new Map<string, typeof edges>();
  for (const edge of edges) {
    const previous = inbound.get(edge.target);
    if (previous) previous.push(edge); else inbound.set(edge.target, [edge]);
  }
  const owners = new Map<string, Box>();
  function place(path: string, subset: GraphNode[]): {boxes: Box[]; width: number; height: number} {
    const tiles = new Map<string, {scope?: string; nodes: GraphNode[]}>();
    const keys = new Map<string, string>();
    for (const node of subset) {
      const rest = (node.module_path ?? "").slice(path ? path.length + 1 : 0);
      const child = node.module_path !== path && rest ? [path, rest.split(".")[0]].filter(Boolean).join(".") : undefined;
      const key = child ? "module:" + child : "node:" + node.id;
      keys.set(node.id, key);
      const tile = tiles.get(key);
      if (tile) tile.nodes.push(node); else tiles.set(key, {scope: child, nodes: [node]});
    }
    const ranks = new Map<string, number>(), columns = new Map<number, {box: Box; children: Box[]}[]>();
    for (const [id, tile] of tiles) {
      let rank = 0;
      for (const node of tile.nodes) for (const edge of inbound.get(node.id) ?? []) {
        const key = keys.get(edge.source);
        if (key && key !== id && ranks.has(key)) rank = Math.max(rank, ranks.get(key)! + 1);
      }
      ranks.set(id, rank);
      const open = tile.scope !== undefined && expanded.has(tile.scope);
      const child = open ? place(tile.scope!, tile.nodes) : undefined;
      const box: Box = {id, x: 0, y: 0, w: child ? Math.max(174, child.width + 28) : 174, h: child ? child.height + 65 : 102,
        scope: tile.scope, ids: tile.nodes.map(n => n.id), node: tile.scope === undefined ? tile.nodes[0] : undefined, expanded: open};
      const column = columns.get(rank) ?? [];
      column.push({box, children: child?.boxes ?? []}); columns.set(rank, column);
      if (!open) for (const node of tile.nodes) owners.set(node.id, box);
    }
    const boxes: Box[] = [];
    let x = 0, height = 0;
    for (const column of [...columns.entries()].sort(([a], [b]) => a - b).map(([, boxes]) => boxes)) {
      let y = 0, wide = 0;
      for (const {box, children} of column) {
        box.x = x; box.y = y; boxes.push(box);
        for (const child of children) {child.x += x + 14; child.y += y + 48; boxes.push(child);}
        y += box.h + 28; wide = Math.max(wide, box.w);
      }
      x += wide + 92; height = Math.max(height, y - 28);
    }
    return {boxes, width: Math.max(0, x - 92), height};
  }
  const result = place(scope, nodes.filter(node => under(node.module_path ?? "", scope)));
  for (const box of result.boxes) {box.x += 24; box.y += 32;}
  const used = new Set<string>();
  return {...result, width: result.width + 112, height: result.height + 64, owners,
    edges: edges.flatMap(edge => {
      const from = owners.get(edge.source), to = owners.get(edge.target);
      const key = [from?.id, to?.id, edge.value].join("/");
      if (!from || !to || from === to || used.has(key)) return [];
      used.add(key);
      return [{...edge, from, to}];
    })};
}

export function groupNodes(group: Group | undefined, nodes: GraphNode[], details: Record<string, NodeDetail>) {
  if (!group) return new Set<string>();
  const values = new Set(group.spans.map(span => span.value));
  const tensors = new Set(group.members.map(member => member.tensor));
  const boundary = new Set(group.boundary.map(item => item.node));
  return new Set(nodes.filter(node => boundary.has(node.id) || node.outputs.some(out => values.has(out.value)) ||
    details[node.id]?.parameters.some(tensor => tensors.has(tensor))).map(node => node.id));
}
