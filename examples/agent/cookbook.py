"""Runnable recipes of the agent cookbook (docs/guides/agent-cookbook.md), v0.4-S7.

Each recipe is a deterministic script over the agent tools. It sees only what a model would see: tool results. It never reads the
workspace. The same recipes run over MCP (`mcp_caller`: the real `layerdelta-mcp` stdio bridge) or in-process through the bridge's
`execute` (`execute_caller`, no MCP SDK needed), so the guide's examples are tested exactly as written.

Run one against a service started with `layerdelta --agent-token-file connection.json --agent-weights-read --agent-weights-write
--agent-no-values`:

    python examples/agent/cookbook.py --connection connection.json --checkpoint <id> --recipe prune
"""

import argparse
import json
import sys
import uuid


class ToolError(Exception):
    """A tool answered with an error result: `code` is the backend's error code, `body` the whole structured result."""

    def __init__(self, tool, body):
        self.tool, self.body = tool, body
        self.code = body.get("error", {}).get("code")
        super().__init__(f"{tool}: {self.code}")


def caller(raw):
    """`call(tool, **args) -> result body` from `raw(tool, args) -> (structured result, is_error)`; errors raise ToolError."""
    def call(tool, **args):
        body, is_error = raw(tool, args)
        if is_error:
            raise ToolError(tool, body)
        return body
    return call


def execute_caller(client):
    """In-process: the bridge's own `execute` (same errors, size guard and parity as MCP)."""
    from layerdelta import agent_bridge

    def raw(tool, args):
        body, is_error, _ = agent_bridge.execute(client, tool, args)
        return body, is_error
    return caller(raw)


class mcp_caller:
    """The real thing: start `layerdelta-mcp` over stdio and call tools through the MCP SDK (`pip install "layerdelta[mcp]"`).
    Use as a context manager; `.call` is the recipe interface, `.meta` the last result's `_meta` (budget usage)."""

    def __init__(self, connection_path, *, extra_args=(), env=None):
        import os
        self.args = ["-m", "layerdelta.agent_bridge", "--connection", str(connection_path), *extra_args]
        self.env, self.meta = (dict(os.environ) if env is None else env), None  # the SDK's default env drops PYTHONPATH

    def __enter__(self):
        from anyio.from_thread import start_blocking_portal
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._portal_cm = start_blocking_portal()
        self.portal = self._portal_cm.__enter__()
        params = StdioServerParameters(command=sys.executable, args=self.args, env=self.env)
        self._stdio = self.portal.wrap_async_context_manager(stdio_client(params))
        read, write = self._stdio.__enter__()
        self._session_cm = self.portal.wrap_async_context_manager(ClientSession(read, write))
        self.session = self._session_cm.__enter__()
        self.portal.call(self.session.initialize)

        def raw(tool, args):
            result = self.portal.call(self.session.call_tool, tool, args)
            self.meta = result.meta
            return result.structuredContent, bool(result.isError)
        self.call = caller(raw)
        return self

    def tools(self):
        return [t.name for t in self.portal.call(self.session.list_tools).tools]

    def __exit__(self, *exc):
        for cm in (self._session_cm, self._stdio, self._portal_cm):
            cm.__exit__(*exc)


# ---------------------------------------------------------------- recipe 1: orient

def orient(call, checkpoint_id):
    """What can I do, and what is there? Capabilities, budgets, the tensor table and the closed groups."""
    session = call("session_info")
    tensors = call("describe", checkpoint_id=checkpoint_id, limit=200)["result"]["tensors"]
    try:
        groups = call("groups", checkpoint_id=checkpoint_id, closed=True, limit=50)["result"]["groups"]
    except ToolError as exc:
        if exc.code != "no_graph":  # without a graph there are no groups: tensor selectors still work
            raise
        groups = []
    return {"capabilities": session["capabilities"], "egress_class": session.get("egress_class"),
            "budgets": session.get("budgets"), "tensors": len(tensors), "closed_groups": len(groups), "groups": groups}


def _largest_closed(groups, minimum=2):
    usable = [g for g in groups if g["closed"] and g["size"] >= minimum]
    if not usable:
        raise LookupError("no closed group with enough units")
    return max(usable, key=lambda g: (g["size"], len(g["members"]), g["group_id"]))


# ---------------------------------------------------------------- recipe 2: find weak units and zero them through a group

def prune_weak_units(call, checkpoint_id, *, key, count=2, groups=None):
    """Rank a closed group's units by L2 (query), zero the weakest `count` in every member at once (plan, apply), check the
    child against its parent (diff from the apply's diff_hint)."""
    groups = groups if groups is not None else orient(call, checkpoint_id)["groups"]
    candidates = sorted((g for g in groups if g["closed"] and g["size"] > count), key=lambda g: (-g["size"], g["group_id"]))
    for group in candidates:
        ranked = call("query", checkpoint_id=checkpoint_id, limit=1,
                      where={"group": group["group_id"], "stat": "l2", "op": "lt", "value": 1e30})["result"]
        if ranked["matches"]:
            break
        # skipped_small: in egress class 1 the ranking member's units are too small to rank (a bias); try the next group
    else:
        raise LookupError("no closed group could be ranked")
    weakest = sorted(u["unit"] for u in ranked["matches"][0]["units"][:count])
    select = {"group": group["group_id"], "units": weakest}
    planned = call("plan", checkpoint_id=checkpoint_id, program=[{"op": "zero", "args": {}, "select": select}],
                   hypothesis=f"Units {weakest} have the lowest L2 in group {group['group_id']}; zeroing them in every member "
                              "should change the output least.")
    recipe = planned["result"]["recipe"]
    applied = call("apply", checkpoint_id=checkpoint_id, key=key, recipe=recipe)
    hint = applied["result"]["diff_hint"]
    diff = call("diff", checkpoint_id=hint["checkpoint_id"], other=hint["other"], limit=64)
    changed = sorted(t["name"] for t in diff["result"]["tensors"] if t.get("status") == "changed")
    return {"recipe": "prune", "child": hint["other"], "parent": checkpoint_id, "group": group["group_id"], "units": weakest,
            "recipe_hash": recipe["recipe_hash"], "changed_tensors": changed,
            "members": sorted({m["tensor"] for m in group["members"]})}


# ---------------------------------------------------------------- recipe 3: a function-preserving permutation

def permute_closed_group(call, checkpoint_id, *, key, groups=None):
    """Reverse the units of the largest closed group with permute_units. The function is unchanged (the caller can check it
    with a forward pass outside LayerDelta); every member tensor is reordered consistently."""
    groups = groups if groups is not None else orient(call, checkpoint_id)["groups"]
    group = _largest_closed(groups)
    perm = list(range(group["size"]))[::-1]
    planned = call("plan", checkpoint_id=checkpoint_id, hypothesis="Reordering a closed group's units preserves the function.",
                   program=[{"op": "permute_units", "args": {"perm": perm}, "select": {"group": group["group_id"]}}])
    applied = call("apply", checkpoint_id=checkpoint_id, key=key, recipe=planned["result"]["recipe"])
    return {"recipe": "permute", "child": applied["result"]["diff_hint"]["other"], "parent": checkpoint_id,
            "group": group["group_id"], "perm": perm, "recipe_hash": planned["result"]["recipe"]["recipe_hash"],
            "moved_units": planned["result"]["moved_units"][0]["count"]}


# ---------------------------------------------------------------- recipe 4: two children of one parent, and lineage

def compare_children(call, parent):
    """List a parent's children (lineage) and diff the first two against each other."""
    children = call("get_checkpoint", checkpoint_id=parent)["children"]
    if len(children) < 2:
        return {"parent": parent, "children": children, "diff": None}
    first, second = children[:2]
    for child in (first, second):
        if call("get_checkpoint", checkpoint_id=child)["parent_id"] != parent:
            raise RuntimeError(f"{child} does not name {parent} as its parent")
    summary = call("diff", checkpoint_id=first, other=second, limit=64)["result"]["summary"]
    return {"parent": parent, "children": children, "diff": {"a": first, "b": second, "changed": summary["changed"]}}


# ---------------------------------------------------------------- recipe 5: recovering from errors

def recover(call, checkpoint_id, groups):
    """The errors an agent meets most, and the right reaction to each (see the error catalog's hints)."""
    seen = {}
    open_group = next((g for g in groups if not g["closed"]), None)
    if open_group is not None:  # permute_units on an open group is refused: pick a closed one instead
        try:
            call("plan", checkpoint_id=checkpoint_id, hypothesis="x",
                 program=[{"op": "permute_units", "args": {"perm": list(range(open_group["size"]))[::-1]},
                           "select": {"group": open_group["group_id"]}}])
        except ToolError as exc:
            seen["open_group"] = exc.code
    try:  # a limit over the cap: lower the argument the error names
        call("describe", checkpoint_id=checkpoint_id, limit=10 ** 9)
    except ToolError as exc:
        seen["cap"] = (exc.code, exc.body["error"]["field"])
    return seen


# ---------------------------------------------------------------- recipe 6: the hand-off report

def handoff(call, results):
    """What to give a person: each child that exists, its recipe hash, and the trace events that made it."""
    events, after = [], 0
    while True:  # page the whole timeline: a long session has more than one page
        page = call("trace", limit=500, after=after)["result"]
        events += page["events"]
        if page["next_after"] is None:
            break
        after = page["next_after"]
    applies = {e["child_id"]: e["seq"] for e in events if e["tool"] == "apply" and e.get("child_id") and e["status"] < 400}
    report = []
    for r in results:
        exists = True
        try:
            call("get_checkpoint", checkpoint_id=r["child"])
        except ToolError:
            exists = False
        if exists:
            report.append({"child": r["child"], "parent": r["parent"], "recipe": r["recipe"], "recipe_hash": r["recipe_hash"],
                           "trace_seq": applies.get(r["child"])})
    return report


# ---------------------------------------------------------------- the scripted multi-family agent (acceptance T7)

def scripted_agent(call, checkpoints, *, user_budget, tag=None):
    """Prune then permute on each checkpoint until the PERSON's apply budget is spent, the server refuses (budget or quota), or the
    work is done. Stops on refusals instead of looking for another way. The report lists only children that exist."""
    tag = tag or uuid.uuid4().hex[:8]
    results, skipped, used, stop = [], [], 0, "done"
    for cid in checkpoints:
        for name, recipe in (("prune", prune_weak_units), ("permute", permute_closed_group)):
            if used >= user_budget:
                stop = "user_budget"
                break
            try:
                results.append(recipe(call, cid, key=f"{tag}-{name}-{cid[:12]}"))
                used += 1
            except LookupError as exc:  # no closed group this recipe can use (or rank, in class 1): note it and move on
                skipped.append({"checkpoint": cid, "recipe": name, "reason": str(exc)})
                continue
            except ToolError as exc:
                if exc.code in ("budget_exhausted", "quota_exhausted", "forbidden", "session_revoked"):
                    stop = exc.code
                    break
                raise
        if stop != "done":
            break
    return {"stop_reason": stop, "applies": used, "skipped": skipped, "children": handoff(call, results)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one cookbook recipe through the real MCP bridge")
    parser.add_argument("--connection", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--recipe", choices=("orient", "prune", "permute", "children"), default="orient")
    parser.add_argument("--key", default=None, help="idempotency key for prune/permute (reuse it to retry)")
    args = parser.parse_args(argv)
    with mcp_caller(args.connection) as mcp:
        call = mcp.call
        if args.recipe == "orient":
            out = orient(call, args.checkpoint)
            out.pop("groups")
        elif args.recipe == "prune":
            out = prune_weak_units(call, args.checkpoint, key=args.key or "cookbook-prune")
        elif args.recipe == "permute":
            out = permute_closed_group(call, args.checkpoint, key=args.key or "cookbook-permute")
        else:
            out = compare_children(call, args.checkpoint)
    json.dump(out, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
