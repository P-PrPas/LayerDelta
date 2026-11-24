"""The `structure` profile (v0.5-E6): what a session WITHOUT weights:read may see of an edit.

An edit's own bookkeeping is a weight probe: `changed_elems` after "scale by 1.0" says whether a region is all zero, a `noop` refusal says
the same, shas of a tensor before and after pin its content, and `changed_units` of a diff says which units an edit moved. A benchmark arm
that must not read weights therefore gets responses rebuilt from an ALLOW-LIST, not filtered by a deny-list: a field added to a response
later is invisible to this profile until it is listed here and the route walk (tests/test_structure_profile.py) says so.

Residual, documented in docs/reference/egress.md: one bit per apply (a child whose bytes equal the parent's has the parent's id), and one
bit per plan/validate/apply from a value-dependent refusal (`dtype_overflow`, `non_finite`); both are bounded by the session budgets.
"""

OP = {"index": True, "op": True, "tensor": True, "selection_id": True, "selected_elems": True, "via": True}
PLAN = {"ops": [OP], "tensors": [{"tensor": True, "dtype": True, "ops": True}], "overlaps": True, "overlaps_total": True,
        "child_bytes": True, "numeric": True, "rounding": True, "groups": True, "moved_units": True, "warnings": True}
MANIFEST = {"schema_version": True, "parent": True, "child_sha256": True, "recipe_hash": True, "child_bytes": True, "engine": True}
APPLY = {**PLAN, "manifest": MANIFEST, "checkpoint": True, "recipe": True, "replayed": True, "existing": True, "diff_hint": True}
VALIDATE = {"valid": True, "checks": True, "error": True, "recipe": True}
DIFF_ROW = {"name": True, "status": True, "dtype": True, "shape": True, "a": True, "b": True}
DIFF_SUMMARY = {"tensors_total": True, "changed": True, "same": True, "only_a": True, "only_b": True, "shape_mismatch": True,
                "dtype_mismatch": True, "only_a_count": True, "only_b_count": True, "shape_mismatch_count": True,
                "dtype_mismatch_count": True, "skipped_small": True, "skipped_small_count": True}
DIFF = {"summary": DIFF_SUMMARY, "offset": True, "tensors": [DIFF_ROW]}
RESULTS = {"plan": PLAN, "apply": APPLY, "validate": VALIDATE, "diff": DIFF}


def keep(value, spec):
    """`value` rebuilt from `spec`: True keeps it whole, a dict keeps the listed keys, a one-item list applies its spec to every item."""
    if spec is True or value is None:
        return value
    if isinstance(spec, list):
        return [keep(item, spec[0]) for item in value]
    return {key: keep(value[key], sub) for key, sub in spec.items() if key in value}


def view(kind: str, envelope: dict) -> dict:
    """An edit or diff envelope as a session without weights:read sees it (the envelope's own fields are structure already)."""
    return {**envelope, "result": keep(envelope["result"], RESULTS[kind])}
