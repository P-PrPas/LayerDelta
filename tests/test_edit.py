"""Edit programs: recipe v2, plan/validate/apply, deterministic child writer, v1 translation."""

import hashlib
import json
import os
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from layerdelta import edit
from layerdelta.checkpoint import load_checkpoint
from layerdelta.edit import Limits, apply, freeze_recipe, plan, translate_v1, validate, verify_recipe
from layerdelta.engine import patch_weight
from layerdelta.model import ReferenceCNN, target_catalog
from layerdelta.selector import InspectError, canonical_json
from layerdelta.tensors import open_tensors

BASE_ID = "ab" * 32
EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "recipes" / "zero-block2-channel17.json"


def write(tmp_path, tensors, name="parent.safetensors"):
    path = tmp_path / name
    save_file(tensors, path)
    return path


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def err(call, **fields):
    with pytest.raises(InspectError) as info:
        call()
    for key, value in fields.items():
        assert getattr(info.value, key) == value, info.value.to_dict()
    return info.value


def recipe_for(table, program, base_id=BASE_ID, hypothesis="h", **kw):
    return freeze_recipe(table, base_id, program, hypothesis, **kw)


def step(op, select, **args):
    return {"op": op, "args": args, "select": select}


def bits_of(path, name):
    table = open_tensors(path)
    info = table.tensors[name]
    return {"F32": np.uint32, "F16": np.uint16, "BF16": np.uint16}[info.dtype], table.read_bytes(name)


def changed_mask(parent, child, name):
    dtype, a = bits_of(parent, name)
    _, b = bits_of(child, name)
    shape = open_tensors(parent).tensors[name].shape
    return (np.frombuffer(a, dtype) != np.frombuffer(b, dtype)).reshape(shape)


# ------------------------------------------------------------------ v1 -> v2 equivalence

def v1_recipe(target_id, channel, operation, scale=None):
    base = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    base.pop("recipe_hash")
    mapping = asdict(target_catalog(ReferenceCNN())[target_id])
    base["mapping"] = {k: mapping[k] for k in ("target_id", "mode", "module_path", "channel_count", "channel_axis", "parameter_name")}
    base["intervention"] = {"target_id": target_id, "mode": "weight", "channel_index": channel, "operation": operation,
                            **({"scale": scale} if scale is not None else {})}
    return {**base, "recipe_hash": edit.digest(base)}


def v01_export(tmp_path, parent, recipe, name):
    """What v0.1's checkpoint_export job does: load the model, patch the kernel, save the state dict."""
    from safetensors.torch import save_file as save
    model = load_checkpoint(parent)
    patch_weight(model, recipe["intervention"])
    out = tmp_path / name
    save(model.state_dict(), str(out))
    return out


def test_example_recipe_v1_hash_verifies_and_translates(tmp_path):
    recipe = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    table = open_tensors(write(tmp_path, ReferenceCNN().state_dict()))
    v2 = translate_v1(recipe, table)
    assert v2 == translate_v1(recipe, table)  # one v1 recipe, exactly one v2 hash
    assert v2["schema_version"] == 2 and v2["base"]["checkpoint_sha256"] == recipe["dependencies"]["checkpoint_sha256"]
    assert v2["program"][0]["op"] == "zero"
    select = v2["program"][0]["select"]  # the resolved, canonical form (shape guard, every axis)
    assert select["tensor"] == "features.block2.conv.weight" and select["shape"] == [24, 8, 3, 3]
    assert select["index"]["0"] == [{"start": 17, "stop": 18}] and select["index"]["1"] == [{"start": 0, "stop": 8}]
    assert verify_recipe(v2) == v2
    tampered = {**recipe, "hypothesis": "changed"}
    err(lambda: translate_v1(tampered, table), code="recipe_hash_mismatch")
    activation = v1_recipe("features.block1.conv.kernel", 0, "zero")
    activation["intervention"]["mode"] = "activation"
    activation["recipe_hash"] = edit.digest({k: v for k, v in activation.items() if k != "recipe_hash"})
    err(lambda: translate_v1(activation, table), code="unsupported_mode")
    err(lambda: translate_v1({"schema_version": 2}, table), code="unsupported_schema")
    with pytest.raises(TypeError):
        translate_v1(recipe)  # the table is required: it fixes the canonical selector


@pytest.mark.parametrize("target,channel,operation,scale", [
    ("features.block2.conv.kernel", 17, "zero", None),   # the shipped example
    ("features.block1.conv.kernel", 0, "zero", None),
    ("features.block2.conv.kernel", 23, "scale", 0.5),
    ("features.block1.conv.kernel", 7, "scale", 1.7),
])
def test_v1_translation_child_equals_v01_export(tmp_path, target, channel, operation, scale):
    torch.manual_seed(7)
    parent = write(tmp_path, ReferenceCNN().state_dict())
    v1 = json.loads(EXAMPLE.read_text(encoding="utf-8")) if (target, channel, operation) == ("features.block2.conv.kernel", 17, "zero") \
        else v1_recipe(target, channel, operation, scale)
    v2 = translate_v1(v1, open_tensors(parent))
    child = tmp_path / "child.safetensors"
    envelope = apply(open_tensors(parent), v2, child, base_id=v1["dependencies"]["checkpoint_sha256"])
    legacy = v01_export(tmp_path, parent, v1, "legacy.safetensors")
    assert child.read_bytes() == legacy.read_bytes()
    assert envelope["result"]["manifest"]["child_sha256"] == file_sha(legacy) == file_sha(child)
    assert envelope["side_effects"] == [{"kind": "checkpoint_created", "id": file_sha(child), "bytes": child.stat().st_size}]


# ------------------------------------------------------------------ every op: touched mask == selection

SELECT = {"tensor": "w", "index": {"0": [1, 3], "2": {"start": 1, "stop": 3}}}


def selection_mask(shape=(5, 4, 3)):
    mask = np.zeros(shape, bool)
    mask[np.ix_([1, 3], range(4), [1, 2])] = True
    return mask


def lerp_blob():
    return np.linspace(-1, 1, 16, dtype="<f4").tobytes()  # 2 * 4 * 2 selected elements


OP_CASES = {
    "zero": ({}, lambda x, b: np.zeros_like(x)),
    "scale": ({"k": 2.0}, lambda x, b: x * np.float32(2)),
    "set": ({"value": 7.5}, lambda x, b: np.full_like(x, 7.5)),
    "add": ({"value": 1.25}, lambda x, b: x + np.float32(1.25)),
    "clamp": ({"max": 0.0}, lambda x, b: np.minimum(x, np.float32(0))),
    "lerp": ({"t": 0.25}, None),
    "copy_from": ({}, None),
    "set_blob": ({"blob": "sha256:" + hashlib.sha256(lerp_blob()).hexdigest()}, None),
    "add_blob": ({"blob": "sha256:" + hashlib.sha256(lerp_blob()).hexdigest()}, None),
}


@pytest.mark.parametrize("case", sorted(OP_CASES))
def test_touched_mask_equals_selection_for_every_op(tmp_path, case):
    torch.manual_seed(3)
    w = torch.randn(5, 4, 3) + 0.5
    donor = torch.randn(5, 4, 3) * 4
    other = torch.randn(7)
    parent = write(tmp_path, {"w": w, "other": other})
    table = open_tensors(parent)
    source = open_tensors(write(tmp_path, {"w": donor, "other": other}, "donor.safetensors"))
    args, _ = OP_CASES[case]
    op = {"set_blob": "set", "add_blob": "add"}.get(case, case)
    args = dict(args)
    if op in ("lerp", "copy_from"):
        args["source"] = "cd" * 32
    recipe = recipe_for(table, [step(op, SELECT, **args)])
    child = tmp_path / "child.safetensors"
    report = apply(table, recipe, child, base_id=BASE_ID, sources={"cd" * 32: source}, blobs={args.get("blob"): lerp_blob()} if "blob" in args else None)
    mask = changed_mask(parent, child, "w")
    sel = selection_mask()
    x, d = w.numpy(), donor.numpy()
    sel_idx = np.ix_([1, 3], range(4), [1, 2])
    old = x[sel_idx]
    blob = np.frombuffer(lerp_blob(), "<f4").reshape(2, 4, 2)
    new = {"zero": np.zeros_like(old), "scale": old * np.float32(2), "set": np.full_like(old, 7.5), "add": old + np.float32(1.25),
           "clamp": np.minimum(old, np.float32(0)), "lerp": old + (d[sel_idx] - old) * np.float32(0.25),
           "copy_from": d[sel_idx], "set_blob": blob, "add_blob": old + blob}[case]
    expected = x.copy()
    expected[sel_idx] = new
    result = np.frombuffer(open_tensors(child).read_bytes("w"), "<f4").reshape(5, 4, 3)
    assert np.array_equal(result.view(np.uint32), expected.view(np.uint32))
    assert not (mask & ~sel).any()  # nothing outside the selection ever changes
    if case in ("clamp",):
        assert (mask == (sel & (x > 0))).all()  # only the elements the clamp really moved
    else:
        assert (mask == sel).all()
    assert open_tensors(child).read_bytes("other") == table.read_bytes("other")  # untouched tensor, byte for byte
    touched = report["result"]["manifest"]["touched"]
    assert [(t["tensor"], t["elems"]) for t in touched] == [("w", 16)]
    assert touched[0]["changed_elems"] == int(mask.sum())
    assert touched[0]["before_sha"] == hashlib.sha256(table.read_bytes("w")).hexdigest()
    assert touched[0]["after_sha"] == hashlib.sha256(open_tensors(child).read_bytes("w")).hexdigest()
    assert report["result"]["touched_elems_total"] == 16


def test_child_keeps_the_parent_header_and_layout(tmp_path):
    parent = write(tmp_path, {"a": torch.randn(3, 3), "b": torch.randn(4).half(), "c": torch.arange(5)})
    table = open_tensors(parent)
    child = tmp_path / "c.safetensors"
    apply(table, recipe_for(table, [step("scale", {"tensor": "b"}, k=0.5)]), child, base_id=BASE_ID)
    assert open_tensors(child).header_bytes() == table.header_bytes()
    assert child.stat().st_size == parent.stat().st_size
    for name in ("a", "c"):
        assert open_tensors(child).read_bytes(name) == table.read_bytes(name)


# ------------------------------------------------------------------ no-op, refusal, replay

def test_scale_by_one_is_a_detected_noop(tmp_path):
    parent = write(tmp_path, {"w": torch.randn(4, 4)})
    table = open_tensors(parent)
    recipe = recipe_for(table, [step("scale", {"tensor": "w", "index": {"0": [1]}}, k=1.0)])
    assert plan(table, recipe, base_id=BASE_ID)["result"]["noop"] is True
    out = tmp_path / "c.safetensors"
    err(lambda: apply(table, recipe, out, base_id=BASE_ID), code="no_op")
    assert not out.exists()
    envelope = apply(table, recipe, out, base_id=BASE_ID, allow_noop=True)
    assert file_sha(out) == file_sha(parent) == envelope["result"]["manifest"]["child_sha256"]
    assert envelope["result"]["manifest"]["touched"][0]["changed_elems"] == 0


@pytest.mark.parametrize("dtype,tensor,op,args,code", [
    (torch.float16, torch.tensor([60000.0, 1.0]), "scale", {"k": 2.0}, "dtype_overflow"),
    (torch.bfloat16, torch.tensor([3.0e38, 1.0]), "scale", {"k": 1.5}, "non_finite"),       # overflows the float32 compute
    (torch.float16, torch.tensor([1.0, 2.0]), "set", {"value": 70000.0}, "dtype_overflow"),
    (torch.float32, torch.tensor([float("nan"), 1.0]), "scale", {"k": 2.0}, "non_finite"),
    (torch.float32, torch.tensor([float("inf"), 1.0]), "add", {"value": 1.0}, "non_finite"),
    (torch.float32, torch.tensor([3.0e38, 1.0]), "add", {"value": 3.0e38}, "non_finite"),
])
def test_non_finite_and_overflow_are_refused_before_any_write(tmp_path, dtype, tensor, op, args, code):
    parent = write(tmp_path, {"w": tensor.to(dtype), "z": torch.ones(3)})
    table = open_tensors(parent)
    out = tmp_path / "child.safetensors"
    recipe = recipe_for(table, [step("zero", {"tensor": "z"}), step(op, {"tensor": "w"}, **args)])
    error = err(lambda: apply(table, recipe, out, base_id=BASE_ID), code=code)
    assert error.field in ("program", "program[1]")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["parent.safetensors"]  # no child, no partial file
    assert validate(table, recipe, base_id=BASE_ID)["valid"] is False
    assert validate(table, recipe, base_id=BASE_ID)["error"]["code"] == code


def test_non_finite_inputs_are_fine_when_the_op_overwrites_them(tmp_path):
    parent = write(tmp_path, {"w": torch.tensor([float("nan"), 1.0, 2.0])})
    table = open_tensors(parent)
    recipe = recipe_for(table, [step("set", {"tensor": "w", "index": {"0": [0]}}, value=3.0)])
    apply(table, recipe, tmp_path / "c.safetensors", base_id=BASE_ID)
    assert open_tensors(tmp_path / "c.safetensors").read_bytes("w")[:4] == np.float32(3).tobytes()


def test_untouched_nan_payloads_survive_bit_exactly(tmp_path):
    bits = torch.tensor([0x7E01, 0x3C00, 0x4000, 0x7C01], dtype=torch.int16).view(torch.float16)  # NaN payloads, 1.0, 2.0
    parent = write(tmp_path, {"w": bits})
    table = open_tensors(parent)
    recipe = recipe_for(table, [step("scale", {"tensor": "w", "index": {"0": {"start": 1, "stop": 3}}}, k=2.0)])
    apply(table, recipe, tmp_path / "c.safetensors", base_id=BASE_ID)
    out = np.frombuffer(open_tensors(tmp_path / "c.safetensors").read_bytes("w"), np.uint16)
    assert out.tolist() == [0x7E01, 0x4000, 0x4400, 0x7C01]


def test_replay_gives_the_same_child_sha(tmp_path):
    torch.manual_seed(1)
    parent = write(tmp_path, {"w": torch.randn(6, 5).bfloat16(), "b": torch.randn(6)})
    table = open_tensors(parent)
    recipe = recipe_for(table, [step("scale", {"tensor": "w", "index": {"0": [2, 4]}}, k=0.3),
                                step("add", {"tensor": "b"}, value=0.1)], hypothesis="replay")
    shas = set()
    for i, version in enumerate((recipe, json.loads(json.dumps(recipe)), verify_recipe(recipe))):
        out = tmp_path / f"c{i}.safetensors"
        shas.add(apply(table, version, out, base_id=BASE_ID)["result"]["manifest"]["child_sha256"])
        assert file_sha(out) in shas
    assert len(shas) == 1
    reopened = open_tensors(parent)  # a fresh table over the same bytes, as another workspace would have
    assert apply(reopened, recipe, tmp_path / "again.safetensors", base_id=BASE_ID)["result"]["manifest"]["child_sha256"] in shas


# ------------------------------------------------------------------ rounding goldens (values from torch, independent path)

BF16_IN = [0x3F80, 0x4040, 0x437F, 0x3F81, 0xC020, 0x38D2, 0x4700, 0x42F7]    # 1, 3, 255, 1.0078125, -2.5, 1e-4, 32768, 123.456
BF16_OUT = [0x3DCD, 0x3E9A, 0x41CC, 0x3DCE, 0xBE80, 0x3728, 0x454D, 0x4146]   # each * float32(0.1), nearest even
F16_IN = [0x3C00, 0x4200, 0x2E66, 0x63D1, 0xC100, 0x7B53]
F16_OUT = [0x34CD, 0x3B33, 0x27AE, 0x5CB1, 0xBA00, 0x7465]                    # each * float32(0.3)


@pytest.mark.parametrize("dtype,raw,k,golden", [(torch.bfloat16, BF16_IN, 0.1, BF16_OUT), (torch.float16, F16_IN, 0.3, F16_OUT)])
def test_scale_rounding_golden(tmp_path, dtype, raw, k, golden):
    values = torch.from_numpy(np.array(raw, np.uint16).view(np.int16)).view(dtype)
    parent = write(tmp_path, {"w": values})
    table = open_tensors(parent)
    out = tmp_path / "c.safetensors"
    apply(table, recipe_for(table, [step("scale", {"tensor": "w"}, k=k)]), out, base_id=BASE_ID)
    got = np.frombuffer(open_tensors(out).read_bytes("w"), np.uint16).tolist()
    assert got == golden
    assert got == (values.float() * np.float32(k)).to(dtype).view(torch.int16).numpy().astype(np.uint16).tolist()


def test_bf16_rounding_matches_torch_on_many_values(tmp_path):
    torch.manual_seed(11)
    values = (torch.randn(4096) * 10 ** torch.randint(-6, 6, (4096,)).float()).bfloat16()
    parent = write(tmp_path, {"w": values})
    table = open_tensors(parent)
    out = tmp_path / "c.safetensors"
    apply(table, recipe_for(table, [step("add", {"tensor": "w"}, value=0.37)]), out, base_id=BASE_ID)
    expected = (values.float() + np.float32(0.37)).bfloat16().view(torch.int16).numpy().astype(np.uint16)
    assert np.array_equal(np.frombuffer(open_tensors(out).read_bytes("w"), np.uint16), expected)


# ------------------------------------------------------------------ crash safety

def test_crash_during_write_leaves_no_visible_child(tmp_path, monkeypatch):
    parent = write(tmp_path, {"a": torch.randn(500), "b": torch.randn(500), "c": torch.randn(500)})
    table = open_tensors(parent)
    recipe = recipe_for(table, [step("scale", {"tensor": "a"}, k=2.0), step("scale", {"tensor": "c"}, k=2.0)])
    out = tmp_path / "child.safetensors"
    real = edit._run_tensor
    calls = []

    def flaky(*args, **kw):
        calls.append(args[1])
        if len(calls) == 4:  # the second pass over the second touched tensor: the file is half written
            raise RuntimeError("power loss")
        return real(*args, **kw)

    monkeypatch.setattr(edit, "_run_tensor", flaky)
    with pytest.raises(RuntimeError):
        apply(table, recipe, out, base_id=BASE_ID)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["parent.safetensors"]
    monkeypatch.undo()
    monkeypatch.setattr(edit.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        apply(table, recipe, out, base_id=BASE_ID)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["parent.safetensors"]
    monkeypatch.undo()
    apply(table, recipe, out, base_id=BASE_ID)  # and a retry afterwards works
    assert out.exists()
    err(lambda: apply(table, recipe, out, base_id=BASE_ID), code="output_exists")


# ------------------------------------------------------------------ sequencing, overlaps, sources, blobs

def test_ops_run_in_order_and_overlaps_are_reported(tmp_path):
    w = torch.randn(4, 6)
    parent = write(tmp_path, {"w": w})
    table = open_tensors(parent)
    recipe = recipe_for(table, [step("scale", {"tensor": "w", "index": {"0": [0, 1, 2]}}, k=2.0),
                                step("add", {"tensor": "w", "index": {"0": [2, 3], "1": {"start": 0, "stop": 3}}}, value=1.0),
                                step("clamp", {"tensor": "w", "index": {"0": [0]}}, min=0.0)])
    report = plan(table, recipe, base_id=BASE_ID)["result"]
    assert report["overlaps"] == [{"tensor": "w", "ops": [0, 1], "elems": 3}, {"tensor": "w", "ops": [0, 2], "elems": 6}]
    assert report["overlaps_total"] == 2
    assert report["touched_elems_total"] == 3 * 6 + 3  # union: rows 0-2 plus the three new elements of row 3
    out = tmp_path / "c.safetensors"
    apply(table, recipe, out, base_id=BASE_ID)
    expected = w.numpy().copy()
    expected[:3] *= np.float32(2)
    expected[2:4, :3] += np.float32(1)
    expected[0] = np.maximum(expected[0], np.float32(0))
    assert np.array_equal(np.frombuffer(open_tensors(out).read_bytes("w"), "<f4").reshape(4, 6), expected)


def test_lerp_endpoints_and_source_errors(tmp_path):
    a, b = torch.randn(3, 4), torch.randn(3, 4)
    table = open_tensors(write(tmp_path, {"w": a}))
    other = open_tensors(write(tmp_path, {"w": b}, "other.safetensors"))
    sid = "ef" * 32
    for t, expected in ((0.0, a.numpy()), (1.0, b.numpy()), (0.5, a.numpy() + (b.numpy() - a.numpy()) * np.float32(0.5))):
        out = tmp_path / f"l{t}.safetensors"
        recipe = recipe_for(table, [step("lerp", {"tensor": "w"}, source=sid, t=t)])
        report = apply(table, recipe, out, base_id=BASE_ID, sources={sid: other}, allow_noop=True)
        assert np.array_equal(np.frombuffer(open_tensors(out).read_bytes("w"), "<f4").reshape(3, 4), expected)
        assert report["result"]["touched_elems_total"] == 12
    recipe = recipe_for(table, [step("copy_from", {"tensor": "w"}, source=sid)])
    err(lambda: plan(table, recipe, base_id=BASE_ID), code="source_missing", field="program[0].args.source")
    wrong = open_tensors(write(tmp_path, {"w": torch.randn(3, 5)}, "wrong.safetensors"))
    err(lambda: plan(table, recipe, base_id=BASE_ID, sources={sid: wrong}), code="incompatible_source")
    half = open_tensors(write(tmp_path, {"w": torch.randn(3, 4).half()}, "half.safetensors"))
    err(lambda: plan(table, recipe, base_id=BASE_ID, sources={sid: half}), code="incompatible_source")
    err(lambda: recipe_for(table, [step("lerp", {"tensor": "w"}, source=sid, t=1.5)]), code="invalid_argument", field="program[0].args.t")


def test_blob_validation(tmp_path):
    table = open_tensors(write(tmp_path, {"w": torch.randn(2, 3)}))
    data = np.arange(6, dtype="<f4").tobytes()
    blob_id = "sha256:" + hashlib.sha256(data).hexdigest()
    recipe = recipe_for(table, [step("add", {"tensor": "w"}, blob=blob_id)])
    err(lambda: plan(table, recipe, base_id=BASE_ID), code="blob_missing", field="program[0].args.blob")
    err(lambda: plan(table, recipe, base_id=BASE_ID, blobs={blob_id: data[:-4]}), code="blob_mismatch")
    short = np.arange(5, dtype="<f4").tobytes()
    short_id = "sha256:" + hashlib.sha256(short).hexdigest()
    short_recipe = recipe_for(table, [step("set", {"tensor": "w"}, blob=short_id)])
    err(lambda: plan(table, short_recipe, base_id=BASE_ID, blobs={short_id: short}), code="blob_size")
    nan = np.array([1, 2, 3, 4, 5, float("nan")], dtype="<f4").tobytes()
    nan_id = "sha256:" + hashlib.sha256(nan).hexdigest()
    err(lambda: plan(table, recipe_for(table, [step("set", {"tensor": "w"}, blob=nan_id)]), base_id=BASE_ID, blobs={nan_id: nan}),
        code="non_finite")
    ok = plan(table, recipe, base_id=BASE_ID, blobs={blob_id: data})
    assert ok["result"]["ops"][0]["selected_elems"] == 6
    err(lambda: plan(table, recipe, base_id=BASE_ID, blobs={blob_id: data}, limits=Limits(max_blob_bytes=8)), code="cap_exceeded")


# ------------------------------------------------------------------ schema, caps, refusals

def test_recipe_is_canonical_and_strict(tmp_path):
    table = open_tensors(write(tmp_path, {"w": torch.randn(6, 4), "i": torch.arange(4)}))
    a = recipe_for(table, [step("scale", {"tensor": "w", "index": {"0": [3, 1, 2, 2]}}, k=2)])
    b = recipe_for(table, [step("scale", {"tensor": "w", "index": {"0": {"start": 1, "stop": 4}}, "shape": [6, 4]}, k=2.0)])
    assert a == b and a["recipe_hash"] == b["recipe_hash"] and a["program"][0]["args"] == {"k": 2.0}
    assert a["program"][0]["select"]["index"]["0"] == [{"start": 1, "stop": 4}]
    assert a["engine"] == {"layerdelta_ops_revision": 1, "plugins": {}, "numeric": "cpu-f32-round-per-op"} and a["evaluation"] is None
    assert verify_recipe(a) == a and verify_recipe(json.loads(canonical_json(a))) == a
    assert a["recipe_hash"] == hashlib.sha256(canonical_json({k: v for k, v in a.items() if k != "recipe_hash"}).encode()).hexdigest()
    bad = {**a, "hypothesis": "edited"}
    err(lambda: verify_recipe(bad), code="recipe_hash_mismatch")
    err(lambda: verify_recipe({**a, "engine": {**a["engine"], "numeric": "gpu"}}), code="unsupported_engine")
    err(lambda: verify_recipe({**a, "extra": 1}), code="unsupported_schema")
    err(lambda: verify_recipe({**a, "schema_version": 1}), code="unsupported_schema")
    err(lambda: verify_recipe({**a, "program": []}), code="invalid_argument")
    noncanonical = json.loads(json.dumps(a))
    noncanonical["program"][0]["args"]["k"] = True
    err(lambda: verify_recipe(noncanonical), code="invalid_argument")
    err(lambda: plan(table, a, base_id="cd" * 32), code="base_mismatch")
    cases = [
        (step("explode", {"tensor": "w"}), "invalid_argument", "program[0].op"),
        (step("scale", {"tensor": "w"}), "invalid_argument", "program[0].args.k"),
        (step("scale", {"tensor": "w"}, k=1, extra=2), "unknown_field", "program[0].args.extra"),
        (step("zero", {"tensor": "w"}, k=1), "unknown_field", "program[0].args.k"),
        (step("scale", {"tensor": "w"}, k=float("nan")), "invalid_argument", "program[0].args.k"),
        (step("scale", {"tensor": "w"}, k=1e39), "invalid_argument", "program[0].args.k"),
        (step("scale", {"tensor": "w"}, k=True), "invalid_argument", "program[0].args.k"),
        (step("set", {"tensor": "w"}), "invalid_argument", "program[0].args"),
        (step("set", {"tensor": "w"}, value=1, blob="sha256:" + "0" * 64), "invalid_argument", "program[0].args"),
        (step("clamp", {"tensor": "w"}), "invalid_argument", "program[0].args"),
        (step("clamp", {"tensor": "w"}, min=2, max=1), "invalid_argument", "program[0].args.min"),
        (step("add", {"tensor": "w"}, blob="nope"), "invalid_argument", "program[0].args.blob"),
        (step("copy_from", {"tensor": "w"}, source="short"), "invalid_argument", "program[0].args.source"),
        (step("scale", {"tensor": "nope"}, k=1), "unknown_tensor", "program[0].select.tensor"),
        (step("scale", {"tensor": "w", "index": {"0": [6]}}, k=1), "index_out_of_range", "program[0].select.index.0[0]"),
        (step("scale", {"tensor": "w", "index": {"0": [-1]}}, k=1), "invalid_index", "program[0].select.index.0[0]"),
        (step("scale", {"tensor": "i"}, k=2), "read_only_dtype", None),
        ({"op": "zero", "args": {}}, "invalid_argument", "program[0]"),
    ]
    for program_step, code, field in cases:
        if code == "read_only_dtype":  # the table says so, not the schema: freeze accepts it, plan refuses
            frozen = recipe_for(table, [program_step])
            err(lambda: plan(table, frozen, base_id=BASE_ID), code=code, field="program[0].select.tensor")
        else:
            err(lambda: recipe_for(table, [program_step]), code=code, field=field)
    err(lambda: recipe_for(table, [step("zero", {"tensor": "w"})], hypothesis=" "), code="invalid_argument", field="hypothesis")
    err(lambda: recipe_for(table, [step("zero", {"tensor": "w"})], base_id="xyz"), code="invalid_argument")


def test_caps(tmp_path):
    table = open_tensors(write(tmp_path, {"w": torch.randn(50, 4)}))
    program = [step("scale", {"tensor": "w", "index": {"0": [i]}}, k=2.0) for i in range(5)]
    recipe = recipe_for(table, program)
    err(lambda: plan(table, recipe, base_id=BASE_ID, limits=Limits(max_program_ops=4)), code="cap_exceeded", field="program")
    err(lambda: plan(table, recipe, base_id=BASE_ID, limits=Limits(max_touched_elems=19)), code="cap_exceeded", field="program")
    err(lambda: plan(table, recipe, base_id=BASE_ID, limits=Limits(max_tensor_elems=100)), code="cap_exceeded", field="program[0]")
    assert plan(table, recipe, base_id=BASE_ID, limits=Limits(max_touched_elems=20))["result"]["touched_elems_total"] == 20


def test_validate_reports_data_and_plan_has_no_side_effects(tmp_path):
    parent = write(tmp_path, {"w": torch.randn(4, 4)})
    table = open_tensors(parent)
    good = recipe_for(table, [step("scale", {"tensor": "w", "index": {"1": [0]}}, k=3.0)])
    assert validate(table, good, base_id=BASE_ID) == {"valid": True, "checks": list(edit.CHECKS), "error": None}
    env = plan(table, good, base_id=BASE_ID)
    assert env["side_effects"] == [] and env["fingerprints"] == {"checkpoint": BASE_ID, "recipe": good["recipe_hash"],
                                                                  "selection": None, "op_revision": 1}
    assert env["result"]["tensors"][0]["touched_elems"] == 4 and env["result"]["child_bytes"] == parent.stat().st_size
    assert env["result"]["numeric"] == "cpu-f32-round-per-op" and env["result"]["rounding"].startswith("nearest-even")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["parent.safetensors"]
    assert json.dumps(env, sort_keys=True) == json.dumps(plan(table, good, base_id=BASE_ID), sort_keys=True)  # deterministic


def test_sharded_base_is_refused(tmp_path):
    save_file({"x": torch.ones(2)}, tmp_path / "a.safetensors")
    index = tmp_path / "m.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"x": "a.safetensors"}}))
    table = open_tensors(index)
    recipe = recipe_for(table, [step("zero", {"tensor": "x"})])
    err(lambda: plan(table, recipe, base_id=BASE_ID), code="unsupported_base")


def test_multi_range_axes_and_scalars(tmp_path):
    w = torch.randn(6, 5, 4)
    s = torch.tensor(2.5)
    table = open_tensors(write(tmp_path, {"w": w, "s": s}))
    select = {"tensor": "w", "index": {"0": [0, 1, 4], "1": [{"start": 0, "stop": 1}, {"start": 3, "stop": 5}]}}
    recipe = recipe_for(table, [step("scale", select, k=-1.0), step("set", {"tensor": "s"}, value=9.0)])
    out = tmp_path / "c.safetensors"
    apply(table, recipe, out, base_id=BASE_ID)
    expected = w.numpy().copy()
    expected[np.ix_([0, 1, 4], [0, 3, 4])] *= np.float32(-1)
    assert np.array_equal(np.frombuffer(open_tensors(out).read_bytes("w"), "<f4").reshape(6, 5, 4), expected)
    assert np.frombuffer(open_tensors(out).read_bytes("s"), "<f4").tolist() == [9.0]


# ------------------------------------------------------------------ PR #20 review fixes

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_program_equals_stepwise_applies_bitwise(tmp_path, dtype):
    """Rounding after every op: [a, b, c] must equal apply(apply(apply(a), b), c) even for half precision."""
    torch.manual_seed(5)
    parent = write(tmp_path, {"w": (torch.randn(8, 6) * 3).to(dtype)})
    table = open_tensors(parent)
    first = step("scale", {"tensor": "w", "index": {"0": [1, 2, 3]}}, k=0.3)
    second = step("add", {"tensor": "w", "index": {"0": [2, 3, 4]}}, value=0.1)
    third = step("scale", {"tensor": "w", "index": {"0": [3]}}, k=1.7)
    both = tmp_path / "both.safetensors"
    apply(table, recipe_for(table, [first, second, third]), both, base_id=BASE_ID)
    current, base = parent, BASE_ID
    for i, one in enumerate((first, second, third)):
        t = open_tensors(current)
        out = tmp_path / f"step{i}.safetensors"
        apply(t, recipe_for(t, [one], base_id=base), out, base_id=base)
        current, base = out, file_sha(out)
    assert both.read_bytes() == current.read_bytes()
    assert file_sha(both) != file_sha(parent)


def test_overflow_is_detected_per_op(tmp_path):
    parent = write(tmp_path, {"w": torch.tensor([30000.0, 1.0], dtype=torch.float16)})
    table = open_tensors(parent)
    recipe = recipe_for(table, [step("scale", {"tensor": "w"}, k=3.0), step("scale", {"tensor": "w"}, k=0.01)])
    err(lambda: plan(table, recipe, base_id=BASE_ID), code="dtype_overflow", field="program[0]")  # 90000 never fits F16


def test_noncanonical_select_and_evaluation_are_refused(tmp_path):
    table = open_tensors(write(tmp_path, {"w": torch.randn(6, 4)}))
    good = recipe_for(table, [step("scale", {"tensor": "w", "index": {"0": [1, 2, 3]}}, k=2.0)])

    def mutate(change):
        recipe = json.loads(json.dumps(good))
        change(recipe["program"][0]["select"])
        recipe["recipe_hash"] = edit.digest({k: v for k, v in recipe.items() if k != "recipe_hash"})  # the hash stays valid
        return recipe

    variants = {
        "raw index list": lambda sel: sel.update(index={"0": [1, 2, 3]}),
        "missing shape": lambda sel: sel.pop("shape"),
        "extra key": lambda sel: sel.update(checkpoint="x"),
        "missing axis": lambda sel: sel["index"].pop("1"),
        "extra axis": lambda sel: sel["index"].update({"2": [{"start": 0, "stop": 1}]}),
        "empty ranges": lambda sel: sel["index"].update({"1": []}),
        "unsorted": lambda sel: sel["index"].update({"0": [{"start": 4, "stop": 5}, {"start": 1, "stop": 2}]}),
        "overlapping": lambda sel: sel["index"].update({"0": [{"start": 1, "stop": 3}, {"start": 2, "stop": 4}]}),
        "adjacent": lambda sel: sel["index"].update({"0": [{"start": 1, "stop": 2}, {"start": 2, "stop": 3}]}),
        "empty range": lambda sel: sel["index"].update({"0": [{"start": 2, "stop": 2}]}),
        "beyond shape": lambda sel: sel["index"].update({"0": [{"start": 1, "stop": 7}]}),
        "extra range key": lambda sel: sel["index"]["0"][0].update(step=1),
        "float bound": lambda sel: sel["index"].update({"0": [{"start": 1.0, "stop": 3}]}),
        "negative shape": lambda sel: sel.update(shape=[-6, 4]),
    }
    for name, change in variants.items():
        err(lambda: verify_recipe(mutate(change)), code="noncanonical_recipe", field="program[0].select")
    assert verify_recipe(mutate(lambda sel: None)) == json.loads(json.dumps(good))
    # a canonical-looking selector that is not the base's own resolution is refused when resolving
    stale = mutate(lambda sel: sel.update(shape=[7, 4], index={"0": [{"start": 1, "stop": 4}], "1": [{"start": 0, "stop": 4}]}))
    err(lambda: plan(table, stale, base_id=BASE_ID), code="invalid_argument", field="program[0].select.shape")
    for bad in ({"x": float("nan")}, ["list"], "text"):
        err(lambda: recipe_for(table, [step("zero", {"tensor": "w"})], evaluation=bad), code="invalid_argument", field="evaluation")
    ok = recipe_for(table, [step("zero", {"tensor": "w"})], evaluation={"evaluator": "classification@1"})
    assert ok["evaluation"] == {"evaluator": "classification@1"}


def test_apply_never_overwrites_an_existing_file(tmp_path, monkeypatch):
    parent = write(tmp_path, {"w": torch.randn(8)})
    table = open_tensors(parent)
    recipe = recipe_for(table, [step("scale", {"tensor": "w"}, k=2.0)])
    out = tmp_path / "child.safetensors"
    real = edit._verify_child

    def appears(*args):  # somebody creates the output between the exists() check and publication
        real(*args)
        out.write_bytes(b"precious")

    monkeypatch.setattr(edit, "_verify_child", appears)
    err(lambda: apply(table, recipe, out, base_id=BASE_ID), code="output_exists")
    assert out.read_bytes() == b"precious"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["child.safetensors", "parent.safetensors"]  # no partial left


def fixture_dir():
    default = Path(__file__).resolve().parents[1] / ".layerdelta" / "reference"
    return Path(os.environ.get("LAYERDELTA_FIXTURE_DIR", default))


@pytest.mark.parametrize("variant", ["example", "block1-scale"])
def test_real_reference_checkpoint_v1_translation_equals_v01_export(tmp_path, variant):
    """The acceptance criterion on the real trained checkpoint (CI builds it and requires this test)."""
    model_path = fixture_dir() / "model.safetensors"
    if not model_path.is_file():
        if os.environ.get("LAYERDELTA_REQUIRE_FIXTURE"):
            pytest.fail(f"LAYERDELTA_REQUIRE_FIXTURE is set but {model_path} does not exist")
        pytest.skip("no reference fixture (build it with python -m layerdelta.fixture, or set LAYERDELTA_FIXTURE_DIR)")
    v1 = (json.loads(EXAMPLE.read_text(encoding="utf-8")) if variant == "example"
          else v1_recipe("features.block1.conv.kernel", 3, "scale", 0.5))
    actual = file_sha(model_path)
    pinned = v1["dependencies"]["checkpoint_sha256"]
    if variant == "example" and actual != pinned:
        # A fixture trained on other hardware or numeric libraries is not byte-identical to the pinned one. The claim
        # under test (the v2 child equals the v0.1 export of the SAME parent) does not depend on which parent that is,
        # so rebase the recipe onto the checkpoint we actually have, say so loudly, and fail only if asked to.
        message = f"reference fixture drift: built {actual} vs pinned {pinned} in the example recipe"
        print(message)
        warnings.warn(message)
        if os.environ.get("LAYERDELTA_REQUIRE_PINNED_DIGEST"):
            pytest.fail(message)
        v1 = json.loads(json.dumps(v1))
        v1["dependencies"]["checkpoint_sha256"] = actual
        v1["recipe_hash"] = edit.digest({k: v for k, v in v1.items() if k != "recipe_hash"})
    table = open_tensors(model_path)
    v2 = translate_v1(v1, table)
    child = tmp_path / "child.safetensors"
    envelope = apply(table, v2, child, base_id=v1["dependencies"]["checkpoint_sha256"])
    legacy = v01_export(tmp_path, model_path, v1, "legacy.safetensors")
    assert file_sha(child) == file_sha(legacy) == envelope["result"]["manifest"]["child_sha256"]
    assert child.read_bytes() == legacy.read_bytes()
