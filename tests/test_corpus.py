"""Cross-family corpus: the generality proof for the v0.2 tensor-space core.

Small seeded toy models of five unrelated families (ResNet-like with BatchNorm and residuals, U-Net
with concat skips and ConvTranspose, ViT-tiny with packed qkv, CRNN with conv + LSTM + linear,
YOLO-like backbone/neck) are saved as safetensors in the test, then run through the SAME generic
pipeline: T1 open, T2 describe/read/stats/diff/query, T3 plan/apply/replay. Nothing in src/ knows any
of them (see test_no_model_specific_code_in_src). Targets of the edit program are picked from the
tensor table through `query`, never by family-specific names.
"""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from layerdelta import edit
from layerdelta.checkpoints import Catalog, import_checkpoint
from layerdelta.inspect_ops import describe, diff, query, read_tensor, stats, to_json
from layerdelta.selector import InspectError
from layerdelta.tensors import open_tensors

ST_DTYPE = {torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16", torch.int64: "I64"}


# ------------------------------------------------------------------ toy model families

class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.downsample = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
                           if stride != 1 or cin != cout else None)

    def forward(self, x):
        y = self.bn2(self.conv2(torch.relu(self.bn1(self.conv1(x)))))
        return torch.relu(y + (x if self.downsample is None else self.downsample(x)))


class ResNetLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(3, 8, 3, 1, 1, bias=False), nn.BatchNorm2d(8)
        self.layer1 = nn.Sequential(BasicBlock(8, 8, 1), BasicBlock(8, 8, 1))
        self.layer2 = nn.Sequential(BasicBlock(8, 16, 2), BasicBlock(16, 16, 1))
        self.layer3 = nn.Sequential(BasicBlock(16, 32, 2))
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.layer3(self.layer2(self.layer1(torch.relu(self.bn1(self.conv1(x))))))
        return self.fc(x.mean((2, 3)))


def double_conv(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU())


class UNetLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1, self.enc2, self.bottleneck = double_conv(3, 8), double_conv(8, 16), double_conv(16, 32)
        self.up2, self.dec2 = nn.ConvTranspose2d(32, 16, 2, 2), double_conv(32, 16)  # input is the concat of skip and up
        self.up1, self.dec1 = nn.ConvTranspose2d(16, 8, 2, 2), double_conv(16, 8)
        self.head = nn.Conv2d(8, 2, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(nn.functional.max_pool2d(e1, 2))
        b = self.bottleneck(nn.functional.max_pool2d(e2, 2))
        d2 = self.dec2(torch.cat([e2, self.up2(b)], 1))
        return self.head(self.dec1(torch.cat([e1, self.up1(d2)], 1)))


class Attention(nn.Module):
    def __init__(self, d, heads=2):
        super().__init__()
        self.heads, self.qkv, self.proj = heads, nn.Linear(d, 3 * d), nn.Linear(d, d)  # packed qkv

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1) * (d // self.heads) ** -.5).softmax(-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(b, n, d))


class ViTBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm1, self.attn, self.norm2 = nn.LayerNorm(d), Attention(d), nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViTTiny(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, d, 4, 4)
        self.cls_token, self.pos_embed = nn.Parameter(torch.randn(1, 1, d)), nn.Parameter(torch.randn(1, 17, d) * .02)
        self.blocks = nn.ModuleList([ViTBlock(d) for _ in range(2)])
        self.norm, self.head = nn.LayerNorm(d), nn.Linear(d, 10)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(len(x), -1, -1), x], 1) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x)[:, 0])


class CRNNLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d((2, 1)),
                                 nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((2, 1)))
        self.rnn = nn.LSTM(32, 24, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(48, 11)

    def forward(self, x):
        f = self.cnn(x)  # (b, 16, 2, w)
        out, _ = self.rnn(f.flatten(1, 2).transpose(1, 2))
        return self.fc(out)


class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, stride=1):
        super().__init__()
        self.conv, self.bn, self.act = nn.Conv2d(cin, cout, k, stride, k // 2, bias=False), nn.BatchNorm2d(cout), nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class YoloLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = ConvBNAct(3, 8, 3, 2)
        self.stage3, self.stage4, self.stage5 = ConvBNAct(8, 16, 3, 2), ConvBNAct(16, 32, 3, 2), ConvBNAct(32, 64, 3, 2)
        self.lateral5, self.lateral4 = ConvBNAct(64, 32, 1), ConvBNAct(32, 16, 1)
        self.fuse4, self.fuse3 = ConvBNAct(64, 32), ConvBNAct(32, 16)  # after upsample + concat
        self.head3, self.head4, self.head5 = (nn.Conv2d(c, 15, 1) for c in (16, 32, 64))

    def forward(self, x):
        c3 = self.stage3(self.stem(x))
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        p4 = self.fuse4(torch.cat([c4, nn.functional.interpolate(self.lateral5(c5), scale_factor=2)], 1))
        p3 = self.fuse3(torch.cat([c3, nn.functional.interpolate(self.lateral4(p4), scale_factor=2)], 1))
        return self.head3(p3), self.head4(p4), self.head5(c5)


FAMILIES = {
    "resnet": (ResNetLike, (2, 3, 16, 16)),
    "unet": (UNetLike, (1, 3, 16, 16)),
    "vit": (ViTTiny, (2, 3, 16, 16)),
    "crnn": (CRNNLike, (2, 1, 8, 12)),
    "yolo": (YoloLike, (1, 3, 32, 32)),
}
CASES = [("resnet", torch.float32), ("resnet", torch.float16), ("unet", torch.float32), ("unet", torch.bfloat16),
         ("vit", torch.float32), ("vit", torch.bfloat16), ("crnn", torch.float32), ("crnn", torch.float16),
         ("yolo", torch.float32), ("yolo", torch.bfloat16)]


def build(family, dtype, seed=0):
    torch.manual_seed(seed)
    model = FAMILIES[family][0]().eval()
    for module in model.modules():  # make BatchNorm statistics distinct from the 0/1 defaults
        if isinstance(module, nn.BatchNorm2d):
            module.running_mean.normal_(0, .1)
            module.running_var.uniform_(.5, 1.5)
    return model, {k: v.detach().to(dtype) if v.is_floating_point() else v.clone() for k, v in model.state_dict().items()}


def save(tmp_path, state, name="parent.safetensors"):
    path = tmp_path / name
    save_file({k: v.contiguous().clone() for k, v in state.items()}, path)
    return path


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def as_torch(values, dtype):
    if dtype == torch.int64:
        values = [int(v) for v in values]
    return torch.tensor(values, dtype=dtype)


# ------------------------------------------------------------------ the generic pipeline, per family and dtype

@pytest.mark.parametrize("family,dtype", CASES, ids=[f"{f}-{str(d).split('.')[1]}" for f, d in CASES])
def test_family_through_the_whole_tensor_space_pipeline(tmp_path, family, dtype):
    model, state = build(family, dtype)
    parent = save(tmp_path, state)
    table = open_tensors(parent)  # T1
    base_id = file_sha(parent)

    # T1: the table equals the state dict, names, dtypes and shapes (nothing about the family is known)
    assert {n: (i.dtype, i.shape) for n, i in table.tensors.items()} == {
        n: (ST_DTYPE[t.dtype], tuple(t.shape)) for n, t in state.items()}

    # T2 describe: structure only, no roles guessed, integer buffers read-only
    env = describe(table, checkpoint_id=base_id, limit=2000)
    rows = {r["name"]: r for r in env["result"]["tensors"]}
    assert len(rows) == len(state) and env["truncated"] is False
    assert env["result"]["checkpoint"]["element_count"] == sum(t.numel() for t in state.values())
    assert env["result"]["coverage"]["unknown"] == sum(t.ndim for t in state.values())
    assert all(r["roles"] == [] and r["identical_bytes_group"] is None for r in rows.values())
    for name, tensor in state.items():
        assert rows[name]["editable"] == tensor.is_floating_point()
        assert (rows[name]["read_only_reason"] is None) == tensor.is_floating_point()

    # T2 read_tensor: first and last unit of axis 0 of every tensor equal the original values
    for name, tensor in state.items():
        if tensor.ndim == 0:
            continue
        for index in (0, tensor.shape[0] - 1):
            got = read_tensor(table, {"tensor": name, "index": {"0": [index]}}, max_elems=10000)["result"]["values"]
            assert torch.equal(as_torch(got, tensor.dtype), tensor[index].flatten()), name
    # selector contract: resolved selectors resubmit
    first = read_tensor(table, {"tensor": next(n for n, t in state.items() if t.ndim >= 2 and t.shape[0] >= 2),
                                "index": {"0": [1, 0]}})
    assert read_tensor(table, first["resolved"]["selector"])["fingerprints"] == first["fingerprints"]

    # T2 stats: per out-unit statistics equal numpy float64 over the stored values
    weight = next(n for n, t in state.items() if t.ndim >= 2 and t.is_floating_point())
    result = stats(table, {"tensor": weight}, reduce_over_axes=list(range(1, state[weight].ndim)), bins=8)["result"]
    wide = state[weight].double().numpy()
    for row in result["units"]:
        values = wide[row["index"]["0"]].ravel()
        assert row["count"] == values.size and row["min"] == values.min() and row["max"] == values.max()
        assert row["mean"] == pytest.approx(values.mean(), rel=1e-9, abs=1e-12)
        assert row["l2"] == pytest.approx(float(np.sqrt((values ** 2).sum())), rel=1e-9)
    assert all(sum(c) == result["units"][0]["count"] for c in result["histogram"]["counts"])

    # T2 query: pick edit targets from the table itself
    matches = query(table, editable=True, min_ndim=2, limit=500)["result"]["matches"]
    names = [m["name"] for m in matches]
    assert len(names) >= 3 and names == sorted(names)
    a, b, c = names[0], names[len(names) // 2], names[-1]
    assert len({a, b, c}) == 3
    ranked = query(table, editable=True, min_ndim=2, where={"stat": "l2", "axis": 0, "op": "gt", "value": 0.0})
    assert [m["name"] for m in ranked["result"]["matches"]] == names  # every float matrix has non-zero units
    chosen = ranked["result"]["matches"][0]
    assert read_tensor(table, chosen["selector"])["fingerprints"]["selection"] == chosen["selection_id"]

    # T3: a three-op program picked generically; plan -> apply -> replay to the identical child
    program = [
        {"op": "scale", "args": {"k": 0.5}, "select": {"tensor": a, "index": {"0": [0]}}},
        {"op": "add", "args": {"value": 0.25}, "select": {"tensor": b}},
        {"op": "set", "args": {"value": 0.125}, "select": {"tensor": c, "index": {"0": [0]}}},
    ]
    recipe = edit.freeze_recipe(table, base_id, program, f"corpus {family}")
    planned = edit.plan(table, recipe, base_id=base_id)["result"]
    touched = sorted({a, b, c})
    assert [t["tensor"] for t in planned["tensors"]] == touched and not planned["noop"] and planned["overlaps"] == []
    child = tmp_path / "child.safetensors"
    applied = edit.apply(table, recipe, child, base_id=base_id)
    manifest = applied["result"]["manifest"]
    assert manifest["child_sha256"] == file_sha(child) and manifest["parent"] == base_id
    assert [t["tensor"] for t in manifest["touched"]] == touched
    for replay_input in (recipe, json.loads(json.dumps(recipe))):  # same recipe, fresh table, fresh path
        again = tmp_path / f"replay{id(replay_input)}.safetensors"
        envelope = edit.apply(open_tensors(parent), replay_input, again, base_id=base_id)
        assert envelope["result"]["manifest"]["child_sha256"] == manifest["child_sha256"]
        assert again.read_bytes() == child.read_bytes()
    assert to_json(edit.plan(table, recipe, base_id=base_id)) == to_json(edit.plan(open_tensors(parent), recipe, base_id=base_id))

    # T2 diff(parent, child): exactly the touched tensors, with per-unit rows
    delta = diff(table, open_tensors(child), checkpoint_a=base_id, checkpoint_b=manifest["child_sha256"], limit=1000)["result"]
    assert delta["summary"]["changed"] == 3 and delta["summary"]["same"] == len(state) - 3
    assert delta["summary"]["only_a"] == delta["summary"]["only_b"] == delta["summary"]["shape_mismatch"] == []
    assert [t["name"] for t in delta["tensors"]] == touched
    units = {t["name"]: t["changed_units"]["listed"] for t in delta["tensors"]}
    assert units[a] == [0] and units[c] == [0] and units[b] == list(range(state[b].shape[0]))
    assert diff(open_tensors(child), table)["result"]["summary"]["changed"] == 3  # symmetric
    child_values = read_tensor(open_tensors(child), {"tensor": c, "index": {"0": [0]}}, max_elems=10000)["result"]["values"]
    assert set(child_values) == {0.125}
    # integer buffers are never editable
    int_names = [n for n, t in state.items() if not t.is_floating_point()]
    for name in int_names:
        bad = edit.freeze_recipe(table, base_id, [{"op": "zero", "args": {}, "select": {"tensor": name}}], "h")
        with pytest.raises(InspectError) as info:
            edit.plan(table, bad, base_id=base_id)
        assert info.value.code == "read_only_dtype"
    assert [m["name"] for m in query(table, editable=False, limit=500)["result"]["matches"]] == sorted(int_names)
    if dtype != torch.float32:  # numeric contract: round per op, so a program equals its steps, bit for bit
        current, current_id = parent, base_id
        for i, one in enumerate(program):
            t = open_tensors(current)
            out = tmp_path / f"step{i}.safetensors"
            edit.apply(t, edit.freeze_recipe(t, current_id, [one], "step"), out, base_id=current_id)
            current, current_id = out, file_sha(out)
        assert current.read_bytes() == child.read_bytes()


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_toy_models_are_real_networks(family):
    model, _ = build(family, torch.float32)
    with torch.no_grad():
        out = model(torch.randn(FAMILIES[family][1]))
    assert all(torch.isfinite(o).all() for o in (out if isinstance(out, tuple) else (out,)))


# ------------------------------------------------------------------ sharded checkpoint

def test_sharded_checkpoint_is_read_compared_and_imported_but_not_edited(tmp_path):
    _, state = build("vit", torch.float32)
    single = save(tmp_path, state, "single.safetensors")
    folder = tmp_path / "hf"
    folder.mkdir()
    names = sorted(state)
    parts = {f"model-0000{i + 1}-of-00003.safetensors": names[i::3] for i in range(3)}
    for shard, members in parts.items():
        save_file({n: state[n].contiguous().clone() for n in members}, folder / shard)
    index = folder / "model.safetensors.index.json"
    index.write_text(json.dumps({"metadata": {}, "weight_map": {n: s for s, ms in parts.items() for n in ms}}))
    sharded, whole = open_tensors(index), open_tensors(single)
    rows = describe(sharded, limit=1000)["result"]
    assert rows["checkpoint"]["sharded"] is True and rows["checkpoint"]["shard_count"] == 3
    assert {r["shard"] for r in rows["tensors"]} == set(parts)
    assert {r["name"]: (r["dtype"], r["shape"]) for r in rows["tensors"]} == {
        r["name"]: (r["dtype"], r["shape"]) for r in describe(whole, limit=1000)["result"]["tensors"]}
    # the same bytes behind a different file layout compare as identical
    same = diff(sharded, whole)["result"]["summary"]
    assert (same["changed"], same["same"], same["tensors_total"]) == (0, len(state), len(state))
    probe = next(n for n, t in state.items() if t.ndim == 3)
    assert read_tensor(sharded, {"tensor": probe}, max_elems=10000)["result"]["values"] == \
        read_tensor(whole, {"tensor": probe}, max_elems=10000)["result"]["values"]
    recipe = edit.freeze_recipe(whole, "ab" * 32, [{"op": "zero", "args": {}, "select": {"tensor": probe}}], "h")
    with pytest.raises(InspectError) as info:
        edit.plan(sharded, recipe, base_id="ab" * 32)
    assert info.value.code == "unsupported_base"  # documented limit: the base of an edit is one file
    catalog = Catalog(tmp_path / "ws")
    catalog.start()
    entry, existing = import_checkpoint(catalog, index, index.name)
    assert existing is False and entry["sharded"] and entry["shard_count"] == 3 and entry["tensor_count"] == len(state)
    assert entry["adapter_compatible"] is False and entry["element_count"] == sum(t.numel() for t in state.values())


# ------------------------------------------------------------------ the generality claim itself

def test_no_model_specific_code_in_src():
    src = Path(__file__).resolve().parents[1] / "src" / "layerdelta"
    generic = ["tensors.py", "selector.py", "inspect_ops.py", "edit.py", "checkpoints.py", "v2.py"]
    families = re.compile(r"\b(resnet|unet|u-net|vit|crnn|yolo|lstm|transformer|bert|gpt\d*|llama|timm|torchvision|"
                          r"ultralytics|diffusers|huggingface)\b", re.IGNORECASE)
    checked = 0
    for name in generic:
        path = src / name
        if not path.exists():
            continue
        checked += 1
        text = path.read_text(encoding="utf-8")
        assert not families.findall(text), f"{name} mentions a model family: {families.findall(text)}"
        assert "ReferenceCNN" not in text and "from .model" not in text and "import torch.nn" not in text, name
    assert checked >= 5
    # and every tensor of every corpus model went through those modules without any per-family branch:
    assert len({f for f, _ in CASES}) == len(FAMILIES)
