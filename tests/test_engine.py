import pytest
import torch

from layerdelta.engine import infer, intervention, patch_weight, preview
from layerdelta.model import ReferenceCNN, target_catalog


@pytest.fixture
def model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        return ReferenceCNN().eval()


def specification(target, channel, scale=None):
    spec = {"mode": target.mode, "target_id": target.target_id, "channel_index": channel,
            "operation": "zero" if scale is None else "scale"}
    if scale is not None:
        spec["scale"] = scale
    return spec


@pytest.mark.parametrize("block", ["block1", "block2"])
@pytest.mark.parametrize("last", [False, True])
def test_exact_activation_and_kernel_mapping(model, block, last):
    catalog = target_catalog(model)
    activation = catalog[f"features.{block}.relu.output"]
    weight = catalog[f"features.{block}.conv.kernel"]
    channel = activation.channel_count - 1 if last else 0
    image = torch.linspace(0, 1, 784).reshape(1, 1, 28, 28)
    relu = model.get_submodule(activation.module_path)
    original = {}

    def save(output_store):
        def capture(module, args, output):
            output_store["output"] = output.clone()
        return capture

    with relu.register_forward_hook(save(original)), torch.inference_mode():
        model(image)
    observed = {}
    with intervention(model, specification(activation, channel)), relu.register_forward_hook(save(observed)):
        model(image)
    expected = original["output"].clone()
    expected[:, channel].zero_()
    assert torch.equal(expected, observed["output"])
    assert not relu._forward_hooks and not model._forward_hooks and not model._forward_pre_hooks

    expected_state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    metadata = patch_weight(model, specification(weight, channel))
    expected_state[weight.parameter_name][channel].zero_()
    for name, tensor in model.state_dict().items():
        assert torch.equal(expected_state[name], tensor)
    assert metadata["parameter_name"] == weight.parameter_name
    assert metadata["channel_index"] == channel
    assert metadata["after"]["zero_fraction"] == 1


@pytest.mark.parametrize("target_id", ["features.block1.relu.output", "features.block1.conv.kernel"])
def test_identity_zero_and_restoration_after_error(model, target_id):
    images = torch.linspace(0, 1, 2 * 784).reshape(2, 1, 28, 28)
    rows = [{"sample_id": "a"}, {"sample_id": "b"}]
    target = target_catalog(model)[target_id]
    baseline = infer(model, images, rows, "cpu", 1)
    assert [row["sample_id"] for row in baseline] == ["a", "b"]
    assert infer(model, images, rows, "cpu", 1, specification(target, 0, 1)) == baseline
    zero = infer(model, images, rows, "cpu", 1, specification(target, 0))
    assert infer(model, images, rows, "cpu", 1, specification(target, 0, 0)) == zero
    with pytest.raises(RuntimeError, match="interrupted"):
        with intervention(model, specification(target, 0)):
            raise RuntimeError("interrupted")
    assert infer(model, images, rows, "cpu", 1) == baseline


def test_bias_separates_activation_and_weight_edits(model):
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.features.block2.conv.bias[0] = 2
        model.classifier.weight[1, 0] = 1
    data = (model, torch.zeros(1, 1, 28, 28), [{"sample_id": "bias"}], "cpu", 1)
    targets = target_catalog(model)
    activation = infer(*data, specification(targets["features.block2.relu.output"], 0))
    weight = infer(*data, specification(targets["features.block2.conv.kernel"], 0))
    assert (activation[0]["prediction"], weight[0]["prediction"]) == (0, 1)
    assert (activation[0]["logits"][1], weight[0]["logits"][1]) == (0, 2)


def test_missing_hook_invocation_fails_and_is_removed(model, monkeypatch):
    target = target_catalog(model)["features.block2.relu.output"]
    monkeypatch.setattr(model, "forward", lambda x: x.new_zeros(len(x), 10))
    with pytest.raises(ValueError, match="Missing forward invocation"):
        infer(model, torch.zeros(1, 1, 28, 28), [{}], "cpu", 1, specification(target, 0))
    assert not model.features.block2.relu._forward_hooks


def test_preview_matches_direct_tensor(model):
    image = torch.linspace(0, 1, 784).reshape(1, 1, 28, 28)
    block = model.features.block2
    with torch.inference_mode():
        activation = block.relu(block.conv(model.features.block1(image)))[0, -1]
    result = preview(model, image[0], "features.block2.conv.kernel", 23, "cpu")
    assert result["hook"] == "features.block2.relu"
    assert result["kernel"]["shape"] == [8, 3, 3]
    assert result["kernel"]["mean"] == block.conv.weight[-1].detach().double().mean().item()
    assert result["activation"]["values"] == activation.tolist()
    assert result["activation"]["zero_fraction"] == activation.eq(0).double().mean().item()
    assert result["image"] == image[0, 0].tolist()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_export_after_cuda_preview_and_inference(model):
    image = torch.zeros(1, 1, 28, 28)
    target = target_catalog(model)["features.block2.conv.kernel"]
    spec = specification(target, 23, 0.5)
    result = preview(model, image, target.target_id, 23, "cuda")
    assert result["actual_device"].startswith("cuda:")
    staged = infer(model, image, [{}], "cuda", 1, spec)
    patch_weight(model, spec)
    assert infer(model, image, [{}], "cuda", 1) == staged
