"""Versioned HTTP request schemas shared by the API and typed client."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Body(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Ticket(Body):
    ticket: str


class Edit(Body):
    hypothesis: str
    intervention: dict
    checkpoint_id: str | None = None
    dataset_revision_id: str | None = None
    evaluation: dict | None = None


class Run(Body):
    kind: Literal["baseline", "intervention"]
    device: Literal["cpu", "cuda"] = "cpu"
    recipe_id: str | None = None
    checkpoint_id: str | None = None
    dataset_revision_id: str | None = None
    evaluation: dict | None = None
    retry_of_run_id: str | None = None


class Preview(Body):
    sample_id: str
    target_id: str
    channel_index: int
    device: Literal["cpu", "cuda"] = "cpu"
    checkpoint_id: str | None = None
    dataset_revision_id: str | None = None


class ImportPath(Body):
    path: str
    display_name: str | None = None


class Final(Body):
    recipe_id: str
    recipe_hash: str
    confirmed: bool
    device: Literal["cpu", "cuda"] = "cpu"


class Export(Body):
    source_run_id: str
    confirmed: bool


class Cursored(Body):
    cursor: str | None = None


class ReadTensorRequest(Cursored):
    selector: dict
    max_elems: int | None = None
    include_range: bool | None = None


class StatsRequest(Cursored):
    selector: dict
    reduce_over_axes: list[int] | None = None
    bins: int | None = None
    limit: int | None = None


class DiffRequest(Cursored):
    other: str
    selector: dict | None = None
    name_glob: str | None = None
    unit_axis: int | None = 0  # null turns the per-unit breakdown off
    values: bool = False
    include_same: bool = False
    limit: int | None = None
    max_elems: int | None = None
    include_range: bool | None = None


class QueryRequest(Cursored):
    name_glob: str | None = None
    dtype: list[str] | None = None
    min_ndim: int | None = None
    max_ndim: int | None = None
    editable: bool | None = None
    min_bytes: int | None = None
    max_bytes: int | None = None
    where: dict | None = None
    limit: int | None = None
    max_scan_elems: int | None = None


class EditRequest(Body):
    """Either a frozen `recipe` (recipe v2) or a `program` plus `hypothesis` that the server freezes."""
    program: list[dict] | None = None
    hypothesis: str | None = None
    evaluation: dict | None = None
    recipe: dict | None = None


class ApplyRequest(EditRequest):
    allow_noop: bool = False


class BudgetChange(Body):
    dimension: Literal["values", "scan", "applies", "blob_bytes", "evals", "eval_samples"]
    limit: int | None = Field(ge=0)  # null removes the limit (not for applies)


class ResolveSelections(Body):
    ids: list[str] = Field(max_length=500)


class EvaluateRequest(Body):
    """Evaluate a checkpoint on a registered dataset. `baseline` absent means the checkpoint's parent; null means absolute metrics only."""
    dataset_id: str
    evaluator: str | dict
    split: str = "exploration"
    baseline: str | None = None
    limit_samples: int | None = None
    graph: str | None = None


class ProbeRequest(Body):
    """Aggregate the output of graph nodes per unit over exploration samples (weights:read; raw values need weights:values)."""
    dataset_id: str
    nodes: list[str]
    unit_axis: int = 1
    split_by: str | None = None
    bins: int = 0
    values: bool = False
    limit_samples: int | None = None
    graph: str | None = None


class FinalEvaluateRequest(Body):
    """A person confirms the child's frozen recipe hash and evaluates it on the holdout split of a dataset."""
    dataset_id: str
    evaluator: str | dict
    recipe_hash: str | None = None   # null only for a checkpoint no recipe made (an imported root)
    confirmed: bool
    baseline: str | None = None
    graph: str | None = None


class DatasetImport(Body):
    """Import a directory of labelled images in one of the closed formats (datasets.py). A person does this; agents only choose."""
    path: str
    format: str
    name: str
    preprocessing: dict | None = None


class AttachGraph(Body):
    """Attach a graph: build one with a provider, or reuse one that was offered for this layout (never automatic)."""
    provider: str | None = None
    model: str | None = None            # builtin
    reuse: str | None = None
    entry: str | None = None            # torch_export: "package.module:callable"
    path: str | None = None             # torch_export: the directory put on sys.path; onnx: the .onnx file
    example_inputs: list[dict] | None = None
    root: str | None = None
    dynamic_dim: int | None = None      # onnx: value for dynamic input dimensions (default 1)
