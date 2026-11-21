"""Versioned HTTP request schemas shared by the API and typed client."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


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


class Final(Body):
    recipe_id: str
    recipe_hash: str
    confirmed: bool
    device: Literal["cpu", "cuda"] = "cpu"


class Export(Body):
    source_run_id: str
    confirmed: bool
