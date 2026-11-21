"""Loopback HTTP tools for exploration-only Agent sessions."""

from collections.abc import Callable, Iterator
import time
from typing import Literal, TypedDict
from urllib.parse import quote, urlsplit
from uuid import uuid4

import httpx

from .schemas import Body, Edit, Preview, Run


class Quota(TypedDict):
    limit: int
    charged: int
    remaining: int
    timeout_seconds: float


class Sample(TypedDict):
    sample_id: str
    source_sample_id: str
    label: int
    split: str
    condition: str


class Job(TypedDict, total=False):
    run_id: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    result: dict
    error: dict | None


class ClientError(RuntimeError):
    def __init__(self, code: str, message: str, status: int):
        super().__init__(message)
        self.code, self.status = code, status


class RunFailed(RuntimeError):
    def __init__(self, job: Job):
        self.job = job
        super().__init__(f"{job['run_id']} {job['status']}: {job.get('error')}")


def wait_for(get_status: Callable[[], Job], *, timeout: float = 180) -> Job:
    deadline = time.monotonic() + timeout
    interval = 0.25
    while time.monotonic() < deadline:
        job = get_status()
        if job["status"] == "succeeded":
            return job
        if job["status"] not in ("queued", "running"):
            raise RunFailed(job)
        time.sleep(min(interval, max(0, deadline - time.monotonic())))
        interval = min(1.0, interval * 1.5)
    raise TimeoutError("Polling deadline reached. Recover the run using its ID.")


def _segment(value: str) -> str:
    if not value or value in (".", "..") or "/" in value or "\\" in value:
        raise ValueError("Invalid object ID")
    return quote(value, safe="")


class AgentClient:
    def __init__(self, url: str, project_id: str, token: str, *, transport: httpx.BaseTransport | None = None):
        origin = urlsplit(url)
        if (origin.scheme != "http" or origin.hostname not in ("localhost", "127.0.0.1")
                or origin.username is not None or origin.password is not None or origin.query
                or origin.fragment or origin.path not in ("", "/") or not token):
            raise ValueError("A local HTTP origin and session token are required")
        prefix = f"{origin.scheme}://{origin.netloc}/api/v1/projects/{_segment(project_id)}/"
        self.http = httpx.Client(base_url=prefix, headers={"Authorization": "Bearer " + token},
                                 timeout=15, trust_env=False, follow_redirects=False, transport=transport)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.http.close()

    def _request(self, method, path, body=None, *, key=None, params=None):
        data = body.model_dump(exclude_none=True) if isinstance(body, Body) else body
        headers = {"Idempotency-Key": key} if key else {}
        for attempt in range(3):
            try:
                response = self.http.request(method, path, json=data, headers=headers, params=params)
            except httpx.TransportError:
                if attempt == 2 or not key:
                    raise
                continue
            if response.is_success:
                return response.json()
            error = response.json()["error"]
            if key and attempt < 2 and error["code"] in ("queue_full", "service_unavailable"):
                time.sleep(0.5 * (attempt + 1))
                continue
            raise ClientError(error["code"], error["message"], response.status_code)

    def catalog(self) -> dict:
        return self._request("GET", "catalog")

    def quota(self) -> Quota:
        return self._request("GET", "agent-session")

    def _pages(self, path, **filters):
        params = {k: v for k, v in filters.items() if v is not None}
        while True:
            page = self._request("GET", path, params=params)
            yield from page["items"]
            if not page["next_cursor"]:
                return
            params["cursor"] = page["next_cursor"]

    def samples(self, source_sample_id: str | None = None, condition: str | None = None) -> Iterator[Sample]:
        return self._pages("samples", source_sample_id=source_sample_id, condition=condition)

    def preview(self, request: Preview, *, key: str | None = None) -> Job:
        return self._request("POST", "previews", request, key=key or uuid4().hex)

    def preview_status(self, run_id: str) -> Job:
        return self._request("GET", f"previews/{_segment(run_id)}")

    def propose(self, request: Edit) -> dict:
        return self._request("POST", "recipes", request)

    def import_recipe(self, recipe: dict) -> dict:
        return self._request("POST", "recipes/import", recipe)

    def get_recipe(self, recipe_id: str) -> dict:
        return self._request("GET", f"recipes/{_segment(recipe_id)}")

    def submit(self, request: Run, *, key: str) -> Job:
        if not key:
            raise ValueError("Run submission needs a reusable idempotency key")
        return self._request("POST", "runs", request, key=key)

    def status(self, run_id: str) -> Job:
        return self._request("GET", f"runs/{_segment(run_id)}")

    def cancel(self, run_id: str) -> Job:
        return self._request("POST", f"runs/{_segment(run_id)}/cancel")

    def comparison(self, run_id: str) -> dict:
        return self._request("GET", f"runs/{_segment(run_id)}/comparison")

    def changes(self, run_id: str) -> Iterator[dict]:
        return self._pages(f"runs/{_segment(run_id)}/changes")
