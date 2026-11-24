"""Loopback HTTP tools for exploration-only Agent sessions."""

import base64
import binascii
from collections.abc import Callable, Iterator
import time
from typing import Literal, TypedDict
from urllib.parse import quote, urlsplit
from uuid import uuid4

import httpx

from . import registry
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
    def __init__(self, code: str, message: str, status: int, field: str | None = None):
        super().__init__(message)
        self.code, self.status, self.field = code, status, field


_UNSET = object()  # "argument not given", distinct from an explicit None (null)


def _given(**fields) -> dict:
    return {k: v for k, v in fields.items() if v is not None and v is not _UNSET}


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
    def __init__(self, url: str, project_id: str, token: str, *, transport: httpx.BaseTransport | None = None,
                 timeout: float = 15, mutate_timeout: float = 120, egress_class: int | None = None, via: str | None = None,
                 profile: str | None = None):
        origin = urlsplit(url)
        if (origin.scheme != "http" or origin.hostname not in ("localhost", "127.0.0.1")
                or origin.username is not None or origin.password is not None or origin.query
                or origin.fragment or origin.path not in ("", "/") or not token):
            raise ValueError("A local HTTP origin and session token are required")
        prefix = f"{origin.scheme}://{origin.netloc}/api/v1/projects/{_segment(project_id)}/"
        self.http = httpx.Client(base_url=prefix, headers={"Authorization": "Bearer " + token, **({"X-LayerDelta-Via": via} if via else {})},
                                 timeout=timeout, trust_env=False, follow_redirects=False, transport=transport)
        self.profile = profile  # from connection.json: the route set the service allows this token
        self.egress_class = egress_class  # from connection.json: what the service releases to this token (0, 1 or 2)
        self.last_budget = None  # the LayerDelta-Budget header of the last answer: what the session has used
        self.mutate_timeout = mutate_timeout  # POST can wait on the server (apply waits up to 60 s, a large plan takes longer than a read)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.http.close()

    def _request(self, method, path, body=None, *, key=None, params=None, content=None):
        data = body.model_dump(exclude_none=True) if isinstance(body, Body) else body
        headers = {"Idempotency-Key": key} if key else {}
        if content is not None:
            headers["Content-Type"] = "application/octet-stream"
        for attempt in range(3):
            try:
                response = self.http.request(method, path, json=data if content is None else None, content=content,
                                             headers=headers, params=params,
                                             timeout=self.mutate_timeout if method != "GET" else httpx.USE_CLIENT_DEFAULT)
            except httpx.TransportError as exc:
                if attempt == 2 or not key or isinstance(exc, httpx.TimeoutException):  # a timeout may have completed: the caller decides
                    raise
                continue
            self.last_budget = response.headers.get("LayerDelta-Budget")
            try:
                payload = response.json()
                if response.is_success:
                    return payload
                error = payload["error"]
                error["code"], error["message"]
            except (ValueError, KeyError, TypeError):
                raise ClientError("backend_error", f"the service answered {response.status_code} without a structured result",
                                  response.status_code) from None
            if key and attempt < 2 and error["code"] in ("queue_full", "service_unavailable"):
                time.sleep(0.5 * (attempt + 1))
                continue
            raise ClientError(error["code"], error["message"], response.status_code, error.get("field"))

    def call(self, tool: str, **arguments) -> dict:
        """Call any registry tool by name with its input-schema arguments (the MCP bridge goes through here). Same errors as the
        named methods; a malformed call raises `ClientError('invalid_argument', ..., 422)` before anything is sent."""
        spec = registry.tools_by_name().get(tool)
        if spec is None:
            raise ClientError("unknown_tool", f"no tool named {tool!r}", 404)
        route, args = spec["route"], dict(arguments)

        def refuse(message, field):
            raise ClientError("invalid_argument", message, 422, field)

        path = route["path"]
        for name in route["path_params"]:
            if not isinstance(args.get(name), str):
                refuse(f"{name} is required and must be a string", name)
            try:
                path = path.replace("{" + name + "}", _segment(args.pop(name)))
            except ValueError:
                refuse(f"{name} is not a valid object ID", name)
        params = {k: args.pop(k) for k in route["query"] if args.get(k) is not None}
        for k in route["query"]:
            args.pop(k, None)
        key = args.pop("key", None) if route["key"] else None
        if route["key"] and (not isinstance(key, str) or not key):
            refuse("this tool needs an idempotency key (reuse it to retry)", "key")
        content, body = None, None
        if route["body"] == "octet":
            try:
                content = base64.b64decode(args.pop("data_base64", ""), validate=True)
            except (binascii.Error, ValueError, TypeError):
                refuse("data_base64 must be base64", "data_base64")
            if not content or len(content) > registry.BLOB_CALL_MAX:
                refuse(f"a blob is 1 to {registry.BLOB_CALL_MAX} bytes per call", "data_base64")
        elif route["body"] == "json":
            body = args  # unknown fields go to the backend, which refuses them with its own error
            args = {}
        if args:
            refuse(f"unknown argument {sorted(args)[0]!r}", sorted(args)[0])
        return self._request(route["method"], path.lstrip("/"), body, key=key, params=params or None, content=content)

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

    # ---- v2: checkpoints, tensor-space inspection, edit programs (one method per route) ----
    def list_checkpoints(self) -> Iterator[dict]:
        return self._pages("checkpoints")

    def get_checkpoint(self, checkpoint_id: str) -> dict:
        return self._request("GET", f"checkpoints/{_segment(checkpoint_id)}")

    def _v2(self, checkpoint_id: str, name: str) -> str:
        return f"checkpoints/{_segment(checkpoint_id)}/{name}"

    def describe(self, checkpoint_id: str, *, name_glob: str | None = None, limit: int | None = None,
                 cursor: str | None = None) -> dict:
        return self._request("GET", self._v2(checkpoint_id, "describe"),
                             params=_given(name_glob=name_glob, limit=limit, cursor=cursor))

    def read_tensor(self, checkpoint_id: str, selector: dict, *, max_elems: int | None = None,
                    include_range: bool | None = None, cursor: str | None = None) -> dict:
        return self._request("POST", self._v2(checkpoint_id, "read-tensor"), _given(
            selector=selector, max_elems=max_elems, include_range=include_range, cursor=cursor))

    def stats(self, checkpoint_id: str, selector: dict, *, reduce_over_axes: list[int] | None = None,
              bins: int | None = None, limit: int | None = None, cursor: str | None = None) -> dict:
        return self._request("POST", self._v2(checkpoint_id, "stats"), _given(
            selector=selector, reduce_over_axes=reduce_over_axes, bins=bins, limit=limit, cursor=cursor))

    def diff(self, checkpoint_id: str, other: str, *, selector: dict | None = None, name_glob: str | None = None,
             unit_axis=_UNSET, values: bool | None = None, include_same: bool | None = None, limit: int | None = None,
             max_elems: int | None = None, include_range: bool | None = None, cursor: str | None = None) -> dict:
        body = _given(other=other, selector=selector, name_glob=name_glob, values=values, include_same=include_same,
                      limit=limit, max_elems=max_elems, include_range=include_range, cursor=cursor)
        if unit_axis is not _UNSET:
            body["unit_axis"] = unit_axis  # None switches the per-unit breakdown off
        return self._request("POST", self._v2(checkpoint_id, "diff"), body)

    def query(self, checkpoint_id: str, *, name_glob: str | None = None, dtype: list[str] | None = None,
              min_ndim: int | None = None, max_ndim: int | None = None, editable: bool | None = None,
              min_bytes: int | None = None, max_bytes: int | None = None, where: dict | None = None,
              limit: int | None = None, max_scan_elems: int | None = None, cursor: str | None = None) -> dict:
        return self._request("POST", self._v2(checkpoint_id, "query"), _given(
            name_glob=name_glob, dtype=dtype, min_ndim=min_ndim, max_ndim=max_ndim, editable=editable,
            min_bytes=min_bytes, max_bytes=max_bytes, where=where, limit=limit, max_scan_elems=max_scan_elems,
            cursor=cursor))

    def graph(self, checkpoint_id: str, *, view: str | None = None, module_prefix: str | None = None, op: str | None = None,
              limit: int | None = None, cursor: str | None = None) -> dict:
        """The attached graph (summary and a page of nodes or of the module tree), or the attach card when there is none."""
        return self._request("GET", self._v2(checkpoint_id, "graph"),
                             params=_given(view=view, module_prefix=module_prefix, op=op, limit=limit, cursor=cursor))

    def graph_node(self, checkpoint_id: str, node_id: str) -> dict:
        return self._request("GET", self._v2(checkpoint_id, f"graph/nodes/{_segment(node_id)}"))

    def groups(self, checkpoint_id: str, *, tensor: str | None = None, node: str | None = None,
               module_prefix: str | None = None, closed: bool | None = None, kind: str | None = None,
               limit: int | None = None, cursor: str | None = None) -> dict:
        return self._request("GET", self._v2(checkpoint_id, "groups"), params=_given(
            tensor=tensor, node=node, module_prefix=module_prefix, closed=closed, kind=kind, limit=limit, cursor=cursor))

    def plan(self, checkpoint_id: str, *, program: list[dict] | None = None, hypothesis: str | None = None,
             evaluation: dict | None = None, recipe: dict | None = None) -> dict:
        return self._request("POST", self._v2(checkpoint_id, "plan"), _given(
            program=program, hypothesis=hypothesis, evaluation=evaluation, recipe=recipe))

    def validate(self, checkpoint_id: str, *, program: list[dict] | None = None, hypothesis: str | None = None,
                 evaluation: dict | None = None, recipe: dict | None = None) -> dict:
        return self._request("POST", self._v2(checkpoint_id, "validate"), _given(
            program=program, hypothesis=hypothesis, evaluation=evaluation, recipe=recipe))

    def apply(self, checkpoint_id: str, *, key: str, program: list[dict] | None = None, hypothesis: str | None = None,
              evaluation: dict | None = None, recipe: dict | None = None, allow_noop: bool | None = None) -> dict:
        """Write a child checkpoint. Reuse `key` to retry after a lost response: same child, charged once."""
        if not key:
            raise ValueError("apply needs a reusable idempotency key")
        return self._request("POST", self._v2(checkpoint_id, "apply"), _given(
            program=program, hypothesis=hypothesis, evaluation=evaluation, recipe=recipe, allow_noop=allow_noop), key=key)

    def upload_blob(self, data: bytes) -> dict:
        """Store a flat little-endian float32 array; the returned blob_id can be named by set/add ops."""
        return self._request("POST", "blobs", content=bytes(data))
