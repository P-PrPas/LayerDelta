import json

import httpx
import pytest

from layerdelta.agent_demo import experiment
from layerdelta.client import AgentClient, ClientError, RunFailed, wait_for
from layerdelta.schemas import Run


def test_lost_response_reuses_submission_identity():
    received = []

    def handle(request):
        assert request.url.path == "/api/v1/projects/project/runs"
        assert request.headers["authorization"] == "Bearer local"
        received.append((request.headers["idempotency-key"], json.loads(request.content)))
        if len(received) == 1:
            raise httpx.ReadError("response lost", request=request)
        return httpx.Response(202, json={"run_id": "saved", "status": "queued"})

    with AgentClient("http://127.0.0.1:8000", "project", "local", transport=httpx.MockTransport(handle)) as client:
        assert client.submit(Run(kind="baseline"), key="single-charge")["run_id"] == "saved"
    assert received == [("single-charge", {"kind": "baseline", "device": "cpu"})] * 2


@pytest.mark.parametrize(("budget", "remaining", "reason"), [(0, 9, "user_budget"), (2, 0, "quota_exhausted")])
def test_budget_stops_before_preview_or_submission(budget, remaining, reason):
    def handle(request):
        assert request.url.path.endswith("/agent-session") and request.method == "GET"
        return httpx.Response(200, json={"limit": 10, "charged": 10 - remaining, "remaining": remaining})

    with AgentClient("http://localhost:8181", "p", "local", transport=httpx.MockTransport(handle)) as client:
        assert experiment(client, budget=budget) == {"submissions": 0, "stop_reason": reason}


def test_publish_failure_is_terminal_and_reports_run_id():
    job = {"run_id": "abc", "status": "failed", "error": {"code": "worker_timeout"}}
    with pytest.raises(RunFailed) as failure:
        wait_for(lambda: job)
    assert failure.value.job == job


def test_error_codes_and_pagination():
    def handle(request):
        if request.url.path.endswith("/agent-session"):
            return httpx.Response(403, json={"error": {"message": "Forbidden", "code": "forbidden"}})
        assert request.url.params["source_sample_id"] == "src/1"
        cursor = request.url.params.get("cursor")
        return httpx.Response(200, json={"items": [cursor or "first"], "next_cursor": None if cursor else "second"})

    with AgentClient("http://localhost:8181", "p", "local", transport=httpx.MockTransport(handle)) as client:
        assert list(client.samples("src/1")) == ["first", "second"]
        with pytest.raises(ClientError) as failure:
            client.quota()
        assert failure.value.code == "forbidden" and failure.value.status == 403


@pytest.mark.parametrize("origin", ["https://example.com", "http://localhost/#ticket", "http://x@localhost:9000"])
def test_agent_token_restricted_to_launcher_origin(origin):
    with pytest.raises(ValueError):
        AgentClient(origin, "p", "token")
