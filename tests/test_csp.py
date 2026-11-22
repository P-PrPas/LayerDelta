"""The served Desktop page, its assets and every API response carry a strict Content-Security-Policy."""

import os
import re
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save_file

from layerdelta import api
from layerdelta.model import ReferenceCNN, describe_model

REPO = Path(__file__).resolve().parents[1]
INDEX = """<!doctype html><html><head><meta charset="utf-8">
<link rel="icon" href="/layerdelta.png"><script type="module" crossorigin src="/assets/app.js"></script>
<link rel="stylesheet" crossorigin href="/assets/app.css"></head><body><div id="root"></div></body></html>"""


def directives(header: str) -> dict:
    parsed = {}
    for part in header.split(";"):
        name, *sources = part.split()
        parsed[name] = sources
    return parsed


@pytest.fixture
def client(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    (workspace / "reference").mkdir(parents=True)
    save_file(ReferenceCNN().state_dict(), workspace / "reference" / "model.safetensors")
    fixture = {"artifacts": {"model.safetensors": "a" * 64, "dataset.json": "b" * 64, "exploration.json": "c" * 64},
               "manifest_sha256": "d" * 64, "training": {"seed": 1}, "model": describe_model(ReferenceCNN())}
    monkeypatch.setattr(api, "load_fixture", lambda _: (fixture, {"dataset_revision_id": "v", "split_sha256": {}, "samples": []}, {}))
    assets = tmp_path / "dist"
    (assets / "assets").mkdir(parents=True)
    (assets / "index.html").write_text(INDEX, encoding="utf-8")
    (assets / "assets" / "app.js").write_text("export {}", encoding="utf-8")
    (assets / "assets" / "app.css").write_text("body{}", encoding="utf-8")
    creds = api.Credentials("demo")
    with TestClient(api.create_app(workspace, creds, assets=assets), base_url="http://127.0.0.1") as client:
        client.post("/api/v1/session", json={"ticket": creds.ticket})
        yield client


def test_every_kind_of_response_carries_the_policy(client):
    responses = {
        "index": client.get("/"), "asset js": client.get("/assets/app.js"), "asset css": client.get("/assets/app.css"),
        "health": client.get("/api/v1/health"), "api json": client.get("/api/v1/projects/demo/catalog"),
        "api error": client.get("/api/v1/projects/other/catalog"), "unauthenticated": TestClient(client.app, base_url="http://127.0.0.1").get("/api/v1/projects/demo/catalog"),
        "static 404": client.get("/nope.txt"), "foreign host": client.get("/", headers={"Host": "evil.example"}),
        "oversize body": client.post("/api/v1/projects/demo/recipes", content=bytes(200_000)),
    }
    assert responses["index"].status_code == 200 and responses["foreign host"].status_code == 403
    assert responses["unauthenticated"].status_code == 401 and responses["oversize body"].status_code == 413
    for name, response in responses.items():
        assert response.headers.get("Content-Security-Policy") == api.CSP, name


def test_the_policy_is_strict():
    policy = directives(api.CSP)
    assert policy["default-src"] == ["'none'"]
    assert policy["script-src"] == ["'self'"] and policy["style-src"] == ["'self'"]  # no inline code, no unsafe-*, no remote
    assert policy["connect-src"] == ["'self'"]  # only back to the loopback origin that served the page
    assert policy["object-src"] == ["'none'"] and policy["frame-ancestors"] == ["'none'"]
    assert policy["base-uri"] == ["'none'"] and policy["form-action"] == ["'none'"]
    assert policy["font-src"] == ["'self'"] and set(policy["img-src"]) == {"'self'", "data:", "blob:"}
    every = " ".join(api.CSP.split())
    for forbidden in ("unsafe-inline", "unsafe-eval", "*", "http:", "https:", "ws:"):
        assert forbidden not in every


def dist_dir():
    return Path(os.environ.get("LAYERDELTA_DESKTOP_DIST", REPO / "desktop" / "dist"))


def test_built_bundle_is_compatible_with_the_policy():
    """The built Desktop page needs nothing the policy forbids (CI builds it and requires this test)."""
    index = dist_dir() / "index.html"
    if not index.is_file():
        if os.environ.get("LAYERDELTA_REQUIRE_DESKTOP_DIST"):
            pytest.fail(f"LAYERDELTA_REQUIRE_DESKTOP_DIST is set but {index} does not exist")
        pytest.skip("the Desktop bundle is not built (npm run build in desktop/)")
    html = index.read_text(encoding="utf-8")
    scripts = re.findall(r"<script\b([^>]*)>(.*?)</script>", html, re.S | re.I)
    assert scripts and all("src=" in attrs and not body.strip() for attrs, body in scripts), "inline script"
    assert not re.search(r"<style\b", html, re.I) and not re.search(r"\sstyle\s*=", html, re.I), "inline style"
    assert not re.search(r"\son[a-z]+\s*=", html, re.I) and "javascript:" not in html.lower(), "inline handler"
    for url in re.findall(r"(?:src|href)\s*=\s*[\"']([^\"']+)[\"']", html):
        assert not re.match(r"(?:[a-z][a-z0-9+.-]*:)?//", url, re.I) or url.startswith("data:"), f"remote reference {url}"
    for css in dist_dir().rglob("*.css"):
        text = css.read_text(encoding="utf-8")
        assert not re.search(r"@import|url\(\s*[\"']?(?:https?:)?//", text, re.I), css.name
    for js in dist_dir().rglob("*.js"):
        text = js.read_text(encoding="utf-8")
        assert "eval(" not in text and "new Function(" not in text, f"{js.name} needs unsafe-eval"
