"""Agent-facing content (v0.4-S2, acceptance T3): descriptions, examples, defaults and the error catalog."""

import os
from pathlib import Path

import pytest

import layerdelta
from layerdelta import agent_bridge, error_catalog, registry

SRC = Path(layerdelta.__file__).parent
ERRORS_MD = SRC.parent.parent / "docs" / "reference" / "errors.md"


def test_every_registry_tool_has_content_and_nothing_else_does():
    content = registry.agent_content()["tools"]
    assert set(content) == {r.tool for r in registry.ROUTES if r.tool}
    for name, entry in content.items():
        assert 0 < len(entry["description"]) <= 600, name
        assert registry.tools_by_name()[name]["description"] == entry["description"]


def test_every_example_validates_against_its_tool_schema_and_every_default_is_a_property():
    jsonschema = pytest.importorskip("jsonschema")
    for tool in registry.load()["tools"]:
        schema = {k: v for k, v in tool["inputSchema"].items() if k != "examples"}
        for example in tool["inputSchema"].get("examples", []):
            jsonschema.validate(example, schema)  # raises with the tool and the example on a mismatch
        for name, prop in tool["inputSchema"]["properties"].items():
            if "default" in prop and prop["default"] is not None:
                jsonschema.validate(prop["default"], {**{k: v for k, v in prop.items() if k != "default"}, "$defs": tool["inputSchema"].get("$defs", {})})


def test_every_code_the_code_base_can_raise_has_a_catalog_entry():
    known = error_catalog.catalog()
    missing = sorted(set(error_catalog.raise_sites(SRC)) - set(known))
    assert missing == [], f"add these codes to agent_content.json errors: {missing}"
    for code, entry in known.items():
        assert isinstance(entry["retryable"], bool) and entry["hint"].strip() and len(entry["hint"]) <= 400, code


def test_the_agent_must_be_told_to_stop_on_the_refusals_that_mean_stop():
    errors = error_catalog.catalog()
    for code in ("forbidden", "budget_exhausted", "quota_exhausted", "session_revoked"):
        assert errors[code]["retryable"] is False and "stop" in errors[code]["hint"].lower(), code


def test_errors_md_is_generated_and_current():
    text = error_catalog.render(error_catalog.raise_sites(SRC))
    if os.environ.get("LAYERDELTA_WRITE_ERRORS"):
        ERRORS_MD.write_text(text, encoding="utf-8")
    assert ERRORS_MD.read_text(encoding="utf-8").replace("\r\n", "\n") == text, "errors.md is stale: LAYERDELTA_WRITE_ERRORS=1"
    assert "| `group_not_closed` | 422 |" in text and "| `budget_exhausted` | 429 |" in text and "| `session_revoked` | 401 |" in text


def test_a_bridge_error_carries_the_catalog_hint_and_unknown_codes_stay_null():
    known = agent_bridge._error("group_not_closed", "m", 422, "program[0].select.group")["error"]
    assert known["hint"] == error_catalog.catalog()["group_not_closed"]["hint"] and known["retryable"] is False
    unknown = agent_bridge._error("not_in_the_catalog", "m", 400)["error"]
    assert unknown["hint"] is None and unknown["retryable"] is None
    own = agent_bridge._error("result_too_large", "m", None, "limit", hint="lower limit", retryable=False)["error"]
    assert own["hint"] == "lower limit"  # a hint the bridge sets itself wins over the catalog
