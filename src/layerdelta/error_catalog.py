"""The error catalog (v0.4-S2, acceptance T3): every error code the code base can raise, with its HTTP status and the hint and
`retryable` flag an agent reads (from agent_content.json). Stdlib only.

`raise_sites` scans the source for the places a code is raised (`InspectError`, `StoreError`, `fail`, `GraphBuildError`,
`TensorFileError`, `ImportFailure`, `ClientError`, the bridge's own `_error`, and `{"code": ...}` bodies), so a code added to the
code base without a catalog entry fails the coverage test. `render` produces docs/reference/errors.md.
"""

import json
import re
from pathlib import Path

CONTENT = Path(__file__).with_name("agent_content.json")
_CALL = re.compile(r"\b(InspectError|StoreError|GraphBuildError|TensorFileError|ImportFailure|fail|ClientError|_error|_bad|refuse|bad)\(\s*[\"']([a-z][a-z0-9_]{2,48})[\"']")
_BODY = re.compile(r"[\"']code[\"']\s*:\s*[\"']([a-z][a-z0-9_]{2,48})[\"']")
_FIXED = {"InspectError": 422, "TensorFileError": 409}  # what the app's exception handlers answer
_THIRD = ("fail", "StoreError", "ClientError")          # (code, message, status)


def catalog() -> dict:
    return json.loads(CONTENT.read_text(encoding="utf-8"))["errors"]


def _args(text: str, start: int) -> list[str]:
    """The top-level arguments of the call whose opening parenthesis is at text[start]."""
    depth, quote, out, current = 0, None, [], []
    i = start
    while i < len(text):
        c = text[i]
        if quote:
            current.append(c)
            if c == "\\":
                current.append(text[i + 1])
                i += 1
            elif c == quote:
                quote = None
        elif c in "\"'":
            quote = c
            current.append(c)
        elif c in "([{":
            depth += 1
            if depth > 1:
                current.append(c)
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                out.append("".join(current).strip())
                return out
            current.append(c)
        elif c == "," and depth == 1:
            out.append("".join(current).strip())
            current = []
        else:
            current.append(c)
        i += 1
    return out


def raise_sites(root: Path) -> dict[str, set]:
    """code -> the HTTP statuses it is raised with (empty when it is not an HTTP response of its own)."""
    found: dict[str, set] = {}
    for path in sorted(Path(root).glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in _CALL.finditer(text):
            name, code = match.groups()
            statuses = found.setdefault(code, set())
            if name in _FIXED:
                statuses.add(_FIXED[name])
            elif name in _THIRD:
                args = _args(text, match.end(0) - len(match.group(0)) + match.group(0).index("("))
                if len(args) >= 3 and re.fullmatch(r"\d{3}", args[2]):
                    statuses.add(int(args[2]))
        for match in _BODY.finditer(text):  # an error body written out by an exception handler: its status follows in the same call
            statuses = found.setdefault(match.group(1), set())
            after = re.search(r"status_code\s*=\s*(\d{3})|\}\s*,\s*(\d{3})\)", text[match.end():match.end() + 300])
            if after:
                statuses.add(int(after.group(1) or after.group(2)))
    return found


def render(sites: dict[str, set]) -> str:
    known = catalog()
    lines = ["# Error codes (generated)", "",
             "Generated from the error catalog (`agent_content.json`) and the raise sites in `src/layerdelta`: do not edit. "
             "Regenerate with `LAYERDELTA_WRITE_ERRORS=1 pytest tests/test_agent_content.py`. "
             "A backend error is `{error: {code, message, field}}`; the MCP bridge adds `hint`, `retryable` and `status` "
             "from this table ([mcp.md](mcp.md)). A status of `-` means the code is not an HTTP response of its own "
             "(an attach or import job result, a library error, or a status chosen at run time).", "",
             "| Code | Status | Retryable | Hint |", "|---|---|---|---|"]
    for code in sorted(set(sites) | set(known)):
        entry = known.get(code, {})
        statuses = ", ".join(str(s) for s in sorted(sites.get(code, ()))) or "-"
        hint = str(entry.get("hint", "")).replace("|", "\\|")
        lines.append(f"| `{code}` | {statuses} | {'yes' if entry.get('retryable') else 'no'} | {hint} |")
    return "\n".join(lines) + "\n"
