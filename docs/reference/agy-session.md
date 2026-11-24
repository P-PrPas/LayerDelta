# The Antigravity session driver and its audit (v0.5-E7b, the part both isolation options need)

Code: `tools/bench/agy_session.py`. Tests: `tests/test_agy_session.py` (a fake `agy`, no host). Evidence for the host's behaviour: [the E7s spike](../research/v0.5-e7s-antigravity-spike.md). Design: [v0.5 §5.2](../architecture/v0.5-evaluators-benchmark.md).

`run_session(arm, session_dir, task, results=..., mcp=..., protected=[...], limits=Limits(...))` runs one benchmark session and returns its record (also `results/session.json`; the raw stream is `results/transcript.ndjson`). One session at a time.

- **Launch.** `agy -p <task> --model gemini-4-hydrogen-high --effort high --output-format stream-json --print-timeout <wall>s` in the session directory. `--dangerously-skip-permissions` is passed only with `skip_permissions=True`: that is the owner's decision and is off by default.
- **Budgets** (values pre-registered after the dry run), enforced from the stream while the session runs; the whole process tree is killed on a limit and the record says `stopped`:
  - `wall_seconds` (also given to the host as `--print-timeout`);
  - `steps`: the DONE tool steps of the stream;
  - `completion_tokens`: output plus thinking tokens of the agent responses.
  Total tokens (as the host counts them, prompt and cache reads included), input and cache-read tokens are reported, not limited.
- **The conversation store** (`~/.gemini/antigravity-cli/conversations`) is shared by every session of the OS account, so a later session could read the other arm's transcripts. The driver moves it aside for the session (it is empty while the agent runs), copies the session's own conversation to `results/conversations`, and puts the old store back, also when the run fails. A backup left by a crashed run (`conversations.bench-backup`) stops the next session (`StoreBusy`) rather than being overwritten.
- **The bridge.** MCP configuration is global to the account, so arm L's session adds `layerdelta` (`agy mcp add layerdelta -- <command>`) before the run and removes it after; arm R refuses to start while `layerdelta` is configured (`McpConflict`), and so does arm L (someone else's session may be running).
- **The audit** reads every tool call of the transcript. A `violation` is what the pre-registered rule counts; a `review` is reported only.
  - `web_tool`: any web or browser tool (`search_web`, `read_url_content`, `browser_*`, ...).
  - `read_outside` / `write_outside`: a file tool whose path is outside the session directory (relative paths are resolved against it).
  - `protected_path`: a path under the agent's own store (`~/.gemini`), or under a root passed as `protected` (the LayerDelta workspace, the MNIST archives, the holdout), in a file tool or in a command.
  - `download` and `training`: the signatures of `mnist_bench.audit_text` (URLs, `curl`/`wget`/`Invoke-WebRequest`, `pip install`, dataset loaders; optimisers, `backward`, `fit`, ...) in a command or in code the agent wrote.
  - `path_outside` (review): an absolute path or `..` in a command that is not under the session directory and not protected (a program elsewhere, for example).
- **What it does not do.** Prevent. Everything is found after the fact. Prevention (a dedicated low-privilege OS account for the sessions and an outbound firewall rule) waits for the owner's decision (Q5) and is not here. The audit is needed under both options, and the audit must also be read for the real host once, because the tool and parameter names are those seen in the spike (`run_command` with `CommandLine`, `read_file` with `Path`); a tool the spike did not exercise (`view_file`, `write_to_file`, `grep_search`, ...) is matched by the parameter-name pattern (`path|file|dir|folder|target|cwd`) and should be checked in the dry run.
