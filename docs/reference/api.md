# LayerDelta — API Specification v1

The Desktop app and the headless CLI `layerdelta` use the same FastAPI application at `/api/v1`. For the schema as actually running, see `/openapi.json` on the session's loopback address, or the [request models](../../src/layerdelta/schemas.py) and [routes](../../src/layerdelta/api.py).

### M2 reference profile

- Start `layerdelta --workspace PATH --agent-token-file NEW_FILE` for a local Agent. The first run prepares MNIST under `PATH/mnist` and the fixture under `PATH/reference`. The launcher prints its assigned URL and project ID. The connection file contains the bearer token. Desktop exchanges a one-use fragment ticket at `POST /api/v1/session` for a cookie and CSRF header value.
- Set `device` to `cpu` or `cuda` on previews, runs and final evaluations. Catalog advertises availability. Poll status for terminal errors. Successful comparisons include `runtime.actual_device`.
- Use the SHA-256 from `catalog.checkpoints[].checkpoint_id`. The fixed evaluation profile contains both image conditions, accuracy and confusion matrices. For this release omit `evaluation` to use the complete profile. If supplied it must match the frozen profile in [domain.py](../../src/layerdelta/domain.py). `primary` selects all blurred rows with the demonstration's true digit. `baseline_error` selects the same digit misclassified as the frozen confusion partner by the baseline.
- `POST /previews` takes `sample_id`, `target_id`, zero-based `channel_index` and device. The response contains `run_id`. A successful GET returns `result` with input pixels (0–1), post-ReLU map, kernel and activation shapes, min/max/mean and zero fraction (0–1). Each summary reduces the entire selected channel for one image. The cache key covers parent, input, target, channel and runtime.
- **`weights:read` capability.** Weight values and statistics are visible only to the human session and to an agent token that was granted `weights:read` when the service started (`layerdelta --agent-token-file FILE --agent-weights-read`; default off). Without it the backend removes `result.kernel` from `GET /previews/{id}`, and removes the `before`/`after` kernel statistics from `intervention_audit` in `GET /runs/{id}/report` (the target mapping `parameter_name`, `channel_axis`, `channel_index` stays). Comparison, changes, history, catalog and recipes never contain weight values or statistics. The preview result is an allowlist (`PREVIEW_PUBLIC` in `api.py`). **This closes the direct channels only:** the full activation map plus the source image in a preview are weight-revealing (block1 weights follow from a least-squares fit over about 8 previews, block2 from about 24 more), so a token without `weights:read` is not guaranteed ignorant of the weights. Gating activation values/image behind a capability or capping previews per session is planned (T4c / v0.5 benchmark gate). `report.json` and the preview cache on disk still hold the full statistics, so a benchmark runner must deny the agent filesystem access to the workspace. `GET /agent-session` returns `capabilities`, `["weights:read"]` or `[]`. The check is on the server for every response; a route that is added later must be added to `tests/test_weights_read.py`, which calls every route as an agent token and scans the JSON for kernel values.
- **Checkpoint catalog and import (v0.2).** Every checkpoint the workspace can use is an entry in a SQLite catalog, independent of the reference fixture (which registers itself as origin `fixture`). `GET /checkpoints` (human and agent, paginated) and `GET /checkpoints/{checkpoint_id}` return `checkpoint_id`, `display_name`, `origin` (`imported | fixture | child`), `parent_id`, `recipe_hash`, `format`, `sharded`, `shard_count`, `file_bytes`, `tensor_count`, `element_count`, `adapter_compatible` and `imported_at`; no filesystem paths. `checkpoint_id` is the sha256 of the file (single file) or, for a sharded set, the sha256 of `layerdelta-fileset-v1\n` followed by the canonical JSON of the sorted `[file name, sha256]` list (the index file's name is part of the id, and the domain prefix means it can never equal a single-file id). Shard names that Windows would reinterpret (`:`, device names such as `NUL.safetensors`) are rejected as `index_invalid`. `GET /catalog` still lists only the entries the MNIST adapter can run (`adapter_compatible`), the fixture first.
- Import is human-only (agent tokens get 403) and never loads a model or runs code from the file: it validates with the T1 safetensors reader, copies into the workspace while hashing, re-validates the copy, then publishes it atomically under `checkpoints/`. Importing identical bytes again returns the same `checkpoint_id` (`existing: true`). `POST /checkpoint-imports` takes `{"path": "<absolute file>", "display_name"?}` for a `.safetensors` file or a `*.safetensors.index.json` (sharded: shards are copied; each shard must resolve inside the index directory, else `shard_outside_directory`). `POST /checkpoint-imports/upload?filename=x.safetensors&display_name=` streams the raw request body (requires `Content-Length`, no chunked encoding) into a workspace staging file in chunks, never into memory; uploads are single files. Both return 202 with an import record; poll `GET /checkpoint-imports/{import_id}`: `status` (`queued | running | succeeded | failed`), `stage` (`validating | copying | verifying | publishing`), `stage_index`/`stage_count`, `done`/`total`/`unit` (`files` or `bytes`), `tensor_count`, `checkpoint_id`, `existing`, and on failure `error: {code, message}`. `validating` covers the browser's header/extent/table rows. Caps are configurable (`layerdelta --max-import-bytes`, default 16 GiB per file, 4 times that per sharded set; header 100 MiB). Refusals have stable codes: the T1 reader's (`file_size`, `header_size`, `header_json`, `dtype_unknown`, `extent_*`, `data_size`, `shard_missing`, `index_*`, ...) plus `unsupported_format` (pickle/zip content or `.pt/.pth/.bin/.ckpt/.pkl` names, anything not safetensors), `source_not_found`, `total_size`, `shard_outside_directory`, `file_changed`, `io_error`, and for HTTP `payload_too_large` (413), `length_required` (411), `length_mismatch`, `invalid_path`, `invalid_filename`, `invalid_display_name`. Offline: `layerdelta-import --workspace W path [--name N]`.
- The legacy `POST /checkpoints` (multipart, MNIST-schema checkpoints, 1 MiB cap) behaves as before and now also registers the file in the catalog. A non-MNIST checkpoint can be imported and browsed but is rejected (`dependency_mismatch`) as the `checkpoint_id` of previews, recipes and runs.
- **v2 tensor-space API (T4c).** All routes are under `/checkpoints/{checkpoint_id}/...` because inspection and editing are per checkpoint and the id is the catalog's content-addressed id (the URL text is looked up in the catalog and never used as an identity). `GET describe` (query `name_glob`, `limit`, `cursor`), `POST read-tensor`, `stats`, `diff` (body `other`), `query`, `plan`, `validate`, `apply` (requires `Idempotency-Key`), plus `POST /blobs`. Request and response fields are the library contracts: selectors and ops in [inspect-ops.md](inspect-ops.md) (rev 6) and [edit-program.md](edit-program.md); `plan`/`validate`/`apply` take either a frozen `recipe` (recipe v2) or a `program` plus `hypothesis` that the server freezes against the base table, and `plan`/`apply` return the frozen recipe in `result.recipe`. Every response is the envelope `schema_version, resolved, result, fingerprints, side_effects, truncated, cursor`; pure routes always return `side_effects: []`; `resolved.selector` can be sent back unchanged. Errors are `{error: {code, message, field, request_id}}`: 422 for request errors with a field path, 404 unknown checkpoint, 409 for an unreadable checkpoint file (T1 code), 403 missing capability, 429 `quota_exhausted`.
- **Capabilities.** `model:read` (always granted to agent tokens): `describe`, catalog routes. `weights:read` (`--agent-weights-read`): `read-tensor`, `stats`, `diff`, `query`, `plan`, `validate`, legacy preview activation/image and kernel statistics. `weights:write` (`--agent-weights-write`, needs `--agent-weights-read`): `apply` and `POST /blobs`. The human session has all three. `GET /agent-session` lists the token's `capabilities`. The grant is stored per principal in the catalog's `sessions` table at start, and every apply snapshots the capabilities of its caller in the `edits` table (where the T5 event trace reads them).
- **Cursors** in v2 responses are HMAC-signed (`c2.`): a cursor that was edited, forged or issued by another process is `invalid_cursor` (field `cursor`); one is also bound to its request and checkpoint.
- **apply** validates the whole program first (so an invalid program costs nothing), then charges one run to an agent token (humans are never charged) and writes the child. The child is published into the catalog as origin `child` with `parent_id` and `recipe_hash`. Repeating the same `Idempotency-Key` with the same body returns the same child with `result.replayed: true` and no new charge or side effects, including when the requests are concurrent; the same key with a different body is 409 `idempotency_conflict`. A failed apply releases its charge and its key. The same recipe applied twice (different keys) gives the same child id.
- **Identity probe.** `GET /whoami` is human-only: 200 `{project_id, principal: "human", capabilities}` for a valid human session, 403 for an agent token, 401 without a session. The desktop launcher uses it (instead of `/holdout-audit`) to validate the WebView cookie before it opens the checkpoint file dialog.
- **Content-Security-Policy.** Every response (the Desktop index and assets, API JSON, errors and the early 403/413 rejections) carries `default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; font-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'` (`api.CSP`). The built bundle needs no inline script or style element (React sets style properties through the CSSOM, which `style-src` does not restrict) and no `eval`; `tests/test_csp.py` asserts the header on every kind of response and checks the built `desktop/dist` (CI builds it and requires that test). `connect-src 'self'` is the loopback origin that served the page.
- **Import cancel and stall timeout.** `POST /checkpoint-imports/{import_id}/cancel` (human only) cancels a queued import at once and a running one within a fraction of a second, even if its worker is blocked in a read; a finished import is returned unchanged, and an import already at its atomic publish completes (the cancel is refused, so a cancelled import never leaves a published checkpoint). New terminal status `cancelled` (error code `cancelled`). An import that makes no progress for `ImportLimits.stall_seconds` (default 120, `layerdelta --import-stall-seconds`), for example on a hung network path, becomes `failed` with `import_stalled`, and the queue continues; the abandoned worker stops at its next check and cannot publish. Statuses are now `queued | running | succeeded | failed | cancelled`. Known limit: the staging file of an upload whose worker is stalled inside a Windows file operation may stay on disk until the next start, when the staging sweep removes it.
- **Idempotency keys in the job store.** The legacy job table now stores `sha256(key)[:32]` (the same digest as the trace and the apply ledger), looked up first; a row written by an older version with the raw key is still found by a second, raw lookup, so in-flight submissions keep working. A key that holds a secret therefore no longer reaches the database anywhere.
- **Session trace (T5).** Every request an authenticated principal makes (every v2 route, every legacy agent route, 403s and 404s included; requests that fail authentication have no session and are not recorded) leaves one event in the catalog's `events` table (SQLite, so it survives restarts). An event holds ids only: `session` (the principal: `agent-<hex>` per server process, or `human`), a per-session `seq`, `route` (the route template, e.g. `POST /checkpoints/{checkpoint_id}/apply`), `tool` (`apply`, `read_tensor`, ...), `request_id` (also returned in the `X-Request-Id` header and in error bodies), `status`, `error_code`, `checkpoint_id`, `other_checkpoint_id`, `selection_id`, `recipe_hash`, `child_id`, `run_id`, a digest of the `Idempotency-Key` (sha256, first 32 hex: it matches a retry to its original and never reveals a key that holds a secret; the apply ledger keeps the same digest), `charged` (set for every charged submission: apply and legacy `POST /runs`, so `summary.charged_runs` equals `/agent-session` `charged`), the principal's `capabilities` snapshot, and for plan/apply the `selections` ids and op count. Routes can only add whitelisted, pattern-checked id fields, so tensor values, tokens, auth headers, request bodies, images and prompts are never stored, and user-supplied URL text that is not an id is dropped.
- **Cost and durability.** An event is one small write made before the response, with `synchronous=NORMAL` on that connection (the catalog is in WAL mode, so the file cannot be corrupted, but the last few events can be lost on a power cut). `GET /trace` pages in SQL from indexes (`after` + `limit`, plus indexed replay and link lookups and aggregate counts for the summary), so its cost follows the page and not the length of the log. The legacy job table keeps the same digest.
- **`GET /trace`** (`model:read`; query `session`, `after`, `limit` 1..500) returns the v2 envelope with `result.events`, `result.summary` (`events`, `applies`, `replays`, `charged_runs`) and `result.next_after` (pass it as `after`; `truncated` is true when more follow). An agent sees only its own session (naming another is 403); the human session sees every session, or one with `session=`. For every successful `apply` event, `links` gives `plan_seqs` (earlier plan events with the same `recipe_hash`), `selection_seqs` (earlier reads and stats of the same checkpoint whose `selection_id` is one of the apply's `selections`) and `child_id`, so a child can be traced back to the reads that motivated it. A repeat of an idempotent request under the same `Idempotency-Key` (an apply replay, a retried run submission) is marked `replayed`/`replay_of` and points at the original event; it is not counted as another apply or another charge (`summary.applies`, `summary.charged_runs`). A new server process gets a new agent session id; the human session keeps its sequence.
- **apply and existing children.** If the child's bytes are already in the catalog (the same recipe under a new key, a different recipe with identical bytes, a no-op under `allow_noop`), nothing is written: `side_effects` has `checkpoint_existing` instead of `checkpoint_created` (and `quota_charged` for an agent), `result.existing` is true, and `result.checkpoint` is the catalog entry only if its lineage is this edit's, otherwise just `{checkpoint_id}`. A replay or key conflict is answered before the program is re-planned, so a replay still works if a blob or source has since gone.
- **Capability pairing.** `weights:write` requires `weights:read` (the credentials refuse it otherwise, and `apply`/`blobs` check both), because the plan data in an apply result (hashes, changed counts, `no_op`) would otherwise probe weights. Cursors are signed with a dedicated random per-process key.
- **Known limits.** The blob store's 1 GiB total cap is checked before each upload and is racy under concurrent uploads (it can be exceeded by the size of the concurrent blobs). A concurrent twin of an in-flight apply waits (sleeps up to 60 s) inside the worker thread pool, so a burst of same-key retries can occupy pool threads.
- **Blobs**: `POST /blobs` takes the raw body (Content-Length required, a multiple of 4, at most 64 MiB, workspace total 1 GiB) of little-endian float32 values, stores it content-addressed (`sha256:<hex>`, idempotent) and returns `blob_id`; `set`/`add` ops name it in `args.blob` and the server checks hash and size against the selection.
- **Legacy preview gating (R2).** `GET /previews/{id}` for an agent token without `weights:read` no longer includes `activation`, `image` or `kernel` (a full activation map plus the source image reveals the weights algebraically). `layerdelta.agent_demo` therefore needs a token started with `--agent-weights-read` and says so when it is missing.
- POST multipart checkpoints with field `file`. Download a confirmed exported child from the scoped `result.download_url`. Reports contain the checkpoint digest and immutable comparison fingerprint. Job cancellation returns its current state. The human-only `GET /holdout-audit` lists confirmations.

## Transport and identity

- The base URL is the loopback endpoint that the launcher announces, followed by `/api/v1`. The client gets the endpoint from its local configuration once the backend is ready.
- JSON is UTF-8, field names use snake_case, timestamps are UTC RFC 3339, IDs are opaque strings, and SHA-256 values are lowercase hex with 64 characters.
- The Desktop uses a same-origin human session with a CSRF token for mutations. An agent uses `Authorization: Bearer <token>`, bound to a project/session. The backend checks permissions before it resolves an object and before it reads any artifact, every time.
- Pagination uses an opaque cursor and `limit`. The proposed default is 50, with a maximum of 200. A cursor is bound to the filters and to the ordering of the collection, and the ID is the tiebreaker when timestamps are equal.
- A mutation that creates an asynchronous job must send an `Idempotency-Key`. Store the key per principal + route + project, together with the hash of the normalized payload. When the outcome of a submit is unknown, send the same key and the same payload again. The same key with a changed payload returns 409.

## Endpoint catalog

Every path in the table, except health, uses the prefix `/projects/{project_id}`.

| Method and path | Users | Success result |
|---|---|---|
| GET /health | Launcher | 200 when the backend is ready. Returns no workspace data. |
| GET /catalog | Human, Agent | Project, checkpoints, architecture/adapter revisions, dataset revisions and the target catalog, subject to permissions |
| GET /samples | Human, Agent | Paginated list of exploration samples, conditions and labels |
| POST /previews | Human, Agent | 202, preview job ID for the given sample/target/checkpoint |
| GET /previews/{id} | Human, Agent | Status, and a size-limited activation map when complete. `result.kernel` (weight statistics) is returned only to the human or to an agent with `weights:read`. |
| POST /checkpoints | Human | 201 for a validated multipart tensor-only checkpoint (legacy, MNIST schema; also added to the catalog) |
| GET /checkpoints/{id}/describe | Human, Agent (`model:read`) | v2 envelope: tensor table, no tensor data |
| POST /checkpoints/{id}/read-tensor, stats, diff, query | Human, Agent (`weights:read`) | v2 envelope: values, statistics, comparison, selection |
| POST /checkpoints/{id}/plan, validate | Human, Agent (`weights:read`) | Dry run of an edit program, no writes |
| POST /checkpoints/{id}/apply | Human, Agent (`weights:write`) | Child checkpoint registered in the catalog; idempotent, charged once |
| POST /blobs | Human, Agent (`weights:write`) | Content-addressed float32 blob for set/add |
| GET /checkpoints, GET /checkpoints/{id} | Human, Agent | Checkpoint catalog: id, display_name, origin, parent_id, recipe_hash, sizes, tensor/element counts, imported_at (no paths); `GET /checkpoints/{id}` also returns `children` (ids of the checkpoints whose parent it is, oldest first) |
| POST /checkpoint-imports | Human | 202, import a safetensors file or `*.safetensors.index.json` from an absolute path |
| POST /checkpoint-imports/upload | Human | 202, streaming upload of a safetensors file (Content-Length required) into staging on disk |
| GET /checkpoint-imports/{id} | Human | Stage, done/total, error code and message on failure, checkpoint_id on success |
| POST /recipes | Human, Agent | 201, immutable recipe revision with hash and dependencies |
| POST /recipes/import | Human, Agent | 201, import of a canonical recipe after schema checks and dependency resolution |
| GET /recipes/{id} | Human, Agent | Recipe readable within scope |
| POST /runs | Human, Agent | 202, baseline or intervention run on exploration |
| GET /runs | Human, Agent | Paginated run history that the actor can read |
| GET /runs/{id} | Human, Agent | State, progress, provenance, and the error when it failed |
| POST /runs/{id}/cancel | Human, Agent | 202 when the cancel request is accepted, or 200 with the existing status when the run is already terminal |
| GET /runs/{id}/comparison | Human, Agent | 200 for a published intervention |
| GET /runs/{id}/changes | Human, Agent | Paginated prediction changes from the same comparison |
| GET /runs/{id}/report | Human, Agent | JSON report for a run the actor may read. `intervention_audit.before/after` (kernel statistics) requires `weights:read`. |
| GET /trace | Human (all sessions), Agent (own session) | v2 envelope: ID-only event timeline with read, plan, apply and child links |
| GET /whoami | Human | 200 identity probe (403 for an agent token): project, principal, capabilities |
| POST /checkpoint-imports/{id}/cancel | Human | Cancel a queued or running import; terminal imports are returned unchanged |
| GET /agent-session | Agent | Quota limit, charged, remaining, run limits, and `capabilities` (`weights:read` when the operator enabled it) |
| POST /final-evaluations | Human | 202 after the frozen recipe hash is confirmed and the audit is recorded |
| POST /checkpoint-exports | Human | 202 for a successful and confirmed weight run |
| GET /checkpoint-exports/{id} | Human | Status, lineage and download URL once published |

POST previews takes the checkpoint ID, dataset revision, exploration sample ID, and the target ID and channel index from the catalog. GET catalog returns the target mode, channel count, zero-based indexes, the hook position or parameter mapping, the allowed operations and the scale bounds. The backend sets the size and time limits for previews. They are separate from the quota for experiment runs.

## Recipe, submit and comparison

```json
{
  "hypothesis": "Scaling this channel may reduce 8-to-3 errors on blurred samples.",
  "checkpoint_id": "ck_demo",
  "dataset_revision_id": "ds_demo_v1",
  "evaluation": {
    "split": "exploration",
    "condition_ids": ["normal", "blurred"],
    "metric_ids": ["accuracy", "confusion_matrix"],
    "slice_ids": ["overall", "normal", "blurred", "baseline_8_as_3_blurred"]
  },
  "intervention": {
    "mode": "activation",
    "target_id": "features.block2.relu.output",
    "channel_index": 17,
    "operation": "scale",
    "scale": 0.5
  }
}
```

The IDs and targets in this example are placeholders. They must be resolved from the actual catalog. A request has one intervention per recipe. Use `activation` or `weight` for the mode, and either a `scale` that is finite and inside the policy range, or `zero`, which omits the scale field. A weight target must lie in a Conv kernel slice that the adapter declares. The server returns the persisted recipe, with the schema version, recipe ID and hash, hypothesis, creator type, model/adapter revision, checkpoint SHA-256, dataset hash, split manifest hash, preprocessing revision and parameters, the seed where relevant, the evaluator revision, and the intervention mapping. Fingerprints and mappings come from the backend. On import, the hash is verified from the canonical payload and dependencies are resolved, without relying on absolute paths or the original workspace IDs.

- **Baseline** sends `{"kind":"baseline","checkpoint_id":"ck_demo","dataset_revision_id":"ds_demo_v1","evaluation":{...}}`, using the same evaluation schema. GET report returns the baseline metrics and fingerprint.
- **Intervention** sends `{"kind":"intervention","recipe_id":"rec_123"}`. The backend computes or loads a baseline whose evaluation fingerprint matches.
- **Retry** sends a new submit with `retry_of_run_id` and a new key. The backend confirms that this is the same request as the failed or cancelled run, and that the actor is authorized. A successful run is then replayed through a normal submit.
- **Final evaluation** sends `recipe_id`, `recipe_hash` and `confirmed:true` for the frozen recipe to be used on the holdout. Creating the holdout recipe and all of its results use the human scope. The audit links the recipe hash, the split manifest hash and the time.
- **Checkpoint export** sends `source_run_id` and `confirmed:true`. The export record holds the parent/child SHA-256, the recipe hash and the source run ID. The service uses a file name that it manages, and sends a scoped download URL when the artifact is complete.

A `Run` returns `run_id`, `kind`, `status`, `recipe_id` when present, `progress` that gives the phase and the completed/total batches when known, timestamps, and the error when it failed. The states are `queued → running → succeeded | failed | cancelled`, and a queued run can be cancelled directly. If the cancel commits before the success commit, the run ends as cancelled. Poll every 0.5–1 seconds at first, then increase the interval up to the configured value.

A `Comparison` returns `run_id`, `baseline_run_id`, `recipe_hash`, `evaluation_fingerprint`, the runtime metadata and `slices[]`. Each slice has an ID, N, before/after accuracy (0–1), `delta_pp`, confusion matrices (row = truth, column = prediction, order 0–9) for the selected metrics, and the counts of fixed/regressed/changed_wrong. An empty group returns N = 0 and accuracy/delta as null, with the reason. `changes` returns the sample ID, condition, truth, before/after labels and category. Every match uses the same manifest and sample IDs.

## Error contract and quota

```json
{
  "error": {
    "code": "invalid_target",
    "message": "Channel index is outside the target catalog range.",
    "field": "intervention.channel_index",
    "request_id": "req_example"
  }
}
```

| Status | Codes and expected response |
|---|---|
| 400 / 422 | `invalid_json`, `invalid_target`, `invalid_scale`, `unsupported_schema`. Fix the payload using the field error. |
| 401 / 403 | `unauthenticated`, `forbidden`. Stop the action and use only the scope that was granted. |
| 404 | `not_found` for an object that does not exist or is outside the project/session that is visible |
| 409 | `dependency_mismatch`, `idempotency_conflict`, `run_not_ready`, `artifact_unavailable`. Check the context before going further. |
| 413 | `payload_too_large`. The file or request exceeds the declared cap. |
| 429 | `quota_exhausted` or `queue_full`. Separates an exhausted budget from a full queue, which can be retried. |
| 500 / 503 | `internal_error`, `service_unavailable`. Check the status, and send the same key again if the outcome of the submit is still unknown. |

The agent quota is charged when a run is accepted, including runs that later fail or are cancelled. A baseline embedded inside an intervention counts as one submission, and an explicit baseline is another submission. The proposed value is 10 submissions per session, and the actual value is announced through the API. Accepting a job and deducting quota happen in a single transaction. Worker failures such as timeout or interruption are reported as terminal run errors through the status endpoint. Results can be read only after they have been published successfully.

## Contract checks

- The Desktop and the Agent send the same recipe on the same runtime and manifest, and get the same sample changes, counts and metrics.
- A lost response followed by a replay with the same key returns the same run ID, and the budget is charged once.
- Cancel and publish produce one terminal state. A run after a failure or cancel can be compared with the original baseline.
- An agent cannot read the holdout through the catalog, preview, history, comparison, report or artifact URL, and cannot create a checkpoint export.
- A replay in a new workspace resolves artifacts by hash, and a weight child that is loaded back gives the same results as the source run, on the reference runtime.
