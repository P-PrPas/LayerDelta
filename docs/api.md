# LayerDelta — API Specification v1

Desktop และ headless CLI `layerdelta` ใช้ FastAPI application เดียวกันที่ `/api/v1`. ดู schema ที่รันจริงจาก `/openapi.json` บน loopback ของ session หรือ [request models](../src/layerdelta/schemas.py) และ [routes](../src/layerdelta/api.py)

### M2 reference profile

- Start `layerdelta --workspace PATH --agent-token-file NEW_FILE` for a local Agent. The first run prepares MNIST under `PATH/mnist` and the fixture under `PATH/reference`. The launcher prints its assigned URL and project ID. The connection file contains the bearer token. Desktop exchanges a one-use fragment ticket at `POST /api/v1/session` for a cookie and CSRF header value.
- Set `device` to `cpu` or `cuda` on previews, runs and final evaluations. Catalog advertises availability. Poll status for terminal errors. Successful comparisons include `runtime.actual_device`.
- Use the SHA-256 from `catalog.checkpoints[].checkpoint_id`. The fixed evaluation profile contains both image conditions, accuracy and confusion matrices. For this release omit `evaluation` to use the complete profile. If supplied it must match the frozen profile in [domain.py](../src/layerdelta/domain.py). `primary` selects all blurred rows with the demonstration's true digit. `baseline_error` selects the same digit misclassified as the frozen confusion partner by the baseline.
- `POST /previews` takes `sample_id`, `target_id`, zero-based `channel_index` and device. The response contains `run_id`. A successful GET returns `result` with input pixels (0–1), post-ReLU map, kernel and activation shapes, min/max/mean and zero fraction (0–1). Each summary reduces the entire selected channel for one image. The cache key covers parent, input, target, channel and runtime.
- **`weights:read` capability.** Weight values and statistics are visible only to the human session and to an agent token that was granted `weights:read` when the service started (`layerdelta --agent-token-file FILE --agent-weights-read`; default off). Without it the backend removes `result.kernel` from `GET /previews/{id}`, and removes the `before`/`after` kernel statistics from `intervention_audit` in `GET /runs/{id}/report` (the target mapping `parameter_name`, `channel_axis`, `channel_index` stays). Comparison, changes, history, catalog and recipes never contain weight values or statistics. The preview result is an allowlist (`PREVIEW_PUBLIC` in `api.py`). **This closes the direct channels only:** the full activation map plus the source image in a preview are weight-revealing (block1 weights follow from a least-squares fit over about 8 previews, block2 from about 24 more), so a token without `weights:read` is not guaranteed ignorant of the weights. Gating activation values/image behind a capability or capping previews per session is planned (T4c / v0.5 benchmark gate). `report.json` and the preview cache on disk still hold the full statistics, so a benchmark runner must deny the agent filesystem access to the workspace. `GET /agent-session` returns `capabilities`, `["weights:read"]` or `[]`. The check is on the server for every response; a route that is added later must be added to `tests/test_weights_read.py`, which calls every route as an agent token and scans the JSON for kernel values.
- **Checkpoint catalog and import (v0.2).** Every checkpoint the workspace can use is an entry in a SQLite catalog, independent of the reference fixture (which registers itself as origin `fixture`). `GET /checkpoints` (human and agent, paginated) and `GET /checkpoints/{checkpoint_id}` return `checkpoint_id`, `display_name`, `origin` (`imported | fixture | child`), `parent_id`, `recipe_hash`, `format`, `sharded`, `shard_count`, `file_bytes`, `tensor_count`, `element_count`, `adapter_compatible` and `imported_at`; no filesystem paths. `checkpoint_id` is the sha256 of the file (single file) or, for a sharded set, the sha256 of `layerdelta-fileset-v1\n` followed by the canonical JSON of the sorted `[file name, sha256]` list (the index file's name is part of the id, and the domain prefix means it can never equal a single-file id). Shard names that Windows would reinterpret (`:`, device names such as `NUL.safetensors`) are rejected as `index_invalid`. `GET /catalog` still lists only the entries the MNIST adapter can run (`adapter_compatible`), the fixture first.
- Import is human-only (agent tokens get 403) and never loads a model or runs code from the file: it validates with the T1 safetensors reader, copies into the workspace while hashing, re-validates the copy, then publishes it atomically under `checkpoints/`. Importing identical bytes again returns the same `checkpoint_id` (`existing: true`). `POST /checkpoint-imports` takes `{"path": "<absolute file>", "display_name"?}` for a `.safetensors` file or a `*.safetensors.index.json` (sharded: shards are copied; each shard must resolve inside the index directory, else `shard_outside_directory`). `POST /checkpoint-imports/upload?filename=x.safetensors&display_name=` streams the raw request body (requires `Content-Length`, no chunked encoding) into a workspace staging file in chunks, never into memory; uploads are single files. Both return 202 with an import record; poll `GET /checkpoint-imports/{import_id}`: `status` (`queued | running | succeeded | failed`), `stage` (`validating | copying | verifying | publishing`), `stage_index`/`stage_count`, `done`/`total`/`unit` (`files` or `bytes`), `tensor_count`, `checkpoint_id`, `existing`, and on failure `error: {code, message}`. `validating` covers the browser's header/extent/table rows. Caps are configurable (`layerdelta --max-import-bytes`, default 16 GiB per file, 4 times that per sharded set; header 100 MiB). Refusals have stable codes: the T1 reader's (`file_size`, `header_size`, `header_json`, `dtype_unknown`, `extent_*`, `data_size`, `shard_missing`, `index_*`, ...) plus `unsupported_format` (pickle/zip content or `.pt/.pth/.bin/.ckpt/.pkl` names, anything not safetensors), `source_not_found`, `total_size`, `shard_outside_directory`, `file_changed`, `io_error`, and for HTTP `payload_too_large` (413), `length_required` (411), `length_mismatch`, `invalid_path`, `invalid_filename`, `invalid_display_name`. Offline: `layerdelta-import --workspace W path [--name N]`.
- The legacy `POST /checkpoints` (multipart, MNIST-schema checkpoints, 1 MiB cap) behaves as before and now also registers the file in the catalog. A non-MNIST checkpoint can be imported and browsed but is rejected (`dependency_mismatch`) as the `checkpoint_id` of previews, recipes and runs.
- POST multipart checkpoints with field `file`. Download a confirmed exported child from the scoped `result.download_url`. Reports contain the checkpoint digest and immutable comparison fingerprint. Job cancellation returns its current state. The human-only `GET /holdout-audit` lists confirmations.

## Transport และ identity

- Base URL คือ endpoint loopback ที่ launcher ประกาศ ต่อด้วย `/api/v1`. Client รับ endpoint จาก local configuration หลัง backend พร้อมใช้งาน
- JSON เป็น UTF-8, field names แบบ snake_case, timestamps เป็น UTC RFC 3339, IDs เป็น opaque strings และ SHA-256 ใช้ lowercase hex 64 ตัว
- Desktop ใช้ same-origin human session พร้อม CSRF token สำหรับ mutation. Agent ใช้ `Authorization: Bearer <token>` ที่ผูก project/session. Backend ตรวจสิทธิ์ก่อน resolve object และก่อนอ่าน artifact ทุกครั้ง
- Pagination ใช้ opaque cursor และ `limit` ค่าเสนอ 50 สูงสุด 200. Cursor ผูกกับ filters และลำดับของ collection ใช้ ID เป็นตัวตัดสินเมื่อเวลาเท่ากัน
- Mutation ที่สร้าง asynchronous job ต้องส่ง `Idempotency-Key`. เก็บ key ต่อ principal + route + project และ hash ของ payload ที่ผ่าน normalization. ส่ง key เดิมและ payload เดิมเมื่อไม่ทราบผลของการ submit. Key เดิมกับ payload ที่เปลี่ยนคืน 409

## Endpoint catalog

ทุก path ในตารางยกเว้น health ใช้ prefix `/projects/{project_id}`.

| Method และ path | ผู้ใช้ | ผลสำเร็จ |
|---|---|---|
| GET /health | Launcher | 200 เมื่อ backend พร้อม ไม่คืนข้อมูล workspace |
| GET /catalog | Human, Agent | Project, checkpoints, architecture/adapter revisions, dataset revisions และ target catalog ตามสิทธิ์ |
| GET /samples | Human, Agent | รายการ exploration samples, condition และ labels แบบแบ่งหน้า |
| POST /previews | Human, Agent | 202, preview job ID สำหรับ sample/target/checkpoint ที่ระบุ |
| GET /previews/{id} | Human, Agent | สถานะและ activation map แบบจำกัดขนาดเมื่อเสร็จ. `result.kernel` (weight statistics) ส่งให้ Human หรือ Agent ที่มี `weights:read` เท่านั้น |
| POST /checkpoints | Human | 201 สำหรับ multipart tensor-only checkpoint ที่ validate แล้ว (legacy, MNIST schema; ลง catalog ด้วย) |
| GET /checkpoints, GET /checkpoints/{id} | Human, Agent | catalog ของ checkpoint: id, display_name, origin, parent_id, ขนาด, จำนวน tensor/element, imported_at (ไม่มี path) |
| POST /checkpoint-imports | Human | 202, import safetensors หรือ `*.safetensors.index.json` จาก absolute path |
| POST /checkpoint-imports/upload | Human | 202, upload ไฟล์ safetensors แบบ streaming (ต้องมี Content-Length) ลง staging บนดิสก์ |
| GET /checkpoint-imports/{id} | Human | stage, done/total, error code+message เมื่อล้มเหลว, checkpoint_id เมื่อสำเร็จ |
| POST /recipes | Human, Agent | 201, immutable recipe revision พร้อม hash และ dependencies |
| POST /recipes/import | Human, Agent | 201, import canonical recipe หลังตรวจ schema และ resolve dependencies |
| GET /recipes/{id} | Human, Agent | Recipe ที่อ่านได้ตาม scope |
| POST /runs | Human, Agent | 202, baseline หรือ intervention run บน exploration |
| GET /runs | Human, Agent | ประวัติ runs ที่ actor อ่านได้ แบบแบ่งหน้า |
| GET /runs/{id} | Human, Agent | State, progress, provenance และ error เมื่อ fail |
| POST /runs/{id}/cancel | Human, Agent | 202 เมื่อรับคำขอยกเลิก หรือ 200 พร้อมสถานะเดิมเมื่อ terminal |
| GET /runs/{id}/comparison | Human, Agent | 200 สำหรับ intervention ที่ publish แล้ว |
| GET /runs/{id}/changes | Human, Agent | Prediction changes แบบแบ่งหน้าจาก comparison เดียวกัน |
| GET /runs/{id}/report | Human, Agent | JSON report ของ run ที่ actor มีสิทธิ์อ่าน. `intervention_audit.before/after` (kernel statistics) ต้องมี `weights:read` |
| GET /agent-session | Agent | Quota limit, charged, remaining, ข้อจำกัดรัน และ `capabilities` (`weights:read` เมื่อ operator เปิดไว้) |
| POST /final-evaluations | Human | 202 หลังยืนยัน frozen recipe hash และบันทึก audit |
| POST /checkpoint-exports | Human | 202 สำหรับ weight run ที่สำเร็จและได้รับการยืนยัน |
| GET /checkpoint-exports/{id} | Human | สถานะ, lineage และ download URL เมื่อ publish แล้ว |

POST previews รับ checkpoint ID, dataset revision, exploration sample ID และ target ID/channel index จาก catalog. GET catalog คืน target mode, channel count, index เริ่มที่ 0, hook position หรือ parameter mapping, allowed operations และ scale bounds. Backend เป็นผู้กำหนดข้อจำกัดขนาดและเวลาให้ preview แยกจาก quota ของ experiment run.

## Recipe, submit และ comparison

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

ID และ target เป็นตัวอย่าง ต้อง resolve จาก catalog จริง. Request มีหนึ่ง intervention ต่อ recipe ใช้ `activation` หรือ `weight`, `scale` ที่ finite ในช่วง policy หรือ `zero` ซึ่งละ field scale. Weight target อยู่ใน Conv kernel slice ที่ adapter ประกาศ. Server คืน persisted recipe ที่มี schema version, recipe ID/hash, hypothesis, creator type, model/adapter revision, checkpoint SHA-256, dataset hash, split manifest hash, preprocessing revision/parameters, seed เมื่อเกี่ยวข้อง, evaluator revision และ intervention mapping. Fingerprints และ mapping มาจาก backend. Import ตรวจ hash จาก canonical payload และ resolve dependencies โดยไม่ผูกกับ absolute paths หรือ workspace IDs เดิม.

- **Baseline** ส่ง `{"kind":"baseline","checkpoint_id":"ck_demo","dataset_revision_id":"ds_demo_v1","evaluation":{...}}` โดยใช้ evaluation schema เดียวกัน. GET report คืน baseline metrics และ fingerprint.
- **Intervention** ส่ง `{"kind":"intervention","recipe_id":"rec_123"}`. Backend คำนวณหรือโหลด baseline ที่ evaluation fingerprint ตรงกัน.
- **Retry** ส่ง submit ใหม่พร้อม `retry_of_run_id` และ key ใหม่ Backend ยืนยันว่าเป็นคำขอเดิมของ run ที่ failed/cancelled และ actor มีสิทธิ์ รันสำเร็จนำมา replay ผ่าน submit ปกติ.
- **Final evaluation** ส่ง `recipe_id`, `recipe_hash` และ `confirmed:true` ของสูตรที่ freeze สำหรับ holdout. การสร้างสูตรสำหรับ holdout และผลทุกเส้นทางใช้ human scope. Audit ผูก recipe hash, split manifest hash และเวลา.
- **Checkpoint export** ส่ง `source_run_id` และ `confirmed:true`. Export record มี parent/child SHA-256, recipe hash และ source run ID. Service ใช้ชื่อไฟล์ที่ระบบจัดการและส่ง download URL แบบ scoped เมื่อ artifact เสร็จ.

`Run` คืน `run_id`, `kind`, `status`, `recipe_id` เมื่อมี, `progress` ที่ระบุ phase และ completed/total batches เมื่อทราบ, timestamps และ error เมื่อ failed. `queued → running → succeeded | failed | cancelled` และ queued ยกเลิกได้โดยตรง. หาก cancel commit ก่อน success commit ให้จบเป็น cancelled. Poll ทุก 0.5–1 วินาทีในช่วงแรกแล้วเพิ่ม interval จนถึงค่าที่กำหนด.

`Comparison` คืน `run_id`, `baseline_run_id`, `recipe_hash`, `evaluation_fingerprint`, runtime metadata และ `slices[]`. แต่ละ slice มี ID, N, before/after accuracy (0–1), `delta_pp`, confusion matrices (row = truth, column = prediction, order 0–9) ตาม metric ที่เลือก และ counts ของ fixed/regressed/changed_wrong. กลุ่มว่างคืน N = 0 และ accuracy/delta เป็น null พร้อมเหตุผล. `changes` คืน sample ID, condition, truth, before/after labels และ category. การจับคู่ใช้ manifest และ sample IDs เดียวกันทั้งหมด.

## Error contract และ quota

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

| Status | Codes และการตอบสนอง |
|---|---|
| 400 / 422 | `invalid_json`, `invalid_target`, `invalid_scale`, `unsupported_schema` แก้ payload จาก field error |
| 401 / 403 | `unauthenticated`, `forbidden` หยุด action และใช้ scope ที่ได้รับอนุญาต |
| 404 | `not_found` สำหรับ object ที่ไม่พบหรืออยู่นอก project/session ที่มองเห็นได้ |
| 409 | `dependency_mismatch`, `idempotency_conflict`, `run_not_ready`, `artifact_unavailable` ตรวจ context ก่อนดำเนินการ |
| 413 | `payload_too_large` ใช้ไฟล์หรือ request ตาม cap ที่ประกาศ |
| 429 | `quota_exhausted` หรือ `queue_full` แยกงบหมดจากคิวเต็มที่ลองใหม่ได้ |
| 500 / 503 | `internal_error`, `service_unavailable` ตรวจสถานะและส่ง key เดิมเมื่อผล submit ยังไม่ทราบ |

Agent quota คิดเมื่อรับ run รวมงานที่ล้มเหลว/ยกเลิก. การฝัง baseline ภายใน intervention คิดตาม submission เดียว ส่วน explicit baseline เป็นอีก submission. ค่าเสนอ 10 submissions/session และประกาศค่าจริงผ่าน API. การรับงานและหักโควตาทำใน transaction เดียว. Worker failures เช่น timeout/interrupted เป็น terminal run errors ผ่าน status endpoint. Results อ่านได้หลัง publish สำเร็จเท่านั้น.

## Contract checks

- Desktop และ Agent ส่งสูตรเดียวกันบน runtime/manifest เดียวกันแล้วได้ sample changes, counts และ metrics ตรงกัน
- Lost response + replay key คืน run ID เดิมและหักงบครั้งเดียว
- Cancel กับ publish มีผล terminal เดียว. Run หลัง failure/cancel เทียบ baseline เดิมได้
- Agent อ่าน holdout ผ่าน catalog, preview, history, comparison, report หรือ artifact URL ไม่ได้ และสร้าง checkpoint export ไม่ได้
- Replay ใน workspace ใหม่ resolve artifacts ด้วย hash และ weight child ที่โหลดกลับให้ผลเท่ากับ source run ตาม runtime อ้างอิง
