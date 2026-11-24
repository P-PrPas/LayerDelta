# Error codes (generated)

Generated from the error catalog (`agent_content.json`) and the raise sites in `src/layerdelta`: do not edit. Regenerate with `LAYERDELTA_WRITE_ERRORS=1 pytest tests/test_agent_content.py`. A backend error is `{error: {code, message, field}}`; the MCP bridge adds `hint`, `retryable` and `status` from this table ([mcp.md](mcp.md)). A status of `-` means the code is not an HTTP response of its own (an attach or import job result, a library error, or a status chosen at run time).

| Code | Status | Retryable | Hint |
|---|---|---|---|
| `aggregate_too_small` | 403 | no | Your session gets statistics only over at least K elements. Reduce over more axes, select more elements, use fewer bins, or pass unit_axis: null. |
| `already_revoked` | 409 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `artifact_unavailable` | 409 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `backend_error` | - | yes | The service answered without a structured result. Retry once; if it persists, stop and report the status. |
| `backend_unreachable` | - | yes | The service did not answer. Retry once; if it persists, stop and report. |
| `base_mismatch` | 422 | no | The recipe belongs to another base checkpoint; apply it to the checkpoint named in recipe.base, or plan again. |
| `blob_mismatch` | 422 | no | The blob's size does not match the selection; it needs one float32 per selected element. |
| `blob_missing` | 422 | no | Upload the blob first with upload_blob and use the id it returns. |
| `blob_quota` | 413 | no | The workspace blob store is full; reuse existing blobs or use scalar values. |
| `blob_size` | 422 | no | A blob is a flat float32 array: its byte length must be a multiple of 4. |
| `budget_exhausted` | 429 | no | The budget named in field is used up. Stop and report; only a person can raise it. |
| `cancelled` | 409 | no | The run was cancelled; do not resubmit unless asked. |
| `cap_exceeded` | 422 | no | The request is over a size cap; field names it. Narrow the selection, lower the limit, or filter with name_glob. |
| `child_invalid` | 422 | no | The written child failed verification and was discarded; report it. |
| `confirmation_required` | 422 | no | This needs a person's confirmation; it is not available to an agent. |
| `constraint_violation` | 422 | no | The permutation moves units across group-normalisation blocks; keep each unit inside its block of n (see constraints). |
| `data_size` | 409 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `dependency_mismatch` | 409 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `device_unavailable` | 409 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `dtype_overflow` | 422 | no | The result does not fit the tensor's dtype (F16/BF16 range); use smaller values. |
| `dtype_unknown` | 409 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `duplicate_field` | 409 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `duplicate_tensor` | 409 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `empty_selection` | 422 | no | The selection has no elements; widen it. |
| `executable_changed` | - | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `executable_mismatch` | 422 | no | The checkpoint does not fit the attached graph (a missing or reshaped tensor, or weights that carry no checkpoint name). Use a checkpoint of the same layout, or ask a person to attach a graph that binds. |
| `executable_missing` | 422 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `export_failed` | - | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `extent` | 409 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `extent_mismatch` | 409 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `extent_overlap_or_gap` | 409 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `file_changed` | 409 | yes | The checkpoint file changed on disk while reading; retry once, then report. |
| `file_size` | 409 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `forbidden` | 403 | no | Your token lacks the capability this needs (see session_info). Stop and report it; do not try another tool or path to get the same data. |
| `forward_failed` | 422 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `gpu_oom` | - | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `group_not_closed` | 422 | no | permute_units needs a closed group; list them with groups(closed=true). Other ops are allowed on open groups. |
| `group_open` | - | no | Warning, not an error: the edit goes through an open group, so only its known members change. |
| `header_json` | 409 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `header_size` | 409 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `idempotency_conflict` | 409 | no | This key was used for a different request. Use a new key for a new program; reuse a key only to retry the same request. |
| `import_failed` | - | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `import_stalled` | - | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `in_progress` | 409 | yes | The same keyed request is still running. Retry with the same key in a moment. |
| `incompatible_source` | 422 | no | The source tensor differs in dtype or shape; choose a compatible source. |
| `incompatible_tensors` | 422 | no | The two checkpoints differ in dtype or shape for this tensor; diff without a selector or pick another tensor. |
| `index_invalid` | 409 | no | An index in the request is malformed; see field. |
| `index_mismatch` | 409 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `index_out_of_range` | 422 | no | An index is past the end of its axis; check the shape in describe. |
| `interrupted` | 503 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `invalid_argument` | 422 | no | A request field is wrong; field names it. Fix that field only and resend. |
| `invalid_arguments` | - | yes | Send the tool arguments as one JSON object that matches the tool's input schema. |
| `invalid_axis` | 422 | no | Axes are 0 to ndim-1 (decimal strings as index keys, integers elsewhere); check the shape in describe. |
| `invalid_checkpoint` | 422 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `invalid_cursor` | 422 | no | Pass the cursor exactly as the previous answer gave it, with the same other arguments, or drop it to start over. |
| `invalid_display_name` | 422 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `invalid_filename` | 422 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `invalid_graph` | 422 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `invalid_index` | 422 | no | Indices are non-negative integers, or {start, stop} with start < stop. |
| `invalid_limit` | 422 | no | Use a limit inside the documented range. |
| `invalid_model` | - | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `invalid_path` | 422 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `invalid_permutation` | 422 | no | perm must list each of 0..size-1 exactly once, for the whole group. |
| `invalid_range` | 422 | no | A range needs 0 <= start < stop <= the axis size. |
| `invalid_request` | 422 | no | The request does not match the schema; check required fields and types. |
| `invalid_scale` | 422 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `invalid_spec` | - | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `invalid_target` | 422 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `io_error` | - | yes | A file could not be read; retry once, then report. |
| `ir_hash_mismatch` | 422 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `length_mismatch` | 400 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `length_required` | 411 | no | The upload needs a known length; send the data in one call. |
| `metadata` | 409 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `no_graph` | 422 | no | No graph is attached to this checkpoint. Group selectors and graph tools need one and only a person can attach it; use tensor selectors meanwhile. |
| `no_members` | 422 | no | The members filter leaves nothing; loosen roles or exclude. |
| `no_op` | 422 | no | The program changes no stored value. Check the selection and arguments (scale by 1, clamp outside the range); pass allow_noop only if you mean it. |
| `non_affine_group` | 422 | no | This group cannot be expanded to unit indices; use tensor-local selectors from its members. |
| `non_finite` | 422 | no | The result would contain NaN or Inf; use smaller factors or clamp first. |
| `non_persistent_buffer` | - | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `noncanonical_recipe` | 422 | no | The recipe was edited by hand; send the recipe exactly as plan returned it, or send the program instead. |
| `nondeterministic_edit` | 422 | no | The edit did not reproduce; report it. |
| `not_found` | 404, 422 | no | The id does not exist here. List again (list_checkpoints, describe, groups) and use an id from the result. |
| `not_offered` | 422 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `not_pending` | 409 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `output_exists` | 422 | yes | The child already exists; retry with the same key to get its result. |
| `payload_too_large` | 413 | no | The upload is too large; split it or use a scalar value. |
| `proposal_rejected` | 409 | no | A person rejected this apply. Do not retry it; use a new key only for a different program. |
| `proposal_stale` | 409 | no | The stored proposal no longer matches; plan and apply again with a new key. |
| `provider_disabled` | 422 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `provider_mismatch` | 422 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `queue_full` | 429 | yes | The run queue is full. Retry later with the same key. |
| `quota_exhausted` | 429 | no | The apply quota is used up. Stop applying; report the children you made. |
| `read_only_dtype` | 422 | no | Only F32, F16 and BF16 tensors can be edited. |
| `recipe_hash_mismatch` | 422 | no | The recipe was modified; send plan's result.recipe unchanged. |
| `request_timeout` | - | yes | The service did not answer in time. The request may have completed: retry with the same key to get its result. |
| `result_too_large` | - | no | Lower the argument named in field (limit, max_elems) or narrow the selection. |
| `run_not_ready` | 409 | yes | The run has not finished; poll status with a short backoff. |
| `schema_mismatch` | 409 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `service_unavailable` | 503 | yes | The service is busy. Retry after a short wait with the same key. |
| `session_gone` | 409 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `session_revoked` | 401 | no | A person revoked this session. Stop and report what you have. |
| `shape` | 409 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `shard_missing` | 409 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `shard_outside_directory` | - | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `source_missing` | 422 | no | The source checkpoint of lerp or copy_from is not in this workspace; use an id from list_checkpoints. |
| `source_not_found` | - | no | The source checkpoint of lerp or copy_from is not in this workspace; use an id from list_checkpoints. |
| `tensor_record` | 409 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `timeout` | 503 | yes | The operation took too long. Narrow it, or retry once. |
| `too_many_proposals` | 429 | no | Many of your applies are waiting for a person. Wait for decisions (trace shows them) instead of proposing more. |
| `total_size` | - | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `unauthenticated` | 401 | no | The session token is not valid for this service. Stop: a person must restart the service or the bridge. |
| `unknown_field` | 422 | no | Remove the field named in field; the schema does not have it. |
| `unknown_group` | 422 | no | Use a group_id from groups for this checkpoint; ids change when the graph changes. |
| `unknown_model` | - | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `unknown_node` | 422 | no | Use a node id from graph. |
| `unknown_provider` | - | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `unknown_tensor` | 422 | no | Use an exact tensor name from describe. |
| `unknown_tool` | 404 | no | Use a tool name from the tool list. |
| `unreadable` | 409 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `unsupported_base` | 422 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `unsupported_dtype` | - | no | This dtype cannot be used here; choose a float tensor. |
| `unsupported_engine` | 422 | no | The recipe was made for another ops revision; plan the program again. |
| `unsupported_format` | 422 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `unsupported_ir_version` | 422 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `unsupported_literal` | - | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `unsupported_mode` | 422 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `unsupported_node` | - | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `unsupported_placeholder` | - | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `unsupported_schema` | 422 | no | Not a recipe this service reads; plan the program again. |
| `unsupported_selector` | 422 | no | Use {tensor, index} or a group selector {group, units, members}. |
| `worker_crashed` | - | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `worker_failed` | 500 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `worker_output` | 422 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `worker_timeout` | 422 | no | This concerns the workspace or a person's action, not your request; report the code and message, then continue with other work or stop. |
| `workspace_locked` | 409 | yes | Another process holds the workspace; retry later. |
