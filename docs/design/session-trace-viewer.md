# Session trace viewer: design spec (agent supervision)

Owner: Senior UX/UI Designer · Status: **accepted, rev 2** (AI Scientist review of the §8 gaps and alignment with [v0.4-agent-surface.md](../architecture/v0.4-agent-surface.md) folded in) · Data: `GET /trace`, `GET /agent-session` ([api.md](../reference/api.md)) and the v0.4 S5 routes · Roadmap: v0.4 task S5 (session viewer, revoke, budgets, approval gate), owner priority "agent surface first" · Mockup: [session-trace-viewer-mockup.html](session-trace-viewer-mockup.html) · Links into: [tensor-browser.md](tensor-browser.md), [architecture-navigator.md](architecture-navigator.md)

## 1. Purpose and non-goals

A human who lets an agent work on a workspace needs to answer five questions without reading logs:

1. **What did the agent look at?** Which checkpoints and tensors it described, read, ran stats on, diffed or queried.
2. **What did it try?** Plans and validations, including the ones it dropped.
3. **What did it change?** Every apply, the child it produced, and whether that child was new, already existed, or was a replay.
4. **What did it spend?** Each v0.4 budget dimension (`values`, `scan`, `applies`, `blob_bytes`) against its limit, and which egress class the session has.
5. **What went wrong or was blocked?** 403 for a missing capability or egress class, 429 for an exhausted budget, 401 after a revoke, 422 request errors and 409 conflicts.

And it lets the human act on the answers, with the three supervision controls v0.4 puts here (S5): **revoke** the session, **change a budget**, and, when the approval gate is on, **approve or reject** a proposed apply.

From every answer the human can jump to the evidence: the tensor in the tensor browser, the group in the architecture navigator, the child compared with its parent.

**The trace stores ids only.** It has no tensor values, request bodies, prompts or tokens (T5). The viewer never makes that look otherwise. It shows ids and resolves them through other read routes as the human session. It never says what the agent "thought" or "wanted". The only words in the agent's own voice are the recipe `hypothesis`, fetched from the stored recipe and labelled *"agent's stated hypothesis"* (§8 gap 4). It is **untrusted text**: plain text only, never markdown, HTML or links, and never used to word anything the viewer says itself.

Non-goals:
- Steering the agent: the viewer does not send the agent instructions, edit its plans, or change capabilities or the egress class (set when the service starts). Its only controls are revoke, budget change and approval (§4.7).
- Chat and prompt transcripts: they are not in the trace by design.
- Multi-workspace views.
- Any analytics beyond the server's `summary`.

## 2. Principles

| Reference | What works there | Principle here |
|---|---|---|
| Browser DevTools network panel | One row per request, a status colour, a waterfall, a detail pane with headers verbatim, filter chips, "preserve log" | One row per event, **verbatim fields in the inspector**, filter chips by kind and status, live tail that pauses when you scroll |
| Distributed tracing (Jaeger, Honeycomb) | A request id links everything; select a span and its parents and children light up | **Select an apply and its provenance chain lights up**: the plan events and the reads that `links` names. The client draws only links the server returned |
| Git log with graph | A commit points at its parent; history reads as a chain of immutable states | Applies read as a **lineage strip**: parent → child, with recipe and replay markers |
| Cloud billing / quota consoles | The budget meter is always visible; the moment a limit is hit is marked on the timeline | **Budget meters in the session header**; the first 429 is a marked event, not a toast |
| CI run logs (GitHub Actions) | Steps collapse; failures expand; links go to the artifact | Runs of consecutive reads collapse into one **read burst** row; errors and writes never collapse |

## 3. Information architecture

Same Desktop shell. The rail gains **Sessions** below **Architecture**. It shows a live dot when an agent session has had an event in the last 60 s.

```
┌ rail ┬ context bar: crumb · live ● · egress class · budget meters · Approvals (n) · Revoke… · ? ────┐
│      ├ A · Sessions (280px) ┬ B · Timeline (flex) ─────────────────────┬ C · Event inspector (400px) ┤
│      │ agent sessions,      │ filter chips · view: Timeline | Applies │ fields verbatim, copy each   │
│      │ newest first         │ lanes: Read · Plan · Write · Run · Other│ links: provenance chain      │
│      │ human session (off   │ seq · time · tool · target · status     │ actions: Open in Tensors,    │
│      │ by default)          │ read bursts collapsed                   │ Compare child, Open in Arch. │
│      │ per row: events,     │ live tail, "12 new events" chip         │                              │
│      │ applies, charged     │                                         │                              │
└──────┴──────────────────────┴─────────────────────────────────────────┴──────────────────────────────┘
```

- **Approvals:** when the approval gate is on, a third timeline tab **Proposals (n)** appears and the rail item shows the pending count. When the gate is off, neither appears.
- **Selection model:** one session, plus an optional event `seq`. Both are in the URL: `?session=agent-3f9c…&seq=41`.
- **Human session:** listed separately and **excluded by default**. Browsing as a human writes human events (the tensor browser's own reads), and they would drown the agent's. The human session can be opened explicitly.
- **Session identity:** `agent-<hex>` is per server process, so one agent working across a restart shows up as two sessions. The list shows start time and process boundary, and never merges sessions on the client.

## 4. Screens and behaviour

### 4.1 Sessions list (A)

One row per session. Each row shows a short id (`agent-3f9c…`, full id in a tooltip), kind (agent or human), started, last event, the capability chips (`model:read`, `weights:read`, `weights:write`), the egress class (`class 1 · stats only` or `class 2 · values`), the counts `events · applies · charged`, and the call mix (`mcp 180 · http 6`, from `via_mix`). A row with an event in the last 60 s has a live dot. A row whose last event was a 429 shows "budget used up". A revoked session shows **Revoked** and its time, and is never shown as live.

This needs the sessions route (§8 gap 1, accepted). Until it exists, the list is built from `GET /trace` without `session=` (the human session sees every session) by grouping the events of the loaded pages by `session`. It is labelled "*from loaded events*" and is incomplete beyond the loaded pages.

### 4.2 Session header and budgets (context bar)

- **Egress class:** a chip, `class 1 · stats only` (default) or `class 2 · values allowed`, with a tooltip saying what leaves the machine. It is read-only here: the class is a human choice made when the agent is connected.
- **Budgets:** one meter per dimension that has a limit, from `summary.budget` (§8 gap 2, accepted): `values 41k / 100k` · `scan 2.1M / 10M` · `applies 3 / 10` · `blob 0 / 8 MB`. A dimension with no limit shows its usage as text without a bar. A meter at ≥ 80% turns amber, and an exhausted one turns alert and links to its 429 event. Each meter has an *Edit* action (§4.7). Until `summary.budget` exists, `applies` comes from `summary.charged_runs` with the text *"limit not visible to the human session"*, and the other dimensions are absent. A limit is never guessed.
- **Counts:** `events 186 · applies 3 · replays 1 · errors 7` from `summary`. `errors` comes from `summary.errors` (§8 gap 5); until then it is counted from loaded pages and labelled *"in loaded events"*.
- **Capabilities:** the snapshot of the latest event. If the snapshot changes inside the session (a restart with other flags shows up as a new session, so this should not happen), the viewer marks it on the timeline.
- **Revoke…:** a button at the right end of the context bar (§4.7). After a revoke it is replaced by `Revoked 12:31`.

### 4.3 Timeline (B)

Rows are oldest first, the order `GET /trace` returns, with the page loaded at the bottom. The live tail polls with `after = next_after` every 2 s while the view is visible. If the human has scrolled up, polling continues but the view does not jump: a **"12 new events ↓"** chip appears instead. The connection state is shown as `Live` or `Paused (scrolled up)` or `Disconnected: retrying`.

Each row shows `seq` · time (`at`, relative and absolute in a tooltip) · lane icon · `tool` · target · status · flags. The lane comes from `tool`, which S5 adds to every event with the same name for HTTP and MCP calls. Until then the viewer maps the route template to the tool. The tool names are the v0.4 registry names (AI Scientist, gap 8):

| Lane | Tools | Colour |
|---|---|---|
| Read | `list_checkpoints`, `get_checkpoint`, `describe`, `graph`, `graph_node`, `groups`, `stats`, `query`, `diff`, `read_tensor` | muted |
| Plan | `plan`, `validate` | accent |
| Write | `apply`, `upload_blob` | changed (amber) |
| Run | legacy `propose`, `run`, `status`, `compare`, `changes`, `cancel` (only with `--legacy-tools`); `evaluate` (v0.5) | violet |
| Meta | `session_info`, `trace` | dim |
| Control | human actions on this session: revoke, budget change, approve, reject (§4.7) | mint, full-width row |

An unknown tool goes to Meta and is shown verbatim. `read_tensor` and `diff` with `values: true` are the class-2 tools: in a class-1 session they appear as 403 rows.

- **Target:** a short checkpoint name (resolved through `GET /checkpoints/{id}`, falling back to the first 8 hex), then the selection, then the other checkpoint for diffs (`→ parent`). Selections are resolved to tensor names and index ranges with one batch call per loaded page (`POST /selections/resolve`, §8 gap 3). Unresolved ids stay as short `sel_…` ids. Resolved names are file-derived text and are rendered as plain text.
- **Status:** a 2xx shows nothing extra. A 4xx or 5xx shows the code and `error_code` in mono, with the row tinted: 403 (blocked, capability), 429 (budget), 409 (conflict), 422 (request error), 404.
- **Flags:** `charged` (with the dimension and amount from `budget_charged`, e.g. `scan 4,096`), `replayed` (with `↩ #seq` of the original), `existing` (child already in the catalog), `mcp` or `http` (from `via`, shown only when a session mixes both).
- **Read bursts:** three or more consecutive successful Read-lane events on the same checkpoint collapse into one row: *"14 reads · large.matrix, conv1.weight +3 tensors · #22–#35"*. The burst expands in place. Errors, Plan and Write events never collapse and always break a burst. A read that a selected apply links to is lifted out of its burst while the highlight is on, so the chain is always visible row by row.
- **Filters:** chips for `Reads` `Plans` `Writes` `Runs` `Errors only` `Charged`, plus a checkpoint filter. Filters dim rows rather than hide them in Timeline view, so the order stays readable, and hide them in a flat list when *Hide filtered* is on.
- **Provenance highlight:** when an apply is selected, the rows named in its `links.plan_seqs` and `links.selection_seqs` get a mint bar on the left and the rest dims. If a linked seq is not on a loaded page, a row placeholder says *"#12 · not loaded · Load"*. The highlight draws server links only and never infers a link from timing or tensor names.

### 4.4 Applies view (B, second tab)

There is one card per successful apply, so the human can read what changed without the reads around it:

```
#41  apply · charged · 12:04:31
parent  mnist-base (84e3…)  →  child 9fac…c09   checkpoint_created
recipe 7be2…41fa · 2 ops · 1 selection
motivated by: reads #22 #27 #30 · plans #36 #39
[Compare child with parent]  [Open recipe]  [Show in timeline]
```

The hypothesis line is labelled *"agent's stated hypothesis"* and rendered as plain text from `GET /recipes/by-hash/{hash}` (§8 gap 4). Replays appear on the original card as *"replayed at #44 (no charge)"*. An `existing` child says *"already in the catalog; nothing written"*. A failed apply is not a card. It stays in the timeline as an error, because a failed apply releases its charge.

### 4.5 Event inspector (C)

The inspector shows every field of the event **verbatim**: `session`, `seq`, `at`, `route` (template), `tool`, `via`, `request_id`, `status`, `error_code`, `checkpoint_id`, `other_checkpoint_id`, `selection_id`, `selections` (list), `ops`, `recipe_hash`, `child_id`, `run_id`, the idempotency digest, `replayed`/`replay_of`, `charged`, `budget_charged`, `capabilities`, and `links`. Ids are mono, with a copy button on each. An absent field is not shown, never shown as "—". Below the fields:

- **What this means** (one line, from a fixed table keyed by status and `error_code`): for example a class-2 tool in a class-1 session, 403 `forbidden` → *"Blocked: values are not released to this session (egress class 1)."*; otherwise 403 `forbidden` → *"Blocked: this token lacks `weights:read`. Nothing was read."* (the missing capability comes from a fixed route → capability table checked against the event's `capabilities` snapshot), 429 `budget_exhausted` (with `field`) → *"The scan budget is used up. Nothing was read."* (or *"…applies budget… Nothing was written."*), 401 `session_revoked` → *"This session was revoked at 12:31. Nothing was done."*, `idempotency_conflict` → *"Same key reused with a different request. Nothing was written."* It explains the code and never speculates about intent.
- **Actions** (only when the ids allow them):
  - *Open in Tensors*: opens the checkpoint, and the selection when it can be resolved (§8 gap 3). Without that, it opens the checkpoint only.
  - *Compare child with parent*: opens tensor-browser compare mode for an apply with a `child_id`.
  - *Open in Architecture*: when the selection is a group selector (G4 rev7).
  - *Open run*: for a `run_id`.
  - *Copy request id*: matches server logs and error bodies.
- **Privacy note** (always shown, small): *"The trace records ids, not values. Opening a tensor reads it as you (human session)."*

### 4.6 States

| State | Shown |
|---|---|
| No agent has connected | *"No agent sessions yet. Start the service with an agent token, then the agent's requests appear here."* plus a link to the agent-integration guide |
| Session with no events | *"This session has made no requests."* |
| Loading the first page | skeleton rows and `Loading events…` |
| Trace route fails | an error block with the code and request id, and *Retry*. Earlier loaded rows stay, marked stale |
| Budget exhausted | the 429 event is pinned as a marker on the timeline scrollbar, and that dimension's meter turns alert |
| Revoked | a Control row at the revoke, every later agent row is 401 `session_revoked`, and the header shows `Revoked 12:31` instead of *Revoke…* |
| Proposal waiting | the Proposals tab and rail badge show the count; the timeline row is `apply · 202 · proposal p_…` with a *Review* action |
| Blocked requests | 403 rows grouped in the filter count, `errors 7 (5 blocked)` |
| Very long session | paging with `limit` 500, a virtualised list, and `summary` counts that stay exact (server-side) |

### 4.7 Supervision controls (v0.4 S5)

The viewer is read-only apart from these three controls. Each is human only, each asks for confirmation, and each is written to the trace as a Control row in the affected session's timeline (§8 gap 9).

**Revoke** (`POST /agent-sessions/{session}/revoke`). The *Revoke…* button opens a confirm dialog:

> **Revoke agent-3f9c…?**
> Every later request from this agent is refused with `session_revoked`. Children it already created stay in the catalog. This cannot be undone for this session: the agent needs a new server process to get a new session.
> [Cancel] [Revoke session]

The destructive button is not the default. Focus starts on *Cancel*.

**Budget change.** *Edit* on a meter opens a small popover for that dimension: the current `used / limit` and a new limit field. The popover says what the new limit means (*"The agent can read 59k more values"*). Lowering below `used` is allowed and means *"no more calls that charge this dimension"*. *Save* asks once more only when lowering. The change appears as a Control row (`budget · scan 10M → 2M · by human`).

**Approval queue** (only when the service runs with `--agent-apply-requires-approval`; default off). An agent `apply` returns `202 {proposal_id}` and writes nothing. The **Proposals** tab lists one card per pending proposal:

```
proposal p_51d2…   from #41 apply · 12:04:31
parent  mnist-base (84e3…)   recipe 7be2…41fa · 2 ops · 1 selection
agent's stated hypothesis: "conv2 unit 5 is redundant"      (plain text)
motivated by: reads #22 #27 #30 · plans #36 #39
[Preview plan]  [Approve]  [Reject]
```

- *Preview plan* runs `plan` on the same recipe as the human and shows the result inline (refusals, warnings, side effects that would happen) using the navigator's plan table (§4.7 there). Nothing is written.
- *Approve* runs the same recipe as a human-confirmed apply under the same idempotency key. The card then becomes an Applies card that links proposal → child, with the side effects as returned.
- *Reject* asks for no reason (no free text enters the trace) and marks the card *Rejected*.
- Approving does not charge the agent's `applies` budget twice. The card shows the charge as the server reports it.

## 5. Components

| Component | Notes | Reuses |
|---|---|---|
| `SessionList` | rows, live dot, capability chips | tree row styles from the tensor browser |
| `BudgetMeter` | value / limit bar; text-only when the limit is unknown | the progress bar from import stages |
| `TraceTimeline` | virtualised rows, lanes, bursts, provenance highlight, live tail chip | — |
| `ApplyCard` | lineage strip, links, actions | the side-effects list from the navigator §4.7 |
| `EventInspector` | verbatim field list, meaning line, actions | inspector field rows from the navigator |
| `IdChip` | short id, full in tooltip, copy | the `g_7f3a…` pattern |
| `RevokeDialog` | confirm, focus on Cancel | dialog pattern |
| `BudgetPopover` | per-dimension limit edit | — |
| `ProposalCard` | plan preview, Approve, Reject | `ApplyCard`, the navigator plan table |

## 6. Wording rules

- Use "requested", "read", "planned" and "applied". Never "thought", "decided", "wanted" or "tried to".
- "Blocked" is used for 403 only. 429 is "budget used up". 401 after revoke is "revoked". 422 is "request error".
- **Agent-originated text is plain text everywhere**: resolved tensor names, module paths and hypotheses. No markdown, HTML or auto-links, and it is never used inside the viewer's own sentences.
- "Charged" means one run was deducted. "Replayed" means the same key returned the earlier result with no new charge.
- Ids are always short form with a tooltip. The time is always shown with the `seq`, because `seq` is the order and `at` is only a timestamp.

## 7. Keyboard and accessibility

- `j`/`k` move between events, `Enter` opens one in the inspector, `[`/`]` move to the previous or next apply, `e` jumps to the next error, `l` toggles live tail, and `/` focuses the filter.
- The timeline is a `role="grid"` with row headers. The live tail announces through `aria-live="polite"` at most once per 5 s ("12 new events"), never per event.
- Status is never colour-only: the code is always printed and the lane has an icon.

## 8. Backend gaps (requests to the AI Scientist / API owner)

Priority **P1** means the viewer cannot answer one of the five questions without it. **P2** means the viewer works without it, in a degraded way.

| # | Field / route | Why | Priority | Decision (AI Scientist, PR #45) |
|---|---|---|---|---|
| 1 | `GET /trace/sessions` (human only): `[{session, kind, created_at, capabilities, events, applies, charged_runs, last_at, last_status}]` | The sessions list without loading every event | P1 | **accepted** (S5), plus `egress_class`, `revoked`, `via_mix` |
| 2 | Budgets visible to the human per session | Budget meters | P1 | **accepted, generalised**: `summary.budget: {values, scan, applies, blob_bytes: {limit, used, remaining}}`; the quota becomes `applies` |
| 3 | Resolve a selection id to its canonical selector | "What did the agent look at"; *Open in Tensors* at the exact selection | P1 | **accepted**: a separate `selections` table keyed by `(checkpoint_id, selection_id)`, human only, plus batch `POST /selections/resolve {ids}` (≤ 500) |
| 4 | `GET /recipes/by-hash/{recipe_hash}` → frozen recipe | Applies and proposal cards show ops and the stated hypothesis | P1 | **accepted**: human only; hypothesis is untrusted text (plain text, labelled) |
| 5 | `summary.errors`, `summary.by_status` | Exact error counts | P2 | **accepted** |
| 6 | Event `duration_ms` | Slow reads; waterfall later | P2 | **accepted** |
| 7 | Event `tensor` name | Cheaper row targets | P2 | **rejected**: names in the trace break T5; use gap 3's batch resolve |
| 8 | Tool names and lanes | The lane table (§4.3) | P1 | **answered**: v0.4 registry names, table in §4.3 |
| 9 | Control events in the target session: revoke, budget change, approve and reject are written with `target_session` (or as events of the affected session with `principal: human`), plus `budget_change {dimension, from, to}` | Control rows in the agent's timeline (§4.7) | P1 | **accepted**: `principal: human`, `target_session`, `budget_change {dimension, from, to}` (ids and numbers only, T5) |
| 10 | Proposal routes for the approval gate: `GET /proposals?session=&status=pending` → `[{proposal_id, seq, checkpoint_id, recipe_hash, selections, created_at}]`, `POST /proposals/{id}/approve`, `POST /proposals/{id}/reject`; and a budget change route, `POST /agent-sessions/{session}/budget {dimension, limit}` and `POST /agent-sessions/{session}/revoke` (target session in the path; singular `/agent-session` stays the agent's own view) | The Proposals tab, the budget popover and Revoke | P1 | **accepted** (S5), all human only |

The trace itself stays ids only (T5): selectors live in their own table (gap 3) and recipes in the recipe store (gap 4).

## 9. Risks

- **The human session's own reads.** Opening the evidence writes human events. Those events are excluded by default, but the human still sees them in the human session. The privacy note in §4.5 tells them.
- **Over-reading intent.** A timeline invites a story ("it read X, so it applied Y"). The provenance highlight uses only server `links`, and the wording rules forbid intent words.
- **Session split on restart.** A single agent can appear as two sessions. That is honest but confusing, so the list shows the process boundary, and merging stays on the server (a future `agent_label`), never on the client.
- **Untrusted agent text.** Hypotheses and file-derived names can carry text written to mislead a supervisor ("approved by the owner"). They are always plain text, always labelled with their source, and never used in the viewer's own sentences.
- **Approval fatigue.** If every apply needs approval, people click through. Each card leads with the plan preview and its warnings, and *Approve* is never pre-focused.
- **Revoke is final for the session.** The dialog says so, and children already created stay.
- **Polling cost.** At 2 s with `after`, the cost follows the page (T5 indexes). Polling stops when the view is hidden.
