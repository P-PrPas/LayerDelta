# Session trace viewer: design spec (agent supervision)

Owner: Senior UX/UI Designer · Status: **proposed, rev 1: information architecture** (the tool vocabulary waits for the AI Scientist's v0.4 agent-surface design; see §8) · Data: `GET /trace`, `GET /agent-session` ([api.md](../reference/api.md)) · Roadmap: v0.5 "trace timeline", owner priority "agent surface first" · Mockup: [session-trace-viewer-mockup.html](session-trace-viewer-mockup.html) · Links into: [tensor-browser.md](tensor-browser.md), [architecture-navigator.md](architecture-navigator.md)

## 1. Purpose and non-goals

A human who lets an agent work on a workspace needs to answer five questions without reading logs:

1. **What did the agent look at?** Which checkpoints and tensors it described, read, ran stats on, diffed or queried.
2. **What did it try?** Plans and validations, including the ones it dropped.
3. **What did it change?** Every apply, the child it produced, and whether that child was new, already existed, or was a replay.
4. **What did it spend?** Charged runs against the quota, and later the read budget (v0.5).
5. **What went wrong or was blocked?** 403 for a missing capability, 429 for an exhausted budget, 422 request errors and 409 conflicts.

From every answer the human can jump to the evidence: the tensor in the tensor browser, the group in the architecture navigator, the child compared with its parent.

**The trace stores ids only.** It has no tensor values, request bodies, prompts or tokens (T5). The viewer never makes that look otherwise. It shows ids and resolves them through other read routes as the human session. It never says what the agent "thought" or "wanted". The only words in the agent's own voice are the recipe `hypothesis`, and only when it is fetched from the stored recipe and labelled as such (§8 gap 4).

Non-goals:
- Controlling the agent: no pause, kill or revoke buttons. Capabilities and quota are set when the service starts. A *Stop agent* control is an owner decision that is not on the roadmap.
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
┌ rail ┬ context bar: crumb · session picker · live ● · budget meters · Export ids · ? ───────────────┐
│      ├ A · Sessions (280px) ┬ B · Timeline (flex) ─────────────────────┬ C · Event inspector (400px) ┤
│      │ agent sessions,      │ filter chips · view: Timeline | Applies │ fields verbatim, copy each   │
│      │ newest first         │ lanes: Read · Plan · Write · Run · Other│ links: provenance chain      │
│      │ human session (off   │ seq · time · tool · target · status     │ actions: Open in Tensors,    │
│      │ by default)          │ read bursts collapsed                   │ Compare child, Open in Arch. │
│      │ per row: events,     │ live tail, "12 new events" chip         │                              │
│      │ applies, charged     │                                         │                              │
└──────┴──────────────────────┴─────────────────────────────────────────┴──────────────────────────────┘
```

- **Selection model:** one session, plus an optional event `seq`. Both are in the URL: `?session=agent-3f9c…&seq=41`.
- **Human session:** listed separately and **excluded by default**. Browsing as a human writes human events (the tensor browser's own reads), and they would drown the agent's. The human session can be opened explicitly.
- **Session identity:** `agent-<hex>` is per server process, so one agent working across a restart shows up as two sessions. The list shows start time and process boundary, and never merges sessions on the client.

## 4. Screens and behaviour

### 4.1 Sessions list (A)

One row per session. Each row shows a short id (`agent-3f9c…`, full id in a tooltip), kind (agent or human), started, last event, the capability chips (`model:read`, `weights:read`, `weights:write`), and the counts `events · applies · charged`. A row with an event in the last 60 s has a live dot. A row whose last event was a 429 shows a "budget exhausted" mark.

This needs a sessions route (§8 gap 1). Until it exists, the list is built from `GET /trace` without `session=` (the human session sees every session) by grouping the events of the loaded pages by `session`. It is labelled "*from loaded events*" and is incomplete beyond the loaded pages.

### 4.2 Session header and budgets (context bar)

- **Runs:** a meter `charged 3 / 10` from the quota limit and `summary.charged_runs`. The limit is only available to the agent itself through `/agent-session` (§8 gap 2). Until the human can read it, the meter shows `3 charged · limit not visible to the human session` without a bar. A guessed limit is never drawn.
- **Reads (v0.5):** `reads 412 / 2,000` once the per-session read budget exists. Until then the meter is absent, not shown as zero.
- **Counts:** `events 186 · applies 3 · replays 1 · errors 7` from `summary`. Errors are counted only from the loaded pages, labelled "*in loaded events*" until the summary returns them (§8 gap 5).
- **Capabilities:** the snapshot of the latest event. If the snapshot changes inside the session (a restart with other flags shows up as a new session, so this should not happen), the viewer marks it on the timeline.

### 4.3 Timeline (B)

Rows are oldest first, the order `GET /trace` returns, with the page loaded at the bottom. The live tail polls with `after = next_after` every 2 s while the view is visible. If the human has scrolled up, polling continues but the view does not jump: a **"12 new events ↓"** chip appears instead. The connection state is shown as `Live` or `Paused (scrolled up)` or `Disconnected: retrying`.

Each row shows `seq` · time (`at`, relative and absolute in a tooltip) · lane icon · `tool` · target · status · flags. Lanes are derived from `tool`:

| Lane | Tools | Colour |
|---|---|---|
| Read | `describe`, `read_tensor`, `stats`, `diff`, `query`, catalog reads | muted |
| Plan | `plan`, `validate` | accent |
| Write | `apply`, `blobs` | changed (amber) |
| Run | legacy `runs` and previews | violet |
| Other | `trace`, `agent-session`, everything unknown | dim |

An unknown tool goes to Other and is shown verbatim. The lane table is the part that waits for the v0.4 tool list (§8).

- **Target:** a short checkpoint name (resolved through `GET /checkpoints/{id}`, falling back to the first 8 hex), then a tensor or selection, then the other checkpoint for diffs (`→ parent`).
- **Status:** a 2xx shows nothing extra. A 4xx or 5xx shows the code and `error_code` in mono, with the row tinted: 403 (blocked, capability), 429 (budget), 409 (conflict), 422 (request error), 404.
- **Flags:** `charged`, `replayed` (with `↩ #seq` of the original), `existing` (child already in the catalog).
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

Replays appear on the original card as *"replayed at #44 (no charge)"*. An `existing` child says *"already in the catalog; nothing written"*. A failed apply is not a card. It stays in the timeline as an error, because a failed apply releases its charge.

### 4.5 Event inspector (C)

The inspector shows every field of the event **verbatim**: `session`, `seq`, `at`, `route` (template), `tool`, `request_id`, `status`, `error_code`, `checkpoint_id`, `other_checkpoint_id`, `selection_id`, `selections` (list), `ops`, `recipe_hash`, `child_id`, `run_id`, the idempotency digest, `replayed`/`replay_of`, `charged`, `capabilities`, and `links`. Ids are mono, with a copy button on each. An absent field is not shown, never shown as "—". Below the fields:

- **What this means** (one line, from a fixed table keyed by status and `error_code`): for example 403 `forbidden` → *"Blocked: this token lacks `weights:read`. Nothing was read."* (the missing capability comes from a fixed route → capability table checked against the event's `capabilities` snapshot), 429 `quota_exhausted` → *"The run budget is used up. Nothing was written."*, `idempotency_conflict` → *"Same key reused with a different request. Nothing was written."* It explains the code and never speculates about intent.
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
| Budget exhausted | the 429 event is pinned as a marker on the timeline scrollbar, and the header meter turns alert |
| Blocked requests | 403 rows grouped in the filter count, `errors 7 (5 blocked)` |
| Very long session | paging with `limit` 500, a virtualised list, and `summary` counts that stay exact (server-side) |

## 5. Components

| Component | Notes | Reuses |
|---|---|---|
| `SessionList` | rows, live dot, capability chips | tree row styles from the tensor browser |
| `BudgetMeter` | value / limit bar; text-only when the limit is unknown | the progress bar from import stages |
| `TraceTimeline` | virtualised rows, lanes, bursts, provenance highlight, live tail chip | — |
| `ApplyCard` | lineage strip, links, actions | the side-effects list from the navigator §4.7 |
| `EventInspector` | verbatim field list, meaning line, actions | inspector field rows from the navigator |
| `IdChip` | short id, full in tooltip, copy | the `g_7f3a…` pattern |

## 6. Wording rules

- Use "requested", "read", "planned" and "applied". Never "thought", "decided", "wanted" or "tried to".
- "Blocked" is used for 403 only. 429 is "budget used up". 422 is "request error".
- "Charged" means one run was deducted. "Replayed" means the same key returned the earlier result with no new charge.
- Ids are always short form with a tooltip. The time is always shown with the `seq`, because `seq` is the order and `at` is only a timestamp.

## 7. Keyboard and accessibility

- `j`/`k` move between events, `Enter` opens one in the inspector, `[`/`]` move to the previous or next apply, `e` jumps to the next error, `l` toggles live tail, and `/` focuses the filter.
- The timeline is a `role="grid"` with row headers. The live tail announces through `aria-live="polite"` at most once per 5 s ("12 new events"), never per event.
- Status is never colour-only: the code is always printed and the lane has an icon.

## 8. Backend gaps (requests to the AI Scientist / API owner)

Priority **P1** means the viewer cannot answer one of the five questions without it. **P2** means the viewer works without it, in a degraded way.

| # | Field / route | Why | Priority |
|---|---|---|---|
| 1 | `GET /trace/sessions` (human only): `[{session, kind, created_at, capabilities, events, applies, charged_runs, last_at, last_status}]` from the `sessions` table and the event indexes | The sessions list without loading every event | P1 |
| 2 | Quota visible to the human per session: `summary.quota: {limit, charged, remaining}` on `GET /trace?session=` (and `reads: {limit, used}` once the v0.5 read budget exists) | Budget meters. Today the limit is only in the agent's own `/agent-session` | P1 |
| 3 | Resolve a selection id: `GET /checkpoints/{id}/selections/{selection_id}` → canonical selector, for selections the server has resolved (stored when a v2 route resolves one; a selector holds tensor names and index ranges, not values) | "What did the agent look at" needs tensor names; *Open in Tensors* at the exact selection | P1 |
| 4 | Recipe by hash for applied recipes: `GET /recipes/by-hash/{recipe_hash}` → frozen recipe (program, `hypothesis`) when an apply stored it | The Applies card shows the ops and the agent's stated hypothesis, labelled as such | P1 |
| 5 | `summary.errors` and `summary.by_status` (counts of 403, 409, 422 and 429) | Exact error counts in the header for long sessions | P2 |
| 6 | Event `duration_ms` | Spotting slow reads; waterfall later | P2 |
| 7 | Event `tensor` (tensor name) for read routes, as a pattern-checked id-like field | Cheaper than gap 3 for row targets; gap 3 still needed for *Open in Tensors* | P2 (alternative to part of 3) |
| 8 | The v0.4/v0.5 tool list: canonical `tool` names (1 endpoint ≡ 1 client method ≡ 1 MCP tool) and their lane (read, plan, write, run, meta) | The lane table (§4.3) and the meaning lines. **Waiting for the AI Scientist's agent-surface design** | P1 |

Gaps 3, 4 and 7 add stored data, so each needs the AI Scientist's OK on privacy against the T5 rule (ids only). Tensor names and index ranges are not values, but they are more than ids today.

## 9. Risks

- **The human session's own reads.** Opening the evidence writes human events. Those events are excluded by default, but the human still sees them in the human session. The privacy note in §4.5 tells them.
- **Over-reading intent.** A timeline invites a story ("it read X, so it applied Y"). The provenance highlight uses only server `links`, and the wording rules forbid intent words.
- **Session split on restart.** A single agent can appear as two sessions. That is honest but confusing, so the list shows the process boundary, and merging stays on the server (a future `agent_label`), never on the client.
- **Polling cost.** At 2 s with `after`, the cost follows the page (T5 indexes). Polling stops when the view is hidden.
