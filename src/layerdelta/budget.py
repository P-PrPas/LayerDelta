"""Per-session read and write budgets of an agent session (v0.4-S3). Metered and refused in the backend; torch-free.

One server process is one agent session. The human sets limits at start (`--agent-budget values=N,scan=N,applies=N,blob_bytes=N`);
a dimension without a limit is unlimited but still counted. The dimensions:

- `values`: elements returned to the agent (`read_tensor`, group reads, `diff` with `values: true`).
- `scan`: elements read from checkpoint files (every read goes through `TensorTable`, so `stats`, `query`, `diff`, group tools,
  `plan` and `validate` are all metered the same way, and a cached tensor hash costs nothing). Charged chunk by chunk BEFORE each
  chunk is read: a call that would pass the limit stops there and answers `budget_exhausted` with no result.
- `blob_bytes`: bytes of `upload_blob`.
- `applies`: the existing apply quota, reported here under the same name (`--agent-budget applies=N` sets it).

`apply` charges `applies` when it makes a child; its planning reads are checked against `scan` but charged only if no child comes of it, and writing the child is not metered. A replay of a keyed apply reads nothing,
so it costs nothing. Only agent principals are metered (the request is marked by the `actor` dependency); a human is never charged.
The meter of one request lives in a context variable, like the trace fields, so the ops need no extra argument.
"""

import threading
from contextvars import ContextVar

DIMENSIONS = ("values", "scan", "applies", "blob_bytes")
METERED = ("values", "scan", "blob_bytes")  # `applies` is the store's quota ledger


class AggregateTooSmall(Exception):
    """A statistic over fewer than K elements, asked by a token without weights:values. `field` names the argument to widen."""

    def __init__(self, count: int, minimum: int, field: str):
        self.count, self.minimum, self.field = count, minimum, field
        how = {"unit_axis": " (pass unit_axis: null for the whole-selection summary)",
               "reduce_over_axes": " (reduce over more axes or select more elements)"}.get(field, " (select more elements)")
        super().__init__(f"this statistic would aggregate {count} elements; a session without weights:values needs at least {minimum}{how}")


class BudgetExhausted(Exception):
    """Refused before reading. `field` names the dimension."""

    def __init__(self, dimension: str, cost: int, remaining: int):
        self.dimension, self.cost, self.remaining = dimension, cost, remaining
        super().__init__(f"the {dimension} budget has {remaining} left and this call needs at least {cost}")


def parse(text: str) -> dict[str, int]:
    """`values=1000,scan=1e6`-style flag value into limits; raises ValueError with a message for the user."""
    limits: dict[str, int] = {}
    for part in filter(None, (p.strip() for p in text.split(","))):
        name, _, number = part.partition("=")
        if name not in DIMENSIONS:
            raise ValueError(f"unknown budget {name!r}; choose from {', '.join(DIMENSIONS)}")
        try:
            value = int(float(number))
        except ValueError:
            raise ValueError(f"budget {name} needs a number, got {number!r}") from None
        if value < 0 or name in limits:
            raise ValueError(f"budget {name} must be given once and be non-negative")
        limits[name] = value
    return limits


class Budgets:
    """The session's limits and what has been used. Thread-safe."""

    def __init__(self, limits: dict[str, int] | None = None):
        self.limits = {k: v for k, v in (limits or {}).items() if k in METERED}
        self.used = {k: 0 for k in METERED}
        self._lock = threading.Lock()

    def add(self, dimension: str, amount: int) -> None:
        """Count a cost that was already incurred (no refusal): the reads of an apply that did not create a child."""
        with self._lock:
            self.used[dimension] += amount

    def check(self, dimension: str, amount: int) -> None:
        """Refuse if `amount` more would pass the limit (nothing is counted)."""
        limit = self.limits.get(dimension)
        if limit is not None and self.used[dimension] + amount > limit:
            raise BudgetExhausted(dimension, amount, max(0, limit - self.used[dimension]))

    def charge(self, dimension: str, amount: int) -> None:
        if amount <= 0:
            return
        with self._lock:
            self.check(dimension, amount)
            self.used[dimension] += amount

    def snapshot(self) -> dict:
        with self._lock:
            return {k: {"limit": self.limits.get(k), "used": self.used[k],
                        "remaining": None if k not in self.limits else max(0, self.limits[k] - self.used[k])} for k in METERED}


class Meter:
    """One request: charges go to the session's budgets and into this request's tally (the trace field `budget_charged`)."""

    def __init__(self, budgets: Budgets):
        self.budgets, self.tally = budgets, {}
        self.active = False   # set by the actor dependency for an agent principal
        self.min_aggregate = 0  # K for a token without weights:values (egress class 1); 0 everywhere else
        self.scan_mode = None  # apply: "defer" while it plans (checked, charged only if no child comes of it), "off" while it writes
        self.deferred = 0      # scan elements read while deferring

    def charge(self, dimension: str, amount: int) -> None:
        if not self.active or amount <= 0:
            return
        if dimension == "scan" and self.scan_mode:
            if self.scan_mode == "defer":
                with self.budgets._lock:  # the limit still stops an apply's reads; the charge waits for the outcome (`settle`)
                    self.budgets.check("scan", self.deferred + int(amount))
                self.deferred += int(amount)
            return
        self.budgets.charge(dimension, amount)
        self.tally[dimension] = self.tally.get(dimension, 0) + int(amount)

    def settle(self, status: int) -> None:
        """An apply that made no child (an error, a no-op, a refusal) is a read like any other: charge what it read. Otherwise a
        refused apply would be a free oracle (`clamp(min=-1e9)` answers `no_op` exactly when every value is above it)."""
        if self.deferred and status >= 400:
            self.budgets.add("scan", self.deferred)
            self.tally["scan"] = self.tally.get("scan", 0) + self.deferred
        self.deferred = 0


_current: ContextVar = ContextVar("layerdelta_meter", default=None)


def begin(budgets: Budgets):
    meter = Meter(budgets)
    return meter, _current.set(meter)


def end(token) -> None:
    _current.reset(token)


def activate(min_aggregate: int = 0) -> None:
    meter = _current.get()
    if meter is not None:
        meter.active = True
        meter.min_aggregate = min_aggregate


def min_aggregate() -> int:
    """K for this request: statistics over fewer elements are refused (egress class 1). 0 means no rule."""
    meter = _current.get()
    return meter.min_aggregate if meter is not None else 0


def require_aggregate(count: int, field: str) -> None:
    """Refuse a statistic over `count` elements when the session is class 1 and `count` < K. A statistic over one element is that element."""
    k = min_aggregate()
    if k and count < k:
        raise AggregateTooSmall(count, k, field)


def defer_scan() -> None:
    """From here the request's scan reads are checked but charged only if the request fails (an apply's planning)."""
    meter = _current.get()
    if meter is not None:
        meter.scan_mode = "defer"


def free_scan() -> None:
    """From here the request's scan reads are not metered (an apply that goes on to write its child)."""
    meter = _current.get()
    if meter is not None:
        meter.scan_mode = "off"


def charge(dimension: str, amount: int) -> None:
    """Charge the current request, if it is an agent's. A no-op everywhere else (human, tests, library use)."""
    meter = _current.get()
    if meter is not None:
        meter.charge(dimension, amount)


def header(budgets: Budgets, applies: tuple[int, int | None] | None = None) -> str:
    """`values=used/limit;scan=used/limit;applies=used/limit;blob_bytes=used/limit` (`-` for no limit)."""
    snap = budgets.snapshot()
    parts = {k: f"{v['used']}/{'-' if v['limit'] is None else v['limit']}" for k, v in snap.items()}
    if applies is not None:
        parts["applies"] = f"{applies[0]}/{'-' if applies[1] is None else applies[1]}"
    return ";".join(f"{k}={parts[k]}" for k in DIMENSIONS if k in parts)
