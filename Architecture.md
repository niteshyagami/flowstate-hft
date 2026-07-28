# FlowState HFT — System Architecture

**Version:** 0.1.0
**Design principle:** async at the I/O boundary, synchronous in the hot path.

---

## 1. High-level flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          HYPERLIQUID (external)                              │
│                    wss://api.hyperliquid.xyz/ws · l2Book                      │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │ JSON frames (~10-50/sec on BTC)
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1 · INGESTION                                    src/ingestion.py     │
│  ────────────────────────────────────────────────────────────────────────    │
│  HyperliquidL2Client                                                         │
│    · persistent WebSocket session, ping/pong keepalive                       │
│    · exponential backoff + full jitter on drop                               │
│    · staleness watchdog (silent-socket detection)                            │
│    · json.loads → parse_l2_book → BookSnapshot                               │
│    · ordering invariants enforced; crossed books rejected                    │
│                                                                              │
│  Output: BookSnapshot(coin, exch_ts, local_ts, bids[n,2], asks[m,2], seq)     │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │ asyncio.Queue(maxsize=512)
                                    │ ── backpressure: drop OLDEST ──
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2 · FEATURES                                       src/features.py    │
│  ────────────────────────────────────────────────────────────────────────    │
│  FeatureEngine.update(snapshot) → FeatureVector | None                       │
│    · micro_price          p = (Qa·pb + Qb·pa)/(Qb+Qa)                        │
│    · weighted_micro_price geometric depth decay over N levels                │
│    · order_flow_imbalance Cont-Kukanov-Stoikov e_n                           │
│    · EWMA(OFI), EWMA(|OFI|)   time-aware α = 1 − 2^(−Δt/h)                    │
│    · realized_volatility  σ per second, and σ_price = σ · mid                 │
│    · trend                EWMA of mid drift per second                       │
│                                                                              │
│  State: fixed-capacity deques (ring buffers). Bounded memory, no GC pauses.  │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │ FeatureVector (frozen, slots)
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3 · STRATEGY                                       src/strategy.py    │
│  ────────────────────────────────────────────────────────────────────────    │
│  AvellanedaStoikovStrategy.quote(features, inventory) → Quote                │
│                                                                              │
│    1. reference    s = w·micro_deep + (1−w)·mid                              │
│    2. reservation  r = s − q̃·γ·σ²·(T−t)      [q̃ = cubic soft clamp]         │
│    3. half-spread  δ = ½[γσ²(T−t) + (2/γ)ln(1+γ/κ)]                          │
│    4. OFI skew     centre = r + β·OFI_norm·s   (clipped to ±δ)               │
│    5. tick grid    bid ↓ floor, ask ↑ ceil                                   │
│    6. risk gate    suppress the side that would breach q_max                 │
│                                                                              │
│  Adaptive: κ̂ = 1/EWMA(fill distance), fed back from Stage 4                  │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │ Quote(bid_px, ask_px, sizes, r, δ, …)
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  STAGE 4 · EXECUTION SIMULATION                          src/execution.py    │
│  ────────────────────────────────────────────────────────────────────────    │
│  PaperExecutionSimulator                                                     │
│    submit(quote, snap)   → latency gate, snapshot queue-ahead depth          │
│    on_book(snap)         → two-stage fill test:                              │
│                              (a) crossing:    ask_t ≤ p_bid ?                │
│                              (b) probability: queue × penetration            │
│                            → cash / inventory / fee / edge accounting        │
│                            → markout settlement at horizon h                 │
│                            → equity + inventory curve points                 │
│    flatten(snap)         → taker liquidation at session end                  │
│    report()              → PerformanceReport                                 │
│                                                                              │
│  ⚠ NO VENUE CONNECTIVITY. Paper fills only.                                  │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │ Fill events, equity curve
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  ORCHESTRATION & REPORTING                                       main.py     │
│    TradingLoop · status heartbeat · signal handling · CSV/JSON export         │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Concurrency model

Three `asyncio` tasks, one bounded queue.

| Task | Role | Blocking behaviour |
|---|---|---|
| `ingestion` | Owns the socket. Reads, parses, enqueues. | Never awaits a consumer. |
| `pipeline` | Drains the queue, runs Stages 2-4 synchronously per tick. | Pure CPU, ~50-200 µs/tick. |
| `status` | Periodic heartbeat, exits on shutdown event. | Sleeps on an `Event.wait` with timeout. |

### Why the pipeline is synchronous inside

Stages 2-4 are microsecond-scale pure computation on NumPy arrays and floats.
Inserting `await` points between them would:

- add event-loop scheduling latency (~10-50 µs per hop) for zero concurrency
  gain, since there is no I/O to overlap with;
- open a window where inventory could change between the strategy computing a
  quote and the simulator acting on it, creating a state-consistency bug class
  that is genuinely hard to debug.

`asyncio` earns its keep at the socket boundary. Past that boundary, straight-line
code is both faster and more correct.

### Backpressure policy: drop oldest

When the queue is full, `_offer` evicts the **oldest** snapshot and enqueues the
newest. This is the correct choice for market making and the wrong choice for
almost anything else. A market maker quoting off a 300 ms old book is quoting off
a fiction; skipping frames to stay current is strictly better than processing a
backlog. The drop count is tracked in `_ConnStats.drops` so silent degradation
is visible.

---

## 3. Data model

### `BookSnapshot` (frozen, slots)

```
coin      str          instrument symbol
exch_ts   float        exchange event time, seconds epoch
local_ts  float        local receipt time, seconds epoch
bids      (n,2) f64    col 0 = price (DESC), col 1 = size
asks      (m,2) f64    col 0 = price (ASC),  col 1 = size
seq       int          client-assigned monotonic sequence
```

Design choices:

- **Column-major-ish `(n,2)` matrix, not a list of dicts.** One contiguous
  allocation per side per frame. Cumulative depth is `np.cumsum(book[:,1])`;
  imbalance is `np.dot`. No Python-level loops downstream.
- **`float64`, not `Decimal`.** Crypto prices carry ~10 significant digits;
  `float64` provides 15-16. `Decimal` is ~40× slower for zero accuracy benefit
  here. (This would be the wrong call for a settlement ledger — it is the right
  call for a signal pipeline.)
- **Frozen + slots.** Immutability means a snapshot can be handed to multiple
  consumers without defensive copies; `slots` removes the per-instance `__dict__`
  (~30% memory reduction, faster attribute access).
- **`local_ts − exch_ts`** is a usable one-way latency proxy, contaminated by
  clock skew. Useful for detecting *changes* in latency, not absolute values.

### `FeatureVector` (frozen, slots)

Carries both `sigma` (relative, per second) and `sigma_price` (absolute, per
second). A-S assumes arithmetic Brownian motion on the mid, so the model needs
absolute price units. Passing relative vol into the A-S formula is a silent
unit bug that produces plausible-looking but meaningless spreads — hence both
are computed and named explicitly.

---

## 4. Latency budget

Measured on a laptop-class CPU, single BTC instrument. These are *pipeline*
latencies; wire latency to Hyperliquid dominates end-to-end.

| Stage | Typical | p99 |
|---|---|---|
| `json.loads` | 15 µs | 60 µs |
| `parse_l2_book` (20 levels × 2) | 20 µs | 45 µs |
| Queue handoff | 5 µs | 20 µs |
| `FeatureEngine.update` | 25 µs | 70 µs |
| `Strategy.quote` | 8 µs | 20 µs |
| `Simulator.on_book` | 15 µs | 40 µs |
| **Total tick-to-quote** | **~90 µs** | **~260 µs** |

**Honest framing:** this is fine for research and hopeless for competitive
liquidity provision. Professional makers operate in the single-digit-microsecond
range on colocated C++/FPGA stacks. The value of this project is that the *model*
and the *measurement methodology* are correct, not that the plumbing is fast.

### Where the time goes, if optimization is ever needed

1. JSON parsing is the single largest cost. `orjson` gives ~3× over stdlib.
2. `parse_l2_book`'s Python loop over levels — vectorizable with a
   pre-allocated buffer and `np.frombuffer` if the venue offered a binary feed.
3. Everything else is already NumPy or scalar arithmetic.

---

## 5. Failure modes and handling

| Failure | Detection | Response |
|---|---|---|
| TCP disconnect | `ConnectionClosed` | Backoff reconnect, resubscribe |
| Silent socket (black-hole route) | `wait_for` timeout on `recv` | Force-close, treat as disconnect |
| Malformed JSON | `JSONDecodeError` | Count, log at DEBUG, skip frame |
| Missing/empty book side | `ValueError` from `_levels_to_array` | Skip frame; loop continues |
| Crossed book | `is_crossed()` check in `parse_l2_book` | Reject frame (data corruption) |
| Out-of-order levels | Explicit sort in `parse_l2_book` | Repair, then proceed |
| Consumer slower than producer | `QueueFull` | Drop oldest, increment counter |
| Feature/strategy exception | `try/except` in `TradingLoop._on_snapshot` | Log with traceback, continue |
| Volatility not yet estimable | `n_samples < min_samples` | Quote `reason="warmup"`, no quotes placed |
| Inventory at hard limit | Check in `Strategy.quote` | Suppress the offending side |
| SIGINT / SIGTERM | Signal handler → `asyncio.Event` | Cancel tasks, flatten, report |

The invariant across all of these: **a single bad tick never terminates the
process.** The only fatal path is a programming error in the top-level wiring.

---

## 6. Extension points

| Extension | Where | Notes |
|---|---|---|
| Additional venues | New client in `ingestion.py` emitting `BookSnapshot` | Downstream is venue-agnostic by construction |
| Trade-tape features | Subscribe `trades` channel, extend `FeatureEngine` | Enables true trade-flow imbalance vs. book-derived OFI |
| Alternative strategies | New class with a `quote(fv, inventory) -> Quote` method | `TradingLoop` depends only on that signature |
| ML fair value | Replace `_reference_price` | Feature vector is already array-serializable via `as_array()` |
| Multi-instrument | One client + engine + strategy per coin, shared simulator | Enables cross-asset inventory netting |
| Historical replay | `BookSnapshot` producer reading from Parquet | Same pipeline, deterministic input — the Phase 4 backtest path |

---

## 7. Security posture

- The system connects to **one** endpoint: a public, unauthenticated market-data
  WebSocket.
- No private keys, API secrets, or credentials exist anywhere in the codebase or
  configuration surface.
- No code path constructs, signs, or transmits an order payload.
- `.gitignore` excludes `runs/`, `.env`, and generated CSV/JSON so session
  artifacts are never committed.

Adding live trading would require a signing module, a nonce manager, a kill
switch, position reconciliation against venue state, and an independent risk
process. **None of that is in scope here, and the absence is deliberate.**
