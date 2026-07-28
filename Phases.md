# FlowState HFT — Implementation Plan

Four phases. Each has a hard exit gate: the next phase does not begin until the
current gate is demonstrably passed.

```
Phase 1 ── Data Foundation ──────────► Phase 2 ── Signal & Model ──┐
  (in progress)                                                     │
                                                                    ▼
Phase 4 ── Validation & Portfolio ◄── Phase 3 ── Simulation & Risk ─┘
```

---

## Phase 1 — Data Foundation

**Objective:** a market-data pipeline that does not lie and does not die.

### Scope

| # | Deliverable | File | Status |
|---|---|---|---|
| 1.1 | `BookSnapshot` normalized data model | `src/ingestion.py` | ✅ Done |
| 1.2 | Async WebSocket client with `l2Book` subscription | `src/ingestion.py` | ✅ Done |
| 1.3 | Exponential backoff + full jitter reconnection | `src/ingestion.py` | ✅ Done |
| 1.4 | Staleness watchdog (silent-socket detection) | `src/ingestion.py` | ✅ Done |
| 1.5 | Bounded queue with drop-oldest backpressure | `src/ingestion.py` | ✅ Done |
| 1.6 | Top-5 ladder renderer | `src/ingestion.py` | ✅ Done |
| 1.7 | Ordering invariants + crossed-book rejection | `src/ingestion.py` | ✅ Done |
| 1.8 | 60-minute soak test, zero unhandled exceptions | — | ⬜ Pending |
| 1.9 | Forced-drop recovery test | — | ⬜ Pending |
| 1.10 | Parquet recorder for offline replay | `src/recorder.py` | ⬜ Pending |

### Exit gate

- 60 minutes of continuous live data, zero unhandled exceptions
- Recovery from a deliberately severed connection, verified in logs
- Parse-error rate < 0.1% of received frames
- Latency proxy (`local_ts − exch_ts`) logged and within plausible bounds
- Recorded session replayable and byte-identical to the live parse

### Known risks

| Risk | Mitigation |
|---|---|
| Hyperliquid changes the `l2Book` schema | Structural validation in `parse_l2_book`; frames rejected loudly, not silently misparsed |
| Clock skew corrupts the latency proxy | Documented as a *relative* measure; watch for changes, not absolute values |
| Sustained high update rate saturates the consumer | Drop counter surfaces it immediately; drop-oldest keeps quotes current |

---

## Phase 2 — Signal & Model

**Objective:** turn raw books into a fair-value estimate and an optimal quote.

### Scope

| # | Deliverable | File | Status |
|---|---|---|---|
| 2.1 | Level-1 micro-price | `src/features.py` | ✅ Done |
| 2.2 | Depth-weighted micro-price with geometric decay | `src/features.py` | ✅ Done |
| 2.3 | Order Flow Imbalance (Cont-Kukanov-Stoikov) | `src/features.py` | ✅ Done |
| 2.4 | Time-aware EWMA smoothing and OFI normalization | `src/features.py` | ✅ Done |
| 2.5 | Realized volatility, relative and absolute units | `src/features.py` | ✅ Done |
| 2.6 | A-S reservation price with cubic inventory clamp | `src/strategy.py` | ✅ Done |
| 2.7 | A-S optimal spread with tick and bps guardrails | `src/strategy.py` | ✅ Done |
| 2.8 | OFI-based quote skew | `src/strategy.py` | ✅ Done |
| 2.9 | Online κ estimation from fill distances | `src/strategy.py` | ✅ Done |
| 2.10 | Feature predictive-power study (OFI → forward return) | `research/` | ⬜ Pending |
| 2.11 | Volatility estimator comparison (realized vs. Garman-Klass vs. EWMA) | `research/` | ⬜ Pending |

### Exit gate

- OFI shows statistically significant correlation with forward mid returns at a
  1-10 second horizon (target: `|IC| > 0.05`, `t > 3`) on recorded data
- Micro-price beats mid as a forward-value predictor, measured by RMSE
- Reservation price demonstrably skews in the correct direction under a synthetic
  inventory path (unit test)
- Optimal spread responds monotonically to σ, γ, and κ (unit test)
- Volatility estimator stable — no explosion during a live volatility spike

### Open research questions

1. What decay constant for the depth-weighted micro-price maximizes forward-return
   predictive power on Hyperliquid specifically? The default `0.5` is a prior, not
   a result.
2. Does OFI computed from *book* changes match OFI computed from the *trade tape*?
   They diverge in the presence of heavy quote flickering, and the tape version is
   generally the better signal — Phase 2.10 should measure the gap.
3. Is a single volatility timescale sufficient, or does the model need fast/slow
   blending to avoid over-widening after transient spikes?

---

## Phase 3 — Simulation & Risk

**Objective:** measure honestly. This is the phase where most projects deceive
themselves, and the one this project exists to get right.

### Scope

| # | Deliverable | File | Status |
|---|---|---|---|
| 3.1 | Two-stage fill model (crossing + queue/penetration) | `src/execution.py` | ✅ Done |
| 3.2 | Latency-gated quote placement | `src/execution.py` | ✅ Done |
| 3.3 | Cash / inventory / fee / gross-edge accounting | `src/execution.py` | ✅ Done |
| 3.4 | Markout computation at a configurable horizon | `src/execution.py` | ✅ Done |
| 3.5 | Hard inventory limits with fill rejection | `src/execution.py` | ✅ Done |
| 3.6 | Sharpe, max drawdown, inventory variance | `src/execution.py` | ✅ Done |
| 3.7 | Adverse-selection ratio | `src/execution.py` | ✅ Done |
| 3.8 | Session flattening at teardown | `src/execution.py` | ✅ Done |
| 3.9 | CSV/JSON artifact export | `main.py` | ✅ Done |
| 3.10 | Fill-model calibration against a recorded trade tape | `research/` | ⬜ Pending |
| 3.11 | Markout curve across horizons (0.1s → 60s) | `research/` | ⬜ Pending |
| 3.12 | Volatility-regime-conditioned PnL attribution | `research/` | ⬜ Pending |

### Exit gate

- Simulated fill rate within 2× of the rate implied by the recorded trade tape at
  comparable quote distances
- Markout curve computed and interpreted — the *shape* matters more than the level
- PnL decomposed into gross edge, fees, adverse selection, and inventory MTM, with
  the four components summing to total PnL to within floating-point tolerance
- Inventory variance below 25% of `q_max²` across at least three sessions
- No session where a single fill accounts for more than 10% of total PnL

### The honesty requirement

Phase 3's deliverable is **not** a good Sharpe ratio. It is a *believable* one.
Specific failure patterns to actively hunt for:

- Sharpe > 5 in simulation almost always means the fill model is too generous.
  Investigate before celebrating.
- Positive PnL with strongly negative markout means the strategy is being adversely
  selected and is only profitable because inventory happened to mark favourably.
  That is luck, and it will reverse.
- PnL concentrated in a handful of fills means the result is a small-sample artifact.
- A strategy that is unprofitable once realistic fees are applied is unprofitable.
  There is no reframing that changes this.

---

## Phase 4 — Validation & Portfolio

**Objective:** establish whether anything found in Phases 2-3 survives contact
with out-of-sample data.

### Scope

| # | Deliverable | File | Status |
|---|---|---|---|
| 4.1 | Deterministic historical replay from Parquet | `src/replay.py` | ⬜ Pending |
| 4.2 | Walk-forward harness (rolling train/test split) | `src/validation.py` | ⬜ Pending |
| 4.3 | Parameter sweep over (γ, κ, τ, micro_weight) | `src/validation.py` | ⬜ Pending |
| 4.4 | Parameter-stability surface plots | `research/` | ⬜ Pending |
| 4.5 | Multi-instrument concurrent quoting | `main.py` | ⬜ Pending |
| 4.6 | Cross-asset inventory netting | `src/execution.py` | ⬜ Pending |
| 4.7 | Regime detection and parameter switching | `src/strategy.py` | ⬜ Pending |
| 4.8 | Performance tearsheet generator | `src/reporting.py` | ⬜ Pending |
| 4.9 | Latency profiling harness | `bench/` | ⬜ Pending |
| 4.10 | Unit + property test suite, ≥80% coverage | `tests/` | ⬜ Pending |

### Exit gate

- Out-of-sample Sharpe within 50% of in-sample. A larger gap means the parameters
  are fitted to noise.
- Parameter surface is a **plateau, not a spike**. If performance collapses when γ
  moves 20%, the result is not real.
- Results hold across at least two instruments and two volatility regimes
- Tearsheet reproducible from a recorded session and a config file alone
- Test suite green; property tests confirm the model's invariants (reservation price
  monotone in `q`, spread monotone in σ)

### Deliverables for external review

1. **Tearsheet** — equity curve, inventory path, markout distribution, drawdown
   profile, fill scatter overlaid on the book.
2. **Parameter stability report** — heatmaps over (γ, τ) showing where the strategy
   is robust versus where it is fitted.
3. **Written limitations section** — what the simulator cannot capture, stated
   plainly. This is the section a quant reviewer reads first, and its absence is
   the fastest way to lose credibility.

---

## Timeline

| Phase | Estimated effort | Dependency |
|---|---|---|
| 1 — Data Foundation | 1 week | — |
| 2 — Signal & Model | 2 weeks | Phase 1 recorded data |
| 3 — Simulation & Risk | 2 weeks | Phase 2 signals |
| 4 — Validation & Portfolio | 3 weeks | Phase 3 simulator |

Estimates assume part-time work and no venue-side surprises. Phase 4 is the one
that reliably overruns, because it is the phase where you find out the earlier
phases were wrong.
