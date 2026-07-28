# FlowState HFT — Project Memory

Rolling engineering log. Update on every meaningful change (see Rules R8 checklist).
Newest entries at the top of each section.

---

## Current state

| Field | Value |
|---|---|
| **Active phase** | **Phase 1 — Data Foundation · IN PROGRESS** |
| Version | 0.1.0 |
| Last updated | Session 1 — initial scaffold |
| Blocking item | 1.8 — 60-minute live soak test not yet run |
| Next action | Run `python main.py --coin BTC --duration 3600 --show-book` and inspect logs for reconnects, parse errors, and drop counts |

### Phase progress

```
Phase 1 · Data Foundation      ███████░░░  70%   IN PROGRESS
Phase 2 · Signal & Model       ███████░░░  70%   code complete, unvalidated
Phase 3 · Simulation & Risk    ███████░░░  70%   code complete, uncalibrated
Phase 4 · Validation           ░░░░░░░░░░   0%   not started
```

> Phases 2 and 3 show code-complete but are **gated behind Phase 1's exit
> criteria**. Code existing is not the same as code being validated, and no
> number produced by Phases 2-3 should be quoted until Phase 1 passes its gate.

---

## Component status

| Component | File | State | Validated? |
|---|---|---|---|
| Book data model | `src/ingestion.py` | Implemented | ⬜ |
| WebSocket client + reconnect | `src/ingestion.py` | Implemented | ⬜ Soak test pending |
| Micro-price (L1 + weighted) | `src/features.py` | Implemented | ⬜ Predictive power unmeasured |
| Order Flow Imbalance | `src/features.py` | Implemented | ⬜ IC unmeasured |
| Realized volatility | `src/features.py` | Implemented | ⬜ Stability under spike untested |
| A-S reservation price | `src/strategy.py` | Implemented | ⬜ Unit tests pending |
| A-S optimal spread | `src/strategy.py` | Implemented | ⬜ Unit tests pending |
| Adaptive κ | `src/strategy.py` | Implemented | ⬜ Convergence unverified |
| Fill simulator | `src/execution.py` | Implemented | ⬜ **Uncalibrated — treat output as indicative only** |
| Performance analytics | `src/execution.py` | Implemented | ⬜ |
| Orchestrator | `main.py` | Implemented | ⬜ |
| Parquet recorder | `src/recorder.py` | Not started | — |
| Walk-forward harness | `src/validation.py` | Not started | — |
| Test suite | `tests/` | Not started | — |

---

## Decision log

| # | Decision | Rationale | Revisit when |
|---|---|---|---|
| D-1 | `float64` over `Decimal` | ~40× faster; 15-16 sig digits vs. crypto's ~10. Correct for signals, wrong for a settlement ledger. | Never, within this scope |
| D-2 | Drop **oldest** frame under backpressure | A stale book is worse than no book for a quoter | Never |
| D-3 | Pipeline stages run synchronously inside one task | Stages are µs-scale pure CPU; awaits would add scheduler latency and open a state-consistency bug window | If a stage ever does real I/O |
| D-4 | Rolling risk horizon τ, not terminal T | Perps never close; τ = expected unwind time is the meaningful quantity | If a session-based product is added |
| D-5 | Cubic inventory clamp past the soft limit | Vanilla linear A-S is too gentle near `q_max`, where marginal inventory cost is genuinely convex | After Phase 3 inventory-variance data |
| D-6 | Conservative two-stage fill model | Optimistic fill assumptions are the single largest source of fake backtest Sharpe | Phase 3.10 calibration against the trade tape |
| D-7 | Fees modelled by default, never zeroed | A strategy that only works at zero fees does not work | Never |
| D-8 | Strategy does not own inventory | Keeps it pure and unit-testable against synthetic inventory paths | Never |
| D-9 | Zero live-trading surface | Research scope; live trading needs signing, nonce management, kill switch, reconciliation, and an independent risk process — none of which is in scope | Out of scope for this repository |

---

## Known issues and open questions

### Open

- **OQ-1** — Default `gamma=0.01` is a placeholder. The correct scale depends on
  `σ_price²·τ` for the instrument; on BTC these units make γ's natural range quite
  different from the ETH range. Needs a proper sweep in Phase 4.
- **OQ-2** — `queue_depth_scale=2.0` in `SimConfig` is a guess. Should be
  calibrated against typical top-of-book depth per instrument.
- **OQ-3** — OFI is computed from *book* changes, not the trade tape. These
  diverge under heavy quote flickering, and the tape version is generally the
  better signal. Phase 2.10 should measure the gap.
- **OQ-4** — Volatility uses a single timescale. A transient spike over-widens
  quotes for the full window length. Fast/slow blending may fix this.
- **OQ-5** — Latency proxy is contaminated by clock skew. Useful for detecting
  *changes*, not absolute values. NTP sync would narrow but not eliminate this.
- **OQ-6** — No market impact modelling. Fine at the default sizes; false at scale.

### Resolved

*(none yet)*

---

## Session log

### Session 1 — initial scaffold

**Done**

- Repository structure, docs (PRD, Architecture, Rules, Phases, Memory)
- `src/ingestion.py` — `BookSnapshot`, `HyperliquidL2Client`, reconnect, watchdog,
  ladder renderer
- `src/features.py` — micro-price, weighted micro-price, OFI, EWMA smoothing,
  realized volatility, `FeatureEngine`
- `src/strategy.py` — `ASParams`, `Quote`, `AvellanedaStoikovStrategy` with
  reservation price, optimal spread, OFI skew, cubic clamp, adaptive κ
- `src/execution.py` — `PaperExecutionSimulator`, two-stage fill model, markout,
  full performance analytics
- `main.py` — orchestrator, CLI, signal handling, artifact export

**Learned**

- Enforcing the bid-descending / ask-ascending invariant *at parse time* rather
  than assuming it removes an entire bug class from every downstream consumer.
- Carrying both `sigma` and `sigma_price` as separately-named fields is worth the
  small redundancy: A-S needs absolute price units, and passing the relative
  figure produces plausible-looking but meaningless spreads — a silent unit bug.

**Not done / deferred**

- Live soak test (Phase 1 exit gate)
- Parquet recorder
- Any empirical validation whatsoever

**Next session**

1. Run the 60-minute soak; record reconnects, parse errors, drops
2. Implement `src/recorder.py`
3. Record 2-3 hours of BTC book data for offline feature research
4. Begin Phase 2.10 — measure OFI's information coefficient against forward returns

---

## Template for new entries

```markdown
### Session N — <title>

**Done**
-

**Learned**
-

**Not done / deferred**
-

**Next session**
1.
```
