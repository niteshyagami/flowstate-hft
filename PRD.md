# FlowState HFT — Product Requirements Document

**Version:** 0.1.0
**Status:** Phase 1 in progress
**Owner:** Quantitative Research
**Target venue:** Hyperliquid (decentralized perpetuals)
**Scope:** Research and simulation only — no live order transmission

---

## 1. Executive summary

FlowState HFT is a research-grade market-making stack that streams live L2 order
book data from Hyperliquid, computes microstructure features, generates optimal
two-sided quotes using the Avellaneda-Stoikov (A-S) framework, and evaluates
those quotes in a latency- and queue-aware paper-trading simulator.

The purpose of the system is not to make money. It is to **measure**, honestly,
whether a passive quoting policy earns more from the spread than it loses to
adverse selection and inventory risk — and to produce the diagnostic artifacts
(markout curves, inventory paths, drawdown profiles) that make that question
answerable.

---

## 2. Problem statement

### 2.1 The market maker's dilemma

A market maker posts a bid at `p_b` and an ask at `p_a` around a fair value `s`,
earning the spread `p_a − p_b` on every round trip. Two forces work against this:

1. **Inventory risk.** Fills arrive asymmetrically. A maker who is repeatedly hit
   on the bid accumulates a long position and is now a directional trader who
   never chose to be one. Unhedged inventory turns a spread-capture business
   into a delta-exposure business, and the variance of that exposure dominates
   PnL variance at any meaningful size.

2. **Adverse selection.** This is the core problem. The counterparties who trade
   against a resting quote are not random. They are, disproportionately, the ones
   who know something. A maker's bid gets lifted precisely when the price is
   about to fall.

### 2.2 Quantifying adverse selection

Adverse selection is measurable through **markout** — the drift of the mid price
after a fill, signed in the maker's direction:

```
markout(h) = side · (mid[t + h] − mid[t]) · size,   side = +1 for buys
```

A maker capturing half a tick of spread while suffering one tick of negative
5-second markout is losing money and will not notice for weeks if they only
track realized PnL. FlowState makes markout a first-class metric, reported
alongside PnL in every session summary.

The decomposition the system targets:

```
Total PnL  =  Gross spread captured  −  Fees  −  Adverse selection  ±  Inventory MTM
```

Each term is reported separately. A strategy that shows positive total PnL
driven entirely by a lucky inventory mark is not a market-making strategy, and
the report will say so.

### 2.3 Why Hyperliquid

- **Public, unauthenticated market data.** The `l2Book` WebSocket channel is
  open, so research requires no API keys and carries no key-leak risk.
- **On-chain order book.** Unlike AMM venues, Hyperliquid runs a genuine CLOB,
  so classical limit-order-book theory applies directly.
- **Meaningful maker economics.** Maker fee tiers and rebates make the passive
  side economically distinct from the aggressive side, which is the precondition
  for a market-making strategy to exist at all.

---

## 3. Solution: Avellaneda-Stoikov

### 3.1 Model setup

The efficient mid price follows arithmetic Brownian motion:

```
dS_t = σ dW_t
```

The maker holds inventory `q`, has CARA utility with risk aversion `γ`, and
faces order arrivals with intensity exponential in quote distance:

```
λ(δ) = A · exp(−κ · δ)
```

Solving the resulting Hamilton-Jacobi-Bellman equation yields two closed-form
quantities.

### 3.2 Reservation price

```
r(s, q, t) = s − q · γ · σ² · (T − t)
```

This is the price at which the maker is indifferent between holding `q` and
holding `q ± 1`. It is the entire inventory risk engine in one term:

| State | Effect |
|---|---|
| `q = 0` | `r = s`. Quotes are symmetric around fair value. |
| `q > 0` (long) | `r < s`. Both quotes shift **down** — the ask becomes more attractive, the bid less so. Inventory bleeds off passively. |
| `q < 0` (short) | `r > s`. Both quotes shift **up**. |

Crucially, the maker never crosses the spread to flatten. Inventory mean-reverts
through *quote placement*, not through taker fees.

### 3.3 Optimal spread

```
δ_a + δ_b = γ σ² (T − t) + (2/γ) · ln(1 + γ/κ)
```

Two economically distinct terms:

- `γ σ² (T − t)` — **inventory risk premium**. Widens with volatility, risk
  aversion, and remaining horizon.
- `(2/γ) ln(1 + γ/κ)` — **monopolistic rent**. Determined by how elastic order
  arrivals are to quote distance. Independent of inventory.

Quotes are then placed symmetrically about the reservation price:

```
p_bid = r − (δ_a + δ_b)/2
p_ask = r + (δ_a + δ_b)/2
```

### 3.4 Extensions beyond vanilla A-S

Vanilla A-S is a beautiful model with known gaps. FlowState addresses four:

| Gap | Extension | Rationale |
|---|---|---|
| Mid price is a biased fair-value estimator when the book is lopsided | **Micro-price anchoring**: substitute `s → w·p_micro + (1−w)·mid` | Direct adverse-selection reduction — the largest single improvement in practice |
| Model is blind to aggressive flow | **OFI skew**: shift quotes by `β · OFI_norm · s` | Step away from the side being run over |
| Linear inventory penalty is too gentle near limits | **Cubic soft clamp** past `soft_ratio · q_max` | Marginal cost of inventory is genuinely convex near the limit |
| Fixed terminal time `T` is meaningless for perps | **Rolling horizon** `τ` (Guéant-Lehalle-Fernandez-Tapia stationary limit) | Perps never close |

### 3.5 Estimating κ online

Under `λ(δ) = A·exp(−κδ)`, the MLE of `κ` from observed fill distances is the
reciprocal of their mean: `κ̂ = 1 / mean(δ_filled)`. FlowState maintains this
mean as a time-decayed EWMA so `κ` tracks regime shifts rather than averaging
over the whole session.

---

## 4. Functional requirements

| ID | Requirement | Priority |
|---|---|---|
| FR-1 | Stream `l2Book` for a configurable coin over WebSocket | P0 |
| FR-2 | Reconnect automatically with exponential backoff + full jitter | P0 |
| FR-3 | Detect and recycle silent sockets via a staleness watchdog | P0 |
| FR-4 | Normalize book frames into NumPy arrays with enforced ordering invariants | P0 |
| FR-5 | Compute micro-price (L1 and depth-weighted) | P0 |
| FR-6 | Compute Order Flow Imbalance and a time-aware EWMA of it | P0 |
| FR-7 | Estimate per-second realized volatility in absolute price units | P0 |
| FR-8 | Compute A-S reservation price and optimal spread each tick | P0 |
| FR-9 | Enforce hard and soft inventory limits, suppressing quotes at the limit | P0 |
| FR-10 | Snap all quotes to the venue tick grid, conservatively | P0 |
| FR-11 | Simulate fills with queue-position and penetration modelling | P0 |
| FR-12 | Model quote placement latency | P0 |
| FR-13 | Track cash, inventory, fees, gross edge, and mark-to-market equity | P0 |
| FR-14 | Compute markout at a configurable horizon for every fill | P0 |
| FR-15 | Report Sharpe, max drawdown, inventory variance, adverse-selection ratio | P0 |
| FR-16 | Export fills, equity curve, and summary as CSV/JSON | P1 |
| FR-17 | Adapt κ online from realized fill distances | P1 |
| FR-18 | Graceful SIGINT/SIGTERM shutdown with position flattening | P1 |
| FR-19 | Walk-forward parameter sweep harness | P2 (Phase 4) |
| FR-20 | Multi-instrument concurrent quoting | P2 (Phase 4) |

---

## 5. Non-functional requirements

| ID | Requirement | Target |
|---|---|---|
| NFR-1 | Tick-to-quote latency (feature + strategy, excluding I/O) | p99 < 500 µs |
| NFR-2 | Sustained throughput | ≥ 500 book updates/sec single-instrument |
| NFR-3 | Memory footprint stable over a multi-hour session | Bounded ring buffers only, no unbounded growth |
| NFR-4 | Uptime across network partitions | Auto-recovery within 30 s worst case |
| NFR-5 | Reproducibility | Fixed RNG seed reproduces a session bit-for-bit given identical input |
| NFR-6 | Python version | 3.11+ (uses `dataclass(slots=True)` and modern typing) |
| NFR-7 | Zero live-trading surface | No signing, no auth, no private endpoints, anywhere in the codebase |

---

## 6. Target metrics

These are the numbers the project is evaluated on. They are stated as targets,
not as claims — the entire point of the simulator is to find out whether they
are achievable.

### 6.1 Primary

| Metric | Definition | Target |
|---|---|---|
| **Sharpe ratio** | `mean(ΔEquity) / std(ΔEquity) · √(365·86400/Δt)` | > 2.0 annualized, net of fees, on out-of-sample sessions |
| **Max drawdown** | `max_t( max_{s≤t} E_s − E_t )` | < 2× the mean daily PnL |
| **Inventory variance** | `Var(q_t)` over the session | < 25% of `q_max²` — the direct test of whether the reservation-price mechanism works |

### 6.2 Secondary

| Metric | Definition | Target |
|---|---|---|
| Adverse-selection ratio | Fraction of fills with negative markout | < 55% (50% is neutral; above 55% means the quoting policy is being picked off) |
| Mean markout (5s) | Average signed post-fill drift | ≥ 0 |
| Fill rate | Fills per minute | 2-20/min — too low means quotes are irrelevant, too high means they are mispriced |
| Inventory half-life | Time for `|q|` to decay by half after a shock | < 5× the risk horizon τ |
| Quote uptime | Fraction of ticks with a two-sided quote live | > 90% |

### 6.3 Explicit non-goals

- Beating any specific benchmark return.
- Live capital deployment. There is no execution path to a venue and none is
  planned within this repository's scope.
- Latency competition with co-located professional makers. A Python + WebSocket
  stack is 2-3 orders of magnitude off that pace, and pretending otherwise would
  be dishonest. The research value is in the *model*, not the plumbing speed.

### 6.4 Honest caveats

A simulator's Sharpe is an upper bound, not a forecast. Specifically:

- **The fill model is a model.** Real queue position depends on order-by-order
  history the L2 feed does not expose. Our estimate is deliberately conservative
  but remains an estimate.
- **No market impact.** Quotes are assumed not to change the behaviour of other
  participants. At small size this is close to true; at scale it is false.
- **Survivorship in parameter choice.** Any parameter set tuned on a session and
  evaluated on that same session will look good. Phase 4 walk-forward validation
  exists specifically to break this, and results before Phase 4 should be read
  as diagnostic, not predictive.

---

## 7. Success criteria for Phase 1

- [ ] Live `l2Book` stream sustained for 60 minutes with zero unhandled exceptions
- [ ] Automatic recovery demonstrated from a forced network drop
- [ ] Top-5 ladder rendered correctly with ordering invariants enforced
- [ ] Latency proxy (`local_ts − exch_ts`) logged and within a plausible range
- [ ] Parse-error rate below 0.1% of received frames
