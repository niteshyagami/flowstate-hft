# FlowState HFT

**Avellaneda-Stoikov market making on Hyperliquid — research, simulation & backtesting stack.**

Python 3.11+ · asyncio · NumPy · pandas · websockets

> ⚠️ **Simulation only.** This project connects to one public, unauthenticated
> market-data endpoint. It contains no signing, no authentication, and no code
> path that transmits an order to any venue. That boundary is deliberate and
> documented in `Rules.md` (R5.1).

**→ See [`RESULTS.md`](RESULTS.md) for the cross-asset backtest findings.**

---

## What this is

A market maker earns the spread and pays for it in two currencies: **inventory
risk** (fills arrive asymmetrically, leaving you an unwilling directional
trader) and **adverse selection** (the people who trade against your quote
disproportionately know something you don't).

FlowState streams live order books from Hyperliquid, computes microstructure
signals, generates optimal quotes from the Avellaneda-Stoikov framework, and
evaluates them in a **latency- and queue-aware** paper simulator that reports
markout alongside PnL — because a strategy with positive PnL and strongly
negative markout is not making money, it is getting lucky.

The workflow is the point: **capture live data once, replay it deterministically,
and compare parameters on identical data** — the only honest way to backtest.

---

## Headline results

A 3-hour capture of six crypto perps, backtested with a per-asset risk-aversion sweep:

| Asset | Verdict | Best γ | PnL ($) | Adverse selection | Fills (3h) |
|---|---|---|---|---|---|
| **AVAX** | ✅ Profitable | 0.10 | +2.29 | 43.3% | 71 |
| **BTC** | ✅ Profitable | 0.03 | +2.79 | 45.2% | 127 |
| **ETH** | ⚠️ Borderline | — | ≈ −1.8 | 48.5% | 104 |
| **DOGE** | ❌ Excluded | — | −0.44 | 44.7% | 274 |

Core Avellaneda-Stoikov mechanism confirmed empirically: **raising risk-aversion γ
drops adverse selection from 54% to 45% and flips PnL positive.** DOGE was excluded
after losing money across a 2000× γ range rather than overfitting to force a positive
number. Full analysis and honest caveats in [`RESULTS.md`](RESULTS.md).

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

python tests/test_invariants.py      # verify the model layer (25 tests, no network)
python -m src.ingestion BTC bbo      # stream a live top-of-book ladder
python main.py --coin BTC --duration 300 --export ./runs   # live paper session
```

---

## The full workflow

### 1. Live paper trading

```bash
python main.py --coin BTC --channel bbo --duration 300 \
  --gamma 0.03 --min-samples 5 --vol-window 60 --export runs
```

### 2. Record data (capture once, backtest forever)

Live data vanishes when the process exits, so you cannot compare parameters on it —
each run sees a different market. Record it, then replay the *same* capture.

```bash
# Single asset
python main.py --coin BTC --record captures --duration 3600

# All six majors in parallel, 3 hours, each to its own rotating file
python record_multi.py --coins BTC ETH SOL AVAX DOGE XRP --duration 10800
```

Captures are JSON-Lines + gzip (~7× compression, ~2–5 MB per hour of `bbo`),
append-only and crash-safe: a killed capture keeps every line written before the crash.

### 3. Backtest & sweep parameters

```bash
# Backtest one capture (runs in seconds, no network)
python main.py --replay captures/btc-bbo-<stamp>.jsonl.gz --gamma 0.03 --export runs

# Cross-asset comparison: backtest every capture, rank by Sharpe
python analyze.py --dir captures

# Parameter sweep: vary one param on one instrument, same data, comparable rows
python analyze.py --dir captures --coin BTC --sweep gamma --grid 0.001 0.003 0.01 0.03
```

Replay is deterministic — same file + same seed reproduces a run exactly. `analyze.py`
groups an instrument's hourly chunks into one continuous session and infers per-asset
tick/order sizes so BTC and DOGE are quoted on comparable ~$25-notional terms.

---

## Key flags

| Flag | Meaning |
|---|---|
| `--channel` | `bbo` (high-frequency top-of-book, default) or `l2Book` (full depth, slower) |
| `--record DIR` | Capture live data to DIR (no trading) |
| `--replay FILE` | Backtest against a recorded capture instead of connecting live |
| `--gamma` | Risk aversion γ. **Scale-sensitive** — see the note below |
| `--kappa` | Order-arrival elasticity κ (adapts online from fill distances) |
| `--risk-horizon` | τ in seconds: how long you expect to need to unwind |
| `--max-inventory` | Hard position limit; enforced in *both* strategy and simulator |
| `--latency-ms` | Quote placement latency — raise it to see how fragile an edge is |
| `--maker-fee-bps` | Negative for a rebate. Never set to zero and believe the result |
| `--min-samples` | Returns needed before quoting begins (lower for calm/slow books) |

**On γ:** the inventory term is `γ·σ_px²·τ`. Its natural scale depends on the asset's
price and volatility; on a live BTC book the profitable region was γ ∈ [0.01, 0.03].
Too small quotes so tight it gets picked off; too large quotes so wide nothing fills.
Always sweep γ against recorded data rather than guessing.

---

## Layout

```
flowstate_hft/
├── PRD.md              Problem statement, A-S math, target metrics, caveats
├── Architecture.md     System flow, concurrency model, latency budget
├── Rules.md            Engineering guardrails (reviewed on every PR)
├── Phases.md           4-phase plan with hard exit gates
├── Memory.md           Rolling engineering log
├── RESULTS.md          Cross-asset backtest findings
├── LINKEDIN_POST.md    Post drafts for sharing
├── main.py             Orchestrator + CLI: live / --record / --replay modes
├── record_multi.py     Record several instruments concurrently
├── analyze.py          Batch backtest: portfolio comparison + parameter sweep
├── requirements.txt
├── src/
│   ├── ingestion.py    Async WS client (l2Book + bbo), reconnect, BookSnapshot
│   ├── features.py     Micro-price, order-flow imbalance, realized volatility
│   ├── strategy.py     Avellaneda-Stoikov reservation price + optimal spread
│   ├── execution.py    Paper fill simulator, markout, performance analytics
│   └── recorder.py     Capture to JSONL+gzip; deterministic replay
└── tests/
    └── test_invariants.py   25 property tests on the model layer
```

---

## The model in two formulas

**Reservation price** — where you are indifferent between holding `q` and `q ± 1`:

```
r = s − q · γ · σ² · (T − t)
```

Long inventory pushes `r` below fair value, so both quotes shift down: your ask
gets more attractive, your bid less so. Inventory mean-reverts through *quote
placement*, never by crossing the spread.

**Optimal spread**:

```
δ_a + δ_b = γ σ² (T − t)  +  (2/γ) ln(1 + γ/κ)
            └─ inventory risk ─┘   └─ arrival-elasticity rent ─┘
```

Extensions over vanilla A-S: micro-price anchoring, OFI-based quote skew, a
cubic inventory clamp past the soft limit, a rolling risk horizon for perps, and
online κ estimation via `κ̂ = 1/EWMA(fill distance)`.

---

## Reading the output

- **Adverse-selection ratio** — fraction of fills followed by adverse drift. 50% is
  neutral; above 55% means the quotes are being picked off, regardless of PnL. This is
  the metric the whole project exists to measure.
- **Inventory variance** — how well the reservation-price mechanism controls position.
  Lower = tighter. `analyze.py` normalizes it to order-size² for cross-asset comparison.
- **Markout** — signed price drift after a fill. Positive means price moves in the
  maker's favour (the opposite of adverse selection).
- **Sharpe** is annualized from tick-frequency increments, which are not IID. Treat its
  **sign and cross-parameter ranking** as signal; treat the magnitude as an artifact.
  A Sharpe above 5 usually means the fill model is too generous — investigate first.

---

## Known limitations

Stated plainly, because this is the section a quant reviewer reads first.

1. **The fill model is a model.** True queue position depends on order-by-order history
   the L2/bbo feed does not expose. Conservative but uncalibrated.
2. **Single 3-hour window.** One capture, one regime. Not yet walk-forward validated.
3. **No microstructure alpha yet.** This is *raw* Avellaneda-Stoikov — the micro-price
   and OFI signals are computed but not yet used to actively steer quotes (Phase 2).
4. **No market impact.** True at default sizes, false at scale.
5. **Python latency.** ~90 µs tick-to-quote — fine for research, orders of magnitude off
   competitive liquidity provision. The value is the model and methodology, not speed.

---

## References

- Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book.* Quantitative Finance, 8(3).
- Cont, R., Kukanov, A. & Stoikov, S. (2014). *The price impact of order book events.* Journal of Financial Econometrics, 12(1).
- Guéant, O., Lehalle, C.-A. & Fernandez-Tapia, J. (2013). *Dealing with the inventory risk: a solution to the market making problem.* Mathematics and Financial Economics, 7(4).
- Stoikov, S. (2018). *The micro-price: a high-frequency estimator of future prices.* Quantitative Finance, 18(12).

---

## License

MIT. Provided for research and educational purposes. Nothing here is investment
advice, and no result in this repository should be relied on to deploy capital.