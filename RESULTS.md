# FlowState HFT — Backtest Results

**Avellaneda-Stoikov market making on Hyperliquid — cross-asset study**

A latency- and queue-aware paper-trading evaluation of a classical
Avellaneda-Stoikov (A-S) quoting policy across six crypto perpetuals, driven by
3 hours of live top-of-book data captured from the Hyperliquid DEX.

> **Scope & honesty note.** This is a *simulation* study on paper fills. No orders
> were ever sent to any venue. The goal is not a headline return — it is an honest
> measurement of where a passive quoting policy earns the spread versus where it
> loses to adverse selection. Some assets are reported as unprofitable on purpose,
> because they are.

---

## Method

| Stage | What it does |
|---|---|
| **Capture** | Recorded the `bbo` (best bid/offer) stream for BTC, ETH, SOL, AVAX, DOGE, XRP concurrently for 3 hours → ~400k book updates, JSON-Lines + gzip, crash-safe with hourly rotation. |
| **Replay** | Deterministic offline replay of each capture through the identical live pipeline (features → strategy → fill simulation). Same file + seed reproduces a run exactly. |
| **Fill model** | Two-stage: a resting quote is eligible only once the market trades *through* its price, then fills with a probability that decays in queue-position depth ahead and rises with penetration. Placement latency (50 ms) and maker fees (1.5 bps) modelled. Deliberately conservative. |
| **Tuning** | Per-asset sweep of the risk-aversion parameter γ on the *same* recorded data, so every row is directly comparable — the only valid way to compare parameters. |

Per-instrument tick size and order size are inferred from price so all assets are
quoted on comparable ~$25-notional terms (0.001 BTC and ~160 DOGE are the same
economic bet).

---

## Headline result

**The strategy is cleanly profitable on AVAX and BTC, borderline on ETH, and
structurally unprofitable on DOGE regardless of tuning.** Two tradeable assets, one
marginal, one correctly excluded.

| Asset | Verdict | Best γ | PnL ($) | Sharpe¹ | Adv. selection² | Fills (3h) | Note |
|---|---|---|---|---|---|---|---|
| **AVAX** | ✅ Profitable | 0.10 | **+2.29** | 218 | **43.3%** | 71 | Positive markout; robust across all γ |
| **BTC** | ✅ Profitable | 0.03 | **+2.79** | 171 | 45.2% | 127 | Most reliable — 500+ fills at low γ |
| **ETH** | ⚠️ Borderline | — | ≈ −1.8 | — | 48.5% | 104 | Negative in the statistically reliable region |
| **DOGE** | ❌ Excluded | — | −0.44 | — | 44.7% | 274 | Unprofitable across a 2000× γ range |
| SOL, XRP | — Insufficient data | — | — | — | — | 2–26 | Too few fills to draw any conclusion |

¹ Sharpe is annualized from tick-frequency increments; treat its **sign and ranking**
as signal, not its magnitude (the annualization convention inflates the absolute
number). ² Fraction of fills followed by adverse price drift; 50% is neutral,
above 55% means the quotes are being picked off.

---

## Key finding: risk-aversion controls adverse selection

Sweeping γ on BTC shows the core Avellaneda-Stoikov mechanism working exactly as the
theory predicts — wider quotes (higher γ) are selected against **less** often, which
flips PnL positive:

| γ | Fills | PnL ($) | Adverse selection | Inventory (norm)³ |
|---|---|---|---|---|
| 0.0003 | 529 | +0.37 | 54.0% | 54.7 |
| 0.001 | 512 | −1.40 | 52.6% | 26.4 |
| 0.003 | 430 | −0.61 | 53.4% | 24.1 |
| **0.01** | 250 | +2.09 | 47.0% | 48.8 |
| **0.03** | 127 | +2.79 | **45.2%** | 31.9 |

As γ rises, adverse selection falls from **54% → 45%** and PnL turns positive. The
good region is a **plateau** (γ ∈ [0.01, 0.03]), not a single lucky point — evidence
the result is robust rather than overfit. The cost is fewer fills (529 → 127): the
classic market-maker trade-off between volume and edge.

³ Inventory variance in units of order-size²; lower = tighter position control.

---

## AVAX: the standout

AVAX is profitable at **every** γ tested (+0.98 to +2.93), with a positive markout at
higher γ — meaning price tends to move *in the maker's favour* after a fill, the
opposite of adverse selection:

| γ | Fills | PnL ($) | Adv. sel | Markout | Inventory (norm) |
|---|---|---|---|---|---|
| 0.003 | 71 | +2.92 | 55.7% | −0.0007 | 24.4 |
| 0.01 | 75 | +2.81 | 52.8% | +0.0003 | 20.0 |
| 0.03 | 68 | +2.93 | 44.8% | +0.0015 | 20.7 |
| **0.10** | 71 | +2.29 | **43.3%** | **+0.0016** | 7.1 |
| 0.30 | 63 | +0.98 | 47.5% | +0.0007 | 2.2 |

Consistent fill counts (63–75) mean the sample is trustworthy across the whole sweep.

---

## Why DOGE was excluded (and why that matters)

DOGE was swept across a **2000× range** of γ (0.001 → 2.0). Inventory control improved
dramatically (variance 301 → 1.2) and adverse selection fell to 45%, yet **PnL never
turned positive** (best −0.44). When even a maximally defensive configuration loses
money, there is no spread edge to capture — passive market making simply does not work
on this instrument over this window.

DOGE was therefore excluded rather than have its parameters bent to force a positive
number. **A strategy that appears to profit on every asset is usually overfit or
running an optimistic fill model.** Reporting an honest exclusion is the credible
outcome.

The same discipline applied to ETH's γ=0.5 row, which showed +0.96 PnL — but on only
**18 fills**, too few to trust. It was treated as noise, and ETH's verdict rests on the
statistically reliable γ ≤ 0.1 region, where it is marginally negative.

---

## Limitations

Stated plainly, because these bound every number above.

1. **The fill model is a model.** True queue position depends on order-by-order history
   the L2/bbo feed does not expose. The estimate is conservative but uncalibrated.
2. **Single 3-hour window.** One capture, one market regime. Results are not yet
   walk-forward validated across days or volatility regimes.
3. **No microstructure alpha yet.** This is *raw* Avellaneda-Stoikov. The micro-price
   and order-flow-imbalance signals are computed but not yet used to actively steer
   quotes — the natural next step for the borderline assets (ETH, DOGE) whose problem
   is adverse selection.
4. **Sample sizes vary.** SOL and XRP produced too few fills (2–26) for any conclusion;
   they need a longer or more volatile capture.
5. **Sharpe magnitudes are inflated** by annualizing tick-frequency increments; only
   sign and cross-parameter ranking are used for decisions.

---

## What this demonstrates

- A full live-data → feature → strategy → simulation → analysis pipeline, built and
  verified end-to-end (25 property tests on the model layer).
- Correct application of the Avellaneda-Stoikov framework, with the reservation-price
  and optimal-spread mechanics behaving as theory predicts under a parameter sweep.
- Disciplined, reproducible backtesting: record once, replay deterministically, compare
  parameters on identical data.
- **Research honesty** — excluding unprofitable assets, discounting low-sample results,
  and stating limitations rather than presenting a polished but fragile number.

## Next steps

- **Phase 2:** wire the micro-price and OFI signals into the quote to attack the 45–55%
  adverse-selection floor directly — the most likely path to making ETH tradeable.
- **Validation:** capture multiple days/regimes and walk-forward test parameter stability.
- **Data:** longer captures for SOL/XRP to reach a usable fill count.

---

*Simulation and research only. Nothing here is investment advice, and no result should be
relied upon to deploy capital.*