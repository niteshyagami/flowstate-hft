"""
research_signals.py
===================

Phase 2 pre-work: measure whether the microstructure signals actually predict
short-horizon price moves, *before* wiring them into the quoter.

The discipline here matters. It is tempting to skip straight to "add OFI skew to
the strategy and see if PnL improves." But if OFI has no predictive power on this
instrument, any PnL change is noise, and tuning against noise is how you overfit.
So first we answer one question per signal: **does it lead price?**

Information Coefficient (IC)
----------------------------
For a signal :math:`s_t` and the forward mid return over horizon :math:`h`,
:math:`r_{t\\to t+h} = \\ln(m_{t+h} / m_t)`, the IC is their correlation:

.. math:: \\mathrm{IC}(h) = \\mathrm{corr}(s_t,\\; r_{t \\to t+h})

Interpretation, roughly, for high-frequency signals:

* :math:`|\\mathrm{IC}| < 0.02` — no usable edge.
* :math:`0.02 \\le |\\mathrm{IC}| < 0.05` — weak but potentially tradeable.
* :math:`|\\mathrm{IC}| \\ge 0.05` — strong for this domain.

We also report the t-statistic :math:`t = \\mathrm{IC}\\sqrt{n-2}/\\sqrt{1-\\mathrm{IC}^2}`;
with tens of thousands of samples even a tiny IC is "significant", so the
**magnitude** matters more than significance here.

Signals tested
--------------
* **OFI EWMA** — smoothed order-flow imbalance. The headline candidate.
* **Book imbalance − 0.5** — top-of-book size skew.
* **Micro-price premium** — :math:`(p^{micro} - m)/m`, the micro-price's own
  deviation from mid, which is mechanically a one-step-ahead predictor.
* **Trend** — the EWMA drift estimate (a momentum sanity check).

Horizons are measured in **events** (book updates), not seconds, since that is
the natural clock for a quoting decision.

Usage
-----
::

    python research_signals.py --dir captures --coin BTC
    python research_signals.py --dir captures --coin AVAX --horizons 1 5 10 30 60
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
from pathlib import Path

import numpy as np

from src.features import FeatureEngine, book_imbalance
from src.recorder import replay_file

logger = logging.getLogger("research")


def _coin_from_filename(path: str) -> str:
    return Path(path).name.split("-")[0].upper()


def collect_signals(files: list[str]) -> dict[str, np.ndarray]:
    """Replay captures and collect aligned signal + mid arrays.

    Returns a dict of equal-length float64 arrays: ``mid`` plus one column per
    signal. Warmup frames (before the feature engine is ready) are skipped so
    volatility-dependent fields are valid.
    """
    fe = FeatureEngine(min_samples=5, vol_window=120)
    mids: list[float] = []
    ofi_ewma: list[float] = []
    imb: list[float] = []
    micro_prem: list[float] = []
    trend: list[float] = []

    for path in sorted(files):
        for snap in replay_file(path):
            fv = fe.update(snap)
            if fv is None:
                continue
            mids.append(fv.mid)
            ofi_ewma.append(fv.ofi_ewma)
            imb.append(fv.imbalance - 0.5)
            micro_prem.append((fv.micro - fv.mid) / fv.mid if fv.mid > 0 else 0.0)
            trend.append(fv.trend)

    return {
        "mid": np.asarray(mids, dtype=np.float64),
        "ofi_ewma": np.asarray(ofi_ewma, dtype=np.float64),
        "imbalance": np.asarray(imb, dtype=np.float64),
        "micro_premium": np.asarray(micro_prem, dtype=np.float64),
        "trend": np.asarray(trend, dtype=np.float64),
    }


def forward_returns(mid: np.ndarray, horizon: int) -> np.ndarray:
    """Log return from each event to ``horizon`` events later (NaN-padded tail)."""
    out = np.full(mid.shape, np.nan, dtype=np.float64)
    if horizon < mid.size:
        with np.errstate(divide="ignore", invalid="ignore"):
            out[:-horizon] = np.log(mid[horizon:] / mid[:-horizon])
    return out


def information_coefficient(signal: np.ndarray, fwd_ret: np.ndarray) -> tuple[float, float, int]:
    """Return (IC, t-stat, n) over the finite, non-degenerate overlap."""
    mask = np.isfinite(signal) & np.isfinite(fwd_ret)
    s, r = signal[mask], fwd_ret[mask]
    n = s.size
    if n < 30 or np.std(s) < 1e-15 or np.std(r) < 1e-15:
        return 0.0, 0.0, n
    ic = float(np.corrcoef(s, r)[0, 1])
    ic = max(min(ic, 1.0), -1.0)
    t = ic * np.sqrt(max(n - 2, 1)) / np.sqrt(max(1.0 - ic * ic, 1e-12))
    return ic, float(t), n


def rating(abs_ic: float) -> str:
    """Human label for an IC magnitude."""
    if abs_ic >= 0.05:
        return "STRONG"
    if abs_ic >= 0.02:
        return "weak"
    return "none"


def analyze(files: list[str], horizons: list[int]) -> None:
    """Compute and print an IC table (signals × horizons) for one instrument."""
    coin = _coin_from_filename(files[0])
    logger.info("[%s] loading %d file(s)…", coin, len(files))
    data = collect_signals(files)
    n_events = data["mid"].size
    if n_events < 200:
        raise SystemExit(f"[{coin}] only {n_events} usable events — need a bigger capture")

    signals = ["ofi_ewma", "imbalance", "micro_premium", "trend"]

    print("\n" + "=" * 78)
    print(f"  SIGNAL PREDICTIVE POWER — {coin} — {n_events:,} events".ljust(78))
    print(f"  Information Coefficient = corr(signal_t, forward_return)".ljust(78))
    print("=" * 78)
    header = f"{'SIGNAL':<16}" + "".join(f"{'h=' + str(h):>12}" for h in horizons)
    print(header)
    print("-" * 78)

    best = ("", 0.0, 0)
    for sig in signals:
        cells = []
        for h in horizons:
            fwd = forward_returns(data["mid"], h)
            ic, t, n = information_coefficient(data[sig], fwd)
            cells.append(f"{ic:>+8.4f}{'*' if abs(t) > 3 else ' '}   ")
            if abs(ic) > abs(best[1]):
                best = (f"{sig}@h={h}", ic, n)
        print(f"{sig:<16}" + "".join(cells))
    print("=" * 78)
    print("  * = |t| > 3 (statistically significant).  Magnitude matters more "
          "than significance\n  at this sample size.  |IC|>=0.05 STRONG, "
          ">=0.02 weak, else none.")
    print(f"\n  Strongest: {best[0]}  IC={best[1]:+.4f} ({rating(abs(best[1]))})  "
          f"over {best[2]:,} obs")

    # Verdict for Phase 2.
    print("\n  " + "-" * 74)
    if abs(best[1]) >= 0.05:
        print(f"  VERDICT: strong signal present. Wiring '{best[0].split('@')[0]}' skew into\n"
              f"  the quoter is justified — proceed to strategy integration.")
    elif abs(best[1]) >= 0.02:
        print(f"  VERDICT: weak signal. Worth a cautious skew with a small coefficient,\n"
              f"  but expect modest improvement. Validate out-of-sample before trusting.")
    else:
        print(f"  VERDICT: no usable linear signal at these horizons on {coin}.\n"
              f"  Don't force an OFI skew here — it would be tuning against noise.\n"
              f"  Consider trade-tape OFI, longer horizons, or a different instrument.")
    print("  " + "-" * 74 + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="research_signals",
        description="Measure microstructure signal predictive power (IC) on captures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dir", default="captures", help="directory of capture files")
    p.add_argument("--glob", default="*.jsonl*", help="filename pattern")
    p.add_argument("--coin", required=True, help="instrument to analyze")
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 10, 30, 60],
                   help="forward horizons in events (book updates)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)-7s %(name)s | %(message)s",
    )

    files = [f for f in glob.glob(os.path.join(args.dir, args.glob))
             if _coin_from_filename(f) == args.coin.upper()]
    if not files:
        raise SystemExit(f"no capture files for {args.coin} in {args.dir}")

    analyze(files, args.horizons)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())