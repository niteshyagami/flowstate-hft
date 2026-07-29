"""
calibrate.py
============

Calibrate the fill-model assumptions against real recorded book statistics.

Why this matters
---------------
The paper simulator's fill probability depends on a few parameters that were, until
now, *guesses*:

* ``queue_depth_scale`` — the "typical" resting size at the touch, used to convert
  queue-position depth into a fill probability. It shipped at ``2.0`` for every
  instrument, which is obviously wrong: BTC and DOGE rest wildly different sizes.
* ``tick_size`` — the venue price grid, used to normalize penetration.

A backtest built on guessed microstructure constants is only as trustworthy as those
guesses. This tool measures the real distributions from the capture and reports
data-driven values, so the fill model is grounded in what the book actually did rather
than a round number that happened to make the PnL look good.

What it measures
---------------
For each capture, over every book update:

* **Resting size at the touch** — the median (and quartiles) of best-bid and best-ask
  sizes. The median is the honest ``queue_depth_scale``.
* **Spread distribution** — how often the spread is one tick vs. wider, which tells you
  whether the book is tight (queue position dominates fills) or wide (penetration does).
* **Tick size** — inferred from the smallest non-zero price change actually observed,
  rather than assumed.
* **Price-change frequency** — how often the mid moves at all, a proxy for how much
  genuine fill opportunity exists.

Usage
-----
::

    python calibrate.py --dir captures --coin BTC
    python calibrate.py --dir captures --coin BTC --apply   # print ready-to-use flags
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
from pathlib import Path

import numpy as np

from src.ingestion import PX, SZ
from src.recorder import replay_file

logger = logging.getLogger("calibrate")


def _coin_from_filename(path: str) -> str:
    return Path(path).name.split("-")[0].upper()


def collect_book_stats(files: list[str], max_frames: int | None = None) -> dict:
    """Replay captures and gather touch-size, spread, and tick statistics.

    Returns a dict of raw arrays plus a couple of derived scalars. Memory stays
    bounded: only scalar per-frame summaries are retained, not full books.
    """
    bid_sz: list[float] = []
    ask_sz: list[float] = []
    spreads: list[float] = []
    mids: list[float] = []
    price_changes: list[float] = []

    prev_mid = None
    n = 0
    for path in sorted(files):
        for snap in replay_file(path):
            bid_sz.append(snap.best_bid_size)
            ask_sz.append(snap.best_ask_size)
            spreads.append(snap.spread)
            mids.append(snap.mid)
            if prev_mid is not None:
                dp = abs(snap.mid - prev_mid)
                if dp > 0:
                    price_changes.append(dp)
            prev_mid = snap.mid
            n += 1
            if max_frames and n >= max_frames:
                break
        if max_frames and n >= max_frames:
            break

    return {
        "bid_sz": np.asarray(bid_sz, dtype=np.float64),
        "ask_sz": np.asarray(ask_sz, dtype=np.float64),
        "spreads": np.asarray(spreads, dtype=np.float64),
        "mids": np.asarray(mids, dtype=np.float64),
        "price_changes": np.asarray(price_changes, dtype=np.float64),
        "n_frames": n,
    }


def infer_tick(price_changes: np.ndarray, spreads: np.ndarray) -> float:
    """Infer the venue tick size, robust to float64 subtraction noise.

    Subtracting two nearby float64 prices (e.g. ETH ~1881.7 or DOGE ~0.07) leaves
    rounding dust on the order of 1e-13, which a naive "smallest positive change"
    would wrongly report as the tick. We instead take the smallest *positive*
    spread — the tightest the book gets is one tick — and snap it to the nearest
    power of ten, which is how crypto venues actually define tick grids. Anything
    below a sane floor (1e-9) is treated as float noise and discarded.
    """
    import math

    NOISE_FLOOR = 1e-9
    pos_spreads = spreads[spreads > NOISE_FLOOR]
    if pos_spreads.size == 0:
        return 1.0

    # The modal (most common) spread is the most reliable tick proxy: in a book
    # that is usually 1 tick wide, the mode of the spread *is* the tick.
    vals, counts = np.unique(np.round(pos_spreads, 12), return_counts=True)
    modal_spread = float(vals[int(np.argmax(counts))])

    # Snap to the nearest power of ten (crypto tick grids are 1eN).
    if modal_spread <= 0:
        return 1.0
    exp = round(math.log10(modal_spread))
    return float(10.0 ** exp)


def report(coin: str, s: dict, apply: bool) -> None:
    """Print the calibration summary and, optionally, ready-to-use flags."""
    bid, ask = s["bid_sz"], s["ask_sz"]
    touch = np.concatenate([bid, ask])
    spreads = s["spreads"]
    tick = infer_tick(s["price_changes"], spreads)

    q_med = float(np.median(touch))
    q_p25, q_p75 = (float(np.percentile(touch, 25)), float(np.percentile(touch, 75)))
    spread_ticks = spreads / tick if tick > 0 else spreads
    one_tick_frac = float(np.mean(spread_ticks <= 1.5)) if spread_ticks.size else 0.0
    move_frac = s["price_changes"].size / max(s["n_frames"] - 1, 1)

    print("\n" + "=" * 74)
    print(f"  FILL-MODEL CALIBRATION — {coin} — {s['n_frames']:,} frames".ljust(74))
    print("=" * 74)
    print(f"  Inferred tick size:            {tick:g}")
    print(f"  Median price:                  {float(np.median(s['mids'])):,.6g}")
    print()
    print("  Resting size at the touch (base units):")
    print(f"    median (→ queue_depth_scale) {q_med:.4g}")
    print(f"    25th / 75th percentile       {q_p25:.4g} / {q_p75:.4g}")
    print()
    print("  Spread:")
    print(f"    median                       {float(np.median(spreads)):g} "
          f"({float(np.median(spread_ticks)):.1f} ticks)")
    print(f"    fraction at 1 tick           {one_tick_frac:.1%}")
    print()
    print("  Activity:")
    print(f"    frames where mid moved       {move_frac:.1%}")
    print("=" * 74)

    # Interpretation.
    print("\n  Interpretation:")
    if one_tick_frac > 0.8:
        print("  • Book is almost always 1 tick wide → fills are dominated by QUEUE")
        print("    POSITION, not penetration. queue_depth_scale is the key parameter.")
    else:
        print("  • Book spends meaningful time wider than 1 tick → penetration matters")
        print("    too; both queue_depth_scale and penetration_eta are relevant.")
    if q_med < 0.5:
        print(f"  • Thin touch (median {q_med:.3g}) → orders rest near the front of a short")
        print("    queue; fills are relatively easy. The old default 2.0 was too high here.")
    elif q_med > 5:
        print(f"  • Deep touch (median {q_med:.3g}) → long queues; fills are harder and the")
        print("    old default 2.0 was too low, overstating fill rates.")
    else:
        print(f"  • Touch depth (median {q_med:.3g}) is moderate.")

    print(f"\n  The old shipped default queue_depth_scale=2.0 vs measured {q_med:.4g} — "
          f"{'close' if 0.5 <= q_med <= 5 else 'materially different'}.")

    if apply:
        print("\n  Ready-to-use calibrated flags:")
        print(f"    --tick-size {tick:g} \\")
        print(f"    --queue-depth-scale {q_med:.4g}")
        print("  (queue_depth_scale flag: add to analyze.py/main.py SimConfig if wired.)")
    print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="calibrate",
        description="Calibrate fill-model constants from recorded book statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dir", default="captures")
    p.add_argument("--glob", default="*.jsonl*")
    p.add_argument("--coin", required=True)
    p.add_argument("--max-frames", type=int, default=None,
                   help="cap frames scanned (for a quick estimate)")
    p.add_argument("--apply", action="store_true",
                   help="print ready-to-use calibrated CLI flags")
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

    stats = collect_book_stats(files, args.max_frames)
    if stats["n_frames"] < 100:
        raise SystemExit(f"only {stats['n_frames']} frames — need a bigger capture")
    report(args.coin.upper(), stats, args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())