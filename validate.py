"""
validate.py
===========

Walk-forward (out-of-sample) validation — the honesty test for every result.

The problem it solves
---------------------
Every number in ``RESULTS.md`` so far is *in-sample*: parameters were chosen on the
same 3-hour capture they were then evaluated on. That is circular. Any flexible
model can be tuned to look good on data it has already seen; the only question that
matters is whether the chosen parameters keep working on data they were **not** fitted
to.

Method
------
Split each instrument's chronological chunks into **train** and **test**:

* **Train** on the earlier chunk(s): sweep the parameter grid, pick the value with the
  best in-sample Sharpe.
* **Test** that single chosen value on the later chunk(s), which the selection never saw.

If test-set performance holds up, the edge is real. If it collapses, the in-sample
result was overfit — and finding that out *now*, on paper, is exactly the point.

This is a single split (earliest → latest). A fuller version rolls the split forward
repeatedly; with only three one-hour chunks per asset a single split is the honest
maximum, and the code notes that limit rather than overstating it.

Usage
-----
::

    python validate.py --dir captures --coin BTC --sweep gamma \\
        --grid 0.003 0.01 0.03 0.1
    python validate.py --dir captures --coin ETH --sweep signal_skew \\
        --grid 0 20 50 100 --gamma 0.1
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
from dataclasses import replace
from pathlib import Path

from analyze import BacktestConfig, backtest_file, infer_scale, _coin_from_filename

logger = logging.getLogger("validate")


def _split_chunks(files: list[str]) -> tuple[list[str], list[str]]:
    """Split sorted chunk files into (train, test), earliest→latest.

    With N chunks, train on the first ``ceil(N/2)`` and test on the rest, so a
    3-chunk asset trains on 2 hours and tests on the final hour.
    """
    ordered = sorted(files)
    if len(ordered) < 2:
        return ordered, []
    k = (len(ordered) + 1) // 2
    return ordered[:k], ordered[k:]


def _best_param(train: list[str], param: str, grid: list[float],
                base: BacktestConfig) -> tuple[float, float]:
    """Sweep ``param`` over ``grid`` on the training set; return (best_value, its_sharpe).

    Selection is by in-sample Sharpe among configurations with a usable fill count,
    so a 2-fill fluke can't win the sweep.
    """
    best_val, best_sharpe = grid[0], -1e18
    for val in grid:
        rep = backtest_file(train, replace(base, **{param: val}))
        if rep.n_fills < 20:
            continue  # too few fills to trust this row
        if rep.sharpe > best_sharpe:
            best_val, best_sharpe = val, rep.sharpe
    return best_val, best_sharpe


def validate(files: list[str], param: str, grid: list[float], base: BacktestConfig,
             auto_tick: bool) -> None:
    """Run one train/test split and print the in-sample vs out-of-sample comparison."""
    coin = _coin_from_filename(files[0])
    train, test = _split_chunks(files)
    if not test:
        raise SystemExit(f"[{coin}] needs at least 2 chunks to split; found {len(files)}")

    cfg = base
    if auto_tick:
        tick, osize, maxinv = infer_scale(train[0])
        cfg = replace(base, tick_size=tick, order_size=osize, max_inventory=maxinv)

    # 1. Select the parameter purely on training data.
    best_val, train_sharpe = _best_param(train, param, grid, cfg)
    chosen = replace(cfg, **{param: best_val})

    # 2. Evaluate the same value on both sets.
    train_rep = backtest_file(train, chosen)
    test_rep = backtest_file(test, chosen)

    print("\n" + "=" * 84)
    print(f"  WALK-FORWARD VALIDATION — {coin} — parameter '{param}'".ljust(84))
    print("=" * 84)
    print(f"  Train chunks: {len(train)}   Test chunks: {len(test)}   "
          f"(chronological split, test is unseen)")
    print(f"  Swept {param} over {grid}")
    print(f"  Selected {param} = {best_val:g}  (best in-sample Sharpe among ≥20-fill configs)")
    print("-" * 84)
    print(f"  {'':<14}{'FILLS':>7}{'PnL($)':>11}{'SHARPE':>10}{'ADV SEL':>10}{'MARKOUT':>11}")
    print(f"  {'IN-SAMPLE':<14}{train_rep.n_fills:>7}{train_rep.total_pnl:>+11.4f}"
          f"{train_rep.sharpe:>10.1f}{train_rep.adverse_selection_ratio:>9.1%}"
          f"{train_rep.mean_markout:>+11.5f}")
    print(f"  {'OUT-OF-SAMPLE':<14}{test_rep.n_fills:>7}{test_rep.total_pnl:>+11.4f}"
          f"{test_rep.sharpe:>10.1f}{test_rep.adverse_selection_ratio:>9.1%}"
          f"{test_rep.mean_markout:>+11.5f}")
    print("=" * 84)
    _verdict(train_rep, test_rep)


def _verdict(train_rep, test_rep) -> None:
    """Interpret the train→test transition honestly."""
    print()
    if test_rep.n_fills < 20:
        print("  VERDICT: too few out-of-sample fills to conclude. Need a longer capture.")
        print()
        return

    pnl_sign_held = (train_rep.total_pnl > 0) == (test_rep.total_pnl > 0)
    profitable_oos = test_rep.total_pnl > 0
    adv_ok = test_rep.adverse_selection_ratio < 0.52

    if profitable_oos and pnl_sign_held:
        print("  VERDICT: ✅ EDGE HELD OUT-OF-SAMPLE. The chosen parameter stayed profitable")
        print("  on data it was never fitted to — evidence the result is real, not overfit.")
    elif profitable_oos:
        print("  VERDICT: ⚠️ Profitable out-of-sample, but magnitude shifted from in-sample.")
        print("  Directionally encouraging; treat the specific numbers with caution.")
    else:
        print("  VERDICT: ❌ EDGE DID NOT SURVIVE. In-sample performance did not carry to the")
        print("  unseen test set — the in-sample result was at least partly overfit. This is a")
        print("  finding, not a failure: it's exactly what walk-forward testing exists to catch.")
    if not adv_ok:
        print(f"  Note: out-of-sample adverse selection {test_rep.adverse_selection_ratio:.1%} "
              f"is high (>52%).")
    print("\n  Caveat: single train/test split over ~3h of data — a directional check, not a")
    print("  full rolling walk-forward. Treat as a sanity test, not a performance guarantee.\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="validate",
        description="Walk-forward out-of-sample validation of a chosen parameter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dir", default="captures")
    p.add_argument("--glob", default="*.jsonl*")
    p.add_argument("--coin", required=True)
    p.add_argument("--sweep", required=True,
                   help="parameter to select on train and test on holdout")
    p.add_argument("--grid", nargs="+", type=float, required=True)
    p.add_argument("--no-auto-tick", action="store_true")

    g = p.add_argument_group("fixed strategy defaults")
    g.add_argument("--gamma", type=float, default=0.03)
    g.add_argument("--risk-horizon", type=float, default=60.0)
    g.add_argument("--order-size", type=float, default=0.001)
    g.add_argument("--max-inventory", type=float, default=0.05)
    g.add_argument("--tick-size", type=float, default=1.0)
    g.add_argument("--maker-fee-bps", type=float, default=0.15)
    g.add_argument("--min-samples", type=int, default=5)
    g.add_argument("--vol-window", type=int, default=120)
    g.add_argument("--signal-skew", type=float, default=0.0)
    g.add_argument("--imbalance-skew", type=float, default=0.0)
    g.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)-7s %(name)s | %(message)s",
    )

    base = BacktestConfig(
        gamma=args.gamma, risk_horizon=args.risk_horizon, order_size=args.order_size,
        max_inventory=args.max_inventory, tick_size=args.tick_size,
        maker_fee_bps=args.maker_fee_bps, min_samples=args.min_samples,
        vol_window=args.vol_window, signal_skew=args.signal_skew,
        imbalance_skew=args.imbalance_skew,
    )

    files = [f for f in glob.glob(os.path.join(args.dir, args.glob))
             if _coin_from_filename(f) == args.coin.upper()]
    if not files:
        raise SystemExit(f"no capture files for {args.coin} in {args.dir}")

    validate(files, args.sweep, args.grid, base, not args.no_auto_tick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())