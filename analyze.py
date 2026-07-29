"""
analyze.py
==========

Batch-backtest recorded captures and produce comparison tables.

Two modes:

**Portfolio mode** (default) — run one backtest per capture file and print a
cross-asset comparison: which instrument the strategy handles best, ranked by
Sharpe, with fills / PnL / inventory-variance / adverse-selection alongside.

    python analyze.py --dir captures

**Sweep mode** — hold the instrument fixed and vary one parameter across a grid,
on the *same* file, so the rows are directly comparable. This is the honest way
to choose a parameter value.

    python analyze.py --dir captures --coin BTC --sweep gamma --grid 0.0001 0.0003 0.001 0.003

Both modes replay from disk, so a three-hour capture crunches in seconds and the
whole sweep is reproducible (fixed RNG seed).

Everything here is offline analysis over paper fills. No network, no orders.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from src.execution import PaperExecutionSimulator, PerformanceReport, SimConfig
from src.features import FeatureEngine
from src.recorder import replay_file
from src.strategy import ASParams, AvellanedaStoikovStrategy

logger = logging.getLogger("analyze")


# --------------------------------------------------------------------------- #
# Core backtest
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class BacktestConfig:
    """Everything needed to reproduce one backtest run."""

    gamma: float = 0.0003
    kappa: float = 1.5
    order_size: float = 0.001
    max_inventory: float = 0.05
    risk_horizon: float = 60.0
    tick_size: float = 1.0
    micro_weight: float = 0.7
    signal_skew: float = 0.0
    imbalance_skew: float = 0.0
    min_samples: int = 5
    vol_window: int = 120
    maker_fee_bps: float = 0.15
    latency_ms: float = 50.0
    queue_depth_scale: float = 2.0
    seed: int = 42


def backtest_file(path: str | Path, cfg: BacktestConfig) -> PerformanceReport:
    """Replay one capture (or several chunks) through the pipeline; return report.

    ``path`` may be a single file or a list of chunk files that belong to the
    same instrument and are replayed back-to-back as one continuous session.
    Mirrors the live loop's ordering: test the resting quote against each new
    book *before* re-quoting, so fills are causal.
    """
    paths = [path] if isinstance(path, (str, Path)) else list(path)

    strategy = AvellanedaStoikovStrategy(ASParams(
        gamma=cfg.gamma, kappa=cfg.kappa, order_size=cfg.order_size,
        max_inventory=cfg.max_inventory, risk_horizon=cfg.risk_horizon,
        tick_size=cfg.tick_size, micro_weight=cfg.micro_weight,
        signal_skew_coefficient=cfg.signal_skew,
        imbalance_skew_coefficient=cfg.imbalance_skew,
    ))
    features = FeatureEngine(min_samples=cfg.min_samples, vol_window=cfg.vol_window)
    sim = PaperExecutionSimulator(SimConfig(
        maker_fee_bps=cfg.maker_fee_bps, latency_ms=cfg.latency_ms,
        max_inventory=cfg.max_inventory, rng_seed=cfg.seed,
        queue_depth_scale=cfg.queue_depth_scale,
    ))

    resting_ref = 0.0
    last = None
    prev_fill_ts: float | None = None
    for p in paths:
        for snap in replay_file(p):
            last = snap
            # 1. resting quote fills against new data
            for fill in sim.on_book(snap):
                dist = abs(fill.price - resting_ref) / max(resting_ref, 1e-12)
                dt = (fill.ts - prev_fill_ts) if prev_fill_ts else 1.0
                prev_fill_ts = fill.ts
                strategy.observe_fill(dist, max(dt, 1e-6))
            # 2. re-quote on post-fill inventory
            fv = features.update(snap)
            if fv is None:
                continue
            q = strategy.quote(fv, sim.inventory)
            if q.bid_px is not None or q.ask_px is not None:
                sim.submit(q, snap)
                resting_ref = q.reference_px

    if last is not None and abs(sim.inventory) > 1e-12:
        sim.flatten(last)
        sim.mark(last)
    return sim.report()


# --------------------------------------------------------------------------- #
# Tick-size heuristic
# --------------------------------------------------------------------------- #
def infer_tick_size(path: str | Path) -> float:
    """Guess a sensible tick size from the first few frames of a capture.

    Different instruments quote on wildly different grids (BTC ~$1, DOGE
    ~$0.00001). Using BTC's tick on DOGE would make every quote span thousands
    of ticks. We sample early spreads and pick a tick a couple of orders of
    magnitude below the price, rounded to a power of ten.
    """
    import math

    mids = []
    for i, snap in enumerate(replay_file(path)):
        mids.append(snap.mid)
        if i >= 20:
            break
    if not mids:
        return 1.0
    price = sum(mids) / len(mids)
    # tick ~ 10^(floor(log10(price)) - 4): BTC~$60k -> 1, DOGE~$0.1 -> 1e-5
    exp = math.floor(math.log10(price)) - 4 if price > 0 else 0
    return float(10.0 ** exp)


def infer_scale(path: str | Path) -> tuple[float, float, float]:
    """Infer (tick_size, order_size, max_inventory) sensible for this instrument.

    Order size and inventory limits must scale with price, not be fixed in base
    units: 0.001 BTC is ~$60 of notional, but 0.001 DOGE is a fraction of a
    cent — far below the tick and effectively unquotable. We target a roughly
    constant *notional* order size (~$25) and a max inventory of ~40 orders, so
    every instrument is quoted on comparable economic terms.
    """
    import math

    mids = []
    for i, snap in enumerate(replay_file(path)):
        mids.append(snap.mid)
        if i >= 20:
            break
    price = (sum(mids) / len(mids)) if mids else 1.0
    exp = math.floor(math.log10(price)) - 4 if price > 0 else 0
    tick = float(10.0 ** exp)

    target_notional = 25.0            # ~$25 per order
    order_size = target_notional / max(price, 1e-9)
    # round order size to 2 significant figures for tidy output
    if order_size > 0:
        mag = 10.0 ** math.floor(math.log10(order_size))
        order_size = round(order_size / mag, 1) * mag
    max_inventory = order_size * 40.0
    return tick, order_size, max_inventory


# --------------------------------------------------------------------------- #
# Portfolio comparison
# --------------------------------------------------------------------------- #
def _coin_from_filename(path: str) -> str:
    """Extract the coin symbol from a ``<coin>-bbo-<stamp>...`` filename."""
    return Path(path).name.split("-")[0].upper()


def portfolio(files: list[str], base: BacktestConfig, auto_tick: bool) -> None:
    """Backtest each instrument (all its chunks together) and rank by Sharpe."""
    # Group chunk files by coin so an instrument's hourly rotations replay as one.
    by_coin: dict[str, list[str]] = {}
    for f in files:
        by_coin.setdefault(_coin_from_filename(f), []).append(f)
    for coin in by_coin:
        by_coin[coin].sort()  # chronological by timestamp in filename

    rows: list[tuple[str, PerformanceReport, float, float]] = []
    for coin, chunks in sorted(by_coin.items()):
        cfg = base
        osize = base.order_size
        if auto_tick:
            tick, osize, maxinv = infer_scale(chunks[0])
            cfg = _replace(base, tick_size=tick, order_size=osize, max_inventory=maxinv)
            logger.info("[%s] %d chunk(s), tick=%g order_size=%g max_inv=%g",
                        coin, len(chunks), tick, osize, maxinv)
        rep = backtest_file(chunks, cfg)
        # Normalize inventory variance to units of order_size^2 so it is
        # comparable across instruments whose base-unit sizes differ by 1e6.
        inv_norm = rep.inventory_variance / (osize ** 2) if osize > 0 else 0.0
        rows.append((coin, rep, osize, inv_norm))

    rows.sort(key=lambda r: r[1].sharpe, reverse=True)

    print("\n" + "=" * 100)
    print("  CROSS-ASSET MARKET-MAKING BACKTEST  ·  3h capture  ·  ranked by Sharpe".ljust(100))
    print("=" * 100)
    hdr = (f"{'COIN':<6} {'FILLS':>6} {'PnL($)':>10} {'SHARPE':>9} {'MAXDD':>9} "
           f"{'INV(norm)':>10} {'ADV SEL':>8} {'MARKOUT':>10} {'FILLS/hr':>9}")
    print(hdr)
    print("-" * 100)
    for coin, r, osize, inv_norm in rows:
        fph = r.n_fills / (r.duration_seconds / 3600.0) if r.duration_seconds > 0 else 0.0
        print(f"{coin:<6} {r.n_fills:>6} {r.total_pnl:>+10.4f} {r.sharpe:>9.1f} "
              f"{r.max_drawdown:>9.4f} {inv_norm:>10.2f} {r.adverse_selection_ratio:>7.1%} "
              f"{r.mean_markout:>+10.5f} {fph:>9.1f}")
    print("=" * 100)

    total_fills = sum(r.n_fills for _, r, _, _ in rows)
    print(f"\n  {len(rows)} instruments · {total_fills} total fills over ~3h")
    print("  Read: SHARPE sign+ranking (magnitude inflated by tick-freq annualization),")
    print("  INV(norm) = inventory variance in units of order_size² (lower = tighter control),")
    print("  ADV SEL near 50%% = neutral, >55%% = being picked off.\n")


# --------------------------------------------------------------------------- #
# Parameter sweep
# --------------------------------------------------------------------------- #
def _replace(cfg: BacktestConfig, **kw) -> BacktestConfig:
    """Return a copy of ``cfg`` with fields overridden."""
    from dataclasses import replace
    return replace(cfg, **kw)


def sweep(files: list[str], param: str, grid: list[float], base: BacktestConfig,
          auto_tick: bool) -> None:
    """Vary one parameter across ``grid`` on one instrument's chunks; print a table."""
    coin = _coin_from_filename(files[0])
    chunks = sorted(files)
    cfg = base
    osize = base.order_size
    if auto_tick:
        tick, osize, maxinv = infer_scale(chunks[0])
        cfg = _replace(base, tick_size=tick, order_size=osize, max_inventory=maxinv)

    if not hasattr(cfg, param):
        raise SystemExit(f"unknown sweep parameter '{param}'. "
                         f"Try one of: gamma, kappa, risk_horizon, micro_weight, "
                         f"order_size, max_inventory, latency_ms")

    print("\n" + "=" * 92)
    print(f"  PARAMETER SWEEP — {coin} — varying '{param}'  ·  {len(chunks)} chunk(s) · "
          f"same data".ljust(92))
    print("=" * 92)
    hdr = (f"{param:>12} {'FILLS':>6} {'PnL($)':>10} {'SHARPE':>9} "
           f"{'INV(norm)':>10} {'ADV SEL':>8} {'MARKOUT':>10}")
    print(hdr)
    print("-" * 92)
    best = None
    for val in grid:
        rep = backtest_file(chunks, _replace(cfg, **{param: val}))
        inv_norm = rep.inventory_variance / (osize ** 2) if osize > 0 else 0.0
        print(f"{val:>12g} {rep.n_fills:>6} {rep.total_pnl:>+10.4f} {rep.sharpe:>9.1f} "
              f"{inv_norm:>10.2f} {rep.adverse_selection_ratio:>7.1%} {rep.mean_markout:>+10.5f}")
        if best is None or rep.sharpe > best[1]:
            best = (val, rep.sharpe)
    print("=" * 92)
    if best:
        print(f"\n  Best Sharpe at {param}={best[0]:g} (Sharpe {best[1]:.1f}). "
              f"Prefer a plateau over a spike — a lone good value is likely overfit.\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    p = argparse.ArgumentParser(
        prog="analyze",
        description="Batch backtest recorded captures (portfolio compare or parameter sweep).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dir", default="captures", help="directory of capture files")
    p.add_argument("--glob", default="*.jsonl*", help="filename pattern within --dir")
    p.add_argument("--coin", default=None, help="restrict to one coin (needed for --sweep)")
    p.add_argument("--sweep", default=None,
                   help="parameter to sweep (gamma, kappa, risk_horizon, micro_weight, ...)")
    p.add_argument("--grid", nargs="+", type=float, default=None,
                   help="values for the swept parameter")
    p.add_argument("--no-auto-tick", action="store_true",
                   help="disable per-coin tick-size inference; use --tick-size for all")

    g = p.add_argument_group("strategy defaults")
    g.add_argument("--gamma", type=float, default=0.0003)
    g.add_argument("--risk-horizon", type=float, default=60.0)
    g.add_argument("--order-size", type=float, default=0.001)
    g.add_argument("--max-inventory", type=float, default=0.05)
    g.add_argument("--tick-size", type=float, default=1.0)
    g.add_argument("--maker-fee-bps", type=float, default=0.15)
    g.add_argument("--queue-depth-scale", type=float, default=2.0,
                   help="typical resting size at touch (calibrate.py measures this)")
    g.add_argument("--min-samples", type=int, default=5)
    g.add_argument("--vol-window", type=int, default=120)
    g.add_argument("--signal-skew", type=float, default=0.0,
                   help="micro-price-premium predictive skew strength (0 = off)")
    g.add_argument("--imbalance-skew", type=float, default=0.0,
                   help="book-imbalance predictive skew strength (0 = off)")
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
        vol_window=args.vol_window,
        signal_skew=args.signal_skew, imbalance_skew=args.imbalance_skew,
        queue_depth_scale=args.queue_depth_scale,
    )
    auto_tick = not args.no_auto_tick

    pattern = os.path.join(args.dir, args.glob)
    files = [f for f in glob.glob(pattern) if not f.endswith(".summary.json")]
    if args.coin:
        files = [f for f in files if _coin_from_filename(f) == args.coin.upper()]
    if not files:
        raise SystemExit(f"no capture files matched {pattern}"
                         + (f" for coin {args.coin}" if args.coin else ""))

    if args.sweep:
        if not args.grid:
            raise SystemExit("--sweep requires --grid with values")
        if not args.coin:
            raise SystemExit("--sweep requires --coin to pick one instrument")
        sweep(files, args.sweep, args.grid, base, auto_tick)
    else:
        portfolio(files, base, auto_tick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())