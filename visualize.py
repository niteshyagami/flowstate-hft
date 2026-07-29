"""
visualize.py
============

Turn a backtest into publication-quality charts (PNG) — the tearsheet.

Produces four panels that tell the project's story visually:

1. **Equity curve** — cumulative PnL over the session.
2. **Inventory path** — signed position over time, with the ±limit band, showing
   the reservation-price mechanism keeping inventory bounded.
3. **Signal on/off comparison** — adverse selection and PnL with vanilla A-S vs
   the signal-enhanced quote, on the same data (the Phase 2 result).
4. **Walk-forward panel** — in-sample vs out-of-sample bars for the validated
   parameter (the credibility result).

Charts are saved as PNGs under ``--out`` (default ``charts/``), ready to drop into
a README or a LinkedIn post.

Usage
-----
::

    # equity + inventory tearsheet for one instrument
    python visualize.py --dir captures --coin BTC --gamma 0.03 --signal-skew 20

    # add the signal on/off comparison panel
    python visualize.py --dir captures --coin BTC --gamma 0.03 --signal-skew 20 --compare

Requires matplotlib (``pip install matplotlib``).
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

logger = logging.getLogger("visualize")

# Colour palette — muted, professional, colour-blind-safe.
INK = "#1a1a2e"
ACCENT = "#0f4c81"        # deep blue
ACCENT2 = "#c1666b"       # muted red
GOOD = "#2a9d8f"          # teal
GRID = "#e0e0e0"


def _require_matplotlib():
    """Import matplotlib with a friendly message if it's missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless: write files, no display window
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required for charts.\n"
            "  Install it with:  pip install matplotlib"
        ) from exc


def _run_backtest(files, cfg):
    """Backtest and return (report, equity_df, fills_df)."""
    from analyze import backtest_file  # noqa: F401 (kept for parity)
    from src.execution import PaperExecutionSimulator, SimConfig
    from src.features import FeatureEngine
    from src.recorder import replay_file
    from src.strategy import ASParams, AvellanedaStoikovStrategy

    strat = AvellanedaStoikovStrategy(ASParams(
        gamma=cfg.gamma, kappa=cfg.kappa, order_size=cfg.order_size,
        max_inventory=cfg.max_inventory, risk_horizon=cfg.risk_horizon,
        tick_size=cfg.tick_size, micro_weight=cfg.micro_weight,
        signal_skew_coefficient=cfg.signal_skew,
        imbalance_skew_coefficient=cfg.imbalance_skew,
    ))
    fe = FeatureEngine(min_samples=cfg.min_samples, vol_window=cfg.vol_window)
    sim = PaperExecutionSimulator(SimConfig(
        maker_fee_bps=cfg.maker_fee_bps, latency_ms=cfg.latency_ms,
        max_inventory=cfg.max_inventory, rng_seed=cfg.seed,
    ))
    ref = 0.0
    last = None
    for p in sorted(files):
        for snap in replay_file(p):
            last = snap
            for _ in sim.on_book(snap):
                pass
            fv = fe.update(snap)
            if fv is None:
                continue
            q = strat.quote(fv, sim.inventory)
            if q.bid_px is not None or q.ask_px is not None:
                sim.submit(q, snap)
                ref = q.reference_px
    if last is not None and abs(sim.inventory) > 1e-12:
        sim.flatten(last)
        sim.mark(last)
    return sim.report(), sim.equity_dataframe(), sim.fills_dataframe(), cfg.max_inventory


def _style(ax, plt):
    """Apply a clean, consistent style to an axis."""
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#999999")
    ax.tick_params(colors="#555555", labelsize=9)


def tearsheet(files, cfg, coin, out_dir, plt):
    """Equity curve + inventory path, stacked, saved as one PNG."""
    rep, eq, fills, max_inv = _run_backtest(files, cfg)
    if eq.empty:
        raise SystemExit(f"[{coin}] no data to plot")

    t = (eq["ts"] - eq["ts"].iloc[0]) / 60.0  # minutes from start

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), height_ratios=[2, 1], sharex=True)
    fig.patch.set_facecolor("white")

    # --- Equity curve ---
    ax1.plot(t, eq["equity"], color=ACCENT, linewidth=1.6, zorder=3)
    ax1.axhline(0, color="#bbbbbb", linewidth=1.0, linestyle="--", zorder=1)
    ax1.fill_between(t, eq["equity"], 0, where=(eq["equity"] >= 0),
                     color=ACCENT, alpha=0.10, zorder=2)
    ax1.fill_between(t, eq["equity"], 0, where=(eq["equity"] < 0),
                     color=ACCENT2, alpha=0.10, zorder=2)
    _style(ax1, plt)
    ax1.set_ylabel("Cumulative PnL ($)", fontsize=10, color=INK)
    skew_txt = f"signal_skew={cfg.signal_skew:g}" if cfg.signal_skew else "raw A-S"
    ax1.set_title(
        f"FlowState HFT  ·  {coin}  ·  γ={cfg.gamma:g}, {skew_txt}",
        fontsize=13, color=INK, fontweight="bold", loc="left", pad=12)

    stats = (f"PnL ${rep.total_pnl:+.3f}   Sharpe {rep.sharpe:.0f}   "
             f"Fills {rep.n_fills}   Adv-sel {rep.adverse_selection_ratio:.0%}")
    ax1.text(0.5, 0.03, stats, transform=ax1.transAxes, ha="center", va="bottom",
             fontsize=9, color="#555555")

    # --- Inventory path ---
    ax2.plot(t, eq["inventory"], color=GOOD, linewidth=1.2, zorder=3)
    ax2.axhline(0, color="#bbbbbb", linewidth=1.0, zorder=1)
    ax2.axhline(max_inv, color=ACCENT2, linewidth=0.9, linestyle=":", zorder=1)
    ax2.axhline(-max_inv, color=ACCENT2, linewidth=0.9, linestyle=":", zorder=1)
    ax2.fill_between(t, eq["inventory"], 0, color=GOOD, alpha=0.10, zorder=2)
    _style(ax2, plt)
    ax2.set_ylabel("Inventory", fontsize=10, color=INK)
    ax2.set_xlabel("Minutes", fontsize=10, color=INK)
    ax2.text(0.99, 0.92, f"±{max_inv:g} limit", transform=ax2.transAxes,
             ha="right", va="top", fontsize=8, color=ACCENT2)

    fig.tight_layout()
    path = Path(out_dir) / f"{coin.lower()}_tearsheet.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    logger.warning("wrote %s", path)
    return path


def comparison(files, cfg, coin, out_dir, plt):
    """Signal on vs off: adverse selection + PnL bars, on the same data."""
    off = _run_backtest(files, replace(cfg, signal_skew=0.0, imbalance_skew=0.0))[0]
    on = _run_backtest(files, cfg)[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor("white")
    labels = ["Raw A-S", "Signal-enhanced"]
    colors = ["#9aa0b4", ACCENT]

    # Adverse selection (lower is better) with 50% neutral line.
    adv = [off.adverse_selection_ratio * 100, on.adverse_selection_ratio * 100]
    b1 = ax1.bar(labels, adv, color=colors, width=0.55, zorder=3)
    ax1.axhline(50, color=ACCENT2, linewidth=1.0, linestyle="--", zorder=2)
    ax1.text(1.4, 50.5, "50% neutral", color=ACCENT2, fontsize=8, ha="right")
    _style(ax1, plt)
    ax1.set_ylabel("Adverse selection (%)", fontsize=10, color=INK)
    ax1.set_title("Lower = fewer pick-offs", fontsize=10, color=INK, loc="left")
    for rect, v in zip(b1, adv):
        ax1.text(rect.get_x() + rect.get_width() / 2, v + 0.4, f"{v:.1f}%",
                 ha="center", fontsize=9, color=INK, fontweight="bold")

    # PnL.
    pnl = [off.total_pnl, on.total_pnl]
    b2 = ax2.bar(labels, pnl, color=colors, width=0.55, zorder=3)
    ax2.axhline(0, color="#bbbbbb", linewidth=1.0, zorder=2)
    _style(ax2, plt)
    ax2.set_ylabel("PnL ($)", fontsize=10, color=INK)
    ax2.set_title("Same data, signal on vs off", fontsize=10, color=INK, loc="left")
    for rect, v in zip(b2, pnl):
        off_y = 0.02 * (max(abs(p) for p in pnl) + 1e-9)
        ax2.text(rect.get_x() + rect.get_width() / 2,
                 v + (off_y if v >= 0 else -off_y), f"${v:+.2f}",
                 ha="center", va="bottom" if v >= 0 else "top",
                 fontsize=9, color=INK, fontweight="bold")

    fig.suptitle(f"{coin} · Phase 2 signal enhancement", fontsize=13,
                 color=INK, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = Path(out_dir) / f"{coin.lower()}_signal_comparison.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    logger.warning("wrote %s", path)
    return path


def main(argv: list[str] | None = None) -> int:
    from analyze import BacktestConfig, infer_scale, _coin_from_filename

    p = argparse.ArgumentParser(
        prog="visualize",
        description="Render backtest tearsheet charts (PNG).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dir", default="captures")
    p.add_argument("--glob", default="*.jsonl*")
    p.add_argument("--coin", required=True)
    p.add_argument("--out", default="charts", help="output directory for PNGs")
    p.add_argument("--compare", action="store_true",
                   help="also render the signal on/off comparison panel")
    p.add_argument("--no-auto-tick", action="store_true")

    g = p.add_argument_group("strategy")
    g.add_argument("--gamma", type=float, default=0.03)
    g.add_argument("--signal-skew", type=float, default=0.0)
    g.add_argument("--imbalance-skew", type=float, default=0.0)
    g.add_argument("--risk-horizon", type=float, default=60.0)
    g.add_argument("--order-size", type=float, default=0.001)
    g.add_argument("--max-inventory", type=float, default=0.05)
    g.add_argument("--tick-size", type=float, default=1.0)
    g.add_argument("--maker-fee-bps", type=float, default=0.15)
    g.add_argument("--min-samples", type=int, default=5)
    g.add_argument("--vol-window", type=int, default=120)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    plt = _require_matplotlib()

    files = [f for f in glob.glob(os.path.join(args.dir, args.glob))
             if _coin_from_filename(f) == args.coin.upper()]
    if not files:
        raise SystemExit(f"no capture files for {args.coin} in {args.dir}")

    base = BacktestConfig(
        gamma=args.gamma, risk_horizon=args.risk_horizon, order_size=args.order_size,
        max_inventory=args.max_inventory, tick_size=args.tick_size,
        maker_fee_bps=args.maker_fee_bps, min_samples=args.min_samples,
        vol_window=args.vol_window, signal_skew=args.signal_skew,
        imbalance_skew=args.imbalance_skew,
    )
    if not args.no_auto_tick:
        tick, osize, maxinv = infer_scale(sorted(files)[0])
        base = replace(base, tick_size=tick, order_size=osize, max_inventory=maxinv)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    tearsheet(files, base, args.coin.upper(), args.out, plt)
    if args.compare:
        comparison(files, base, args.coin.upper(), args.out, plt)
    print(f"\nCharts written to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())