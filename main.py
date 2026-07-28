"""
main.py
=======

FlowState HFT — orchestrator.

Wires the four stages into a single asyncio application::

    Hyperliquid WS ──▶ [ingestion] ──queue──▶ [features] ──▶ [strategy] ──▶ [execution sim]
                                                                                │
                                                                     periodic reporting

Concurrency design
------------------
Two tasks, one bounded queue:

* **Producer** (``HyperliquidL2Client.run``) owns the socket. It never awaits a
  consumer; on backpressure it drops the oldest snapshot. A market maker would
  rather skip a frame than quote off a stale book.
* **Consumer** (``TradingLoop.run``) drains the queue and executes the
  feature → strategy → simulation pipeline synchronously per tick. The whole
  pipeline is pure CPU work measured in microseconds, so introducing awaits
  between the stages would add scheduler latency for no concurrency gain.
  ``asyncio`` earns its keep at the I/O boundary, not inside the hot path.

A third supervisory task prints a periodic status line and, at shutdown,
flattens the book and renders the performance report. ``SIGINT``/``SIGTERM``
trigger a graceful unwind rather than a stack trace.

Safety
------
This program is **simulation only**. It opens a read-only public market-data
socket and never authenticates, signs, or transmits an order. There is no code
path from this repository to a live venue.

Usage
-----
::

    python main.py --coin BTC --duration 300 --gamma 0.05 --verbose
    python main.py --coin ETH --duration 600 --export ./runs
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from src.execution import PaperExecutionSimulator, SimConfig
from src.features import FeatureEngine
from src.ingestion import BookSnapshot, HyperliquidL2Client
from src.strategy import ASParams, AvellanedaStoikovStrategy

logger = logging.getLogger("flowstate")

BANNER: Final[str] = r"""
  ______ _               _____ _        _         _   _ ______ _____
 |  ____| |             / ____| |      | |       | | | |  ____|_   _|
 | |__  | | _____      | (___ | |_ __ _| |_ ___  | |_| | |__    | |
 |  __| | |/ _ \ \ /\ / \___ \| __/ _` | __/ _ \ |  _  |  __|   | |
 | |    | | (_) \ V  V /____) | || (_| | ||  __/ | | | | |     _| |_
 |_|    |_|\___/ \_/\_/|_____/ \__\__,_|\__\___| |_| |_|_|    |_____|

  Avellaneda-Stoikov market making  ·  PAPER TRADING ONLY  ·  no venue orders
"""


# --------------------------------------------------------------------------- #
# Runtime configuration
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class RunConfig:
    """Top-level runtime settings resolved from the CLI."""

    coin: str = "BTC"
    duration: float = 300.0
    channel: str = "bbo"
    status_interval: float = 10.0
    ladder_levels: int = 5
    show_book: bool = False
    export_dir: Path | None = None
    verbose: bool = False


# --------------------------------------------------------------------------- #
# Trading loop
# --------------------------------------------------------------------------- #
class TradingLoop:
    """Consumes book snapshots and drives features → strategy → simulation."""

    def __init__(
        self,
        features: FeatureEngine,
        strategy: AvellanedaStoikovStrategy,
        simulator: PaperExecutionSimulator,
        config: RunConfig,
    ) -> None:
        self.features = features
        self.strategy = strategy
        self.sim = simulator
        self.cfg = config

        self.ticks = 0
        self.quotes = 0
        self.last_snapshot: BookSnapshot | None = None
        self._last_fill_ts: float | None = None
        self._resting_ref: float = 0.0
        self._stop = asyncio.Event()

    def stop(self) -> None:
        """Request a graceful exit of the consumer loop."""
        self._stop.set()

    async def run(self, queue: asyncio.Queue[BookSnapshot]) -> None:
        """Drain ``queue`` and run the pipeline until stopped."""
        while not self._stop.is_set():
            try:
                snap = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue  # heartbeat: lets us notice self._stop promptly
            except asyncio.CancelledError:
                raise

            try:
                self._on_snapshot(snap)
            except Exception:  # noqa: BLE001 - one bad tick must not kill the loop
                logger.exception("pipeline error on seq=%s; continuing", snap.seq)

    # -- Pipeline ----------------------------------------------------------- #
    def _on_snapshot(self, snap: BookSnapshot) -> None:
        """Execute one full tick of the pipeline."""
        self.ticks += 1
        self.last_snapshot = snap

        if self.cfg.show_book and self.ticks % 20 == 0:
            print(snap.format_ladder(self.cfg.ladder_levels), end="\n\n")

        # Stage 2 — features
        fv = self.features.update(snap)
        if fv is None:
            self.sim.mark(snap)
            return

        # Stage 4a — test the quote resting from the PREVIOUS tick against this
        # new book, before it gets replaced. Order matters: new data arrives,
        # resting orders may fill, and only then do we re-quote. Submitting
        # first would mean a quote is always centred on the very book it is
        # tested against, and nothing could ever fill.
        fills = self.sim.on_book(snap)
        for fill in fills:
            self._feed_back_kappa(fill.price, self._resting_ref, fill.ts)
            logger.info(
                "FILL %-4s %.5f @ %.2f | q=%+.5f | equity=%+.4f",
                fill.side, fill.size, fill.price, self.sim.inventory, self.sim.equity,
            )

        # Stage 3 — strategy (inventory now reflects any fill above)
        quote = self.strategy.quote(fv, self.sim.inventory)
        if quote.bid_px is not None or quote.ask_px is not None:
            self.sim.submit(quote, snap)
            self._resting_ref = quote.reference_px
            self.quotes += 1

    def _feed_back_kappa(self, fill_px: float, reference_px: float, ts: float) -> None:
        """Feed realized fill distance back into the adaptive kappa estimator."""
        distance = abs(fill_px - reference_px) / max(reference_px, 1e-12)
        dt = ts - self._last_fill_ts if self._last_fill_ts is not None else 1.0
        self._last_fill_ts = ts
        self.strategy.observe_fill(distance, max(dt, 1e-6))

    # -- Reporting ---------------------------------------------------------- #
    def status_line(self, started: float) -> str:
        """One-line operational summary."""
        elapsed = time.time() - started
        rate = self.ticks / elapsed if elapsed > 0 else 0.0
        mid = self.last_snapshot.mid if self.last_snapshot else float("nan")
        return (
            f"t={elapsed:6.1f}s | ticks={self.ticks:6d} ({rate:5.1f}/s) "
            f"| quotes={self.quotes:6d} | fills={len(self.sim.fills):4d} "
            f"| mid={mid:,.2f} | q={self.sim.inventory:+.5f} "
            f"| pnl={self.sim.equity:+.4f} | κ={self.strategy.kappa:.3f}"
        )


# --------------------------------------------------------------------------- #
# Application wiring
# --------------------------------------------------------------------------- #
async def run_session(
    cfg: RunConfig, as_params: ASParams, sim_cfg: SimConfig, features: FeatureEngine
) -> PaperExecutionSimulator:
    """Run one complete paper-trading session and return the simulator."""
    queue: asyncio.Queue[BookSnapshot] = asyncio.Queue(maxsize=512)

    client = HyperliquidL2Client(coin=cfg.coin, channel=cfg.channel, queue_size=512)
    strategy = AvellanedaStoikovStrategy(as_params)
    simulator = PaperExecutionSimulator(sim_cfg)
    loop_task_obj = TradingLoop(features, strategy, simulator, cfg)

    started = time.time()
    producer = asyncio.create_task(client.run(queue), name="ingestion")
    consumer = asyncio.create_task(loop_task_obj.run(queue), name="pipeline")

    shutdown = asyncio.Event()
    _install_signal_handlers(shutdown)

    async def _status() -> None:
        """Emit a periodic heartbeat until shutdown."""
        while not shutdown.is_set():
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(shutdown.wait(), timeout=cfg.status_interval)
                break
            logger.info(loop_task_obj.status_line(started))

    status = asyncio.create_task(_status(), name="status")

    # Wait for whichever comes first: the duration budget, or a signal.
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(shutdown.wait(), timeout=cfg.duration)

    logger.info("shutting down…")
    shutdown.set()
    client.stop()
    loop_task_obj.stop()

    for task in (producer, consumer, status):
        task.cancel()
    await asyncio.gather(producer, consumer, status, return_exceptions=True)

    # Flatten any residual inventory so the report reflects a closed book.
    if loop_task_obj.last_snapshot is not None:
        simulator.flatten(loop_task_obj.last_snapshot)
        simulator.mark(loop_task_obj.last_snapshot)

    logger.info(
        "session complete: %d ticks, %d quotes, %d fills, %d reconnects",
        loop_task_obj.ticks, loop_task_obj.quotes, len(simulator.fills), client.stats.reconnects,
    )
    return simulator


def run_replay(
    path: str, cfg: RunConfig, as_params: ASParams, sim_cfg: SimConfig, features: FeatureEngine
) -> PaperExecutionSimulator:
    """Backtest the strategy against a recorded capture file.

    Fully synchronous and as fast as the disk allows — no network, no waiting,
    no event loop. Hours of recorded ticks process in seconds. Because the input
    is fixed, this is the path where parameter comparisons are actually valid:
    two runs with different ``as_params`` see the *identical* market, so any
    performance difference is attributable to the parameters, not to luck of the
    draw in when you happened to be connected.
    """
    from src.recorder import replay_file

    strategy = AvellanedaStoikovStrategy(as_params)
    simulator = PaperExecutionSimulator(sim_cfg)
    loop = TradingLoop(features, strategy, simulator, cfg)

    logger.info("replaying %s …", path)
    for snap in replay_file(path):
        loop._on_snapshot(snap)  # same pipeline the live loop uses

    if loop.last_snapshot is not None:
        simulator.flatten(loop.last_snapshot)
        simulator.mark(loop.last_snapshot)

    logger.info(
        "replay complete: %d ticks, %d quotes, %d fills",
        loop.ticks, loop.quotes, len(simulator.fills),
    )
    return simulator


def _install_signal_handlers(event: asyncio.Event) -> None:
    """Route SIGINT/SIGTERM to a graceful shutdown event where supported."""
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, AttributeError, ValueError):
            loop.add_signal_handler(sig, event.set)


def _export(sim: PaperExecutionSimulator, directory: Path, coin: str) -> None:
    """Write the fill blotter, equity curve, and summary to ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    prefix = directory / f"{coin.lower()}-{stamp}"

    sim.fills_dataframe().to_csv(f"{prefix}-fills.csv", index=False)
    sim.equity_dataframe().to_csv(f"{prefix}-equity.csv", index=False)

    import json

    with open(f"{prefix}-summary.json", "w", encoding="utf-8") as fh:
        json.dump(sim.report().to_dict(), fh, indent=2)
    logger.info("exported artifacts to %s*", prefix)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface."""
    p = argparse.ArgumentParser(
        prog="flowstate",
        description="FlowState HFT — Avellaneda-Stoikov market-making simulator (paper only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--coin", default="BTC", help="Hyperliquid symbol to quote")
    p.add_argument("--duration", type=float, default=300.0, help="session length in seconds")
    p.add_argument("--channel", default="bbo", choices=["l2Book", "bbo"],
                   help="market-data feed: 'bbo' is far more frequent (more ticks/fills), "
                        "'l2Book' gives full depth but updates less often")

    m = p.add_argument_group("mode")
    m.add_argument("--replay", metavar="FILE", default=None,
                   help="backtest against a recorded capture instead of connecting live")
    m.add_argument("--record", metavar="DIR", default=None,
                   help="record live data to DIR (no trading); use with --duration")

    g = p.add_argument_group("strategy")
    g.add_argument("--gamma", type=float, default=0.001, help="risk aversion γ")
    g.add_argument("--kappa", type=float, default=1.5, help="order-arrival elasticity κ")
    g.add_argument("--order-size", type=float, default=0.001, help="quote size in base units")
    g.add_argument("--max-inventory", type=float, default=0.01, help="hard position limit |q|")
    g.add_argument("--risk-horizon", type=float, default=60.0, help="τ in seconds (rolling mode)")
    g.add_argument("--tick-size", type=float, default=1.0, help="venue price increment")
    g.add_argument("--micro-weight", type=float, default=0.7, help="micro-price blend weight")

    fg = p.add_argument_group("features")
    fg.add_argument("--min-samples", type=int, default=30,
                    help="returns needed before quoting begins (lower for slow/calm books)")
    fg.add_argument("--vol-window", type=int, default=512,
                    help="returns retained for the volatility estimate")
    fg.add_argument("--ofi-halflife", type=float, default=5.0,
                    help="OFI EWMA half-life in seconds")
    fg.add_argument("--trend-halflife", type=float, default=10.0,
                    help="drift EWMA half-life in seconds")

    s = p.add_argument_group("simulation")
    s.add_argument("--maker-fee-bps", type=float, default=0.0, help="maker fee (negative = rebate)")
    s.add_argument("--latency-ms", type=float, default=25.0, help="quote placement latency")
    s.add_argument("--seed", type=int, default=42, help="RNG seed for fill sampling")

    o = p.add_argument_group("output")
    o.add_argument("--status-interval", type=float, default=10.0, help="heartbeat period")
    o.add_argument("--show-book", action="store_true", help="periodically print the L2 ladder")
    o.add_argument("--export", type=Path, default=None, help="directory for CSV/JSON artifacts")
    o.add_argument("--verbose", "-v", action="store_true", help="DEBUG logging")
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-12s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("websockets").setLevel(logging.WARNING)

    print(BANNER)

    run_cfg = RunConfig(
        coin=args.coin,
        duration=args.duration,
        channel=args.channel,
        status_interval=args.status_interval,
        show_book=args.show_book,
        export_dir=args.export,
        verbose=args.verbose,
    )
    as_params = ASParams(
        gamma=args.gamma,
        kappa=args.kappa,
        order_size=args.order_size,
        max_inventory=args.max_inventory,
        risk_horizon=args.risk_horizon,
        tick_size=args.tick_size,
        micro_weight=args.micro_weight,
    )
    sim_cfg = SimConfig(
        maker_fee_bps=args.maker_fee_bps,
        latency_ms=args.latency_ms,
        max_inventory=args.max_inventory,
        rng_seed=args.seed,
    )
    features = FeatureEngine(
        vol_window=args.vol_window,
        ofi_halflife=args.ofi_halflife,
        trend_halflife=args.trend_halflife,
        min_samples=args.min_samples,
    )

    # --- Record mode: capture only, no trading -------------------------------
    if args.record is not None:
        from src.recorder import MarketDataRecorder

        rec = MarketDataRecorder(args.coin, channel=args.channel, out_dir=args.record)
        dur = args.duration if args.duration > 0 else None
        try:
            asyncio.run(rec.record(duration=dur))
        except KeyboardInterrupt:
            logger.warning("recording interrupted by user")
            rec.close()
            return 130
        return 0

    # --- Replay mode (backtest) vs live session ------------------------------
    try:
        if args.replay is not None:
            sim = run_replay(args.replay, run_cfg, as_params, sim_cfg, features)
        else:
            sim = asyncio.run(run_session(run_cfg, as_params, sim_cfg, features))
    except KeyboardInterrupt:
        logger.warning("interrupted by user")
        return 130
    except FileNotFoundError as exc:
        logger.error("replay file not found: %s", exc)
        return 1
    except Exception:  # noqa: BLE001
        logger.exception("fatal error")
        return 1

    print()
    print(sim.report().render())

    if run_cfg.export_dir is not None:
        _export(sim, run_cfg.export_dir, run_cfg.coin)
    return 0


if __name__ == "__main__":
    sys.exit(main())