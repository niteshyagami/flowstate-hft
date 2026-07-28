"""
record_multi.py
===============

Record several instruments concurrently, each to its own capture file.

The single-coin path (``python main.py --record``) opens one WebSocket and
writes one file. For a cross-asset study you want BTC, ETH, SOL, ... captured
over the *same* wall-clock window so their books are directly comparable. This
runs one :class:`MarketDataRecorder` per coin as independent asyncio tasks
sharing one event loop — each owns its own socket and reconnect logic, so a
disconnect on SOL never disturbs the BTC capture.

Usage
-----
::

    # 3 hours of the six majors, into captures/
    python record_multi.py --coins BTC ETH SOL AVAX DOGE XRP --duration 10800

    # quick 5-minute smoke test
    python record_multi.py --coins BTC ETH SOL --duration 300

Each coin produces its own ``<coin>-bbo-<stamp>.jsonl.gz`` (rotating hourly).
A combined status line prints every 30 s so you can watch all captures at once.

Everything here is capture-only. No strategy, no quotes, no orders.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
import time
from pathlib import Path

from src.recorder import MarketDataRecorder

logger = logging.getLogger("record_multi")


async def _run(coins: list[str], duration: float | None, channel: str,
               out_dir: str, status_interval: float) -> None:
    """Launch one recorder per coin and supervise them until done."""
    recorders = [
        MarketDataRecorder(coin, channel=channel, out_dir=out_dir)
        for coin in coins
    ]
    started = time.time()

    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, AttributeError, ValueError):
            loop.add_signal_handler(sig, shutdown.set)

    # One capture task per coin.
    tasks = [
        asyncio.create_task(rec.record(duration=duration), name=f"rec-{rec.coin}")
        for rec in recorders
    ]

    async def _status() -> None:
        """Print a combined per-coin heartbeat until shutdown."""
        while not shutdown.is_set():
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(shutdown.wait(), timeout=status_interval)
                break
            elapsed = time.time() - started
            parts = " | ".join(
                f"{r.coin}:{r.stats.frames}f/{r.stats.bytes_written / 1e6:.1f}MB"
                for r in recorders
            )
            logger.info("t=%6.0fs | %s", elapsed, parts)

    status = asyncio.create_task(_status(), name="status")

    # Wait for the duration budget or a Ctrl-C.
    with contextlib.suppress(asyncio.TimeoutError):
        if duration is not None:
            await asyncio.wait_for(shutdown.wait(), timeout=duration + 5)
        else:
            await shutdown.wait()

    shutdown.set()
    for rec in recorders:
        rec.close()
    for t in (*tasks, status):
        t.cancel()
    await asyncio.gather(*tasks, status, return_exceptions=True)

    # Final tally.
    total_frames = sum(r.stats.frames for r in recorders)
    total_mb = sum(r.stats.bytes_written for r in recorders) / 1e6
    logger.info("=" * 60)
    logger.info("ALL CAPTURES DONE — %.1f min, %d frames, %.2f MB total",
                (time.time() - started) / 60.0, total_frames, total_mb)
    for r in recorders:
        logger.info("  %-6s %6d frames  %6.2f MB  ->  %s",
                    r.coin, r.stats.frames, r.stats.bytes_written / 1e6,
                    r.current_path.name if r.current_path else "(none)")
    logger.info("=" * 60)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        prog="record_multi",
        description="Record several Hyperliquid instruments concurrently (capture only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--coins", nargs="+", default=["BTC", "ETH", "SOL"],
                   help="symbols to record in parallel")
    p.add_argument("--duration", type=float, default=10800.0,
                   help="seconds to record (0 = until Ctrl-C)")
    p.add_argument("--channel", default="bbo", choices=["l2Book", "bbo"],
                   help="feed type; bbo is higher frequency")
    p.add_argument("--out-dir", default="captures", help="output directory")
    p.add_argument("--status-interval", type=float, default=30.0, help="heartbeat period")
    p.add_argument("--verbose", "-v", action="store_true", help="DEBUG logging")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-14s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("websockets").setLevel(logging.WARNING)
    # Quiet the per-recorder 500-frame chatter; the combined status line covers it.
    logging.getLogger("src.recorder").setLevel(logging.WARNING)

    coins = [c.upper() for c in args.coins]
    dur = args.duration if args.duration > 0 else None
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    logger.info("recording %s for %s to %s/",
                ", ".join(coins),
                f"{dur / 60:.0f} min" if dur else "until Ctrl-C",
                args.out_dir)

    try:
        asyncio.run(_run(coins, dur, args.channel, args.out_dir, args.status_interval))
    except KeyboardInterrupt:
        logger.warning("interrupted by user")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())