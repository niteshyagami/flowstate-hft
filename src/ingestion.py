"""
ingestion.py
============

Asynchronous, self-healing WebSocket market-data client for the Hyperliquid
decentralized perpetuals exchange.

Responsibilities
----------------
1. Maintain a persistent WebSocket session against ``wss://api.hyperliquid.xyz/ws``.
2. Subscribe to the ``l2Book`` channel for a configurable coin (e.g. ``BTC``).
3. Parse the raw JSON payload and *normalize* it into a zero-copy-friendly
   :class:`BookSnapshot` backed by ``numpy`` arrays.
4. Survive network partitions via exponential backoff with jitter, plus an
   application-level staleness watchdog (silent-socket detection).

Wire format
-----------
Hyperliquid pushes L2 book updates shaped as::

    {
      "channel": "l2Book",
      "data": {
        "coin": "BTC",
        "time": 1716400000123,                 # exchange ms epoch
        "levels": [
          [ {"px": "64000.0", "sz": "1.25", "n": 3}, ... ],   # index 0 -> BIDS
          [ {"px": "64001.0", "sz": "0.80", "n": 2}, ... ]    # index 1 -> ASKS
        ]
      }
    }

Bids arrive descending in price, asks ascending. We preserve that ordering
invariant and assert it on ingest, because every downstream feature
(micro-price, OFI, imbalance) assumes ``levels[0]`` is the inside quote.

Design notes
------------
* **Non-blocking end-to-end.** The reader never awaits a consumer. Snapshots are
  handed off through an ``asyncio.Queue`` with a bounded ``maxsize``; on
  backpressure we drop the *oldest* frame rather than stall the socket. In
  market making, a stale book is worse than no book.
* **Parse cost.** String→float conversion dominates. We build a single
  ``np.empty((n, 2), dtype=np.float64)`` per side and fill it in one pass
  rather than constructing intermediate Python lists of tuples.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Final, Sequence

import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatus, WebSocketException

__all__ = ["BookSnapshot", "HyperliquidL2Client", "HYPERLIQUID_WS_URL", "parse_l2_book", "parse_bbo"]

logger = logging.getLogger(__name__)

HYPERLIQUID_WS_URL: Final[str] = "wss://api.hyperliquid.xyz/ws"

# Column indices into the (n, 2) level matrices.
PX: Final[int] = 0
SZ: Final[int] = 1


# --------------------------------------------------------------------------- #
# Normalized data model
# --------------------------------------------------------------------------- #
@dataclass(slots=True, frozen=True)
class BookSnapshot:
    """An immutable, normalized L2 order-book snapshot.

    Attributes
    ----------
    coin:
        Instrument symbol, e.g. ``"BTC"``.
    exch_ts:
        Exchange-stamped event time in **seconds** (float epoch).
    local_ts:
        Local monotonic-adjusted receipt time in seconds. ``local_ts - exch_ts``
        is a usable (clock-skew-contaminated) proxy for one-way latency.
    bids:
        ``(n, 2)`` float64 array, column 0 = price, column 1 = size.
        Sorted **descending** by price, so ``bids[0]`` is the best bid.
    asks:
        ``(m, 2)`` float64 array, sorted **ascending** by price, so ``asks[0]``
        is the best ask.
    seq:
        Monotonically increasing local sequence number assigned by the client.
        Used by OFI to detect dropped/reordered frames.
    """

    coin: str
    exch_ts: float
    local_ts: float
    bids: np.ndarray
    asks: np.ndarray
    seq: int = 0

    # -- Derived quantities (cheap, computed on demand) --------------------- #
    @property
    def best_bid(self) -> float:
        """Price of the highest resting buy order."""
        return float(self.bids[0, PX])

    @property
    def best_ask(self) -> float:
        """Price of the lowest resting sell order."""
        return float(self.asks[0, PX])

    @property
    def best_bid_size(self) -> float:
        """Aggregate size resting at the best bid."""
        return float(self.bids[0, SZ])

    @property
    def best_ask_size(self) -> float:
        """Aggregate size resting at the best ask."""
        return float(self.asks[0, SZ])

    @property
    def mid(self) -> float:
        r"""Arithmetic mid price :math:`m = (p^{bid}_0 + p^{ask}_0) / 2`."""
        return 0.5 * (self.best_bid + self.best_ask)

    @property
    def spread(self) -> float:
        """Absolute top-of-book spread in quote currency."""
        return self.best_ask - self.best_bid

    @property
    def depth(self) -> int:
        """Number of usable levels (min of the two sides)."""
        return int(min(self.bids.shape[0], self.asks.shape[0]))

    def is_crossed(self) -> bool:
        """Return ``True`` if the book is crossed or locked (data corruption)."""
        return self.best_bid >= self.best_ask

    def format_ladder(self, levels: int = 5) -> str:
        """Render the top ``levels`` of each side as an aligned ASCII ladder.

        Bids are printed best-first on the left, asks best-first on the right,
        which is how a trader reads a depth-of-market widget.
        """
        n = min(levels, self.depth)
        header = (
            f"{self.coin}  mid={self.mid:,.2f}  spread={self.spread:.2f}  "
            f"seq={self.seq}  lat={1e3 * (self.local_ts - self.exch_ts):.1f}ms"
        )
        rule = "-" * 62
        lines = [header, rule, f"{'BID SIZE':>12} {'BID PX':>14} | {'ASK PX':<14} {'ASK SIZE':<12}", rule]
        for i in range(n):
            lines.append(
                f"{self.bids[i, SZ]:>12.4f} {self.bids[i, PX]:>14,.2f} | "
                f"{self.asks[i, PX]:<14,.2f} {self.asks[i, SZ]:<12.4f}"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def _levels_to_array(raw_levels: Sequence[dict[str, Any]], max_levels: int) -> np.ndarray:
    """Convert Hyperliquid level dicts into an ``(n, 2)`` float64 matrix.

    Parameters
    ----------
    raw_levels:
        Sequence of ``{"px": str, "sz": str, "n": int}`` dicts.
    max_levels:
        Hard cap on retained depth. Truncating early keeps the working set
        inside L1/L2 cache for the feature stage.

    Raises
    ------
    ValueError
        If the side is empty (a one-sided book is unusable for quoting) or if a
        price/size field is malformed.
    """
    n = min(len(raw_levels), max_levels)
    if n == 0:
        raise ValueError("empty book side")

    out = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        lvl = raw_levels[i]
        # float() on the raw str is ~2x faster than Decimal and precision is
        # ample: float64 gives 15-16 significant digits vs. crypto's ~10.
        out[i, PX] = float(lvl["px"])
        out[i, SZ] = float(lvl["sz"])
    return out


def parse_l2_book(payload: dict[str, Any], max_levels: int, seq: int) -> BookSnapshot:
    """Normalize a decoded ``l2Book`` frame into a :class:`BookSnapshot`.

    Raises
    ------
    ValueError
        On any structural violation (missing keys, empty side, crossed book).
        Callers are expected to log-and-skip rather than terminate: a single
        corrupt frame must not take down the quoting loop.
    """
    data = payload["data"]
    levels = data["levels"]
    if len(levels) < 2:
        raise ValueError(f"expected 2 book sides, got {len(levels)}")

    bids = _levels_to_array(levels[0], max_levels)
    asks = _levels_to_array(levels[1], max_levels)

    # Enforce ordering invariants downstream code relies on.
    if bids.shape[0] > 1 and bids[0, PX] < bids[1, PX]:
        bids = bids[np.argsort(-bids[:, PX])]
    if asks.shape[0] > 1 and asks[0, PX] > asks[1, PX]:
        asks = asks[np.argsort(asks[:, PX])]

    snap = BookSnapshot(
        coin=str(data.get("coin", "UNKNOWN")),
        exch_ts=float(data.get("time", 0)) / 1e3,
        local_ts=time.time(),
        bids=bids,
        asks=asks,
        seq=seq,
    )
    if snap.is_crossed():
        raise ValueError(f"crossed book: bid={snap.best_bid} >= ask={snap.best_ask}")
    return snap


def parse_bbo(payload: dict[str, Any], seq: int) -> BookSnapshot:
    """Normalize a ``bbo`` (best bid/offer) frame into a :class:`BookSnapshot`.

    The ``bbo`` channel is *far* more frequent than ``l2Book`` — it fires on
    every top-of-book change on a block, whereas ``l2Book`` is throttled. The
    trade-off is depth: a bbo frame carries exactly one level per side, so the
    resulting snapshot has single-row ``bids``/``asks`` matrices. Every
    downstream consumer already handles arbitrary depth via ``snapshot.depth``,
    so this drops in transparently — the depth-weighted micro-price simply
    collapses to the level-1 micro-price.

    Wire shape (per Hyperliquid docs, ``WsBbo``)::

        {"channel": "bbo",
         "data": {"coin": "BTC", "time": <ms>,
                  "bbo": [ {"px","sz","n"} | null,   # bid
                           {"px","sz","n"} | null ]} # ask

    Raises
    ------
    ValueError
        If either side is ``null`` (a one-sided book cannot be quoted around)
        or the book is crossed.
    """
    data = payload["data"]
    bbo = data["bbo"]
    if len(bbo) < 2 or bbo[0] is None or bbo[1] is None:
        raise ValueError("bbo frame missing a side")

    bid = np.array([[float(bbo[0]["px"]), float(bbo[0]["sz"])]], dtype=np.float64)
    ask = np.array([[float(bbo[1]["px"]), float(bbo[1]["sz"])]], dtype=np.float64)

    snap = BookSnapshot(
        coin=str(data.get("coin", "UNKNOWN")),
        exch_ts=float(data.get("time", 0)) / 1e3,
        local_ts=time.time(),
        bids=bid,
        asks=ask,
        seq=seq,
    )
    if snap.is_crossed():
        raise ValueError(f"crossed bbo: bid={snap.best_bid} >= ask={snap.best_ask}")
    return snap


# --------------------------------------------------------------------------- #
# WebSocket client
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class _ConnStats:
    """Lightweight counters for operational observability."""

    frames: int = 0
    parse_errors: int = 0
    reconnects: int = 0
    drops: int = 0
    connected_at: float = field(default_factory=time.time)


class HyperliquidL2Client:
    """Self-healing L2 order-book streamer.

    Usage
    -----
    Either consume it as an async iterator::

        client = HyperliquidL2Client(coin="BTC")
        async for snapshot in client.stream():
            print(snapshot.format_ladder(5))

    ...or run it as a background producer feeding an external queue::

        task = asyncio.create_task(client.run(queue))

    Parameters
    ----------
    coin:
        Instrument symbol as listed on Hyperliquid (``"BTC"``, ``"ETH"``, ...).
    url:
        WebSocket endpoint. Overridable for testing against a local mock.
    max_levels:
        Depth retained per side after truncation.
    queue_size:
        Bounded handoff buffer. On overflow the *oldest* snapshot is discarded.
    stale_timeout:
        Seconds without any inbound frame before the socket is presumed dead and
        force-recycled. Hyperliquid is chatty; silence means a black-hole route.
    max_backoff:
        Ceiling on exponential reconnect delay, in seconds.
    """

    def __init__(
        self,
        coin: str = "BTC",
        url: str = HYPERLIQUID_WS_URL,
        *,
        channel: str = "l2Book",
        max_levels: int = 20,
        queue_size: int = 256,
        stale_timeout: float = 30.0,
        max_backoff: float = 30.0,
        ping_interval: float = 20.0,
    ) -> None:
        if channel not in ("l2Book", "bbo"):
            raise ValueError("channel must be 'l2Book' or 'bbo'")
        self.coin = coin.upper()
        self.url = url
        self.channel = channel
        self.max_levels = max_levels
        self.queue_size = queue_size
        self.stale_timeout = stale_timeout
        self.max_backoff = max_backoff
        self.ping_interval = ping_interval

        self._seq = 0
        self._stop = asyncio.Event()
        self.stats = _ConnStats()

    # -- Public API --------------------------------------------------------- #
    def stop(self) -> None:
        """Signal the reader loop to unwind at the next opportunity."""
        self._stop.set()

    @property
    def subscription_message(self) -> str:
        """The JSON subscribe frame for this client's coin and channel."""
        return json.dumps(
            {"method": "subscribe",
             "subscription": {"type": self.channel, "coin": self.coin}}
        )

    async def run(self, queue: asyncio.Queue[BookSnapshot]) -> None:
        """Produce snapshots into ``queue`` forever, reconnecting as needed.

        Never raises on transport errors; it logs, backs off, and retries. Only
        :class:`asyncio.CancelledError` propagates.
        """
        attempt = 0
        while not self._stop.is_set():
            try:
                await self._session(queue)
                attempt = 0  # clean exit -> reset backoff ladder
            except asyncio.CancelledError:
                raise
            except (ConnectionClosed, InvalidStatus, WebSocketException, OSError) as exc:
                attempt += 1
                self.stats.reconnects += 1
                delay = self._backoff(attempt)
                logger.warning(
                    "[%s] transport error (%s): %s — reconnect #%d in %.2fs",
                    self.coin, type(exc).__name__, exc, attempt, delay,
                )
                await self._sleep_or_stop(delay)
            except Exception as exc:  # noqa: BLE001 - last line of defence
                attempt += 1
                delay = self._backoff(attempt)
                logger.exception("[%s] unexpected reader failure: %s", self.coin, exc)
                await self._sleep_or_stop(delay)

        logger.info("[%s] ingestion stopped. %s", self.coin, self.stats)

    async def stream(self) -> AsyncIterator[BookSnapshot]:
        """Yield snapshots as an async iterator (convenience wrapper over :meth:`run`)."""
        queue: asyncio.Queue[BookSnapshot] = asyncio.Queue(maxsize=self.queue_size)
        producer = asyncio.create_task(self.run(queue), name=f"ingest-{self.coin}")
        try:
            while not (self._stop.is_set() and queue.empty()):
                yield await queue.get()
        finally:
            producer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await producer

    # -- Internals ---------------------------------------------------------- #
    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with full jitter.

        .. math:: d = \\min(\\text{max\\_backoff},\\, 2^{a-1}) \\cdot U(0.5, 1.5)

        Full jitter prevents thundering-herd reconnects when the venue itself
        bounces and every bot in the world retries on the same tick.
        """
        base = min(self.max_backoff, 2.0 ** max(0, attempt - 1))
        return base * random.uniform(0.5, 1.5)

    async def _sleep_or_stop(self, delay: float) -> None:
        """Sleep for ``delay`` seconds, waking early if :meth:`stop` is called."""
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._stop.wait(), timeout=delay)

    async def _session(self, queue: asyncio.Queue[BookSnapshot]) -> None:
        """One connect → subscribe → read-loop lifecycle."""
        logger.info("[%s] connecting to %s", self.coin, self.url)
        async with websockets.connect(
            self.url,
            ping_interval=self.ping_interval,
            ping_timeout=self.ping_interval / 2,
            close_timeout=5.0,
            max_queue=self.queue_size,
        ) as ws:
            await ws.send(self.subscription_message)
            self.stats.connected_at = time.time()
            logger.info("[%s] subscribed to %s", self.coin, self.channel)

            while not self._stop.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=self.stale_timeout)
                except asyncio.TimeoutError as exc:
                    raise ConnectionClosed(None, None) from exc  # force recycle

                snap = self._handle_frame(raw)
                if snap is not None:
                    self._offer(queue, snap)

    def _handle_frame(self, raw: str | bytes) -> BookSnapshot | None:
        """Decode one frame. Returns ``None`` for non-book or malformed frames."""
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            self.stats.parse_errors += 1
            logger.debug("[%s] undecodable frame dropped", self.coin)
            return None

        if not isinstance(payload, dict):
            return None

        channel = payload.get("channel")
        if channel == "subscriptionResponse":
            logger.debug("[%s] subscription ack: %s", self.coin, payload.get("data"))
            return None

        try:
            if channel == "l2Book":
                snap = parse_l2_book(payload, self.max_levels, self._seq)
            elif channel == "bbo":
                snap = parse_bbo(payload, self._seq)
            else:
                return None  # pongs, errors, other channels
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            self.stats.parse_errors += 1
            logger.debug("[%s] rejecting frame: %s", self.coin, exc)
            return None

        self._seq += 1
        self.stats.frames += 1
        return snap

    def _offer(self, queue: asyncio.Queue[BookSnapshot], snap: BookSnapshot) -> None:
        """Non-blocking enqueue with oldest-first eviction under backpressure."""
        try:
            queue.put_nowait(snap)
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty):
                queue.get_nowait()
                self.stats.drops += 1
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(snap)


# --------------------------------------------------------------------------- #
# Standalone smoke test:  python -m src.ingestion BTC
# --------------------------------------------------------------------------- #
async def _demo(coin: str, channel: str = "l2Book", ladder_levels: int = 5, max_frames: int = 25) -> None:
    """Print the top-of-book ladder for a handful of live frames."""
    client = HyperliquidL2Client(coin=coin, channel=channel)
    printed = 0
    async for snap in client.stream():
        print(snap.format_ladder(ladder_levels), end="\n\n")
        printed += 1
        if printed >= max_frames:
            client.stop()
            break


if __name__ == "__main__":  # pragma: no cover
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTC"
    chan = sys.argv[2] if len(sys.argv) > 2 else "l2Book"
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_demo(symbol, chan))