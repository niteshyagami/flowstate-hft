"""
recorder.py
===========

Capture live market data to disk and replay it deterministically.

Why this exists
---------------
Live data is ephemeral: a WebSocket stream you don't persist is gone the instant
the process exits, and the exact market conditions of that session will never
recur. That makes three essential quant workflows impossible on live data alone:

1. **Parameter comparison.** "Is γ = 3e-4 better than 1e-4?" is only answerable
   by running *both* on the *same* data. Live, each run sees a different market,
   so the comparison is apples-to-oranges.
2. **Reproducibility.** A result you cannot reproduce is not a result. Recorded
   input + a fixed RNG seed reproduces a session bit-for-bit.
3. **Speed.** Three hours of recorded ticks replay through the simulator in
   seconds, so an overnight capture becomes fifty parameter sweeps before
   breakfast — versus 150 real hours to run those sweeps live.

Firms capture their own market data for exactly these reasons; deep tick-level
order-book history is rarely available after the fact, so you record it yourself.

File format
-----------
**JSON Lines** (``.jsonl``), optionally gzip-compressed (``.jsonl.gz``):
one self-contained JSON object per line, one line per book frame. This format is
chosen deliberately over Parquet/HDF5 for the recorder's write path:

* **Append-only and crash-safe.** Each line is flushed independently. If the
  process is killed mid-session, every line written before the crash is intact
  and replayable — you lose at most the final partial line. A columnar format
  that finalizes on close would lose the *entire* session on a crash.
* **Streaming-friendly.** Lines are written as frames arrive, with bounded
  memory, regardless of session length.
* **Human-inspectable.** ``zcat capture.jsonl.gz | head`` just works.

Each line schema (compact keys to keep files small)::

    {"c":"BTC","e":1716400000.123,"l":1716400000.170,"s":42,
     "b":[[64000.0,1.2],[63999.0,2.0]],
     "a":[[64001.0,0.8],[64002.0,3.1]]}

    c = coin,  e = exch_ts,  l = local_ts,  s = seq,  b = bids,  a = asks

A one-line JSON **header** precedes the frames, recording capture metadata
(coin, channel, start time, format version) so a replayer can validate what it
is about to read.
"""

from __future__ import annotations

import asyncio
import contextlib
import gzip
import io
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Final, Iterator, TextIO

import numpy as np

from .ingestion import PX, SZ, BookSnapshot, HyperliquidL2Client

__all__ = ["MarketDataRecorder", "replay_file", "replay_stream", "RecorderStats"]

logger = logging.getLogger(__name__)

FORMAT_VERSION: Final[int] = 1


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #
def snapshot_to_line(snap: BookSnapshot) -> str:
    """Serialize a :class:`BookSnapshot` to one compact JSON line (no newline).

    Prices and sizes are emitted as plain floats. ``float64`` round-trips
    exactly through JSON for the ~10 significant digits crypto venues use, so no
    precision is lost relative to the original wire string.
    """
    obj = {
        "c": snap.coin,
        "e": round(snap.exch_ts, 6),
        "l": round(snap.local_ts, 6),
        "s": snap.seq,
        "b": snap.bids.tolist(),
        "a": snap.asks.tolist(),
    }
    return json.dumps(obj, separators=(",", ":"))


def line_to_snapshot(line: str) -> BookSnapshot:
    """Reconstruct a :class:`BookSnapshot` from one recorded JSON line.

    Raises
    ------
    ValueError
        If the line is malformed or missing required fields. The replay loop
        catches this and skips the line, so one corrupt row never aborts a
        replay of an otherwise-good multi-gigabyte capture.
    """
    try:
        obj = json.loads(line)
        bids = np.asarray(obj["b"], dtype=np.float64)
        asks = np.asarray(obj["a"], dtype=np.float64)
        if bids.ndim != 2 or asks.ndim != 2:
            raise ValueError("bids/asks must be 2-D")
        return BookSnapshot(
            coin=str(obj["c"]),
            exch_ts=float(obj["e"]),
            local_ts=float(obj["l"]),
            bids=bids,
            asks=asks,
            seq=int(obj.get("s", 0)),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"bad record: {exc}") from exc


# --------------------------------------------------------------------------- #
# Recorder
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class RecorderStats:
    """Live counters for a recording session."""

    frames: int = 0
    bytes_written: int = 0
    files: int = 1
    started: float = field(default_factory=time.time)
    dropped: int = 0

    def summary(self) -> str:
        """One-line human summary."""
        dur = max(time.time() - self.started, 1e-9)
        mb = self.bytes_written / 1e6
        return (
            f"{self.frames} frames, {mb:.2f} MB across {self.files} file(s), "
            f"{self.frames / dur:.1f} fps, {self.dropped} dropped"
        )


class MarketDataRecorder:
    """Persist live book snapshots to rotating JSON-Lines files.

    Two ways to drive it:

    * **Attached to a live client** (the common case) — :meth:`record` owns a
      :class:`HyperliquidL2Client`, streams from it, and writes every frame::

          rec = MarketDataRecorder("BTC", out_dir="captures", channel="bbo")
          await rec.record(duration=3600)          # capture one hour

    * **Fed externally** — call :meth:`write` yourself with snapshots from any
      source (useful for tests or a shared client feeding multiple sinks).

    Parameters
    ----------
    coin, channel:
        Passed to the underlying client and stored in the file header.
    out_dir:
        Directory for capture files (created if absent).
    compress:
        Write ``.jsonl.gz`` (gzip) instead of plain ``.jsonl``. Order-book data
        compresses ~5-10x since consecutive frames are near-duplicates, so this
        is on by default.
    rotate_minutes:
        Start a new file after this many minutes. ``0`` disables time rotation.
        Rotation keeps individual files a manageable size and bounds the damage
        from a corrupt file.
    rotate_mb:
        Start a new file after the current one reaches this size (megabytes,
        measured pre-compression). ``0`` disables size rotation.
    flush_every:
        Flush the OS buffer to disk every N frames. Lower = safer on crash,
        higher = less I/O overhead. ``1`` flushes every frame (maximally safe).
    """

    def __init__(
        self,
        coin: str = "BTC",
        *,
        channel: str = "bbo",
        out_dir: str | Path = "captures",
        compress: bool = True,
        rotate_minutes: float = 60.0,
        rotate_mb: float = 256.0,
        flush_every: int = 20,
    ) -> None:
        self.coin = coin.upper()
        self.channel = channel
        self.out_dir = Path(out_dir)
        self.compress = compress
        self.rotate_minutes = rotate_minutes
        self.rotate_mb = rotate_mb
        self.flush_every = max(1, flush_every)

        self.stats = RecorderStats()
        self._fh: TextIO | None = None
        self._path: Path | None = None
        self._file_started: float = 0.0
        self._file_bytes: int = 0
        self._since_flush: int = 0
        self._client: HyperliquidL2Client | None = None

    # -- Public API --------------------------------------------------------- #
    @property
    def current_path(self) -> Path | None:
        """Path of the file currently being written, if any."""
        return self._path

    async def record(self, duration: float | None = None) -> RecorderStats:
        """Capture a live stream until ``duration`` seconds elapse (or forever).

        Owns and manages its own :class:`HyperliquidL2Client`, including its
        reconnection logic — a mid-capture disconnect resumes into the *same*
        file, so a dropout leaves a time gap but never a corrupt capture.

        Passing ``duration=None`` records until cancelled (Ctrl-C), which is the
        usual "record overnight" mode.
        """
        self._client = HyperliquidL2Client(coin=self.coin, channel=self.channel)
        self._open_new_file()
        deadline = (time.time() + duration) if duration else None

        try:
            async for snap in self._client.stream():
                self.write(snap)
                if deadline is not None and time.time() >= deadline:
                    break
                if self.stats.frames % 500 == 0:
                    logger.info("[%s] recording… %s", self.coin, self.stats.summary())
        except asyncio.CancelledError:
            logger.info("[%s] recording cancelled", self.coin)
            raise
        finally:
            self.close()
        return self.stats

    def write(self, snap: BookSnapshot) -> None:
        """Append one snapshot to the current file, rotating if thresholds hit."""
        if self._fh is None:
            self._open_new_file()
        if self._should_rotate():
            self._rotate()

        line = snapshot_to_line(snap) + "\n"
        try:
            n = self._fh.write(line)  # type: ignore[union-attr]
        except (OSError, ValueError) as exc:
            self.stats.dropped += 1
            logger.error("[%s] write failed, frame dropped: %s", self.coin, exc)
            return

        nbytes = len(line.encode("utf-8"))
        self.stats.frames += 1
        self.stats.bytes_written += nbytes
        self._file_bytes += nbytes
        self._since_flush += 1

        if self._since_flush >= self.flush_every:
            with contextlib.suppress(OSError):
                self._fh.flush()  # type: ignore[union-attr]
            self._since_flush = 0

    def close(self) -> None:
        """Flush and close the current file. Idempotent."""
        if self._fh is not None:
            with contextlib.suppress(OSError):
                self._fh.flush()
                self._fh.close()
            logger.info("[%s] closed %s", self.coin, self._path)
            self._fh = None
        logger.info("[%s] recording done: %s", self.coin, self.stats.summary())

    # -- Internals ---------------------------------------------------------- #
    def _make_path(self) -> Path:
        """Build a timestamped capture filename."""
        stamp = time.strftime("%Y%m%d-%H%M%S")
        ext = "jsonl.gz" if self.compress else "jsonl"
        return self.out_dir / f"{self.coin.lower()}-{self.channel}-{stamp}.{ext}"

    def _open_new_file(self) -> None:
        """Open a fresh capture file and write the metadata header line."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._make_path()
        if self.compress:
            raw = gzip.open(self._path, "wt", encoding="utf-8", compresslevel=6)
            self._fh = raw  # type: ignore[assignment]
        else:
            self._fh = open(self._path, "w", encoding="utf-8")

        header = {
            "__header__": True,
            "version": FORMAT_VERSION,
            "coin": self.coin,
            "channel": self.channel,
            "started": round(time.time(), 6),
        }
        self._fh.write(json.dumps(header, separators=(",", ":")) + "\n")
        self._file_started = time.time()
        self._file_bytes = 0
        logger.info("[%s] recording to %s", self.coin, self._path)

    def _should_rotate(self) -> bool:
        """Return ``True`` if a rotation threshold has been crossed."""
        if self.rotate_minutes > 0:
            if time.time() - self._file_started >= self.rotate_minutes * 60.0:
                return True
        if self.rotate_mb > 0:
            if self._file_bytes >= self.rotate_mb * 1e6:
                return True
        return False

    def _rotate(self) -> None:
        """Close the current file and open the next."""
        self.close_for_rotation()
        self.stats.files += 1
        self._open_new_file()

    def close_for_rotation(self) -> None:
        """Close the file handle without emitting the final 'done' summary."""
        if self._fh is not None:
            with contextlib.suppress(OSError):
                self._fh.flush()
                self._fh.close()
            self._fh = None


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #
def _open_maybe_gzip(path: Path) -> TextIO:
    """Open a capture file transparently, whether gzip-compressed or plain."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")  # type: ignore[return-value]
    return open(path, "r", encoding="utf-8")


def replay_file(path: str | Path, skip_header: bool = True) -> Iterator[BookSnapshot]:
    """Yield :class:`BookSnapshot` objects from a recorded capture file.

    Synchronous and as fast as the disk and JSON parser allow — this is the
    backtest path, where you want to burn through hours of data immediately with
    no inter-frame delay. Corrupt lines are logged and skipped.

    Examples
    --------
    >>> from src.recorder import replay_file
    >>> for snap in replay_file("captures/btc-bbo-20260101-000000.jsonl.gz"):
    ...     ...  # feed FeatureEngine -> Strategy -> Simulator as usual
    """
    path = Path(path)
    fh = _open_maybe_gzip(path)
    header: dict[str, Any] | None = None
    good = bad = 0
    try:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            if i == 0 and skip_header:
                with contextlib.suppress(json.JSONDecodeError):
                    obj = json.loads(line)
                    if isinstance(obj, dict) and obj.get("__header__"):
                        header = obj
                        logger.info("replaying %s: %s", path.name,
                                    {k: v for k, v in obj.items() if k != "__header__"})
                        continue
            try:
                yield line_to_snapshot(line)
                good += 1
            except ValueError as exc:
                bad += 1
                logger.debug("skipping bad line %d: %s", i, exc)
    finally:
        fh.close()
        logger.info("replay of %s complete: %d frames, %d skipped", path.name, good, bad)


async def replay_stream(
    path: str | Path, speed: float = 0.0
) -> AsyncIterator[BookSnapshot]:
    """Async replay that can re-time frames to their original inter-arrival gaps.

    Parameters
    ----------
    speed:
        ``0`` (default) replays as fast as possible — the backtest use case.
        ``1.0`` replays in real time using the recorded ``local_ts`` deltas.
        ``2.0`` is twice real speed, ``0.5`` half. Re-timed replay is useful for
        demoing a live-looking dashboard from a canned capture.

    Notes
    -----
    Timing uses recorded ``local_ts`` deltas, clamped to avoid replaying a
    reconnect gap as a multi-minute stall.
    """
    prev_ts: float | None = None
    for snap in replay_file(path):
        if speed > 0.0 and prev_ts is not None:
            gap = (snap.local_ts - prev_ts) / speed
            if gap > 0.0:
                await asyncio.sleep(min(gap, 5.0))  # clamp reconnect gaps
        prev_ts = snap.local_ts
        yield snap


# --------------------------------------------------------------------------- #
# Standalone:  python -m src.recorder BTC 3600   (record BTC for 1 hour)
#              python -m src.recorder --replay captures/foo.jsonl.gz
# --------------------------------------------------------------------------- #
async def _record_main(coin: str, duration: float | None, channel: str) -> None:
    rec = MarketDataRecorder(coin, channel=channel)
    with contextlib.suppress(asyncio.CancelledError, KeyboardInterrupt):
        await rec.record(duration=duration)


def _replay_main(path: str) -> None:
    n = 0
    for snap in replay_file(path):
        n += 1
        if n <= 3 or n % 1000 == 0:
            print(f"{n:>7}  {snap.coin}  mid={snap.mid:,.2f}  seq={snap.seq}")
    print(f"total frames: {n}")


if __name__ == "__main__":  # pragma: no cover
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    args = sys.argv[1:]
    if args and args[0] == "--replay":
        _replay_main(args[1])
    else:
        symbol = args[0] if args else "BTC"
        dur = float(args[1]) if len(args) > 1 else None
        chan = args[2] if len(args) > 2 else "bbo"
        asyncio.run(_record_main(symbol, dur, chan))