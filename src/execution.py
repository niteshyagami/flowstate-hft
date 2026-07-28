"""
execution.py
============

Local paper-trading simulator: fill modelling, inventory accounting, and
performance analytics. **Nothing in this module ever sends an order to a
venue.** It consumes live market data and simulated quotes only.

Fill model
----------
Naive backtests assume a resting quote fills the instant the market touches it.
That single assumption is responsible for most of the fake Sharpe ratios on
the internet, because it ignores queue position: when the best bid trades, the
orders *ahead of you* fill first.

We use a two-stage model.

**Stage 1 — crossing test.** A resting bid at :math:`p^{bid}` is only eligible
once the market's best ask has traded down to :math:`p^{bid}`, i.e.
:math:`a_t \\le p^{bid}`. (Mirror for asks.)

**Stage 2 — queue-position probability.** Conditional on eligibility, fill
probability decays in the depth resting ahead of us, and rises with how deeply
the market penetrated our price:

.. math::

    P(\\text{fill}) = \\underbrace{\\frac{1}{1 + Q_{\\text{ahead}} / \\bar{Q}}}_{\\text{queue}}
                     \\cdot
                     \\underbrace{\\left(1 - e^{-\\eta \\Delta / \\text{tick}}\\right)}_{\\text{penetration}}

where :math:`Q_{\\text{ahead}}` is the size resting at or better than our price
when we joined, :math:`\\bar{Q}` a normalizing depth scale, and
:math:`\\Delta = p^{bid} - a_t \\ge 0` the amount by which the market traded
through us. A quote that is merely touched fills rarely; one that is traded
several ticks through fills nearly always. This is deliberately conservative.

Adverse selection accounting
----------------------------
Every fill records the mid price at fill time and the mid :math:`h` seconds
later. **Markout** is the signed post-fill drift against the maker:

.. math::

    \\text{markout}_h = \\text{side} \\cdot (m_{t+h} - m_t) \\cdot \\text{size}

with ``side = +1`` for buys. Persistently negative markout is the definition of
adverse selection, and it is the metric this project is fundamentally about.

Performance metrics
-------------------
* **Sharpe** on PnL increments, annualized by :math:`\\sqrt{365\\cdot 86400/\\Delta t}`.
* **Max drawdown** on the mark-to-market equity curve.
* **Inventory variance** :math:`\\mathrm{Var}(q_t)` — the direct empirical proxy
  for how well the A-S reservation-price mechanism is working.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Final, Literal

import numpy as np
import pandas as pd

from .ingestion import BookSnapshot
from .strategy import Quote

__all__ = ["Fill", "SimConfig", "PerformanceReport", "PaperExecutionSimulator"]

logger = logging.getLogger(__name__)

_EPS: Final[float] = 1e-12
SECONDS_PER_YEAR: Final[float] = 365.0 * 86400.0

Side = Literal["buy", "sell"]


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Fill:
    """A single simulated execution.

    Attributes
    ----------
    ts:
        Fill timestamp (seconds, float epoch).
    side:
        ``"buy"`` if our bid was lifted, ``"sell"`` if our ask was hit.
    price / size:
        Execution price and base-asset quantity.
    mid_at_fill:
        Market mid at fill time, used as the markout baseline.
    fee:
        Signed cost in quote currency. Negative fees (maker rebates) increase PnL.
    inventory_after:
        Signed position immediately after this fill.
    markout:
        Post-fill drift, populated later by
        :meth:`PaperExecutionSimulator._settle_markouts`. ``NaN`` until settled.
    edge:
        Immediate captured spread, ``side_sign * (mid_at_fill - price) * size``.
        Positive means we bought below / sold above the prevailing mid.
    """

    ts: float
    side: Side
    price: float
    size: float
    mid_at_fill: float
    fee: float
    inventory_after: float
    markout: float = float("nan")
    edge: float = 0.0

    @property
    def signed_size(self) -> float:
        """``+size`` for buys, ``-size`` for sells."""
        return self.size if self.side == "buy" else -self.size


@dataclass(slots=True)
class SimConfig:
    """Simulator configuration.

    Attributes
    ----------
    maker_fee_bps:
        Maker fee in basis points of notional. Use a **negative** value for a
        rebate. Hyperliquid's maker tier is small but non-zero; assuming zero
        fees is a classic way to manufacture a fake edge.
    latency_ms:
        One-way quote-placement latency. A quote is not eligible to fill until
        ``latency_ms`` after it was computed, which correctly penalizes
        strategies that only look profitable with instant cancels.
    queue_depth_scale:
        :math:`\\bar Q` in the queue-position formula, in base units. Roughly
        "typical size resting at the touch".
    penetration_eta:
        :math:`\\eta` controlling how fast fill probability rises as the market
        trades through the quote.
    max_inventory:
        Hard risk limit. Fills that would breach it are rejected outright, which
        models the risk system your quotes sit behind in production.
    markout_horizon:
        Seconds after a fill at which markout is measured.
    rng_seed:
        Seed for the fill RNG. Fixed by default so backtests are reproducible.
    """

    maker_fee_bps: float = 0.0
    latency_ms: float = 25.0
    queue_depth_scale: float = 2.0
    penetration_eta: float = 1.0
    max_inventory: float = 0.10
    markout_horizon: float = 5.0
    rng_seed: int | None = 42


@dataclass(slots=True)
class PerformanceReport:
    """Aggregated session statistics."""

    n_fills: int
    n_buys: int
    n_sells: int
    fill_rate_per_min: float
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    fees_paid: float
    gross_edge: float
    sharpe: float
    max_drawdown: float
    max_drawdown_pct: float
    inventory_variance: float
    inventory_mean: float
    inventory_max_abs: float
    mean_markout: float
    adverse_selection_ratio: float
    duration_seconds: float
    turnover: float

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict (JSON/CSV friendly)."""
        return asdict(self)

    def render(self) -> str:
        """Pretty multi-line report for the console."""
        rows = [
            ("Duration", f"{self.duration_seconds:,.1f} s"),
            ("Fills (buy/sell)", f"{self.n_fills} ({self.n_buys}/{self.n_sells})"),
            ("Fill rate", f"{self.fill_rate_per_min:.2f} /min"),
            ("Turnover", f"{self.turnover:,.4f} base"),
            ("", ""),
            ("Gross edge", f"{self.gross_edge:+,.4f}"),
            ("Fees paid", f"{self.fees_paid:,.4f}"),
            ("Realized PnL", f"{self.realized_pnl:+,.4f}"),
            ("Unrealized PnL", f"{self.unrealized_pnl:+,.4f}"),
            ("TOTAL PnL", f"{self.total_pnl:+,.4f}"),
            ("", ""),
            ("Sharpe (annualized)", f"{self.sharpe:,.2f}"),
            ("Max drawdown", f"{self.max_drawdown:,.4f} ({self.max_drawdown_pct:.2f}%)"),
            ("Inventory variance", f"{self.inventory_variance:.6f}"),
            ("Inventory mean", f"{self.inventory_mean:+.5f}"),
            ("Inventory |max|", f"{self.inventory_max_abs:.5f}"),
            ("", ""),
            ("Mean markout", f"{self.mean_markout:+,.5f}"),
            ("Adverse-selection ratio", f"{self.adverse_selection_ratio:.2%}"),
        ]
        width = 62
        out = ["=" * width, "  FLOWSTATE HFT — PAPER TRADING PERFORMANCE".ljust(width), "=" * width]
        for k, v in rows:
            out.append("" if not k else f"  {k:<28} {v:>30}")
        out.append("=" * width)
        return "\n".join(out)


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #
class PaperExecutionSimulator:
    """Event-driven paper-fill engine with inventory and PnL accounting.

    Lifecycle per tick — **order matters**::

        fills = sim.on_book(snap)   # 1. test the RESTING quote against new data
        quote = strategy.quote(...) # 2. re-quote using post-fill inventory
        sim.submit(quote, snap)     # 3. rest the new quote (latency-gated)

    Testing before submitting is the correct causality: market data arrives,
    resting orders may fill, and only then does the maker re-quote. Submitting
    first would centre every quote on the very book it is then tested against,
    and nothing could ever fill. ``on_book`` calls :meth:`mark` internally.

    Inventory convention: ``q > 0`` is long base asset. Cash is tracked in quote
    currency, so total equity is ``cash + q * mid``.
    """

    def __init__(self, config: SimConfig | None = None) -> None:
        self.cfg = config or SimConfig()
        self._rng = np.random.default_rng(self.cfg.rng_seed)

        # Position state
        self.inventory: float = 0.0
        self.cash: float = 0.0
        self.fees_paid: float = 0.0
        self.gross_edge: float = 0.0
        self.turnover: float = 0.0

        # Resting quote state
        self._quote: Quote | None = None
        self._quote_ready_ts: float = 0.0
        self._bid_queue_ahead: float = 0.0
        self._ask_queue_ahead: float = 0.0

        # History
        self.fills: list[Fill] = []
        self._equity_ts: list[float] = []
        self._equity: list[float] = []
        self._inventory_path: list[float] = []
        self._pending_markout: Deque[tuple[Fill, float]] = deque()
        self._start_ts: float | None = None
        self._last_ts: float = 0.0
        self._last_mid: float = 0.0

    # -- Public API --------------------------------------------------------- #
    @property
    def position_value(self) -> float:
        """Mark-to-market value of the open position in quote currency."""
        return self.inventory * self._last_mid

    @property
    def equity(self) -> float:
        """Total account equity: cash plus marked position."""
        return self.cash + self.position_value

    def submit(self, quote: Quote, snapshot: BookSnapshot) -> None:
        """Register the strategy's desired quote, subject to placement latency.

        The queue-ahead estimate is snapshotted **at submission time**, which is
        the honest thing to do: you join the back of whatever queue exists when
        your order lands, not the queue that exists when it fills.
        """
        self._quote = quote
        self._quote_ready_ts = quote.ts + self.cfg.latency_ms / 1e3
        self._bid_queue_ahead = self._depth_at_or_better(snapshot, "buy", quote.bid_px)
        self._ask_queue_ahead = self._depth_at_or_better(snapshot, "sell", quote.ask_px)

    def on_book(self, snapshot: BookSnapshot) -> list[Fill]:
        """Test resting quotes against a new book state; return any fills.

        Also advances mark-to-market and settles matured markouts.
        """
        if self._start_ts is None:
            self._start_ts = snapshot.local_ts
        self._last_ts = snapshot.local_ts
        self._last_mid = snapshot.mid

        fills: list[Fill] = []
        q = self._quote
        if q is not None and snapshot.local_ts >= self._quote_ready_ts:
            if q.bid_px is not None:
                f = self._try_fill(snapshot, "buy", q.bid_px, q.bid_sz, self._bid_queue_ahead)
                if f is not None:
                    fills.append(f)
            if q.ask_px is not None:
                f = self._try_fill(snapshot, "sell", q.ask_px, q.ask_sz, self._ask_queue_ahead)
                if f is not None:
                    fills.append(f)

        self._settle_markouts(snapshot)
        self.mark(snapshot)
        return fills

    def mark(self, snapshot: BookSnapshot) -> None:
        """Record one point on the equity and inventory curves."""
        self._last_mid = snapshot.mid
        self._equity_ts.append(snapshot.local_ts)
        self._equity.append(self.equity)
        self._inventory_path.append(self.inventory)

    def flatten(self, snapshot: BookSnapshot) -> Fill | None:
        """Liquidate the open position at the touch (taker), for session end.

        Returns the closing :class:`Fill`, or ``None`` if already flat.
        """
        if abs(self.inventory) < _EPS:
            return None
        side: Side = "sell" if self.inventory > 0 else "buy"
        px = snapshot.best_bid if side == "sell" else snapshot.best_ask
        fill = self._book_fill(snapshot, side, px, abs(self.inventory), forced=True)
        logger.info("Flattened %+.5f at %.2f", -fill.signed_size, px)
        return fill

    def report(self) -> PerformanceReport:
        """Compute the full performance summary for the session."""
        eq = np.asarray(self._equity, dtype=np.float64)
        ts = np.asarray(self._equity_ts, dtype=np.float64)
        inv = np.asarray(self._inventory_path, dtype=np.float64)

        duration = float(ts[-1] - ts[0]) if ts.size >= 2 else 0.0
        realized = self.cash
        unrealized = self.position_value
        markouts = np.array([f.markout for f in self.fills], dtype=np.float64)
        finite_mo = markouts[np.isfinite(markouts)]

        buys = sum(1 for f in self.fills if f.side == "buy")
        return PerformanceReport(
            n_fills=len(self.fills),
            n_buys=buys,
            n_sells=len(self.fills) - buys,
            fill_rate_per_min=(len(self.fills) / duration * 60.0) if duration > 0 else 0.0,
            realized_pnl=realized,
            unrealized_pnl=unrealized,
            total_pnl=realized + unrealized,
            fees_paid=self.fees_paid,
            gross_edge=self.gross_edge,
            sharpe=self._sharpe(eq, ts),
            max_drawdown=self._max_drawdown(eq)[0],
            max_drawdown_pct=self._max_drawdown(eq)[1],
            inventory_variance=float(np.var(inv)) if inv.size else 0.0,
            inventory_mean=float(np.mean(inv)) if inv.size else 0.0,
            inventory_max_abs=float(np.max(np.abs(inv))) if inv.size else 0.0,
            mean_markout=float(np.mean(finite_mo)) if finite_mo.size else 0.0,
            adverse_selection_ratio=(
                float(np.mean(finite_mo < 0.0)) if finite_mo.size else 0.0
            ),
            duration_seconds=duration,
            turnover=self.turnover,
        )

    def fills_dataframe(self) -> pd.DataFrame:
        """Return the fill blotter as a ``pandas.DataFrame`` (offline analysis only).

        pandas is intentionally confined to post-session reporting; it never
        touches the tick loop.
        """
        if not self.fills:
            return pd.DataFrame(columns=list(Fill.__slots__))
        df = pd.DataFrame([asdict(f) for f in self.fills])
        df["datetime"] = pd.to_datetime(df["ts"], unit="s", utc=True)
        return df

    def equity_dataframe(self) -> pd.DataFrame:
        """Return the equity/inventory curve as a ``pandas.DataFrame``."""
        return pd.DataFrame(
            {
                "ts": self._equity_ts,
                "datetime": pd.to_datetime(self._equity_ts, unit="s", utc=True),
                "equity": self._equity,
                "inventory": self._inventory_path,
            }
        )

    # -- Fill mechanics ----------------------------------------------------- #
    def _try_fill(
        self,
        snapshot: BookSnapshot,
        side: Side,
        price: float,
        size: float,
        queue_ahead: float,
    ) -> Fill | None:
        """Stage 1 + Stage 2 of the fill model for one side."""
        if side == "buy":
            penetration = price - snapshot.best_ask   # >0 once market trades down to us
        else:
            penetration = snapshot.best_bid - price

        if penetration < 0.0:
            return None  # not eligible: market never reached our price

        # Risk gate: reject a fill that would breach the position limit.
        signed = size if side == "buy" else -size
        if abs(self.inventory + signed) > self.cfg.max_inventory + _EPS:
            return None

        p_fill = self._fill_probability(penetration, queue_ahead, snapshot)
        if self._rng.random() > p_fill:
            return None

        return self._book_fill(snapshot, side, price, size)

    def _fill_probability(
        self, penetration: float, queue_ahead: float, snapshot: BookSnapshot
    ) -> float:
        r"""Combined queue and penetration fill probability, clipped to ``[0, 1]``.

        .. math::

            P = \\frac{1}{1 + Q_{ahead}/\\bar Q}\\cdot\\left(1 - e^{-\\eta\\Delta/\\text{tick}}\\right)

        The tick scale is approximated by the current spread when it is
        positive, so the penetration term is dimensionless and adapts to regime.
        """
        queue_factor = 1.0 / (1.0 + max(queue_ahead, 0.0) / max(self.cfg.queue_depth_scale, _EPS))
        tick_scale = max(snapshot.spread, _EPS)
        pen_factor = 1.0 - math.exp(-self.cfg.penetration_eta * penetration / tick_scale)
        # A quote exactly at the touch (penetration == 0) still gets a small
        # chance of filling from a marketable order sweeping the level.
        pen_factor = max(pen_factor, 0.05)
        return float(np.clip(queue_factor * pen_factor, 0.0, 1.0))

    def _book_fill(
        self, snapshot: BookSnapshot, side: Side, price: float, size: float, forced: bool = False
    ) -> Fill:
        """Apply a fill to cash, inventory, fees, and history."""
        signed = size if side == "buy" else -size
        notional = price * size
        fee = abs(notional) * self.cfg.maker_fee_bps * 1e-4

        self.cash -= signed * price      # buying spends cash, selling earns it
        self.cash -= fee
        self.inventory += signed
        self.fees_paid += fee
        self.turnover += size

        sign = 1.0 if side == "buy" else -1.0
        edge = -sign * (price - snapshot.mid) * size  # bought below mid -> positive
        self.gross_edge += edge

        fill = Fill(
            ts=snapshot.local_ts,
            side=side,
            price=price,
            size=size,
            mid_at_fill=snapshot.mid,
            fee=fee,
            inventory_after=self.inventory,
            edge=edge,
        )
        self.fills.append(fill)
        if not forced:
            self._pending_markout.append((fill, snapshot.local_ts + self.cfg.markout_horizon))
        logger.debug(
            "FILL %-4s %.5f @ %.2f (mid %.2f, edge %+0.4f) -> q=%+.5f",
            side, size, price, snapshot.mid, edge, self.inventory,
        )
        return fill

    def _settle_markouts(self, snapshot: BookSnapshot) -> None:
        """Populate ``markout`` for fills whose horizon has now elapsed."""
        while self._pending_markout and self._pending_markout[0][1] <= snapshot.local_ts:
            fill, _ = self._pending_markout.popleft()
            sign = 1.0 if fill.side == "buy" else -1.0
            fill.markout = sign * (snapshot.mid - fill.mid_at_fill) * fill.size

    @staticmethod
    def _depth_at_or_better(snapshot: BookSnapshot, side: Side, price: float | None) -> float:
        """Aggregate resting size at or better than ``price`` — our queue ahead."""
        if price is None:
            return 0.0
        book = snapshot.bids if side == "buy" else snapshot.asks
        mask = book[:, 0] >= price if side == "buy" else book[:, 0] <= price
        return float(np.sum(book[mask, 1]))

    # -- Analytics ---------------------------------------------------------- #
    @staticmethod
    def _sharpe(equity: np.ndarray, ts: np.ndarray) -> float:
        r"""Annualized Sharpe ratio of equity increments.

        .. math::

            SR = \\frac{\\mu_{\\Delta}}{\\sigma_{\\Delta}}
                 \\sqrt{\\frac{365 \\cdot 86400}{\\overline{\\Delta t}}}

        Risk-free rate is taken as zero: a market maker's PnL is already an
        excess-return series over a nominally flat book.

        **Caveat.** Annualizing from tick-frequency increments assumes those
        increments are IID, which they are not — book updates are strongly
        autocorrelated. The scaling factor at 10 Hz is :math:`\\sqrt{3\\times10^8}`,
        so magnitudes come out large in both directions. Treat the **sign and
        the relative ordering across parameter sets** as the signal; treat the
        absolute magnitude as an artifact of the annualization convention.
        Phase 4 should re-report this on 1-minute resampled PnL for a figure
        comparable to published strategy Sharpes.
        """
        if equity.size < 3:
            return 0.0
        d = np.diff(equity)
        sd = float(np.std(d, ddof=1))
        if sd < _EPS:
            return 0.0
        mean_dt = float(np.mean(np.diff(ts)))
        if mean_dt <= 0.0:
            return 0.0
        periods = SECONDS_PER_YEAR / mean_dt
        return float(np.mean(d) / sd * math.sqrt(periods))

    @staticmethod
    def _max_drawdown(equity: np.ndarray) -> tuple[float, float]:
        r"""Maximum peak-to-trough decline of the equity curve.

        .. math:: MDD = \\max_t\\left(\\max_{s \\le t} E_s - E_t\\right)

        Returns ``(absolute, percent_of_peak)``.

        **Caveat on the percentage.** A market maker's equity curve starts at
        zero PnL, so "percent of peak" is ill-defined early in a session: an
        absolute drawdown of 8 against a running peak of 0.0003 yields a
        meaningless 2,800%. We therefore report the percentage only once the
        peak exceeds the drawdown itself, capping the figure at 100%, and
        return ``0.0`` otherwise. **The absolute figure is the one to trust.**
        """
        if equity.size < 2:
            return 0.0, 0.0
        peak = np.maximum.accumulate(equity)
        dd = peak - equity
        i = int(np.argmax(dd))
        abs_dd = float(dd[i])
        denom = abs(float(peak[i]))
        pct = (abs_dd / denom * 100.0) if denom > abs_dd else 0.0
        return abs_dd, pct
