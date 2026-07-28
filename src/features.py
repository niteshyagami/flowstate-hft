"""
features.py
===========

Stateless and stateful microstructure feature extraction.

The two headline signals are the ones that actually pay in a passive
market-making book:

**1. Micro-price (Gatheral & Oomen; Stoikov 2018).**
The mid price is a poor estimator of the "true" value when the book is
lopsided. The size-weighted micro-price leans toward the thin side, because
the thin side is where the next trade will push price:

.. math::

    p^{micro} = \\frac{Q^{ask} \\cdot p^{bid} + Q^{bid} \\cdot p^{ask}}
                     {Q^{bid} + Q^{ask}}

Note the *cross* weighting — bid price is weighted by **ask** size. Intuition:
a large resting bid (:math:`Q^{bid} \\gg Q^{ask}`) means buying pressure, so the
fair value should sit near the ask. Setting
:math:`I = Q^{bid} / (Q^{bid} + Q^{ask})` gives the equivalent form
:math:`p^{micro} = m + (I - \\tfrac12)\\,s`, where :math:`m` is the mid and
:math:`s` the spread.

**2. Order Flow Imbalance (Cont, Kukanov & Stoikov 2014).**
OFI is the signed net change in depth at the top of book between consecutive
snapshots. It is the single best-known linear predictor of short-horizon price
change. For successive snapshots :math:`n-1 \\to n`:

.. math::

    e_n = \\underbrace{\\mathbb{1}_{\\{P^b_n \\ge P^b_{n-1}\\}} q^b_n
                     - \\mathbb{1}_{\\{P^b_n \\le P^b_{n-1}\\}} q^b_{n-1}}_{\\text{bid contribution}}
        - \\underbrace{\\left(\\mathbb{1}_{\\{P^a_n \\le P^a_{n-1}\\}} q^a_n
                     - \\mathbb{1}_{\\{P^a_n \\ge P^a_{n-1}\\}} q^a_{n-1}\\right)}_{\\text{ask contribution}}

Reading the bid term: if the bid price *improved*, the whole new bid size is
added liquidity; if it *worsened*, the whole old size was pulled or hit; if the
price is unchanged both indicators fire and the term collapses to the size
delta :math:`q^b_n - q^b_{n-1}`. The ask term is the mirror image, and is
subtracted because ask-side additions are bearish.

Everything here is vectorized or O(1). No pandas in the hot path — a
``DataFrame`` construction per tick costs ~100 µs, which is two orders of
magnitude more than the entire feature computation.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Final

import numpy as np

from .ingestion import PX, SZ, BookSnapshot

__all__ = [
    "FeatureVector",
    "FeatureEngine",
    "micro_price",
    "weighted_micro_price",
    "book_imbalance",
    "order_flow_imbalance",
    "realized_volatility",
]

_EPS: Final[float] = 1e-12


# --------------------------------------------------------------------------- #
# Stateless primitives
# --------------------------------------------------------------------------- #
def book_imbalance(bid_size: float, ask_size: float) -> float:
    r"""Top-of-book volume imbalance :math:`I \\in [0, 1]`.

    .. math:: I = \\frac{Q^{bid}}{Q^{bid} + Q^{ask}}

    ``I > 0.5`` indicates buy-side pressure. Returns ``0.5`` for an empty book.
    """
    total = bid_size + ask_size
    if total < _EPS:
        return 0.5
    return bid_size / total


def micro_price(snapshot: BookSnapshot) -> float:
    r"""Level-1 size-weighted micro-price.

    .. math:: p^{micro} = \\frac{Q^a p^b + Q^b p^a}{Q^b + Q^a}

    Degrades gracefully to the arithmetic mid when both sides are empty.
    """
    qb, qa = snapshot.best_bid_size, snapshot.best_ask_size
    total = qb + qa
    if total < _EPS:
        return snapshot.mid
    return (qa * snapshot.best_bid + qb * snapshot.best_ask) / total


def weighted_micro_price(snapshot: BookSnapshot, depth: int = 5, decay: float = 0.5) -> float:
    r"""Multi-level micro-price with geometric depth decay.

    Level :math:`i` (0-indexed) contributes with weight
    :math:`w_i = e^{-\\lambda i}`, so far-touch liquidity — which is more often
    spoofed or simply not executable — is discounted:

    .. math::

        \\tilde{Q}^{b} = \\sum_{i<d} w_i q^b_i, \\quad
        \\tilde{Q}^{a} = \\sum_{i<d} w_i q^a_i, \\quad
        p = \\frac{\\tilde{Q}^{a} p^b_0 + \\tilde{Q}^{b} p^a_0}
                  {\\tilde{Q}^{b} + \\tilde{Q}^{a}}

    Parameters
    ----------
    depth:
        Number of levels aggregated per side.
    decay:
        Decay constant :math:`\\lambda`. ``0`` recovers a flat-weighted sum;
        large values recover the level-1 :func:`micro_price`.
    """
    d = min(depth, snapshot.depth)
    if d <= 0:
        return snapshot.mid

    w = np.exp(-decay * np.arange(d, dtype=np.float64))
    qb = float(np.dot(w, snapshot.bids[:d, SZ]))
    qa = float(np.dot(w, snapshot.asks[:d, SZ]))
    total = qb + qa
    if total < _EPS:
        return snapshot.mid
    return (qa * snapshot.best_bid + qb * snapshot.best_ask) / total


def order_flow_imbalance(prev: BookSnapshot, curr: BookSnapshot) -> float:
    r"""Single-step Order Flow Imbalance :math:`e_n` (Cont-Kukanov-Stoikov).

    Positive values indicate net buying pressure at the touch. Units are the
    base-asset size unit (e.g. BTC), so OFI is comparable across time but not
    across instruments without normalization — see
    :meth:`FeatureEngine.normalized_ofi`.
    """
    pb_n, qb_n = curr.best_bid, curr.best_bid_size
    pb_o, qb_o = prev.best_bid, prev.best_bid_size
    pa_n, qa_n = curr.best_ask, curr.best_ask_size
    pa_o, qa_o = prev.best_ask, prev.best_ask_size

    bid_term = (qb_n if pb_n >= pb_o else 0.0) - (qb_o if pb_n <= pb_o else 0.0)
    ask_term = (qa_n if pa_n <= pa_o else 0.0) - (qa_o if pa_n >= pa_o else 0.0)
    return bid_term - ask_term


def realized_volatility(log_returns: np.ndarray, dt: float, annualize: bool = False) -> float:
    r"""Realized volatility from a vector of log returns.

    .. math::

        \\sigma_{\\text{per second}}
          = \\sqrt{\\frac{1}{\\Delta t}\\cdot\\frac{1}{N-1}\\sum_i (r_i - \\bar r)^2}

    Parameters
    ----------
    log_returns:
        1-D array of :math:`r_i = \\ln(p_i / p_{i-1})`.
    dt:
        Mean sampling interval in seconds. Scaling by :math:`1/\\Delta t`
        converts a per-sample variance into a per-second variance, which is the
        unit Avellaneda-Stoikov expects when :math:`T-t` is measured in seconds.
    annualize:
        If ``True``, multiply by :math:`\\sqrt{365 \\cdot 86400}` (crypto trades
        continuously, so there is no 252-day convention).
    """
    n = log_returns.size
    if n < 2 or dt <= 0.0:
        return 0.0
    var_per_sample = float(np.var(log_returns, ddof=1))
    sigma = math.sqrt(max(var_per_sample, 0.0) / dt)
    if annualize:
        sigma *= math.sqrt(365.0 * 86400.0)
    return sigma


# --------------------------------------------------------------------------- #
# Feature vector
# --------------------------------------------------------------------------- #
@dataclass(slots=True, frozen=True)
class FeatureVector:
    """Everything the strategy layer needs from one book update.

    Attributes
    ----------
    ts:
        Local receipt timestamp (seconds, float epoch).
    mid / micro / micro_deep:
        Price estimators, in quote currency.
    spread:
        Top-of-book spread in quote currency.
    imbalance:
        Level-1 volume imbalance in ``[0, 1]``.
    ofi:
        Raw single-step order flow imbalance (base units).
    ofi_ewma:
        Exponentially smoothed OFI — the tradeable version. Raw OFI is far too
        noisy to steer quotes on directly.
    sigma:
        Per-second volatility of log mid returns.
    sigma_price:
        Per-second volatility in **absolute price units**
        (:math:`\\sigma_{\\text{price}} = \\sigma \\cdot p`). Avellaneda-Stoikov
        is formulated on an arithmetic Brownian mid, so the strategy layer needs
        the absolute-unit figure, not the relative one.
    trend:
        Short-horizon drift estimate: EWMA of mid deltas per second.
    """

    ts: float
    mid: float
    micro: float
    micro_deep: float
    spread: float
    imbalance: float
    ofi: float
    ofi_ewma: float
    sigma: float
    sigma_price: float
    trend: float
    n_samples: int

    def as_array(self) -> np.ndarray:
        """Flatten to a float64 vector for logging or model input."""
        return np.array(
            [self.mid, self.micro, self.micro_deep, self.spread, self.imbalance,
             self.ofi, self.ofi_ewma, self.sigma, self.sigma_price, self.trend],
            dtype=np.float64,
        )

    @staticmethod
    def columns() -> list[str]:
        """Column names matching :meth:`as_array`."""
        return ["mid", "micro", "micro_deep", "spread", "imbalance",
                "ofi", "ofi_ewma", "sigma", "sigma_price", "trend"]


# --------------------------------------------------------------------------- #
# Stateful engine
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class FeatureEngine:
    """Incremental feature computation over a stream of :class:`BookSnapshot`.

    State is held in fixed-capacity ring buffers (``collections.deque`` with
    ``maxlen``), so memory is bounded regardless of session length and there is
    no periodic reallocation pause.

    Parameters
    ----------
    vol_window:
        Number of log returns retained for the realized-volatility estimate.
        At ~10 book updates/sec on BTC, 512 samples ≈ 50 seconds of history.
    micro_depth:
        Depth used by :func:`weighted_micro_price`.
    micro_decay:
        Geometric decay constant for depth weighting.
    ofi_halflife:
        Half-life, in **seconds**, of the OFI EWMA. Converted to a per-update
        alpha using the observed inter-arrival time, so the smoothing is
        wall-clock consistent even when update frequency varies.
    trend_halflife:
        Half-life, in seconds, of the drift EWMA.
    min_samples:
        Volatility is reported as ``0.0`` until this many returns accumulate;
        the strategy treats that as "not ready" and stands down.
    """

    vol_window: int = 512
    micro_depth: int = 5
    micro_decay: float = 0.5
    ofi_halflife: float = 5.0
    trend_halflife: float = 10.0
    min_samples: int = 30

    _prev: BookSnapshot | None = field(default=None, init=False, repr=False)
    _returns: Deque[float] = field(init=False, repr=False)
    _intervals: Deque[float] = field(init=False, repr=False)
    _ofi_ewma: float = field(default=0.0, init=False, repr=False)
    _trend_ewma: float = field(default=0.0, init=False, repr=False)
    _ofi_abs_ewma: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._returns = deque(maxlen=self.vol_window)
        self._intervals = deque(maxlen=self.vol_window)

    # -- Public API --------------------------------------------------------- #
    def reset(self) -> None:
        """Clear all state. Call after a reconnect gap to avoid stale returns."""
        self._prev = None
        self._returns.clear()
        self._intervals.clear()
        self._ofi_ewma = 0.0
        self._trend_ewma = 0.0
        self._ofi_abs_ewma = 0.0

    @property
    def ready(self) -> bool:
        """``True`` once enough history exists for a usable volatility estimate."""
        return len(self._returns) >= self.min_samples

    @property
    def mean_dt(self) -> float:
        """Mean inter-update interval in seconds (defaults to 0.1 when cold)."""
        if not self._intervals:
            return 0.1
        return max(float(np.mean(self._intervals)), 1e-4)

    def normalized_ofi(self) -> float:
        r"""OFI scaled by its own recent absolute magnitude.

        .. math:: \\widehat{OFI} = \\frac{\\text{EWMA}(e_n)}{\\text{EWMA}(|e_n|) + \\epsilon}

        Result is roughly bounded in ``[-1, 1]`` and instrument-agnostic, which
        makes it safe to use as a direct skew multiplier in the quoter.
        """
        return self._ofi_ewma / (self._ofi_abs_ewma + _EPS)

    def update(self, snapshot: BookSnapshot) -> FeatureVector | None:
        """Ingest one snapshot and emit the corresponding feature vector.

        Returns ``None`` on the very first snapshot (no predecessor, so no OFI
        and no return), or when the snapshot is non-advancing (duplicate
        timestamp). Callers should simply skip a ``None``.
        """
        prev = self._prev
        self._prev = snapshot

        if prev is None:
            return None

        dt = snapshot.local_ts - prev.local_ts
        if dt <= 0.0:
            dt = self.mean_dt  # clock jitter guard; never divide by zero

        # --- Returns / volatility ----------------------------------------- #
        if prev.mid > 0.0 and snapshot.mid > 0.0:
            self._returns.append(math.log(snapshot.mid / prev.mid))
            self._intervals.append(dt)

        sigma = realized_volatility(np.fromiter(self._returns, dtype=np.float64), self.mean_dt) \
            if self.ready else 0.0
        sigma_price = sigma * snapshot.mid

        # --- Order flow imbalance ------------------------------------------ #
        ofi = order_flow_imbalance(prev, snapshot)
        a_ofi = self._alpha(dt, self.ofi_halflife)
        self._ofi_ewma += a_ofi * (ofi - self._ofi_ewma)
        self._ofi_abs_ewma += a_ofi * (abs(ofi) - self._ofi_abs_ewma)

        # --- Drift ---------------------------------------------------------- #
        a_tr = self._alpha(dt, self.trend_halflife)
        inst_drift = (snapshot.mid - prev.mid) / dt
        self._trend_ewma += a_tr * (inst_drift - self._trend_ewma)

        return FeatureVector(
            ts=snapshot.local_ts,
            mid=snapshot.mid,
            micro=micro_price(snapshot),
            micro_deep=weighted_micro_price(snapshot, self.micro_depth, self.micro_decay),
            spread=snapshot.spread,
            imbalance=book_imbalance(snapshot.best_bid_size, snapshot.best_ask_size),
            ofi=ofi,
            ofi_ewma=self._ofi_ewma,
            sigma=sigma,
            sigma_price=sigma_price,
            trend=self._trend_ewma,
            n_samples=len(self._returns),
        )

    # -- Internals ---------------------------------------------------------- #
    @staticmethod
    def _alpha(dt: float, halflife: float) -> float:
        r"""Time-aware EWMA weight.

        .. math:: \\alpha = 1 - 2^{-\\Delta t / h}

        Using elapsed wall time rather than a fixed per-tick alpha keeps the
        effective smoothing horizon constant even when update rates spike during
        volatile periods — exactly when a fixed-alpha filter would over-react.
        """
        if halflife <= 0.0:
            return 1.0
        return 1.0 - math.pow(2.0, -dt / halflife)
