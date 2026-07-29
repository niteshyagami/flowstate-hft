"""
strategy.py
===========

Avellaneda-Stoikov optimal market making, with microstructure extensions.

The model (Avellaneda & Stoikov, *High-frequency trading in a limit order
book*, Quantitative Finance 2008)
-----------------------------------------------------------------------
Assume the efficient mid price follows arithmetic Brownian motion
:math:`dS_t = \\sigma\\, dW_t`, the maker holds inventory :math:`q`, and has
CARA utility with risk aversion :math:`\\gamma`. Solving the resulting HJB
equation yields two quantities.

**1. Reservation (indifference) price** — the price at which the maker is
indifferent between holding :math:`q` and holding :math:`q \\pm 1`:

.. math::

    r(s, q, t) = s - q\\,\\gamma\\,\\sigma^{2}\\,(T - t)

This is the entire risk-management engine in one term. Long inventory
(:math:`q > 0`) pushes :math:`r` *below* the mid, so both quotes shift down,
making the maker's ask more aggressive and its bid less aggressive — the book
mean-reverts inventory toward flat without ever crossing the spread.

**2. Optimal total spread** — the sum of the two half-spreads:

.. math::

    \\delta^{a} + \\delta^{b}
      = \\gamma \\sigma^{2} (T-t) + \\frac{2}{\\gamma}\\ln\\!\\left(1 + \\frac{\\gamma}{\\kappa}\\right)

The first term is inventory-risk compensation; the second is the monopolistic
rent extractable given order-arrival elasticity :math:`\\kappa`. Quotes are then
placed symmetrically about :math:`r`:

.. math::

    p^{bid} = r - \\tfrac{1}{2}(\\delta^a + \\delta^b), \\quad
    p^{ask} = r + \\tfrac{1}{2}(\\delta^a + \\delta^b)

Order arrival intensity is assumed exponential in distance from mid:
:math:`\\lambda(\\delta) = A e^{-\\kappa \\delta}`, which is what makes
:math:`\\kappa` estimable from fill data.

Infinite-horizon variant
------------------------
Crypto perps never close, so a fixed terminal time :math:`T` is artificial.
Guéant, Lehalle & Fernandez-Tapia (2013) give the stationary limit, which we
emulate by replacing :math:`(T-t)` with a constant *risk horizon*
:math:`\\tau`: the time over which the maker expects to be able to unwind.
Set ``horizon_mode="rolling"`` for that behaviour; ``"session"`` reproduces the
original terminating-inventory dynamics.

Extensions layered on top of vanilla A-S
----------------------------------------
* **Micro-price anchoring.** Substitute :math:`s \\to p^{micro}` (blended). The
  mid is a biased estimator when the book is lopsided; anchoring on the
  micro-price is a direct adverse-selection reduction.
* **OFI skew.** Shift both quotes by :math:`\\beta \\cdot \\widehat{OFI} \\cdot s`.
  When flow is aggressively buying, step the ask away so you are not run over.
* **Inventory clamp.** Beyond a soft limit the model widens super-linearly, and
  at the hard limit the offending side is withdrawn entirely.
* **Tick/spread floors.** No quote is ever placed inside the venue's tick grid,
  and the half-spread is floored at half a tick.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from .features import FeatureVector

__all__ = ["ASParams", "Quote", "AvellanedaStoikovStrategy"]

_EPS: Final[float] = 1e-12


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class ASParams:
    """Tunable parameters for the Avellaneda-Stoikov quoter.

    Attributes
    ----------
    gamma:
        Risk aversion :math:`\\gamma`. Higher → tighter inventory control, wider
        spreads, lower fill rate. Typical range ``1e-4 .. 1e-1`` for BTC where
        prices are ~1e4-1e5 and :math:`\\sigma^2` is in price² units.
    kappa:
        Order-arrival elasticity :math:`\\kappa` in :math:`\\lambda = Ae^{-\\kappa\\delta}`.
        Higher → arrivals fall off fast with distance → quote tighter.
        Re-estimated online when ``adaptive_kappa`` is on.
    order_size:
        Size quoted on each side, in base units.
    max_inventory:
        Hard position limit :math:`q_{max}` in base units. Quoting on the side
        that would breach it is suspended.
    soft_inventory_ratio:
        Fraction of ``max_inventory`` beyond which the inventory penalty is
        amplified (see :meth:`_inventory_penalty`).
    horizon_mode:
        ``"rolling"`` → constant risk horizon ``risk_horizon`` (perp default).
        ``"session"`` → classic :math:`(T-t)` countdown over ``session_seconds``.
    risk_horizon:
        :math:`\\tau` in seconds for rolling mode: how long you expect to need
        to flatten. Short horizon → the model tolerates inventory; long horizon
        → aggressive skewing.
    session_seconds:
        Total session length :math:`T` for session mode.
    micro_weight:
        Blend weight :math:`w` in :math:`s = w\\,p^{micro} + (1-w)\\,p^{mid}`.
    ofi_skew_coefficient:
        :math:`\\beta`. Quote shift in *relative* terms per unit smoothed OFI.
    signal_skew_coefficient:
        Predictive skew strength on the **micro-price premium**
        :math:`(p^{micro}-m)/m`, which measured the highest information
        coefficient (IC≈0.4-0.5 at h=1) of any signal. Unlike the OFI term,
        which pushes quotes *away* from flow to avoid being run over, this term
        shifts the quote *centre toward* the predicted move: if the micro-price
        says price is about to rise, both quotes lift so the maker buys before
        the rise and doesn't sell too cheap. ``0`` disables it (vanilla A-S).
        Because the premium is already a fraction, the shift is
        :math:`c \\cdot \\text{premium} \\cdot s`, clipped to the half-spread.
    imbalance_skew_coefficient:
        Predictive skew on top-of-book imbalance :math:`(I - 0.5)`, the
        second-strongest signal (IC≈0.4 at h=5-10). Complements the micro-price
        term, which leads at h=1 while imbalance leads slightly later.
    tick_size:
        Venue price increment. Bids round down, asks round up — never quote
        tighter than the grid allows.
    min_half_spread_ticks:
        Floor on each half-spread, in ticks.
    max_half_spread_bps:
        Ceiling on each half-spread, in basis points of mid. Prevents a
        volatility spike from producing a quote nobody will ever hit.
    adaptive_kappa:
        Re-estimate :math:`\\kappa` from realized fill distances.
    """

    gamma: float = 0.001
    kappa: float = 1.5
    order_size: float = 0.01
    max_inventory: float = 0.10
    soft_inventory_ratio: float = 0.60
    horizon_mode: str = "rolling"
    risk_horizon: float = 60.0
    session_seconds: float = 3600.0
    micro_weight: float = 0.70
    ofi_skew_coefficient: float = 1.0e-5
    signal_skew_coefficient: float = 0.0
    imbalance_skew_coefficient: float = 0.0
    tick_size: float = 0.5
    min_half_spread_ticks: float = 1.0
    max_half_spread_bps: float = 50.0
    adaptive_kappa: bool = True
    kappa_halflife: float = 300.0

    def __post_init__(self) -> None:
        if self.gamma <= 0.0:
            raise ValueError("gamma must be strictly positive")
        if self.kappa <= 0.0:
            raise ValueError("kappa must be strictly positive")
        if self.order_size <= 0.0:
            raise ValueError("order_size must be strictly positive")
        if self.max_inventory <= 0.0:
            raise ValueError("max_inventory must be strictly positive")
        if self.tick_size <= 0.0:
            raise ValueError("tick_size must be strictly positive")
        if not 0.0 <= self.micro_weight <= 1.0:
            raise ValueError("micro_weight must lie in [0, 1]")
        if self.horizon_mode not in ("rolling", "session"):
            raise ValueError("horizon_mode must be 'rolling' or 'session'")


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
@dataclass(slots=True, frozen=True)
class Quote:
    """A two-sided quote instruction produced by the strategy.

    A side priced ``None`` means *do not quote there* (inventory limit breached
    or the model is not yet warmed up).
    """

    ts: float
    bid_px: float | None
    ask_px: float | None
    bid_sz: float
    ask_sz: float
    reservation_px: float
    reference_px: float
    half_spread: float
    inventory: float
    sigma_price: float
    horizon: float
    reason: str = "ok"

    @property
    def is_two_sided(self) -> bool:
        """``True`` when both sides are live."""
        return self.bid_px is not None and self.ask_px is not None

    @property
    def skew(self) -> float:
        """Signed distance of the reservation price from the reference price.

        Negative when long (quotes pushed down to encourage selling).
        """
        return self.reservation_px - self.reference_px

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        b = f"{self.bid_px:,.2f}" if self.bid_px is not None else "  --  "
        a = f"{self.ask_px:,.2f}" if self.ask_px is not None else "  --  "
        return (
            f"[{self.reason:>9}] ref={self.reference_px:,.2f} r={self.reservation_px:,.2f} "
            f"skew={self.skew:+.2f} | {b} / {a} | δ={self.half_spread:.2f} "
            f"q={self.inventory:+.4f} σ={self.sigma_price:.3f}"
        )


# --------------------------------------------------------------------------- #
# Strategy
# --------------------------------------------------------------------------- #
class AvellanedaStoikovStrategy:
    """Stateless-per-tick quoter with a small amount of adaptive state.

    The only mutable state is (a) the online :math:`\\kappa` estimate and
    (b) the session start time. Inventory is *injected* by the caller each tick
    rather than tracked internally, which keeps the strategy pure enough to
    unit-test against synthetic inventory paths.

    Examples
    --------
    >>> strat = AvellanedaStoikovStrategy(ASParams(gamma=0.05, tick_size=0.5))
    >>> # quote(features, inventory) -> Quote
    """

    def __init__(self, params: ASParams | None = None) -> None:
        self.p = params or ASParams()
        self._kappa = self.p.kappa
        self._t0: float | None = None
        self._quotes_emitted = 0

    # -- Public API --------------------------------------------------------- #
    @property
    def kappa(self) -> float:
        """Current (possibly adapted) order-arrival elasticity."""
        return self._kappa

    def reset(self) -> None:
        """Reset session clock and the adaptive kappa estimate."""
        self._kappa = self.p.kappa
        self._t0 = None
        self._quotes_emitted = 0

    def quote(self, f: FeatureVector, inventory: float) -> Quote:
        """Compute the optimal two-sided quote for the current state.

        Parameters
        ----------
        f:
            Latest :class:`~src.features.FeatureVector`.
        inventory:
            Current signed position :math:`q` in base units. Positive = long.

        Returns
        -------
        Quote
            With ``reason`` describing why a side may be suppressed:
            ``"warmup"``, ``"long_limit"``, ``"short_limit"``, or ``"ok"``.
        """
        if self._t0 is None:
            self._t0 = f.ts

        horizon = self._horizon(f.ts)
        reference = self._reference_price(f)
        sigma_px = max(f.sigma_price, _EPS)

        # ---- 1. Reservation price ---------------------------------------- #
        #     r = s - q * γ * σ² * (T - t)   (with a super-linear soft clamp)
        penalty = self._inventory_penalty(inventory)
        reservation = reference - penalty * self.p.gamma * (sigma_px ** 2) * horizon

        # ---- 2. Optimal half-spread --------------------------------------- #
        #     δ = ½[ γσ²(T-t) + (2/γ) ln(1 + γ/κ) ]
        half_spread = self.cap_half_spread(self._half_spread(sigma_px, horizon), reference)

        # ---- 3. Microstructure skew --------------------------------------- #
        # Two conceptually opposite adjustments, both clipped to the spread:
        #   (a) OFI skew: push AWAY from aggressive flow (defensive, anti-run-over).
        #   (b) Predictive skew: shift the centre TOWARD the move the leading
        #       signals (micro-price premium, book imbalance) forecast. These
        #       had the highest information coefficient in Phase-2 research
        #       (IC ~ 0.4-0.5), so acting on them directly attacks adverse
        #       selection — the maker steps ahead of informed flow instead of
        #       being picked off by it.
        ofi_shift = self.p.ofi_skew_coefficient * f.ofi_ewma * reference

        micro_premium = (f.micro - f.mid) / f.mid if f.mid > 0 else 0.0
        predictive_shift = (
            self.p.signal_skew_coefficient * micro_premium
            + self.p.imbalance_skew_coefficient * (f.imbalance - 0.5)
        ) * reference

        total_shift = float(np.clip(ofi_shift + predictive_shift,
                                    -half_spread, half_spread))
        centre = reservation + total_shift

        raw_bid = centre - half_spread
        raw_ask = centre + half_spread

        # ---- 4. Grid alignment and sanity ---------------------------------- #
        bid_px = self._round_to_tick(raw_bid, "down")
        ask_px = self._round_to_tick(raw_ask, "up")
        if ask_px - bid_px < self.p.tick_size:
            ask_px = bid_px + self.p.tick_size

        # ---- 5. Risk gating ------------------------------------------------ #
        reason = "ok"
        if f.n_samples < 2 or f.sigma_price <= 0.0:
            bid_px = ask_px = None
            reason = "warmup"
        else:
            if inventory + self.p.order_size > self.p.max_inventory + _EPS:
                bid_px = None
                reason = "long_limit"
            if inventory - self.p.order_size < -self.p.max_inventory - _EPS:
                ask_px = None
                reason = "short_limit" if reason == "ok" else "both_limits"

        self._quotes_emitted += 1
        return Quote(
            ts=f.ts,
            bid_px=bid_px,
            ask_px=ask_px,
            bid_sz=self.p.order_size,
            ask_sz=self.p.order_size,
            reservation_px=reservation,
            reference_px=reference,
            half_spread=half_spread,
            inventory=inventory,
            sigma_price=f.sigma_price,
            horizon=horizon,
            reason=reason,
        )

    def observe_fill(self, fill_distance: float, dt: float) -> None:
        r"""Update the online :math:`\\kappa` estimate from a realized fill.

        Under :math:`\\lambda(\\delta) = Ae^{-\\kappa\\delta}`, the maximum-likelihood
        estimator of :math:`\\kappa` from observed fill distances is the
        reciprocal of their mean:
        :math:`\\hat\\kappa = 1 / \\overline{\\delta}`. We maintain
        :math:`\\overline{\\delta}` as a time-decayed EWMA so the estimate tracks
        regime shifts instead of averaging over the whole session.

        Parameters
        ----------
        fill_distance:
            Distance of the filled quote from the reference price, in **relative**
            terms (:math:`\\delta / s`), so :math:`\\kappa` stays dimensionless.
        dt:
            Seconds elapsed since the previous fill observation.
        """
        if not self.p.adaptive_kappa or fill_distance <= 0.0:
            return
        alpha = 1.0 - math.pow(2.0, -max(dt, 1e-6) / self.p.kappa_halflife)
        current_mean = 1.0 / max(self._kappa, _EPS)
        new_mean = current_mean + alpha * (fill_distance - current_mean)
        self._kappa = float(np.clip(1.0 / max(new_mean, _EPS), 1e-3, 1e6))

    # -- Internals ---------------------------------------------------------- #
    def _horizon(self, now: float) -> float:
        r"""Return the effective :math:`(T - t)` in seconds.

        In ``"session"`` mode this decays linearly to a small floor, reproducing
        A-S's terminal-inventory urgency: as :math:`t \\to T` the inventory term
        vanishes and quotes converge on the reservation price. In ``"rolling"``
        mode it is the constant :math:`\\tau`.
        """
        if self.p.horizon_mode == "rolling":
            return self.p.risk_horizon
        elapsed = max(0.0, now - (self._t0 or now))
        return max(self.p.session_seconds - elapsed, 1.0)

    def _reference_price(self, f: FeatureVector) -> float:
        r"""Blend micro-price and mid: :math:`s = w\\,p^{micro} + (1-w)\\,m`.

        Uses the depth-weighted micro-price when it is finite and within one
        spread of the mid; otherwise falls back to level-1 micro to avoid a
        thin-book outlier dragging the anchor.
        """
        w = self.p.micro_weight
        micro = f.micro_deep
        if not math.isfinite(micro) or abs(micro - f.mid) > max(f.spread, _EPS):
            micro = f.micro
        return w * micro + (1.0 - w) * f.mid

    def _inventory_penalty(self, q: float) -> float:
        r"""Effective inventory used in the reservation-price term.

        Vanilla A-S is linear in :math:`q`. That is fine near flat but too gentle
        near the position limit, where the marginal cost of one more unit is
        genuinely convex. We apply a cubic amplification past the soft limit:

        .. math::

            \\tilde q = q\\left[1 + \\left(\\frac{\\max(0,\\,|q| - q_{soft})}
                                             {q_{max} - q_{soft}}\\right)^{3}
                          \\cdot (\\Lambda - 1)\\right]

        with amplification cap :math:`\\Lambda = 4`. Below the soft limit this is
        exactly :math:`\\tilde q = q` and the model is unmodified A-S.
        """
        soft = self.p.soft_inventory_ratio * self.p.max_inventory
        excess = abs(q) - soft
        if excess <= 0.0:
            return q
        span = max(self.p.max_inventory - soft, _EPS)
        ratio = min(excess / span, 1.0)
        return q * (1.0 + (ratio ** 3) * 3.0)

    def _half_spread(self, sigma_px: float, horizon: float) -> float:
        r"""Optimal half-spread with tick and bps guardrails.

        .. math::

            \\delta = \\tfrac12\\left[\\gamma\\sigma^{2}(T-t)
                      + \\frac{2}{\\gamma}\\ln\\!\\left(1 + \\frac{\\gamma}{\\kappa}\\right)\\right]
        """
        g = self.p.gamma
        inventory_term = g * (sigma_px ** 2) * horizon
        rent_term = (2.0 / g) * math.log1p(g / max(self._kappa, _EPS))
        half = 0.5 * (inventory_term + rent_term)

        floor = self.p.min_half_spread_ticks * self.p.tick_size
        return max(half, floor)

    def _round_to_tick(self, price: float, direction: str) -> float:
        """Snap ``price`` onto the venue tick grid, conservatively.

        ``"down"`` for bids and ``"up"`` for asks, so rounding never makes a
        quote more aggressive than the model intended.
        """
        t = self.p.tick_size
        if direction == "down":
            return math.floor(price / t) * t
        return math.ceil(price / t) * t

    def cap_half_spread(self, half: float, mid: float) -> float:
        """Apply the ``max_half_spread_bps`` ceiling (exposed for backtests)."""
        ceiling = self.p.max_half_spread_bps * 1e-4 * mid
        floor = self.p.min_half_spread_ticks * self.p.tick_size
        return max(min(half, ceiling), floor)