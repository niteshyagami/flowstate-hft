"""
Invariant tests for the FlowState HFT model layer.

These are *property* tests, not regression tests. They assert the economic
invariants the Avellaneda-Stoikov model must satisfy regardless of parameters.
If any of these fail, the model is wrong — not merely differently-tuned.

Run with::

    python -m pytest tests/ -v
    python tests/test_invariants.py        # no pytest required
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.execution import PaperExecutionSimulator, SimConfig  # noqa: E402
from src.features import (  # noqa: E402
    FeatureEngine,
    FeatureVector,
    book_imbalance,
    micro_price,
    order_flow_imbalance,
)
from src.ingestion import BookSnapshot, parse_l2_book  # noqa: E402
from src.strategy import ASParams, AvellanedaStoikovStrategy  # noqa: E402

T0 = time.time()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_book(mid: float = 64_000.0, spread: float = 1.0,
              bid_sz: float = 1.0, ask_sz: float = 1.0, seq: int = 0,
              ts: float = T0) -> BookSnapshot:
    """Construct a synthetic 5-level book around ``mid``."""
    bids = np.array([[mid - spread / 2 - i, bid_sz + 0.2 * i] for i in range(5)])
    asks = np.array([[mid + spread / 2 + i, ask_sz + 0.2 * i] for i in range(5)])
    return BookSnapshot("TEST", ts, ts, bids, asks, seq)


def make_features(sigma_price: float = 5.0, mid: float = 64_000.0,
                  ofi_ewma: float = 0.0, n: int = 100) -> FeatureVector:
    """Construct a feature vector with a neutral book and controllable σ."""
    return FeatureVector(
        ts=T0, mid=mid, micro=mid, micro_deep=mid, spread=1.0, imbalance=0.5,
        ofi=0.0, ofi_ewma=ofi_ewma, sigma=sigma_price / mid,
        sigma_price=sigma_price, trend=0.0, n_samples=n,
    )


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
def test_parser_handles_real_payload_shape() -> None:
    """A well-formed l2Book frame parses with correct ordering and derived values."""
    payload = {
        "channel": "l2Book",
        "data": {"coin": "BTC", "time": 1_716_400_000_123, "levels": [
            [{"px": "64000.0", "sz": "1.2", "n": 3}, {"px": "63999.0", "sz": "2.0", "n": 5}],
            [{"px": "64001.0", "sz": "0.8", "n": 2}, {"px": "64002.0", "sz": "3.1", "n": 7}],
        ]},
    }
    s = parse_l2_book(payload, max_levels=20, seq=0)
    assert s.best_bid == 64_000.0
    assert s.best_ask == 64_001.0
    assert math.isclose(s.mid, 64_000.5)
    assert math.isclose(s.spread, 1.0)
    assert not s.is_crossed()


def test_parser_rejects_crossed_book() -> None:
    """A crossed book is data corruption and must be rejected, not quoted on."""
    payload = {"channel": "l2Book", "data": {"coin": "BTC", "time": 0, "levels": [
        [{"px": "64002.0", "sz": "1.0", "n": 1}],
        [{"px": "64000.0", "sz": "1.0", "n": 1}],
    ]}}
    try:
        parse_l2_book(payload, 20, 0)
    except ValueError:
        return
    raise AssertionError("crossed book was accepted")


def test_parser_repairs_unsorted_levels() -> None:
    """Out-of-order levels are sorted rather than silently misinterpreted."""
    payload = {"channel": "l2Book", "data": {"coin": "BTC", "time": 0, "levels": [
        [{"px": "63999.0", "sz": "1.0", "n": 1}, {"px": "64000.0", "sz": "1.0", "n": 1}],
        [{"px": "64002.0", "sz": "1.0", "n": 1}, {"px": "64001.0", "sz": "1.0", "n": 1}],
    ]}}
    s = parse_l2_book(payload, 20, 0)
    assert s.best_bid == 64_000.0, "bids must end up descending"
    assert s.best_ask == 64_001.0, "asks must end up ascending"


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def test_micro_price_leans_toward_thin_side() -> None:
    """Heavy bid depth pulls the micro-price up toward the ask, and vice versa."""
    heavy_bid = make_book(bid_sz=10.0, ask_sz=1.0)
    heavy_ask = make_book(bid_sz=1.0, ask_sz=10.0)
    assert micro_price(heavy_bid) > heavy_bid.mid
    assert micro_price(heavy_ask) < heavy_ask.mid


def test_micro_price_equals_mid_when_balanced() -> None:
    """A perfectly balanced book has micro-price exactly at the mid."""
    b = make_book(bid_sz=3.0, ask_sz=3.0)
    assert math.isclose(micro_price(b), b.mid, rel_tol=1e-12)


def test_book_imbalance_bounds() -> None:
    """Imbalance is bounded in [0,1] and is 0.5 for an empty book."""
    assert math.isclose(book_imbalance(1.0, 1.0), 0.5)
    assert book_imbalance(9.0, 1.0) > 0.5
    assert book_imbalance(1.0, 9.0) < 0.5
    assert math.isclose(book_imbalance(0.0, 0.0), 0.5)


def test_ofi_sign_conventions() -> None:
    """Bid improvement is bullish (+); ask improvement is bearish (−)."""
    prev = make_book(mid=64_000.0)
    bid_up = make_book(mid=64_000.5)   # bid stepped up, ask stepped up
    assert order_flow_imbalance(prev, bid_up) > 0

    ask_down = make_book(mid=63_999.5)  # both sides stepped down
    assert order_flow_imbalance(prev, ask_down) < 0

    # Pure size addition on the bid with no price change is bullish.
    more_bid = make_book(mid=64_000.0, bid_sz=5.0)
    assert order_flow_imbalance(prev, more_bid) > 0


def test_feature_engine_warmup_and_readiness() -> None:
    """The engine returns None on the first snapshot and warms up deterministically."""
    fe = FeatureEngine(min_samples=10)
    assert fe.update(make_book(seq=0, ts=T0)) is None
    for i in range(1, 30):
        fv = fe.update(make_book(mid=64_000.0 + i * 0.5, seq=i, ts=T0 + i * 0.1))
        assert fv is not None
    assert fe.ready
    assert fv.sigma_price > 0.0


# --------------------------------------------------------------------------- #
# Strategy — the core A-S invariants
# --------------------------------------------------------------------------- #
def test_reservation_price_decreases_in_inventory() -> None:
    """r(q) = s − qγσ²(T−t) must be strictly decreasing in q.

    This is *the* inventory risk mechanism. If it fails, the strategy has no
    means of mean-reverting its position without crossing the spread.
    """
    st = AvellanedaStoikovStrategy(ASParams(gamma=0.001, max_inventory=0.10, tick_size=1.0))
    fv = make_features()
    prices = [st.quote(fv, q).reservation_px for q in (-0.06, -0.03, 0.0, 0.03, 0.06)]
    assert all(a > b for a, b in zip(prices, prices[1:])), prices


def test_reservation_equals_reference_when_flat() -> None:
    """With zero inventory the reservation price collapses onto fair value."""
    st = AvellanedaStoikovStrategy(ASParams(gamma=0.001, tick_size=1.0))
    q = st.quote(make_features(), 0.0)
    assert math.isclose(q.reservation_px, q.reference_px, rel_tol=1e-12)


def test_spread_increases_with_volatility() -> None:
    """δ = ½[γσ²(T−t) + (2/γ)ln(1+γ/κ)] must be increasing in σ."""
    st = AvellanedaStoikovStrategy(ASParams(gamma=0.001, tick_size=1.0, max_half_spread_bps=1e6))
    widths = [st.quote(make_features(sigma_price=s), 0.0).half_spread for s in (1, 5, 20, 60)]
    assert all(a < b for a, b in zip(widths, widths[1:])), widths


def test_spread_increases_with_risk_aversion() -> None:
    """Higher γ must produce a wider quote for identical market state."""
    fv = make_features(sigma_price=20.0)
    narrow = AvellanedaStoikovStrategy(
        ASParams(gamma=1e-4, tick_size=1.0, max_half_spread_bps=1e6)).quote(fv, 0.0)
    wide = AvellanedaStoikovStrategy(
        ASParams(gamma=1e-2, tick_size=1.0, max_half_spread_bps=1e6)).quote(fv, 0.0)
    assert wide.half_spread > narrow.half_spread


def test_inventory_limits_suppress_the_offending_side() -> None:
    """At the long limit the bid is pulled; at the short limit the ask is pulled."""
    p = ASParams(gamma=0.001, order_size=0.01, max_inventory=0.05, tick_size=1.0)
    st = AvellanedaStoikovStrategy(p)
    at_long = st.quote(make_features(), 0.05)
    at_short = st.quote(make_features(), -0.05)
    assert at_long.bid_px is None and at_long.ask_px is not None
    assert at_short.ask_px is None and at_short.bid_px is not None
    assert st.quote(make_features(), 0.0).is_two_sided


def test_quotes_respect_tick_grid_conservatively() -> None:
    """Bids round down and asks round up — rounding never adds aggression."""
    tick = 0.5
    st = AvellanedaStoikovStrategy(ASParams(gamma=0.001, tick_size=tick))
    q = st.quote(make_features(mid=64_000.37), 0.0)
    assert q.bid_px is not None and q.ask_px is not None
    assert math.isclose(q.bid_px / tick, round(q.bid_px / tick), abs_tol=1e-9)
    assert math.isclose(q.ask_px / tick, round(q.ask_px / tick), abs_tol=1e-9)
    assert q.ask_px > q.bid_px


def test_warmup_suppresses_all_quoting() -> None:
    """With no volatility estimate the strategy must stand down entirely."""
    st = AvellanedaStoikovStrategy(ASParams(gamma=0.001, tick_size=1.0))
    cold = make_features(sigma_price=0.0, n=1)
    q = st.quote(cold, 0.0)
    assert q.bid_px is None and q.ask_px is None and q.reason == "warmup"


def test_invalid_params_rejected_at_construction() -> None:
    """Bad configuration fails fast, not at tick 40,000."""
    for kwargs in ({"gamma": 0.0}, {"kappa": -1.0}, {"tick_size": 0.0},
                   {"micro_weight": 1.5}, {"horizon_mode": "nonsense"}):
        try:
            ASParams(**kwargs)  # type: ignore[arg-type]
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid params: {kwargs}")


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def test_inventory_limit_rejects_breaching_fills() -> None:
    """The simulator enforces the hard limit independently of the strategy."""
    sim = PaperExecutionSimulator(SimConfig(max_inventory=0.02, latency_ms=0.0, rng_seed=1))
    st = AvellanedaStoikovStrategy(
        ASParams(gamma=1e-6, order_size=0.01, max_inventory=1.0, tick_size=1.0))
    px = 64_000.0
    for i in range(400):
        px -= 0.5  # relentless downtrend: every bid gets run over
        snap = make_book(mid=px, seq=i, ts=T0 + i * 0.1)
        sim.on_book(snap)
        fv = make_features(mid=px)
        sim.submit(st.quote(fv, sim.inventory), snap)
    assert abs(sim.inventory) <= 0.02 + 1e-9, sim.inventory


def test_accounting_identity_holds() -> None:
    """Cash must equal −Σ(signed size × price) − Σfees, exactly."""
    sim = PaperExecutionSimulator(SimConfig(maker_fee_bps=1.0, latency_ms=0.0, rng_seed=3))
    snap = make_book()
    for side, px in (("buy", 63_999.0), ("sell", 64_002.0), ("buy", 63_998.0)):
        sim._book_fill(snap, side, px, 0.01)  # type: ignore[arg-type]
    expected_cash = -sum(f.signed_size * f.price for f in sim.fills) - sim.fees_paid
    assert math.isclose(sim.cash, expected_cash, rel_tol=1e-12)
    assert math.isclose(sim.inventory, sum(f.signed_size for f in sim.fills), rel_tol=1e-12)


def test_drawdown_and_sharpe_are_finite_on_degenerate_input() -> None:
    """Analytics must not raise or return NaN on flat or near-empty curves."""
    sim = PaperExecutionSimulator(SimConfig())
    for i in range(50):
        sim.on_book(make_book(seq=i, ts=T0 + i * 0.1))
    r = sim.report()
    assert math.isfinite(r.sharpe) and math.isfinite(r.max_drawdown)
    assert r.max_drawdown >= 0.0 and 0.0 <= r.max_drawdown_pct <= 100.0


# --------------------------------------------------------------------------- #
# Runner (works without pytest installed)
# --------------------------------------------------------------------------- #
def main() -> int:
    """Execute every ``test_*`` function in this module and report."""
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001 - test harness reporting
            failures += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
