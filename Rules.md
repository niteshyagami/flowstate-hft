# FlowState HFT — Engineering Rules

Non-negotiable guardrails. Every PR is reviewed against this document.

---

## 1. Language and tooling

| Rule | Detail |
|---|---|
| **R1.1** | Python **3.11+** only. The codebase uses `dataclass(slots=True)`, `X \| None` unions, and `Self`-free modern typing. No 3.10 backports. |
| **R1.2** | `from __future__ import annotations` at the top of **every** module. Keeps annotations lazy — zero import-time cost. |
| **R1.3** | Full type annotations on every public function, method, and dataclass field. `Any` requires a comment justifying it. |
| **R1.4** | Runtime dependencies limited to `numpy`, `pandas`, `websockets`. Adding a fourth requires written justification. |
| **R1.5** | No wildcard imports. Every module declares `__all__`. |
| **R1.6** | Line length 100. Formatter: `black -l 100`. Linter: `ruff`. |

---

## 2. Numerical code

| Rule | Detail |
|---|---|
| **R2.1** | **Vectorize.** No Python `for` loop over price levels, ticks, or samples where a NumPy operation exists. The one sanctioned exception is `_levels_to_array`, where string→float conversion has no vectorized equivalent. |
| **R2.2** | **No pandas in the hot path.** `DataFrame` construction costs ~100 µs — two orders of magnitude above the entire feature computation. pandas is confined to post-session reporting (`fills_dataframe`, `equity_dataframe`). |
| **R2.3** | `float64` everywhere. Declare `dtype=np.float64` explicitly; never rely on inference. |
| **R2.4** | **Guard every division.** Compare the denominator against `_EPS = 1e-12` and return a documented fallback. A `ZeroDivisionError` or silent `NaN` in the quoting loop is a production incident. |
| **R2.5** | **Preallocate.** `np.empty(shape)` then fill. Never `np.append` in a loop — it reallocates every call. |
| **R2.6** | **Units are documented and enforced.** Every price is quote currency, every size is base units, every time is seconds. `sigma` (relative) and `sigma_price` (absolute) are separate named fields precisely because conflating them is a silent, plausible-looking bug. |
| **R2.7** | Every formula in a docstring uses LaTeX-style math with defined symbols. If a reader cannot reconstruct the math from the docstring, the docstring is incomplete. |
| **R2.8** | Bounded memory. Rolling state uses `deque(maxlen=N)` or a fixed NumPy buffer. No unbounded list accumulation in any loop that runs per tick. (The fill blotter is exempt — it is O(fills), not O(ticks).) |

---

## 3. Async and concurrency

| Rule | Detail |
|---|---|
| **R3.1** | **Never block the event loop.** No `time.sleep`, no synchronous `requests`, no unguarded file I/O inside a coroutine. Use `await asyncio.sleep` and offload real blocking work to `run_in_executor`. |
| **R3.2** | **All queues are bounded.** `asyncio.Queue(maxsize=N)`. An unbounded queue converts a slow consumer into an OOM kill. |
| **R3.3** | **Producers never block on consumers.** Use `put_nowait` with an explicit overflow policy. FlowState drops the oldest frame and counts it. |
| **R3.4** | **Every `recv` has a timeout.** `asyncio.wait_for(ws.recv(), timeout=...)`. A socket that stops delivering without closing is the most common and most silent failure mode in WebSocket clients. |
| **R3.5** | **Name your tasks.** `asyncio.create_task(coro, name="ingestion")`. Anonymous tasks are unreadable in a debugger. |
| **R3.6** | `asyncio.CancelledError` is **re-raised**, never swallowed. Catching it in a bare `except Exception` is why applications hang on shutdown — always order the handlers with `CancelledError` first. |
| **R3.7** | Cancelled tasks are awaited during teardown, with `contextlib.suppress(asyncio.CancelledError)`. Fire-and-forget cancellation leaks. |
| **R3.8** | Signal handlers set an `asyncio.Event`; they never do work directly. |

---

## 4. Error handling

| Rule | Detail |
|---|---|
| **R4.1** | **Catch specific exceptions.** `except (KeyError, ValueError, TypeError)`, not `except Exception`. |
| **R4.2** | A broad `except Exception` is permitted **only** at a loop boundary whose job is to survive, and must be annotated `# noqa: BLE001` with a comment explaining the survival requirement. There are exactly two such sites: the reconnect loop and the per-tick pipeline guard. |
| **R4.3** | **Never `except: pass`.** Every handler either logs or increments a counter. Silent failure in a trading system is the worst possible outcome — worse than a crash, because a crash is visible. |
| **R4.4** | Reconnect logic uses exponential backoff **with jitter**. Deterministic backoff synchronizes every bot on the venue into a thundering herd when the venue bounces. |
| **R4.5** | Parameter validation lives in `__post_init__` and raises `ValueError` with a message naming the offending field. Fail at construction, not at tick 40,000. |
| **R4.6** | Log levels: `DEBUG` per-frame detail, `INFO` lifecycle and fills, `WARNING` recoverable degradation, `ERROR` with `logger.exception` for unexpected failures. Never `print` from library code — `print` is for `main.py` presentation only. |
| **R4.7** | Log messages use `%`-style lazy formatting (`logger.info("q=%.5f", q)`), not f-strings. The formatting cost is skipped entirely when the level is disabled. |

---

## 5. Risk and safety

| Rule | Detail |
|---|---|
| **R5.1** | **No live order transmission. Ever.** No signing, no authentication, no private endpoints. This is a hard architectural boundary, not a default. |
| **R5.2** | Every position-changing path checks the inventory limit. The check exists in **both** the strategy (quote suppression) and the simulator (fill rejection) — defence in depth, mirroring the maker + exchange risk-gate structure of a real system. |
| **R5.3** | No secrets in code, config, or logs. No `.env` file is read, because none is needed. |
| **R5.4** | Fill assumptions must be **conservative**. When two fill models are defensible, implement the pessimistic one. An optimistic backtest is a lie told to yourself. |
| **R5.5** | Fees are modelled by default and must never be silently zeroed. A strategy that only works at zero fees does not work. |
| **R5.6** | Markout is reported in every session summary. A strategy cannot be evaluated on PnL alone. |
| **R5.7** | Any performance claim in documentation states its assumptions inline. No number appears without its caveat. |

---

## 6. Code structure

| Rule | Detail |
|---|---|
| **R6.1** | Strict layering: `ingestion → features → strategy → execution`. Imports flow one direction only. `strategy.py` importing from `execution.py` is a design failure. |
| **R6.2** | Data-transfer objects are `@dataclass(slots=True, frozen=True)`. Mutable state lives in explicitly-named engine classes. |
| **R6.3** | The strategy does **not** own inventory. It receives `q` as an argument, which keeps it pure enough to unit-test against synthetic inventory paths. |
| **R6.4** | Configuration is dataclasses with defaults and validation, not module-level constants and not dicts. |
| **R6.5** | Every module is runnable standalone (`python -m src.ingestion BTC`) for isolated smoke testing. |
| **R6.6** | Public API surface is explicit via `__all__`. Anything not listed is `_`-prefixed and may change without notice. |

---

## 7. Documentation

| Rule | Detail |
|---|---|
| **R7.1** | NumPy-style docstrings: summary line, `Parameters`, `Returns`, `Raises`, `Notes`, `Examples` where useful. |
| **R7.2** | Every module opens with a docstring explaining what it does **and why the approach was chosen**. |
| **R7.3** | Mathematical functions state the formula, define every symbol, and cite the source paper. |
| **R7.4** | Comments explain **why**, not what. `# increment counter` is noise. `# drop oldest: a stale book is worse than no book` is the reason the code exists. |
| **R7.5** | Known limitations are documented **in the code**, next to the limitation, not buried in a README. |

---

## 8. Review checklist

Before merge:

- [ ] Runs clean on Python 3.11 and 3.12
- [ ] `ruff check .` and `black -l 100 --check .` pass
- [ ] No new dependency without justification (R1.4)
- [ ] No Python loop where NumPy would do (R2.1)
- [ ] No pandas in the tick path (R2.2)
- [ ] Every division guarded (R2.4)
- [ ] Every queue bounded, every `recv` timed out (R3.2, R3.4)
- [ ] `CancelledError` re-raised before any broad handler (R3.6)
- [ ] No `except: pass` (R4.3)
- [ ] Inventory limits enforced on every new position path (R5.2)
- [ ] Zero live-trading surface introduced (R5.1)
- [ ] Docstrings carry the math and the citation (R7.3)
- [ ] `Memory.md` updated with what changed and what is now known
