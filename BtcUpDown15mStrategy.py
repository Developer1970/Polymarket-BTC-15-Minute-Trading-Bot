from datetime import timedelta

from nautilus_trader.trading import Strategy, StrategyConfig
from nautilus_trader.model import (
    InstrumentId,
    QuoteTick,
    BinaryOption,
    InstrumentClose,
    PositionClosed,
)
from nautilus_trader.adapters.polymarket import POLYMARKET_VENUE

class BtcUpDown15mConfig(StrategyConfig):
    asset: str = "btc"
    interval_mins: int = 15
    flatten_before_close_secs: int = 15   # stop trading / flatten this long before expiry
    order_notional_usdc: float = 5.0


class BtcUpDown15mStrategy(Strategy):
    """
    Rolling strategy for Polymarket "{ASSET} Up or Down {N}m" binary-option pairs.

    Relies on the node's PolymarketInstrumentProviderConfig(event_slug_builder=...)
    to keep discovering upcoming windows; this class only tracks which pair is
    currently tradable and rolls off the previous one.
    """

    def __init__(self, config: BtcUpDown15mConfig):
        super().__init__(config)
        # window_key -> {"up": InstrumentId, "down": InstrumentId,
        #                "activation_ns": int, "expiration_ns": int}
        self._windows: dict[int, dict] = {}
        self.active_expiration_ns: int | None = None
        self.active_up_id: InstrumentId | None = None
        self.active_down_id: InstrumentId | None = None
        # True while on_start() is feeding the initial cache.instruments()
        # batch through on_instrument — see _maybe_activate.
        self._bootstrapping = False

    # ------------------------------------------------------------------ #
    # Discovery: instruments arrive from bootstrap / periodic refresh /
    # new-market discovery, configured at the node/client level. Nautilus
    # doesn't auto-deliver on_instrument for cached instruments — a strategy
    # has to explicitly subscribe, or this whole class stays inert (nothing
    # else here ever fires without on_instrument kicking off the chain:
    # _activate -> subscribe_quotes -> on_quote -> set_time_alert -> on_time_event).
    # ------------------------------------------------------------------ #
    def on_start(self) -> None:
        self.subscribe_instruments(POLYMARKET_VENUE)
        self.log.info(f"Subscribed to instrument updates for {POLYMARKET_VENUE}")

        # subscribe_instruments only delivers instruments discovered AFTER
        # this point — it does NOT replay instruments the data client's
        # Gamma bootstrap (PolymarketInstrumentProviderConfig.event_slug_builder)
        # already loaded into the cache before we subscribed. Process
        # whatever's already there through the exact same path as a live
        # on_instrument callback, so we don't depend on winning a race
        # against the data client's own startup.
        existing = self.cache.instruments(venue=POLYMARKET_VENUE)
        self.log.info(f"Processing {len(existing)} instrument(s) already in cache at startup")
        self._bootstrapping = True
        for instrument in existing:
            try:
                self.on_instrument(instrument)
            except Exception as e:
                # Don't let one bad instrument silently kill the rest of the
                # loop (or look identical to "nothing happened").
                self.log.error(f"on_instrument raised for {self._instrument_label(instrument.id)}: {e!r}")
        self._bootstrapping = False
        # Now that the full startup batch is known, pick correctly (see
        # _select_current_window) instead of per-instrument during the loop.
        self._maybe_activate()

    def on_instrument(self, instrument) -> None:
        self.log.info(f"on_instrument: {self._instrument_label(instrument.id)} ({type(instrument).__name__})")
        if not isinstance(instrument, BinaryOption):
            self.log.debug(f"  reject {self._instrument_label(instrument.id)}: not a BinaryOption")
            return
        if not self._belongs_to_family(instrument):
            return  # _belongs_to_family logs its own reason

        exp_ns = instrument.expiration_ns
        window = self._windows.setdefault(
            exp_ns, {"expiration_ns": exp_ns, "activation_ns": instrument.activation_ns}
        )

        outcome = (instrument.outcome or "").lower()
        if "up" in outcome:
            window["up"] = instrument.id
            self.log.info(f"BTC 15m window {exp_ns}: UP leg = {self._instrument_label(instrument.id)}")
        elif "down" in outcome:
            window["down"] = instrument.id
            self.log.info(f"BTC 15m window {exp_ns}: DOWN leg = {self._instrument_label(instrument.id)}")
        else:
            self.log.warning(
                f"Unrecognized outcome '{instrument.outcome}' for {self._instrument_label(instrument.id)}"
            )
            return

        # Once both legs of a window are known, consider activating it.
        if "up" in window and "down" in window:
            self.log.debug(f"Both legs known for window {exp_ns}, checking activation")
            self._maybe_activate()

    def _belongs_to_family(self, instrument: BinaryOption) -> bool:
        if not (hasattr(instrument, 'info') and instrument.info):
            self.log.debug(f"  reject {self._instrument_label(instrument.id)}: no .info metadata on instrument")
            return False
        question = instrument.info.get('question', '').lower()
        slug = instrument.info.get('market_slug', '').lower()
        asset = self.config.asset.lower()

        if not ((asset in question or asset in slug) and '15m' in slug):
            self.log.debug(
                f"  reject {self._instrument_label(instrument.id)}: question={question!r} slug={slug!r} "
                f"doesn't match {asset!r} + 15m"
            )
            return False

        return True

    def _select_current_window(self) -> dict | None:
        return self._select_window(self._windows, self.clock.timestamp_ns())

    @staticmethod
    def _select_window(windows: dict, now_ns: int) -> dict | None:
        """
        Among fully-known windows (both legs discovered), pick the one that
        SHOULD be active right now: the currently-live one if one exists,
        else the soonest upcoming one.

        This is NOT "whichever window's legs finished being discovered
        first" — cache.instruments() iteration order has no relation to
        market timing, so picking by discovery-completion order can (and
        did) latch onto a future window instead of the actually-live one.

        Pure function of (windows, now_ns) — no Nautilus clock/cache access
        — so it's testable without a live kernel.
        """
        complete = [w for w in windows.values() if "up" in w and "down" in w]
        live = [w for w in complete if w["activation_ns"] <= now_ns < w["expiration_ns"]]
        if live:
            return min(live, key=lambda w: w["expiration_ns"])
        future = [w for w in complete if w["activation_ns"] > now_ns]
        if future:
            return min(future, key=lambda w: w["activation_ns"])
        return None

    def _maybe_activate(self) -> None:
        if self._bootstrapping:
            # Wait until the whole startup batch from cache.instruments() has
            # been processed before picking — otherwise we might lock onto
            # the first window discovered rather than the correct one, if a
            # better candidate's legs complete later in the same batch.
            return
        if self.active_expiration_ns is not None:
            return  # already trading/waiting on something; rollover changes this, not fresh discovery
        selected = self._select_current_window()
        if selected is not None:
            self._activate(selected)

    def _activate(self, window: dict) -> None:
        exp_ns = window["expiration_ns"]
        act_ns = window["activation_ns"]
        self.active_expiration_ns = exp_ns
        self.active_up_id = window["up"]
        self.active_down_id = window["down"]

        # Subscribe now regardless of whether the window has started — we
        # want to be receiving quotes the instant it opens, not discover it
        # late. Evaluation itself is gated separately (see on_quote /
        # _active_window_is_open) so we never trade a market that hasn't
        # opened yet.
        self.subscribe_quotes(self.active_up_id)
        self.subscribe_quotes(self.active_down_id)

        exp_dt = self.clock.utc_now().__class__.fromtimestamp(exp_ns / 1e9, tz=self.clock.utc_now().tzinfo)
        flatten_at = exp_dt - timedelta(seconds=self.config.flatten_before_close_secs)

        self.clock.set_time_alert(name=f"flatten-{exp_ns}", alert_time=flatten_at)
        self.clock.set_time_alert(name=f"rollover-{exp_ns}", alert_time=exp_dt)

        status = "open" if act_ns <= self.clock.timestamp_ns() else "not open yet"
        self.log.info(
            f"Activated window expiring {exp_dt.isoformat()} ({status}): "
            f"up={self._instrument_label(self.active_up_id)} down={self._instrument_label(self.active_down_id)}"
        )

    def on_time_event(self, event) -> None:
        # TimeEvent handling for the alerts set above. v2 renamed the generic
        # on_event hook to on_time_event (separate from on_order_event/
        # on_position_event) — on_event doesn't exist at all in this version,
        # so this was silently never being called.
        name = getattr(event, "name", "")
        if name.startswith("flatten-"):
            self._flatten_active_window()
        elif name.startswith("rollover-"):
            self._rollover()

    # ------------------------------------------------------------------ #
    # Trading — only act on the currently active pair.
    # ------------------------------------------------------------------ #
    def on_quote(self, tick: QuoteTick) -> None:
        self.log.debug(
            f"on_quote: {self._instrument_label(tick.instrument_id)} "
            f"bid={tick.bid_price} ask={tick.ask_price} "
            f"age={self._tick_age_secs(tick):.1f}s"
        )
        if tick.instrument_id not in (self.active_up_id, self.active_down_id):
            return  # stale tick from a rolled-off or not-yet-active window
        if not self._active_window_is_open() or self._past_flatten_cutoff():
            return

        self._evaluate_signal(tick)

    def _active_window_is_open(self) -> bool:
        # We subscribe to quotes as soon as a window is chosen, which can be
        # before it opens (see _activate) — quotes/snapshots can arrive
        # early and would otherwise be evaluated against a market that
        # hasn't started trading for real yet.
        if self.active_expiration_ns is None:
            return False
        window = self._windows.get(self.active_expiration_ns)
        if window is None:
            return False
        return self.clock.timestamp_ns() >= window["activation_ns"]

    def _past_flatten_cutoff(self) -> bool:
        if self.active_expiration_ns is None:
            return True
        cutoff_ns = self.active_expiration_ns - self.config.flatten_before_close_secs * 1_000_000_000
        return self.clock.timestamp_ns() >= cutoff_ns

    def _instrument_tail_10(self, instrument_id: InstrumentId) -> str:
        instrument_id_str = str(instrument_id)
        before_dot = instrument_id_str.rsplit(".", 1)[0] if "." in instrument_id_str else instrument_id_str
        # Capture the contiguous digit run right before the venue suffix and keep its last 10 digits.
        i = len(before_dot) - 1
        while i >= 0 and before_dot[i].isdigit():
            i -= 1
        return before_dot[i + 1 :][-10:] if i < len(before_dot) - 1 else "UNKNOWN"

    def _instrument_label(self, instrument_id: InstrumentId) -> str:
        # A raw token-id tail is meaningless for cross-checking against what
        # you actually see on the Polymarket web UI. Look up the real market
        # slug/side so log lines are directly comparable to the site.
        tail = self._instrument_tail_10(instrument_id)
        instrument = self.cache.instrument(instrument_id)
        if instrument is None or not getattr(instrument, "info", None):
            return f"{tail} (no instrument info in cache)"
        slug = instrument.info.get("market_slug", "?")
        outcome = getattr(instrument, "outcome", None) or "?"
        return f"{tail} [{slug} / {outcome}]"

    def _tick_age_secs(self, tick: QuoteTick) -> float:
        # How old this tick's own event timestamp is relative to now — the
        # decisive check for "is this genuinely fresh data" vs "the exact
        # same tick being redelivered/cached".
        return (self.clock.timestamp_ns() - tick.ts_event) / 1_000_000_000

    def _evaluate_signal(self, tick: QuoteTick) -> None:
        # Plug in your actual edge here (e.g. delta of live BTC spot vs the
        # window's reference/opening price, fed via RTDS crypto price
        # subscription — see PolymarketRtdsCryptoPrice in the adapter docs).
        self.log.debug(
            f"Evaluating signal: {self._instrument_label(tick.instrument_id)} "
            f"bid={tick.bid_price} ask={tick.ask_price} age={self._tick_age_secs(tick):.1f}s"
        )
        pass

    # ------------------------------------------------------------------ #
    # Rollover mechanics
    # ------------------------------------------------------------------ #
    def _flatten_active_window(self) -> None:
        self.log.info(f"Flattening window expiring {self.active_expiration_ns}")
        for instrument_id in (self.active_up_id, self.active_down_id):
            if instrument_id is None:
                continue
            self.cancel_all_orders(instrument_id)
            for position in self.cache.positions_open(instrument_id=instrument_id):
                self.close_position(position)

    def _rollover(self) -> None:
        self.log.info(f"Rolling over from window expiring {self.active_expiration_ns}")
        old_up, old_down = self.active_up_id, self.active_down_id
        if old_up:
            self.unsubscribe_quotes(old_up)
        if old_down:
            self.unsubscribe_quotes(old_down)

        old_exp = self.active_expiration_ns
        self.active_expiration_ns = None
        self.active_up_id = None
        self.active_down_id = None
        if old_exp in self._windows:
            del self._windows[old_exp]

        # Pick up whichever window should be next — same live-preferring
        # selection used at startup, not just "next by discovery order".
        selected = self._select_current_window()
        if selected is not None:
            self._activate(selected)
        else:
            self.log.warning("No upcoming window discovered yet — waiting on next on_instrument.")

    # ------------------------------------------------------------------ #
    # Resolution + housekeeping — reactive only.
    # ------------------------------------------------------------------ #
    def on_instrument_close(self, data: InstrumentClose) -> None:
        self.log.info(f"{self._instrument_label(data.instrument_id)} resolved: close_price={data.close_price}")

    def on_position_closed(self, event: PositionClosed) -> None:
        instrument_id = event.instrument_id
        self.unsubscribe_quotes(instrument_id)
        self.cache.purge_instrument(instrument_id)