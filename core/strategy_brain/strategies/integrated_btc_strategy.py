"""
IntegratedBTCStrategy

Nautilus Strategy for trading Polymarket BTC Up/Down markets. This class
contains ONLY strategy/execution logic (signal processing, trading
decisions, order placement, paper trading, risk/performance tracking). It
knows nothing about how the market it's trading was found — it asks an
active-market finder (e.g. BTC15minActiveMarketFinder) for a MarketData at
startup and whenever the market-switch timer fires, and adapts.
"""
import asyncio
import os
import math
import time
import random
import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import List, Optional

from loguru import logger

from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.identifiers import ClientOrderId
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.objects import Quantity
from nautilus_trader.model.data import QuoteTick

from core.market_finder.market_data import MarketData
from core.market_finder.btc_15min_market_finder import BTC15minActiveMarketFinder

from core.strategy_brain.signal_processors.spike_detector import SpikeDetectionProcessor
from core.strategy_brain.signal_processors.sentiment_processor import SentimentProcessor
from core.strategy_brain.signal_processors.divergence_processor import PriceDivergenceProcessor
from core.strategy_brain.signal_processors.orderbook_processor import OrderBookImbalanceProcessor
from core.strategy_brain.signal_processors.tick_velocity_processor import TickVelocityProcessor
from core.strategy_brain.signal_processors.deribit_pcr_processor import DeribitPCRProcessor
from core.strategy_brain.fusion_engine.signal_fusion import get_fusion_engine
from execution.risk_engine import get_risk_engine
from monitoring.performance_tracker import get_performance_tracker
from monitoring.grafana_exporter import get_grafana_exporter
from feedback.learning_engine import get_learning_engine


# =============================================================================
# CONSTANTS
# =============================================================================
QUOTE_STABILITY_REQUIRED = 3      # Need only 3 valid ticks to be stable (faster startup)
PAPER_TRADES_FILE = "paper_trades.json"
MAX_PAPER_TRADES_HISTORY = 1000   # matches PerformanceTracker's _max_trades_history
# _record_paper_trade's simulated exit moves at most -8%/+8% from entry (see
# `movement = random.uniform(...)` there). These must be inside that range or
# they can never actually clamp anything — a wide threshold like 20-30% would
# silently never trigger, which is the "not functioning" bug this exists to
# fix. Stop-loss is tight (trades enter at minute 13-14 of a 15-min market,
# when the outcome is nearly decided, so a sharp reversal is the rare case);
# take-profit has more room since late-window entries are usually right.
STOP_LOSS_PCT = 0.01               # 1% adverse move clamps the simulated exit
TAKE_PROFIT_PCT = 0.05             # 5% favorable move clamps the simulated exit


@dataclass
class PaperTrade:
    """Track paper/simulation trades"""
    timestamp: datetime
    direction: str
    size_usd: float
    price: float
    signal_score: float
    signal_confidence: float
    outcome: str = "PENDING"
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    exit_reason: str = "TIME"  # "STOP_LOSS" | "TAKE_PROFIT" | "TIME"

    def to_dict(self):
        return {
            'timestamp': self.timestamp.isoformat(),
            'direction': self.direction,
            'size_usd': self.size_usd,
            'price': self.price,
            'signal_score': self.signal_score,
            'signal_confidence': self.signal_confidence,
            'outcome': self.outcome,
            'exit_price': self.exit_price,
            'pnl': self.pnl,
            'exit_reason': self.exit_reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PaperTrade":
        return cls(
            timestamp=datetime.fromisoformat(d['timestamp']),
            direction=d['direction'],
            size_usd=d['size_usd'],
            price=d['price'],
            signal_score=d['signal_score'],
            signal_confidence=d['signal_confidence'],
            outcome=d.get('outcome', 'PENDING'),
            # Older records (before P&L was tracked) won't have these — default to unknown.
            exit_price=d.get('exit_price'),
            pnl=d.get('pnl'),
            exit_reason=d.get('exit_reason', 'TIME'),
        )


class IntegratedBTCStrategy(Strategy):
    """
    Integrated BTC Strategy - FIXED VERSION
    - Subscribes immediately at startup
    - Forces stability for first trade
    - Correct timing for market switching
    """

    def __init__(self, redis_client=None, enable_grafana=True, test_mode=False):
        super().__init__()

        self.bot_start_time = datetime.now(timezone.utc)
        self.restart_after_minutes = 90

        # Nautilus
        self.instrument_id = None
        self.redis_client = redis_client
        self.current_simulation_mode = False

        # Market finder — finds which BTC 15-min market to trade. Strategy
        # only knows the MarketData it returns (see _apply_market).
        self.market_finder: Optional[BTC15minActiveMarketFinder] = None
        self.current_market: Optional[MarketData] = None
        self.next_switch_time: Optional[datetime] = None

        # Quote-stability tracking
        self._stable_tick_count = 0
        self._market_stable = False
        self._last_instrument_switch = None

        # =========================================================================
        # FIX 1: Force first trade by setting last_trade_time to -1
        # =========================================================================
        self.last_trade_time = -1  # Force first trade immediately!
        self._waiting_for_market_open = False  # True when waiting for a future market to open
        self._last_bid_ask = None  # (bid_decimal, ask_decimal) from last tick, for liquidity checks

        # Tick buffer: rolling 90s of ticks for TickVelocityProcessor
        self._tick_buffer: deque = deque(maxlen=500)  # ~500 ticks = well over 90s

        # YES token id for the current market (set via _apply_market)
        self._yes_token_id: Optional[str] = None
        self._no_instrument_id = None

        # Phase 4: Signal Processors
        self.spike_detector = SpikeDetectionProcessor(
            spike_threshold=0.05,       # FIXED: was 0.15 (too high for probabilities)
            lookback_periods=20,
        )
        self.sentiment_processor = SentimentProcessor(
            extreme_fear_threshold=25,
            extreme_greed_threshold=75,
        )
        self.divergence_processor = PriceDivergenceProcessor(
            divergence_threshold=0.05,
        )
        self.orderbook_processor = OrderBookImbalanceProcessor(
            imbalance_threshold=0.30,   # 30% skew to signal
            min_book_volume=50.0,       # ignore illiquid books
        )
        self.tick_velocity_processor = TickVelocityProcessor(
            velocity_threshold_60s=0.015,  # 1.5% move in 60s
            velocity_threshold_30s=0.010,  # 1.0% move in 30s
        )
        self.deribit_pcr_processor = DeribitPCRProcessor(
            bullish_pcr_threshold=1.20,
            bearish_pcr_threshold=0.70,
            max_days_to_expiry=2,
            cache_seconds=300,          # refresh every 5 min
        )

        # Phase 4: Signal Fusion — update weights for 6 processors
        self.fusion_engine = get_fusion_engine()
        # Rebalanced weights (must sum ≤ 1.0; higher = more influence)
        self.fusion_engine.set_weight("OrderBookImbalance", 0.30)  # best real-time signal
        self.fusion_engine.set_weight("TickVelocity",       0.25)  # fast poly momentum
        self.fusion_engine.set_weight("PriceDivergence",    0.18)  # spot momentum
        self.fusion_engine.set_weight("SpikeDetection",     0.12)  # mean reversion
        self.fusion_engine.set_weight("DeribitPCR",         0.10)  # institutional sentiment
        self.fusion_engine.set_weight("SentimentAnalysis",  0.05)  # daily F&G (weak)

        # Phase 5: Risk Management
        self.risk_engine = get_risk_engine()

        # Phase 6: Performance Tracking
        self.performance_tracker = get_performance_tracker()

        # Phase 7: Learning Engine
        self.learning_engine = get_learning_engine()

        # Phase 6: Grafana (optional)
        if enable_grafana:
            self.grafana_exporter = get_grafana_exporter()
        else:
            self.grafana_exporter = None

        # Price history
        self.price_history = []
        self.max_history = 100

        # Paper trading tracker — reload prior sessions' trades so a restart
        # (auto-restart fires every restart_after_minutes) doesn't clobber
        # paper_trades.json with just the new session's trades.
        self.paper_trades: List[PaperTrade] = self._load_paper_trades()

        self.test_mode = test_mode

        if test_mode:
            logger.info("=" * 80)
            logger.info("  TEST MODE ACTIVE - Trading every minute!")
            logger.info("=" * 80)

        logger.info("=" * 80)
        logger.info("INTEGRATED BTC STRATEGY INITIALIZED - FIXED VERSION")
        logger.info("  Phase 4: Signal processors ready")
        logger.info("  Phase 5: Risk engine ready")
        logger.info("  Phase 6: Performance tracking ready")
        logger.info("  Phase 7: Learning engine ready")
        logger.info("  $1 per trade maximum")
        logger.info("=" * 80)

    # ------------------------------------------------------------------
    # Redis
    # ------------------------------------------------------------------

    async def check_simulation_mode(self) -> bool:
        """Check Redis for current simulation mode."""
        if not self.redis_client:
            return self.current_simulation_mode
        try:
            sim_mode = self.redis_client.get('btc_trading:simulation_mode')
            if sim_mode is not None:
                redis_simulation = sim_mode == '1'
                if redis_simulation != self.current_simulation_mode:
                    self.current_simulation_mode = redis_simulation
                    mode_text = "SIMULATION" if redis_simulation else "LIVE TRADING"
                    logger.warning(f"Trading mode changed to: {mode_text}")
                    if not redis_simulation:
                        logger.warning("LIVE TRADING ACTIVE - Real money at risk!")
                return redis_simulation
        except Exception as e:
            logger.warning(f"Failed to check Redis simulation mode: {e}")
        return self.current_simulation_mode

    # ------------------------------------------------------------------
    # Strategy lifecycle
    # ------------------------------------------------------------------

    def on_start(self):
        """Called when strategy starts - LOAD ALL MARKETS AND SUBSCRIBE IMMEDIATELY"""
        logger.info("=" * 80)
        logger.info("INTEGRATED BTC STRATEGY STARTED - FIXED VERSION")
        logger.info("=" * 80)

        # =========================================================================
        # FIX 2 & 3: Find the market to trade and subscribe to it immediately
        # =========================================================================
        self.market_finder = BTC15minActiveMarketFinder(self.cache)
        market = self.market_finder.load()
        self._apply_market(market)

        now = datetime.now(timezone.utc)
        if market.is_open(now):
            self.next_switch_time = market.end_time
            self._waiting_for_market_open = False
        else:
            self.next_switch_time = market.start_time
            self._waiting_for_market_open = True
            logger.info(f"  Starts at {market.start_time.strftime('%H:%M:%S')} UTC — waiting for it to open")

        self.subscribe_quote_ticks(self.instrument_id)
        logger.info(f"✓ SUBSCRIBED to market: {self.instrument_id}")

        # Try to get current price from cache
        try:
            quote = self.cache.quote_tick(self.instrument_id)
            if quote and quote.bid_price and quote.ask_price:
                current_price = (quote.bid_price + quote.ask_price) / 2
                self.price_history.append(current_price)
                logger.info(f"✓ Initial price: ${float(current_price):.4f}")
        except Exception as e:
            logger.debug(f"No initial price yet: {e}")

        # Generate synthetic history if needed
        if len(self.price_history) < 20:
            self._generate_synthetic_history(target_count=20, existing_count=len(self.price_history))

        # =========================================================================
        # Schedule market switching via the strategy clock instead of a background
        # thread. clock.set_time_alert() fires its callback on the kernel thread
        # (the same thread that dispatches on_quote_tick), so it's safe to call
        # subscribe_quote_ticks/unsubscribe_quote_ticks from it. A homegrown
        # asyncio loop on a separate thread is not — Nautilus's message bus and
        # strategy API are only safe to call from the single kernel thread.
        # =========================================================================
        if self.next_switch_time:
            self._schedule_market_switch(self.next_switch_time)

        restart_time = self.bot_start_time + timedelta(minutes=self.restart_after_minutes)
        self.clock.set_time_alert(
            "auto_restart", restart_time, callback=self._on_restart_alert, override=True,
        )

        # Grafana's HTTP server lifecycle is process-level infra, not strategy
        # logic — it's started/stopped by the runner (run_integrated_bot),
        # not here. self.grafana_exporter (see __init__) is only used to
        # record metrics (increment_trade_counter, record_trade_duration),
        # which works regardless of who started the server.

        logger.info("=" * 80)
        logger.info("Strategy active - will trade every 15 minutes")
        logger.info(f"Price history: {len(self.price_history)} points")
        if len(self.price_history) >= 20:
            logger.info("✓ READY TO TRADE NOW!")
        else:
            logger.warning(f"⚠ Need more history ({len(self.price_history)}/20)")
        logger.info("=" * 80)

    def _generate_synthetic_history(self, target_count: int = 20, existing_count: int = 0):
        """Generate synthetic price history for testing"""
        if self.price_history:
            base_price = self.price_history[-1]
        else:
            base_price = Decimal("0.5")
        needed = target_count - existing_count
        if needed <= 0:
            return
        for _ in range(needed):
            change = Decimal(str(random.uniform(-0.03, 0.03)))
            new_price = base_price * (Decimal("1.0") + change)
            new_price = max(Decimal("0.01"), min(Decimal("0.99"), new_price))
            self.price_history.append(new_price)
            base_price = new_price

    # ------------------------------------------------------------------
    # Market switching
    # ------------------------------------------------------------------

    def _apply_market(self, market: MarketData) -> None:
        """Adopt a MarketData as the market this strategy is currently trading."""
        self.current_market = market
        self.instrument_id = market.instrument_id
        self._yes_token_id = market.yes_token_id
        self._no_instrument_id = market.no_instrument_id

    def _switch_to_next_market(self) -> bool:
        """Ask the market finder to advance, and adopt the new market if it's ready."""
        next_market = self.market_finder.advance()
        if next_market is None:
            return False

        old_instrument_id = self.instrument_id
        self._apply_market(next_market)
        self.next_switch_time = next_market.end_time

        logger.info("=" * 80)
        logger.info(f"SWITCHING TO NEXT MARKET: {next_market.slug}")
        logger.info(f"  Current time: {datetime.now(timezone.utc).strftime('%H:%M:%S')}")
        logger.info(f"  Market ends at: {self.next_switch_time.strftime('%H:%M:%S')}")
        logger.info("=" * 80)

        # =========================================================================
        # FIX 5: Force stability for new market and reset trade timer correctly
        # =========================================================================
        self._stable_tick_count = QUOTE_STABILITY_REQUIRED  # Force stable immediately
        self._market_stable = True
        self._waiting_for_market_open = False  # Market is now active

        # Reset trade timer so we trade at the NEXT quote we receive
        # Use -1 so any interval will trigger (same as startup)
        self.last_trade_time = -1
        logger.info(f"  Trade timer reset — will trade on next tick")

        # Polymarket's WS protocol has no unsubscribe frame — the adapter's
        # _unsubscribe_quote_ticks is a hard no-op that only logs an error
        # (see nautilus_trader/adapters/polymarket/data.py). old_instrument_id's
        # socket keeps streaming until the next auto-restart tears the whole
        # data client down; on_quote_tick already filters ticks by
        # self.instrument_id so this doesn't affect trading correctness.
        if old_instrument_id and old_instrument_id != self.instrument_id:
            logger.debug(f"  Leaving previous market subscribed (Polymarket has no unsubscribe): {old_instrument_id}")

        self.subscribe_quote_ticks(self.instrument_id)
        return True

    # ------------------------------------------------------------------
    # Market switch & restart scheduling
    #
    # These callbacks are invoked by Strategy.clock as TimeEvents, which are
    # dispatched on the kernel thread — the same thread that runs on_quote_tick,
    # on_order_filled, etc. That makes it safe to mutate strategy state and call
    # subscribe_quote_ticks/unsubscribe_quote_ticks/submit_order from here. Do
    # NOT reintroduce a background thread or a second asyncio event loop for
    # this: Nautilus's message bus and strategy API are only safe to call from
    # the single kernel thread (see nautilus_trader issue #3322).
    # ------------------------------------------------------------------

    def _schedule_market_switch(self, when: datetime):
        """Schedule the next market-switch/open check at an exact time."""
        self.clock.set_time_alert(
            "market_switch", when, callback=self._on_market_switch_alert, override=True,
        )

    def _on_market_switch_alert(self, event):
        """
        Fired by the strategy clock exactly when the current market ends, or
        when a future market we were waiting on is due to open.
        """
        if self._waiting_for_market_open:
            # The future market we were waiting for has now opened.
            # Treat it like a market switch so the trade timer resets.
            now = datetime.now(timezone.utc)
            logger.info("=" * 80)
            logger.info(f"⏰ WAITING MARKET NOW OPEN: {now.strftime('%H:%M:%S')} UTC")
            logger.info("=" * 80)
            if self.current_market is not None:
                self.next_switch_time = self.current_market.end_time
                logger.info(f"  Market ends at {self.next_switch_time.strftime('%H:%M:%S')} UTC")
            self._waiting_for_market_open = False
            self._market_stable = True
            self._stable_tick_count = QUOTE_STABILITY_REQUIRED
            self.last_trade_time = -1  # Trade immediately on next tick
            logger.info("  ✓ MARKET OPEN — ready to trade on next tick")
            self._schedule_market_switch(self.next_switch_time)
            return

        if self._switch_to_next_market():
            self._schedule_market_switch(self.next_switch_time)
        else:
            # Next market not loaded yet or not ready — retry shortly, same
            # cadence as the old poll loop, until the auto-restart alert fires.
            retry_at = datetime.now(timezone.utc) + timedelta(seconds=10)
            self._schedule_market_switch(retry_at)

    def _on_restart_alert(self, event):
        """Fired once by the strategy clock after restart_after_minutes uptime."""
        logger.warning("AUTO-RESTART TIME - Loading fresh filters")
        import signal as _signal
        os.kill(os.getpid(), _signal.SIGTERM)

    # ------------------------------------------------------------------
    # Quote tick handler - SIMPLIFIED
    # ------------------------------------------------------------------

    def on_quote_tick(self, tick: QuoteTick):
        """Handle quote tick - TRADE when market opens and at each 15-min boundary"""
        try:
            # Only process ticks from current instrument
            if self.instrument_id is None or tick.instrument_id != self.instrument_id:
                return

            now = datetime.now(timezone.utc)
            bid = tick.bid_price
            ask = tick.ask_price

            if bid is None or ask is None:
                return

            try:
                bid_decimal = bid.as_decimal()
                ask_decimal = ask.as_decimal()
            except:
                return

            # Always store price history
            mid_price = (bid_decimal + ask_decimal) / 2
            self.price_history.append(mid_price)
            if len(self.price_history) > self.max_history:
                self.price_history.pop(0)

            # Store latest bid/ask for liquidity check before order placement
            self._last_bid_ask = (bid_decimal, ask_decimal)

            # Tick buffer for TickVelocityProcessor (rolling 90s window)
            self._tick_buffer.append({'ts': now, 'price': mid_price})

            # Stability gate
            if not self._market_stable:
                self._stable_tick_count += 1
                if self._stable_tick_count >= 1:
                    self._market_stable = True
                    logger.info(f"✓ Market STABLE immediately")
                else:
                    return

            # =========================================================================
            # FIXED TRADING LOGIC:
            #
            # We trade once per market interval. Instead of checking wall-clock
            # boundaries (which caused the 2-hour wait), we use a simple counter
            # keyed to the Polymarket market's OWN start time.
            #
            # Within each market, we compute a "sub-interval" index:
            #   sub_interval = elapsed_seconds_since_market_open // market_length
            # Trade ID = (market_start_timestamp, sub_interval)
            # This fires once at market open AND once after every interval within
            # the same market if it's a multi-interval market.
            #
            # If _waiting_for_market_open is True (started before market opens),
            # we block trading until the timer loop calls _switch_to_next_market.
            # =========================================================================

            # Block trading if waiting for a future market to open
            if self._waiting_for_market_open:
                return

            current_market = self.current_market
            if current_market is None:
                return

            market_start_ts = current_market.market_timestamp  # Slug timestamp = market start (Unix)
            market_length_secs = (current_market.end_time - current_market.start_time).total_seconds()

            # How many intervals have elapsed since this market opened?
            elapsed_secs = now.timestamp() - market_start_ts
            if elapsed_secs < 0:
                # Market hasn't started yet — block
                return

            sub_interval = int(elapsed_secs // market_length_secs)

            # Unique trade key: (market_start_timestamp, sub_interval)
            trade_key = (market_start_ts, sub_interval)

            # =========================================================================
            # TRADE WINDOW: minutes 13–14 of each 15-min market (780–840 seconds in)
            #
            # WHY LATE IN THE MARKET:
            #   At 13 minutes in, the UP/DOWN result is nearly decided. The price IS
            #   the trend — if YES is at $0.78, BTC went up during this interval.
            #   We're not predicting anymore, we're reading a nearly-resolved outcome.
            #
            # WHY NOT EARLIER (the old 30–90s window):
            #   At 30 seconds in, nobody knows which way BTC will move. The signals
            #   have no edge. This is why we were losing at prices near $0.50.
            #
            # TREND FILTER (applied in _make_trading_decision):
            #   Price > 0.60 → clear UP trend → buy YES
            #   Price < 0.40 → clear DOWN trend → buy NO
            #   Price 0.40–0.60 → coin flip → SKIP (don't trade)
            #
            # Share count intuition:
            #   1.4 shares = price $0.71 → strong trend, win rate ~71%
            #   1.9 shares = price $0.53 → weak trend, near coin flip
            #   2.0+ shares = price $0.50 → pure coin flip, SKIP
            # =========================================================================
            seconds_into_sub_interval = elapsed_secs % market_length_secs
            TRADE_WINDOW_START = 780   # 13 minutes in
            TRADE_WINDOW_END   = 840   # 14 minutes in (60s window)

            if TRADE_WINDOW_START <= seconds_into_sub_interval < TRADE_WINDOW_END and trade_key != self.last_trade_time:
                self.last_trade_time = trade_key

                logger.info("=" * 80)
                logger.info(f" LATE-WINDOW TRADE: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                logger.info(f"   Market: {current_market.slug}")
                logger.info(f"   Sub-interval #{sub_interval} ({seconds_into_sub_interval:.1f}s in = {seconds_into_sub_interval/60:.1f} min)")
                logger.info(f"   Price: ${float(mid_price):,.4f} | Bid: ${float(bid_decimal):,.4f} | Ask: ${float(ask_decimal):,.4f}")
                logger.info(f"   Trend strength: {'STRONG ✓' if float(mid_price) > 0.60 or float(mid_price) < 0.40 else 'WEAK — may skip'}")
                logger.info(f"   Price history: {len(self.price_history)} points")
                logger.info("=" * 80)

                # Schedule the (already-async) trading decision on the kernel's
                # own running event loop instead of a thread-pool executor.
                # _fetch_market_context/_make_trading_decision use httpx async
                # I/O, so there's nothing blocking to offload to a thread — and
                # submit_order must run on the kernel thread anyway (see the
                # scheduling note above _on_market_switch_alert).
                asyncio.get_running_loop().create_task(self._run_trading_decision(mid_price))

        except Exception as e:
            logger.error(f"Error processing quote tick: {e}")

    # ------------------------------------------------------------------
    # Trading decision
    # ------------------------------------------------------------------

    async def _run_trading_decision(self, price_decimal: Decimal):
        """
        Task wrapper around _make_trading_decision.

        Runs as an asyncio task on the kernel's own event loop (scheduled from
        on_quote_tick via create_task) rather than a separate thread/loop, so
        submit_order and friends stay on the single kernel thread. Errors are
        caught here since asyncio otherwise only logs "Task exception was
        never retrieved" once the task is garbage collected.
        """
        try:
            await self._make_trading_decision(price_decimal)
        except Exception as e:
            logger.error(f"Error in trading decision task: {e}")
            import traceback
            traceback.print_exc()

    async def _fetch_market_context(self, current_price: Decimal) -> dict:
        """
        Fetch REAL external data to populate signal processor metadata.

        Returns a dict with:
          - sentiment_score (float 0-100): live Fear & Greed index, or None
          - spot_price (float): live BTC-USD from Coinbase, or None
          - deviation (float): polymarket price vs SMA-20 (always computed)
          - momentum (float): 5-period rate of change (always computed)
          - volatility (float): price std-dev over last 20 ticks (always computed)
        """
        current_price_float = float(current_price)

        # --- Always-available stats from local price_history ---
        recent_prices = [float(p) for p in self.price_history[-20:]]
        sma_20 = sum(recent_prices) / len(recent_prices)
        deviation = (current_price_float - sma_20) / sma_20
        momentum = (
            (current_price_float - float(self.price_history[-5])) / float(self.price_history[-5])
            if len(self.price_history) >= 5 else 0.0
        )
        variance = sum((p - sma_20) ** 2 for p in recent_prices) / len(recent_prices)
        volatility = math.sqrt(variance)

        metadata = {
            "deviation": deviation,
            "momentum": momentum,
            "volatility": volatility,
            # Tick buffer for TickVelocityProcessor
            "tick_buffer": list(self._tick_buffer),
            # YES token id for OrderBookImbalanceProcessor
            "yes_token_id": self._yes_token_id,
        }

        # --- Real sentiment: Fear & Greed Index via NewsSocialDataSource ---
        try:
            from data_sources.news_social.adapter import NewsSocialDataSource
            news_source = NewsSocialDataSource()
            await news_source.connect()
            fg = await news_source.get_fear_greed_index()
            await news_source.disconnect()
            if fg and "value" in fg:
                metadata["sentiment_score"] = float(fg["value"])
                metadata["sentiment_classification"] = fg.get("classification", "")
                logger.info(
                    f"Fear & Greed: {metadata['sentiment_score']:.0f} "
                    f"({metadata['sentiment_classification']})"
                )
            else:
                logger.warning("Fear & Greed fetch returned no data — sentiment processor skipped")
        except Exception as e:
            logger.warning(f"Could not fetch Fear & Greed index: {e} — sentiment processor skipped")

        # --- Real spot price: Coinbase BTC-USD REST API ---
        try:
            from data_sources.coinbase.adapter import CoinbaseDataSource
            coinbase = CoinbaseDataSource()
            await coinbase.connect()
            spot = await coinbase.get_current_price()
            await coinbase.disconnect()
            if spot:
                metadata["spot_price"] = float(spot)
                logger.info(f"Coinbase spot price: ${float(spot):,.2f}")
            else:
                logger.warning("Coinbase price fetch returned None — divergence processor skipped")
        except Exception as e:
            logger.warning(f"Could not fetch Coinbase spot price: {e} — divergence processor skipped")

        logger.info(
            f"Market context — deviation={deviation:.2%}, "
            f"momentum={momentum:.2%}, volatility={volatility:.4f}, "
            f"sentiment={'%.0f' % metadata['sentiment_score'] if 'sentiment_score' in metadata else 'N/A'}, "
            f"spot=${'%.2f' % metadata['spot_price'] if 'spot_price' in metadata else 'N/A'}"
        )
        return metadata

    async def _make_trading_decision(self, current_price: Decimal):
        """
        Make trading decision using our 7-phase system.

        Position size is always $1.00 — no variable sizing, no risk-engine
        calculation needed. The risk engine is still used to check that we
        don't already have too many open positions.
        """
        # --- Mode check ---
        is_simulation = await self.check_simulation_mode()
        logger.info(f"Mode: {'SIMULATION' if is_simulation else 'LIVE TRADING'}")

        # --- Minimum history guard ---
        if len(self.price_history) < 20:
            logger.warning(f"Not enough price history ({len(self.price_history)}/20)")
            return

        logger.info(f"Current price: ${float(current_price):,.4f}")

        # --- Phase 4a: Build real metadata for processors ---
        metadata = await self._fetch_market_context(current_price)

        # --- Phase 4b: Run all three signal processors ---
        signals = self._process_signals(current_price, metadata)

        if not signals:
            logger.info("No signals generated — no trade this interval")
            return

        logger.info(f"Generated {len(signals)} signal(s):")
        for sig in signals:
            logger.info(
                f"  [{sig.source}] {sig.direction.value}: "
                f"score={sig.score:.1f}, confidence={sig.confidence:.2%}"
            )

        # --- Phase 4c: Fuse signals into one consensus ---
        # min_score lowered to 40 because the TREND FILTER (price at min 11-13)
        # is now the primary decision maker. Fusion is informational context,
        # not the trade gate. The trend gate below is the real filter.
        fused = self.fusion_engine.fuse_signals(signals, min_signals=1, min_score=40.0)
        if not fused:
            logger.info("Fusion produced no actionable signal — no trade this interval")
            return

        logger.info(
            f"FUSED SIGNAL: {fused.direction.value} "
            f"(score={fused.score:.1f}, confidence={fused.confidence:.2%})"
        )

        # --- Phase 5: Position size is always exactly $1.00 ---
        POSITION_SIZE_USD = Decimal("1.00")

        # =========================================================================
        # TREND FILTER — replaces signal-based direction at the late trade window
        #
        # At minute 13, the Polymarket price IS the market's verdict on BTC direction.
        # We ignore what the signal processors say and simply follow the price:
        #
        #   price > 0.60 → market says UP with >60% confidence → buy YES
        #   price < 0.40 → market says DOWN with >60% confidence → buy NO
        #   price 0.40–0.60 → too close to call → SKIP (this is where we were losing)
        #
        # This directly addresses the observation that trades at 1.9–2.0+ shares
        # (price near $0.50) almost always lose, while trades at 1.4 shares
        # (price ~$0.71) mostly win.
        # =========================================================================
        TREND_UP_THRESHOLD   = 0.60   # price above this → buy YES (UP)
        TREND_DOWN_THRESHOLD = 0.40   # price below this → buy NO (DOWN)

        price_float = float(current_price)

        if price_float > TREND_UP_THRESHOLD:
            direction = "long"
            trend_confidence = price_float  # e.g. 0.72 = 72% confident UP
            logger.info(
                f" TREND: UP ({price_float:.2%} YES probability) → buying YES"
            )
        elif price_float < TREND_DOWN_THRESHOLD:
            direction = "short"
            trend_confidence = 1.0 - price_float  # e.g. 0.31 price = 69% confident DOWN
            logger.info(
                f" TREND: DOWN ({price_float:.2%} YES probability = {1-price_float:.2%} NO) → buying NO"
            )
        else:
            logger.info(
                f"⏭ TREND: NEUTRAL ({price_float:.2%}) — price too close to 0.50, SKIPPING trade "
                f"(coin flip territory: {TREND_DOWN_THRESHOLD:.0%}–{TREND_UP_THRESHOLD:.0%})"
            )
            return

        # Risk engine: only check position-count / exposure limits (no sizing math)
        is_valid, error = self.risk_engine.validate_new_position(
            size=POSITION_SIZE_USD,
            direction=direction,
            current_price=current_price,
        )
        if not is_valid:
            logger.warning(f"Risk engine blocked trade: {error}")
            return

        logger.info(f"Position size: $1.00 (fixed) | Direction: {direction.upper()}")

        # --- Stop-loss / take-profit levels ---
        # These bound how much the simulated paper-trade exit can move against
        # or in favor of us (see _record_paper_trade). For LIVE orders they're
        # informational only — Polymarket orders here are BUY-and-hold-to-
        # resolution (no active position monitoring/selling exists), and the
        # trade window (13-14 min into a 15-min market) leaves at most ~1-2
        # minutes before settlement anyway, so there's no real window for an
        # early exit to matter. Logged so the levels are visible, not enforced.
        if direction == "long":
            stop_loss_price = max(Decimal("0.01"), current_price * (Decimal("1") - Decimal(str(STOP_LOSS_PCT))))
            take_profit_price = min(Decimal("0.99"), current_price * (Decimal("1") + Decimal(str(TAKE_PROFIT_PCT))))
        else:
            stop_loss_price = min(Decimal("0.99"), current_price * (Decimal("1") + Decimal(str(STOP_LOSS_PCT))))
            take_profit_price = max(Decimal("0.01"), current_price * (Decimal("1") - Decimal(str(TAKE_PROFIT_PCT))))
        logger.info(
            f"Risk levels: stop_loss=${float(stop_loss_price):.4f} "
            f"take_profit=${float(take_profit_price):.4f}"
        )

        # --- Liquidity guard: don't place if market has no real depth ---
        # The current bid/ask come from the last processed quote tick.
        # If ask <= 0.02 or bid <= 0.02, the orderbook is essentially empty
        # and a FAK (IOC market) order will be rejected immediately.
        last_tick = getattr(self, '_last_bid_ask', None)
        if last_tick:
            last_bid, last_ask = last_tick
            MIN_LIQUIDITY = Decimal("0.02")
            if direction == "long" and last_ask <= MIN_LIQUIDITY:
                logger.warning(
                    f"⚠ No liquidity for BUY: ask=${float(last_ask):.4f} ≤ {float(MIN_LIQUIDITY):.2f} — skipping trade, will retry next tick"
                )
                self.last_trade_time = -1  # Allow retry next tick
                return
            if direction == "short" and last_bid <= MIN_LIQUIDITY:
                logger.warning(
                    f"⚠ No liquidity for SELL: bid=${float(last_bid):.4f} ≤ {float(MIN_LIQUIDITY):.2f} — skipping trade, will retry next tick"
                )
                self.last_trade_time = -1  # Allow retry next tick
                return

        # --- Phase 5 / 6: Execute ---
        if is_simulation:
            await self._record_paper_trade(
                fused, POSITION_SIZE_USD, current_price, direction, stop_loss_price, take_profit_price,
            )
        else:
            await self._place_real_order(
                fused, POSITION_SIZE_USD, current_price, direction, stop_loss_price, take_profit_price,
            )

    async def _record_paper_trade(
        self, signal, position_size, current_price, direction, stop_loss_price, take_profit_price,
    ):
        """
        Simulate a paper trade's exit and record its P&L.

        Stands in for a real on_position_closed: Polymarket binary options
        settle on-chain automatically (no sell, no fill/close event Nautilus
        can react to — see the note in _place_real_order), so there's no
        real exit event to record for either live or paper trades. This
        generates a plausible one, clamped to stop_loss_price/take_profit_price
        so those levels actually bound the simulated outcome instead of being
        pure decoration.
        """
        exit_delta = timedelta(minutes=1) if self.test_mode else timedelta(minutes=15)
        exit_time = datetime.now(timezone.utc) + exit_delta

        if "BULLISH" in str(signal.direction):
            movement = random.uniform(-0.02, 0.08)
        else:
            movement = random.uniform(-0.08, 0.02)

        raw_exit_price = current_price * (Decimal("1.0") + Decimal(str(movement)))

        # Clamp the simulated exit to the stop-loss/take-profit bounds so they
        # actually constrain the outcome rather than sitting unused.
        if direction == "long":
            exit_price = max(stop_loss_price, min(take_profit_price, raw_exit_price))
        else:
            exit_price = min(stop_loss_price, max(take_profit_price, raw_exit_price))
        exit_price = max(Decimal("0.01"), min(Decimal("0.99"), exit_price))

        if exit_price == take_profit_price:
            exit_reason = "TAKE_PROFIT"
        elif exit_price == stop_loss_price:
            exit_reason = "STOP_LOSS"
        else:
            exit_reason = "TIME"

        if direction == "long":
            pnl = position_size * (exit_price - current_price) / current_price
        else:
            pnl = position_size * (current_price - exit_price) / current_price

        outcome = "WIN" if pnl > 0 else "LOSS"
        paper_trade = PaperTrade(
            timestamp=datetime.now(timezone.utc),
            direction=direction.upper(),
            size_usd=float(position_size),
            price=float(current_price),
            signal_score=signal.score,
            signal_confidence=signal.confidence,
            outcome=outcome,
            exit_price=float(exit_price),
            pnl=float(pnl),
            exit_reason=exit_reason,
        )
        self.paper_trades.append(paper_trade)

        self.performance_tracker.record_trade(
            trade_id=f"paper_{int(datetime.now().timestamp())}",
            direction=direction,
            entry_price=current_price,
            exit_price=exit_price,
            size=position_size,
            entry_time=datetime.now(timezone.utc),
            exit_time=exit_time,
            signal_score=signal.score,
            signal_confidence=signal.confidence,
            metadata={
                "simulated": True,
                "num_signals": signal.num_signals if hasattr(signal, 'num_signals') else 1,
                "fusion_score": signal.score,
                "exit_reason": exit_reason,
            }
        )

        if hasattr(self, 'grafana_exporter') and self.grafana_exporter:
            self.grafana_exporter.increment_trade_counter(won=(pnl > 0))
            self.grafana_exporter.record_trade_duration(exit_delta.total_seconds())

        logger.info("=" * 80)
        logger.info("[SIMULATION] PAPER TRADE RECORDED")
        logger.info(f"  Direction: {direction.upper()}")
        logger.info(f"  Size: ${float(position_size):.2f}")
        logger.info(f"  Entry Price: ${float(current_price):,.4f}")
        logger.info(f"  Simulated Exit: ${float(exit_price):,.4f} ({exit_reason})")
        logger.info(f"  Simulated P&L: ${float(pnl):+.2f} ({movement*100:+.2f}%)")
        logger.info(f"  Outcome: {outcome}")
        logger.info(f"  Total Paper Trades: {len(self.paper_trades)}")
        logger.info("=" * 80)

        self._save_paper_trades()

    def _load_paper_trades(self) -> List["PaperTrade"]:
        if not os.path.exists(PAPER_TRADES_FILE):
            return []
        try:
            with open(PAPER_TRADES_FILE, 'r') as f:
                trades_data = json.load(f)
            trades = [PaperTrade.from_dict(d) for d in trades_data]
            logger.info(f"Loaded {len(trades)} paper trade(s) from previous session(s)")
            return trades
        except Exception as e:
            logger.error(f"Failed to load paper trades from {PAPER_TRADES_FILE}: {e}")
            return []

    def _save_paper_trades(self):
        try:
            if len(self.paper_trades) > MAX_PAPER_TRADES_HISTORY:
                self.paper_trades = self.paper_trades[-MAX_PAPER_TRADES_HISTORY:]
            trades_data = [t.to_dict() for t in self.paper_trades]
            with open(PAPER_TRADES_FILE, 'w') as f:
                json.dump(trades_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save paper trades: {e}")

    # ------------------------------------------------------------------
    # Real order (unchanged)
    # ------------------------------------------------------------------

    async def _place_real_order(
        self, signal, position_size, current_price, direction, stop_loss_price, take_profit_price,
    ):
        if not self.instrument_id:
            logger.error("No instrument available")
            return

        try:
            logger.info("=" * 80)
            logger.info("LIVE MODE - PLACING REAL ORDER!")
            logger.info("=" * 80)

            # NOTE: stop_loss_price/take_profit_price are logged for visibility
            # only — they are NOT enforced. There is no active monitoring or
            # SELL path here: Polymarket orders are BUY-and-hold-to-resolution
            # (both UP and DOWN bets are BUY orders — see below), and the
            # trade window (13-14 min into a 15-min market) leaves at most
            # ~1-2 minutes before the market settles automatically on-chain
            # anyway, so an early exit has essentially no window to matter.
            # An actual stop-loss would require subscribing to post-entry
            # ticks and submitting a SELL order on breach — not built.
            logger.info(
                f"  Risk levels (informational only): "
                f"stop_loss=${float(stop_loss_price):.4f} take_profit=${float(take_profit_price):.4f}"
            )

            # On Polymarket, both UP and DOWN are BUY orders.
            # Bullish = buy YES token (self.instrument_id — the market's primary/YES token)
            # Bearish = buy NO token  (self._no_instrument_id)
            # There is NO sell — you always buy whichever side you want.
            side = OrderSide.BUY

            if direction == "long":
                trade_instrument_id = self.instrument_id
                trade_label = "YES (UP)"
            else:
                no_id = self._no_instrument_id
                if no_id is None:
                    logger.warning(
                        "NO token instrument not found for this market — "
                        "cannot bet DOWN. Skipping trade."
                    )
                    return
                trade_instrument_id = no_id
                trade_label = "NO (DOWN)"

            instrument = self.cache.instrument(trade_instrument_id)
            if not instrument:
                logger.error(f"Instrument not in cache: {trade_instrument_id}")
                return

            logger.info(f"Buying {trade_label} token: {trade_instrument_id}")

            trade_price = float(current_price)
            max_usd_amount = float(position_size)

            precision = instrument.size_precision

            # Always BUY — the market-order patch converts this to a USD amount.
            # Pass dummy qty=5 (minimum) so Nautilus risk engine doesn't deny it.
            min_qty_val = float(getattr(instrument, 'min_quantity', None) or 5.0)
            token_qty = max(min_qty_val, 5.0)
            token_qty = round(token_qty, precision)
            logger.info(
                f"BUY {trade_label}: dummy qty={token_qty:.6f} "
                f"(patch converts to ${max_usd_amount:.2f} USD)"
            )

            qty = Quantity(token_qty, precision=precision)
            timestamp_ms = int(time.time() * 1000)
            unique_id = f"BTC-15MIN-${max_usd_amount:.0f}-{timestamp_ms}"

            order = self.order_factory.market(
                instrument_id=trade_instrument_id,
                order_side=side,
                quantity=qty,
                client_order_id=ClientOrderId(unique_id),
                quote_quantity=False,
                time_in_force=TimeInForce.IOC,
            )

            self.submit_order(order)

            logger.info(f"REAL ORDER SUBMITTED!")
            logger.info(f"  Order ID: {unique_id}")
            logger.info(f"  Direction: {trade_label}")
            logger.info(f"  Side: BUY")
            logger.info(f"  Token Quantity: {token_qty:.6f}")
            logger.info(f"  Estimated Cost: ~${max_usd_amount:.2f}")
            logger.info(f"  Price: ${trade_price:.4f}")
            logger.info("=" * 80)

            self._track_order_event("placed")

        except Exception as e:
            logger.error(f"Error placing real order: {e}")
            import traceback
            traceback.print_exc()
            self._track_order_event("rejected")

    # ------------------------------------------------------------------
    # Signal processing
    # ------------------------------------------------------------------

    def _process_signals(self, current_price, metadata=None):
        signals = []
        if metadata is None:
            metadata = {}

        processed_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, float):
                processed_metadata[key] = Decimal(str(value))
            else:
                processed_metadata[key] = value

        spike_signal = self.spike_detector.process(
            current_price=current_price,
            historical_prices=self.price_history,
            metadata=processed_metadata,
        )
        if spike_signal:
            signals.append(spike_signal)

        if 'sentiment_score' in processed_metadata:
            sentiment_signal = self.sentiment_processor.process(
                current_price=current_price,
                historical_prices=self.price_history,
                metadata=processed_metadata,
            )
            if sentiment_signal:
                signals.append(sentiment_signal)

        if 'spot_price' in processed_metadata:
            divergence_signal = self.divergence_processor.process(
                current_price=current_price,
                historical_prices=self.price_history,
                metadata=processed_metadata,
            )
            if divergence_signal:
                signals.append(divergence_signal)

        # --- Order Book Imbalance (real-time Polymarket CLOB depth) ---
        if processed_metadata.get('yes_token_id'):
            ob_signal = self.orderbook_processor.process(
                current_price=current_price,
                historical_prices=self.price_history,
                metadata=processed_metadata,
            )
            if ob_signal:
                signals.append(ob_signal)

        # --- Tick Velocity (last 60s of Polymarket probability movement) ---
        if processed_metadata.get('tick_buffer'):
            tv_signal = self.tick_velocity_processor.process(
                current_price=current_price,
                historical_prices=self.price_history,
                metadata=processed_metadata,
            )
            if tv_signal:
                signals.append(tv_signal)

        # --- Deribit Put/Call Ratio (institutional options sentiment) ---
        pcr_signal = self.deribit_pcr_processor.process(
            current_price=current_price,
            historical_prices=self.price_history,
            metadata=processed_metadata,
        )
        if pcr_signal:
            signals.append(pcr_signal)

        return signals

    # ------------------------------------------------------------------
    # Order events
    # ------------------------------------------------------------------

    def _track_order_event(self, event_type: str) -> None:
        """
        Safely track an order event on the performance tracker.

        PerformanceTracker does not expose `increment_order_counter`, so we
        use whichever method is actually available, or fall back to a no-op.
        Supported event_type values: "placed", "filled", "rejected".
        """
        try:
            pt = self.performance_tracker
            # Try the method that actually exists first
            if hasattr(pt, 'record_order_event'):
                pt.record_order_event(event_type)
            elif hasattr(pt, 'increment_counter'):
                pt.increment_counter(event_type)
            elif hasattr(pt, 'increment_order_counter'):
                pt.increment_order_counter(event_type)
            else:
                # No suitable method found – log and carry on
                logger.debug(
                    f"PerformanceTracker has no order-counter method; "
                    f"ignoring event '{event_type}'"
                )
        except Exception as e:
            logger.warning(f"Failed to track order event '{event_type}': {e}")

    def on_order_filled(self, event):
        logger.info("=" * 80)
        logger.info(f"ORDER FILLED!")
        logger.info(f"  Order: {event.client_order_id}")
        logger.info(f"  Fill Price: ${float(event.last_px):.4f}")
        logger.info(f"  Quantity: {float(event.last_qty):.6f}")
        logger.info("=" * 80)
        self._track_order_event("filled")

    def on_order_denied(self, event):
        logger.error("=" * 80)
        logger.error(f"ORDER DENIED!")
        logger.error(f"  Order: {event.client_order_id}")
        logger.error(f"  Reason: {event.reason}")
        logger.error("=" * 80)
        self._track_order_event("rejected")

    def on_order_rejected(self, event):
        """Handle order rejection — reset trade timer so we can retry next tick."""
        reason = str(getattr(event, 'reason', ''))
        reason_lower = reason.lower()
        if 'no orders found' in reason_lower or 'fak' in reason_lower or 'no match' in reason_lower:
            logger.warning(
                f"⚠ FAK rejected (no liquidity) — resetting timer to retry next tick\n"
                f"  Reason: {reason}"
            )
            self.last_trade_time = -1  # Allow retry on next quote tick
        else:
            logger.warning(f"Order rejected: {reason}")

    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------

    def on_stop(self):
        logger.info("Integrated BTC strategy stopped")
        logger.info(f"Total paper trades recorded: {len(self.paper_trades)}")
        for name in ("market_switch", "auto_restart"):
            if name in self.clock.timer_names:
                self.clock.cancel_timer(name)
        # Grafana's HTTP server is started/stopped by the runner
        # (run_integrated_bot), not here — see on_start.
