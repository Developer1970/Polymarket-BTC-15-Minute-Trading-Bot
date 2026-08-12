import asyncio
import os
import signal
import sys
import threading
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# nautilus_trader 2.0 moved to a compiled Rust/PyO3 core: no more
# nautilus_trader.adapters.polymarket.{gamma_markets,providers} Python
# submodules to monkey-patch, so patch_gamma_markets.py's target no longer
# exists. It's also no longer needed — PolymarketUpDownEventSlugConfig
# (see run_integrated_bot) is the native v2 replacement for what that patch
# was working around (server-side slug/time filtering of Gamma markets).

# Now import Nautilus
from nautilus_trader.common import Environment, FileWriterConfig, LoggerConfig, LogLevel
from nautilus_trader.config import LiveExecEngineConfig, LiveRiskEngineConfig
from nautilus_trader.model import StrategyId, TraderId
from nautilus_trader.live import LiveNode
from nautilus_trader.adapters.polymarket import (
    POLYMARKET,
    PolymarketDataClientConfig,
    PolymarketDataClientFactory,
    PolymarketExecClientConfig,
    PolymarketExecutionClientFactory,
    PolymarketInstrumentProviderConfig,
    PolymarketUpDownEventSlugConfig,
    SignatureType,
)

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# patch_market_orders.py monkey-patches PolymarketExecutionClient._submit_market_order
# (a pure-Python v1 class) to force BUY market orders to use a USD notional
# instead of a token quantity. That class no longer exists in v2 (compiled
# Rust core, same as gamma_markets above), so this patch can't apply — it
# fails its own ImportError handling and just logs a warning below, it
# doesn't crash. BtcUpDown15mStrategy doesn't place orders yet
# (_evaluate_signal is a stub), so this isn't blocking anything yet, but
# whatever replaces it will need a v2-native way to submit USD-denominated
# market BUY orders — investigate that when order placement is implemented.
from patch_market_orders import apply_market_order_patch
patch_applied = apply_market_order_patch()
if patch_applied:
    logger.info("Market order patch applied successfully")
else:
    logger.warning("Market order patch failed - orders may be rejected")

# Strategy lives in its own module so different strategies/market timeframes
# can be swapped in later without touching this runner.
from BtcUpDown15mStrategy import BtcUpDown15mStrategy, BtcUpDown15mConfig
from monitoring.grafana_exporter import get_grafana_exporter
from core.redis_client import get_redis_client


# ---------------------------------------------------------------------------
# Grafana lifecycle
#
# The metrics HTTP server is process-level infra, not strategy logic, so it's
# owned here rather than by the strategy. It needs its own event loop
# on a background thread since run_integrated_bot() itself is synchronous
# (node.run() blocks the main thread). Note this loop must be kept alive with
# run_forever() — calling run_until_complete(exporter.start()) and returning
# (as the strategy used to do) lets the loop stop right after the update-loop
# task is scheduled but before it ever executes, so metrics are posted once
# at startup and never refreshed again. Confirmed with a standalone repro.
# ---------------------------------------------------------------------------

_grafana_loop: "asyncio.AbstractEventLoop | None" = None
_grafana_thread: "threading.Thread | None" = None


def _start_grafana(exporter) -> None:
    global _grafana_loop, _grafana_thread

    def _run():
        global _grafana_loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _grafana_loop = loop
        try:
            loop.run_until_complete(exporter.start())
            loop.run_forever()
        finally:
            loop.close()

    _grafana_thread = threading.Thread(target=_run, daemon=True)
    _grafana_thread.start()


def _stop_grafana(exporter) -> None:
    global _grafana_loop, _grafana_thread
    if _grafana_loop is None:
        return
    try:
        future = asyncio.run_coroutine_threadsafe(exporter.stop(), _grafana_loop)
        future.result(timeout=5)
    except Exception as e:
        logger.warning(f"Error stopping Grafana exporter: {e}")
    _grafana_loop.call_soon_threadsafe(_grafana_loop.stop)
    if _grafana_thread:
        _grafana_thread.join(timeout=5)
    _grafana_loop = None
    _grafana_thread = None


# ---------------------------------------------------------------------------
# Auto-restart
#
# Process lifecycle — it just kills the process so
# 15m_bot_runner.py's wrapper loop relaunches it with fresh filters.
# ---------------------------------------------------------------------------

RESTART_AFTER_MINUTES = 90


def _trigger_restart() -> None:
    logger.warning("AUTO-RESTART TIME - Loading fresh filters")
    os.kill(os.getpid(), signal.SIGTERM)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_integrated_bot(
    simulation: bool = False,
    enable_grafana: bool = True,
    test_mode: bool = False,
    restart_after_minutes: int = RESTART_AFTER_MINUTES,
):
    """Run the integrated BTC 15-min trading bot - LOADS ALL BTC MARKETS FOR THE DAY"""

    print("=" * 80)
    print("INTEGRATED POLYMARKET BTC 15-MIN TRADING BOT")
    print("Nautilus + 7-Phase System + Redis Control")
    print("=" * 80)

    redis_client = get_redis_client()

    if redis_client:
        try:
            # ALWAYS overwrite Redis with the current session mode.
            # This prevents a stale value from a previous --live run
            # silently overriding --test-mode or --simulation runs.
            mode_value = '1' if simulation else '0'
            redis_client.set('btc_trading:simulation_mode', mode_value)
            mode_label = 'SIMULATION' if simulation else 'LIVE'
            logger.info(f"Redis simulation_mode forced to: {mode_label} ({mode_value})")
        except Exception as e:
            logger.warning(f"Could not set Redis simulation mode: {e}")

    print(f"\nConfiguration:")
    print(f"  Initial Mode: {'SIMULATION' if simulation else 'LIVE TRADING'}")
    print(f"  Redis Control: {'Enabled' if redis_client else 'Disabled'}")
    print(f"  Grafana: {'Enabled' if enable_grafana else 'Disabled'}")
    print(f"  Max Trade Size: ${os.getenv('MARKET_BUY_USD', '1.00')}")
    print()

    # =========================================================================
    # PolymarketUpDownEventSlugConfig is the v2-native replacement for the old
    # manual btc-updown-15m-<unix_ts> slug loop + generic filters={"slug": ...}
    # hack (which needed patch_gamma_markets.py to work around v1 Gamma API
    # filtering bugs — see the note near the top of this file). The adapter
    # now generates and refreshes the slugs itself.
    # =========================================================================
    instrument_config = PolymarketInstrumentProviderConfig(
        event_slug_builder=PolymarketUpDownEventSlugConfig(
            assets=["btc"],
            interval_mins=15,
            periods=3,
            start_offset_periods=0,
        ),
    )

    data_client_config = PolymarketDataClientConfig(
        instrument_config=instrument_config,
        update_instruments_interval_mins=15,
        # subscribe_new_markets subscribes to a PLATFORM-WIDE new-market feed
        # (every category — sports, esports, politics, everything), not just
        # markets matching event_slug_builder below. That's what was pulling
        # in unrelated slugs (e.g. League of Legends markets) and warning
        # when they didn't produce loadable instruments. We only want BTC
        # 15-min windows, which event_slug_builder + the 15-min periodic
        # refresh above already catches as each new window rolls around —
        # no need for the global feed.
        subscribe_new_markets=False,
    )

    # Credentials live only on the exec client config in v2 — the data client
    # no longer needs them (public market data requires no auth).
    exec_client_config = PolymarketExecClientConfig(
        private_key=os.getenv("POLYMARKET_PK"),
        api_key=os.getenv("POLYMARKET_API_KEY"),
        api_secret=os.getenv("POLYMARKET_API_SECRET"),
        passphrase=os.getenv("POLYMARKET_PASSPHRASE"),
        signature_type=SignatureType.PolyProxy,
    )

    # Without an explicit strategy_id, Nautilus logs this strategy under a
    # generic "DataActor-<object id>" component name — unreadable in logs.
    strategy = BtcUpDown15mStrategy(
        BtcUpDown15mConfig(strategy_id=StrategyId("BTCUPDOWN15M-001"))
    )

    print("\nBuilding Nautilus node...")
    trader_id = TraderId.from_str("BTC-15MIN-INTEGRATED-001")
    node = (
        LiveNode.builder("BTC-15MIN-INTEGRATED-001", trader_id, Environment.LIVE)
        .with_logging(
            LoggerConfig(
                stdout_level=LogLevel.INFO,
                file_config=FileWriterConfig(directory="./logs/nautilus"),
                # Polymarket's book goes one-sided as each market resolves (winning
                # side -> ~0.999, losing side -> ~0.001/no size), so the adapter's
                # own drop_quotes_missing_side=True path (the default) logs a
                # WARNING per dropped tick — expected, not a bug. Since unsubscribe
                # isn't supported by Polymarket, every market we've switched away
                # from keeps streaming until the next auto-restart, so these
                # warnings pile up from closed markets too. Silence them at the
                # component level; real errors still surface.
                component_levels={
                    "DataClient-POLYMARKET": "ERROR",
                    "BTCUPDOWN15M-001": "DEBUG",
                },
            )
        )
        .with_risk_engine_config(LiveRiskEngineConfig(bypass=simulation))
        # Reconciliation (the startup "mass status" sync of pre-existing
        # orders/trades/positions from the venue) is off: this bot is
        # currently paper-trading only (tuning the predictive algorithm, not
        # placing real orders), so there's nothing on the venue to sync with.
        # Re-enable this before any real order placement goes live.
        .with_exec_engine_config(LiveExecEngineConfig(reconciliation=False))
        .add_data_client(POLYMARKET, PolymarketDataClientFactory(), data_client_config)
        .add_exec_client(POLYMARKET, PolymarketExecutionClientFactory(), exec_client_config)
        .build()
    )
    node.add_strategy(strategy)
    logger.info("Nautilus node built successfully")

    grafana_exporter = get_grafana_exporter() if enable_grafana else None
    if grafana_exporter:
        _start_grafana(grafana_exporter)
        logger.info("Grafana metrics started on port 8000")

    restart_timer = threading.Timer(restart_after_minutes * 60, _trigger_restart)
    restart_timer.daemon = True
    restart_timer.start()

    print()
    print("=" * 80)
    print("BOT STARTING")
    print("=" * 80)

    try:
        node.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        restart_timer.cancel()
        node.dispose()
        if grafana_exporter:
            _stop_grafana(grafana_exporter)
        logger.info("Bot stopped")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Integrated BTC 15-Min Trading Bot")
    parser.add_argument("--live", action="store_true",
                        help="Run in LIVE mode (real money at risk!). Default is simulation.")
    parser.add_argument("--no-grafana", action="store_true", help="Disable Grafana metrics")
    parser.add_argument("--test-mode", action="store_true",
                        help="Run in TEST MODE (trade every minute for faster testing)")

    args = parser.parse_args()
    enable_grafana = not args.no_grafana
    test_mode = args.test_mode

    # --test-mode ALWAYS forces simulation even if --live is also passed
    if args.test_mode:
        simulation = True
    else:
        simulation = not args.live

    if not simulation:
        logger.warning("=" * 80)
        logger.warning("LIVE TRADING MODE — REAL MONEY AT RISK!")
        logger.warning("=" * 80)
    else:
        logger.info("=" * 80)
        logger.info(f"SIMULATION MODE — {'TEST MODE (fast clock)' if test_mode else 'paper trading only'}")
        logger.info("No real orders will be placed.")
        logger.info("=" * 80)

    run_integrated_bot(simulation=simulation, enable_grafana=enable_grafana, test_mode=test_mode)


if __name__ == "__main__":
    main()
