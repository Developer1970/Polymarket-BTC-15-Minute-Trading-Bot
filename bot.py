import asyncio
import os
import sys
import threading
from pathlib import Path
from datetime import datetime, timezone

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from patch_gamma_markets import apply_gamma_markets_patch, verify_patch
    patch_applied = apply_gamma_markets_patch()
    if patch_applied:
        verify_patch()
    else:
        print("ERROR: Failed to apply gamma_market patch")
        sys.exit(1)
except ImportError as e:
    print(f"ERROR: Could not import patch module: {e}")
    print("Make sure patch_gamma_markets.py is in the same directory")
    sys.exit(1)

# Now import Nautilus
from nautilus_trader.config import (
    InstrumentProviderConfig,
    LiveDataEngineConfig,
    LiveExecEngineConfig,
    LiveRiskEngineConfig,
    LoggingConfig,
    TradingNodeConfig,
)
from nautilus_trader.live.node import TradingNode
from nautilus_trader.adapters.polymarket import POLYMARKET
from nautilus_trader.adapters.polymarket import (
    PolymarketDataClientConfig,
    PolymarketExecClientConfig,
)
from nautilus_trader.adapters.polymarket.factories import (
    PolymarketLiveDataClientFactory,
    PolymarketLiveExecClientFactory,
)

from dotenv import load_dotenv
from loguru import logger
import redis

load_dotenv()
from patch_market_orders import apply_market_order_patch
patch_applied = apply_market_order_patch()
if patch_applied:
    logger.info("Market order patch applied successfully")
else:
    logger.warning("Market order patch failed - orders may be rejected")

# Strategy lives in its own module so different strategies/market timeframes
# can be swapped in later without touching this runner.
from core.strategy_brain.strategies.integrated_btc_strategy import (
    IntegratedBTCStrategy,
    QUOTE_STABILITY_REQUIRED,
)
from monitoring.grafana_exporter import get_grafana_exporter

def init_redis():
    """Initialize Redis connection for simulation mode control."""
    try:
        redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_DB', 2)),
            decode_responses=True,
            socket_connect_timeout=5,
            socket_keepalive=True
        )
        redis_client.ping()
        logger.info("Redis connection established")
        return redis_client
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        logger.warning("Simulation mode will be static (from .env)")
        return None


# ---------------------------------------------------------------------------
# Grafana lifecycle
#
# The metrics HTTP server is process-level infra, not strategy logic, so it's
# owned here rather than by IntegratedBTCStrategy (which only records metrics
# into the already-running exporter singleton). It needs its own event loop
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
# Runner
# ---------------------------------------------------------------------------

def run_integrated_bot(simulation: bool = False, enable_grafana: bool = True, test_mode: bool = False):
    """Run the integrated BTC 15-min trading bot - LOADS ALL BTC MARKETS FOR THE DAY"""

    print("=" * 80)
    print("INTEGRATED POLYMARKET BTC 15-MIN TRADING BOT")
    print("Nautilus + 7-Phase System + Redis Control")
    print("=" * 80)

    redis_client = init_redis()

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
    print(f"  Quote stability gate: {QUOTE_STABILITY_REQUIRED} valid ticks")
    print()

    # =========================================================================
    # Slug timestamps ARE standard Unix timestamps (no offset) aligned to
    # 15-min boundaries. Generate slugs for current + next 24 hours.
    # =========================================================================
    now = datetime.now(timezone.utc)
    unix_interval_start = (int(now.timestamp()) // 900) * 900  # current 15-min boundary

    btc_slugs = []
    for i in range(-1, 97):  # include 1 prior interval (in case we're just after boundary)
        timestamp = unix_interval_start + (i * 900)
        btc_slugs.append(f"btc-updown-15m-{timestamp}")

    filters = {
        "active": True,
        "closed": False,
        "archived": False,
        "slug": tuple(btc_slugs),
        "limit": 100,
    }

    logger.info("=" * 80)
    logger.info("LOADING BTC 15-MIN MARKETS BY SLUG")
    logger.info(f"  Interval start: {unix_interval_start} | Count: {len(btc_slugs)}")
    logger.info(f"  First: {btc_slugs[0]}  Last: {btc_slugs[-1]}")
    logger.info("=" * 80)

    instrument_cfg = InstrumentProviderConfig(
        load_all=True,
        filters=filters,
        use_gamma_markets=True,
    )

    poly_data_cfg = PolymarketDataClientConfig(
        private_key=os.getenv("POLYMARKET_PK"),
        api_key=os.getenv("POLYMARKET_API_KEY"),
        api_secret=os.getenv("POLYMARKET_API_SECRET"),
        passphrase=os.getenv("POLYMARKET_PASSPHRASE"),
        signature_type=1,
        instrument_provider=instrument_cfg,
    )

    poly_exec_cfg = PolymarketExecClientConfig(
        private_key=os.getenv("POLYMARKET_PK"),
        api_key=os.getenv("POLYMARKET_API_KEY"),
        api_secret=os.getenv("POLYMARKET_API_SECRET"),
        passphrase=os.getenv("POLYMARKET_PASSPHRASE"),
        signature_type=1,
        instrument_provider=instrument_cfg,
    )

    config = TradingNodeConfig(
        environment="live",
        trader_id="BTC-15MIN-INTEGRATED-001",
        logging=LoggingConfig(
            log_level="INFO",
            log_directory="./logs/nautilus",
            # Polymarket's book goes one-sided as each market resolves (winning
            # side -> ~0.999, losing side -> ~0.001/no size), so the adapter's
            # own drop_quotes_missing_side=True path (the default) logs a
            # WARNING per dropped tick — expected, not a bug. Since unsubscribe
            # isn't supported by Polymarket (see _switch_to_next_market), every
            # market we've switched away from keeps streaming until the next
            # auto-restart, so these warnings pile up from closed markets too.
            # Silence them at the component level; real errors still surface.
            log_component_levels={"DataClient-POLYMARKET": "ERROR"},
        ),
        data_engine=LiveDataEngineConfig(qsize=6000),
        exec_engine=LiveExecEngineConfig(qsize=6000),
        risk_engine=LiveRiskEngineConfig(bypass=simulation),
        data_clients={POLYMARKET: poly_data_cfg},
        exec_clients={POLYMARKET: poly_exec_cfg},
    )

    strategy = IntegratedBTCStrategy(
        redis_client=redis_client,
        enable_grafana=enable_grafana,
        test_mode=test_mode,
    )

    print("\nBuilding Nautilus node...")
    node = TradingNode(config=config)
    node.add_data_client_factory(POLYMARKET, PolymarketLiveDataClientFactory)
    node.add_exec_client_factory(POLYMARKET, PolymarketLiveExecClientFactory)
    node.trader.add_strategy(strategy)
    node.build()
    logger.info("Nautilus node built successfully")

    grafana_exporter = get_grafana_exporter() if enable_grafana else None
    if grafana_exporter:
        _start_grafana(grafana_exporter)
        logger.info("Grafana metrics started on port 8000")

    print()
    print("=" * 80)
    print("BOT STARTING")
    print("=" * 80)

    try:
        node.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
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
