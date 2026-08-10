"""
BTC15minActiveMarketFinder

Scans Nautilus's instrument cache for Polymarket BTC 15-minute Up/Down
markets and selects the one a strategy should currently be trading:
- the active market, if one is currently open, or
- the nearest upcoming market, if none is currently open.

This is pure market-discovery logic. It never subscribes/unsubscribes to
data, schedules timers, or touches strategy state — a Strategy asks it for
a MarketData and decides what to do with that information.
"""
from datetime import datetime, timezone
from typing import List, Optional

from loguru import logger

from core.market_finder.market_data import MarketData


class BTC15minActiveMarketFinder:
    """Finds the currently-active (or nearest upcoming) BTC 15-min Up/Down market."""

    INTERVAL_SECONDS = 900

    def __init__(self, cache):
        """
        Args:
            cache: A Nautilus strategy's `self.cache` (nautilus_trader.cache.Cache),
                used to read instruments loaded by the Polymarket instrument provider.
        """
        self._cache = cache
        self._markets: List[MarketData] = []
        self._current_index: int = -1

    @property
    def current(self) -> Optional[MarketData]:
        """The currently-selected market, or None if load() hasn't run yet."""
        if 0 <= self._current_index < len(self._markets):
            return self._markets[self._current_index]
        return None

    @property
    def markets(self) -> List[MarketData]:
        return list(self._markets)

    def load(self) -> MarketData:
        """
        Scan the instrument cache for BTC 15-min markets, sort them by start
        time, and select the market to trade: the active one if one exists,
        otherwise the nearest upcoming one. Call once at strategy startup,
        after the instrument provider has populated the cache.
        """
        self._markets = self._scan_markets()
        if not self._markets:
            raise RuntimeError("No BTC 15-min markets found in the instrument cache")

        now = datetime.now(timezone.utc)
        active = [m for m in self._markets if m.is_open(now)]

        if active:
            selected = active[0]
            logger.info(f"✓ CURRENT MARKET: {selected.slug} (active)")
        else:
            future = [m for m in self._markets if m.start_time > now]
            selected = min(future, key=lambda m: m.start_time) if future else self._markets[-1]
            logger.info(
                f"⚠ NO CURRENT MARKET — nearest: {selected.slug} "
                f"(starts {selected.start_time.strftime('%H:%M:%S')} UTC)"
            )

        self._current_index = self._markets.index(selected)
        return selected

    def advance(self) -> Optional[MarketData]:
        """
        Move to the next market in the loaded list if it's due to have
        started. Returns the new MarketData, or None if the next market
        isn't ready yet (caller should retry shortly) or there are no more
        markets loaded (caller should reload/restart).
        """
        next_index = self._current_index + 1
        if next_index >= len(self._markets):
            logger.warning("No more BTC 15-min markets loaded")
            return None

        next_market = self._markets[next_index]
        if datetime.now(timezone.utc) < next_market.start_time:
            logger.info(f"Waiting for next market at {next_market.start_time.strftime('%H:%M:%S')}")
            return None

        self._current_index = next_index
        logger.info(f"→ ADVANCED TO NEXT MARKET: {next_market.slug}")
        return next_market

    def _scan_markets(self) -> List[MarketData]:
        """Scan the instrument cache for BTC 15-min Up/Down markets, pairing YES/NO tokens by slug."""
        instruments = self._cache.instruments()
        now = datetime.now(timezone.utc)
        current_timestamp = int(now.timestamp())

        raw = []
        for instrument in instruments:
            try:
                if not (hasattr(instrument, 'info') and instrument.info):
                    continue
                question = instrument.info.get('question', '').lower()
                slug = instrument.info.get('market_slug', '').lower()

                if not (('btc' in question or 'btc' in slug) and '15m' in slug):
                    continue

                timestamp_part = slug.split('-')[-1]
                market_timestamp = int(timestamp_part)

                # The slug timestamp IS the market start time (Unix, no offset).
                # end_date_iso is a DATE-only string (e.g. "2026-02-20"), NOT a
                # datetime, so parsing it gives midnight UTC which is wrong for
                # intraday markets. Always derive end from start + interval.
                end_timestamp = market_timestamp + self.INTERVAL_SECONDS

                # Only include markets that haven't ended yet
                if end_timestamp <= current_timestamp:
                    continue

                # Extract the CLOB token_id for REST order-book calls.
                # Nautilus instrument ID format: {condition_id}-{token_id}.POLYMARKET
                raw_id = str(instrument.id)
                without_suffix = raw_id.split('.')[0] if '.' in raw_id else raw_id
                token_id = without_suffix.split('-')[-1] if '-' in without_suffix else without_suffix

                raw.append({
                    'instrument_id': instrument.id,
                    'slug': slug,
                    'start_time': datetime.fromtimestamp(market_timestamp, tz=timezone.utc),
                    'end_time': datetime.fromtimestamp(end_timestamp, tz=timezone.utc),
                    'market_timestamp': market_timestamp,
                    'token_id': token_id,
                })
            except (ValueError, IndexError, AttributeError):
                continue

        # Pair YES and NO tokens by slug. Each Polymarket market has two
        # tokens loaded as separate Nautilus instruments — the first one
        # found for a slug is treated as the YES/UP token (the one we
        # subscribe to and trade against by default), the second as NO/DOWN.
        seen_slugs = {}
        ordered_slugs = []
        for inst in raw:
            slug = inst['slug']
            if slug not in seen_slugs:
                seen_slugs[slug] = {'yes': inst, 'no': None}
                ordered_slugs.append(slug)
            else:
                seen_slugs[slug]['no'] = inst

        markets = [
            MarketData(
                slug=slug,
                instrument_id=seen_slugs[slug]['yes']['instrument_id'],
                no_instrument_id=(
                    seen_slugs[slug]['no']['instrument_id'] if seen_slugs[slug]['no'] else None
                ),
                yes_token_id=seen_slugs[slug]['yes']['token_id'],
                market_timestamp=seen_slugs[slug]['yes']['market_timestamp'],
                start_time=seen_slugs[slug]['yes']['start_time'],
                end_time=seen_slugs[slug]['yes']['end_time'],
            )
            for slug in ordered_slugs
        ]

        markets.sort(key=lambda m: m.market_timestamp)

        logger.info("=" * 80)
        logger.info(f"FOUND {len(markets)} BTC 15-MIN MARKETS:")
        for i, m in enumerate(markets):
            status = "ACTIVE" if m.is_open(now) else "FUTURE" if m.start_time > now else "PAST"
            logger.info(
                f"  [{i}] {m.slug}: {status} "
                f"(starts at {m.start_time.strftime('%H:%M:%S')}, ends at {m.end_time.strftime('%H:%M:%S')})"
            )
        logger.info("=" * 80)

        return markets
