"""
MarketData: a snapshot of one Polymarket market instrument, produced by an
active-market finder (e.g. BTC15minActiveMarketFinder) and handed to a
Strategy to trade against. Deliberately dumb — no market-discovery logic,
no Nautilus API calls — so it stays reusable across different finders
(other timeframes) and different strategies.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from nautilus_trader.model.identifiers import InstrumentId


@dataclass(frozen=True)
class MarketData:
    """Everything a strategy needs to know about the market it should trade."""

    slug: str
    instrument_id: InstrumentId              # YES/primary token — subscribe to this for quotes
    no_instrument_id: Optional[InstrumentId]  # NO/opposing token, for buying the DOWN side
    yes_token_id: Optional[str]              # raw CLOB token id (for REST calls, e.g. order book depth)
    market_timestamp: int                    # market start, unix seconds
    start_time: datetime
    end_time: datetime

    def is_open(self, at: datetime) -> bool:
        """True if this market has started and not yet ended, at the given time."""
        return self.start_time <= at < self.end_time
