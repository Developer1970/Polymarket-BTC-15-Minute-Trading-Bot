"""
Shared Redis connection for simulation-mode control (btc_trading:simulation_mode).

Singleton getter, matching the get_risk_engine()/get_performance_tracker()/
get_grafana_exporter()/get_learning_engine() pattern used elsewhere in this
project — callers ask for the client when they need it instead of having it
threaded through constructors and stored as instance state.
"""
import os
from typing import Optional

import redis
from loguru import logger

_redis_client: Optional["redis.Redis"] = None
_connection_failed = False


def get_redis_client() -> Optional["redis.Redis"]:
    """
    Return the shared Redis client, connecting on first call.

    Returns None if Redis isn't reachable — callers should treat that as
    "no Redis available" rather than raising. Only attempts to connect once;
    a failed connection is remembered so later calls don't retry every time.
    """
    global _redis_client, _connection_failed

    if _redis_client is not None:
        return _redis_client
    if _connection_failed:
        return None

    try:
        client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_DB', 2)),
            decode_responses=True,
            socket_connect_timeout=5,
            socket_keepalive=True,
        )
        client.ping()
        logger.info("Redis connection established")
        _redis_client = client
        return client
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        logger.warning("Simulation mode will be static (from .env)")
        _connection_failed = True
        return None
