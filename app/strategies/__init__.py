"""Strategy registry. Add new strategies here so every page can list them."""
from app.strategies.base import Signal, Strategy
from app.strategies.breakout import Breakout
from app.strategies.ema_rsi import EmaRsi
from app.strategies.rules import Rules
from app.strategies.trend_pullback import TrendPullback

STRATEGIES: dict[str, Strategy] = {s.name: s for s in [EmaRsi(), TrendPullback(), Breakout(), Rules()]}


def get_strategy(name: str) -> Strategy:
    return STRATEGIES[name]


__all__ = ["Signal", "Strategy", "STRATEGIES", "get_strategy"]
