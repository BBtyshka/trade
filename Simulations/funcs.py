"""
Backward-compatible functions that wrap the new object-oriented engine.

For new code, consider using the TradingEngine directly from objects.py
"""

import sys
import os

# Add parent directory to path to import objects
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Simulations.objects import (
    brownian_prices,
    sma,
    TradingEngine,
    YFinanceDataSource,
    SimulatedDataSource,
    SMAStrategy,
    StrategyResult
)

# Re-export for backward compatibility
__all__ = [
    'brownian_prices',
    'sma',
    'TradingEngine',
    'YFinanceDataSource',
    'SimulatedDataSource',
    'SMAStrategy',
    'StrategyResult'
]