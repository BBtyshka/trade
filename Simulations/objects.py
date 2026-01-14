"""
Trading Engine - Object-Oriented Trading Simulation Framework

This module provides a modular, extensible framework for backtesting
and simulating trading strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Callable
from enum import Enum
import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================================
# ENUMS & DATA CLASSES
# ============================================================================

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class PositionStatus(Enum):
    FLAT = "FLAT"
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Trade:
    """Represents a single executed trade."""
    timestamp: int
    side: OrderSide
    price: float
    quantity: float = 1.0
    
    @property
    def value(self) -> float:
        return self.price * self.quantity


@dataclass
class Position:
    """Tracks current position state."""
    status: PositionStatus = PositionStatus.FLAT
    entry_price: Optional[float] = None
    entry_time: Optional[int] = None
    quantity: float = 0.0
    
    def open_long(self, price: float, time: int, qty: float = 1.0):
        self.status = PositionStatus.LONG
        self.entry_price = price
        self.entry_time = time
        self.quantity = qty
    
    def close(self):
        self.status = PositionStatus.FLAT
        self.entry_price = None
        self.entry_time = None
        self.quantity = 0.0
    
    @property
    def is_flat(self) -> bool:
        return self.status == PositionStatus.FLAT
    
    @property
    def is_long(self) -> bool:
        return self.status == PositionStatus.LONG


@dataclass
class StrategyResult:
    """Contains the results of a strategy backtest."""
    trades: List[Trade] = field(default_factory=list)
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    buy_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    sell_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    buy_prices: np.ndarray = field(default_factory=lambda: np.array([]))
    sell_prices: np.ndarray = field(default_factory=lambda: np.array([]))
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    
    @property
    def num_trades(self) -> int:
        return len(self.trades)
    
    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        # Calculate win rate from paired trades
        pairs = min(len(self.buy_prices), len(self.sell_prices))
        if pairs == 0:
            return 0.0
        wins = sum(1 for i in range(pairs) if self.sell_prices[i] > self.buy_prices[i])
        return wins / pairs


# ============================================================================
# DATA SOURCES
# ============================================================================

class DataSource(ABC):
    """Abstract base class for data sources."""
    
    @abstractmethod
    def fetch(self) -> np.ndarray:
        """Fetch price data."""
        pass
    
    @abstractmethod
    def get_dataframe(self) -> pd.DataFrame:
        """Get full dataframe with OHLCV data."""
        pass


class YFinanceDataSource(DataSource):
    """Fetches data from Yahoo Finance."""
    
    def __init__(self, ticker: str, period: str = "1y", interval: str = "1h"):
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self._df: Optional[pd.DataFrame] = None
        self._prices: Optional[np.ndarray] = None
    
    def fetch(self) -> np.ndarray:
        """Fetch mid prices as numpy array."""
        if self._prices is None:
            df = self.get_dataframe()
            self._prices = np.array(df["Mid"])
        return self._prices
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get full dataframe with computed Mid price."""
        if self._df is None:
            ticker = yf.Ticker(self.ticker)
            self._df = pd.DataFrame(
                ticker.history(period=self.period, interval=self.interval)
            ).reset_index()
            self._df["Mid"] = (self._df["High"] + self._df["Low"]) / 2
        return self._df
    
    def refresh(self):
        """Clear cache and refetch data."""
        self._df = None
        self._prices = None


class SimulatedDataSource(DataSource):
    """Generates simulated price data using Brownian motion."""
    
    def __init__(self, base_data: np.ndarray, steps: int = 200):
        self.base_data = base_data
        self.steps = steps
        self._prices: Optional[np.ndarray] = None
    
    def fetch(self) -> np.ndarray:
        """Generate new simulated prices."""
        self._prices = self._brownian_motion()
        return self._prices
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get simulated data as dataframe."""
        prices = self.fetch()
        return pd.DataFrame({
            "Step": range(len(prices)),
            "Price": prices,
            "Mid": prices
        })
    
    def _brownian_motion(self) -> np.ndarray:
        """Generate prices using geometric Brownian motion."""
        r = np.diff(np.log(self.base_data))
        mu = np.mean(r)
        sd = np.std(r)
        
        last = self.base_data[-1]
        norm = np.random.normal(mu, sd, self.steps)
        sim = last * np.exp(np.cumsum(norm))
        return sim
    
    def set_steps(self, steps: int):
        """Update simulation steps."""
        self.steps = steps


# ============================================================================
# STRATEGIES
# ============================================================================

class Strategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
    
    @abstractmethod
    def run(self, prices: np.ndarray) -> StrategyResult:
        """Execute strategy on price data."""
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        pass
    
    @abstractmethod
    def set_params(self, **kwargs):
        """Set strategy parameters."""
        pass


class SMAStrategy(Strategy):
    """Simple Moving Average Crossover Strategy."""
    
    def __init__(self, short_window: int = 3, long_window: int = 9):
        super().__init__(name="SMA Crossover")
        self.short_window = short_window
        self.long_window = long_window
    
    def run(self, prices: np.ndarray) -> StrategyResult:
        """Execute SMA crossover strategy."""
        result = StrategyResult()
        
        s_averages = np.array([])
        l_averages = np.array([])
        
        buy_prices = []
        sell_prices = []
        buy_idx = []
        sell_idx = []
        trades = []
        
        position = Position()
        
        for i in range(len(prices)):
            if i < self.long_window:
                continue
            
            # Calculate moving averages
            long_sma = np.mean(prices[i - self.long_window:i])
            l_averages = np.append(l_averages, long_sma)
            short_sma = np.mean(prices[i - self.short_window:i])
            s_averages = np.append(s_averages, short_sma)
            
            if len(l_averages) < 2:
                continue
            
            l_diff = l_averages[-1] - l_averages[-2]
            
            # Crossover detection
            crossed_up = (
                s_averages[-2] < l_averages[-2] and 
                s_averages[-1] >= l_averages[-1] and 
                l_diff > 0
            )
            crossed_dn = (
                s_averages[-2] > l_averages[-2] and 
                s_averages[-1] <= l_averages[-1] and 
                l_diff < 0
            )
            
            # Generate signals
            if crossed_up:
                buy_prices.append(prices[i])
                buy_idx.append(i)
                trades.append(Trade(timestamp=i, side=OrderSide.BUY, price=prices[i]))
                position.open_long(prices[i], i)
            elif crossed_dn:
                sell_prices.append(prices[i])
                sell_idx.append(i)
                trades.append(Trade(timestamp=i, side=OrderSide.SELL, price=prices[i]))
                position.close()
        
        # Calculate PnL
        result.buy_prices = np.array(buy_prices)
        result.sell_prices = np.array(sell_prices)
        result.buy_indices = np.array(buy_idx)
        result.sell_indices = np.array(sell_idx)
        result.trades = trades
        
        result.realized_pnl = np.sum(result.sell_prices) - np.sum(result.buy_prices[:len(result.sell_prices)])
        
        # Unrealized PnL for open position
        if len(buy_prices) > len(sell_prices) and len(buy_prices) > 0:
            result.unrealized_pnl = prices[-1] - buy_prices[-1]
        
        result.total_pnl = result.realized_pnl + result.unrealized_pnl
        
        # Build equity curve
        result.equity_curve = self._build_equity_curve(prices, result)
        
        return result
    
    def _build_equity_curve(self, prices: np.ndarray, result: StrategyResult) -> np.ndarray:
        """Build cumulative equity curve."""
        equity = np.zeros(len(prices))
        
        for idx in result.buy_indices.astype(int):
            equity[idx] = -prices[idx]
        for idx in result.sell_indices.astype(int):
            equity[idx] = prices[idx]
        
        return np.cumsum(equity)
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "short_window": self.short_window,
            "long_window": self.long_window
        }
    
    def set_params(self, **kwargs):
        if "short_window" in kwargs:
            self.short_window = kwargs["short_window"]
        if "long_window" in kwargs:
            self.long_window = kwargs["long_window"]


# ============================================================================
# TRADING ENGINE
# ============================================================================

class TradingEngine:
    """
    Main trading engine that orchestrates data, strategies, and execution.
    
    Usage:
        engine = TradingEngine()
        engine.set_data_source(YFinanceDataSource("AAPL"))
        engine.add_strategy(SMAStrategy(short_window=5, long_window=20))
        results = engine.run()
    """
    
    def __init__(self):
        self._data_source: Optional[DataSource] = None
        self._strategies: Dict[str, Strategy] = {}
        self._results: Dict[str, StrategyResult] = {}
        self._prices: Optional[np.ndarray] = None
        self._callbacks: Dict[str, List[Callable]] = {
            "on_data_loaded": [],
            "on_strategy_complete": [],
            "on_trade": [],
        }
    
    # ---- Data Management ----
    
    def set_data_source(self, source: DataSource):
        """Set the data source for the engine."""
        self._data_source = source
        return self
    
    def load_data(self) -> np.ndarray:
        """Load data from the configured source."""
        if self._data_source is None:
            raise ValueError("No data source configured. Use set_data_source() first.")
        
        self._prices = self._data_source.fetch()
        self._trigger_callbacks("on_data_loaded", self._prices)
        return self._prices
    
    @property
    def prices(self) -> Optional[np.ndarray]:
        """Get currently loaded prices."""
        return self._prices
    
    @property
    def data_source(self) -> Optional[DataSource]:
        """Get current data source."""
        return self._data_source
    
    # ---- Strategy Management ----
    
    def add_strategy(self, strategy: Strategy, name: Optional[str] = None):
        """Add a strategy to the engine."""
        key = name or strategy.name
        self._strategies[key] = strategy
        return self
    
    def remove_strategy(self, name: str):
        """Remove a strategy by name."""
        if name in self._strategies:
            del self._strategies[name]
        return self
    
    def get_strategy(self, name: str) -> Optional[Strategy]:
        """Get a strategy by name."""
        return self._strategies.get(name)
    
    def list_strategies(self) -> List[str]:
        """List all registered strategy names."""
        return list(self._strategies.keys())
    
    # ---- Execution ----
    
    def run(self, strategy_name: Optional[str] = None) -> Dict[str, StrategyResult]:
        """
        Run strategies on loaded data.
        
        Args:
            strategy_name: If provided, run only this strategy.
                          Otherwise, run all strategies.
        
        Returns:
            Dictionary mapping strategy names to their results.
        """
        if self._prices is None:
            self.load_data()
        
        strategies_to_run = {}
        if strategy_name:
            if strategy_name not in self._strategies:
                raise ValueError(f"Strategy '{strategy_name}' not found.")
            strategies_to_run[strategy_name] = self._strategies[strategy_name]
        else:
            strategies_to_run = self._strategies
        
        for name, strategy in strategies_to_run.items():
            if self._prices:
                result = strategy.run(self._prices)
            self._results[name] = result
            self._trigger_callbacks("on_strategy_complete", name, result)
            
            # Trigger trade callbacks
            for trade in result.trades:
                self._trigger_callbacks("on_trade", trade)
        
        return self._results
    
    def run_with_prices(self, prices: np.ndarray, strategy_name: Optional[str] = None) -> Dict[str, StrategyResult]:
        """Run strategies on provided prices without loading from data source."""
        self._prices = prices
        return self.run(strategy_name)
    
    # ---- Results ----
    
    def get_results(self, strategy_name: Optional[str] = None) -> Dict[str, StrategyResult]:
        """Get results from last run."""
        if strategy_name:
            return {strategy_name: self._results.get(strategy_name)}
        return self._results
    
    def get_summary(self) -> pd.DataFrame:
        """Get a summary DataFrame of all strategy results."""
        data = []
        for name, result in self._results.items():
            data.append({
                "Strategy": name,
                "Total PnL": result.total_pnl,
                "Realized PnL": result.realized_pnl,
                "Unrealized PnL": result.unrealized_pnl,
                "Num Trades": result.num_trades,
                "Win Rate": f"{result.win_rate:.2%}"
            })
        return pd.DataFrame(data)
    
    # ---- Callbacks/Events ----
    
    def on(self, event: str, callback: Callable):
        """Register an event callback."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
        return self
    
    def _trigger_callbacks(self, event: str, *args):
        """Trigger all callbacks for an event."""
        for callback in self._callbacks.get(event, []):
            callback(*args)
    
    # ---- Utility ----
    
    def reset(self):
        """Reset engine state."""
        self._results.clear()
        self._prices = None
        return self
    
    def __repr__(self) -> str:
        return (
            f"TradingEngine("
            f"data_source={type(self._data_source).__name__ if self._data_source else None}, "
            f"strategies={list(self._strategies.keys())})"
        )


# ============================================================================
# SIMULATION HELPERS
# ============================================================================

def brownian_prices(data: np.ndarray, steps: int) -> np.ndarray:
    """
    Generate simulated prices using geometric Brownian motion.
    
    Standalone function for backward compatibility.
    """
    source = SimulatedDataSource(data, steps)
    return source.fetch()


def sma(data: np.ndarray, short_window: int, long_window: int) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run SMA strategy on data.
    
    Standalone function for backward compatibility.
    Returns: (profit, buy_idx, buy_prices, sell_idx, sell_prices)
    """
    strategy = SMAStrategy(short_window, long_window)
    result = strategy.run(data)
    return (
        result.total_pnl,
        result.buy_indices,
        result.buy_prices,
        result.sell_indices,
        result.sell_prices
    )
