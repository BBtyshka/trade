import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import List, Optional, Dict
import pandas as pd

from core import BacktestResult, Strategy, Signal, Trade, Side
from draw import plot_prices_with_trades
from stats import Statistics
from data_loader import DataLoader

# ============================================================================
# ENGINE
# ============================================================================

class Engine:
    """Main backtesting engine with stop-loss and take-profit support.
    
    Orchestrates data loading, strategy execution, and result aggregation.
    Supports multiple tickers and strategies.
    
    The Engine processes signals from strategies and manages:
    - Position tracking (one position at a time per ticker/strategy)
    - Stop-loss execution
    - Take-profit execution
    
    Usage:
        engine = Engine(stop_loss_pct=0.02, take_profit_pct=0.05)
        engine.add_tickers("AAPL", "MSFT", "GOOGL")
        engine.add_strategy(SMAStrategy(short_window=10, long_window=30))
        
        results = engine.run()
        print(engine.summary())
    """
    
    def __init__(
        self, 
        period: str = "5y", 
        interval: str = "1d",
        stop_loss: Optional[float] = None,      # Absolute $ stop-loss
        take_profit: Optional[float] = None,    # Absolute $ take-profit
        stop_loss_pct: Optional[float] = None,  # Percentage stop-loss (0.02 = 2%)
        take_profit_pct: Optional[float] = None, # Percentage take-profit (0.05 = 5%)
        capital: float = 10000.0
    ):
        """
        Args:
            period: Data period for Yahoo Finance
            interval: Data interval for Yahoo Finance
            stop_loss: Absolute dollar amount for stop-loss
            take_profit: Absolute dollar amount for take-profit
            stop_loss_pct: Percentage for stop-loss (0.02 = 2%)
            take_profit_pct: Percentage for take-profit (0.05 = 5%)
        """
        self.data_loader = DataLoader(period=period, interval=interval)
        self._tickers: List[str] = []
        self._strategies: Dict[str, Strategy] = {}
        self._results: Dict[str, Dict[str, BacktestResult]] = {}
        
        # Stop-loss and take-profit settings
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self.capital = capital
    
    # ---- Ticker Management ----
    
    def add_tickers(self, *tickers: str) -> "Engine":
        """Add one or more tickers to backtest."""
        for ticker in tickers:
            if ticker not in self._tickers:
                self._tickers.append(ticker.upper())
        return self
    
    def remove_ticker(self, ticker: str) -> "Engine":
        """Remove a ticker."""
        ticker = ticker.upper()
        if ticker in self._tickers:
            self._tickers.remove(ticker)
        return self
    
    def clear_tickers(self) -> "Engine":
        """Remove all tickers."""
        self._tickers.clear()
        return self
    
    @property
    def tickers(self) -> List[str]:
        """Get list of tickers."""
        return self._tickers.copy()
    
    # ---- Strategy Management ----
    
    def add_strategy(self, strategy: Strategy, name: Optional[str] = None) -> "Engine":
        """Add a strategy."""
        key = name or strategy.name
        self._strategies[key] = strategy
        return self
    
    def remove_strategy(self, name: str) -> "Engine":
        """Remove a strategy by name."""
        if name in self._strategies:
            del self._strategies[name]
        return self
    
    def get_strategy(self, name: str) -> Optional[Strategy]:
        """Get a strategy by name."""
        return self._strategies.get(name)
    
    @property
    def strategies(self) -> List[str]:
        """Get list of strategy names."""
        return list(self._strategies.keys())
    
    # --- Draw ---
    def draw_results(self, results):
        plot_prices_with_trades(results)
    
    # ---- Execution ----
    
    def _process_signals(self, signals: List[Signal], prices: pd.Series) -> List[Trade]:
        """Process signals into trades, applying stop-loss and take-profit.
        
        Logic:
        - BUY signal opens a long position (if not already in one)
        - SELL signal closes the position
        - At each bar, check if SL/TP is hit and close if so
        
        Args:
            signals: List of signals from strategy
            prices: Price series with datetime index
            
        Returns:
            List of completed Trade objects
        """
        trades: List[Trade] = []
        current_position: Optional[Trade] = None
        
        # Create a dict of signals indexed by timestamp for O(1) lookup
        signal_map: Dict[pd.Timestamp, Signal] = {pd.Timestamp(s.timestamp): s for s in signals}
        
        for timestamp, price in prices.items():
            # Cast timestamp to pd.Timestamp (pandas index items are Timestamps)
            timestamp_key: pd.Timestamp = pd.Timestamp(timestamp)  # type: ignore[arg-type]
            
            # Check if we have an open position
            if current_position is not None:
                entry_price = current_position.entry_price
                
                # Calculate SL/TP levels
                sl_price = None
                tp_price = None
                
                if self.stop_loss_pct is not None:
                    sl_price = entry_price * (1 - self.stop_loss_pct)
                elif self.stop_loss is not None:
                    sl_price = entry_price - self.stop_loss
                    
                if self.take_profit_pct is not None:
                    tp_price = entry_price * (1 + self.take_profit_pct)
                elif self.take_profit is not None:
                    tp_price = entry_price + self.take_profit
                
                # Check stop-loss
                if sl_price is not None and price <= sl_price:
                    closed_trade = Trade(
                        entry_time=current_position.entry_time,
                        exit_time=timestamp_key.to_pydatetime(),
                        entry_price=entry_price,
                        exit_price=price,
                        quantity=current_position.quantity,
                        exit_reason="stop_loss"
                    )
                    trades.append(closed_trade)
                    current_position = None
                    continue
                
                # Check take-profit
                if tp_price is not None and price >= tp_price:
                    closed_trade = Trade(
                        entry_time=current_position.entry_time,
                        exit_time=timestamp_key.to_pydatetime(),
                        entry_price=entry_price,
                        exit_price=price,
                        quantity=current_position.quantity,
                        exit_reason="take_profit"
                    )
                    trades.append(closed_trade)
                    current_position = None
                    continue
            
            # Check for signal at this timestamp
            if timestamp_key in signal_map:
                signal = signal_map[timestamp_key]
                
                if signal.side == Side.BUY and current_position is None:
                    # Open a new position
                    current_position = Trade(
                        entry_time=timestamp_key.to_pydatetime(),
                        exit_time=None,
                        entry_price=signal.price,
                        exit_price=None,
                        quantity=signal.quantity
                    )
                    
                elif signal.side == Side.SELL and current_position is not None:
                    # Close the position via signal
                    closed_trade = Trade(
                        entry_time=current_position.entry_time,
                        exit_time=timestamp_key.to_pydatetime(),
                        entry_price=current_position.entry_price,
                        exit_price=signal.price,
                        quantity=current_position.quantity,
                        exit_reason="signal"
                    )
                    trades.append(closed_trade)
                    current_position = None
        
        # If position still open at end, add as open trade
        if current_position is not None:
            trades.append(current_position)
        
        return trades
    
    def run(self) -> Dict[str, Dict[str, BacktestResult]]:
        """Run all strategies on all tickers.
        
        Returns:
            Nested dict: {ticker: {strategy_name: BacktestResult}}
        """
        if not self._tickers:
            raise ValueError("No tickers configured. Use add_tickers() first.")
        if not self._strategies:
            raise ValueError("No strategies configured. Use add_strategy() first.")
        
        self._results.clear()
        
        for ticker in self._tickers:
            prices = self.data_loader.load_prices(ticker)
            self._results[ticker] = {}
            
            for name, strategy in self._strategies.items():
                # Get signals from strategy
                signals = strategy.generate_signals(prices)
                
                # Process signals into trades with SL/TP
                trades = self._process_signals(signals, prices)
                
                # Build result
                result = BacktestResult(
                    ticker=ticker,
                    strategy_name=name,
                    trades=trades,
                    signals=signals,
                    prices=prices
                )
                self.capital += result.total_pnl
                result.build_equity_curve()
                self._results[ticker][name] = result
        
        self.draw_results(self._results)
        
        return self._results
    
    def run_single(self, ticker: str, strategy_name: str) -> BacktestResult:
        """Run a single strategy on a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            strategy_name: Name of strategy to run
        
        Returns:
            BacktestResult for the run
        """
        if strategy_name not in self._strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found.")
        
        prices = self.data_loader.load_prices(ticker.upper())
        strategy = self._strategies[strategy_name]
        
        # Get signals from strategy
        signals = strategy.generate_signals(prices)
        
        # Process signals into trades with SL/TP
        trades = self._process_signals(signals, prices)
        
        # Build result
        result = BacktestResult(
            ticker=ticker.upper(),
            strategy_name=strategy_name,
            trades=trades,
            signals=signals,
            prices=prices
        )
        result.build_equity_curve()
        
        return result
    
    # ---- Results ----
    
    def get_result(self, ticker: str, strategy_name: str) -> Optional[BacktestResult]:
        """Get a specific result."""
        return self._results.get(ticker.upper(), {}).get(strategy_name)
    
    def get_statistics(self, ticker: str, strategy_name: str) -> Optional[Statistics]:
        """Get statistics for a specific result."""
        result = self.get_result(ticker, strategy_name)
        if result:
            return Statistics(result)
        return None
    
    def summary(self) -> pd.DataFrame:
        """Get a summary DataFrame of all results."""
        rows = []
        for ticker, strategies in self._results.items():
            for strategy_name, result in strategies.items():
                rows.append({
                    "Ticker": ticker,
                    "Strategy": strategy_name,
                    "Total PnL": f"${result.total_pnl:.2f}",
                    "Realized PnL": f"${result.realized_pnl:.2f}",
                    "Trades": result.num_trades,
                    "Win Rate": f"{result.win_rate:.1%}",
                    "Stop-Loss Hits": sum(1 for t in result.closed_trades if t.exit_reason == "stop_loss"),
                    "Take-Profit Hits": sum(1 for t in result.closed_trades if t.exit_reason == "take_profit"),
                })
        return pd.DataFrame(rows)
    
    # ---- Utility ----
    
    def reset(self) -> "Engine":
        """Reset results and data cache."""
        self._results.clear()
        self.data_loader.clear_cache()
        return self
    
    def __repr__(self) -> str:
        return f"Engine(tickers={self._tickers}, strategies={self.strategies})"


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def backtest(
    tickers: List[str],
    strategy: Strategy,
    period: str = "1y",
    interval: str = "1d"
) -> Dict[str, BacktestResult]:
    """Quick backtest function.
    
    Args:
        tickers: List of ticker symbols
        strategy: Strategy instance to run
        period: Data period
        interval: Data interval
    
    Returns:
        Dict mapping ticker to BacktestResult
    
    Example:
        results = backtest(["AAPL", "MSFT"], SMAStrategy(10, 30))
        for ticker, result in results.items():
            print(f"{ticker}: ${result.total_pnl:.2f}")
    """
    engine = Engine(period=period, interval=interval)
    engine.add_tickers(*tickers)
    engine.add_strategy(strategy)
    
    all_results = engine.run()
    return {ticker: results[strategy.name] for ticker, results in all_results.items()}
