"""
Example usage of the Trading Engine.

Run backtests on historical data with multiple tickers and strategies.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import Engine, Statistics, backtest
from str.SMA import SMAStrategy
from str.ARIMA import ARIMAStrategy


def main():
    # ========================================
    # Option 1: Using the Engine class
    # ========================================
    
    print("=" * 60)
    print("TRADING ENGINE - BACKTEST EXAMPLE")
    print("=" * 60)
    
    capital = 10000.0  # Starting capital for backtests

    engine = Engine(
        period="1y", 
        interval="1d",
        stop_loss_pct=0.05,   # 5% stop-loss
        take_profit_pct=0.10,# 10% take-profit
        capital=capital  
    )
    
    # Add tickers to analyze
    #engine.add_tickers("AAPL", "MSFT", "GOOGL", "AMZN")
    tickers = ["AAPL","MSFT","NVDA", "TSLA"]  # Add more tickers as needed
    engine.add_tickers(*tickers)
    
    # Add strategies
    engine.add_strategy(SMAStrategy(short_window=10, long_window=30), name="SMA_10_30")
    engine.add_strategy(SMAStrategy(short_window=5, long_window=20), name="SMA_5_20")
    engine.add_strategy(SMAStrategy(short_window=1, long_window=2), name="SMA_1_2")
    engine.add_strategy(SMAStrategy(short_window=1, long_window=3), name="SMA_1_3")
    engine.add_strategy(SMAStrategy(short_window=2, long_window=5), name="SMA_2_5")
    
    #engine.add_strategy(ARIMAStrategy(order=(3,2,0), train_points=100), name="ARIMA_3_2_0")
    
    strat_names = ["SMA_2_5", "SMA_1_2", "SMA_1_3", "SMA_5_20", "SMA_10_30"]  # Add "ARIMA_3_2_0" if ARIMA is used

    # Run backtests
    print("\nRunning backtests...")
    results = engine.run()
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(engine.summary().to_string(index=False))
    print("Starting Capital: ${:.2f} - ending capital: ${:.2f} - total profit: ${:.2f}".format(capital, engine.capital, engine.capital - capital))
    
    # Get detailed stats for a specific ticker/strategy
    print("\n" + "=" * 60)
    print("DETAILED STATS: AAPL - SMA_10_30")
    print("=" * 60)
    
    for ticker in tickers:
        for strat in strat_names:
            stats = engine.get_statistics(ticker, strat)
            if stats:
                if engine.get_result(ticker, strat) is not None:
                    print(f"\nStatistics for {ticker} - {strat}:")
                    print(f"Sharpe ratio for {ticker} - {strat}: {stats.sharpe_ratio():.4f}")
                    print(f"Sortino ratio for {ticker} - {strat}: {stats.sortino_ratio():.4f}")
                    print(f"Max Drawdown for {ticker} - {strat}: {stats.max_drawdown():.4f}")

if __name__ == "__main__":
    main()
