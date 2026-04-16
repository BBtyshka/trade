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
from str.sg_vs_hk import EMA_Bollinger


def main():
    print("=" * 60)
    print("TRADING ENGINE - BACKTEST EXAMPLE")
    print("=" * 60)
    
    capital = 10000.0  # Starting capital for backtests

    engine = Engine(
        period="1y", 
        interval="1h",
        stop_loss_pct=0.05, 
        take_profit_pct=0.10,
        capital=capital  
    )
    
    # Add tickers to analyze
    #engine.add_tickers("AAPL", "MSFT", "GOOGL", "AMZN")
    tickers = ["BTC-USD", 'ETH-USD', 'SOL-USD']  # Add more tickers as needed
    engine.add_tickers(*tickers)
    
    # Add strategies

    engine.add_strategy(EMA_Bollinger(short_window=10, long_window=30), name="EMA_Bollinger_10_30")
    engine.add_strategy(EMA_Bollinger(short_window=5, long_window=20), name="EMA_Bollinger_5_20")
    engine.add_strategy(EMA_Bollinger(short_window=2, long_window=5), name="EMA_Bollinger_2_5")

    engine.add_strategy(SMAStrategy(short_window=7, long_window=20), name="SMA_7_20")
    engine.add_strategy(SMAStrategy(short_window=5, long_window=16), name="SMA_5_16")
    #engine.add_strategy(ARIMAStrategy(order=(3,2,0), train_points=1000), name="ARIMA_3_2_0")
    #engine.add_strategy(ARIMAStrategy(order=(3,2,0), train_points=100), name="ARIMA_3_2_0")
    
    strat_names = ["ARIMA_3_2_0"]  # Add "ARIMA_3_2_0" if ARIMA is used

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
    print("=" * 60)
    
    for ticker in tickers:
        for strat in strat_names:
            stats = engine.get_statistics(ticker, strat)
            if stats:
                if engine.get_result(ticker, strat) is not None and stats.sharpe_ratio()>=0:
                    print(f"\nStatistics for {ticker} - {strat}:")
                    print(f"Sharpe ratio for {ticker} - {strat}: {stats.sharpe_ratio():.4f}")
                    print(f"Sortino ratio for {ticker} - {strat}: {stats.sortino_ratio():.4f}")
                    print(f"Max Drawdown for {ticker} - {strat}: {stats.max_drawdown():.4f}")

if __name__ == "__main__":
    main()
