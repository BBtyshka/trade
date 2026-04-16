from typing import List, Dict
import numpy as np
import pandas as pd
import yfinance as yf

class DataLoader:
    """Fetches historical price data from Yahoo Finance."""
    
    def __init__(self, period: str = "1y", interval: str = "1d"):
        """
        Args:
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        """
        self.period = period
        self.interval = interval
        self._cache: Dict[str, pd.DataFrame] = {}
    
    def load(self, ticker: str) -> pd.DataFrame:
        """Load OHLCV data for a ticker."""
        if ticker not in self._cache:
            yf_ticker = yf.Ticker(ticker)
            df = yf_ticker.history(period=self.period, interval=self.interval)
            df["Mid"] = (df["High"] + df["Low"]) / 2
            self._cache[ticker] = df
        return self._cache[ticker]
    
    def load_prices(self, ticker: str) -> pd.DataFrame:
        """Load mid prices for a ticker as pandas DataFrame with datetime index."""
        df = self.load(ticker)
        return df
    
    def load_multiple(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """Load data for multiple tickers."""
        return {ticker: self.load(ticker) for ticker in tickers}
    
    def clear_cache(self):
        """Clear the data cache."""
        self._cache.clear()
