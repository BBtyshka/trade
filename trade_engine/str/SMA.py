import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict, List
import numpy as np
import pandas as pd
from core import Signal, Side, Strategy


class SMAStrategy(Strategy):
    """Simple Moving Average Crossover Strategy.
    
    Generates BUY signals when short SMA crosses above long SMA (with upward momentum).
    Generates SELL signals when short SMA crosses below long SMA (with downward momentum).
    """
    
    def __init__(self, short_window: int = 10, long_window: int = 30):
        super().__init__(name="SMA")
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, prices: pd.Series) -> List[Signal]:
        """Generate trading signals from price series."""
        # Calculate SMAs using pandas rolling
        short_sma = prices.rolling(window=self.short_window).mean()
        long_sma = prices.rolling(window=self.long_window).mean()
        
        signals = []
        
        # Start after we have enough data for long SMA
        for idx in range(self.long_window + 1, len(prices)):
            timestamp = prices.index[idx]
            
            # Get current and previous values
            curr_short = short_sma.iloc[idx]
            prev_short = short_sma.iloc[idx - 1]
            curr_long = long_sma.iloc[idx]
            prev_long = long_sma.iloc[idx - 1]
            
            # Skip if any NaN
            if pd.isna(curr_short) or pd.isna(prev_short) or pd.isna(curr_long) or pd.isna(prev_long):
                continue
            
            # Calculate momentum
            momentum = curr_long - prev_long
            
            # Detect crossovers
            crossed_up = prev_short < prev_long and curr_short >= curr_long and momentum > 0
            crossed_down = prev_short > prev_long and curr_short <= curr_long and momentum < 0
            
            if crossed_up:
                signals.append(Signal(
                    timestamp=timestamp.to_pydatetime(),
                    side=Side.BUY,
                    price=prices.iloc[idx],
                    quantity=1*10*momentum
                ))
            elif crossed_down:
                signals.append(Signal(
                    timestamp=timestamp.to_pydatetime(),
                    side=Side.SELL,
                    price=prices.iloc[idx],
                    quantity=1*10*momentum
                ))
        
        return signals
    
    def get_params(self) -> Dict[str, Any]:
        return {"short_window": self.short_window, "long_window": self.long_window}