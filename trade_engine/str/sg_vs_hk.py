import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict, List
import pandas as pd
from core import Signal, Side, Strategy

def _calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR)."""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/window, adjust=False).mean()


class EMA_Bollinger(Strategy):
    """
    Exponential Moving Average Crossover Strategy with Bollinger Bands, ATR stop-loss, 
    and risk-based position sizing.
    """
    
    def __init__(self, short_window: int = 10, long_window: int = 30, atr_window: int = 14, atr_multiplier: float = 2.0, risk_per_trade: float = 0.01, ):
        super().__init__(name="EMA_Bollinger_ATR_Stop")
        self.short_window = short_window
        self.long_window = long_window
        self.risk_window = 2 * long_window
        self.atr_window = atr_window
        self.atr_multiplier = atr_multiplier
        self.risk_per_trade = risk_per_trade  # e.g., 0.01 for 1% of portfolio
        self.upper_band = None
        self.lower_band = None

    def generate_signals(self, prices: pd.DataFrame, balance: float) -> List[Signal]:
        """Generate trading signals from a DataFrame with High, Low, and Close prices."""
        close_prices = prices['Close']

        # --- Indicator Calculations ---
        short_ema = close_prices.ewm(span=self.short_window, adjust=False).mean()
        long_ema = close_prices.ewm(span=self.long_window, adjust=False).mean()
        risk_ema = close_prices.ewm(span=self.risk_window, adjust=False).mean()
        atr = _calculate_atr(prices['High'], prices['Low'], prices['Close'], self.atr_window)
        
        signals = []
        
        for idx in range(self.risk_window + 1, len(close_prices)):
            timestamp = close_prices.index[idx]
            
            # --- Indicator Values ---
            curr_short = short_ema.iloc[idx]
            prev_short = short_ema.iloc[idx - 1]
            curr_long = long_ema.iloc[idx]
            prev_long = long_ema.iloc[idx - 1]
            
            if pd.isna(curr_short) or pd.isna(prev_short) or pd.isna(curr_long) or pd.isna(prev_long) or pd.isna(atr.iloc[idx]):
                continue

            # --- Bollinger Bands ---
            rolling_std = close_prices.iloc[idx - self.long_window:idx].std()
            self.upper_band = long_ema.iloc[idx] + (2 * rolling_std)
            self.lower_band = long_ema.iloc[idx] - (2 * rolling_std)
            
            # --- Crossover Logic ---
            momentum = curr_long - prev_long
            crossed_up = prev_short < prev_long and curr_short >= curr_long and momentum > 0
            crossed_down = prev_short > prev_long and curr_short <= curr_long and momentum < 0
            
            # --- Stop-Loss and Sizing ---
            stop_loss_distance = atr.iloc[idx] * self.atr_multiplier
            current_price = close_prices.iloc[idx]

            # --- Entry Conditions ---
            if crossed_up and curr_short > risk_ema.iloc[idx] and current_price > self.lower_band:
                stop_loss = current_price - stop_loss_distance
                # Ensure stop-loss is not negative and there's a valid distance for sizing
                if stop_loss > 0 and (current_price - stop_loss) > 0:
                    # Position size based on fixed risk per trade
                    quantity = (self.risk_per_trade) / (current_price - stop_loss) * balance*0.01
                    signals.append(Signal(
                        timestamp=timestamp.to_pydatetime(),
                        side=Side.BUY,
                        price=current_price,
                        quantity=quantity,
                        stop_loss=stop_loss
                    ))

            elif crossed_down and curr_short < risk_ema.iloc[idx] and current_price < self.upper_band:
                stop_loss = current_price + stop_loss_distance
                if (stop_loss - current_price) > 0:
                    # Position size based on fixed risk per trade
                    quantity = (self.risk_per_trade) / (stop_loss - current_price) * balance*0.01
                    signals.append(Signal(
                        timestamp=timestamp.to_pydatetime(),
                        side=Side.SELL,
                        price=current_price,
                        quantity=quantity,
                        stop_loss=stop_loss
                    ))
        
        return signals
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "short_window": self.short_window, 
            "long_window": self.long_window, 
            "risk_window": self.risk_window,
            "atr_window": self.atr_window,
            "atr_multiplier": self.atr_multiplier,
            "risk_per_trade": self.risk_per_trade
        }
