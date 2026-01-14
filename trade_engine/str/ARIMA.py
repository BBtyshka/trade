##ARIMA strategy 

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
from typing import Any, Dict, List
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
from core import Side, Signal, Strategy


class ARIMAStrategy(Strategy):
    def __init__(self, order=(5, 1, 0), train_points=100):
        super().__init__(name="ARIMA")
        self.order = order
        self.fitted_model = None
        self.train_points = train_points

    def fit(self, data):
        """
        Fit the ARIMA model to the provided time series data.
        
        Parameters:
        data (array-like): The time series data to fit the model on.
        """
        # Use .values to avoid date index warnings
        values = data.values if hasattr(data, 'values') else data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = ARIMA(values, order=self.order)
            self.fitted_model = self.model.fit()

    def predict(self, steps=1):
        """
        Predict future values using the fitted ARIMA model.
        
        Parameters:
        steps (int): The number of future time steps to predict.
        
        Returns:
        array-like: Predicted values for the specified number of steps.
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction.")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self.fitted_model.forecast(steps=steps)
        return forecast
    
    def generate_signals(self, prices: pd.Series) -> List[Signal]:
        """
        Generate trading signals based on ARIMA model predictions.
        
        Parameters:
        prices (pd.Series): The price series to generate signals from.
        
        Returns:
        List[Signal]: A list of generated trading signals.
        """
        if len(prices) < self.train_points:
            raise ValueError("Not enough data points to fit the model.")
        
        signals = []
        
        for idx in range(self.train_points, len(prices)):
            self.fit(prices[idx - self.train_points:idx])       
            predicted_price = self.predict(steps=5)[-1]
            current_price = prices.iloc[idx]
            last_price = prices.iloc[idx - 1]
            if idx % 50 == 0:
                print(f"ARIMA step {idx}")
            dif = abs(predicted_price - last_price) / last_price
            if dif < 0.001:  # Skip insignificant changes
                continue
            
            if predicted_price > last_price:
                signals.append(Signal(
                    timestamp=prices.index[idx].to_pydatetime(),
                    side=Side.BUY,
                    price=current_price,
                    quantity=1 * 10 * dif
                ))
            elif predicted_price < last_price:
                signals.append(Signal(
                    timestamp=prices.index[idx].to_pydatetime(),
                    side=Side.SELL,
                    price=current_price,
                    quantity=1 * 10 * dif
                ))
        
        return signals

    def get_params(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return {"order": self.order, "train_points": self.train_points}