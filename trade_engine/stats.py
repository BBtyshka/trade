from typing import Dict, Any
import numpy as np
from core import BacktestResult

class Statistics:
    """Calculate performance statistics for backtest results.
    
    Usage:
        stats = Statistics(result)
        print(stats.sharpe_ratio())
        print(stats.max_drawdown())
    """
    
    def __init__(self, result: BacktestResult):
        self.result = result
    
    def sharpe_ratio(self, risk_free_rate: float = 0.0, trading_days: int = 252) -> float:
        """Calculate daily Sharpe ratio.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 0)
            trading_days: Number of trading days per year (default 252)
        
        Returns:
            Sharpe ratio (annualized)
        """
        # Formula: (mean_return - daily_rf) / std_return * sqrt(trading_days)
        equity = self.result.equity_curve.dropna()
        
        if len(equity) < 2:
            return 0.0
        
        # Calculate returns, replacing inf with NaN, then dropping
        returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(returns) == 0:
            return 0.0
        
        daily_rf = risk_free_rate / trading_days
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0 or np.isnan(std_return):
            return 0.0
        
        # Annualized Sharpe ratio
        sharpe = (mean_return - daily_rf) / std_return * (trading_days ** 0.5)
        return 0.0 if np.isnan(sharpe) else sharpe
    
    def sortino_ratio(self, risk_free_rate: float = 0.0, trading_days: int = 252) -> float:
        """Calculate Sortino ratio (only penalizes downside volatility).
        
        Args:
            risk_free_rate: Annual risk-free rate (default 0)
            trading_days: Number of trading days per year (default 252)
        
        Returns:
            Sortino ratio (annualized)
        """
        equity = self.result.equity_curve.dropna()
        
        if len(equity) < 2:
            return 0.0
        
        # Calculate returns, replacing inf with NaN, then dropping
        returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

        if len(returns) == 0:
            return 0.0
        
        daily_rf = risk_free_rate / trading_days
        excess_returns = returns - daily_rf
        mean_excess = excess_returns.mean()
        
        # Downside deviation: sqrt of mean of squared negative excess returns
        # Use min(return, 0)^2 for ALL periods, not just negative ones
        downside_returns = np.minimum(excess_returns, 0)
        downside_std = np.sqrt((downside_returns ** 2).mean())
        
        if downside_std == 0 or np.isnan(downside_std):
            return 0.0
        
        # Annualized Sortino ratio
        sortino = (mean_excess / downside_std) * (trading_days ** 0.5)
         
        return 0.0 if np.isnan(sortino) else sortino
    
    def max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve.
        
        Returns:
            Maximum drawdown as a percentage (positive number, e.g., 0.15 = 15% drawdown)
        """
        equity = self.result.equity_curve.dropna()
        
        if len(equity) < 2:
            return 0.0
        
        # Running maximum (peak at each point)
        running_max = equity.cummax()
        
        # Drawdown at each point: (peak - current) / peak
        drawdown = (running_max - equity) / running_max
        
        # Replace inf/nan with 0 (in case of division by zero)
        drawdown = drawdown.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Return maximum drawdown
        return drawdown.max()
    
    def calmar_ratio(self) -> float:
        """Calculate Calmar ratio (return / max drawdown).
        
        Returns:
            Calmar ratio
        """
        # TODO: Implement Calmar ratio calculation
        raise NotImplementedError("Calmar ratio calculation not implemented yet")
    
    def profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss).
        
        Returns:
            Profit factor
        """
        # TODO: Implement profit factor calculation
        raise NotImplementedError("Profit factor calculation not implemented yet")
    
    def avg_trade_pnl(self) -> float:
        """Calculate average PnL per trade.
        
        Returns:
            Average trade PnL
        """
        # TODO: Implement average trade PnL calculation
        raise NotImplementedError("Average trade PnL calculation not implemented yet")
    
    def total_return(self) -> float:
        """Calculate total return as percentage.
        
        Returns:
            Total return percentage
        """
        # TODO: Implement total return calculation
        raise NotImplementedError("Total return calculation not implemented yet")
    
    def volatility(self, annualize: bool = True) -> float:
        """Calculate return volatility.
        
        Args:
            annualize: Whether to annualize the volatility
        
        Returns:
            Volatility (standard deviation of returns)
        """
        # TODO: Implement volatility calculation
        raise NotImplementedError("Volatility calculation not implemented yet")
    
    def summary(self) -> Dict[str, Any]:
        """Get a summary of all statistics.
        
        Returns:
            Dictionary with all computed statistics
        """
        # TODO: Call all statistics methods and return as dict
        return {
            "ticker": self.result.ticker,
            "strategy": self.result.strategy_name,
            "total_pnl": self.result.total_pnl,
            "realized_pnl": self.result.realized_pnl,
            "unrealized_pnl": self.result.unrealized_pnl,
            "num_trades": self.result.num_trades,
            "win_rate": self.result.win_rate,
        }
