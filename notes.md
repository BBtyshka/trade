Models to try:
~~ARIMA, SARIMA~~
Black-Scholes equation
    - Can be connected to a neural network as a parameter to trade on the difference between the theoretical and real value.Or it can be straight up used to trade. 
    - The equation is an equation that supposedly gives yo the theoretical perfect value of the option. 

Pair trading
    - When one stock goes up, a similar stock follows it and vice versa. If the stocks are mispriced, we can hedge both of them to extract a profit

Readme ex:
Conducted independent research into quantitative trading strategies, implementing and evaluating SMA Crossover, ARIMA/SARIMA/SARIMAX time-series models, pair trading (cointegration-based mean reversion), and LSTM neural networks.
Built a backtesting engine to simulate strategy execution on historical equity data with configurable stop-loss/take-profit thresholds, position tracking, and equity curve generation.
Applied ARIMA modelling with Box-Cox transformation and AIC-based hyperparameter tuning (grid search over p, d, q), achieving an RMSE of ~2.13 on AAPL 5-year close prices using rolling one-step-ahead forecasting.
Implemented a Monte Carlo simulation framework using geometric Brownian motion to generate 1,000+ synthetic price paths and evaluate strategy robustness across varying market conditions.
Computed risk-adjusted performance metrics — annualised Sharpe ratio, Sortino ratio, and maximum drawdown — to compare strategies and optimise SMA window parameters via grid search.
Explored alternative data for alpha generation by engineering features from meteorological data (temperature, precipitation, pressure via Meteostat API) and training an LSTM model (Keras) to predict price direction for weather-sensitive equities.
Investigated statistical arbitrage through pair trading, using cointegration tests and z-score thresholds on correlated stock pairs (AAPL/MSFT) to identify mean-reversion entry signals.
Tools: Python, Pandas, NumPy, statsmodels, scikit-learn, Keras/TensorFlow, Matplotlib, yfinance, Meteostat


Improvements: 
Add a README.md — This is the single highest-impact improvement. Include:

One-line description, architecture diagram, how to run, sample output/screenshots
Recruiters and hiring managers look at the README first; right now there isn't one.
Add a requirements.txt or pyproject.toml — Makes the project reproducible and signals professionalism.

Include a screenshot of the Dear PyGui dashboard — A visual in the README instantly communicates the scope of the project.

Medium effort (strengthen the quant story)
Add more performance metrics — You already have Sharpe and Sortino. Adding Calmar ratio, win rate, profit factor, and average trade duration would round out a proper tearsheet.

Produce a strategy comparison table or chart — A single summary DataFrame or plot comparing all strategies side by side (Sharpe, drawdown, total return) is the kind of output quant interviewers love to see.

Add walk-forward / out-of-sample validation — Your ARIMA notebook does rolling refit, but explicitly splitting into train/validation/test and reporting out-of-sample metrics would demonstrate rigour.

Write unit tests — Even a handful of tests for Statistics, BacktestResult, and SMAStrategy show software engineering maturity and make the project more credible.

Ambitious extensions (strong differentiators)
Implement a factor model — Even a simple Fama-French 3-factor regression on your returns would elevate this from a "strategies project" to a "quant research project".

Add transaction costs and slippage — Your backtester currently assumes zero-cost execution. Adding a configurable commission/slippage model makes the results more realistic and is a common interview talking point.

Expand pair trading with proper Engle-Granger or Johansen cointegration — Your current implementation uses sm.tsa.stattools.coint but doesn't act on the results programmatically. Wiring it into the engine as a full strategy would complete the loop.

Deploy a live paper-trading dashboard — Connecting to a paper trading API (e.g., Alpaca) and running strategies in real time, even on a free tier, is a major differentiator at the student level.

