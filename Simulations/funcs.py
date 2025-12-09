import numpy as np
import yfinance as yf
import pandas as pd


def brownian_prices(data:np.ndarray, steps:int):
    r = np.diff(np.log(data))
    mu = np.mean(r)
    sd = np.std(r)

    last = data[-1]

    norm = np.random.normal(mu, sd, steps)
    sim = last * np.exp(np.cumsum(norm))
    return sim


def sma(data:np.ndarray, short_window:int, long_window:int):
    "Returns: profit, buy_prices, sell_prices"
    LONG = long_window
    SHORT = short_window

    val = data

    s_averages = np.array([])
    l_averages = np.array([])

    buy_prices = np.array([])
    sell_prices = np.array([])

    buy_idx = np.array([])
    sell_idx = np.array([])

    position = 0
    entry_price = None

    for i in range(len(val)):
        if i < LONG:
            continue

        long_sma = np.mean(val[i-LONG:i])
        l_averages = np.append(l_averages, long_sma)
        short_sma = np.mean(val[i-SHORT:i])
        s_averages = np.append(s_averages, short_sma)

        if len(l_averages) < 2:
            continue
        l_diff = l_averages[-1] - l_averages[-2]

        crossed_up = (s_averages[-2] < l_averages[-2]) and (s_averages[-1] >= l_averages[-1]) and (l_diff > 0)
        crossed_dn = (s_averages[-2] > l_averages[-2]) and (s_averages[-1] <= l_averages[-1]) and (l_diff < 0)

        if crossed_up: #and position == 0:
            buy_prices = np.append(buy_prices, val[i])
            buy_idx = np.append(buy_idx, i)
            #position = 1
            entry_price = val[i]
        elif crossed_dn: #and position == 1:
            sell_prices = np.append(sell_prices, val[i])
            sell_idx = np.append(sell_idx, i)
            #position = 0
            entry_price = None

    realized_pnl = np.sum(sell_prices) - np.sum(buy_prices)
    #open_pnl = (val[-1] - entry_price) if position == 1 and entry_price is not None else 0.0
    profit = realized_pnl #+ open_pnl
    return profit, buy_idx, buy_prices, sell_idx, sell_prices