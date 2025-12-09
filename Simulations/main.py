import dearpygui.dearpygui as dpg
import numpy as np
import yfinance as yf
import pandas as pd
from funcs import brownian_prices, sma  # ensure this is available

# Cache data once
def getData():
    ticker = yf.Ticker("MSFT")
    df = pd.DataFrame(ticker.history(period="1y", interval="1h")).reset_index()
    df["Mid"] = (df["High"] + df["Low"]) / 2
    return np.array(df["Mid"])

data_cache = getData()

def run_simulation_callback(sender, app_data, user_data):
    input_steps_id = user_data["input_steps_id"]
    line_series_id = user_data["line_series_id"]
    y_axis_path = user_data["y_axis_path"]
    x_axis_path = user_data["x_axis_path"]
    profit_series_id = user_data["profit_series_id"]
    profit_y_axis = user_data["profit_y_axis"]
    profit_x_axis = user_data["profit_x_axis"]
    button_id = sender

    n_steps = dpg.get_value(input_steps_id)
    prices = brownian_prices(data_cache, n_steps)
    profit, buy_idx, buy_prices, sell_idx, sell_prices = sma(prices, short_window=3, long_window=9)

    dpg.set_value(line_series_id, [[], []])
    dpg.configure_item(button_id, enabled=False)

    # Center the view on the generated line with limits based on prices
    y_min = float(np.min(prices))
    y_max = float(np.max(prices))
    # Avoid zero-height range
    if y_min == y_max:
        eps = 1e-6
        y_min -= eps
        y_max += eps
    dpg.set_axis_limits(y_axis_path, y_min, y_max)
    # Set X-axis limits to cover the simulation steps
    dpg.set_axis_limits(x_axis_path, 0, max(1, n_steps - 1))

    # Prepare per-step profit (delta price) and set limits for profit chart

    sma_x = list(range(len(prices)))
    sma_y = np.zeros_like(prices)
    # Use positional mapping: buy_prices/sell_prices align with their respective indices
    for idx in buy_idx.astype(int):
        sma_y[idx] = prices[idx]*-1  # negative for buys
    for idx in sell_idx.astype(int):
        sma_y[idx] = prices[idx]
    sma_y = sma_y.cumsum().tolist()

    ry_min = float(np.min(sma_y))
    ry_max = float(np.max(sma_y))
    if ry_min == ry_max:
        eps = 1e-9
        ry_min -= eps
        ry_max += eps
    dpg.set_axis_limits(profit_y_axis, ymin=ry_min, ymax=ry_max)
    dpg.set_axis_limits(profit_x_axis, ymin=0, ymax=max(1, n_steps - 1))

    state = {
        "idx": 0,
        "x": [],
        "y": [],
        "prices": prices.tolist(),
        "px": [],
        "py": [],
        "returns": sma_y
    }

    batch = max(1, round(n_steps / 100))  # points per frame

    def tick_callback():
        # append up to `batch` points
        for _ in range(batch):
            i = state["idx"]
            if i >= len(state["prices"]):
                dpg.configure_item(button_id, enabled=True)
                return
            state["x"].append(i)
            state["y"].append(state["prices"][i])
            # profits (delta price) at the same step index
            state["px"].append(i)
            state["py"].append(state["returns"][i])
            state["idx"] += 1

        dpg.set_value(line_series_id, [state["x"], state["y"]])
        dpg.set_value(profit_series_id, [state["px"], state["py"]])

        # schedule next tick for the next frame
        dpg.set_frame_callback(dpg.get_frame_count() + 1, tick_callback)

    # kick off first tick next frame
    dpg.set_frame_callback(dpg.get_frame_count() + 1, tick_callback)

dpg.create_context()
dpg.create_viewport(title="Simulation", width=700, height=800)



with dpg.window(label="Simulation", width=700, height=800):
    dpg.add_text("Brownian Price Simulation")
    dpg.add_separator()

    # Input for steps
    input_steps_id = dpg.add_input_int(label="Steps", default_value=200, min_value=1, max_value=5000)

    # Initial data
    init_steps = dpg.get_value(input_steps_id)
    init_prices = brownian_prices(data_cache, init_steps)
    init_x = list(range(len(init_prices)))

    # Line plot
    plot_path = dpg.add_plot(label="Simulated Path", height=260, width=-1)
    x_axis_path = dpg.add_plot_axis(dpg.mvXAxis, label="Step", parent=plot_path)
    y_axis_path = dpg.add_plot_axis(dpg.mvYAxis, label="Price", parent=plot_path)
    line_series_id = dpg.add_line_series([], [], label="Path", parent=y_axis_path)

    # Disable auto-fit; we'll control limits
    dpg.configure_item(y_axis_path, auto_fit=False)
    
    # Profits bar chart directly below simulated path
    profit_plot = dpg.add_plot(label="Profit", height=300, width=-1)
    profit_x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Step", parent=profit_plot)
    profit_y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Profit", parent=profit_plot)
    profit_series_id = dpg.add_bar_series([], [], label="Profit", parent=profit_y_axis)

    # Button: pass IDs via user_data
    dpg.add_button(
        label="Run Simulation",
        width=150,
        callback=run_simulation_callback,
        user_data={
            "input_steps_id": input_steps_id,
            "line_series_id": line_series_id,
            "y_axis_path": y_axis_path,
            "x_axis_path": x_axis_path,
            "profit_series_id": profit_series_id,
            "profit_y_axis": profit_y_axis,
            "profit_x_axis": profit_x_axis,
        }
    )




with dpg.window(label="Historic", pos=(710, 0), width=900, height=800):
    dpg.add_text("Historic MSFT Mid Prices")
    dpg.add_separator()

    historic_plot_path = dpg.add_plot(label="Historic Mid Prices", height=700, width=-1)
    historic_x_axis_path = dpg.add_plot_axis(dpg.mvXAxis, label="Time", parent=historic_plot_path)
    historic_y_axis_path = dpg.add_plot_axis(dpg.mvYAxis, label="Mid Price", parent=historic_plot_path)

    historic_x = list(range(len(data_cache)))
    historic_y = data_cache.tolist()
    dpg.add_line_series(list(historic_x), historic_y, label="Mid Price", parent=historic_y_axis_path)

with dpg.window(label="SMA", pos=(700+900+10, 0), width=900, height=300):
    dpg.add_text("Simple Moving Average (SMA) Strategy on Historic MSFT Mid Prices")
    dpg.add_separator()

    sma_plot_path = dpg.add_plot(label="SMA Strategy", height=250, width=-1)
    sma_x_axis_path = dpg.add_plot_axis(dpg.mvXAxis, label="Time", parent=sma_plot_path)
    sma_y_axis_path = dpg.add_plot_axis(dpg.mvYAxis, label="Price", parent=sma_plot_path)

    profit, buy_idx, buy_prices, sell_idx, sell_prices = sma(data_cache, short_window=7, long_window=28)

    sma_x = list(range(len(data_cache)))
    # Build per-step realized PnL at sell steps using paired trades
    sma_steps = np.zeros_like(data_cache, dtype=float)
    pairs = min(len(buy_idx), len(sell_idx))
    for k in range(pairs):
        sell_i = int(sell_idx[k])
        pnl = float(sell_prices[k]) - float(buy_prices[k])
        sma_steps[sell_i] = pnl
    # Optional: account for an unmatched final buy as unrealized PnL at the end
    if len(buy_idx) > len(sell_idx):
        last_buy_price = float(buy_prices[-1])
        sma_steps[-1] += float(data_cache[-1]) - last_buy_price

    sma_cum = np.cumsum(sma_steps).tolist()

    dpg.set_axis_limits(sma_y_axis_path, min(sma_cum), max(sma_cum))
    # Set X-axis limits to cover the simulation steps
    dpg.set_axis_limits(sma_x_axis_path, 0, len(sma_x) - 1)

    dpg.add_bar_series(list(sma_x), sma_cum, label="SMA Cumulative PnL", parent=sma_y_axis_path)



dpg.setup_dearpygui()
dpg.show_viewport()
dpg.maximize_viewport()
dpg.start_dearpygui()
dpg.destroy_context()