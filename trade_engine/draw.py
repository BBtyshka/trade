import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import math

def plot_prices_with_trades(results):
    total_plots = sum(len(strategies) for strategies in results.values())
    
    if total_plots == 0:
        print("No results to plot")
        return
    
    rows = math.ceil(math.sqrt(total_plots))
    cols = math.ceil(total_plots / rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10), squeeze=False)
    axes = axes.flatten() 
    
    i = 0
    for ticker in results:
        for name in results[ticker]:
            result = results[ticker][name]
            equity_curve = result.equity_curve
            
            # Plot with datetime index
            axes[i].plot(equity_curve.index, equity_curve.values, label=f'{ticker} - {name}')
            axes[i].set_title(f'{ticker} - {name}')
            axes[i].legend()
            
            # Format x-axis for dates
            axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            axes[i].xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            i += 1
    
    for j in range(i, len(axes)):
        axes[j].set_visible(False)
            
    plt.tight_layout()
    plt.show()