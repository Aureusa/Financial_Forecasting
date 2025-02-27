import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pandas_ta as ta
import math


class Plotter:
    def __init__(self, closing_prices, predicted_closing_prices, dates):
        self._closing_prices = closing_prices
        self._predicted_closing_prices = predicted_closing_prices
        self._dates = dates

    def comparison_plot(self, predictions_std: list[float]|None, bollinger_band: bool, stock_name: str, save: bool):
        fig, ax = plt.subplots()
        
        if bollinger_band:
            mid_band = self._calculate_sma(self._closing_prices, 20)
            twenty_day_std = pd.Series(self._closing_prices).rolling(window=20).std().dropna().tolist()

            upper_band = np.array(mid_band) + 2 * np.array(twenty_day_std)
            lower_band = np.array(mid_band) - 2 * np.array(twenty_day_std)

            bolinger_x = self._dates[19:]

            ax.plot(bolinger_x, mid_band, color="g")
            ax.plot(bolinger_x, upper_band, color="b")
            ax.plot(bolinger_x, lower_band, color="b")

        # Plot the line
        ax.plot(self._dates, self._closing_prices, color="blue")

        # Add square markers for data points
        ax.scatter(self._dates, self._closing_prices, color="blue", marker="s", label="Actual")

            
        # Plot the line
        ax.plot(self._dates, self._predicted_closing_prices, color="red")

        # Add square markers for data points
        ax.scatter(self._dates, self._predicted_closing_prices, color="red", marker="^", label="Predicted")

        # Labels and title
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        ax.set_title(f"Actual vs Predicted Closing Prices ({stock_name})")

        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

        # Legend
        ax.legend()

        if save:
            plt.savefig(f"{stock_name}.png")
            plt.close(fig)
        else:
            plt.show()

    def _calculate_sma(
            self,
            close: list[float],
            length: int = 3
            ) -> list[float]:
        # Creating a dataFrame (required for the pandas_ta module)
        close_pd = pd.DataFrame({"close": []})
        close_pd["close"] = close

        # Calculating SMA
        SMA = ta.sma(close_pd["close"], length=length)

        # Converting SMA to list and rounding it,
        # also removing the NAN value
        SMA_list = SMA.tolist()
        SMA_list = [round(x, 2) for x in SMA_list if not math.isnan(x)]

        return SMA_list
