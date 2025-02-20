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

    def comparison_plot(self, predictions_std: list[float]|None = None, bollinger_band: bool = False):
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

        if predictions_std is not None:
            self._closing_prices.insert(0, self._closing_prices[0])
            self._closing_prices.pop()
            lower_bound_fill = self._closing_prices
            # Plotting the shaded area for standard deviation (std) range
            plt.fill_between(
                self._dates,  # X-axis values (indices)
                np.array(self._predicted_closing_prices).T[0] - 3 * np.array(predictions_std).T[0],#lower_bound_fill,  # Lower bound (predicted - std)
                np.array(self._predicted_closing_prices).T[0] + 3 * np.array(predictions_std).T[0],  # Upper bound (predicted + std)
                color='red',  # Color of the shaded area
                alpha=0.3,  # Transparency of the shaded area
            )
            
        # Plot the line
        ax.plot(self._dates, self._predicted_closing_prices, color="red")

        # Add square markers for data points
        ax.scatter(self._dates, self._predicted_closing_prices, color="red", marker="^", label="Predicted")

        # Labels and title
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        ax.set_title("Actual vs Predicted closing prices")

        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

        # Legend
        ax.legend()

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
