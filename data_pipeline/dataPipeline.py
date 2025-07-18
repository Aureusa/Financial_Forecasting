from datetime import datetime
from copy import deepcopy
import pandas as pd
import pandas_ta as ta
import math
import numpy as np
import torch

from .stockGetter import Stock


class DataPipeline(Stock):
    def __init__(
            self,
            name: str,
            start_date: str,
            end_date: str,
            interval: str = "1d"
        ):
        """
        Initializes a DataPipeline object with stock data retrieval capabilities.

        :param name: The name of the stock or financial instrument.
        :type name: str
        :param start_date: The start date for retrieving stock data.
        :type start_date: str
        :param end_date: The end date for retrieving stock data.
        :type end_date: str
        :param interval: The interval for the stock data, defaults to "1d".
        :type interval: str
        """
        super().__init__(name, start_date, end_date, interval)
        (
            self.dates,
            self.open_,
            self.high,
            self.low,
            self.close,
            self.volume
            ) = self.get_all_data()

    def get_sma(
            self,
            lookback: int = 3
            ) -> list[float]:
        """
        Calculates the Simple Moving Average (SMA) for the stock's closing prices.

        :param lookback: The number of periods to calculate the SMA over.
        :type lookback: int
        :return: A list of SMA values.
        :rtype: list[float]
        """
        # Creating a dataFrame (required for the pandas_ta module)
        close_pd = pd.DataFrame({"close": []})
        close_pd["close"] = self.close

        # Calculating SMA
        sma = ta.sma(close_pd["close"], length=lookback)

        # Converting SMA to list and rounding it,
        # also removing the NAN value
        sma_list = sma.tolist()
        sma_list = [round(x, 2) for x in sma_list if not math.isnan(x)]
        return sma_list
    
    def get_residuals_data(
            self,
            sma: list[float]|None = None,
            lookback: int = 3
            ) -> list[float]:
        """
        Calculates the residuals of the stock's closing prices from the SMA.

        :param lookback: The number of periods to calculate the SMA over.
        :type lookback: int
        :return: A list of residuals.
        :rtype: list[float]
        """
        closing_prices = self.close

        if sma is None:
            # If SMA is not provided, calculate it
            sma = self.get_sma(lookback)

        nr_of_residuals = len(sma)
        closing_prices = closing_prices[-nr_of_residuals:]

        residuals = [round(a - b, 2) for a, b in zip(sma, closing_prices)]

        return residuals
    
    def get_raw_data(self) -> list[tuple[str, float, float, float, float]]:
        """
        Retrieves the stock data in a structured format.

        :return: A list of tuples containing stock data in the format:
            (date, open, high, low, close)
        :rtype: list[tuple[str, float, float, float, float]]
        """
        # Combine into DOHLC format
        # (dates, open, high, low, close)
        data = [
            (dat, op, hi, lo, cl)
            for dat, op, hi, lo, cl in zip(self.dates, self.open_, self.high, self.low, self.close)
            ]
        return data
    
    def get_training_data(
            self,
            pointsPerSet: int = 3,
            labelsPerSet: int = 1,
            sma_data: bool = False,
            sma_lookback: int = 3
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates training data and labels from the stock data.

        :param pointsPerSet: The number of data points per set, defaults to 3.
        :type pointsPerSet: int
        :param labelsPerSet: The number of labels per set, defaults to 1.
        :type labelsPerSet: int
        :param sma_data: Whether to use the Simple Moving Average data, defaults to False.
        :type sma_data: bool
        :param sma_lookback: The lookback period for the SMA, defaults to 3.
        :type sma_lookback: int
        :return: A tuple containing the training data and labels.
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        if sma_data:
            data = self.get_sma(sma_lookback)
        else:
            data = self.get_residuals_data(lookback=sma_lookback)
        
        # Generate sets from the data
        sets = self._generate_sets(
            data,
            pointsPerSet
        )

        # Generate labels from the data
        data, labels = self.generate_labels(sets=sets, label_size=labelsPerSet)

        return (
            np.array(data),
            np.array(labels)
            )
    
    def _generate_sets(
            self,
            data: list[float],
            pointsPerSet: int
        ) -> list[list[float]]:
        """
        Generates sets of data points from the provided data.

        :param data: The data to generate sets from.
        :type data: list[float]
        :param pointsPerSet: The number of data points per set.
        :type pointsPerSet: int
        :return: A list of sets, each containing the specified number of data points.
        :rtype: list[list[float]]
        """
        allData = []
        for i in range(len(data)//pointsPerSet):
            dat = data[i*pointsPerSet:(i+1)*pointsPerSet]
            allData.append(dat)
        return allData

    def generate_labels(
            self,
            sets: list[list[float]],
            label_size: int = 1
        ) -> tuple[list[list[float]], list[list[float]]]:
        """
        Generates labels from the provided sets of data.
        
        :param sets: The sets of data to generate labels from.
        :type sets: list[list[float]]
        :param label_size: The number of labels per set, defaults to 1.
        :type label_size: int
        :return: A tuple containing the data and labels.
        :rtype: tuple[list[list[float]], list[list[float]]]
        """
        allData = []
        allLabels = []
        for set in sets:
            allData.append(set[:-label_size])
            allLabels.append(set[-label_size:])
        return allData, allLabels
    