from copy import deepcopy
import pandas as pd
import pandas_ta as ta
import math


class DataProcessor:
    """
    Serves as a way to process stock data from Yahoo's API.
    """
    def __init__(
            self,
            data: list[tuple[str,float,float,float,float]]|None,
            unpack: bool = True
            ) -> None:
        """
        A way of instantating a proccessor object for stock data.

        :param data: the data as a list of tuples, where the first
        element is the date, and the rest are:
        open, high, low, and close prices.
        :type data: list[tuple[str,float,float,float,float]]
        """
        self._dates = None
        self._data: list[tuple[float, float, float, float]] = None
        if data is not None and unpack is True:
            self._unpack_data(data)
        else:
            self._data = data
    
    @property
    def data(self) -> tuple[float,float,float,float]:
        """
        Retunrs a deepcopy of the stock data in OHLC tuple.
        """
        return deepcopy(self._data)

    def calculate_SMA(
            self,
            stock_data: list[tuple[float,float,float,float]],
            length: int = 3
            ) -> list[float]:
        """
        Ccalculates the Simple Moving Average for a
        given dataset over a specified lookback time.
        
        :param stock_data: the stock data of whic to calculate
        the SMA.
        :type stock_data: list[tuple[float,float,float,float]]
        :param length: the length of the period to consider
        when calculating the Simple Moving Average (SMA).
        :type length: int (optional)
        :return SMA_list: lists of floats representing
        the SMA
        :sets_SMA SMA_list: list[list[Float]]
        """
        # Unzipping the close
        _, _, _, close = zip(*stock_data)

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
        
    def calculate_residuals(
            self,
            stock_data: list[tuple[float,float,float,float]],
            sma: list[float]
            ) -> list[float]:
        """
        Calculates the residuals by substracting the closing prices
        from a Simple Moving Average (SMA).

        :param stock_data: the stock data of whic to calculate
        the SMA.
        :type stock_data: list[tuple[float,float,float,float]]
        :param sma: the Simple Moving Average of the data.
        :sma type: list[float]
        :return residuals: the difference between SMA and the closing
        prices.
        :residuals type: list[float]
        """
        _, _, _, closing_prices = zip(*stock_data)

        nr_of_residuals = len(sma)
        closing_prices = closing_prices[-nr_of_residuals:]

        residuals = [round(a - b, 2) for a, b in zip(sma, closing_prices)]

        return residuals
    
    def generate_labels(
            self,
            processed_data: list[list[float]],
            label_size: int = 5
            ) -> tuple[list[list[float]], list[list[float]]]:
        """
        Generates labels for a given data based on
        label size.

        :param processed_data: the data to generate labels on
        :type processed_data: list[list[float]]
        :param label_size: the number of labels (size), defaults to 5
        :type label_size: int, optional
        :return: tuple of data and lebels to be used for test, train,
        validation split.
        :rtype: tuple[list[list[float]], list[list[float]]]
        """
        allData = []
        allLabels = []
        for set in processed_data:
            allData.append(set[:-label_size])
            allLabels.append(set[-label_size:])
        return allData, allLabels
    
    def generate_sets(
            self,
            pointsPerSet: int
            ) -> list[list[float]]:
        """
        Generates sets from the Stock data to be used in training.
        Usually this is used to compute SME and get the residuals
        in order to train a FFNN.

        :param pointsPerSet: the points per data set
        :pointsPerSet type: int
        """
        allData = []
        for i in range(len(self._data)//pointsPerSet):
            data = self._data[i*pointsPerSet:(i+1)*pointsPerSet]
            allData.append(data)
        return allData

    def _unpack_data(
            self,
            data: list[tuple[str,float,float,float,float]]
            ) -> None:
        """
        Unpacks the data and separates Date from the Stock Data.
        Used in the instantiation of the Class

        :param data: stock data containing (date, open, high, low, close) data.
        :type data: list[tuple[str,float,float,float,float]]
        """
        dates, open_, high, low, close = zip(*data)
        self._dates = dates
        data = [(op, hi, lo, cl) for op, hi, lo, cl in zip(open_, high, low, close)]
        rounded_data = self._round_data(data)
        self._data = rounded_data
    
    def _round_data(
            self,
            data: list[tuple[float, float, float, float]]
            ) -> list[tuple[float, float, float, float]]:
        """
        Rounds a Stock Data to two decimals.

        :param data: the data as given by the getData method
        from the dataReader class.
        :data type: list[tuple[float, float, float, float]
        :return: the rounded data
        :return type: list[tuple[float, float, float, float]
        """
        rounded_data = []
        for tup in data:
            # Round each value in the tuple
            rounded_tup = tuple(round(value, 2) for value in tup)
            rounded_data.append(rounded_tup)
        return rounded_data
    