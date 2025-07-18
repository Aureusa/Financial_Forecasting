import pandas as pd
import numpy as np
import pandas_ta as ta
import math

from data_pipeline.stockGetter import Stock


class DataMaker:
    """
    A class to forge data for testing the model in a real-world scenario.
    It generates sets of stock data, including dates, closing prices,
    Simple Moving Averages (SMA), and residuals, based on the stock codes
    provided. The data is sliced into sets of points and labels, which can
    be used for training and testing machine learning models.
    """
    def forge_data(
            self,
            stock_codes: list[str],
            start_date: str,
            end_date: str,
            points_per_set: int,
            labels_per_set: int,
            lookback: int = 3
        ) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        """
        Forges data to be used for testing the model in the way it would work in a real-world
        scenario.

        :param stock_codes: a list containing stock codes whose data is used in the experiment
        :type stock_codes: list[str]
        :param start_date: the starting date to retrieve data from. Must be in the format `yyyy-mm-dd`.
        :type start_date: str
        :param end_date: the end date to retrieve data from. Must be in the format `yyyy-mm-dd`.
        :type end_date: str
        :param points_per_set: the number of points per set
        :type points_per_set: int
        :param labels_per_set: the number of labels
        :type labels_per_set: int
        :param lookback: the number of past points to be considered for the Simple Moving Average (SMA)
        :type lookback: int, optional
        :return: returns a tuple containg:
            stock_codes (as np.ndarray)
            stocks_features (as np.ndarray):
                In the shape of (STOCKS, TYPE_FEATURES, FEATURE, DIM = points_per_set-labels_per_set)
                The type of features are: [dates_features, close_features, sma_features, residuals_features]
            stocks_labels (as np.ndarray):
                In the shape of (STOCKS, TYPE_FEATURES, FEATURE, DIM = labels_per_set)
                The type of features are: [dates_labels, close_labels, sma_labels, residuals_labels]
        :rtype: tuple[list[str],np.ndarray,np.ndarray]
        """
        stocks_features = []
        stocks_labels = []
        for stock in stock_codes:
            dates, _, _, _, close = Stock(
                name=stock,
                start_date=start_date,
                end_date=end_date
            ).get_data()

            # Convert to lists
            dates = [date.strftime("%Y-%m-%d") for date in dates.to_list()]
            close = close.to_list()

            # Compute the SMA and adjust number of closing and data
            sma_data, close = self._calculate_sma(close, lookback)
            dates = dates[-len(close):]

            # Compute residuals
            residuals = [round(a - b, 2) for a, b in zip(sma_data, close)]

            # Generate sets
            dates_sets = self._generate_sets(
                data=dates,
                points_per_set=points_per_set
            )
            close_sets = self._generate_sets(
                data=close,
                points_per_set=points_per_set
            )
            sma_data_sets = self._generate_sets(
                data=sma_data,
                points_per_set=points_per_set
            )
            residuals_sets = self._generate_sets(
                data=residuals,
                points_per_set=points_per_set
            )

            # Generate labels
            dates_features, dates_labels = self._generate_labels(
                data=dates_sets,
                label_size=labels_per_set
            )
            close_features, close_labels = self._generate_labels(
                data=close_sets,
                label_size=labels_per_set
            )
            sma_features, sma_labels = self._generate_labels(
                data=sma_data_sets,
                label_size=labels_per_set
            )
            residuals_features, residuals_labels = self._generate_labels(
                data=residuals_sets,
                label_size=labels_per_set
            )

            # Append all the features
            stocks_features.append([dates_features, close_features, sma_features, residuals_features])

            # Append all the labels
            stocks_labels.append([dates_labels, close_labels, sma_labels, residuals_labels])

        stock_codes = np.array(stock_codes) # (STOCKS,)
        stocks_features = np.array(stocks_features) # (STOCKS, TYPE_FEATURES, FEATURE, DIM = points_per_set-labels_per_set)
        stocks_labels = np.array(stocks_labels) # (STOCKS, TYPE_FEATURES, FEATURE, DIM = labels_per_set)

        return stock_codes, stocks_features, stocks_labels

    def _generate_labels(
            self,
            data: list[list],
            label_size: int
            ) -> tuple[list[list], list[list]]:
        """
        Generates labels for a given data based on
        label size.

        :param data: the data to generate labels on
        :type data: list[list]
        :param label_size: the number of labels (size)
        :type label_size: int, optional
        :return: tuple of data and labels.
        :rtype: tuple[list[list], list[list]]
        """
        all_data = []
        all_labels = []
        for set in data:
            all_data.append(set[:-label_size])
            all_labels.append(set[-label_size:])
        return all_data, all_labels

    def _generate_sets(self, data: list, points_per_set: int) -> list:
        """
        Generates sets from data and a given number of points per set,
        using the sliding window approach (or rolling window).

        :param data: the data to be made into sets
        :type data: list
        :param points_per_set: the number of points per set
        :type points_per_set: int
        :return: a list of sets
        :rtype: list
        """
        sets = [data[i:i + 3] for i in range(len(data) - points_per_set + 1)]
        return sets

    def _calculate_sma(self, close: list[float], lookback: int) -> tuple:
        """
        Calculates the Simple Moving Average (SMA) of the
        closing price given.

        :param close: the closing prices
        :type close: list[flaot]
        :param lookback: the number of past points to be considered
        :type lookback: int
        :return: the simple moving average of the data as well as the
        sliced closing prices (to match the len of the SMA)
        :rtype: tuple
        """
        # Creating a dataFrame (required for the pandas_ta module)
        close_pd = pd.DataFrame({"close": []})
        close_pd["close"] = close

        # Calculating SMA
        SMA = ta.sma(close_pd["close"], length=lookback)

        # Converting SMA to list and rounding it,
        # also removing the NAN value
        SMA_list = SMA.tolist()
        SMA_list = [round(x, 2) for x in SMA_list if not math.isnan(x)]

        close = close[-len(SMA_list):]

        return SMA_list, close
    