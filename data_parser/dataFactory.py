from typing import Any
import numpy as np
import pandas as pd
import pandas_ta as ta
import math

from data_parser.dataReader import DataReader
from data_parser.dataProcessor import DataProcessor

class StockDataFactory:
    """
    In accordance with the design pattern "Factory Method" this
    class is used to generate stock data that is used for training
    a neural netowork.
    """
    def __init__(
            self,
            stock_name: str,
            points_per_set: int,
            labels_per_set: int,
            ) -> None:
        """
        A way of initialising a StockDataFactory.

        :param stock_name: the name of the stock
        :type stock_name: str
        :param labels_per_set: the labels per set
        :type labels_per_set: int
        """
        self._stock_name = stock_name
        self._labels_per_set = labels_per_set
        self._points_per_set = points_per_set

        self._data_reader: DataReader|None = None
        self._data_processor: DataProcessor|None = None

    def get_training_data(self, start_date: str, end_date: str, sma_data: bool = False):
        # Generate the sets 
        sets = self._generate_sets_from_dates(start_date, end_date)

        if sma_data:
            # Preprocess the simple moving average
            processed_data = self._preprocess_data(sets, sma_data)
        else:
            # Preprocess the residuals
            processed_data = self._preprocess_data(sets, sma_data)

        # Generate labels from the data
        data, labels = self._get_labeled_data(processed_data)
        
        return (
            np.array(data),
            np.array(labels)
            )

    def get_grand_ensemble_model_data(
            self,
            start_date: str,
            end_date: str,
            training: bool = True
        ):
        # Generate the sets
        if training:
            (
                metadata_sets,
                residuals_sets,
                three_day_sma_sets,
                close_sets
            ) = self._generate_metasets_from_dates(start_date, end_date)
        else:
            (
                metadata_sets,
                residuals_sets,
                three_day_sma_sets,
                close_sets
            ) = self._generate_metasets_from_dates(start_date, end_date, True)

        # Remove the last element of the sets
        mlp_data = self._remove_last_entry_in_sets(residuals_sets)
        lstm_data = self._remove_last_entry_in_sets(three_day_sma_sets)

        # Get the second to last entry of the metadata
        meta_mlp_data = self._get_entry_in_sets(data=metadata_sets, index=-2)

        # Get the labels (i.e. the last element in each set)
        labels = self._get_entry_in_sets(data=close_sets, index=-1)
        
        return (
            np.array(mlp_data),
            np.array(lstm_data),
            np.array(meta_mlp_data),
            np.array(labels)
        )
    
    def get_raw_data(
            self,
            start_date: str,
            end_date: str,
            interval: str
        ) -> list[tuple[str,float,float,float,float]]:
        return DataReader(
            stock_name = self._stock_name,
            interval = interval
        ).get_data(start_date, end_date)

    def get_sma(
            self,
            data: list[tuple[str,float,float,float,float]],
            sma_lookback_period: int
            ) -> list[float]:
        """
        A way of getting the simple moving average of raw data.

        :param data: the data you want to get the simple moving
        average of.
        :type data: list[tuple[str,float,float,float,float]]
        :param sma_lookback_period: the lookback period for the
        calculation of the SME average. This is the number of datapoints
        used to calculate the SME. Example:
        if sma_lookback_period = 3:
            take: mean(last 3 points)
        :type sma_lookback_period: int
        :return: returns: SMA
        :rtype: list[float]
        """
        stock_data = DataProcessor(data).data
        return DataProcessor(None).\
            calculate_SMA(stock_data, length = sma_lookback_period)
    
    def get_residuals_data(
            self,
            raw_data: list[tuple[str, float, float, float, float]],
            sma: list[float]
            ) -> list[float]:
        """
        A way of getting the residuals of the SMA and the
        closing prices.

        :param raw_data: the raw data
        :type raw_data: _type_
        :param sma: the SMA of the raw data
        :type sma: _type_
        :return: the residuals
        :rtype: list[float]
        """
        stock_data = DataProcessor(raw_data).data
        return DataProcessor(None).\
            calculate_residuals(stock_data, sma)
    
    def _generate_sets_from_dates(self, stard_date: str, end_date: str):
        # Get data
        self._data_reader = DataReader(self._stock_name)
        stock_data = self._data_reader.get_data(
            stard_date,
            end_date
            )
        
        # Generate sets
        self._data_processor = DataProcessor(stock_data)
        sets = self._data_processor.generate_sets(
            self._points_per_set+2)
        return sets
    
    def _generate_metasets_from_dates(self, stard_date: str, end_date: str, rolling_sets: bool = False):
        # Get data
        self._data_reader = DataReader(self._stock_name)
        all_data = self._data_reader.get_all_data(
            stard_date,
            end_date
            )
        
        # Build the features
        metadata, residuals, three_day_sma, close = self._extract_features(all_data)

        if rolling_sets:
            # Generate the test sets
            metadata_sets = [metadata[i:i + 10] for i in range(len(metadata) - 10 + 1)]
            residuals_sets = [residuals[i:i + 10] for i in range(len(residuals) - 10 + 1)]
            three_day_sma_sets = [three_day_sma[i:i + 10] for i in range(len(three_day_sma) - 10 + 1)]
            close_sets = [close[i:i + 10] for i in range(len(close) - 10 + 1)]

            return (
                metadata_sets,
                residuals_sets,
                three_day_sma_sets,
                close_sets
            )

        # Generate the train sets
        (
            metadata_sets,
            residuals_sets,
            three_day_sma_sets,
            close_sets
        ) = self._generate_sets(
            metadata,
            residuals,
            three_day_sma,
            close,
            self._points_per_set
        )

        return (
            metadata_sets,
            residuals_sets,
            three_day_sma_sets,
            close_sets
        )

    def _extract_features(self, all_data):
        (
            dates,
            open_,
            high,
            low,
            close,
            volume
        ) = all_data

        open_ = open_.values.T.tolist()[0]
        high = high.values.T.tolist()[0]
        low = low.values.T.tolist()[0]
        close = close.values.T.tolist()[0]
        volume = volume.values.T.tolist()[0]
        
        h_min_l = np.array(high) - np.array(low)

        o_min_c = np.array(open_) - np.array(close)

        three_day_sma = self._calculate_sma(close, 3)

        seven_day_sma = self._calculate_sma(close, 7)

        fourteen_day_sma = self._calculate_sma(close, 14)

        seven_day_std = pd.Series(close).rolling(window=7).std().dropna().tolist()

        # Bollinger band
        mid_band = self._calculate_sma(close, 20)
        twenty_day_std = pd.Series(close).rolling(window=20).std().dropna().tolist()

        upper_band = np.array(mid_band) + 2 * np.array(twenty_day_std)
        lower_band = np.array(mid_band) - 2 * np.array(twenty_day_std)

        # Resize the arrays
        scaler = - len(upper_band)
        h_min_l = np.array(h_min_l)[scaler:]
        o_min_c = np.array(o_min_c)[scaler:]
        seven_day_sma = np.array(seven_day_sma)[scaler:]
        fourteen_day_sma = np.array(fourteen_day_sma)[scaler:]
        seven_day_std = np.array(seven_day_std)[scaler:]
        volume = np.array(volume)[scaler:]
        dates = np.array(dates)[scaler:]
        mid_band = np.array(mid_band)
        upper_band = np.array(upper_band)
        lower_band = np.array(lower_band)
        rsi = self._compute_rsi(close)[scaler:]
        close = np.array(close)[scaler:]

        upper_band_close_diff = np.abs(upper_band-close)
        lower_band_close_diff = np.abs(lower_band-close)
        mid_band_close_diff = np.abs(mid_band-close)

        metadata = np.column_stack(
            [
                close,
                h_min_l,
                o_min_c,
                seven_day_sma,
                fourteen_day_sma,
                seven_day_std,
                mid_band_close_diff,
                upper_band_close_diff,
                lower_band_close_diff,
                rsi
            ]
        )

        # Slice the 3 day SMA and convert it to ndarray
        three_day_sma = np.array(three_day_sma)[scaler:]

        # Compute residuals
        residuals = self._calculate_residuals(close.tolist(), three_day_sma.tolist())

        return metadata, residuals, three_day_sma, close

    def _compute_rsi(self, closing_prices, period=14):
        df = pd.DataFrame({"Close": closing_prices})
        delta = df["Close"].diff(1)  # Calculate daily price changes

        gain = np.where(delta > 0, delta, 0)  # Keep only positive gains
        loss = np.where(delta < 0, -delta, 0)  # Keep only negative losses

        avg_gain = pd.Series(gain).rolling(window=period, min_periods=1).mean()
        avg_loss = pd.Series(loss).rolling(window=period, min_periods=1).mean()

        rs = avg_gain / avg_loss  # Relative Strength
        rsi = 100 - (100 / (1 + rs))  # RSI Calculation

        return rsi.dropna().to_numpy()
    
    def _calculate_residuals(
            self,
            closing_prices: list[float],
            sma: list[float]
            ) -> list[float]:
        """
        Calculates the residuals by substracting the closing prices
        from a Simple Moving Average (SMA).
        """
        residuals = [round(a - b, 2) for a, b in zip(sma, closing_prices)]

        return residuals
    
    def _generate_sets(
            self,
            metadata, residuals, three_day_sma, close, pointsPerSet
            ):
        # Get the data length
        data_len = len(metadata)

        # Define the different lists that hold the sets
        metadata_sets = []
        residuals_sets = []
        three_day_sma_sets = []
        close_sets = []

        # Populate the lists
        for i in range(data_len//pointsPerSet):
            md = metadata[i*pointsPerSet:(i+1)*pointsPerSet]
            res = residuals[i*pointsPerSet:(i+1)*pointsPerSet]
            tds = three_day_sma[i*pointsPerSet:(i+1)*pointsPerSet]
            cl = close[i*pointsPerSet:(i+1)*pointsPerSet]

            metadata_sets.append(md)
            residuals_sets.append(res)
            three_day_sma_sets.append(tds)
            close_sets.append(cl)

        return (
            metadata_sets,
            residuals_sets,
            three_day_sma_sets,
            close_sets
        )
    
    def _preprocess_data(
            self,
            sets: list[list[float]],
            sma_data: bool,
            ) -> list[list[float]]:
        """
        This method is used to preproccess the data
        of the different sets.

        :param sets: a list of sets of stock data
        :type sets: list[list[float]]
        :return: a list of preprocessed data
        :rtype: list[list[float]]
        """
        data = []
        for set_ in sets:
            simple_moving_average = self._data_processor.calculate_SMA(set_)
            if sma_data:
                # Get data for the LSTM (i.e. SMA data)
                data.append(simple_moving_average)
            else:
                # Get data for the NN (i.e. residual data)
                residual = self._data_processor.calculate_residuals(
                    set_,
                    simple_moving_average
                    )
                data.append(residual)
        return data

    def _remove_last_entry_in_sets(self, data: list[np.ndarray]):
        sliced_data = []
        for dat in data:
            new_dat = dat[:-1]
            sliced_data.append(new_dat)
        return sliced_data
    
    def _get_entry_in_sets(self, data: list[np.ndarray], index: int):
        sliced_data = []
        for dat in data:
            new_dat = dat[index]
            sliced_data.append(new_dat)
        return sliced_data

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
    
    def _get_labeled_data(
            self,
            processed_data: Any
            ) -> tuple[list[list[float]], list[list[float]]]:
        """
        Labels the data to prepare it for train, test, validation split

        :param processed_data: the processed data to generate the labels
        of
        :type processed_data: Any
        :return: a tuple with data and labels
        :rtype: tuple[list[list[float]], list[list[float]]]
        """
        data, labels = self._data_processor.generate_labels(
            processed_data,
            self._labels_per_set
            )
        return data, labels
    