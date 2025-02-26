from typing import Any
import numpy as np
import pandas as pd
import pandas_ta as ta
import math
import seaborn as sns
import matplotlib.pyplot as plt
import torch

from data_parser.dataReader import DataReader
from data_parser.dataProcessor import DataProcessor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        self._data_processor: DataProcessor|None = DataProcessor()

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
    
    def get_all_raw_data(
            self,
            start_date: str,
            end_date: str,
            interval: str
        ) -> list[tuple[str,float,float,float,float]]:
        return DataReader(
            stock_name = self._stock_name,
            interval = interval
        ).get_all_data(start_date, end_date)

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
            calculate_SMA(stock_data=stock_data, length=sma_lookback_period)

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
    
    def generate_training_loaders(self, start_date, end_date, interval: str = "1d"):
        all_data = self.get_all_raw_data(start_date, end_date, interval)

        residual_metadata, residuals = self._extract_residual_features(all_data)

        sma_metadata, sma = self._extract_sma_features(all_data)

        (
            training_sets_residuals,
            labels_residuals,
            training_sets_sma,
            labels_sma
        ) = self._data_processor.generate_sets_from_metadata(
            residuals_metadata=residual_metadata,
            residuals=residuals,
            sma_metadata=sma_metadata,
            sma=sma,
            points_per_set=self._points_per_set,
            labels_per_set=self._labels_per_set
        )

        return (
            training_sets_residuals.to(device),
            labels_residuals.to(device),
            training_sets_sma.to(device),
            labels_sma.to(device)
        )

    def generate_testing_loaders(self, start_date, end_date, interval: str = "1d"):
        all_data = self.get_all_raw_data(start_date, end_date, interval)

        residual_metadata, residuals = self._extract_residual_features(all_data)

        sma_metadata, sma = self._extract_sma_features(all_data)

        num_testing_data = self._points_per_set - self._labels_per_set

        # Generate the test sets
        testing_sets_residuals = [residual_metadata[i:i + num_testing_data] for i in range(len(residual_metadata) - num_testing_data)]
        testing_sets_sma = [sma_metadata[i:i + num_testing_data] for i in range(len(sma_metadata) - num_testing_data)]
        labels_residuals = [residuals[i + num_testing_data] for i in range(len(residuals) - num_testing_data)]
        labels_sma = [sma[i + num_testing_data] for i in range(len(sma) - num_testing_data)]

        # Convert to Tensors and put on the GPU
        testing_sets_residuals = torch.Tensor(testing_sets_residuals)
        testing_sets_sma = torch.Tensor(testing_sets_sma)
        labels_residuals = torch.Tensor(labels_residuals).reshape(-1,1)
        labels_sma = torch.Tensor(labels_sma).reshape(-1,1)

        return (
            testing_sets_residuals.to(device),
            testing_sets_sma.to(device),
            labels_residuals.to(device),
            labels_sma.to(device)
        )
    
    def _extract_residual_features(self, all_data, corr_heatmap: bool = False):
        """
        res model: residuals, o_min_c, upper_band_min_close, close_min_lower_band, close_min_mid_band, rsi
        trend model: three_day_sma, close, h_min_l, seven_day_sma, fourteen_day_sma, seven_day_std, upper_band_min_close, close_min_lower_band, close_min_mid_band, rsi

        :param all_data: _description_
        :type all_data: _type_
        :return: _description_
        :rtype: _type_
        """
        (
            dates,
            open_,
            high,
            low,
            close,
            volume
        ) = all_data

        # Convert values to lists
        open_ = open_.values.T.tolist()
        high = high.values.T.tolist()
        low = low.values.T.tolist()
        close = close.values.T.tolist()
        volume = volume.values.T.tolist()
        
        # High - Low
        h_min_l = np.array(high) - np.array(low)

        # Open - Close
        o_min_c = np.array(open_) - np.array(close)

        # 3-day SMA
        three_day_sma = self._data_processor.calculate_SMA(close=close, length=3)

        # 7-day SMA
        seven_day_sma = self._data_processor.calculate_SMA(close=close, length=7)

        # 14-day SMA
        fourteen_day_sma = self._data_processor.calculate_SMA(close=close, length=14)

        # 7-day closing price standard deviation
        seven_day_std = pd.Series(close).rolling(window=7).std().dropna().tolist()

        # Bollinger bands
        twenty_day_std = pd.Series(close).rolling(window=20).std().dropna().tolist()
        mid_band = self._data_processor.calculate_SMA(close=close, length=20)
        upper_band = np.array(mid_band) + 2 * np.array(twenty_day_std)
        lower_band = np.array(mid_band) - 2 * np.array(twenty_day_std)

        # Compute RSI
        rsi = self._data_processor.compute_rsi(close)

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
        rsi = rsi[scaler:]
        close = np.array(close)[scaler:]
        three_day_sma = np.array(three_day_sma)[scaler:]

        # Compute residuals
        residuals = self._data_processor.calculate_residuals(
            stock_data=None,
            sma=seven_day_sma.tolist(),
            closing_prices=close.tolist()
        )

        # Upper Bollinger Band - Close
        upper_band_min_close = upper_band - close

        # Close - Lower Bollinger Band
        close_min_lower_band = close - lower_band
        
        # Close - Mid Band (Residuals with the Bollinger band)
        close_min_mid_band = close - mid_band

        # Scale the RSI
        rsi = (rsi - 50) / 50

        metadata = np.column_stack(
            [
                residuals,
                o_min_c,
                upper_band_min_close,
                close_min_lower_band,
                close_min_mid_band,
                rsi
            ]
        )

        if corr_heatmap:
            # Assuming metadata is already defined as a NumPy array
            column_names = [
                "residuals",
                "o_min_c",
                "upper_band_min_close",
                "close_min_lower_band",
                "close_min_mid_band",
                "rsi"
            ]

            # Convert to Pandas DataFrame
            df = pd.DataFrame(metadata, columns=column_names)

            # Shift "close" column up by 2
            df["residuals_shifted"] = df["residuals"].shift(-2)

            # Drop the last row (since shifting creates a NaN value)
            df = df.dropna()

            # Compute correlation matrix
            corr_matrix = df.corr()

            # Plot heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
            plt.title("Correlation Heatmap")
            plt.show()

        return metadata, residuals

    
    def _extract_sma_features(self, all_data, corr_heatmap: bool = False):
        """
        res model: residuals, o_min_c, upper_band_min_close, close_min_lower_band, close_min_mid_band, rsi
        trend model: three_day_sma, close, h_min_l, seven_day_sma, fourteen_day_sma, seven_day_std, upper_band_min_close, close_min_lower_band, close_min_mid_band, rsi

        :param all_data: _description_
        :type all_data: _type_
        :return: _description_
        :rtype: _type_
        """
        (
            dates,
            open_,
            high,
            low,
            close,
            volume
        ) = all_data

        # Convert values to lists
        open_ = open_.values.T.tolist()
        high = high.values.T.tolist()
        low = low.values.T.tolist()
        close = close.values.T.tolist()
        volume = volume.values.T.tolist()
        
        # High - Low
        h_min_l = np.array(high) - np.array(low)

        # Open - Close
        o_min_c = np.array(open_) - np.array(close)

        # 3-day SMA
        three_day_sma = self._data_processor.calculate_SMA(close=close, length=3)

        # 7-day SMA
        seven_day_sma = self._data_processor.calculate_SMA(close=close, length=7)

        # 14-day SMA
        fourteen_day_sma = self._data_processor.calculate_SMA(close=close, length=14)

        # 7-day closing price standard deviation
        seven_day_std = pd.Series(close).rolling(window=7).std().dropna().tolist()

        # Bollinger bands
        twenty_day_std = pd.Series(close).rolling(window=20).std().dropna().tolist()
        mid_band = self._data_processor.calculate_SMA(close=close, length=20)
        upper_band = np.array(mid_band) + 2 * np.array(twenty_day_std)
        lower_band = np.array(mid_band) - 2 * np.array(twenty_day_std)

        # Compute RSI
        rsi = self._data_processor.compute_rsi(close)

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
        rsi = rsi[scaler:]
        close = np.array(close)[scaler:]
        three_day_sma = np.array(three_day_sma)[scaler:]

        # Upper Bollinger Band - Close
        upper_band_min_close = upper_band - close

        # Close - Lower Bollinger Band
        close_min_lower_band = close - lower_band
        
        # Close - Mid Band (Residuals with the Bollinger band)
        close_min_mid_band = close - mid_band

        # Scale the RSI
        rsi = (rsi - 50) / 50

        metadata = np.column_stack(
            [
                three_day_sma,
                close,
                h_min_l,
                seven_day_sma,
                fourteen_day_sma,
                seven_day_std,
                upper_band_min_close,
                close_min_lower_band,
                close_min_mid_band,
                rsi
            ]
        )

        if corr_heatmap:
            # Assuming metadata is already defined as a NumPy array
            column_names = [
                "three_day_sma",
                "close",
                "h_min_l",
                "seven_day_sma",
                "fourteen_day_sma",
                "seven_day_std",
                "upper_band_min_close",
                "close_min_lower_band",
                "close_min_mid_band",
                "rsi"
            ]

            # Convert to Pandas DataFrame
            df = pd.DataFrame(metadata, columns=column_names)

            # Shift "three_day_sma" column up by 9
            df["three_day_sma_shifted"] = df["three_day_sma"].shift(-9)

            # Drop the last row (since shifting creates a NaN value)
            df = df.dropna()

            # Compute correlation matrix
            corr_matrix = df.corr()

            # Plot heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
            plt.title("Correlation Heatmap")
            plt.show()

        return metadata, three_day_sma
    
    '''
    def _extract_features(self, all_data):
        """
        res model: residuals, o_min_c, upper_band_min_close, close_min_lower_band, close_min_mid_band, rsi
        trend model: three_day_sma, close, h_min_l, seven_day_sma, fourteen_day_sma, seven_day_std, upper_band_min_close, close_min_lower_band, close_min_mid_band, rsi

        :param all_data: _description_
        :type all_data: _type_
        :return: _description_
        :rtype: _type_
        """
        (
            dates,
            open_,
            high,
            low,
            close,
            volume
        ) = all_data

        # Convert values to lists
        open_ = open_.values.T.tolist()
        high = high.values.T.tolist()
        low = low.values.T.tolist()
        close = close.values.T.tolist()
        volume = volume.values.T.tolist()
        
        # High - Low
        h_min_l = np.array(high) - np.array(low)

        # Open - Close
        o_min_c = np.array(open_) - np.array(close)

        # 3-day SMA
        three_day_sma = self._data_processor.calculate_SMA(close=close, length=3)

        # 7-day SMA
        seven_day_sma = self._data_processor.calculate_SMA(close=close, length=7)

        # 14-day SMA
        fourteen_day_sma = self._data_processor.calculate_SMA(close=close, length=14)

        # 7-day closing price standard deviation
        seven_day_std = pd.Series(close).rolling(window=7).std().dropna().tolist()

        # Bollinger bands
        twenty_day_std = pd.Series(close).rolling(window=20).std().dropna().tolist()
        mid_band = self._data_processor.calculate_SMA(close=close, length=20)
        upper_band = np.array(mid_band) + 2 * np.array(twenty_day_std)
        lower_band = np.array(mid_band) - 2 * np.array(twenty_day_std)

        # Compute RSI
        rsi = self._data_processor.compute_rsi(close)

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
        rsi = rsi[scaler:]
        close = np.array(close)[scaler:]
        three_day_sma = np.array(three_day_sma)[scaler:]

        # Compute residuals
        residuals = self._data_processor.calculate_residuals(
            stock_data=None,
            sma=seven_day_sma.tolist(),
            closing_prices=close.tolist()
        )

        # Upper Bollinger Band - Close
        upper_band_min_close = upper_band - close

        # Close - Lower Bollinger Band
        close_min_lower_band = close - lower_band
        
        # Close - Mid Band (Residuals with the Bollinger band)
        close_min_mid_band = close - mid_band

        # Scale the RSI
        rsi = (rsi - 50) / 50

        metadata = np.column_stack(
            [
                close,
                h_min_l,
                o_min_c,
                three_day_sma,
                residuals,
                seven_day_sma,
                fourteen_day_sma,
                seven_day_std,
                upper_band_min_close,
                close_min_lower_band,
                close_min_mid_band,
                rsi
            ]
        )

        # Assuming metadata is already defined as a NumPy array
        column_names = [
            "close",
            "h_min_l",
            "o_min_c",
            "three_day_sma",
            "residuals",
            "seven_day_sma",
            "fourteen_day_sma",
            "seven_day_std",
            "upper_band_min_close",
            "close_min_lower_band",
            "close_min_mid_band",
            "rsi"
        ]

        # Convert to Pandas DataFrame
        df = pd.DataFrame(metadata, columns=column_names)

        # Shift "close" column up by 1
        df["close_shifted"] = df["close"].shift(-9)
        df["residuals_shifted"] = df["residuals"].shift(-1)
        df["three_day_sma_shifted"] = df["three_day_sma"].shift(-9)

        # Drop the last row (since shifting creates a NaN value)
        df = df.dropna()

        # Compute correlation matrix
        corr_matrix = df.corr()

        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.show()

        return metadata, residuals, three_day_sma, close
    '''

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
            simple_moving_average = self._data_processor.calculate_SMA(stock_data=set_)
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