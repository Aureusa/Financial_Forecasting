from typing import Any
import numpy as np

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
    