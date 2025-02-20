from sklearn.metrics import mean_absolute_error
import tensorflow as tf
import numpy as np
from typing import Any

from data_parser.dataFactory import StockDataFactory
from data_parser.dataProcessor import DataProcessor
from forecast.ensembleModel import EnsembleModel
from visualisation.visualize_simple import Plotter


class ForcastFactoryEnsemble:
    """
    This class provides a way to test the forcasting abilities of an
    Ensemble model.
    """
    def __init__(
            self,
            stock_name: str,
            residual_model: str,
            residual_model_folder: str,
            trend_model: str,
            trend_model_folder: str,
            pointsPerSet: int,
            labelsPerSet: int
            ) -> None:
        """
        A way of instantiating ForcastFactoryEnsemble.

        :param stock_name: the stock code.
        :type stock_name: str
        :param residual_model: the residual model's name
        :type residual_model: str
        :param residual_model_folder: the residual model's folder
        :type residual_model_folder: str
        :param trend_model: the trend model's name
        :type trend_model: str
        :param trend_model_folder: the trend model's folder
        :type trend_model_folder: str
        :param pointsPerSet: points per set
        :type pointsPerSet: int
        :param labelsPerSet: labels per set
        :type labelsPerSet: int
        """
        self._stock_name = stock_name
        
        # Initialize Ensemble Model
        self._ensemble_model = EnsembleModel(
            residual_model, residual_model_folder, trend_model, trend_model_folder
        )

        # Initialize a StockDataFactory
        self._data_factory = StockDataFactory(
            stock_name,
            pointsPerSet,
            labelsPerSet
        )

        # Values that are being calculated
        self._raw_data: list[
            tuple[str, float, float, float, float]
            ]|None = None
        self._sma: list[float]|None = None
        self._residuals: list[float]|None = None

        # Predictions
        self._predicted_residuals: list[float]|None = None
        self._predicted_closing_prices: list[float]|None = None
        self._extrapolated_sma: list[float]|None = None

    def predict(
            self,
            start_date: str,
            end_date: str,
            sma_lookback_period: int = 3,
            interval: str = "1d"
            ) -> None:
        """
        Predicts the closing prices and residuals.

        :param raw_data_amount: the amount of raw data to be generated,
        used for the plotting faculties of this factory as well, defaults to 50
        :type raw_data_amount: int, optional
        :param sma_lookback_period: the lookback time used to calculate
        the simple moving average (this is a hyperparameter for the ML model),
        defaults to 3
        :type sma_lookback_period: int, optional
        :param end_date: the end date of the raw data retrieval,
        defaults to "2024-09-01"
        :type end_date: str, optional
        :param interval: the scale of the candles, defaults to "1d"
        :type interval: str, optional
        """
        self._get_raw_data(start_date, end_date, interval)

        self._predict_residuals(sma_lookback_period)

        self._extrapolate_sma()

        self._predicted_closing_prices, self._predicted_closing_prices_std = self._calculate_predicted_closing_prices()

    def compare_predictions_with_observations(self) -> float:
        """
        Compare the model's predictions with the actual observed data.

        :return: the mean absolute error of the observed closing prices
        and the predicted closing prices
        :rtype: float
        """
        self._validate_predictions(self._predicted_closing_prices)

        actual_closing_prices = self._calculate_actual_closing_prices()

        predicted_closing_prices = self._predicted_closing_prices

        direction_success_rate = self._calculate_direction_success_rate(predicted_closing_prices, actual_closing_prices)

        range_match_success_rate = self._calculate_range_match_success_rate(predicted_closing_prices, actual_closing_prices)
        
        # Calculate Mean Absolute Error (MAE)
        mae = mean_absolute_error(
            actual_closing_prices,
            predicted_closing_prices
            )
        
        return mae, direction_success_rate, range_match_success_rate
    
    def make_comparison_plot(self, bollinger_band: bool):
        closing_prices = self._calculate_actual_closing_prices()

        dates = [t[0] for t in self._raw_data]

        dates = dates[-len(closing_prices):]

        plotter = Plotter(closing_prices, self._predicted_closing_prices, dates)

        plotter.comparison_plot(self._predicted_closing_prices_std, bollinger_band)

    def _calculate_direction_success_rate(self, predicted_closing_prices, actual_closing_prices):
        # Set-up the arrays
        predicted_closing = np.array(predicted_closing_prices).T[0][1:]
        actual_closing = np.array(actual_closing_prices)[:-1]
        actual_closing_prices = np.array(actual_closing_prices)

        # Check whether the predictions indicate an upward trend
        # this is an array containing True (if cond in met) False otherwise
        predictions = predicted_closing > actual_closing

        # Check wether there is an upward trend, if the next actual closing
        # price is greater than the previous one set to True, otherwise False
        ground_truth = actual_closing_prices[:-1] < actual_closing_prices[1:]

        # Compare predictions with ground_truth (element-wise equality)
        success = predictions == ground_truth

        # Calculate success rate as the percentage of correct predictions
        success_rate = np.mean(success) * 100

        return success_rate
    
    def _calculate_range_match_success_rate(self, predicted_closing_prices, actual_closing_prices):
        sigma = np.array(self._predicted_closing_prices_std).T[0]

        upper_bound = np.array(predicted_closing_prices).T[0] + 3 * sigma
        lower_bound = np.array(predicted_closing_prices).T[0] - 3 * sigma

        condition = (actual_closing_prices < upper_bound) & (actual_closing_prices > lower_bound)

        success_rate = np.mean(condition) * 100

        return success_rate
    
    def _calculate_predicted_closing_prices(self) -> list[float]:
        """
        Calculates the closing prices from the extrapolated SMA
        and the predicted residuals.

        :return: a list of the predicted closing prices
        :rtype: list[float]
        """
        # Compute the std of predicted closing prices
        predicted_closing_prices_std = [
            (std[0]**2 + std[1]**2 ) ** 0.5
            for std in
            zip(
                self._extrapolated_sma_std,
                self._predicted_residuals_std
            )
        ]

        predicted_closing_prices = [
            sum(x)
            for x in
            zip(
                self._extrapolated_sma,
                self._predicted_residuals
            )
        ]

        return predicted_closing_prices, predicted_closing_prices_std

    def _calculate_actual_closing_prices(self) -> list[float]:
        """
        Calculates the actual closing prices.

        :return: a list of actual closing prices.
        :rtype: list[float]
        """
        return [round(sum(x+y),2) for x, y in zip(self._actual_residuals,self._actual_sma)]

    def _predict_residuals(self, sma_lookback_period: int) -> None:
        """
        Used to predict the residuals. First it calculates the SMA
        of the raw data, get's their residuals, preprocess them,
        and does the prediction.

        :param sma_lookback_period: the lookback period used to
        compute the SMA
        :type sma_lookback_period: int
        """
        # Calculate the SMA
        self._sma_lookback_period = sma_lookback_period
        self._sma = self._data_factory.get_sma(
            self._raw_data,
            self._sma_lookback_period
            )
        
        # Calculate the residuals
        self._residuals = self._data_factory.get_residuals_data(
            self._raw_data,
            self._sma
            )

        # Makes labels
        test_data, test_labels = self._preprocess_residuals()

        # Predict the residuals
        self._predicted_residuals, self._predicted_residuals_std = self._ensemble_model.\
            predict_residuals(test_data)
        
        # Set the actual residuals to the test labels
        self._actual_residuals = test_labels

    def _extrapolate_sma(
            self
            ) -> None:
        """
        Extrapolates the SMA using the trend model.
        """
        # Gets the SMA data
        sma_data = self._data_factory.get_sma(self._raw_data, 3)

        # Generates sets
        processor = DataProcessor(data=None, unpack=False)

        sets = [sma_data[i:i + 10] for i in range(len(sma_data) - 10 + 1)]

        # Creates labels
        test_data, test_labels = processor.generate_labels(sets, 1)

        # Predict the SMA
        self._extrapolated_sma, self._extrapolated_sma_std = self._ensemble_model.\
            predict_sma(test_data)
        
        # Set the actual SMA to the test labels
        self._actual_sma = test_labels
        
    def _preprocess_residuals(self) -> tf.Tensor:
        """
        This method ensures that there are enough datapoints to fit
        the input shape of the model. After that it reduces the number
        of residuals to fit the input shape. Utilises tensorflow's
        convert_to_tensor method to create a tensor in the right shape
        for the prediciction method of the Model.

        :raises ValueError: if the number of datapoints is smaller
        than the input shape
        :return: resiudals in the form of a tensor object ready to
        be used as an input of a NN Model.
        :rtype: tf.Tensor
        """
        # Generates sets
        processor = DataProcessor(data=None, unpack=False)

        sets = [self._residuals[i:i + 10] for i in range(len(self._residuals) - 10 + 1)]

        # Generates labels
        test_data, test_labels = processor.generate_labels(sets, 1)
        
        return test_data, test_labels
    
    def _get_raw_data(
            self,
            start_date: int,
            end_date: str,
            interval: str
        ) -> None:
        """
        Helper method to get raw data used for plotting
        utilising the DataFactory.

        :param start_date: the start date of the data retrieval
        :type start_date: str
        :param end_date: the end date of the data retrieval
        :type end_date: str
        :param interval: the scale of the candlesticks
        :type interval: str
        """
        self._start_date = start_date
        self._end_date = end_date
        self._interval = interval
        self._raw_data = self._data_factory.get_raw_data(
            start_date,
            end_date,
            interval
        )

    def _validate_predictions(self, predictions: Any) -> None:
        """
        Validates if predictions have been made.

        :param predictions: the predictions
        :type predictions: Any
        :raises ValueError: if predictions haven't been made
        """
        if predictions is None:
            raise ValueError(
                "You need to make predictions first!"
                "Please first use the `predict` method.")
