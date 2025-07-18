from sklearn.metrics import mean_absolute_error
import numpy as np
from typing import Any

from data_pipeline import DataPipeline
from forecast.ensembleModel import EnsembleModel
from visualisation.visualize_simple import Plotter


class ForecastFactoryEnsemble:
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
        A way of instantiating ForecastFactoryEnsemble.

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

        self._pointsPerSet = pointsPerSet
        self._labelsPerSet = labelsPerSet

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
        Predicts the closing prices of the stock using the ensemble model.

        :param start_date: the start date of the data retrieval
        :type start_date: str
        :param end_date: the end date of the data retrieval
        :type end_date: str
        :param sma_lookback_period: the lookback period used to compute the SMA
        :type sma_lookback_period: int
        :param interval: the scale of the candlesticks
        :type interval: str
        """
        # Initialize the DataPipeline
        self._data_pipeline = DataPipeline(
            self._stock_name,
            start_date,
            end_date,
            interval
        )

        self._raw_data = self._data_pipeline.get_raw_data()

        self._predict_residuals(sma_lookback_period)

        self._extrapolate_sma()

        self._predicted_closing_prices, self._predicted_closing_prices_std = self._calculate_predicted_closing_prices()

    def compare_predictions_with_observations(self) -> tuple[list[float], list[float], np.ndarray, list[float], float, float, float]:
        """
        Compares the predicted closing prices with the actual closing prices.

        :return: a tuple containing the actual closing prices, predicted closing prices,
        directional array, sigma, mean absolute error, direction success rate, and range match success rate
        :rtype: tuple[list[float], list[float], np.ndarray, list[float], float, float, float]
        """
        self._validate_predictions(self._predicted_closing_prices)

        actual_closing_prices = self._calculate_actual_closing_prices()

        predicted_closing_prices = self._predicted_closing_prices

        direction_success_rate, directional_arr = self.calculate_direction_success_rate(predicted_closing_prices, actual_closing_prices)

        # sigma = Standart deviation of the model predictions
        range_match_success_rate, sigma = self._calculate_range_match_success_rate(predicted_closing_prices, actual_closing_prices)
        
        # Calculate Mean Absolute Error (MAE)
        mae = mean_absolute_error(
            actual_closing_prices,
            predicted_closing_prices
            )
        
        return actual_closing_prices, predicted_closing_prices, directional_arr, sigma, mae, direction_success_rate, range_match_success_rate
    
    def make_comparison_plot(self, bollinger_band: bool, stock_name: str, save: bool):
        """
        Makes a comparison plot of the predicted closing prices and the actual closing prices.
        
        :param bollinger_band: whether to include the Bollinger band in the plot
        :type bollinger_band: bool
        :param stock_name: the name of the stock
        :type stock_name: str
        :param save: whether to save the plot
        :type save: bool
        """
        closing_prices = self._calculate_actual_closing_prices()

        dates = [t[0] for t in self._raw_data]

        dates = dates[-len(closing_prices):]

        plotter = Plotter(closing_prices, self._predicted_closing_prices, dates)

        plotter.comparison_plot(
            predictions_std=self._predicted_closing_prices_std,
            bollinger_band=bollinger_band,
            stock_name=stock_name,
            save=save
        )

    def calculate_direction_success_rate(
            self,
            predicted_closing_prices: list[float],
            actual_closing_prices: list[float]
        ) -> tuple[float, np.ndarray]:
        """
        Calculates the success rate of the direction predictions.

        :param predicted_closing_prices: the predicted closing prices
        :type predicted_closing_prices: list[float]
        :param actual_closing_prices: the actual closing prices
        :type actual_closing_prices: list[float]
        :return: a tuple containing the success rate and the directional array
        :rtype: tuple[float, np.ndarray]
        """
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

        # Get an array with directions filled with 1 (UP) -1 (DOWN)
        directional_arr = np.where(predictions, 1, -1)

        return success_rate, directional_arr
    
    def _calculate_range_match_success_rate(
            self,
            predicted_closing_prices: list[float],
            actual_closing_prices: list[float]
        ) -> tuple[float, list[float]]:
        """
        Calculates the success rate of the range match predictions.

        :param predicted_closing_prices: the predicted closing prices
        :type predicted_closing_prices: list[float]
        :param actual_closing_prices: the actual closing prices
        :type actual_closing_prices: list[float]
        :return: a tuple containing the success rate and the sigma
        :rtype: tuple[float, list[float]]
        """
        sigma = np.array(self._predicted_closing_prices_std).T[0]

        upper_bound = np.array(predicted_closing_prices).T[0] + 3 * sigma
        lower_bound = np.array(predicted_closing_prices).T[0] - 3 * sigma

        condition = (actual_closing_prices < upper_bound) & (actual_closing_prices > lower_bound)

        success_rate = np.mean(condition) * 100

        return success_rate, sigma.tolist()
    
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
        self._sma = self._data_pipeline.get_sma(sma_lookback_period)
        
        # Calculate the residuals
        self._residuals = self._data_pipeline.get_residuals_data(
            self._sma,
            sma_lookback_period
        )

        # Create sets
        sets = [self._residuals[i:i + self._pointsPerSet] for i in range(len(self._residuals) - self._pointsPerSet + 1)]

        # Generate labels
        test_data, test_labels = self._data_pipeline.generate_labels(
            sets,
            self._labelsPerSet,
        )

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
        if self._sma is None:
            self._sma = self._data_pipeline.get_sma(self._sma_lookback_period)

        # Generates sets
        sets = [self._sma[i:i + self._pointsPerSet] for i in range(len(self._sma) - self._pointsPerSet + 1)]

        # Creates labels
        test_data, test_labels = self._data_pipeline.generate_labels(
            sets,
            self._labelsPerSet
        )

        # Predict the SMA
        self._extrapolated_sma, self._extrapolated_sma_std = self._ensemble_model.\
            predict_sma(test_data)
        
        # Set the actual SMA to the test labels
        self._actual_sma = test_labels
        
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
