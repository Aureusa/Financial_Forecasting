import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle
from datetime import datetime
from typing import Any


def save_pkl(data, filename: str) -> None:
    """
    Saves data to a pickle file.

    :param data: Data to be saved.
    :type data: Any
    :param filename: The name of the file where data will be saved.
    :type filename: str
    """
    if not os.path.exists(filename):
        os.makedirs(filename)
        
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

    print(f"Data saved to {filename}")

def load_pkl(filename: str) -> Any:
    """
    Loads data from a pickle file.

    :param filename: The name of the file from which data will be loaded.
    :type filename: str
    :return: The data loaded from the file.
    :rtype: Any
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    print(f"Data loaded from {filename}")
    return data

def plot_portfolio(
        dates: list,
        portfolio: list,
        alpha: float
    ) -> None:
    """
    Plots the portfolio value over time of the market simulation.

    :param dates: List of dates corresponding to the portfolio values.
    :type dates: list of str
    :param portfolio: List of portfolio values.
    :type portfolio: list of float
    :param alpha: Alpha value used in the model, for labeling purposes.
    :type alpha: float
    """
    # Convert string dates to datetime objects
    dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
    
    plt.figure(figsize=(10, 5))
    plt.plot(dates, portfolio, label='Portfolio Value')
    plt.xlabel('Dates')
    plt.ylabel('Portfolio Value')
    plt.title('Portfolio Value Over Time (Alpha = {})'.format(alpha))
    plt.legend()
    plt.grid(True)
    
    # Format the x-axis to show only the month
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_actual_vs_predicted(
        dates: list,
        actual_prices_dict: dict,
        predicted_prices_dict: dict,
        alpha: float
    ) -> None:
    """
    Plots actual vs predicted prices for multiple stocks over time.
    Typically used to visualize the performance of the forecasting model
    on the market data.

    :param dates: List of dates corresponding to the stock prices.
    :type dates: list of str
    :param actual_prices_dict: Dictionary containing actual stock prices. Usually
    structured as {stock_name: [prices]}.
    :type actual_prices_dict: dict
    :param predicted_prices_dict: Dictionary containing predicted stock prices. Usually
    structured as {stock_name: [predicted_prices]}.
    :type predicted_prices_dict: dict
    :param alpha: Alpha value used in the model, for labeling purposes.
    :type alpha: float
    """
    # Convert string dates to datetime objects
    dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
    
    plt.figure(figsize=(10, 5))
    
    for stock in actual_prices_dict.keys():
        actual_prices = actual_prices_dict[stock]
        predicted_prices = predicted_prices_dict[stock]
        
        plt.plot(dates, actual_prices, label=f'{stock} Actual')
        plt.plot(dates, predicted_prices, label=f'{stock} Predicted', linestyle='--')
    
    plt.xlabel('Dates')
    plt.ylabel('Prices')
    plt.title('Actual vs Predicted Prices Over Time (Alpha = {})'.format(alpha))
    plt.legend()
    plt.grid(True)
    
    # Format the x-axis to show only the month
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def compute_success_rate(
        dir_predictions: dict,
        dir_ground_truth: dict
    ) -> float:
    """
    Computes the success rate of predictions against ground truth.
    
    :param dir_predictions: Dictionary of predicted stock directions.
    :type dir_predictions: dict
    :param dir_ground_truth: Dictionary of actual stock directions.
    :type dir_ground_truth: dict
    :return: Success rate as a float.
    :rtype: float
    """
    total_predictions = 0
    correct_predictions = 0
    
    for stock in dir_predictions.keys():
        predicted_directions = dir_predictions[stock]
        actual_directions = dir_ground_truth[stock]
        
        for pred, actual in zip(predicted_directions, actual_directions):
            total_predictions += 1
            if pred == actual:
                correct_predictions += 1
    
    success_rate = correct_predictions / total_predictions if total_predictions > 0 else 0
    return success_rate
