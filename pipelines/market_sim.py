import os

from market.dynamic import TradingEnv


def roll_the_market(
        stock_codes: list[str] = ["TSLA", "MSFT", "AMZN", "AAPL"],
        start_date: str = "2012-01-01",
        end_date: str = "2025-03-01",
        points_per_set: int = 3,
        labels_per_set: int = 1,
        lookback: int = 3,
        initial_balance: int = 10000,
        alpha: float = 0.0,
        data_folder: str = os.path.join(os.getcwd(), "experiments", "market_experiment2"),
        filename: str = "final.pkl"
    ):
    """
    Rolls the market simulation with the given parameters.
    Chose the stock codes for the market you want to simulate.
    There are hardcoded hyperparmeters for the simulation,
    but you can change them if you want to. Just navigate to
    market/dynamic.py and change the parameters in the
    TradingEnv class. They control how frequently the
    checkpoints are saved, how often the model is retrained,
    and how many steps the simulation runs for.

    :param stock_codes: List of stock codes to simulate.
    :type stock_codes: list[str]
    :param start_date: Start date for the simulation.
    :type start_date: str
    :param end_date: End date for the simulation.
    :type end_date: str
    :param points_per_set: Number of data points per set, defaults to 3.
    :type points_per_set: int
    :param labels_per_set: Number of labels per set, defaults to 1.
    :type labels_per_set: int
    :param lookback: Lookback period for the SMA, defaults to 3.
    :type lookback: int
    :param initial_balance: Initial balance for the trading environment, defaults to 10000.
    :type initial_balance: int
    :param alpha: Alpha value for the trading strategy, defaults to 0.0.
    :type alpha: float
    :param data_folder: Folder to save the experiment data, defaults to "experiments/market_experiment2".
    :type data_folder: str
    :param filename: Name of the file to save the final state, defaults to "final.pkl".
    :type filename: str
    """
    env = TradingEnv(
        stock_codes=stock_codes,
        start_date=start_date,
        end_date=end_date,
        points_per_set=points_per_set,
        labels_per_set=labels_per_set,
        lookback=lookback,
        initial_balance=initial_balance,
        alpha=alpha,
        data_folder=data_folder
    )

    done = False

    while not done:
        done = env.roll_the_market()
    
    env.save_state(filename)
