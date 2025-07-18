from .data import DataMaker
from .dynamic import TradingEnv
from .model import RollingModel


def roll_the_market(
        stock_codes: list[str] = ["TSLA", "MSFT", "AMZN", "AAPL"],
        start_date: str = "2012-01-01",
        end_date: str = "2025-03-01",
        points_per_set: int = 3,
        labels_per_set: int = 1,
        lookback: int = 3,
        initial_balance: int = 10000,
        filename: str = "og_state.pkl"
    ):
    env = TradingEnv(
        stock_codes=stock_codes,
        start_date=start_date,
        end_date=end_date,
        points_per_set=points_per_set,
        labels_per_set=labels_per_set,
        lookback=lookback,
        initial_balance=initial_balance
    )

    done = False

    while not done:
        done = env.roll_the_market()
    
    env.save_state(filename)