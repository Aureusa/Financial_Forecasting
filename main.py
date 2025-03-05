from market.dynamic import TradingEnv


if __name__ == "__main__":
    stock_codes = ["TSLA", "MSFT", "AMZN", "AAPL"]
    env = TradingEnv(
        stock_codes=stock_codes,
        start_date="2012-01-01",
        end_date="2025-03-01",
        points_per_set=3,
        labels_per_set=1,
        lookback=3,
        initial_balance=10000
    )

    done = False

    while not done:
        done = env.yolooo()
    
    env.save_state("og_state.pkl")
