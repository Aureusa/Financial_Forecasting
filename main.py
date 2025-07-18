from market.dynamic import TradingEnv
from utils import load_pkl, compute_success_rate, plot_portfolio, plot_actual_vs_predicted
from pipelines import test_model, train_model


if __name__ == "__main__":
    train_model(
        labelsPerSet=1,
        pointsPerSet=3,
        stockCode="AAPL",
    )
    test_model(
        labelsPerSet=1,
        pointsPerSet=3,
        stock_name="AAPL",
        residual_model_folder = "mlp__foldername_placeholder",
        trend_model_folder = "lstm__foldername_placeholder",
        residual_model="mlp_filename_placeholder_model.keras",
        trend_model="lstm_filename_placeholder_model.keras"
    )

    # stock_codes = ["TSLA", "MSFT", "AMZN", "AAPL"]
    # env = TradingEnv(
    #     stock_codes=stock_codes,
    #     start_date="2012-01-01",
    #     end_date="2025-03-01",
    #     points_per_set=3,
    #     labels_per_set=1,
    #     lookback=3,
    #     initial_balance=10000
    # )

    # done = False

    # while not done:
    #     done = env.roll_the_market()
    
    # env.save_state("og_state.pkl")
