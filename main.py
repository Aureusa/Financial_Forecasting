import numpy as np
import pandas as pd

from market.stock_environment import TradingEnv
from naive_model import test_model, train_model
from sophisticated_model import train_sophisticated_model


if __name__ == "__main__":
    print("ss")
    #  stock_codes = ["ADA-USD", "TSLA", "MSFT", "AMZN"]

    # for stock in stock_codes:
    #     train_model(
    #         stockCode=stock,
    #         mlp_model_foldername = "stock_bundle",
    #         mlp_model_filename = f"mlp_{stock}",
    #         lstm_model_foldername = "stock_bundle",
    #         lstm_model_filename = f"lstm_{stock}",
    #         pointsPerSet=3
    #     )

    # asset_name = ["BTC-USD", "ADA-USD", "TSLA", "MSFT", "AMZN", "AAPL"]
    # closing_prices = []
    # directions = []
    # uncertainties = []

    # (
    #     actual_closing_prices,
    #     predicted_closing_prices,
    #     directional_arr,
    #     sigma,
    #     mae,
    #     direction_success_rate,
    #     range_match_success_rate
    # ) = test_model(
    #     stock_name="BTC-USD",
    #     residual_model="btc_model.keras",
    #     residual_model_folder="btc_mlp",
    #     trend_model="btc_model.keras",
    #     trend_model_folder="btc_lstm",
    #     pointsPerSet=3,
    #     fig_save=True
    # )

    # closing_prices.append(actual_closing_prices[:-1])
    # directions.append(directional_arr.tolist())
    # uncertainties.append(sigma)

    # for stock in stock_codes:
    #     (
    #         actual_closing_prices,
    #         predicted_closing_prices,
    #         directional_arr,
    #         sigma,
    #         mae,
    #         direction_success_rate,
    #         range_match_success_rate
    #     ) = test_model(
    #         stock_name=stock,
    #         pointsPerSet=3,
    #         residual_model=f"mlp_{stock}_model.keras",
    #         residual_model_folder="stock_bundle",
    #         trend_model=f"lstm_{stock}_model.keras",
    #         trend_model_folder="stock_bundle",
    #         fig_save=True
    #     )

    #     closing_prices.append(actual_closing_prices[:-1])
    #     directions.append(directional_arr.tolist())
    #     uncertainties.append(sigma)

    # (
    #     actual_closing_prices,
    #     predicted_closing_prices,
    #     directional_arr,
    #     sigma,
    #     mae,
    #     direction_success_rate,
    #     range_match_success_rate
    # ) = test_model(
    #     stock_name="AAPL",
    #     residual_model="mlp_filename_placeholder_model.keras",
    #     residual_model_folder="mlp_80_percent_model",
    #     trend_model="lstm_filename_placeholder_model.keras",
    #     trend_model_folder="lstm_80_percent_model",
    #     pointsPerSet=3,
    #     fig_save=True
    # )

    # closing_prices.append(actual_closing_prices[:-1])
    # directions.append(directional_arr.tolist())
    # uncertainties.append(sigma)

    # df_stocks = pd.DataFrame({"Stocks": stock_codes})
    # df_close = pd.DataFrame({"Close": closing_prices})
    # df_dir = pd.DataFrame({"Direction": directions})
    # df_unc = pd.DataFrame({"Uncertainties": uncertainties})

    # df_stocks.to_pickle('df_stocks.pkl')
    # df_close.to_pickle('df_close.pkl')
    # df_dir.to_pickle('df_dir.pkl')
    # df_unc.to_pickle('df_unc.pkl')
    
    # df_stocks = pd.DataFrame({"Stocks": ["BTC-USD", "ADA-USD", "TSLA", "MSFT", "AMZN", "AAPL"]})
    # df_close = pd.read_pickle('df_close.pkl')
    # df_dir = pd.read_pickle('df_dir.pkl')
    # df_unc = pd.read_pickle('df_unc.pkl')

    # data = tuple((df_close, df_stocks, df_dir, df_unc))

    # env = TradingEnv(data, 10000)  # Create the environment

    # obs, _ = env.reset()
    # done = False

    # while not done:
    #     obs, reward, done, truncated, _ = env.step(1, 1000)
    #     env.render()

    # env.plot_portfolio()


