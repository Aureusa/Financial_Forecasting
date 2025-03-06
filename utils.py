import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle
from datetime import datetime


def load_state(filename: str):
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        return state

def plot_portfolio(dates, portfolio):
    # Convert string dates to datetime objects
    dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
    
    plt.figure(figsize=(10, 5))
    plt.plot(dates, portfolio, label='Portfolio Value')
    plt.xlabel('Dates')
    plt.ylabel('Portfolio Value')
    plt.title('Portfolio Value Over Time')
    plt.legend()
    plt.grid(True)
    
    # Format the x-axis to show only the month
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_actual_vs_predicted(dates, actual_prices_dict, predicted_prices_dict):
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
    plt.title('Actual vs Predicted Prices Over Time')
    plt.legend()
    plt.grid(True)
    
    # Format the x-axis to show only the month
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def compute_success_rate(dir_predictions, dir_ground_truth):
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

    # train_model(stockCode="NVDA",start_date="2010-09-01", end_date="2024-12-23", pointsPerSet=3)
    # test_model(
    #     stock_name="NVDA",
    #     start_date ="2024-12-23",
    #     end_date = "2025-03-05",
    #     residual_model = "mlp_filename_placeholder_model.keras",
    #     residual_model_folder = "mlp__foldername_placeholder",
    #     trend_model = "lstm_filename_placeholder_model.keras",
    #     trend_model_folder = "lstm__foldername_placeholder",
    #     pointsPerSet = 3
    # )
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






    
# state_0 = load_state("checkpoint_1000_alpha_0.pkl")
    # state_25 = load_state("checkpoint_1000_alpha_25.0.pkl")
    # state_50 = load_state("checkpoint_1000_alpha_50.0.pkl")
    # state_75 = load_state("checkpoint_1000_alpha_75.0.pkl")

    # dir_predictions_0 = state_0["dir_predictions"] 
    # dir_predictions_25 = state_25["dir_predictions"] 
    # dir_predictions_50 = state_50["dir_predictions"] 
    # dir_predictions_75 = state_75["dir_predictions"] 

    # dir_ground_truth_0 = state_0["dir_ground_truth"]
    # dir_ground_truth_25 = state_25["dir_ground_truth"]
    # dir_ground_truth_50 = state_50["dir_ground_truth"]
    # dir_ground_truth_75 = state_75["dir_ground_truth"]

    # portfolio_0 = state_0["portfolio"]
    # portfolio_25 = state_25["portfolio"]
    # portfolio_50 = state_50["portfolio"]
    # portfolio_75 = state_75["portfolio"]

    # portfolio_0.pop(0)
    # portfolio_25.pop(0)
    # portfolio_50.pop(0)
    # portfolio_75.pop(0)

    # trading_dates_0 = state_0["trading_dates"]
    # trading_dates_25 = state_25["trading_dates"]
    # trading_dates_50 = state_50["trading_dates"]
    # trading_dates_75 = state_75["trading_dates"]

    # print(trading_dates_0["TSLA"][100], trading_dates_0["TSLA"][-1])
    # print(trading_dates_25["TSLA"][100], trading_dates_25["TSLA"][-1])
    # print(trading_dates_50["TSLA"][100], trading_dates_50["TSLA"][-1])
    # print(trading_dates_75["TSLA"][100], trading_dates_75["TSLA"][-1])

    # This section of the code is performing the following tasks:
    # predicted_closing_prices_0 = state_0["predicted_closing_prices"]
    # predicted_closing_prices_25 = state_25["predicted_closing_prices"]
    # predicted_closing_prices_50 = state_50["predicted_closing_prices"]
    # predicted_closing_prices_75 = state_75["predicted_closing_prices"]

    # real_closing_prices_0 = state_0["real_closing_prices"]
    # real_closing_prices_25 = state_25["real_closing_prices"]
    # real_closing_prices_50 = state_50["real_closing_prices"]
    # real_closing_prices_75 = state_75["real_closing_prices"]

    # success_rate_0 = compute_success_rate(dir_predictions_0, dir_ground_truth_0)
    # success_rate_25 = compute_success_rate(dir_predictions_25, dir_ground_truth_25)
    # success_rate_50 = compute_success_rate(dir_predictions_50, dir_ground_truth_50)
    # success_rate_75 = compute_success_rate(dir_predictions_75, dir_ground_truth_75)

    # print(f"Success rate for alpha=0: {success_rate_0}")
    # print(f"Success rate for alpha=0.25: {success_rate_25}")
    # print(f"Success rate for alpha=0.50: {success_rate_50}")
    # print(f"Success rate for alpha=0.75: {success_rate_75}")

    # plot_portfolio(trading_dates_0["TSLA"], portfolio_0)
    # plot_portfolio(trading_dates_25["TSLA"], portfolio_25)
    # plot_portfolio(trading_dates_50["TSLA"], portfolio_50)
    # plot_portfolio(trading_dates_75["TSLA"], portfolio_75)

    # plot_actual_vs_predicted(trading_dates_0["TSLA"], real_closing_prices_0, predicted_closing_prices_0)
    # plot_actual_vs_predicted(trading_dates_25["TSLA"], real_closing_prices_25, predicted_closing_prices_25)
    # plot_actual_vs_predicted(trading_dates_50["TSLA"], real_closing_prices_50, predicted_closing_prices_50)
    # plot_actual_vs_predicted(trading_dates_75["TSLA"], real_closing_prices_75, predicted_closing_prices_75)