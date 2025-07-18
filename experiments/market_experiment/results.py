import os

from utils import load_pkl, compute_success_rate, plot_portfolio, plot_actual_vs_predicted


def show_experiment_results():
    # Rebase the path to the current script's directory
    file_dir = os.path.dirname(os.path.abspath(__file__))
    current_dir = os.getcwd() # Get the current working directory

    os.chdir(file_dir)  # Change to the directory of this script

    # Load the states for different alpha values
    state_0 = load_pkl("checkpoint_1000_alpha_0.pkl")
    state_25 = load_pkl("checkpoint_1500_alpha_25.pkl")
    state_50 = load_pkl("checkpoint_1500_alpha_50.pkl")
    state_75 = load_pkl("checkpoint_1500_alpha_75.pkl")

    # Extracting directional data from the states
    dir_predictions_0 = state_0["dir_predictions"] 
    dir_predictions_25 = state_25["dir_predictions"] 
    dir_predictions_50 = state_50["dir_predictions"] 
    dir_predictions_75 = state_75["dir_predictions"] 

    # Extracting ground truth data
    dir_ground_truth_0 = state_0["dir_ground_truth"]
    dir_ground_truth_25 = state_25["dir_ground_truth"]
    dir_ground_truth_50 = state_50["dir_ground_truth"]
    dir_ground_truth_75 = state_75["dir_ground_truth"]
    
    # Extracting portfolio data
    portfolio_0 = state_0["portfolio"]
    portfolio_25 = state_25["portfolio"]
    portfolio_50 = state_50["portfolio"]
    portfolio_75 = state_75["portfolio"]

    # Remove the first element from each portfolio
    # This is often done to remove an initial state or placeholder
    portfolio_0.pop(0)
    portfolio_25.pop(0)
    portfolio_50.pop(0)
    portfolio_75.pop(0)

    # Extracting trading dates
    trading_dates_0 = state_0["trading_dates"]
    trading_dates_25 = state_25["trading_dates"]
    trading_dates_50 = state_50["trading_dates"]
    trading_dates_75 = state_75["trading_dates"]

    # Print trading dates for verification
    print(trading_dates_0["TSLA"][100], trading_dates_0["TSLA"][-1])
    print(trading_dates_25["TSLA"][100], trading_dates_25["TSLA"][-1])
    print(trading_dates_50["TSLA"][100], trading_dates_50["TSLA"][-1])
    print(trading_dates_75["TSLA"][100], trading_dates_75["TSLA"][-1])

    # Extracting predicted closing prices
    predicted_closing_prices_0 = state_0["predicted_closing_prices"]
    predicted_closing_prices_25 = state_25["predicted_closing_prices"]
    predicted_closing_prices_50 = state_50["predicted_closing_prices"]
    predicted_closing_prices_75 = state_75["predicted_closing_prices"]

    # Extracting real closing prices
    real_closing_prices_0 = state_0["real_closing_prices"]
    real_closing_prices_25 = state_25["real_closing_prices"]
    real_closing_prices_50 = state_50["real_closing_prices"]
    real_closing_prices_75 = state_75["real_closing_prices"]

    # Compute success rates for each alpha value
    # This is a measure of how often the predictions match the ground truth
    success_rate_0 = compute_success_rate(dir_predictions_0, dir_ground_truth_0)
    success_rate_25 = compute_success_rate(dir_predictions_25, dir_ground_truth_25)
    success_rate_50 = compute_success_rate(dir_predictions_50, dir_ground_truth_50)
    success_rate_75 = compute_success_rate(dir_predictions_75, dir_ground_truth_75)

    # Print success rates for each alpha value
    print(f"Success rate for alpha=0: {success_rate_0}")
    print(f"Success rate for alpha=0.25: {success_rate_25}")
    print(f"Success rate for alpha=0.50: {success_rate_50}")
    print(f"Success rate for alpha=0.75: {success_rate_75}")

    # Plot the portfolio values over time for each alpha value
    plot_portfolio(trading_dates_0["TSLA"], portfolio_0, alpha=0)
    plot_portfolio(trading_dates_25["TSLA"], portfolio_25, alpha=0.25)
    plot_portfolio(trading_dates_50["TSLA"], portfolio_50, alpha=0.50)
    plot_portfolio(trading_dates_75["TSLA"], portfolio_75, alpha=0.75)

    # Plot actual vs predicted closing prices for each alpha value
    plot_actual_vs_predicted(trading_dates_0["TSLA"], real_closing_prices_0, predicted_closing_prices_0, alpha=0)
    plot_actual_vs_predicted(trading_dates_25["TSLA"], real_closing_prices_25, predicted_closing_prices_25, alpha=0.25)
    plot_actual_vs_predicted(trading_dates_50["TSLA"], real_closing_prices_50, predicted_closing_prices_50, alpha=0.50)
    plot_actual_vs_predicted(trading_dates_75["TSLA"], real_closing_prices_75, predicted_closing_prices_75, alpha=0.75)

    # Change back to the original directory
    os.chdir(current_dir)
    