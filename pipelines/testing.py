from forecast.forecastFactoryEnsemble import ForecastFactoryEnsemble


def test_model(
        stock_name: str = "AAPL",
        start_date: str = "2024-09-01",
        end_date: str = "2025-02-11",
        residual_model="residual_model_model.keras",
        residual_model_folder="residual_model_w_dropout",
        trend_model="trend_model_model.keras",
        trend_model_folder="trend_model_w_dropout",
        interval: str = "1d",
        sma_lookback_period: int = 3,
        pointsPerSet: int = 10,
        labelsPerSet: int = 1,
        bollinger_band: bool = False,
        fig_save: bool = False
        ):
    """
    Tests the model by making predictions and comparing them with actual data.

    The default arguments for this function are set in such a way to
    maximize the performance of the models which was empirically verified.

    :param stock_name: the stock name to test, defaults to "AAPL"
    :type stock_name: str
    :param start_date: the start date for the data, defaults to "2024-09-01"
    :type start_date: str
    :param end_date: the end date for the data, defaults to "2025-02-11"
    :type end_date: str
    :param residual_model: the filename of the residual model, defaults to "residual_model_model.keras"
    :type residual_model: str
    :param residual_model_folder: the folder name for the residual model, defaults to "residual_model_w_dropout"
    :type residual_model_folder: str
    :param trend_model: the filename of the trend model, defaults to "trend_model_model.keras"
    :type trend_model: str
    :param trend_model_folder: the folder name for the trend model, defaults to "trend_model_w_dropout"
    :type trend_model_folder: str
    :param interval: the interval for the data, defaults to "1d"
    :type interval: str
    :param sma_lookback_period: the lookback period for the SMA, defaults to 3
    :type sma_lookback_period: int
    :param pointsPerSet: the number of points per set, defaults to 10
    :type pointsPerSet: int
    :param labelsPerSet: the number of labels per set, defaults to 1
    :type labelsPerSet: int
    :param bollinger_band: whether to use Bollinger Bands, defaults to False
    :type bollinger_band: bool
    :param fig_save: whether to save the figure, defaults to False
    :type fig_save: bool
    :return: a tuple containing actual closing prices, predicted closing prices,
                directional array, sigma, mean absolute error, direction success rate,
                and range match success rate
    :rtype: tuple
    """
    # Create a forecast factory
    forcaster = ForecastFactoryEnsemble(
        stock_name=stock_name,
        residual_model=residual_model,
        residual_model_folder=residual_model_folder,
        trend_model=trend_model,
        trend_model_folder=trend_model_folder,
        pointsPerSet=pointsPerSet,
        labelsPerSet=labelsPerSet
        )

    # Make the forecast
    forcaster.predict(
        start_date=start_date,
        end_date=end_date,
        sma_lookback_period=sma_lookback_period,
        interval=interval
    )
    
    (
        actual_closing_prices,
        predicted_closing_prices,
        directional_arr,
        sigma,
        mae,
        direction_success_rate,
        range_match_success_rate
    ) = forcaster.compare_predictions_with_observations()


    forcaster.make_comparison_plot(
        bollinger_band=bollinger_band,
        stock_name=stock_name,
        save=fig_save
    )
    
    print("MAE:", round(mae,2))
    print("DIRECTION SUCCESS RATE:", round(direction_success_rate,2))
    print("RANGE SUCCESS RATE:", round(range_match_success_rate,2))

    return (
        actual_closing_prices,
        predicted_closing_prices,
        directional_arr,
        sigma,
        mae,
        direction_success_rate,
        range_match_success_rate
    )
