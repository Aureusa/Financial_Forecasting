from naive_model.mlp import Model
from naive_model.lstm import LstmModel
from naive_model.network_constructor import NetworksConstructor
from data_parser.dataFactory import StockDataFactory
from forecast.forecastFactoryEnsemble import ForcastFactoryEnsemble

def train_model(
        stockCode: str = "AAPL",
        pointsPerSet: int = 10,
        labelsPerSet: int = 1,
        maxEpochs: int = 50,
        start_date: str = "2021-09-01",
        end_date: str = "2024-09-01",
        paramList_LSTM: list[tuple]|None = None,
        paramList_MLP: list[tuple]|None = None,
        mlp_model_foldername: str = "mlp__foldername_placeholder",
        mlp_model_filename: str = "mlp_filename_placeholder",
        lstm_model_foldername: str = "lstm__foldername_placeholder",
        lstm_model_filename: str = "lstm_filename_placeholder"
    ) -> dict:
    # Set the param lists
    if paramList_LSTM is None:
        paramList_LSTM = tuple(([9,18], 0.0025, 1))
    if paramList_MLP is None:
        paramList_MLP = tuple(([8,10], 0.0050, 1))

    # Create a StockDataFactory
    dataFactory = StockDataFactory(
        stockCode,
        pointsPerSet,
        labelsPerSet
    )
    
    # Get the training data for the mlp
    (
        training_data_mlp,
        training_labels_mlp
    ) = dataFactory.get_training_data(start_date=start_date, end_date=end_date, sma_data=False)

    # Get the training data for the lstm
    (
        training_data_lstm,
        training_labels_lstm
    ) = dataFactory.get_training_data(start_date=start_date, end_date=end_date, sma_data=True)
    
    # Construct a network
    input_size = len(training_data_mlp[0])
    output_size = len(training_labels_mlp[0])

    # Instantaite MLP Constructor
    netConst_mlp = NetworksConstructor(
        Model,
        input_size,
        output_size,
        maxEpochs
    )

    # Instantaite LSTM Constructor
    netConst_lstm = NetworksConstructor(
        LstmModel,
        input_size,
        output_size,
        maxEpochs
    )
    
    # Train and save MLP model
    netConst_mlp.train_model(
        training_data_mlp,
        training_labels_mlp,
        paramList_MLP,
        mlp_model_filename,
        mlp_model_foldername,
        )
    
    print("[***** MLP model trained successfully *****]")
    
    # Train and save LSTM model
    netConst_lstm.train_model(
        training_data_lstm,
        training_labels_lstm,
        paramList_LSTM,
        lstm_model_filename,
        lstm_model_foldername,
        )
    
    print("[***** LSTM model trained successfully *****]")

    msg = f"|| Successfully trained ensemble model on `{stockCode}` data ||"
    border = len(msg) * "="
    message = border + "\n" + msg + "\n" + border
    print(message)

def test_model(
        stock_name: str = "AAPL",
        start_date: str = "2024-09-01",
        end_date: str = "2025-02-11",
        residual_model="residual_model_model.keras",
        residual_model_folder="residual_model_w_dropout",
        trend_model="trend_model_model.keras",
        trend_model_folder="trend_model_w_dropout",
        interval: str = "1d",
        pointsPerSet: int = 10,
        labelsPerSet: int = 1,
        bollinger_band: bool = False,
        fig_save: bool = False
        ):
    """
    Tests the Ensemble Model.

    :param stock_name: the stock code, defaults to "AAPL"
    :type stock_name: str, optional
    :param raw_data_amount: the amount of raw data to retrieve,
    defaults to 90
    :type raw_data_amount: int, optional
    :param sma_lookback_period: the lookback time for the SMA,
    defaults to 3
    :type sma_lookback_period: int, optional
    :param end_date: the end date, defaults to "2025-01-15"
    :type end_date: str, optional
    :param interval: the candlestick interval, defaults to "1d"
    :type interval: str, optional
    :param points_per_set: points per set, defaults to 10
    :type points_per_set: int, optional
    :param num_sets: number of sets, defaults to 50
    :type num_sets: int, optional
    :param labels_per_set: labels per set, defaults to 1
    :type labels_per_set: int, optional
    :param training_percentage: trainign percentage, defaults to 0.8
    :type training_percentage: float, optional
    :param validation_percentage: validation percentage, defaults to 0.1
    :type validation_percentage: float, optional
    """
    # Create a forecast factory
    forcaster = ForcastFactoryEnsemble(
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
