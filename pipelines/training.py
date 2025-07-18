from naive_model import Model, LstmModel, NetworksConstructor
from data_pipeline import DataPipeline


def train_model(
        stockCode: str = "AAPL",
        pointsPerSet: int = 10,
        labelsPerSet: int = 1,
        maxEpochs: int = 50,
        start_date: str = "2021-09-01",
        end_date: str = "2024-09-01",
        interval: str = "1d",
        sma_lookback: int = 3,
        paramList_LSTM: list[tuple]|None = None,
        paramList_MLP: list[tuple]|None = None,
        mlp_model_foldername: str = "mlp__foldername_placeholder",
        mlp_model_filename: str = "mlp_filename_placeholder",
        lstm_model_foldername: str = "lstm__foldername_placeholder",
        lstm_model_filename: str = "lstm_filename_placeholder"
    ) -> dict:
    """
    Trains the MLP and LSTM models. Sets a default for the parameters
    if they are not provided. The default parameters are:
    - MLP: ([8, 10], 0.0050, 1)
    - LSTM: ([9, 18], 0.0025, 1)
    (Check the `NetworksConstructor` class for more details)

    The default arguments for this function are set in such a way to
    maximize the performance of the models which was empirically verified.

    :param stockCode: the stock code, defaults to "AAPL"
    :type stockCode: str, optional
    :param pointsPerSet: the number of points per set, defaults to 10
    :type pointsPerSet: int, optional
    :param labelsPerSet: the number of labels per set, defaults to 1
    :type labelsPerSet: int, optional
    :param maxEpochs: the maximum number of epochs, defaults to 50
    :type maxEpochs: int, optional
    :param start_date: the start date for the data, defaults to "2021-09-01"
    :type start_date: str, optional
    :param end_date: the end date for the data, defaults to "2024-09-01"
    :type end_date: str, optional
    :param interval: the interval for the data, defaults to "1d"
    :type interval: str, optional
    :param sma_lookback: the lookback period for the SMA, defaults to 3
    :type sma_lookback: int, optional
    :param paramList_LSTM: the parameters for the LSTM model, defaults to None
    :type paramList_LSTM: list[tuple], optional
    :param paramList_MLP: the parameters for the MLP model, defaults to None
    :type paramList_MLP: list[tuple], optional
    :param mlp_model_foldername: the folder name for the MLP model, defaults to "mlp__foldername_placeholder"
    :type mlp_model_foldername: str, optional
    :param mlp_model_filename: the filename for the MLP model, defaults to "mlp_filename_placeholder"
    :type mlp_model_filename: str, optional
    :param lstm_model_foldername: the folder name for the LSTM model, defaults to "lstm__foldername_placeholder"
    :type lstm_model_foldername: str, optional
    :param lstm_model_filename: the filename for the LSTM model, defaults to "lstm_filename_placeholder"
    :type lstm_model_filename: str, optional
    :return: a dictionary containing the trained models
    :rtype: dict
    """
    # Set the param lists
    if paramList_LSTM is None:
        paramList_LSTM = tuple(([9,18], 0.0025, 1))
    if paramList_MLP is None:
        paramList_MLP = tuple(([8,10], 0.0050, 1))
    
    dataPipeline = DataPipeline(
        name=stockCode,
        start_date=start_date,
        end_date=end_date,
        interval=interval
    )

    # Get the training data for the mlp
    (
        training_data_mlp,
        training_labels_mlp
    ) = dataPipeline.get_training_data(
        pointsPerSet=pointsPerSet,
        labelsPerSet=labelsPerSet,
        sma_data=False,
        sma_lookback=sma_lookback,
    )

    # Get the training data for the lstm
    (
        training_data_lstm,
        training_labels_lstm
    ) = dataPipeline.get_training_data(
        pointsPerSet=pointsPerSet,
        labelsPerSet=labelsPerSet,
        sma_data=True,
        sma_lookback=sma_lookback,
    )
    
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
