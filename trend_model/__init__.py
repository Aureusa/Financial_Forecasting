from data_parser.dataFactory import StockDataFactory

#####################################################################################
############################       TO DO LIST     ###################################
#####################################################################################
# 1. Normalise the data before training!!! (Crucial for training RNNs)
#       Simple example:
#               from sklearn.preprocessing import MinMaxScaler
#
#               scaler = MinMaxScaler(feature_range=(0, 1))
#               dataset = scaler.fit_transform(dataset)
# 2. Create and fit LSTM network.
#       Simple example:
#               from tensorflow.keras.models import Sequential
#               from tensorflow.keras.layers import Dense
#               from tensorflow.keras.layers import LSTM
#
#               model = Sequential()
#               model.add(LSTM(4, input_shape=(1, 1)))
#               model.add(Dense(1))
#               model.compile(loss='mean_squared_error', optimizer='adam')
#               model.fit(training_data, training_labels, epochs=100, batch_size=1, verbose=2)
# 3. Evaluate the LSTM network against a given metric, like RMSE.
#       Simple example:
#               from sklearn.metrics import mean_squared_error
#
#               trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
#               print('Train Score: %.2f RMSE' % (trainScore))
#               testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# 4. Repeat the previous 3 steps for different permutations of hyperparameter combination
# for hyperparameter tunning.


def get_training_data(
        stockCode: str = "AAPL",
        pointsPerSet: int = 10,
        numSets: int = 50,
        labelsPerSet: int = 1,
        testingPercentage: float = 0.8,
        validationPercentage: float = 0.1,
    ) -> tuple:
    """
    Generates and retrieves training, validation, and testing
    datasets along with their corresponding labels that are used for
    training an LSTM (RNN) on the Simple Moving Average data.
    
    Note from Peter:
    "I would keep the default setting of these unless you
    really, reaaaally want to change them for some reason."

    :param stockCode: The stock symbol or code for which the data is retrieved,
    defaults to "AAPL".
    :type stockCode: str
    :param pointsPerSet: Number of data points included in each set, defaults to 10.
    :type pointsPerSet: int, optional
    :param numSets: The total number of data sets to be generated, defaults to 50.
    :type numSets: int, optional
    :param labelsPerSet: Number of labels associated with each set, defaults to 1.
    :type labelsPerSet: int, optional
    :param testingPercentage: Proportion of data used for testing, defaults to 0.8.
    :type testingPercentage: float, optional
    :param validationPercentage: Proportion of data used for validation, defaults to 0.1.
    :type validationPercentage: float, optional
    :return: A tuple containing the training, validation, and testing datasets 
             along with their respective labels.
    :rtype: tuple
    """
    # Create a StockDataFactory
    dataFactory = StockDataFactory(
        stockCode,
        pointsPerSet,
        numSets,
        labelsPerSet,
        testingPercentage,
        validationPercentage
        )
    
    # Get the data from the data factory
    (
        training_data,
        validation_data,
        testing_data,
        training_labels,
        validation_labels,
        testing_labels
        ) = dataFactory.get_stock_data(sma_data=True)
    
    return (
        training_data,
        validation_data,
        testing_data,
        training_labels,
        validation_labels,
        testing_labels
    )