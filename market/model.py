import numpy as np

from naive_model import Model
from naive_model import LstmModel


class RollingModel:
    """
    This class provides a way to test the forcasting abilities of an
    Ensemble model. It uses a rolling approach to train the MLP and LSTM models
    on the provided training data and then predicts the closing prices using the
    trained models. The class also provides a method to compute the rolling weights
    for the new model based on the old model's weights using a Simple Moving Average (SMA) approach.
    """
    def train(
            self,
            stockCode: str,
            training_data_mlp: np.ndarray,
            training_labels_mlp: np.ndarray,
            training_data_lstm: np.ndarray,
            training_labels_lstm: np.ndarray,
            epochs: int = 50,
            alpha: float = 0.5,
            old_mlp_model: Model|None = None,
            old_lstm_model: LstmModel|None = None
        ) -> tuple[Model, LstmModel]:
        """
        Trains the MLP and LSTM models on the provided training data.

        :param stockCode: The stock code for which the models are trained
        :type stockCode: str
        :param training_data_mlp: The training data for the MLP model
        :type training_data_mlp: np.ndarray
        :param training_labels_mlp: The training labels for the MLP model
        :type training_labels_mlp: np.ndarray
        :param training_data_lstm: The training data for the LSTM model
        :type training_data_lstm: np.ndarray
        :param training_labels_lstm: The training labels for the LSTM model
        :type training_labels_lstm: np.ndarray
        :param epochs: The number of epochs to train the models
        :type epochs: int, optional
        :param alpha: The weight factor for the old model's weights
        :type alpha: float, optional
        :param old_mlp_model: The previously trained MLP model
        :type old_mlp_model: Model, optional
        :param old_lstm_model: The previously trained LSTM model
        :type old_lstm_model: LstmModel, optional
        :return: The trained MLP and LSTM models
        :rtype: tuple[Model, LstmModel]
        """
        # Set the param lists
        paramList_LSTM = tuple(([9,18], 0.0025, 1))
        paramList_MLP = tuple(([8,10], 0.0050, 1))
        
        # Construct a network
        input_size = len(training_data_mlp[0])
        output_size = len(training_labels_mlp[0])

        # Create a tuple with the parameter set and associated metadata (MLP)
        param_tuple_mlp = (
            paramList_MLP,                  # The current parameter set
            training_data_mlp,              # Training data
            training_labels_mlp,            # Training labels
        )

        # Create a tuple with the parameter set and associated metadata (LSTM)
        param_tuple_lstm = (
            paramList_LSTM,                  # The current parameter set
            training_data_lstm,              # Training data
            training_labels_lstm,            # Training labels
        )

        # Train MLP model
        mlp_model = self._train(
            model=Model,
            input_size=input_size,
            output_size=output_size,
            params=param_tuple_mlp,
            epochs=epochs
        )
        
        print("[***** MLP model trained successfully *****]")
        
        lstm_model = self._train(
            model=LstmModel,
            input_size=input_size,
            output_size=output_size,
            params=param_tuple_lstm,
            epochs=epochs
        )
        
        print("[***** LSTM model trained successfully *****]")


        if old_mlp_model is not None: # Compute the SMA of the weights of the MLP model
            mlp_model = self._compute_rolling_weights(
                old_model=old_mlp_model,
                new_model=mlp_model,
                alpha=alpha
            )

        if old_lstm_model is not None: # Compute the SMA of the weights of the LSTM model
            lstm_model = self._compute_rolling_weights(
                old_model=old_lstm_model,
                new_model=lstm_model,
                alpha=alpha
            )

        msg = f"|| Successfully trained ensemble model on `{stockCode}` data ||"
        border = len(msg) * "="
        message = border + "\n" + msg + "\n" + border
        print(message)

        return mlp_model, lstm_model

    def predict(
            self,
            mlp_model: Model,
            lstm_model: LstmModel,
            testing_data_mlp: np.ndarray,
            testing_data_lstm: np.ndarray,
            mc_realizations: int
            ) -> None:
        """
        Predicts the closing prices using the MLP and LSTM models.

        :param mlp_model: The trained MLP model
        :type mlp_model: Model
        :param lstm_model: The trained LSTM model
        :type lstm_model: LstmModel
        :param testing_data_mlp: The testing data for the MLP model
        :type testing_data_mlp: np.ndarray
        :param testing_data_lstm: The testing data for the LSTM model
        :type testing_data_lstm: np.ndarray
        :param mc_realizations: The number of Monte Carlo realizations
        :type mc_realizations: int
        :return: The predicted closing prices and their standard deviations
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        residual, residual_std = mlp_model.monte_carlo_predictions(
            data=testing_data_mlp,
            num_samples=mc_realizations
        )

        sma, sma_std = lstm_model.monte_carlo_predictions(
            data=testing_data_lstm,
            num_samples=mc_realizations
        )

        closing_price = sma + residual

        closing_prices_std = [
            (std[0]**2 + std[1]**2 ) ** 0.5
            for std in
            zip(
                residual_std,
                sma_std
            )
        ]

        closing_prices_std = np.array(closing_prices_std) / closing_price

        return closing_price, closing_prices_std

    def _compute_rolling_weights(self, old_model: Model, new_model: Model, alpha: float) -> Model:
        """
        Computes the rolling weights for the new model based on the old model's weights
        using a Simple Moving Average (SMA) approach.

        :param old_model: The previously trained model
        :type old_model: Model
        :param new_model: The newly trained model
        :type new_model: Model
        :param alpha: The weight factor for the old model's weights
        :type alpha: float
        :return: The new model with updated weights
        :rtype: Model
        """
        # Get weights from both models
        old_weights = old_model.model.get_weights()
        new_weights = new_model.model.get_weights()

        # Compute moving average (Simple Moving Average)
        averaged_weights = [(alpha * old_w + (1 - alpha) * new_w) for old_w, new_w in zip(old_weights, new_weights)]

        # Apply averaged weights to new model
        new_model.model.set_weights(averaged_weights)

        return new_model
    
    def _train(
            self,
            model: Model,
            input_size: int,
            output_size: int,
            params: tuple,
            epochs: int
        ) -> Model:
        """
        Trains a model based on the provided parameters.

        :param model: the model to be trained
        :type model: Model
        :param input_size: the size of the input layer
        :type input_size: int
        :param output_size: the size of the output layer
        :type output_size: int
        :param params: the parameters for the model
        :type params: tuple
        :param epochs: the number of epochs to train the model
        :type epochs: int
        :return: the trained model
        :rtype: Model
        """
        # Unpacking the parameters
        (
            paramSet,
            training_data,
            training_labels
        ) = params
        
        # Unpacking the parameter sets
        (
            architecture,
            learning_rate,
            batch_size
        ) = paramSet

        # Create list of activation functions for the network,
        # which will be relu for all but the output layer
        activations = ["relu" for _ in range(len(architecture))]    
        activations.append("linear")

        # Create and train the model
        model = self._build_model(
            model=model,
            input_size=input_size,
            output_size=output_size,
            architecture=architecture,
            activations=activations,
            learning_rate=learning_rate
            )
        
        # Train model
        model.trainModel(
            training_data,
            training_labels,
            None,
            None,
            epochs,
            batch_size
            )

        return model
    
    def _build_model(
            self,
            model: Model,
            input_size: int,
            output_size: int,
            architecture: list[int],
            activations: list[str],
            learning_rate: float = 0.001,
            lossFunc: str = "mse",
            metrics: list[str] = ["mae"]
            ) -> Model:
        """
        Builds a model of a network.

        :param model: the model to be built
        :type model: Model
        :param input_size: the size of the input layer
        :type input_size: int
        :param output_size: the size of the output layer
        :type output_size: int
        :param architecture: the architecture of the network
        :type architecture: list[int]
        :param activations: the activation functions for each layer
        :type activations: list[str]
        :param learning_rate: the learning rate for the optimizer
        :type learning_rate: float, optional
        :param lossFunc: the loss function to be used
        :type lossFunc: str, optional
        :param metrics: the metrics to be used for evaluation
        :type metrics: list[str], optional
        """
        model = model()

        model.create_sequential_model(
            architecture,
            activations,
            input_size,
            output_size
            )
        
        model.compileModel(
            learning_rate,
            lossFunc,
            metrics
            )
        return model
    