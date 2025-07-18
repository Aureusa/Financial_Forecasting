import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout  # type: ignore
from tensorflow.keras.layers import LSTM  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore

from naive_model.mlp import Model
from naive_model.normalizer import DataNormalizer


class LstmModel(Model):
    """
    The base class for LSTM models.
    """
    def __init__(self):
        """
        Initializes the LSTM model with a data normalizer.
        """
        super().__init__()
        self.scaler = DataNormalizer()


    def create_sequential_model(
            self,
            architecture: list[int],
            activations: list[str],
            input_shape: int,
            output_size: int
    ) -> None:
        """
        Creates the model architecture and assigns it to the model attribute.

        :param architecture: Holds parameters look_back (0) - the number of data points the LSTM layer uses
        for predictions and model_shape (1) - the number of neurons in the LSTM layer.
        :type architecture: list[int]
        :param activations: Activation function for each layer.
        :type activations: list[str]
        :param input_shape: Number of data points used for input.
        :type input_shape: int
        :param output_size: Number of neurons in the output layer.
        :type output_size: int
        """
        look_back = int(architecture[0])
        model_shape = int(architecture[1])

        model = Sequential([
            LSTM(units=model_shape, input_shape=(input_shape, 2)),
            Dropout(0.123),
            Dense(units=output_size, activation="relu")
        ])

        self.model = model

    def trainModel(
            self,
            training_data: np.ndarray,
            training_labels: np.ndarray,
            validation_data: np.ndarray,
            validation_labels: np.ndarray,
            epochs: int,
            batch_size: int
    ) -> None:
        """
        Trains the model using specified training and validation data.

        :param training_data: Data for training, matching the input shape of
        the model.
        :type training_data: np.ndarray
        :param training_labels: Labels for training data, matching the output
        shape of the model.
        :type training_labels: np.ndarray
        :param validation_data: Data for validation during training to prevent
        overfitting.
        :type validation_data: np.ndarray
        :param validation_labels: Labels for validation data.
        :type validation_labels: np.ndarray
        :param epochs: Number of training iterations.
        :type epochs: int
        :param batch_size: Data points per batch, the number processed before
        updating weights.
        :type batch_size: int
        """

        x_data = [training_data]
        y_data = [training_labels]

        x_data = [self.scaler.scale_data(dt) for dt in x_data]
        x_data = [self.scaler.reshape_input(dt) for dt in x_data]

        y_data = [self.scaler.scale_data(dt) for dt in y_data]

        return super().trainModel(x_data[0], y_data[0], None, None, epochs, batch_size)

    def predict(
            self,
            data: np.ndarray
    ) -> np.ndarray:
        """
        Makes a prediction on the specified data using the trained model

        :param data: the data you want to predict
        :type data: np.ndarray
        :return: the predictions
        :type return: np.ndarray
        """
        x_test = data
        x_test = self.scaler.scale_data(x_test)
        x_test = self.scaler.reshape_input(x_test)

        predictions = super().predict(x_test)

        padded_predictions = np.zeros((predictions.shape[0], 2))    # dummy values to avoid input-output shape mismatch
        padded_predictions[:, 0] = predictions[:, 0]

        y_predictions = self.scaler.inverse_scaled_data(padded_predictions)
        return y_predictions[:, 0].reshape(-1, 1)

    def monte_carlo_predictions(self, data: np.ndarray, num_samples: int = 100):
        """
        Uses Monte Carlo Dropout to obtain a probability distribution of predictions.
        
        :param data: The input data.
        :type data: np.ndarray
        :param num_samples: Number of stochastic forward passes.
        :type num_samples: int
        :return: Tuple containing mean predictions and standard deviation.
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        self._model_validator()

        @tf.function
        def f_model(input_data, training=True):
            return self.model(input_data, training=training)  # Keep dropout active

        # Process input data
        x_test = self.scaler.scale_data(data)
        x_test = self.scaler.reshape_input(x_test)

        # Run multiple stochastic passes and rescale each prediction
        preds_rescaled = []
        for _ in range(num_samples):
            pred = f_model(x_test, training=True).numpy()
            
            # Pad prediction for inverse transformation
            padded_pred = np.zeros((pred.shape[0], 2))
            padded_pred[:, 0] = pred[:, 0]

            # Inverse scale each prediction
            rescaled_pred = self.scaler.inverse_scaled_data(padded_pred)[:, 0].reshape(-1, 1)
            preds_rescaled.append(rescaled_pred)

        # Convert to numpy array
        preds_rescaled = np.array(preds_rescaled)  # Shape: (num_samples, batch_size, 1)

        # Compute mean & standard deviation in original scale
        mean_rescaled = preds_rescaled.mean(axis=0)
        std_rescaled = preds_rescaled.std(axis=0)

        return mean_rescaled, std_rescaled
    