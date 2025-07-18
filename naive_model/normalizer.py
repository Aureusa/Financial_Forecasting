from sklearn import preprocessing
import numpy as np


class DataNormalizer:
    """
    A class to normalize data using Min-Max scaling.
    It scales the data to a range between 0 and 1.
    """
    def __init__(self):
        """
        Initializes the DataNormalizer with a Min-Max scaler.
        """
        self.min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    
    def reshape_input(self, input_data: np.ndarray) -> np.ndarray:
        """
        Reshapes the input data to be compatible with the model.

        :param input_data: The data to be reshaped.
        :type input_data: np.ndarray
        :return: Reshaped data.
        :rtype: np.ndarray
        """
        x = np.reshape(input_data, (input_data.shape[0], 1, input_data.shape[1]))
        return x
    
    def scale_data(self, data: np.ndarray) -> np.ndarray:
        """
        Scales the data using Min-Max scaling.

        :param data: The data to be scaled.
        :type data: np.ndarray
        :return: Scaled data.
        :rtype: np.ndarray
        """
        # Scaling data
        scaled_data = self.min_max_scaler.fit_transform(data)  # .reshape(-1, 1)
        #transformed_data = self.reshape_input(scaled_data)  ?????????
        return scaled_data
    
    def inverse_scaled_data(self, data: np.ndarray) -> np.ndarray:
        """
        Inverses the scaling of the data.

        :param data: The data to be inversely scaled.
        :type data: np.ndarray
        :return: Inversely scaled data.
        :rtype: np.ndarray
        """
        transformed_data = self.min_max_scaler.inverse_transform(data)
        return transformed_data
    