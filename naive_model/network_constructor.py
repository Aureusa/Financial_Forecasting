from typing import Tuple
import tensorflow as tf
from copy import deepcopy

from naive_model import Model

# Disable progress bars
tf.keras.utils.disable_interactive_logging()

# Define the type for a single parameter tuple
ParamTuple = list[
    Tuple[
        Tuple[
            list[int], float, int
            ],   # The current parameter set
        int,                            # Index of the current parameter set
        int,                            # Total number of parameter sets
        list[float],                   # Training data
        list[float],                   # Training labels
        list[float],                   # Validation data
        list[float],                   # Validation labels
        list[float],                   # Testing data
        list[float]                    # Testing labels
        ]
    ]

LIST_BB = list(range(1000, 1000001, 1000))

class NetworksConstructor:
    """
    A way of constructing NNs with different parameters to be
    used in performing a statistical analysis to deremine the
    optimal set of parameters.
    """
    def __init__(
            self,
            model: Model,
            input_size: int,
            output_size: int,
            epochs: int
            ) -> None:
        """
        Initializes parameters used throughout the class.

        :param model: The model to be used, has to be of type Model
        :type model: Model
        :param input_size: Number of data points in the input data.
        :type input_size: int
        :param output_size: Number of points in the output data.
        :type output_size: int
        :param epochs: Maximum number of epochs for training.
        :type epochs: int
        """
        self._results = []
        self._training_loss = []
        self._validation_loss = []
        self.input_size = input_size
        self.output_size = output_size
        self.epochs = epochs
        self.model: Model = model

    @property
    def training_loss(self) -> list[list[float]]:
        """
        A way of retrieving the training loss

        :return: the training loss
        :rtype: list[list[float]]
        """
        return deepcopy(self._training_loss)
    
    @property
    def validation_loss(self) -> list[list[float]]:
        """
        A way of retrieving the validation loss

        :return: the validation loss
        :rtype: list[list[float]]
        """
        return deepcopy(self._validation_loss)

    @property
    def results(self) -> list:
        """
        A way of retrieving the results.

        :return: the results
        :rtype: list
        """
        return deepcopy(self._results)
    
    def _build_model(
            self,
            architecture: list[int],
            activations: list[str],
            learning_rate: float = 0.001,
            lossFunc: str = "mse",
            metrics: list[str] = ["mae"]
            ) -> Model:
        """
        Builds a model of a network.

        :param architecture: the architecture of the NN
        :type architecture: list[int]
        :param activations: list with actiavation functions
        :type activations: list[str]
        :param learning_rate: the learning rate of the model,
        defaults to 0.001
        :type learning_rate: float, optional
        :param lossFunc: the loss function that we are minimising,
        defaults to "mse"
        :type lossFunc: str, optional
        :param metrics: a list of metrics that we are tracking,
        defaults to ["mae"]
        :type metrics: list[str], optional
        :return: the model
        :rtype: Model
        """
        model = self.model()

        model.create_sequential_model(
            architecture,
            activations,
            self.input_size,
            self.output_size
            )
        
        model.compileModel(
            learning_rate,
            lossFunc,
            metrics
            )
        return model
    
    def _train(
            self,
            params: ParamTuple,
        ) -> None:
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
            self.epochs,
            batch_size
            )

        return model
    
    def train_model(
        self,
        training_data: list[float],
        training_labels: list[float],
        paramList: list[tuple[list[int], float, int]],
        model_filename: str,
        model_foldername: str,
        ) -> None:
        # Create a tuple with the parameter set and associated metadata
        param_tuple = (
            paramList,                  # The current parameter set
            training_data,              # Training data
            training_labels,            # Training labels
        )
        
        model = self._train(param_tuple)

        model.save_model(model_foldername, model_filename)
        
        print(f"Model `{model_filename}` saved in `{model_foldername}` folder")
