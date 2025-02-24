from itertools import product


class ParamGenerator:
    """
    Used to generate different hyperparameter permuations
    for each of the components in the sophisticated model.
    """
    def generate_learning_rates(
            self,
            min_lr: float,
            max_lr: float,
            step: float
        ) -> float:
        """
        Generate a list of learning rates between min_lr and max_lr.

        :param min_lr: Minimum learning rate.
        :type min_lr: float
        :param max_lr: Maximum learning rate.
        :type max_lr: float
        :param step: Step size.
        :type step:
            scale (str): "linear" or "log" scale.

        :return: List of learning rates.
        :rtype: list[float]
        """
        return [round(x, 8) for x in self._frange(min_lr, max_lr, step)]
        
    def generate_mlp_permutations(
            self,
            layer_ranges: dict
        ) -> list[list[float]]:
        """
        Generate all possible permutations of MLP architectures 
        by varying the number of neurons per layer.

        :param layer_ranges: Dictionary where keys are layer indices
        (1, 2, 3, ..., N) and values are (min, max, step) tuples for neuron counts.
        Example:
            layer_ranges = {
                1: (min, max, step),
                2: (min, max, step),
                3: (min, max, step),
                .
                .
                .
                N: (min, max, step)
            }

        :return: List of lists, each representing an MLP architecture.
        :rtype: list[list[float]]
        """
        layer_values = [
            list(range(v[0], v[1] + v[2], v[2])) for v in layer_ranges.values()
        ]
        
        # Generate all possible neuron configurations for each layer
        mlp_architectures = [list(values) for values in product(*layer_values)]
        
        return mlp_architectures

    def generate_transformer_permutations(
            self,
            param_ranges: dict
        ) -> list[dict]:
        """
        Generate all permutations of Transformer hyperparameters
        based on min, max, and step values.

        :param param_ranges: Dictionary with parameter names as
        keys and (min, max, step) tuples as values. Typically this dict looks
        like:
            dict = {
                "d_model": (min, max, step),
                "nhead": (min, max, step),
                "dim_feedforward": (min, max, step),
            }
        :return: List of dictionaries, each representing a
        unique set of hyperparameters.
        :rtype: list[dict]
        """
        # Create a list of parameter names and their respective value ranges
        param_keys = param_ranges.keys()
        param_values = [
            list(range(v[0], v[1] + v[2], v[2])) if isinstance(v[0], int)
            else [round(x, 2) for x in self._frange(v[0], v[1], v[2])]
            for v in param_ranges.values()
        ]
        
        # Generate all possible combinations
        param_combinations = [
            dict(zip(param_keys, values)) for values in product(*param_values)
        ]
        
        return param_combinations

    def _frange(
            self,
            start: float,
            stop: float,
            step: float
        ):
        """
        Helper function to generate floating point ranges.

        :param start: the start of the range.
        :type start: float
        :param stop: the end of the range.
        :type stop: float
        :param step: the step size.
        :type step: float
        :yield: the generated value
        :rtype: float
        """
        while start <= stop:
            yield start
            start += step
