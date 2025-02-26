import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from tqdm import tqdm
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(
            self,
            num_timesteps: int,
            num_sources: int,
            time_combiner_params: dict,
            source_combiner_params: dict,
            mlp_layers: list[int],
            lr: float
        ) -> None:
        """
        Instantiates a model whose architecture is as follows:
            - It uses "num_timesteps" MLPs to generate source-dependent
            features.
            - It uses "num_sources" MLPs to generate time-dependent
            features.
            - It uses two TransformerEmbeding layers that enrich the features:
                - One to do self attention on the source-dependent features
                - One to do self attention on the time-dependent features
            - The enriched features are then concatiante and are fed into a
            MLP for final inference

        :param num_timesteps: The number of time steps of the sequence.
        :type num_timesteps: int
        :param num_sources: The number of sources for each time step.
        :type num_sources: int
        :param time_combiner_params: Dictionary containing the parameters
        of the TransformerEmbeding layer tasked with self attention on the
        time-dependent features. The usual structure of the dictionary
        is as follows:
            dict = {
                "d_model": int,
                "nhead": int,
                "dim_feedforward": int,
            }
        :type time_combiner_params: dict
        :param source_combiner_params: Dictionary containing the parameters
        of the TransformerEmbeding layer tasked with self attention on the
        source-dependent features. The usual structure of the dictionary
        is as follows:
            dict = {
                "d_model": int,
                "nhead": int,
                "dim_feedforward": int,
            }
        :type source_combiner_params: dict
        :param mlp_layers: A list containing the neurons in the MLP used
        for inference. The usual structure is as follows:
            list = [int,int,...]
        :type mlp_layers: list[int]
        :param lr: The learning rate.
        :type lr: float
        """
        super(Model, self).__init__()

        self._loss_vals = []

        self._num_timesteps = num_timesteps
        self._num_sources = num_sources
        self._num_features = time_combiner_params["d_model"]

        # Define the embeders
        self._time_embedders = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.Linear(1, self._num_features),
                # torch.nn.BatchNorm1d(num_sources),
                torch.nn.ReLU(),
                torch.nn.Linear(self._num_features, self._num_features),
                # torch.nn.BatchNorm1d(num_sources),
                torch.nn.ReLU(),
            ) # (B, S, T, 1) x (1, F) = (B, S, T, F)
        for _ in range(self._num_timesteps))

        # Time and Source transformer
        self._time_combiner = nn.TransformerEncoderLayer(**time_combiner_params, batch_first=True)
        #self._source_combiner = nn.TransformerEncoderLayer(**source_combiner_params, batch_first=True)

        self._linear = torch.nn.Sequential(
            nn.Linear(self._num_timesteps, 1),
            # torch.nn.BatchNorm1d(self._num_features)
        )

        # Fully connected output MLP
        # self._mlp = self._build_mlp(self._num_features, mlp_layers)

        # Output layer
        self._fc_out = nn.Linear(self._num_features, 1)

        # Loss function and optimizer
        self._criterion = nn.MSELoss()
        self._optimizer = optim.Adam(self.parameters(), lr=lr)
        

    def data_compatibility_check(self, x: Tensor) -> None:
        shape = x.shape

        if len(shape) != 3:
            raise ValueError(
                f"You must provide a data that in (batch_size, num_timesteps, num_source), got {shape} instead"
            )
        
        if shape[1] != self._num_timesteps:
            raise ValueError(
                f"Expected number of timesteps `{self._num_timesteps}` got `{shape[1]}` instead."
            )

        if shape[2] != self._num_sources:
            raise ValueError(
                f"Expected number of sources `{self._num_sources}` got `{shape[2]}` instead."
            )
        
        message = f"|| Your data passed the compatibility check and is of the shape {shape}. ||"
        border = "=" * len(message)

        success_message = f"\n{border}\n{message}\n{border}"

        print(success_message)

    def forward(self, x: Tensor) -> Tensor:
        """
        The forward pass of the model.

        :param x: the data which is in a shape of
        (batch_size, timestep, sources), where sources
        represent the financial indicators.
        :type x: Tensor
        :return: the predictions of the model
        :rtype: Tensor
        """
        # Pass through the Transformer to learn dependencies 
        # throught time of the sources.
        combined_through_time = self.\
            _feature_embeddings(x, num_features=self._num_features)
        combined_through_time = self._time_combiner(combined_through_time)
        result_time = combined_through_time[:, -self._num_timesteps:, :] # (B, T, F)

        result_transposed = result_time.transpose(1, 2) # (B, F, T)

        results = self._linear(result_transposed) # (B, F, T) x (T, 1) = (B, F, 1)

        results_squezed = results.squeeze(-1)  # (B, F)

        final_result = self._fc_out(results_squezed) # (B, F) x (F, 1) = (B, 1)

        return final_result

    def train_model(self, train_loader: Tensor, epochs: int = 50):
        """
        Train the model using training data.
        
        :param train_loader: The training data.
        :type train_loader: Tensor
        :param epochs: The number of epochs the model
        is training, defaults to 50.
        :type epochs: int
        """
        self.train()  # Set to training mode
        for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
            total_loss = 0
            for batch in train_loader:
                x_batch, y_batch = batch

                self._optimizer.zero_grad()  # Clear previous gradients
                predictions = self(x_batch)  # Forward pass
                loss = self._criterion(predictions, y_batch)  # Compute loss
                loss.backward()  # Backpropagation
                self._optimizer.step()  # Update weights

                total_loss += loss.item()

            self._loss_vals.append(total_loss / len(train_loader))
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

    def make_loss_plot(self, name: str) -> None:
        """
        Makes a plot of the loss function and saves it.

        :param name: the name of the plot
        :type name: str
        """
        plt.plot(self._loss_vals)
        plt.savefig(f"{name}.png")
        plt.close()

    def evaluate_model(self, test_loader: Tensor):
        """
        Used to evaluate the model on a test data.

        :param test_loader: The test data.
        :type test_loader: Tensor
        """
        self.eval()  # Set to evaluation mode
        total_loss = 0
        with torch.no_grad():  # Disable gradient computation
            for batch in test_loader:
                x_batch, y_batch = batch
                predictions = self(x_batch)
                loss = self._criterion(predictions, y_batch)
                total_loss += loss.item()
        
        print(f"Test Loss: {total_loss / len(test_loader):.4f}")

    def _build_mlp(self, input_dim: int, layers: list[float]) -> nn.Sequential:
        """
        Used to build the final MLP used for inference.

        :param input_dim: The input dimension.
        :type input_dim: int
        :param layers: A list containing the number of neurons
        in each layer.
        :type layers: list[float]
        :return: A MLP model used for inference.
        :rtype: nn.Sequential
        """
        mlp_layers = []
        for neurons in layers:
            mlp_layers.append(nn.Linear(input_dim, neurons))
            mlp_layers.append(nn.ReLU())  # Activation function
            input_dim = neurons  # Update input size for next layer
        
        # Final output layer (single value prediction)
        mlp_layers.append(nn.Linear(input_dim, 1))
        return nn.Sequential(*mlp_layers)
    
    def _feature_embeddings(self, x: Tensor, num_features: int) -> Tensor:
        """
        Creates feature embeddings.

        :param x: the tensor to do a feauture embeddings on, in
        shape (batch_size, timestep, sources)
        :type x: Tensor
        :param time_embedding: Whether or not to do a time embedding
        or a token embedding. 
            Time embedding (True) - Looks at the releationship of the different
            sources through time.
            Time embedding (False) - Looks at the releationshops of the different
            sources between each other.
        :type time_embedding: bool
        :return: The data with the embedded features.
        :rtype: Tensor
        """
        batch_size = x.shape[0] # (B, T, S)

        embedders = self._time_embedders

        target_size = self._num_sources # (S,)

        x = x.unsqueeze(-1) # (B, S, T, 1)/(B, T, S, 1)

        num_embeders = len(embedders)

        # Featurize each source independently
        features = []
        for si in range(num_embeders):
            s = x[:, si, :, :] # (B, T, 1)

            featurizer_si = embedders[si]
            f = featurizer_si(s) # (B, T, 1) x (1, F) = (B, T, F)

            ts = torch.arange(0, target_size, device=s.device)
            pos_encodings = self._positional_encoding(num_features, ts) # (T, F)
            pos_encodings = pos_encodings.unsqueeze(0) # (1, T, F) || (B, T, F)
            f = f + pos_encodings # (B, T, F)
            features.append(f)

        # Creates the features
        features = torch.stack(features, dim=1) # (B, S, T, F)

        # Reshape the data into the correct shape
        # (batch_size, timestep*source, num_features)
        combined = features.reshape(
            batch_size,
            self._num_sources*self._num_timesteps,
            num_features
        ) # (B, S*T, F)

        return combined

    def _positional_encoding(
            self,
            d_model: int,
            t: Tensor,
        ) -> Tensor:
        """
        Makes a positional encoding using sine and cosine functions.

        :type d_model: int
        :type t: torch.Tensor (shape [batch_size, sequence_length])
        :rtype: torch.Tensor (shape [batch_size, sequence_length, d_model])
        """
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, d_model, 2, device=t.device) / d_model)
        )
        # Ensure `t` has shape [batch_size, sequence_length, 1]
        t = t.unsqueeze(-1)  # Shape [batch_size, sequence_length, 1]
        pos_enc_a = torch.sin(t * inv_freq)  # Shape [batch_size, sequence_length, d_model // 2]
        pos_enc_b = torch.cos(t * inv_freq)  # Shape [batch_size, sequence_length, d_model // 2]
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)  # Shape [batch_size, sequence_length, d_model]
        return pos_enc
    