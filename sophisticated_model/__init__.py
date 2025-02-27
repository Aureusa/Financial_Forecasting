import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from data_parser.dataFactory import StockDataFactory, device
from sophisticated_model.model import Model

def train_sophisticated_model():
    datafactory = StockDataFactory("AAPL", 3, 1)

    # Get training data
    (
        training_sets_residuals,
        training_labels_residuals,
        training_sets_sma,
        training_labels_sma
    ) = datafactory.generate_training_loaders("2021-01-18", "2024-09-01")

    # Get testing data
    (
        testing_sets_residuals,
        testing_sets_sma,
        testing_labels_residuals,
        testing_labels_sma
    ) = datafactory.generate_testing_loaders("2024-08-10", "2025-02-25")

    ###### FEATURES ######

    # Reshape data: (batch_size, timestep, sources) → (batch_size * timestep, sources)
    train_reshaped_sma = training_sets_sma.reshape(-1, training_sets_sma.shape[-1]).cpu().numpy()
    test_reshaped_sma = testing_sets_sma.reshape(-1, testing_sets_sma.shape[-1]).cpu().numpy()
    train_reshaped_res = training_sets_residuals.reshape(-1, training_sets_residuals.shape[-1]).cpu().numpy()
    test_reshaped_res = testing_sets_residuals.reshape(-1, testing_sets_residuals.shape[-1]).cpu().numpy()
    
    # Fit MinMaxScaler on sma data
    scaler_sma = MinMaxScaler(feature_range=(0, 1))
    train_scaled_sma = scaler_sma.fit_transform(train_reshaped_sma)
    test_scaled_sma = scaler_sma.fit_transform(test_reshaped_sma)

    # Fit MinMaxScaler on residuals data
    scaler_res = MinMaxScaler(feature_range=(0, 1))
    train_scaled_residuals = scaler_res.fit_transform(train_reshaped_res)
    test_scaled_residuals = scaler_res.fit_transform(test_reshaped_res)

    ###### LABELS ######

    # Fit MinMaxScaler on labels residuals
    scaler_res_label = MinMaxScaler(feature_range=(0, 1))
    training_labels_scaled_residuals = scaler_res_label.fit_transform(training_labels_residuals.cpu().numpy())
    testing_labels_scaled_residuals = scaler_res_label.fit_transform(testing_labels_residuals.cpu().numpy())

    # Fit MinMaxScaler on labels data sma
    scaler_sma_label = MinMaxScaler(feature_range=(0, 1))
    training_labels_scaled_sma = scaler_sma_label.fit_transform(training_labels_sma.cpu().numpy())
    testing_labels_scaled_sma = scaler_sma_label.fit_transform(testing_labels_sma.cpu().numpy())


    # Reshape back to (batch_size, timestep, sources)
    training_sets_sma = torch.tensor(train_scaled_sma, dtype=torch.float32).reshape(training_sets_sma.shape).to(device)
    testing_sets_sma = torch.tensor(test_scaled_sma, dtype=torch.float32).reshape(testing_sets_sma.shape).to(device)
    training_sets_residuals = torch.tensor(train_scaled_residuals, dtype=torch.float32).reshape(training_sets_residuals.shape).to(device)
    testing_sets_residuals = torch.tensor(test_scaled_residuals, dtype=torch.float32).reshape(testing_sets_residuals.shape).to(device)

    training_labels_scaled_residuals = torch.tensor(training_labels_scaled_residuals).to(device)
    testing_labels_scaled_residuals = torch.tensor(testing_labels_scaled_residuals).to(device)
    training_labels_scaled_sma = torch.tensor(training_labels_scaled_sma).to(device)
    testing_labels_scaled_sma = torch.tensor(testing_labels_scaled_sma).to(device)




    # # Generate the dataset for the residual model
    # training_dataset_residual = TensorDataset(training_sets_residuals, training_labels_residuals)
    # testing_dataset_residual = TensorDataset(testing_sets_residuals, testing_labels_residuals)

    # # Generate the dataset for the sma model
    # training_dataset_sma = TensorDataset(training_sets_sma, training_labels_sma)
    # testing_dataset_sma = TensorDataset(testing_sets_sma, testing_labels_sma)

    # Generate the dataset for the residual model
    training_dataset_residual = TensorDataset(training_sets_residuals, training_labels_scaled_residuals)
    testing_dataset_residual = TensorDataset(testing_sets_residuals, testing_labels_scaled_residuals)

    # Generate the dataset for the sma model
    training_dataset_sma = TensorDataset(training_sets_sma, training_labels_scaled_sma)
    testing_dataset_sma = TensorDataset(testing_sets_sma, testing_labels_scaled_sma)

    # Create a training DataLoader
    training_loader_residual = DataLoader(training_dataset_residual, batch_size=64)
    training_loader_sma = DataLoader(training_dataset_sma, batch_size=64)

    # Create a testing DataLoader
    testing_loader_residual = DataLoader(testing_dataset_residual, batch_size=16)
    testing_loader_sma = DataLoader(testing_dataset_sma, batch_size=16)

    B = 789
    T = 2
    S_res = 6
    S_sma = 10
    F = 256

    time_combiner_params = {
        "d_model": F,
        "nhead": 1,
        "dim_feedforward": 512,
    }

    source_combiner_params = {
        "d_model": F,
        "nhead": 1,
        "dim_feedforward": 512,
    }

    mlp_layers = [F, 512, 512, 256, 128, 64, 32, 16, 8, 4, 2]

    # Creating the residual model
    model_res = Model(T,S_res,time_combiner_params,source_combiner_params,mlp_layers,0.001)
    model_res.to(device)

    # Creating the sma model
    model_sma = Model(T,S_sma,time_combiner_params,source_combiner_params,mlp_layers,0.001)
    model_sma.to(device)

    # Train the model
    model_res.train_model(training_loader_residual, epochs=500)
    model_res.make_loss_plot("residual_model")
    model_sma.train_model(training_loader_sma, epochs=500)
    model_sma.make_loss_plot("sma_model")

    # Test the model
    predicted_res = model_res(testing_sets_residuals)
    predicted_sma = model_sma(testing_sets_sma)

    # Scale predictions
    predicted_res_scaled = scaler_res_label.inverse_transform(predicted_res.detach().cpu().numpy())
    predicted_sma_scaled = scaler_sma_label.inverse_transform(predicted_sma.detach().cpu().numpy())

    predicted_closing = predicted_sma_scaled + predicted_res_scaled

    actual_closing = testing_labels_residuals + testing_labels_sma
    actual_closing = actual_closing.cpu().numpy()

    plt.plot(predicted_closing, label="Predicted Closing")
    plt.plot(actual_closing, label="Actual Closing")
    plt.legend()
    plt.show()