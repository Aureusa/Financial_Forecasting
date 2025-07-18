import os
import warnings

from pipelines import test_model, train_model

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Turn off oneDNN custom operations

# Suppress other warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


if __name__ == "__main__":
    train_model(
        labelsPerSet=1,
        pointsPerSet=3,
        stockCode="AAPL",
    )
    test_model(
        labelsPerSet=1,
        pointsPerSet=3,
        stock_name="AAPL",
        residual_model_folder = "mlp__foldername_placeholder",
        trend_model_folder = "lstm__foldername_placeholder",
        residual_model="mlp_filename_placeholder_model.keras",
        trend_model="lstm_filename_placeholder_model.keras"
    )
