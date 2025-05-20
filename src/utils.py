import os
import logging
import joblib
import json
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def ensure_dir(directory: str) -> None:
    """
    Ensures that a directory exists. Creates it if it doesn't.

    Parameters:
        directory (str): Path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")
    else:
        logging.info(f"Directory already exists: {directory}")


def save_model(model, filepath: str) -> None:
    """
    Saves the trained model to disk using joblib.

    Parameters:
        model: Trained model object.
        filepath (str): File path to save the model.
    """
    ensure_dir(os.path.dirname(filepath))
    joblib.dump(model, filepath)
    logging.info(f"Model saved to {filepath}")


def save_metrics(metrics: dict, filepath: str) -> None:
    """
    Saves evaluation metrics to a JSON file.

    Parameters:
        metrics (dict): Dictionary of metrics (e.g., accuracy, classification report).
        filepath (str): File path to save metrics.
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Metrics saved to {filepath}")


def save_plot(fig, filepath: str) -> None:
    """
    Saves a matplotlib figure to the specified path.

    Parameters:
        fig: Matplotlib figure object.
        filepath (str): File path to save the plot.
    """
    ensure_dir(os.path.dirname(filepath))
    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Plot saved to {filepath}")
