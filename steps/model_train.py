import logging
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin

from src.config import ModelNameConfig
from src.strategy_model_development import LinearRegressionModel

@step
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    config: ModelNameConfig
) -> RegressorMixin:
    """Trains the machine learning model

    Args:
        X_train (pd.DataFrame): pandas Dataframe containing  the training data
        y_train (pd.DataFrame): pandas Dataframe containing the training labels
        config (ModelNameConfig): Model configuration containing the model configuration

    Returns:
        RegressorMixin: Trained model
    """
    model = None
    if config.model_name == "LinearRegression":
        logging.info("Training Linear Regression model")
        model = LinearRegressionModel()
        trained_model = model.train(X_train, y_train)
        return trained_model
    else:
        raise ValueError("Invalid model")