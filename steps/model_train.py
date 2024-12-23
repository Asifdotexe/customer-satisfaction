import logging
import pandas as pd
from zenml import step
from sklearn.base import RegressionMixin

from src.config import ModelNameConfig
from src.model_development import LinearRegressionModel

@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.Series,
    y_train: pd.DataFrame,
    y_test: pd.Series,
    config: ModelNameConfig
) -> RegressionMixin:
    """Trains the machine learning model

    Args:
        X_train (pd.DataFrame): pandas Dataframe containing  the training data
        X_test (pd.Series): pandas Series containing the testing data
        y_train (pd.DataFrame): pandas Dataframe containing the training labels
        y_test (pd.Series): pandas Series containing the testing labels
        config (ModelNameConfig): Model configuration containing the model configuration

    Returns:
        RegressionMixin: Trained model
    """
    model = None
    if config.model_name == "LinearRegression":
        model = LinearRegressionModel()
        trained_model = model.train(X_train, y_train)
        return trained_model
    else:
        raise ValueError("Invalid model")