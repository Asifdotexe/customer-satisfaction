import logging

import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin

from src.strategy_model_evaluation import MeanSquaredError, RootMeanSquaredError, R2Score

@step
def evaluate_model(
    model: 
    df: pd.DataFrame
) -> None:
    """Evaluates the model on the ingested data

    Args:
        df (pd.DataFrame): pandas dataframe containing the ingested data
    """