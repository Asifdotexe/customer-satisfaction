import logging
import pandas as pd
from zenml import step

from src.model_evaluation import MeanSquaredError, RootMeanSquaredError, R2Score

@step
def evaluate_model(df: pd.DataFrame) -> None:
    """Evaluates the model on the ingested data

    Args:
        df (pd.DataFrame): pandas dataframe containing the ingested data
    """