import logging
from typing import Annotated

import mlflow
import pandas as pd
from zenml import step
from zenml.client import Client
from sklearn.base import RegressorMixin

from src.strategy_model_evaluation import MeanSquaredError, RootMeanSquaredError, R2Score

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[
        Annotated[float, "r2_score"],
        Annotated[float, "rmse"],
        Annotated[float, "mse"],
    ]:
    """Evaluate the ingested data

    Args:
        model (RegressorMixin): Model that needs to be evaluated
        X_test (pd.DataFrame): Pandas dataframe containing the testing features
        y_test (pd.Series): Pandas dataframe containing the testing labels

    Returns:
        r2_score (float): R2 score
        rmse (float): Root mean squared error
        mse (float): Mean squared error
    """
    try:
        # predicting the test set results using mean squared error
        y_pred = model.predict(X_test)
        mse = MeanSquaredError()
        mse_result = mse.calculate_scores(y_test, y_pred)
        mlflow.log_metric("mse", mse_result)
        
        # predicting the test set results using root mean squared error
        rmse = RootMeanSquaredError()
        rmse_result = rmse.calculate_scores(y_test, y_pred)
        mlflow.log_metric("rmse", rmse_result)
        
        # predicting the test set results using R2 score
        r2 = R2Score()
        r2_result = r2.calculate_scores(y_test, y_pred)
        mlflow.log_metric("r2 score", r2_result)
        
        return r2_result, rmse_result, mse_result
    
    except Exception as e:
        logging.error(f"Error in evaluating model: {e}")
        raise e