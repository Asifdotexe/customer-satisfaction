import logging
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

class Evaluation(ABC):
    """Abstract class defining evaluation strategy"""
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Evaluate the model by calculate the scores

        Args:
            y_true (np.ndarray): Labels containing true value
            y_pred (np.ndarray): Labels containing predicted value
        """
        pass
    
class MeanSquaredError(Evaluation):
    """Evaluate the model by calculate the mean squared error"""
    def calculate_scores(self, y_true, y_pred):
        try:
            logging.info("Calculating MeanSquaredError")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"Mean Squared Error: {mse}")
            return mse
        
        except Exception as e:
            logging.error(f"Error calculating MeanSquaredError: {e}")
            raise e
        
class R2Score(Evaluation):
    """Evaluate the model by calculate the R2 score"""
    def calculate_scores(self, y_true, y_pred):
        try:
            logging.info("Calculating R2 Score")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2 Score: {r2}")
            return r2
        
        except Exception as e:
            logging.error(f"Error calculating R2 Score: {e}")
            raise e
        
class RootMeanSquaredError(Evaluation):
    """Evaluate the model by calculating the root mean squared error"""
    def calculate_scores(self, y_true, y_pred):
        try:
            logging.info("Calculating root mean squared error")
            rmse = root_mean_squared_error(y_true, y_pred)
            logging.info(f"Root Mean Squared Error: {rmse}")
            return rmse
        
        except Exception as e:
            logging.error(f"Error calculating root mean squared error: {e}")
            raise e