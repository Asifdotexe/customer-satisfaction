import logging
import pandas as pd
from sklearn.linear_model import LinearRegression
from abc import ABC, abstractmethod

class Model(ABC):
    """Abstract class defining a model"""
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Trains the model

        Args:
            X_train (pd.DataFrame): Pandas dataframe containing the training 
                independent features
            y_train (pd.Series): Pandas series containing the training dependent labels
        """
        pass
    
class LinearRegressionModel(Model):
    """Linear Regression Model"""
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        """Trains the model

        Args:
            X_train (pd.DataFrame): Pandas dataframe containing the training 
                independent features
            y_train (pd.Series): Pandas series containing the training dependent labels
        """
        try:
            model = LinearRegressionModel(**kwargs)
            model.fit(X_train, y_train)
            logging.info("Linear Regression model trained successfully")
            return model
        except Exception as e:
            logging.error(f"Error training Linear Regression model: {e}")
            raise e