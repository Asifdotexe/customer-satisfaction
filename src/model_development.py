import logging
import pandas as pd
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
    
