import logging
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """Abstract class defining strategy for handling data"""
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass
    
class DataPreProcessStrategy(DataStrategy):
    """Strategy for preprocessing data"""
    def handle(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data

        Args:
            data (pd.DataFrame): Pandas dataframe containing ingested dataset

        Returns:
            pd.DataFrame: Pandas dataframe containing preprocessed dataset
        """
        logging.info("Preprocessing data")