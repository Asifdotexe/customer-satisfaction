import logging
from typing import Union
from abc import ABC, abstractmethod

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
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocesses the ingested data

        Args:
            data (pd.DataFrame): Pandas dataframe containing ingested dataset

        Returns:
            pd.DataFrame: Pandas dataframe containing preprocessed dataset
        """
        try:
            # dropping all the unwanted columns for now, 
            # these can be integrated in the program later.
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1,
            )
            # filling the missing values with the median
            data["product_weight_g"].fillna(
                data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(
                data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(
                data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(
                data["product_width_cm"].median(), inplace=True)
            # write "No review" in review_comment_message column
            data["review_comment_message"].fillna("No review", inplace=True)

            data = data.select_dtypes(include=[np.number])
            #TODO: currently dropping we can later handle this with another strategy
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)
            
            return data
        
        except Exception as e:
            logging.error(e)
            raise e
        
class DataSplitStrategy(DataStrategy):
    """Strategy for splitting the data into train and test"""
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            # X contains the independent variables
            X = data.drop(['review_score'], axis=1)
            # y contains the dependent variable
            y = data['review_score']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in splitting data: {e}")
            raise e
        
class DataCleaning:
    """
    Class for cleaning data which processes the data and 
    divides it into the train and test data
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy
        
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """Handles data"""
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in handling data: {e}")
            raise e
    
#TODO: Can add more strategies like encoding strategy and outlier detection strategy
# to make the pipline more robust and use more features