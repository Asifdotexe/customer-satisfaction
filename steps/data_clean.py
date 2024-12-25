import logging
import pandas as pd
from zenml import step

from typing import Annotated
from src.strategy_data_cleaning import DataCleaning, \
    DataPreProcessStrategy, DataSplitStrategy

@step
def clean_df(df: pd.DataFrame) -> tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.DataFrame, "y_train"],
    Annotated[pd.DataFrame, "y_test"],
]:
    """Cleans the ingested data

    Args:
        df (pd.DataFrame): pandas Dataframe containing the ingested data

    Returns:
        X_train (pd.DataFrame): Training data
        X_test (pd.DataFrame): Testing data
        y_train (pd.Series): Training labels
        y_test (pd.Series): Testing labels
    """
    try:
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()
        
        split_strategy = DataSplitStrategy()
        data_cleaning = DataCleaning(processed_data, split_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
    except Exception as e:
        logging.info(f"Error in cleaning data: {e}")
        raise e