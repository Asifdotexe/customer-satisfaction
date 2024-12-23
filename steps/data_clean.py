import logging
import pandas as pd
from zenml import step

@step
def clean_df(df: pd.DataFrame) -> None:
    """Cleans the ingested data

    Args:
        df (pd.DataFrame): pandas Dataframe containing the ingested data

    Returns:
        pd.DataFrame: pandas Dataframe containing the cleaned data
    """
    pass