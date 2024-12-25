import logging

import pandas as pd
from zenml import step

class IngestData:
    """Ingesting data from the data_path"""
    def __init__(self, data_path: str):
        """Ingesting the data from data

        Args:
            data_path (str): Path to the data file
        """
        self.data_path = data_path

    def get_data(self):
        """Load data from the specified file and log the ingestion process."""
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path, parse_dates=True)
    
@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """Ingest the data from the specified file

    Args:
        data_path (str): Path to the file

    Returns:
        pd.DataFrame: pandas DataFrame containing the ingested data
    """
    try:
        ingest_data = IngestData(data_path=data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error ingesting data: {e}")
        raise e