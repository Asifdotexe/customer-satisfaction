import logging
import pandas as pd
from zenml import step

class IngestData:
    def __init__(self, data_path: str):
        """Initialize the IngestData class with the data path."""
        self.data_path = data_path

    def get_data(self):
        """Load data from the specified file and log the ingestion process."""
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)