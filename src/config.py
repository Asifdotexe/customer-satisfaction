"""
This file contains all the constant variables for easy configuration
- File paths
- Model configurations
"""
from zenml.steps import BaseParameters

# used in run_pipeline.py
RAW_DATA_PATH = r'data\raw\olist_customers_dataset.csv'

class ModelNameConfig(BaseParameters):
    """Model Configurations"""
    model_name: str = "LinearRegression"