from zenml.client import Client

from src.config import RAW_DATA_PATH
from pipelines.training_pipeline import training_pipeline

if __name__ == '__main__':
    # run the pipeline
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    training_pipeline(data_path=RAW_DATA_PATH)