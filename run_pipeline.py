from pipelines.training_pipeline import training_pipeline
from src.config import RAW_DATA_PATH

if __name__ == '__main__':
    # run the pipeline
    training_pipeline(data_path=RAW_DATA_PATH)