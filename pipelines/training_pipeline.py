from zenml import pipelines
from steps.data_ingest import ingest_data
from steps.data_clean import clean_data
from steps.model_train import train_model
from steps.model_evaluate import evaluate_model

@pipelines()
def training_pipeline(data_path: str):
    df = ingest_data(data_path)
    clean_data(df)
    train_model(df)
    evaluate_model(df)