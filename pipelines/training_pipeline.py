from zenml import pipeline
from steps.data_ingest import ingest_df
from steps.data_clean import clean_df
from steps.model_train import train_model
from steps.model_evaluate import evaluate_model

@pipeline(enable_cache=True)
def training_pipeline(data_path: str) -> None:
    """Calling the training pipeline

    Args:
        data_path (str): Path to the data
        
    Returns: 
        None
    """
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, y_train)
    r2, rmse, mse = evaluate_model(model, X_test, y_test)