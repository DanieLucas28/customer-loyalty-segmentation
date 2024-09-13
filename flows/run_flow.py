from prefect import flow, task, get_run_logger
from scripts.data_cleaning import clean_data
from scripts.feature_engineering import feature_engineering
from scripts.apply_clustering import apply_clustering


@task
def clean_data_task(raw_data_path: str):
    """
    Task to clean the data.
    Args:
        raw_data_path (str): Path to the input file.
    Returns:
        pd.DataFrame: Cleaned data.
    """
    logger = get_run_logger()
    logger.info(f"Starting data cleaning from {raw_data_path}")

    clean_df = clean_data(raw_data_path)

    logger.info("Data cleaning completed successfully.")
    return clean_df


@task
def feature_engineering_task(clean_df):
    """
    Task to perform feature engineering on cleaned data.
    Args:
        clean_df (pd.DataFrame): Cleaned DataFrame.
    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    logger = get_run_logger()
    logger.info("Starting feature engineering...")

    featured_df = feature_engineering(df=clean_df)

    logger.info("Feature engineering completed successfully.")
    return featured_df


@task
def apply_clustering_task(featured_df, n_clusters: int, scaler_file: str):
    """
    Task to apply clustering to the data.
    Args:
        featured_df (pd.DataFrame): DataFrame with features ready for clustering.
        n_clusters (int): Number of clusters for KMeans.
        scaler_file (str): Path to the scaler file.
    Returns:
        pd.DataFrame: DataFrame with cluster labels.
    """
    logger = get_run_logger()
    logger.info(f"Applying clustering with {n_clusters} clusters using scaler from {scaler_file}...")

    clustered_df = apply_clustering(scaler_file=scaler_file, n_clusters=n_clusters, df=featured_df)

    logger.info("Clustering completed successfully.")
    return clustered_df


@flow
def data_pipeline_flow(file_path: str, scaler_file: str, n_clusters: int):
    """
    Flow to orchestrate the data pipeline.
    Args:
        file_path (str): Path to the raw input file.
        scaler_file (str): Path to the scaler file.
        n_clusters (int): Number of clusters for KMeans.
    """
    logger = get_run_logger()
    logger.info(f"Starting data pipeline flow with input file {file_path}...")

    # Step 1: Clean the data
    clean_df = clean_data_task(file_path)  # Use .result() to ensure task completion

    # Step 2: Feature engineering
    featured_df = feature_engineering_task(clean_df)  # Ensure task completion before next step

    # Step 3: Apply clustering
    clustered_df = apply_clustering_task(featured_df, n_clusters, scaler_file)

    logger.info("Data pipeline flow completed successfully.")
    return clustered_df


# Caminho para os arquivos
file_path = "C:/Users/danie/Documents/GitHub/customer-loyalty-segmentation/data/raw/online_retail_II.xlsx"
scaler_file = 'C:/Users/danie/Documents/GitHub/customer-loyalty-segmentation/scalers/standard_scaler.pkl'
n_clusters = 4

# Executar o flow
if __name__ == "__main__":
    data_pipeline_flow(file_path, scaler_file, n_clusters)
