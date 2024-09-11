import pandas as pd
import joblib
from sklearn.cluster import KMeans

def denormalize(scaler_path: str, data: pd.DataFrame) -> pd.DataFrame:
    """
    Denormalize the data using the scaler.

    Args:
        scaler_path (str): Path to the scaler.
        data (pd.DataFrame): Data to denormalize.
    Returns:
        pd.DataFrame: Denormalized data.
    """
    scaler = joblib.load(scaler)
    return scaler.inverse_transform(data)

def apply_clustering(input_file: str,scaler_file: str, n_clusters: int, output_file: str) -> pd.DataFrame:
    """
    Apply clustering to the data using the KMeans algorithm, 
    denormalize the data, and save the results to a CSV file.
    Args:
        input_file (str): Path to the input CSV file.
        scaler_file (str): Path to the scaler.
        n_clusters (int): Number of clusters to use.
        output_file (str): Path to the output CSV file.
    """
    # Load data
    features = ['Recency', 'Frequency', 'Monetary', 'Total Orders', 'Average Price']
    data = pd.read_csv(input_file)
    scaler = joblib.load(scaler_file)
    data['Cluster'] = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit_predict((data[features]))
    data[features] = scaler.inverse_transform(data[features])
    data.to_csv(output_file, index=False)


input_file = 'C:/Users/danie/Documents/GitHub/customer-loyalty-segmentation/data/processed/customer_features.csv'
output_file = 'C:/Users/danie/Documents/GitHub/customer-loyalty-segmentation/data/clusters.csv'
scaler_file = 'C:/Users/danie/Documents/GitHub/customer-loyalty-segmentation/scalers/standard_scaler.pkl'
n_clusters = 4
apply_clustering(input_file, scaler_file, n_clusters, output_file)
