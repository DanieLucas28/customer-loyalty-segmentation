import pandas as pd
import joblib
from sklearn.cluster import KMeans


def apply_clustering(input_file: str = None, scaler_file: str = None, n_clusters: int = 4, output_file: str = None,
                     df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Apply clustering to the data using the KMeans algorithm, 
    denormalize the data, and save the results to a CSV file.
    
    Args:
        input_file (str): Path to the input CSV file. Optional if df is provided.
        scaler_file (str): Path to the scaler (for denormalization).
        n_clusters (int): Number of clusters to use.
        output_file (str): Path to the output CSV file. Optional.
        df (pd.DataFrame): Preloaded DataFrame. Optional, if input_file is not provided.
    
    Returns:
        pd.DataFrame: The data with the cluster labels.
    """
    # Ensure either input_file or df is provided
    if df is None:
        if input_file is not None:
            df = pd.read_csv(input_file)
        else:
            raise ValueError("Either 'input_file' or 'df' must be provided.")

    # Ensure the scaler is provided
    if scaler_file is None:
        raise ValueError("Scaler file must be provided to denormalize the features.")

    # Define the features to be used in clustering
    features = ['Recency', 'Frequency', 'Monetary', 'Total Orders', 'Average Price']

    # Load the scaler
    scaler = joblib.load(scaler_file)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    df['Cluster'] = kmeans.fit_predict(df[features])

    # Denormalize the features
    df[features] = scaler.inverse_transform(df[features])

    # Save the results if an output file is provided
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Clustered data saved to {output_file}")

    return df


if __name__ == "__main__":
    input_file = 'C:/Users/danie/Documents/GitHub/customer-loyalty-segmentation/data/processed/customer_features.csv'
    output_file = 'C:/Users/danie/Documents/GitHub/customer-loyalty-segmentation/data/clusters.csv'
    scaler_file = 'C:/Users/danie/Documents/GitHub/customer-loyalty-segmentation/scalers/standard_scaler.pkl'
    n_clusters = 4
    apply_clustering(input_file, scaler_file, n_clusters, output_file)
