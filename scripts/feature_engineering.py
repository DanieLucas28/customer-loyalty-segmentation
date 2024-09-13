import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


def load_data(file_path: str = None, df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Load the dataset from a CSV file or return the provided DataFrame.

    Args:
        file_path (str): The path to the CSV file. Optional if df is provided.
        df (pd.DataFrame): An existing DataFrame. If provided, it will be used instead of loading from file.
    
    Returns:
        pd.DataFrame: The loaded or filtered dataset.
    """
    # Check if a DataFrame is provided, otherwise load from file
    if df is None:
        if file_path is not None:
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Either a valid 'file_path' or 'df' must be provided.")

    # Convert 'InvoiceDate' to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Filter rows for 'United Kingdom'
    df = df[df['Country'] == 'United Kingdom']

    return df


def create_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features of Recency, Frequency, and Monetary value (RFM)
    at client level.

    Args:
        df (pd.DataFrame): The dataset.
    Returns:
        pd.DataFrame: The dataset with the RFM features.

    """

    latest_date = df['InvoiceDate'].max()

    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (latest_date - x.max()).days,  # Recency
        'Invoice': 'count',  # Frequency
        'Price': 'sum'  # Monetary
    }).rename(columns={'InvoiceDate': 'Recency', 'Invoice': 'Frequency', 'Price': 'Monetary'})

    return rfm


def create_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create aggregate features at client level.
    Features to create:
        - Total orders
        - Average price
    
    Args:
        df (pd.DataFrame): The dataset.
    Returns:
        pd.DataFrame: The dataset with the aggregate features.
    """

    aggregate_features = df.groupby('Customer ID').agg({
        'Invoice': 'count',
        'Price': 'mean'
    }).rename(columns={'Invoice': 'Total Orders', 'Price': 'Average Price'})

    return aggregate_features


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the features.

    Args:
        df (pd.DataFrame): The dataset.
    Returns:
        pd.DataFrame: The dataset with the normalized features.
    """
    scaler = StandardScaler()
    df[['Recency', 'Frequency', 'Monetary', 'Total Orders', 'Average Price']] = scaler.fit_transform(
        df[['Recency', 'Frequency', 'Monetary', 'Total Orders', 'Average Price']])
    joblib.dump(scaler, 'C:/Users/danie/Documents/GitHub/customer-loyalty-segmentation/scalers/standard_scaler.pkl')

    return df


def feature_engineering(input_file: str = None, output_file: str = None, df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Main function to execute the feature engineering pipeline:
        - Loads the data
        - Creates the RFM features
        - Creates the aggregate features
        - Normalizes the features
        - Saves the features to a csv file

    Args:
        input_file (str): The path to the csv file. Optional if df is provided.
        output_file (str): The path to save the processed csv file. Optional.
        df (pd.DataFrame): An existing DataFrame. If provided, it will be used instead of loading from a file.

    Returns:
        pd.DataFrame: The dataset with the normalized features.
    """
    # Ensure either input_file or df is provided
    if df is None:
        if input_file is not None:
            df = load_data(input_file)
        else:
            raise ValueError("Either 'input_file' or 'df' must be provided.")

    # Create the RFM features
    rfm = create_rfm_features(df)

    # Create the aggregate features
    aggregate_features = create_aggregate_features(df)

    # Concatenate the features
    df = pd.concat([rfm, aggregate_features], axis=1)

    # Normalize the features
    df = normalize_features(df)

    # Ensure 'Customer ID' is integer type and reset index
    df = df.reset_index()
    df['Customer ID'] = df['Customer ID'].astype(int)

    # Save the features to a csv file if output_file is provided
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Features saved to {output_file}")

    return df


if __name__ == "__main__":
    input_file = 'C:/Users/danie/Documents/GitHub/customer-loyalty-segmentation/data/processed/clean_data.csv'
    output_file = 'C:/Users/danie/Documents/GitHub/customer-loyalty-segmentation/data/processed/customer_features.csv'
    feature_engineering(input_file, output_file, df=False)
