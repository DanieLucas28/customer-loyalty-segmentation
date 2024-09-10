import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from csv file.

    Args:
        file_path (str): The path to the csv file.
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    df = pd.read_csv(file_path)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
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
        'InvoiceDate': lambda x: (latest_date - x.max()).days, #Recency
        'Invoice': 'count', #Frequency
        'Price': 'sum' # Monetary
    }).rename(columns={'InvoiceDate': 'Recency', 'Invoice':'Frequency', 'Price': 'Monetary'})

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
    df[['Recency', 'Frequency', 'Monetary', 'Total Orders', 'Average Price']] = scaler.fit_transform(df[['Recency', 'Frequency', 'Monetary', 'Total Orders', 'Average Price']])

    return df

def feature_engineering(input_file, output_file) -> pd.DataFrame:
    """
    Main function to execute the feature engineering pipeline:
        - Loads the data
        - Creates the RFM features
        - Creates the aggregate features
        - Normalizes the features
        - Saves the features to a csv file

    Args:
            input_file (str): The path to the csv file.
            output_file (str): The path to the csv file.
    Returns:
            pd.DataFrame: The dataset with the normalized features.
    """

    # Load the data
    df = load_data(input_file)
    # Create the RFM features
    rfm = create_rfm_features(df)
    # Create the aggregate features
    aggregate_features = create_aggregate_features(df)
    # Concatenate the features
    df = pd.concat([rfm, aggregate_features], axis=1)
    # Normalize the features
    df = normalize_features(df)
    # Save the features to a csv file
    df = df.reset_index()
    df['Customer ID'] = df['Customer ID'].astype(int)
    df.to_csv(output_file, index=False)

input_file = 'C:/Users/danie/Documents/GitHub/customer-loyalty-segmentation/data/processed/clean_data.csv'
output_file = 'C:/Users/danie/Documents/GitHub/customer-loyalty-segmentation/data/processed/customer_features.csv'
feature_engineering(input_file, output_file)