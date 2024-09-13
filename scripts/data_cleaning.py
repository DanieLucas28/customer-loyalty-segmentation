import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from csv file.

    Args:
        file_path (str): The path to the csv file.
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    df = pd.read_excel(file_path)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in columns 'Description' and 'Customer ID'.

    Args:
        df (pd.DataFrame): The dataset.
    Returns:
        pd.DataFrame: The dataset with missing values handled.
    """

    df = df.dropna(subset=['Description', 'Customer ID'])
    return df


def remove_negative_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with negative values in columns 'Quantity' and 'Price'.

    Argss:
        df (pd.DataFrame): The dataset.
    Returns:
        pd.DataFrame: The dataset with negative values removed.
    """
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
    return df


def remove_outliers(df: pd.DataFrame, quantile: float = 0.95) -> pd.DataFrame:
    """
    Remove outliers in columns 'Quantity' and 'Price'.

    Args:
        df (pd.DataFrame): The dataset.
        quantile (float): The quantile to use for outlier detection.
    Returns:
        pd.DataFrame: The dataset with outliers removed.
    """
    df = df[(df['Quantity'] < df['Quantity'].quantile(quantile)) & (df['Price'] < df['Price'].quantile(quantile))]
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicates in the dataset.
    Args:
        df (pd.DataFrame): The dataset.
    Returns:
        pd.DataFrame: The dataset with duplicates removed.
    """
    df = df.drop_duplicates()
    return df


def clean_data(file_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values, removing negative values, outliers, and duplicates.
    Saves the cleaned dataset to a csv file if output_path is provided.
    
    Args:
        file_path (str): The path to the csv file.
        output_path (str): The path to save the cleaned dataset. Optional.
    
    Returns:
        pd.DataFrame: The cleaned dataset.
    """

    # Load the dataset
    df = load_data(file_path)

    # Handle missing values
    df = handle_missing_values(df)

    # Remove negative values
    df = remove_negative_values(df)

    # Remove outliers
    df = remove_outliers(df)

    # Remove duplicates
    df = remove_duplicates(df)

    # Save the cleaned data if output_path is provided
    if output_path:
        df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    file_path = "C:/Users/danie/Documents/GitHub/customer-loyalty-segmentation/data/raw/online_retail_II.xlsx"
    output_path = "C:/Users/danie/Documents/GitHub/customer-loyalty-segmentation/data/raw/data_cleaned.csv"
    clean_data(file_path, output_path)
