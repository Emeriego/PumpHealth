import pandas as pd

def load_data(path: str, sep: str = ",") -> pd.DataFrame:
    """
    Load dataset from a file with error handling.

    Parameters:
    - path: file path
    - sep: delimiter (default is comma)

    Returns:
    - pd.DataFrame
    """
    try:
        df = pd.read_csv(path, sep=sep)
        print("DataFrame successfully loaded.")
        print(f"Shape: {df.shape}")
        return df

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {path}")

    except Exception as e:
        raise Exception(f"Error loading data: {e}")