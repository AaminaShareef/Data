import pandas as pd
import os


def load_file(file_path: str) -> pd.DataFrame:
    """
    Load a CSV or Excel file into a pandas DataFrame.

    Args:
        file_path: absolute path to the uploaded file

    Returns:
        pandas DataFrame

    Raises:
        ValueError: if file format is not supported
        FileNotFoundError: if file does not exist
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.csv':
        # try UTF-8 first, fall back to latin-1 for encoding issues
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin-1')

    elif ext in ('.xlsx', '.xls'):
        df = pd.read_excel(file_path, engine='openpyxl')

    else:
        raise ValueError(f"Unsupported file format: {ext}. Only CSV and Excel files are supported.")

    return df