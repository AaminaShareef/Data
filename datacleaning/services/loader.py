import pandas as pd
from pathlib import Path

class DatasetLoader:

    @staticmethod
    def load_dataset(file_path):
        """
        Automatically loads CSV or Excel dataset
        """
        path = Path(file_path)

        if path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)

        elif path.suffix.lower() in [".xls", ".xlsx"]:
            df = pd.read_excel(file_path, engine="openpyxl")

        else:
            raise ValueError("Unsupported file format")

        return df
