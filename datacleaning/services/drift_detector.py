import numpy as np

class DataDriftDetector:

    def __init__(self, old_df, new_df):
        self.old = old_df
        self.new = new_df
        self.drift_report = {}

    def detect(self):

        numeric_cols = self.old.select_dtypes(include=["int64","float64"]).columns

        for col in numeric_cols:
            if col not in self.new.columns:
                continue

            old_mean = self.old[col].mean()
            new_mean = self.new[col].mean()

            if old_mean == 0:
                continue

            shift = abs(new_mean - old_mean) / abs(old_mean)

            if shift > 0.30:  # 30% shift threshold
                self.drift_report[col] = round(float(shift*100),2)

        return self.drift_report
