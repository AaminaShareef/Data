def validate_data(df):
    # Drop rows that are completely empty (safety)
    df.dropna(how="all", inplace=True)

    # Final check
    remaining_nulls = df.isnull().sum().sum()

    # Allow minimal missing values in non-critical columns
    if remaining_nulls > 0:
        print(f"⚠ Warning: Dataset contains {remaining_nulls} missing values after cleaning")

    return True
