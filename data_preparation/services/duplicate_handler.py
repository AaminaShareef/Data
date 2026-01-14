def handle_duplicates(df):
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    return df, before - after
