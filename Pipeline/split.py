def temporal_split(df):

    df = df.sort_values("timestamp")

    train_size = int(len(df)*0.7)
    val_size = int(len(df)*0.85)

    train = df.iloc[:train_size]
    val = df.iloc[train_size:val_size]
    test = df.iloc[val_size:]

    return train, val, test