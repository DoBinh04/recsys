import pandas as pd

def preprocess_events(df, min_user_inter=5, min_item_inter=5):

    # convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # sort by time
    df = df.sort_values("timestamp")

    # remove duplicates
    df = df.drop_duplicates()

    # filter users
    user_counts = df.groupby("user_id").size()
    valid_users = user_counts[user_counts >= min_user_inter].index
    df = df[df["user_id"].isin(valid_users)]

    # filter items
    item_counts = df.groupby("item_id").size()
    valid_items = item_counts[item_counts >= min_item_inter].index
    df = df[df["item_id"].isin(valid_items)]

    return df