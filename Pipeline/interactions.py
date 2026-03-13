def build_interactions(df):

    weights = {
        "view": 1,
        "addtocart": 5,
        "transaction": 10
    }

    df["weight"] = df["event"].map(weights)

    interactions = df[["user_id","item_id","timestamp","weight", "event"]]

    return interactions