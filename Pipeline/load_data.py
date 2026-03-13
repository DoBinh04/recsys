import pandas as pd

def load_events(path):

    df = pd.read_csv(path)

    # chuẩn hóa tên cột
    df.columns = df.columns.str.strip().str.lower()

    df = df.rename(columns={
        "visitorid": "user_id",
        "itemid": "item_id"
    })

    return df