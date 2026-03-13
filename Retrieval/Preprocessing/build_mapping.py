import pandas as pd
import json

train_path = "Retrieval/data/train_ready.parquet"
val_path = "Retrieval/data/val_ready.parquet"

train = pd.read_parquet(train_path)
val = pd.read_parquet(val_path)

df = pd.concat([train, val])

users = df["user_id"].unique()
user2idx = {int(u): i + 1 for i, u in enumerate(users)}

# ITEMS
items = df["item_id"].unique()
item2idx = {int(i): idx + 1 for idx, i in enumerate(items)}

# ROOT
roots = df["root"].dropna().unique()
root2idx = {int(r): i + 1 for i, r in enumerate(roots)}

# LEAF
leafs = df["leaf"].dropna().unique()
leaf2idx = {int(l): i + 1 for i, l in enumerate(leafs)}

# =================
# SAVE
# =================

with open("Retrieval/data/user2idx.json", "w") as f:
    json.dump(user2idx, f)

with open("Retrieval/data/item2idx.json", "w") as f:
    json.dump(item2idx, f)

with open("Retrieval/data/root2idx.json", "w") as f:
    json.dump(root2idx, f)

with open("Retrieval/data/leaf2idx.json", "w") as f:
    json.dump(leaf2idx, f)

print("Mappings saved.")