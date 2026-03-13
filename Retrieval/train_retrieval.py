import torch
import json
from tqdm import tqdm

from Retrieval.Train.two_tower_model import TwoTowerModel
from Retrieval.Train.retrieval_loss import retrieval_loss
from Retrieval.Train.data_loader import build_dataloader


device = "cuda" if torch.cuda.is_available() else "cpu"


# ========================
# PATHS
# ========================

train_path = "data/train_ready.parquet"
val_path = "data/val_ready.parquet"

user_map_path = "data/user2idx.json"
item_map_path = "data/item2idx.json"
root_map_path = "data/root2idx.json"
leaf_map_path = "data/leaf2idx.json"


# LOAD MAPPINGS
with open(user_map_path) as f:
    user2idx = json.load(f)

with open(item_map_path) as f:
    item2idx = json.load(f)

with open(root_map_path) as f:
    root2idx = json.load(f)

with open(leaf_map_path) as f:
    leaf2idx = json.load(f)


# DATALOADER
train_loader, val_loader = build_dataloader(
    train_path,
    val_path,
    user2idx,
    item2idx,
    root2idx,
    leaf2idx,
    batch_size=1024
)


# MODEL
model = TwoTowerModel(
    num_users=len(user2idx),
    num_items=len(item2idx),
    num_roots=len(root2idx),
    num_leafs=len(leaf2idx)
).to(device)


# OPTIMIZER
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3
)


# ========================
# TRAIN LOOP
# ========================

for epoch in range(10):

    model.train()

    total_loss = 0

    for batch in tqdm(train_loader):

        batch = {k: v.to(device) for k, v in batch.items()}

        user_vec, item_vec = model(batch)

        loss = retrieval_loss(
            user_vec,
            item_vec,
            batch["weight"]
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}")
    print("Train Loss:", total_loss / len(train_loader))