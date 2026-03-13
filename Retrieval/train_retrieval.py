import json
from pathlib import Path

import torch
from tqdm import tqdm

from Retrieval.Train.data_loader import build_dataloader
from Retrieval.Train.evaluation import (
    build_item_catalog_from_loader,
    evaluate_retrieval_metrics,
    export_embeddings,
)
from Retrieval.Train.faiss_index import build_faiss_index
from Retrieval.Train.retrieval_loss import retrieval_loss
from Retrieval.Train.two_tower_model import TwoTowerModel


device = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ========================
# PATHS
# ========================

train_path = PROJECT_ROOT / "Retrieval/data/train_ready.parquet"
val_path = PROJECT_ROOT / "Retrieval/data/val_ready.parquet"

user_map_path = PROJECT_ROOT / "Retrieval/data/user2idx.json"
item_map_path = PROJECT_ROOT / "Retrieval/data/item2idx.json"
root_map_path = PROJECT_ROOT / "Retrieval/data/root2idx.json"
leaf_map_path = PROJECT_ROOT / "Retrieval/data/leaf2idx.json"

artifacts_dir = PROJECT_ROOT / "Retrieval/artifacts"
checkpoints_dir = artifacts_dir / "checkpoints"
embeddings_dir = artifacts_dir / "embeddings"
index_dir = artifacts_dir / "index"

for folder in [artifacts_dir, checkpoints_dir, embeddings_dir, index_dir]:
    folder.mkdir(parents=True, exist_ok=True)

# LOAD MAPPINGS
with open(user_map_path, encoding="utf-8") as f:
    user2idx = json.load(f)

with open(item_map_path, encoding="utf-8") as f:
    item2idx = json.load(f)

with open(root_map_path, encoding="utf-8") as f:
    root2idx = json.load(f)

with open(leaf_map_path, encoding="utf-8") as f:
    leaf2idx = json.load(f)

# DATALOADER
train_loader, val_loader = build_dataloader(
    train_path,
    val_path,
    user2idx,
    item2idx,
    root2idx,
    leaf2idx,
    batch_size=128,
)

# MODEL
model = TwoTowerModel(
    num_users=len(user2idx),
    num_items=len(item2idx),
    num_roots=len(root2idx),
    num_leafs=len(leaf2idx),
).to(device)

# OPTIMIZER
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

best_recall_at_20 = -1.0
best_ckpt_path = checkpoints_dir / "two_tower_best.pt"
last_ckpt_path = checkpoints_dir / "two_tower_last.pt"

num_epochs = 10
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        user_vec, item_vec = model(batch)
        loss = retrieval_loss(user_vec, item_vec, batch["weight"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # Validation with unified metrics: Recall@K, NDCG@K
    item_vecs, item_ids = build_item_catalog_from_loader(model, train_loader, device)
    val_metrics = evaluate_retrieval_metrics(
        model,
        val_loader,
        item_vecs,
        item_ids,
        device,
        ks=(20, 50),
    )

    print(f"Epoch {epoch}/{num_epochs}")
    print(f"Train Loss: {avg_train_loss:.6f}")
    print(
        "Validation Metrics: "
        + ", ".join([f"{name}={value:.6f}" for name, value in val_metrics.items()])
    )

    checkpoint_payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": avg_train_loss,
        "val_metrics": val_metrics,
    }

    torch.save(checkpoint_payload, last_ckpt_path)

    if val_metrics.get("Recall@20", 0.0) > best_recall_at_20:
        best_recall_at_20 = val_metrics["Recall@20"]
        torch.save(checkpoint_payload, best_ckpt_path)
        print(f"Saved best checkpoint -> {best_ckpt_path}")

print(f"Training complete. Best Recall@20={best_recall_at_20:.6f}")

# Load best checkpoint for export/indexing
best_state = torch.load(best_ckpt_path, map_location=device)
model.load_state_dict(best_state["model_state_dict"])
model.eval()

# Export tower vectors for downstream ranking (Wide & Deep)
train_export = export_embeddings(model, train_loader, embeddings_dir, device, split_name="train")
val_export = export_embeddings(model, val_loader, embeddings_dir, device, split_name="val")

print(f"Exported train embeddings -> {train_export}")
print(f"Exported val embeddings -> {val_export}")

# Build ANN index with FAISS from unique item vectors
item_vecs, item_ids = build_item_catalog_from_loader(model, train_loader, device)

try:
    index_info = build_faiss_index(item_vecs, item_ids, index_dir)
    print(f"FAISS index built -> {index_info}")
except ImportError as err:
    print(f"Skipped FAISS index build: {err}")
