import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from Retrieval.Train.candidate_retrieval import load_faiss_index, retrieve_topk
from Ranking.wide_deep_model import WideAndDeepRanker


def l2_normalize(vec):
    norm = np.linalg.norm(vec)
    if norm <= 0:
        return vec
    return vec / norm


def load_embedding_npz(npz_path):
    payload = np.load(npz_path, allow_pickle=True)
    ids = payload["ids"]
    vectors = payload["vectors"].astype(np.float32)
    return {str(i): vectors[idx] for idx, i in enumerate(ids)}


def interaction_label(event_name):
    event_name = str(event_name)
    return 1.0 if event_name in {"addtocart", "transaction"} else 0.0


class RankingDataset(Dataset):
    def __init__(self, features, labels, group_ids):
        # Keep backing storage as NumPy to avoid materializing the
        # whole dataset as Torch tensors at construction time.
        self.features = np.asarray(features, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.group_ids = group_ids
        self.feature_dim = self.features.shape[1] if self.features.size > 0 else 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "features": torch.from_numpy(self.features[idx]),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
            "group": self.group_ids[idx],
        }


def build_ranking_rows(
    df,
    user_embedding_map,
    item_embedding_map,
    index,
    item_ids,
    top_k=100,
    force_include_positive=True,
):
    features = []
    labels = []
    group_ids = []

    positives = df[df["event"].isin(["addtocart", "transaction"])].copy()

    for row in tqdm(positives.itertuples(index=False), total=len(positives), desc="Build ranking rows"):
        user_id = str(row.user_id)
        pos_item = str(row.item_id)

        if user_id not in user_embedding_map or pos_item not in item_embedding_map:
            continue

        user_vec = l2_normalize(user_embedding_map[user_id]).astype(np.float32)
        candidate_ids, _ = retrieve_topk(index, item_ids, user_vec, k=top_k)
        candidate_ids = candidate_ids[0] if candidate_ids else []

        if force_include_positive and pos_item not in candidate_ids:
            candidate_ids = [pos_item] + candidate_ids[:-1]

        for cand_id in candidate_ids:
            item_vec = item_embedding_map.get(cand_id)
            if item_vec is None:
                continue

            item_vec = l2_normalize(item_vec)
            cross_vec = user_vec * item_vec
            model_input = np.concatenate([user_vec, item_vec, cross_vec], axis=0)

            features.append(model_input)
            labels.append(1.0 if cand_id == pos_item else 0.0)
            group_ids.append(f"{user_id}:{pos_item}")

    return features, labels, group_ids


def evaluate_topk(model, dataset, device, k=10):
    model.eval()

    grouped = {}
    with torch.no_grad():
        for idx in range(len(dataset)):
            row = dataset[idx]
            x = row["features"].to(device)
            y = float(row["label"].item())
            g = row["group"]
            s = float(torch.sigmoid(model(x.unsqueeze(0))).item())

            grouped.setdefault(g, []).append((s, y))

    hits = 0
    total = 0
    for _, scores in grouped.items():
        ranked = sorted(scores, key=lambda x: x[0], reverse=True)
        top_items = ranked[:k]
        if any(label > 0 for _, label in top_items):
            hits += 1
        total += 1

    return hits / total if total > 0 else 0.0


def evaluate_ndcg_at_k(model, dataset, device, k=10):
    model.eval()

    grouped = {}
    with torch.no_grad():
        for idx in range(len(dataset)):
            row = dataset[idx]
            x = row["features"].to(device)
            y = float(row["label"].item())
            g = row["group"]
            s = float(torch.sigmoid(model(x.unsqueeze(0))).item())

            grouped.setdefault(g, []).append((s, y))

    ndcgs = []
    for _, scores in grouped.items():
        ranked = sorted(scores, key=lambda x: x[0], reverse=True)[:k]

        dcg = 0.0
        for rank, (_, label) in enumerate(ranked, start=1):
            if label > 0:
                dcg += 1.0 / np.log2(rank + 1)

        ideal_hits = int(sum(label > 0 for _, label in scores))
        ideal_hits = min(ideal_hits, k)
        if ideal_hits == 0:
            ndcgs.append(0.0)
            continue

        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
        ndcgs.append(dcg / idcg)

    return float(np.mean(ndcgs)) if ndcgs else 0.0


def train_model(train_ds, val_ds, output_dir, epochs=5, batch_size=512, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim = train_ds.feature_dim

    model = WideAndDeepRanker(input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_ndcg = -1.0
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "wide_deep_best.pt"
    last_path = output_dir / "wide_deep_last.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            x = batch["features"].to(device)
            y = batch["label"].to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        val_recall = evaluate_topk(model, val_ds, device, k=10)
        val_ndcg = evaluate_ndcg_at_k(model, val_ds, device, k=10)

        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_loss,
            "val_recall_at_10": val_recall,
            "val_ndcg_at_10": val_ndcg,
            "input_dim": input_dim,
        }
        torch.save(payload, last_path)

        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            torch.save(payload, best_path)

        print(
            f"Epoch {epoch}: train_loss={avg_loss:.6f}, "
            f"val_recall@10={val_recall:.6f}, val_ndcg@10={val_ndcg:.6f}"
        )

    return {
        "best_model": str(best_path),
        "last_model": str(last_path),
        "best_val_ndcg_at_10": best_ndcg,
    }


def main():
    root = Path(__file__).resolve().parents[1]

    train_path = root / "Retrieval/data/train_ready.parquet"
    val_path = root / "Retrieval/data/val_ready.parquet"

    emb_dir = root / "Retrieval/artifacts/embeddings"
    index_dir = root / "Retrieval/artifacts/index"
    ranker_out = root / "Ranking/artifacts"

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    user_emb_train = load_embedding_npz(emb_dir / "user_embeddings_train.npz")
    item_emb_train = load_embedding_npz(emb_dir / "item_embeddings_train.npz")
    user_emb_val = load_embedding_npz(emb_dir / "user_embeddings_val.npz")

    index, item_ids = load_faiss_index(
        index_dir / "item_faiss.index",
        index_dir / "item_ids.npy",
    )

    train_x, train_y, train_groups = build_ranking_rows(
        train_df,
        user_emb_train,
        item_emb_train,
        index,
        item_ids,
        top_k=100,
        force_include_positive=True,
    )
    val_x, val_y, val_groups = build_ranking_rows(
        val_df,
        user_emb_val,
        item_emb_train,
        index,
        item_ids,
        top_k=100,
        force_include_positive=False,
    )

    if not train_x or not val_x:
        raise RuntimeError(
            "Ranking dataset is empty. Run retrieval training first to export embeddings/index."
        )

    train_ds = RankingDataset(train_x, train_y, train_groups)
    val_ds = RankingDataset(val_x, val_y, val_groups)

    result = train_model(train_ds, val_ds, ranker_out)

    summary_path = ranker_out / "wide_deep_training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved training summary -> {summary_path}")


if __name__ == "__main__":
    main()
