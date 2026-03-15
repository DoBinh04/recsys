import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ranking model
from Ranking.wide_deep_model import WideAndDeepRanker

# Retrieval utilities (FAISS index + search)
from Retrieval.Train.candidate_retrieval import load_faiss_index, retrieve_topk

# Dataset used for retrieval model
from Retrieval.Train.retrieval_dataset import RetrievalDataset

# Two-tower retrieval model
from Retrieval.Train.two_tower_model import TwoTowerModel


# Events considered as positive interactions
POSITIVE_EVENTS = {"addtocart", "transaction"}


# L2 normalize a vector
def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# Compute DCG score from rank position
def dcg_from_rank(rank_1_indexed: int) -> float:
    return 1.0 / np.log2(rank_1_indexed + 1)


# Load a JSON file
def load_json(path: Path) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# Load the trained TwoTower retrieval model
def load_retrieval_model(
    checkpoint_path: Path,
    user2idx: Dict,
    item2idx: Dict,
    root2idx: Dict,
    leaf2idx: Dict,
    device: str,
) -> TwoTowerModel:

    # Initialize model
    model = TwoTowerModel(
        num_users=len(user2idx),
        num_items=len(item2idx),
        num_roots=len(root2idx),
        num_leafs=len(leaf2idx),
    ).to(device)

    # Load checkpoint weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Set evaluation mode
    model.eval()
    return model


# Load Wide & Deep ranking model
def load_ranker(checkpoint_path: Path, device: str) -> WideAndDeepRanker:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    input_dim = int(checkpoint["input_dim"])

    model = WideAndDeepRanker(input_dim=input_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    return model


# Build dataloader from dataset
def build_loader(
    data_path: Path,
    user2idx: Dict,
    item2idx: Dict,
    root2idx: Dict,
    leaf2idx: Dict,
    batch_size: int,
) -> DataLoader:

    dataset = RetrievalDataset(
        data_path,
        user2idx=user2idx,
        item2idx=item2idx,
        root2idx=root2idx,
        leaf2idx=leaf2idx,
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# Create a dictionary: item_id -> item_embedding
def build_item_embedding_map(
    model: TwoTowerModel,
    loader: DataLoader,
    device: str,
) -> Dict[str, np.ndarray]:

    item_map: Dict[str, np.ndarray] = {}

    with torch.no_grad():
        for batch in loader:

            # Move tensors to device
            batch_device = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            # Encode item embeddings
            item_vec = model.encode_item(batch_device)

            # Normalize embeddings
            item_vec = F.normalize(item_vec, dim=1).cpu().numpy()

            raw_ids = [str(x) for x in batch["item_raw_id"]]

            for item_id, vec in zip(raw_ids, item_vec):
                if item_id not in item_map:
                    item_map[item_id] = vec.astype(np.float32)

    return item_map


# Rank retrieved candidates using the ranking model
def score_candidates(
    ranker: WideAndDeepRanker,
    device: str,
    user_vec: np.ndarray,
    candidate_ids: Sequence[str],
    item_embeddings: Dict[str, np.ndarray],
) -> List[Tuple[str, float]]:

    scored = []

    with torch.no_grad():
        for item_id in candidate_ids:

            item_vec = item_embeddings.get(str(item_id))
            if item_vec is None:
                continue

            item_vec = l2_normalize(item_vec).astype(np.float32)

            # Feature vector for ranking model
            features = np.concatenate([user_vec, item_vec, user_vec * item_vec], axis=0)

            x = torch.from_numpy(features).unsqueeze(0).to(device)

            score = float(torch.sigmoid(ranker(x)).item())

            scored.append((str(item_id), score))

    # Sort by score
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored


# Evaluate retrieval-only and retrieval+ranking performance
def evaluate(
    model: TwoTowerModel,
    ranker: WideAndDeepRanker,
    eval_loader: DataLoader,
    item_embeddings: Dict[str, np.ndarray],
    index,
    index_item_ids: Sequence[str],
    ks: Sequence[int],
    candidate_k: int,
    device: str,
):

    # Metrics for retrieval stage
    retrieval_stats = {f"Recall@{k}": 0 for k in ks}
    retrieval_stats.update({f"NDCG@{k}": 0.0 for k in ks})

    # Metrics for full pipeline
    full_stats = {f"Recall@{k}": 0 for k in ks}
    full_stats.update({f"NDCG@{k}": 0.0 for k in ks})

    total_positive = 0
    max_k = max(ks)

    # Number of items to retrieve
    request_k = max(max_k, candidate_k)

    with torch.no_grad():
        for batch in eval_loader:

            batch_device = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            # Encode user embeddings
            user_vecs = model.encode_user(batch_device)

            user_vecs = F.normalize(user_vecs, dim=1).cpu().numpy().astype(np.float32)

            item_raw_ids = [str(x) for x in batch["item_raw_id"]]
            events = [str(x) for x in batch["event"]]

            for idx, (event_name, target_item) in enumerate(zip(events, item_raw_ids)):

                # Only evaluate positive samples
                if event_name not in POSITIVE_EVENTS:
                    continue

                total_positive += 1

                user_vec = user_vecs[idx]

                # Retrieve candidates from FAISS
                candidate_ids_batch, _ = retrieve_topk(index, index_item_ids, user_vec, k=request_k)
                candidate_ids = candidate_ids_batch[0] if candidate_ids_batch else []

                # Retrieval-only metrics
                for k in ks:
                    topk_retrieval = candidate_ids[:k]

                    if target_item in topk_retrieval:
                        retrieval_stats[f"Recall@{k}"] += 1
                        rank = topk_retrieval.index(target_item) + 1
                        retrieval_stats[f"NDCG@{k}"] += dcg_from_rank(rank)

                # Ranking stage
                ranked = score_candidates(
                    ranker=ranker,
                    device=device,
                    user_vec=user_vec,
                    candidate_ids=candidate_ids[:candidate_k],
                    item_embeddings=item_embeddings,
                )

                ranked_ids = [item_id for item_id, _ in ranked]

                for k in ks:
                    topk_full = ranked_ids[:k]

                    if target_item in topk_full:
                        full_stats[f"Recall@{k}"] += 1
                        rank = topk_full.index(target_item) + 1
                        full_stats[f"NDCG@{k}"] += dcg_from_rank(rank)

    # Normalize metrics
    if total_positive == 0:
        retrieval_metrics = {name: 0.0 for name in retrieval_stats}
        full_metrics = {name: 0.0 for name in full_stats}
    else:
        retrieval_metrics = {name: float(value / total_positive) for name, value in retrieval_stats.items()}
        full_metrics = {name: float(value / total_positive) for name, value in full_stats.items()}

    return {
        "retrieval_only": retrieval_metrics,
        "retrieval_plus_ranking": full_metrics,
        "num_positive_samples": total_positive,
    }