from pathlib import Path

import numpy as np


def _load_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "FAISS is not installed. Please install faiss-cpu or faiss-gpu."
        ) from exc
    return faiss


def load_faiss_index(index_path, ids_path):
    faiss = _load_faiss()

    index_path = Path(index_path)
    ids_path = Path(ids_path)

    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")
    if not ids_path.exists():
        raise FileNotFoundError(f"Item ID file not found: {ids_path}")

    index = faiss.read_index(str(index_path))
    item_ids = np.load(ids_path, allow_pickle=True)
    return index, item_ids


def retrieve_topk(index, item_ids, user_vectors, k=100):
    if user_vectors.ndim == 1:
        user_vectors = user_vectors.reshape(1, -1)

    user_vectors = user_vectors.astype(np.float32)
    scores, indices = index.search(user_vectors, k)

    batch_item_ids = []
    for row in indices:
        batch_item_ids.append([str(item_ids[i]) for i in row if i >= 0])

    return batch_item_ids, scores