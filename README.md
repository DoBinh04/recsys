# RecSys Project

A two-stage recommender system:
- **Retrieval**: Two-Tower + FAISS for fast candidate generation.
- **Ranking**: Wide & Deep to re-rank candidates by relevance.
- **Serving**: FastAPI for online inference.

---

## 1) Architecture Overview

### Project structure
- `Pipeline/`: raw data loading, preprocessing, interaction weighting, and temporal train/val/test split.
- `Retrieval/`: user/item feature engineering, ID mappings, Two-Tower training, embedding export, FAISS indexing.
- `Ranking/`: candidate construction and Wide & Deep ranker training.
- `Evaluation/`: offline evaluation scripts and metrics outputs.
- `api.py`: online inference API (artifact loading, retrieval, ranking).

### End-to-end flow
1. **Offline pipeline**: process data + train Retrieval + train Ranking + save artifacts.
2. **Online inference**: receive `user_id` -> retrieval returns candidates -> ranking scores candidates -> return top-k.

---

## 2) Offline Pipeline (Detailed)

### 2.1 Preprocessing and filtering
In `Pipeline/preprocess.py`, interaction data is:
- Converted from millisecond `timestamp` to datetime.
- Sorted by timestamp.
- Deduplicated.
- Filtered to remove **low-activity users** (`min_user_inter=5` by default).
- Filtered to remove **low-activity items** (`min_item_inter=5` by default).

This step reduces sparsity and noise before learning embeddings.

### 2.2 Implicit-feedback event weighting
In `Pipeline/interactions.py`, each event is assigned a weight:
- `view` -> `0.02`
- `addtocart` -> `0.3`
- `transaction` (purchase) -> `1.0`

This `weight` column is carried into Retrieval training and used directly in the loss, so stronger behaviors have larger learning impact.

### 2.3 Temporal train/val/test split
In `Pipeline/split.py`, data is sorted by time, then split as:
- `train`: 70%
- `val`: 15%
- `test`: 15%

Temporal splitting better matches production behavior (predicting future interactions from past data) and helps reduce leakage.

---

## 3) Retrieval: Feature Engineering + Two-Tower

### 3.1 Retrieval feature engineering
After the base pipeline, train/val/test data is enriched in `Retrieval/Preprocessing/build_training_data.py` with two feature groups:

#### A) User features (`Retrieval/Features/user_features.py`)
1. **Recent interaction sequence**
   - `recent_items`: up to 5 most recent items before the current interaction (with fixed-length padding).
2. **User activity statistics**
   - `total_views`
   - `total_addtocart`
   - `total_transactions`
   - `unique_items`
   - `addtocart_rate = total_addtocart / (total_views + 1)`
   - `purchase_rate = total_transactions / (total_views + 1)`

#### B) Item features (`Retrieval/Features/item_features.py`)
1. **Item popularity/statistics**
   - `total_views_item`
   - `total_addtocart_item`
   - `total_transactions_item`
   - `cart_rate`
   - `purchase_rate_item`
2. **Category hierarchy features**
   - `root`: root category
   - `leaf`: leaf/current category
   - `depth`: category path depth

These features are merged to produce `*_ready.parquet` datasets for Retrieval training.

### 3.2 Two-Tower training
In `Retrieval/train_retrieval.py` + `Retrieval/Train/two_tower_model.py`:
- **User tower** learns user vectors from:
  - user ID embedding
  - pooled embedding of `recent_items`
  - user numeric features
- **Item tower** learns item vectors from:
  - item ID embedding
  - category embeddings (`root`, `leaf`)
  - item numeric features

### 3.3 In-batch negative sampling
In `Retrieval/Train/retrieval_loss.py`:
- A full in-batch similarity matrix is computed (`user_vec @ item_vec.T`).
- Aligned pairs in the same row/column index are positives.
- Other items in the batch act as **in-batch negatives**.
- Cross-entropy is multiplied by event `weight` to prioritize purchase > add-to-cart > view.

### 3.4 Retrieval optimization objective
Retrieval is used for **candidate generation**, so validation is tracked with recall-oriented top-k metrics (candidate coverage quality).

### 3.5 Retrieval outputs
After training, Retrieval produces:
- Two-Tower checkpoints (`.pt`) under `Retrieval/artifacts/checkpoints/`.
- `user_embeddings_*.npz` and `item_embeddings_*.npz` under `Retrieval/artifacts/embeddings/`.
- FAISS index files under `Retrieval/artifacts/index/`.

You can consider the `.pt` file as the trained Retrieval model, while user/item embeddings plus FAISS index are the key downstream artifacts for candidate retrieval.

---

## 4) FAISS Indexing (Candidate Retrieval)

After item embeddings are exported from Two-Tower:
1. Normalize item vectors.
2. Build FAISS index (`item_faiss.index`).
3. Save item ID mapping (`item_ids.npy`).

During inference (or ranking data construction):
- Encode user embedding.
- Query FAISS for top-`candidate_k` nearest items.

This provides efficient ANN retrieval at scale.

---

## 5) Ranking: Wide & Deep optimizing NDCG

### 5.1 Ranking input features
In `Ranking/train_wide_deep.py`, ranker inputs are built from Retrieval outputs:
- `user_embedding`
- candidate `item_embedding`
- `cross feature = user_embedding * item_embedding` (element-wise)

Final input vector is concatenated as:
`[user_emb, item_emb, user_emb * item_emb]`

### 5.2 Candidate set for ranking
- Candidates are retrieved from FAISS top-k.
- In training, positive items can be force-included (`force_include_positive=True`) to ensure supervised positive learning signals.

### 5.3 Ranking objective
- Training loss: `BCEWithLogitsLoss`.
- Validation metrics include `Recall@10` and **`NDCG@10`**.
- Best checkpoint is selected by `best_val_ndcg_at_10`.

### 5.4 Ranking outputs
Model artifacts are saved under `Ranking/artifacts/` as `.pt` checkpoints (for example, `wide_deep_best.pt`).
This is the Wide & Deep model used to score and re-rank Retrieval candidates.

---

## 6) Online Inference Pipeline

1. Client calls API with `user_id`.
2. System encodes user -> queries FAISS -> gets `candidate_k` items.
3. Ranker scores candidates using user/item embeddings + cross features.
4. API returns final `top_k` ranked item IDs.

---

## 7) How to run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run API locally
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Run with Docker
```bash
docker build -t recsys-api .
docker run --rm -p 8000:8000 recsys-api
```

Swagger docs: `http://localhost:8000/docs`

---

## 8) Artifact readiness notes
Before running the API, ensure all required artifacts exist:
- Retrieval checkpoints/embeddings/index in `Retrieval/artifacts/`.
- Ranking checkpoints in `Ranking/artifacts/`.

If artifacts are missing, the API may start in degraded mode until offline training is completed.
