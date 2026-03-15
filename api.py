from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from Retrieval.Train.candidate_retrieval import load_faiss_index, retrieve_topk
from Ranking.wide_deep_model import WideAndDeepRanker

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "Retrieval" / "artifacts"
RANKING_ARTIFACT_DIR = ROOT / "Ranking" / "artifacts"

class RecommendRequest(BaseModel):
    user_id: str = Field(..., description="User id used to generate recommendations")
    top_k: int = Field(10, ge=1, le=100)
    candidate_k: int = Field(100, ge=1, le=500)

class RecommendResponse(BaseModel):
    user_id: str
    top_k: int
    recommendations: List[str] # Trả về danh sách chuỗi ID

class RecommenderService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.index, self.index_item_ids = load_faiss_index(
            ARTIFACT_DIR / "index" / "item_faiss.index",
            ARTIFACT_DIR / "index" / "item_ids.npy",
        )
        self.user_embedding_map = self._load_user_embeddings()
        self.item_embedding_map = self._load_embedding_npz(
            ARTIFACT_DIR / "embeddings" / "item_embeddings_train.npz"
        )
        self.ranker = self._load_ranker(RANKING_ARTIFACT_DIR / "wide_deep_best.pt")
        self.ranker.eval()

    @staticmethod
    def _load_embedding_npz(npz_path: Path) -> Dict[str, np.ndarray]:
        payload = np.load(npz_path, allow_pickle=True)
        ids = payload["ids"]
        vectors = payload["vectors"].astype(np.float32)
        return {str(i): vectors[idx] for idx, i in enumerate(ids)}

    def _load_user_embeddings(self) -> Dict[str, np.ndarray]:
        emb_dir = ARTIFACT_DIR / "embeddings"
        users_train = self._load_embedding_npz(emb_dir / "user_embeddings_train.npz")
        users_val = self._load_embedding_npz(emb_dir / "user_embeddings_val.npz")
        users_train.update(users_val)
        return users_train

    def _load_ranker(self, checkpoint_path: Path) -> WideAndDeepRanker:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        input_dim = int(checkpoint["input_dim"])
        model = WideAndDeepRanker(input_dim=input_dim).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def recommend(self, user_id: str, top_k: int, candidate_k: int) -> List[str]:
        user_vec = self.user_embedding_map.get(str(user_id))
        if user_vec is None:
            raise HTTPException(status_code=404, detail=f"user_id '{user_id}' not found")

        user_vec = self._l2_normalize(user_vec).astype(np.float32)
        candidate_ids_batch, _ = retrieve_topk(self.index, self.index_item_ids, user_vec, k=candidate_k)
        candidate_ids = candidate_ids_batch[0] if candidate_ids_batch else []

        # Chạy Ranker để sắp xếp, lấy ID sau khi xong
        scored_items = []
        with torch.no_grad():
            for item_id in candidate_ids:
                item_vec = self.item_embedding_map.get(item_id)
                if item_vec is None: continue
                
                item_vec = self._l2_normalize(item_vec)
                model_input = np.concatenate([user_vec, item_vec, user_vec * item_vec], axis=0)
                x = torch.from_numpy(model_input.astype(np.float32)).unsqueeze(0).to(self.device)
                score = torch.sigmoid(self.ranker(x)).item()
                scored_items.append((item_id, score))

        scored_items.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in scored_items[:top_k]] # Chỉ trả về ID

app = FastAPI(title="RecSys Inference API")
service = RecommenderService()

@app.get("/health")
def healthcheck(): return {"status": "ok"}

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    recommendations = service.recommend(req.user_id, req.top_k, req.candidate_k)
    return RecommendResponse(user_id=req.user_id, top_k=req.top_k, recommendations=recommendations)

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
        <head><title>RecSys Test</title></head>
        <body style="font-family: sans-serif; padding: 20px;">
            <h2>Hệ thống Gợi ý (Chỉ ID)</h2>
            <input type="text" id="userId" placeholder="Nhập User ID...">
            <button onclick="getRecs()">Lấy danh sách</button>
            <div style="margin-top: 20px;">
                <strong>Kết quả:</strong>
                <ul id="itemList"></ul>
            </div>

            <script>
                async function getRecs() {
                    const userId = document.getElementById('userId').value;
                    const list = document.getElementById('itemList');
                    list.innerHTML = "Đang tải...";
                    
                    const response = await fetch('/recommend', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({user_id: userId, top_k: 10, candidate_k: 50})
                    });
                    const data = await response.json();
                    
                    list.innerHTML = "";
                    if (data.recommendations) {
                        data.recommendations.forEach(id => {
                            const li = document.createElement('li');
                            li.textContent = "Sản phẩm ID: " + id;
                            list.appendChild(li);
                        });
                    } else {
                        list.innerHTML = "Lỗi: " + data.detail;
                    }
                }
            </script>
        </body>
    </html>
    """
