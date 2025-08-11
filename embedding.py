import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List

class VectorIndex:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.meta = []  # metadati paralleli alle righe dell’indice

    def add_texts(self, texts: List[str], metadatas: List[Dict]):
        emb = self.model.encode(texts, normalize_embeddings=True)
        emb = np.asarray(emb, dtype="float32")
        if self.index is None:
            self.index = faiss.IndexFlatIP(emb.shape[1])  # cosine (con normalized=True)
        self.index.add(emb)
        self.meta.extend(metadatas)

    def search(self, query: str, k: int = 5) -> List[Dict]:
        q = self.model.encode([query], normalize_embeddings=True).astype("float32")
        D, I = self.index.search(q, k)
        out = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            item = dict(self.meta[idx])
            item["score"] = float(score)
            out.append(item)
        return out

    def save(self, path_idx: str, path_meta: str):
        faiss.write_index(self.index, path_idx)
        with open(path_meta, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def load(self, path_idx: str, path_meta: str):
        self.index = faiss.read_index(path_idx)
        with open(path_meta, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
