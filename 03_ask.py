#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03_ask.py — Query RAG veloce e robusta:
- BM25 di default sul testo arricchito (chunk + summary + keywords + title)
- Denso FAISS in parallelo
- Fusione semplice: score = λ * dense_norm + (1-λ) * bm25_norm
- E5: prefisso 'query:' per la query

Env opzionali:
  EMBED_MODEL  (default intfloat/multilingual-e5-base)
  BM25_K       (default 120)
  DENSE_K      (default 120)
  LAMBDA       (default 0.6)  # peso del denso nella fusione
  GEMINI_MODEL (default gemini-2.0-flash), GEMINI_API_KEY, OLLAMA_MODEL
"""

import os
import json
import argparse
from typing import Any, Dict, List

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import requests

# ====== CONFIG ======
EMBED_MODEL = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-base")
BM25_K  = int(os.environ.get("BM25_K", "120"))
DENSE_K = int(os.environ.get("DENSE_K", "120"))
LAMBDA  = float(os.environ.get("LAMBDA", "0.6"))  # peso del denso

# LLM
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct-q5_K_M")

# ====== MODEL CACHE ======
_MODEL = None
def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(EMBED_MODEL)
    return _MODEL

def _encode(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0,1), dtype="float32")
    model = _get_model()
    emb = model.encode(texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    faiss.normalize_L2(emb)
    return emb

# ====== IO ======
def load_index(index_path: str, meta_path: str):
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta

# ====== HELPERS ======
def _normalize_query(q: str) -> str:
    return q.replace("’","'").replace("‘","'").replace("“",'"').replace("”",'"').strip()

def _bm25_text(m: Dict[str, Any]) -> str:
    parts = [
        m.get("text",""),  # già arricchito in 02_index.py
        # fallback (se qualcuno ha indici vecchi):
        m.get("segment_summary",""),
        " ".join(m.get("segment_keywords",[]) or []),
        m.get("segment_title",""),
    ]
    return " \n ".join([p for p in parts if p]).strip()

def _minmax_on_keys(d: Dict[int,float], keys: List[int]) -> Dict[int,float]:
    vals = [d.get(k, 0.0) for k in keys]
    if not vals:
        return {k: 0.0 for k in keys}
    mn, mx = min(vals), max(vals)
    if mx <= mn:
        return {k: 0.0 for k in keys}
    return {k: (d.get(k,0.0) - mn) / (mx - mn) for k in keys}

def _maybe_add_e5_prefix(texts: List[str], kind: str) -> List[str]:
    if "e5" in EMBED_MODEL.lower():
        if kind == "query":
            return [f"query: {t}" for t in texts]
        else:
            return [f"passage: {t}" for t in texts]
    return texts

# ====== RETRIEVAL ======
def retrieve(index, meta, query: str, top_k: int = 10, disable_bm25: bool = False) -> List[Dict[str, Any]]:
    q = _normalize_query(query)

    # Dense: top DENSE_K
    q_emb = _encode(_maybe_add_e5_prefix([q], "query"))
    D, I = index.search(q_emb, max(DENSE_K, top_k))
    dense_rows = [int(r) for r in I[0].tolist() if r >= 0]
    dense_scores = {row: float(score) for row, score in zip(dense_rows, D[0][:len(dense_rows)])}

    # BM25: top BM25_K (default ON)
    bm25_rows = []
    bm25_scores = {}
    if not disable_bm25:
        docs = [_bm25_text(m).lower().split() for m in meta]
        if any(len(d)>0 for d in docs):
            bm25 = BM25Okapi(docs)
            s = bm25.get_scores(q.lower().split())
            order = np.argsort(s)[::-1][:BM25_K]
            bm25_rows = [int(i) for i in order]
            bm25_scores = {int(i): float(s[i]) for i in order}

    # Candidati = unione
    cand = list(dict.fromkeys(dense_rows + bm25_rows))
    if not cand:
        return []

    # Normalizza per i soli candidati
    dense_norm = _minmax_on_keys(dense_scores, cand)
    bm25_norm  = _minmax_on_keys(bm25_scores,  cand)

    # Fusione lineare
    fused = []
    for row in cand:
        score = LAMBDA * dense_norm.get(row, 0.0) + (1.0 - LAMBDA) * bm25_norm.get(row, 0.0)
        m = dict(meta[row])
        m["row"] = row
        m["score_dense"] = dense_norm.get(row, 0.0)
        m["score_bm25"]  = bm25_norm.get(row, 0.0)
        m["score"] = float(score)
        fused.append(m)

    fused.sort(key=lambda x: x["score"], reverse=True)
    hits = fused[:top_k]
    for r,h in enumerate(hits, start=1):
        h["rank"] = r
    return hits

def dedup_snippets(hits, max_passages=5, max_chars=350):
    out, seen = [], set()
    for h in hits:
        key = (h.get("segment_title",""), tuple(h.get("segment_pages",[])))
        if key in seen:
            continue
        txt = (h.get("text","") or "").strip().replace("\n"," ")
        if not txt:
            continue
        out.append({**h, "text": txt[:max_chars]})
        seen.add(key)
        if len(out) >= max_passages:
            break
    return out

# ====== ANSWER ======
def llm_answer(query: str, hits: List[Dict[str, Any]], provider: str = "auto") -> str:
    evidence = dedup_snippets(hits, max_passages=5, max_chars=350)
    context = []
    for h in evidence:
        libray = h.get("pdf")
        seg = h.get("segment_title","")
        pgs = h.get("segment_pages",[])
        snippet = h["text"]
        context.append(f"Libro: {libray} - {seg} (pagine {pgs}): {snippet}")
    ctx = "\n".join(context)

    prompt = f"""Rispondi in modo accurato alla domanda usando SOLTANTO le informazioni nel CONTENUTO.
Includi una sezione finale 'Fonti' elencando le pagine PDF dei chunk usati e il libro.
Se non trovi la risposta nel contenuto, dì esplicitamente che non è presente.

DOMANDA: {query}

CONTENUTO:
{ctx}
"""
    if (provider == "gemini" or provider == "auto") and GEMINI_API_KEY:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
        headers = {"Content-Type":"application/json", "X-Goog-Api-Key": GEMINI_API_KEY}
        body = {"generationConfig": {"temperature": 0.2}, "contents": [{"role":"user","parts":[{"text": prompt}]}]}
        r = requests.post(url, headers=headers, data=json.dumps(body), timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    try:
        import ollama
        resp = ollama.chat(model=OLLAMA_MODEL, messages=[{"role":"user","content": prompt}], options={"temperature": 0.2})
        return resp["message"]["content"]
    except Exception:
        return ctx  # evidence-only fallback

# ====== CLI ======
def main():
    ap = argparse.ArgumentParser(description="03 — Query RAG (BM25 + Dense fusion)")
    ap.add_argument("--index", required=True, help="Path index.faiss")
    ap.add_argument("--meta", required=True, help="Path index_meta.json")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--no-bm25", action="store_true", help="Disattiva BM25 (solo denso)")
    ap.add_argument("query", nargs="+")
    args = ap.parse_args()
    query = " ".join(args.query)

    index, meta = load_index(args.index, args.meta)
    hits = retrieve(index, meta, query, top_k=args.topk, disable_bm25=args.no_bm25)

    # astensione conservativa
    avg = float(np.mean([h["score"] for h in hits])) if hits else 0.0
    if args.no_llm or avg < 0.1:
        print(json.dumps(hits, ensure_ascii=False, indent=2)); return

    print(llm_answer(query, hits, provider="auto"))

if __name__ == "__main__":
    main()
