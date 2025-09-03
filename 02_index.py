#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02_index.py — Da pages.json + sections.json → chunking → embedding FAISS (E5).
- Arricchisce il testo: title + summary + keywords + chunk
- Pulizia tipica OCR (ligature, sillabazioni, spazi, virgolette)
- Embedding con intfloat/multilingual-e5-base (prefisso 'passage:')
Scrive:
  - index.faiss
  - index_meta.json  (metadati paralleli alle righe FAISS con 'text' arricchito)
"""

import os
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ========= CONFIG =========
CHUNK_MAX_CHARS = int(os.environ.get("CHUNK_MAX_CHARS", "1200"))
CHUNK_OVERLAP   = int(os.environ.get("CHUNK_OVERLAP",   "180"))

EMBED_MODEL   = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-base")
EMBED_BATCH   = int(os.environ.get("EMBED_BATCH", "32"))
EMBED_NORMALIZE = True

# ========= TEXT CLEAN =========
LIGATURES = {
    "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
    "’": "'", "‘": "'", "“": '"', "”": '"', "—": "-", "–": "-"
}

def clean_text(t: str) -> str:
    if not t:
        return ""
    for a,b in LIGATURES.items():
        t = t.replace(a,b)
    # togli sillabazioni a capo: "fortifi-\n cazioni" -> "fortificazioni"
    t = re.sub(r'-\s*\n\s*(?=\w)', '', t)
    # normalizza unicode
    t = unicodedata.normalize("NFC", t)
    # compattazione whitespace
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()

def _split_by_chars(text: str, max_chars: int, overlap: int) -> List[str]:
    t = clean_text(text)
    if len(t) <= max_chars:
        return [t]
    chunks, i = [], 0
    while i < len(t):
        end = min(i + max_chars, len(t))
        cut = end
        rev = t[i:end][::-1]
        m = re.search(r'[.!?]\s', rev)
        if m and (end - (i + m.start())) > max_chars//2:
            cut = i + (end - (m.start()+1))
        chunk = t[i:cut].strip()
        if chunk:
            chunks.append(chunk)
        i = cut - overlap if cut - overlap > i else cut
    return chunks

def _concat_pages(pages_map: Dict[int, Dict[str, Any]], start: int, end: int) -> str:
    parts = []
    for p in range(start, end+1):
        v = pages_map.get(str(p), pages_map.get(p, {})).get("text", "")
        if v:
            parts.append(f"[Pag {p}]\n{v}")
    return "\n\n".join(parts).strip()

def build_chunks(pdf_name: str,
                 pages_map: Dict[int, Dict[str, Any]],
                 sections: List[Dict[str, Any]],
                 max_chars: int = CHUNK_MAX_CHARS,
                 overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    chunks = []
    seg_id = 1
    for s in sections:
        start, end = int(s["pages"][0]), int(s["pages"][1])
        title    = (s.get("title","") or f"Sezione {start}-{end}").strip()
        keywords = s.get("keywords", []) or []
        summary  = s.get("summary","") or ""
        body = _concat_pages(pages_map, start, end)
        if not body:
            continue
        body = clean_text(body)
        parts = _split_by_chars(body, max_chars, overlap)
        for j, part in enumerate(parts, start=1):
            cid = f"{Path(pdf_name).stem}_seg{seg_id}_c{j}"
            # testo ARRICCHITO: title + summary + keywords + chunk
            enrich = " ".join([title, summary, " ".join(keywords)]).strip()
            meta_text = (enrich + "\n" + part).strip() if enrich else part
            meta = {
                "id": cid,
                "pdf": pdf_name,
                "segment_id": seg_id,
                "segment_title": title,
                "segment_pages": list(range(start, end+1)),
                "segment_summary": summary,
                "segment_keywords": keywords,
                "chunk_index": j,
                "chunk_count_in_segment": len(parts),
                "text": meta_text
            }
            chunks.append(meta)
        seg_id += 1
    return chunks

def embed_and_save(chunks_meta: List[Dict[str, Any]], out_dir: Path):
    texts = [m["text"] for m in chunks_meta]
    # E5 richiede prefissi: passages
    if "e5" in EMBED_MODEL.lower():
        texts = [f"passage: {t}" for t in texts]
    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode(texts, batch_size=EMBED_BATCH, convert_to_numpy=True,
                       normalize_embeddings=True).astype("float32")
    if EMBED_NORMALIZE:
        faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(out_dir / "index.faiss"))
    (out_dir / "index_meta.json").write_text(json.dumps(chunks_meta, ensure_ascii=False, indent=2), encoding="utf-8")

def main():
    import argparse
    ap = argparse.ArgumentParser(description="02 — Chunking + Embedding (E5)")
    ap.add_argument("--pages", required=True, help="Path a pages.json")
    ap.add_argument("--sections", required=True, help="Path a sections.json")
    ap.add_argument("--out", default=None, help="Cartella output (default: stessa di pages.json)")
    ap.add_argument("--pdf-name", default=None, help="Nome file PDF (per metadati)")
    ap.add_argument("--max-chars", type=int, default=CHUNK_MAX_CHARS)
    ap.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    args = ap.parse_args()

    pages_path = Path(args.pages)
    out_dir = Path(args.out) if args.out else pages_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    pages = json.loads(pages_path.read_text(encoding="utf-8"))
    sections_obj = json.loads(Path(args.sections).read_text(encoding="utf-8"))
    sections = sections_obj.get("sections", sections_obj)

    pdf_name = args.pdf_name or (pages_path.parent.name + ".pdf")

    chunks = build_chunks(pdf_name, pages, sections, max_chars=args.max_chars, overlap=args.overlap)
    if not chunks:
        raise SystemExit("Nessun chunk costruito — controlla pages/sections.")
    embed_and_save(chunks, out_dir)
    print(json.dumps({"chunks": len(chunks), "out_dir": str(out_dir)}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
