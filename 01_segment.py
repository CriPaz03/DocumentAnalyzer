#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_segment.py — Estrazione per pagina (nativo -> OCR fallback) + segmentazione via LLM.
Scrive SOLO:
- pages.json     (mappa: {page_num: {"text": "...", "source": "pdf|ocr"}})
- sections.json  ({"sections":[{"title":..., "pages":[start,end], "summary":..., "keywords":[...]}, ...]})
"""

import os
import re
import io
import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import requests

# ========= LOG & CONFIG =========
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("segment")

# OCR / PDF
DEFAULT_TESSERACT = os.environ.get("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
DPI = int(os.environ.get("PDF_DPI", "300"))
OCR_LANG = os.environ.get("OCR_LANG", "eng+ita")
NATIVE_MINLEN = int(os.environ.get("NATIVE_MINLEN", "220"))
MAX_PAGE_CHARS = int(os.environ.get("MAX_PAGE_CHARS", "4000"))

# LLM (segmentation)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = "AIzaSyBcG4GNT04c6Zfm1nlT3w5NFngyMltHEJw"  # se non presente -> fallback Ollama
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct-q5_K_M")
LLM_RETRIES = int(os.environ.get("LLM_RETRIES", "2"))
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "60"))
MAX_PAGES_PER_BATCH = int(os.environ.get("MAX_PAGES_PER_BATCH", "30"))
MAX_BATCH_CHARS = int(os.environ.get("MAX_BATCH_CHARS", "60000"))

PROMPT_SEGMENT = """Sei un assistente per la segmentazione di testi accademici/storici.
Riceverai un elenco di pagine e un estratto (troncato) del loro contenuto.

Dividi l’INTERO intervallo di pagine fornite in SEZIONI semantiche coerenti.
Regole:
- Ogni sezione ha:
  - "title": stringa breve (es. "Capitolo 2 – Fortificazioni normanne")
  - "pages": [start, end] (indici 1-based rispetto al PDF reale)
  - "summary": riassunto sintetico (4-6 frasi) dei contenuti della sezione
  - "keywords": 5-15 tag (parole o bigrammi)
- Le sezioni devono coprire l’intero intervallo di pagine dato, senza overlap e senza buchi.
- Rispetta i numeri di pagina reali passati in input (non inventare pagine).

OUTPUT: Rispondi SOLO con JSON valido e con esatta struttura:
{"sections":[{"title":"string","pages":[start,end],"summary":"string","keywords":["string","..."]}]}
""".strip()

REPAIR_PROMPT = """Ti fornisco un oggetto che DOVREBBE essere JSON valido per lo schema
{"sections":[{"title":"string","pages":[start,end],"summary":"string","keywords":["..."]}]}
Se non è valido, restituisci lo stesso contenuto ma in JSON valido AL 100%, senza testo fuori dal JSON.
Rispondi SOLO con il JSON.
"""


# ========= UTILS =========
def set_tesseract(cmd: Optional[str] = None):
    pytesseract.pytesseract.tesseract_cmd = cmd or DEFAULT_TESSERACT


def _json_loads_robust(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        if "{" in text and "}" in text:
            cut = text[text.find("{"): text.rfind("}") + 1]
            return json.loads(cut)
        raise


def _http_retry(method, url, **kwargs):
    last = None
    for attempt in range(LLM_RETRIES + 1):
        try:
            r = requests.request(method, url, timeout=LLM_TIMEOUT, **kwargs)
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"HTTP failed: {last}")


def _ocr_page_pix(page: "fitz.Page", dpi: int = DPI) -> str:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    try:
        return pytesseract.image_to_string(img, lang=OCR_LANG)
    except Exception as e:
        log.error("OCR errore: %s", e)
        return ""

def normalize_text(t: str) -> str:
    t = re.sub(r'-\n(?=\w)', '', t)       # togli sillabazione
    t = re.sub(r'\s+\n', '\n', t)         # spazi prima di newline
    t = re.sub(r'\n{3,}', '\n\n', t)      # max doppio newline
    t = re.sub(r'[ \t]{2,}', ' ', t)      # spazi multipli
    return t.strip()

def extract_text_per_page(pdf_path: Path) -> Dict[int, Dict[str, Any]]:
    """Ritorna {n: {"text": "...", "source": "pdf|ocr"}}"""
    res = {}
    doc = fitz.open(str(pdf_path))
    try:
        for i in range(len(doc)):
            p = doc[i];
            n = i + 1
            txt = p.get_text("text") or ""
            src = "pdf"
            if len(txt.strip()) < NATIVE_MINLEN:
                txt = _ocr_page_pix(p, dpi=DPI)
                src = "ocr"
            if len(txt) > MAX_PAGE_CHARS:
                txt = txt[:MAX_PAGE_CHARS]
            res[n] = {"text": normalize_text(txt), "source": src}
    finally:
        doc.close()
    return res


def _pages_payload(pages: List[Tuple[int, str]]) -> Dict[str, Any]:
    return {"pages": [{"page": p, "text": (t or "")[:MAX_PAGE_CHARS]} for p, t in pages]}


def call_gemini(prompt: str, payload_pages: dict) -> dict:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY non impostata")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    headers = {"Content-Type": "application/json", "X-Goog-Api-Key": GEMINI_API_KEY}
    body = {
        "systemInstruction": {"role": "system", "parts": [{"text": "Rispondi SOLO con JSON valido."}]},
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]},
            {"role": "user", "parts": [{"text": json.dumps(payload_pages, ensure_ascii=False)}]}
        ]
    }
    r = _http_retry("POST", url, headers=headers, data=json.dumps(body))
    data = r.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    return _json_loads_robust(text)


def call_ollama(prompt: str, payload_pages: dict) -> dict:
    import ollama
    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt},
                  {"role": "user", "content": json.dumps(payload_pages, ensure_ascii=False)}],
        format="json",
        options={"temperature": 0, "num_ctx": 8192}
    )
    content = resp["message"]["content"]
    try:
        return _json_loads_robust(content)
    except Exception:
        repair = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "system", "content": "Sei un riparatore di JSON."},
                      {"role": "user", "content": REPAIR_PROMPT + "\n\n---\n" + content + "\n---"}],
            format="json",
            options={"temperature": 0}
        )
        return _json_loads_robust(repair["message"]["content"])


def segment_document(pages_map: Dict[int, Dict[str, Any]], book_title: str) -> List[Dict[str, Any]]:
    """
    Segmenta il documento e include il titolo del libro nei metadati

    Args:
        pages_map: Mappa delle pagine estratte
        book_title: Titolo/nome del libro da includere nelle citazioni
    """
    nums = sorted(pages_map.keys())
    all_secs: List[Dict[str, Any]] = []
    batch: List[Tuple[int, str]] = []
    charcount = 0

    for i, p in enumerate(nums):
        txt = pages_map[p].get("text", "")
        batch.append((p, txt))
        charcount += len(txt)

        is_last = (i == len(nums) - 1)
        need_flush = (len(batch) >= MAX_PAGES_PER_BATCH) or (charcount >= MAX_BATCH_CHARS) or is_last

        if need_flush:
            payload = _pages_payload(batch)
            try:
                res = call_gemini(PROMPT_SEGMENT, payload)
            except Exception as e:
                log.warning("Gemini fallito, uso Ollama fallback: %s", e)
                res = call_ollama(PROMPT_SEGMENT, payload)

            secs = []
            for sec in res.get("sections", []):
                title = str(sec.get("title", "")).strip()
                pages = sec.get("pages", [])
                summary = str(sec.get("summary", "")).strip()
                keywords = [str(k).strip() for k in sec.get("keywords", []) if str(k).strip()]
                if title and isinstance(pages, list) and len(pages) == 2:
                    try:
                        start, end = int(pages[0]), int(pages[1])
                        if start <= end:
                            # ✅ AGGIUNGE IL NOME DEL LIBRO
                            secs.append({
                                "book_title": book_title,  # <-- NUOVO CAMPO
                                "title": title,
                                "pages": [start, end],
                                "summary": summary,
                                "keywords": keywords
                            })
                    except Exception:
                        pass
            all_secs.extend(secs)
            batch, charcount = [], 0

    # Resto della logica di merge invariata
    def jaccard(a: List[str], b: List[str]) -> float:
        sa, sb = set(map(str.lower, a)), set(map(str.lower, b))
        inter = len(sa & sb);
        union = len(sa | sb) or 1
        return inter / union

    all_secs = sorted(all_secs, key=lambda s: (s["pages"][0], s["pages"][1]))
    merged = []
    for s in all_secs:
        if not merged:
            merged.append(s);
            continue
        last = merged[-1]
        contigue = last["pages"][1] + 1 == s["pages"][0]
        same_title = last["title"].strip().lower() == s["title"].strip().lower()
        similar = jaccard(last.get("keywords", []), s.get("keywords", [])) >= 0.6
        if contigue and (same_title or similar):
            last["pages"][1] = s["pages"][1]
            last["summary"] = (last.get("summary", "") + " " + s.get("summary", "")).strip()
            last["keywords"] = sorted(list(set(last.get("keywords", []) + s.get("keywords", []))))
            # Mantiene il book_title del primo chunk
        else:
            merged.append(s)

    return merged


def main():
    import argparse
    ap = argparse.ArgumentParser(description="01 — Estrazione e segmentazione")
    ap.add_argument("--pdf", required=True, help="Percorso PDF")
    ap.add_argument("--out", default=None,
                    help="Cartella dove salvare pages.json e sections.json (default: accanto al PDF)")
    args = ap.parse_args()

    set_tesseract()
    pdf = Path(args.pdf)
    out_dir = Path(args.out) if args.out else pdf.parent

    log.info("Estrazione testo per pagina...")
    pages = extract_text_per_page(pdf)

    (out_dir / "pages.json").write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("pages.json scritto in %s", out_dir)

    log.info("Segmentazione via LLM...")
    sections = segment_document({int(k): v for k, v in pages.items()} if isinstance(pages, dict) else pages)
    (out_dir / "sections.json").write_text(json.dumps({"sections": sections}, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    log.info("sections.json scritto in %s", out_dir)


if __name__ == "__main__":
    main()
