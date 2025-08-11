import re
import json
import base64
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional

import fitz
import ollama
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageOps, ImageFilter

from embedding import VectorIndex

# ============ CONFIG ============
DEFAULT_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
DEFAULT_MODEL_VL = "qwen2.5vl"  # modello vision-language
OCR_LANG = "eng+ita"  # personalizzabile da CLI
DPI = 300
PATH_FILE = Path("File")
PATH_OUTPUT = Path("Output")
MAX_HEIGHT = 20000
RETRIES = 3
RETRY_BASE_DELAY = 1.5
PATH_EXTRACT_IMAGE = Path("immagini_estratte")
# =================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("doc-analyzer")

PROMPT = """
Agisci come un esperto di storia medievale italiana e analista di documenti accademici. Ti fornirò:
- Un'immagine contenente una o più pagine di un documento storico medievale.
- Il testo estratto tramite OCR.

Il tuo compito è:
1. Leggere e comprendere il contenuto testuale.
2. Generare un riassunto coerente dei contenuti storici del documento.
3. Identificare in quali pagine ci sono immagini.
4. Assegnare dei tag per il contenuto (massimo 15, singole parole o bigrammi).

Il risultato deve essere **solo** JSON:

{
  "riassunto": "Testo riassuntivo dei contenuti storici...",
  "tags": ["castelli", "guerra", "fortificazioni", "XII secolo"]
}
""".strip()


def set_tesseract(path: Optional[str] = None) -> None:
    pytesseract.pytesseract.tesseract_cmd = path or DEFAULT_TESSERACT


def safe_json_loads(s: str, extra: Dict) -> Optional[Dict]:
    cleaned = s.strip().replace("```json", "```").strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned[3:-3].strip()
    try:
        return (json.loads(cleaned) | extra)
    except Exception:
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if m:
            try:
                return (json.loads(m.group(0)) | extra)
            except Exception:
                return None
    return None


def ocr_image(img_path: Path, lang: str = OCR_LANG) -> str:
    img = Image.open(img_path)
    img = ImageOps.grayscale(img)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=150, threshold=3))
    return pytesseract.image_to_string(img, lang=lang)


def extract_images_from_pdf(pdf_path: Path) -> Dict:
    PATH_EXTRACT_IMAGE.mkdir(parents=True, exist_ok=True)
    out = {"immagini": []}
    try:
        doc = fitz.open(str(pdf_path))
        try:
            for i in range(len(doc)):
                page_num = i + 1
                page = doc[i]
                images_found = page.get_images(full=True)
                if not images_found:
                    log.info(f"Nessuna immagine embedded trovata nella pagina {page_num}")
                    continue
                for j, img in enumerate(images_found, start=1):
                    try:
                        xref = img[0]
                        base = doc.extract_image(xref)
                        ext = base["ext"]
                        out_path = PATH_EXTRACT_IMAGE / f"pagina{page_num}_img{j}.{ext}"
                        with open(out_path, "wb") as fh:
                            fh.write(base["image"])
                        out["immagini"].append({"pagina": page_num, "path": str(out_path)})
                        log.info(f"Estratta immagine: {out_path}")
                    except Exception as e:
                        log.error(f"Errore estrazione img {j} pag {page_num}: {e}")
        finally:
            doc.close()
    except Exception as e:
        log.error(f"Errore apertura PDF {pdf_path}: {e}")
    return out


def call_ollama_with_retry(*, model: str, messages: List[Dict], images: Optional[List[str]] = None) -> str:
    """
    Chiamata Ollama con retry/backoff, ritorna il contenuto testuale.
    images: lista di immagini base64 (se serve vision)
    """
    payload = [{
        "role": "user",
        "content": messages[0]["content"],
        **({"images": images} if images else {})
    }]

    for attempt in range(1, RETRIES + 1):
        try:
            resp = ollama.chat(model=model, messages=payload)
            return resp["message"]["content"]
        except Exception as e:
            wait = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            log.warning(f"Ollama fallita (tentativo {attempt}/{RETRIES}): {e}. Retry tra {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError("Chiamata a Ollama fallita dopo i retry")


class DocumentAnalyzer:
    CHOICES = {"1": "analyze_multi_page", "2": "analyze_single_page"}

    def __init__(self, pdf_path: Path, max_height: int = MAX_HEIGHT,
                 model_summary: str = DEFAULT_MODEL_VL,
                 ocr_lang: str = OCR_LANG):
        self.pdf_path = Path(pdf_path)
        self.max_height = max_height
        self.model_summary = model_summary
        self.ocr_lang = ocr_lang
        self.images = self.convert_pdf_to_images()
        self.output = {self.pdf_path.name: {}}
        PATH_OUTPUT.mkdir(exist_ok=True)

    def clear_output_directory(self) -> None:
        if not PATH_OUTPUT.exists():
            PATH_OUTPUT.mkdir(parents=True, exist_ok=True)
            return
        for p in PATH_OUTPUT.glob("*"):
            try:
                p.unlink()
            except Exception as e:
                log.warning(f"Impossibile rimuovere {p}: {e}")

    def convert_pdf_to_images(self) -> List[Image.Image]:
        log.info(f"Converto PDF in immagini (DPI={DPI})...")
        images = convert_from_path(str(self.pdf_path), dpi=DPI)
        log.info(f"Pagine convertite: {len(images)}")
        return images

    def start_analyzer(self, choice: str):
        method_name = self.CHOICES.get(choice)
        if not method_name:
            raise ValueError(f"Scelta non valida: {choice}")
        return getattr(self, method_name)()

    # ---------- Modalità 1: immagine unica (merge) ----------
    def analyze_multi_page(self):
        """
        Unisce le pagine in blocchi verticali (rispettando MAX_HEIGHT).
        Registra per ogni immagine di output il vettore di pagine coperte.
        """
        self.clear_output_directory()
        output_prefix = PATH_OUTPUT / "merged_image"

        widths = [img.width for img in self.images]
        max_width = max(widths)

        y = 0
        idx = 1
        pages_in_block = []
        merged = Image.new("RGB", (max_width, self.max_height), color="white")

        for page_idx, img in enumerate(self.images, start=1):
            h = img.height
            if y + h > self.max_height and pages_in_block:
                out_path = f"{output_prefix}_{idx}.jpg"
                merged.save(out_path, "JPEG", quality=95)
                self.output[self.pdf_path.name][Path(out_path).name] = {
                    "pages": pages_in_block[:],
                    "images": []
                }
                idx += 1
                merged = Image.new("RGB", (max_width, self.max_height), color="white")
                y = 0
                pages_in_block = []

            merged.paste(img, (0, y))
            y += h
            pages_in_block.append(page_idx)

        if pages_in_block:
            out_path = f"{output_prefix}_{idx}.jpg"
            merged.save(out_path, "JPEG", quality=95)
            self.output[self.pdf_path.name][Path(out_path).name] = {
                "pages": pages_in_block[:]
            }

        return self._process_output_images()

    # ---------- Modalità 2: pagina per pagina ----------
    def analyze_single_page(self):
        self.clear_output_directory()
        for i, img in enumerate(self.images, start=1):
            out = PATH_OUTPUT / f"page_{i:04d}.jpg"
            out.write_bytes(self._pil_to_jpeg_bytes(img))
            # registra pagine e images per ogni file pagina
            self.output[self.pdf_path.name][out.name] = {"pages": [i], "images": []}
        return self._process_output_images()

    @staticmethod
    def _pil_to_jpeg_bytes(img: Image.Image) -> bytes:
        from io import BytesIO
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return buf.getvalue()

    def _generate_sql_from_summary(self, summary_text: str, tags: List[str], pages: List[int]) -> str:
        """
        Crea SQL deterministico a partire da riassunto+tag.
        - Prende i primi N tag come 'nome'.
        - Usa frasi chiave dal riassunto come 'descrizione'.
        - Collega alle pagine fornite.
        """
        N = min(10, len(tags))
        chosen = tags[:N] or ["generico"]
        sentences = [s.strip() for s in re.split(r'[.!?]\s+', summary_text) if len(s.strip()) > 0]
        snippets = (sentences[:N] or ["caratteristica estratta dal riassunto"])

        page_list = pages or [None] * N

        lines = [
            "CREATE TABLE IF NOT EXISTS caratteristiche (",
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,",
            "  nome TEXT,",
            "  descrizione TEXT,",
            "  pagina INTEGER",
            ");"
        ]
        for i, nome in enumerate(chosen):
            desc = (snippets[i] if i < len(snippets) else snippets[-1])[:400]
            page = page_list[i] if i < len(page_list) else None
            lines.append(
                "INSERT INTO caratteristiche (nome, descrizione, pagina) "
                f"VALUES ({json.dumps(nome)}, {json.dumps(desc)}, {'NULL' if page is None else page});"
            )
        return "\n".join(lines)

    def _summarize_with_vl(self, img_path: Path) -> Optional[Dict]:
        text = ocr_image(img_path, lang=self.ocr_lang)
        self.output[self.pdf_path.name].setdefault(img_path.name, {}).setdefault("ocr_text", text)
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        pages = self.output[self.pdf_path.name].get(img_path.name, {}).get("pages", [])
        prompt = (
                PROMPT + f"\n\nPagine coperte dal blocco: {pages}\n"
                         "Testo estratto OCR:\n" + text
        )
        content = call_ollama_with_retry(
            model=self.model_summary,
            messages=[{"role": "user", "content": prompt}],
            images=[b64]
        )
        data = safe_json_loads(content, {"pages": pages})
        return data

    def _process_output_images(self) -> Dict:
        jpgs = sorted(PATH_OUTPUT.glob("*.jpg"))

        for img_path in jpgs:
            try:
                resp = self._summarize_with_vl(img_path)
                if not resp:
                    log.warning(f"Risposta non JSON per {img_path.name}")
                    continue

                summary = resp.get("riassunto", "").strip()
                tags = resp.get("tags", [])
                pages = resp.get("pages", [])
                sql_query = self._generate_sql_from_summary(summary, tags, pages)

                self.output[self.pdf_path.name][img_path.name] = {
                    **self.output[self.pdf_path.name].get(img_path.name, {}),
                    "riassunto": summary,
                    "tags": tags,
                    "sql": sql_query
                }

            except Exception as e:
                log.error(f"Errore su {img_path.name}: {e}")
        self._extract_images_from_pdf()
        vec = VectorIndex()
        texts, metas = [], []

        for fname, entry in self.output[self.pdf_path.name].items():
            emb_text = self._embedding_text_from_entry(entry)
            if not emb_text:
                continue
            texts.append(emb_text)
            metas.append({
                "file": fname,
                "pages": entry.get("pages", []),
                "tags": entry.get("tags", []),
                "images": entry.get("images", []),
                "pdf": self.pdf_path.name
            })

        vec.add_texts(texts, metas)
        vec.save("index.faiss", "index_meta.json")
        return {self.pdf_path.name: self.output[self.pdf_path.name]}

    def _extract_images_from_pdf(self) -> None:
        PATH_EXTRACT_IMAGE.mkdir(parents=True, exist_ok=True)
        try:
            doc = fitz.open(str(self.pdf_path))
            try:
                for i in range(len(doc)):
                    page_num = i + 1
                    page = doc[i]
                    images_found = page.get_images(full=True)
                    if not images_found:
                        log.info(f"Nessuna immagine embedded trovata nella pagina {page_num}")
                        continue

                    for j, img in enumerate(images_found, start=1):
                        try:
                            xref = img[0]
                            base = doc.extract_image(xref)
                            ext = base["ext"]
                            out_path = PATH_EXTRACT_IMAGE / f"pagina{page_num}_img{j}.{ext}"
                            with open(out_path, "wb") as fh:
                                fh.write(base["image"])

                            for _, entry in self.output[self.pdf_path.name].items():
                                pages = entry.get("pages", [])
                                if page_num in pages:
                                    entry.setdefault("images", []).append(str(out_path))

                            log.info(f"Estratta immagine: {out_path}")
                        except Exception as e:
                            log.error(f"Errore estrazione img {j} pag {page_num}: {e}")
            finally:
                doc.close()
        except Exception as e:
            log.error(f"Errore apertura PDF {self.pdf_path}: {e}")

    def _embedding_text_from_entry(self, entry: Dict) -> str:
        parts = []
        if entry.get("riassunto"):
            parts.append(entry["riassunto"])
        if entry.get("tags"):
            parts.append(" ".join(entry["tags"]))
        if entry.get("ocr_text"):
            parts.append(entry["ocr_text"][:4000])  # limite di sicurezza
        return "\n".join(parts).strip()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analizzatore PDF storico (VL + OCR)")
    parser.add_argument("--pdf", type=str, default=str(PATH_FILE / "The_archaeology_of_the_medieval_Castle_o.pdf"))
    parser.add_argument("--mode", choices=["1", "2"], default="1",
                        help="1=merge multipagina, 2=pagina per pagina (consigliato)")
    parser.add_argument("--tesseract", type=str, default=DEFAULT_TESSERACT)
    parser.add_argument("--model-vl", type=str, default=DEFAULT_MODEL_VL)
    parser.add_argument("--ocr-lang", type=str, default=OCR_LANG)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    set_tesseract(args.tesseract)
    analyzer = DocumentAnalyzer(
        pdf_path=Path(args.pdf),
        model_summary=args.model_vl,
        ocr_lang=args.ocr_lang
    )

    log.info("Scegli il metodo di analisi: 1=merge multi-pagina, 2=pagina per pagina")
    results = analyzer.start_analyzer(args.mode)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    with open("prova.json", "w") as outfile:
        json.dump(results, outfile)


if __name__ == "__main__":
    main()
