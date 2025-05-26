import json
from collections import defaultdict

import fitz  # PyMuPDF
import pdfplumber
import os

def estrai_testo_da_pdf(pdf_path):
    dict_result = {}
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            testo = page.extract_text() or ""
            dict_result.setdefault(i + 1, {}).setdefault("testo", testo.strip())
    return dict_result

def estrai_immagini_da_pdf(*, pdf_path, output_folder="immagini_estratte", dict_result):
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        pagina = doc[page_num]
        immagini_in_pagina = pagina.get_images(full=True)
        for i, img in enumerate(immagini_in_pagina):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_path = f"{output_folder}/pagina{page_num+1}_img{i+1}.{image_ext}"

            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)

            dict_result.setdefault(page_num + 1, {}).setdefault("immagini", []).append(image_path)

    return dict_result

def analizza_pdf(pdf_path):
    dict_result = estrai_testo_da_pdf(pdf_path)
    dict_result = estrai_immagini_da_pdf(pdf_path=pdf_path, dict_result=dict_result)
    return dict_result

# Esempio di utilizzo
risultato = analizza_pdf("File/The_archaeology_of_the_medieval_Castle_o.pdf")

with open("risultato.json", "w") as outfile:
    json.dump(risultato, outfile)

print(risultato)
