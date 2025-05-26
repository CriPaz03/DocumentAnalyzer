import os
import ollama
import base64
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import json

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract.exe'

prompt = """
Agisci come un esperto di storia medievale italiana e analista di documenti accademici. Ti fornirò:
- Un'immagine contenente una o più pagine di un documento storico medievale.
- Il testo estratto tramite OCR.

Il tuo compito è:
1. Leggere e comprendere il contenuto testuale.
2. Generare un riassunto coerente dei contenuti storici del documento.
3. Identificare in quali pagine ci sono immagini.

Il risultato che mi restituisci deve essere formattato come JSON nel seguente modo:

{
  "riassunto": "Testo riassuntivo dei contenuti storici...",
  "immagini": [
    { "pagina": 1 },
    { "pagina": 4 },
    ...
  ],
}

"""

PATH_FILE = "File/"
PATH_OUTPUT = "Output/"

class DocumentAnalyzer:
    choice_function = {
        "1": "analyze_multi_page",
        "2": "analyze_single_page"
    }

    def __init__(self, pdf_path, max_height=20000, model_summary="qwen2.5vl"):
        self.pdf_path = pdf_path
        self.max_height = max_height
        self.model_summary = model_summary
        self.images = self.convert_pdf_to_images()

    def convert_result_to_json(self, result):
        cleaned_result = result.replace("```json", "").replace("```", "").strip()

        try:
            json_result = json.loads(cleaned_result)
        except json.JSONDecodeError as e:
            print(f"Errore nel parsing JSON: {e}")
            return None

        return json_result

    def convert_pdf_to_images(self):
        images = convert_from_path(self.pdf_path, dpi=300)
        return images

    def start_analyzer(self, choice):
        method_name = self.choice_function.get(choice)
        if not method_name:
            raise ValueError(f"Scelta non valida: {choice}")
        method = getattr(self, method_name)
        return method()

    def clear_output_directory(self):
        for file in os.listdir(PATH_OUTPUT):
            os.remove(f"{PATH_OUTPUT}{file}")

    def analyze_multi_page(self):
        self.clear_output_directory()
        output_path = f"{PATH_OUTPUT}merged_image"
        widths, heights = zip(*(img.size for img in self.images))
        max_width = max(widths)
        merged_image = Image.new('RGB', (max_width, self.max_height))
        y_offset = 0
        count = 1
        for img in self.images:
            img_height = img.size[1]
            if img_height + y_offset > self.max_height:
                y_offset = 0
                merged_image.save(f"{output_path}_{count}.jpg", 'JPEG', quality=95)
                count += 1
                merged_image = Image.new('RGB', (max_width, self.max_height))
            merged_image.paste(img, (0, y_offset))
            y_offset += img.height
        merged_image.save(f"{output_path}_{count}.jpg", 'JPEG', quality=95)

        return self.send_request_ollama()

    def analyze_single_page(self):
        self.clear_output_directory()

        for i, img in enumerate(self.images):
            temp_path = f'{PATH_OUTPUT}temp_page_{i + 1}.jpg'
            img.save(temp_path, 'JPEG')

        return self.send_request_ollama()

    def extract_text_from_image(self, image_path):
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang='eng+ita')
        return text

    def generate_sql_query_with_codellama(self, summary_text):
        prompt_sql = f"""
        Sei uno storico medievista specializzato in documentazione. Hai appena ricevuto questo riassunto storico:

        \"\"\"{summary_text}\"\"\"

        Costruisci una query SQL per SQLite3 che:
        - Crea una tabella `caratteristiche` (se non esiste) con i seguenti campi:
          - `id`: INTEGER PRIMARY KEY AUTOINCREMENT
          - `nome`: TEXT – nome della caratteristica
          - `descrizione`: TEXT – spiegazione del suo significato o uso
          - `pagina`: INTEGER – numero della pagina del documento in cui è stata individuata

        Genera la query di creazione e degli INSERT con le caratteristiche più importanti e significanti.
        Rispondi **solo** con il codice SQL, senza testo aggiuntivo.
        """

        messages = [{
            'role': 'user',
            'content': prompt_sql
        }]

        response = ollama.chat(model="codellama:7b", messages=messages)
        return response['message']['content']

    def generate_summary_and_images(self, file_path):
        if file_path != "":
            with open(file_path, 'rb') as f:
                text = self.extract_text_from_image(file_path)
                image_data = base64.b64encode(f.read()).decode('utf-8')
            prompt_with_text = prompt + "\n\nTesto estratto OCR:\n" + text
            messages = [{
                'role': 'user',
                'content': prompt_with_text,
                'images': [image_data]
            }]
        else:
            messages = [{
                'role': 'user',
                'content': prompt,
            }]
        response = ollama.chat(
            model=self.model_summary,
            messages=messages
        )

        return self.convert_result_to_json(response['message']['content'])

    def send_request_ollama(self):
        final_result = []
        for filename in os.listdir(PATH_OUTPUT):
            file_path = os.path.join(PATH_OUTPUT, filename)
            try:
                response = self.generate_summary_and_images(file_path=file_path)
                if response and "riassunto" in response and "immagini" in response:
                    summary = response["riassunto"]
                    images = response["immagini"]
                    query_sql = self.generate_sql_query_with_codellama(summary)
                    final_result.append({
                        "riassunto": summary,
                        "immagini": images,
                        "query": query_sql
                    })
                else:
                    print(f"Risposta malformata per il file: {file_path}")
            except Exception as e:
                print(f"Errore durante l'elaborazione del file {file_path}: {e}")
        return final_result

def main():
    pdf_path = 'File/The_archaeology_of_the_medieval_Castle_o.pdf'
    print("Scegli il metodo di analisi:")
    print("1. Immagine singola ridimensionata")
    print("2. Analisi pagina per pagina")

    choice = input("Inserisci il numero (1-2): ").strip()

    print(DocumentAnalyzer(pdf_path).start_analyzer(choice))


if __name__ == '__main__':
    main()




