from flask import Flask, render_template, request, jsonify, send_file
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from spellchecker import SpellChecker
from PIL import Image
from io import BytesIO
from docx import Document
import os

app = Flask(__name__)

# Load model and processor once at startup
print("⏳ Loading TrOCR model... This will take a moment on first run.")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
spell = SpellChecker()
print("✅ Model loaded successfully!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ocr', methods=['POST'])
def ocr_image():
    file = request.files['image']
    image = Image.open(file.stream).convert("RGB")

    # OCR
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # Spell correction
    corrected = []
    for word in text.split():
        corrected.append(spell.correction(word) or word)
    corrected_text = " ".join(corrected)

    return jsonify({"raw": text, "corrected": corrected_text})

@app.route('/download', methods=['POST'])
def download_doc():
    content = request.json.get("text", "")
    doc = Document()
    doc.add_paragraph(content)
    path = "output.docx"
    doc.save(path)
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
