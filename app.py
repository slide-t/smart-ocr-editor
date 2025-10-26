#!/usr/bin/env python3
"""
Smart OCR Editor (local)
- TrOCR (handwritten) for recognition
- Line segmentation
- Spell correction
- Punctuation restoration (model + rule-based fallback)
- Export to DOCX
Run:
  pip install -r requirements.txt
  python app.py
Open http://127.0.0.1:5000
"""

import os
import io
from flask import Flask, render_template, request, jsonify, send_file
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from spellchecker import SpellChecker
from PIL import Image
from docx import Document
import numpy as np
import cv2
import torch

app = Flask(__name__)

# -------------------------
# Config / Model selection
# -------------------------
TROCR_MODEL = os.environ.get("TROCR_MODEL", "microsoft/trocr-base-handwritten")
PUNCT_MODEL = os.environ.get("PUNCT_MODEL", "oliverguhr/fullstop-punctuation-multilang-large")
USE_PUNCT_MODEL = os.environ.get("USE_PUNCT_MODEL", "1") == "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load TrOCR
# -------------------------
print("Loading TrOCR model (this may take a while)...")
processor = TrOCRProcessor.from_pretrained(TROCR_MODEL)
trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL)
trocr_model.to(device)
print("TrOCR loaded.")

# -------------------------
# Spell checker
# -------------------------
spell = SpellChecker()

# -------------------------
# Load punctuation restoration model (optional)
# -------------------------
punct_pipeline = None
if USE_PUNCT_MODEL:
    try:
        print(f"Loading punctuation model: {PUNCT_MODEL} ...")
        # Use a text2text-generation pipeline if model supports seq2seq, else fallback
        punct_tokenizer = AutoTokenizer.from_pretrained(PUNCT_MODEL)
        punct_model = AutoModelForSeq2SeqLM.from_pretrained(PUNCT_MODEL).to(device)
        punct_pipeline = pipeline("text2text-generation", model=punct_model, tokenizer=punct_tokenizer, device=0 if torch.cuda.is_available() else -1)
        print("Punctuation model loaded.")
    except Exception as e:
        print("Punctuation model load failed — will fallback to rule-based punctuation. Error:", e)
        punct_pipeline = None

# -------------------------
# Helpers
# -------------------------
def preprocess_image_bytes(data: bytes):
    """Return binarized image (numpy uint8) for segmentation + PIL RGB image for whole-page OCR if needed."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Denoise and threshold
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th, Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def segment_lines(binary_img: np.ndarray):
    """
    Segment binary image into line crops (returns list of PIL.Image objects).
    Simple projection profile method.
    """
    inv = cv2.bitwise_not(binary_img)  # text is white on black
    proj = np.sum(inv, axis=1)
    lines = []
    start = None
    threshold = max(1, np.max(proj) * 0.05)
    for i, val in enumerate(proj):
        if val > threshold and start is None:
            start = i
        elif val <= threshold and start is not None:
            end = i
            if end - start > 8:
                lines.append((start, end))
            start = None
    if start is not None:
        lines.append((start, len(proj)))

    crops = []
    h, w = binary_img.shape
    for (y1, y2) in lines:
        crop = binary_img[y1:y2, :]
        # Convert to RGB PIL for TrOCR
        pil = Image.fromarray(cv2.cvtColor(cv2.merge([crop, crop, crop]), cv2.COLOR_BGR2RGB)) if len(crop.shape) == 2 else Image.fromarray(crop)
        crops.append(pil)
    if not crops:
        # fallback: return whole image
        crops = [Image.fromarray(binary_img)]
    return crops

def trocr_recognize(pil_image: Image.Image):
    """Run TrOCR on a PIL image and return text (uses device)."""
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values.to(device)
    generated_ids = trocr_model.generate(pixel_values, max_length=512)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text

def simple_spell_correct(text: str):
    """Word-level spell correction for obvious errors (fast)."""
    words = text.split()
    corrected = []
    for w in words:
        # preserve punctuation-looking tokens
        if not w.isalpha():
            corrected.append(w)
            continue
        # keep case when it's capitalized
        lower = w.lower()
        if lower in spell:
            corrected.append(w)
            continue
        suggestion = spell.correction(lower)
        if suggestion:
            # restore capitalization if original was capitalized
            if w[0].isupper():
                suggestion = suggestion.capitalize()
            corrected.append(suggestion)
        else:
            corrected.append(w)
    return " ".join(corrected)

def restore_punctuation(text: str):
    """Restore punctuation using model if available, else rule-based fallback."""
    if not text.strip():
        return text
    # If model pipeline available, use it
    if punct_pipeline:
        # The punct pipeline expects reasonably sized inputs. We'll chunk by ~200 tokens if long.
        # Simple chunk by characters: 1200 chars per chunk to be safe.
        chunks = []
        s = text.strip()
        max_chars = 1200
        i = 0
        while i < len(s):
            chunk = s[i:i+max_chars]
            # try not to cut in middle of word
            if i + max_chars < len(s):
                # extend to next space
                next_space = s.find(" ", i+max_chars)
                if next_space != -1 and next_space < i+max_chars+50:
                    chunk = s[i:next_space]
                    i = next_space
                else:
                    i += max_chars
            else:
                i += max_chars
            chunks.append(chunk)
        out_chunks = []
        for c in chunks:
            try:
                res = punct_pipeline(c, max_length= len(c.split()) + 200)
                if isinstance(res, list):
                    out_chunks.append(res[0]["generated_text"])
                elif isinstance(res, dict) and "generated_text" in res:
                    out_chunks.append(res["generated_text"])
                else:
                    out_chunks.append(c)
            except Exception as e:
                # fallback to raw chunk
                out_chunks.append(c)
        out = " ".join(out_chunks)
        return out
    # Rule-based fallback (lightweight):
    # - Split on double newlines -> Paragraphs
    # - For each paragraph, split words and insert periods every N words if no punctuation
    paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    restored = []
    for p in paragraphs:
        # If punctuation already present, keep simple normalization
        if any(ch in p for ch in ".?!,;:"):
            # basic tidy: ensure single spaces
            restored.append(" ".join(p.split()))
        else:
            # insert periods near sentence boundaries using simple heuristics:
            words = p.split()
            # insert period every 12-18 words as a naive sentence boundary
            out = []
            i = 0
            while i < len(words):
                chunk = words[i:i+14]  # chunk size
                s_chunk = " ".join(chunk).capitalize()
                if i + 14 < len(words):
                    out.append(s_chunk + ".")
                else:
                    out.append(s_chunk)
                i += 14
            restored.append(" ".join(out))
    return "\n\n".join(restored)

def normalize_whitespace(text: str):
    # Replace multiple spaces/newlines with normalized spacing
    # Keep intentional paragraph breaks (two newlines)
    lines = [ln.strip() for ln in text.splitlines()]
    joined = "\n".join([ln for ln in lines if ln != ""])
    # collapse multiple spaces
    return " ".join(joined.split())

# -------------------------
# Flask endpoints
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ocr", methods=["POST"])
def ocr():
    f = request.files.get("image")
    if not f:
        return jsonify({"error": "image file required"}), 400
    try:
        data = f.read()
        binary_img, pil_rgb = preprocess_image_bytes(data)
        # Segment lines, run TrOCR per line (better accuracy)
        line_images = segment_lines(binary_img)
        texts = []
        for pil in line_images:
            txt = trocr_recognize(pil.convert("RGB"))
            texts.append(txt.strip())
        raw_text = "\n".join([t for t in texts if t])
        # Basic cleanup
        raw_text = normalize_whitespace(raw_text)
        # Spell correction
        spelled = simple_spell_correct(raw_text)
        # Punctuation restoration
        punctuated = restore_punctuation(spelled)
        # Final whitespace normalization
        final = normalize_whitespace(punctuated)
        return jsonify({"raw": raw_text, "spelled": spelled, "final": final})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/download", methods=["POST"])
def download():
    content = request.json.get("text", "")
    doc = Document()
    for para in content.split("\n"):
        doc.add_paragraph(para)
    path = "ocr_output.docx"
    doc.save(path)
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    print("Starting app on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
