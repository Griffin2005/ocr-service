#!/usr/bin/env python3
"""
Layer 2 - Deep Analysis (multiprocessing OCR + summarization)

Usage:
  python layer2_deep_processing.py \
    --input-folder input_docs \
    --layer1-json layer1_results.json \
    --output-json layer2_results.json \
    --workers 4 \
    --dpi 150 \
    --summary-sentences 3 \
    --tesseract-path "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

If --layer1-json is provided, only files listed there are processed; otherwise all files in input-folder are processed.
"""

import os
import time
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# OCR and PDF libraries
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

# Summarization
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# language detect for potential enhancements
from langdetect import detect, LangDetectException

# Optional: try to use easyocr if available (faster on images and optional GPU)
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False

# --------------------------
# Helper functions
# --------------------------
def setup_tesseract(tesseract_path):
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

def fast_extract_text_with_pymupdf(pdf_path):
    """Try to extract text using PyMuPDF (very fast for PDFs that have a text layer)."""
    try:
        doc = fitz.open(pdf_path)
        texts = []
        for page in doc:
            texts.append(page.get_text("text") or "")
        doc.close()
        return "\n".join(texts).strip()
    except Exception:
        return ""

def get_pdf_page_count(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        n = doc.page_count
        doc.close()
        return n
    except Exception:
        return 0

def ocr_page_worker(args_tuple):
    """
    Worker function executed in a separate process.
    It will convert the specified page of pdf_path to an image and run OCR on it.
    args_tuple = (pdf_path, page_number, dpi, tesseract_lang, use_easyocr)
    page_number is 0-based for this function.
    Returns extracted text (string).
    """
    pdf_path, page_number, dpi, tesseract_lang, use_easyocr = args_tuple
    try:
        # convert only the required page (pdf2image supports first_page/last_page indexing from 1)
        pil_pages = convert_from_path(pdf_path, dpi=dpi, first_page=page_number+1, last_page=page_number+1)
        if not pil_pages:
            return ""
        img = pil_pages[0]
        # Prefer easyocr if requested and available
        if use_easyocr and EASYOCR_AVAILABLE:
            reader = easyocr.Reader([tesseract_lang.split('+')[0]] if '+' in tesseract_lang else [tesseract_lang], gpu=False)
            # easyocr accepts numpy arrays or path; use numpy
            import numpy as np
            arr = np.asarray(img.convert("RGB"))
            try:
                result = reader.readtext(arr, detail=0)
                return " ".join(result) if result else ""
            except Exception:
                # fallback to pytesseract
                pass
        # fallback to pytesseract
        text = pytesseract.image_to_string(img, lang=tesseract_lang, config="--psm 3 --oem 1")
        return text or ""
    except Exception as e:
        return f"[OCR error page {page_number+1}: {e}]"

def summarize_with_sumy(text, sentence_count=3):
    if not text or not text.strip():
        return "No text available."
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        sentences = summarizer(parser.document, sentence_count)
        return " ".join(str(s).strip() for s in sentences) or (" ".join(text.split()[:50]))
    except Exception:
        return " ".join(text.split()[:50])

def detect_priority_simple(text):
    t = (text or "").lower()
    if any(k in t for k in ["urgent", "immediate", "critical", "accident", "incident", "deadline"]):
        return "HIGH"
    if any(k in t for k in ["request", "invoice", "payment", "submit", "due"]):
        return "MEDIUM"
    return "LOW"

def make_preview(text, max_words=200):
    words = (text or "").split()
    if not words:
        return ""
    preview = " ".join(words[:max_words])
    if len(words) > max_words:
        preview += "..."
    return preview

# --------------------------
# Per-document processing
# --------------------------
def process_document(path_str, args, department="Unknown", category="General", source="Central Email", uploaded_by="System"):
    """
    Process a single document (full deep analysis).
    Returns a dict with all metadata fields requested.
    """
    start = time.time()
    pdf_path = str(path_str)
    filename = os.path.basename(pdf_path)
    ext = os.path.splitext(filename)[1].lower()

    # 1) Try fast text extraction
    page_count = 0
    full_text = ""
    if ext == ".pdf":
        page_count = get_pdf_page_count(pdf_path)
        full_text = fast_extract_text_with_pymupdf(pdf_path)

    # 2) OCR fallback if text too short
    if not full_text or len(full_text.split()) < args.min_words_for_skip:
        full_text = run_ocr_for_file(pdf_path, ext, page_count, args)

    # --- Title: first heading line or fallback to filename ---
    title = full_text.split("\n")[0].strip() if full_text else filename

    # --- Summary & priority ---
    summary = summarize_with_sumy(full_text, sentence_count=args.summary_sentences) if args.use_sumy else make_preview(full_text, max_words=args.summary_preview_words)
    priority = detect_priority_simple(full_text)

    # --- Type: based on extension or keywords ---
    if ext in [".pdf", ".docx", ".doc"]:
        doc_type = "Document"
    elif ext in [".jpg", ".jpeg", ".png"]:
        doc_type = "Image"
    elif ext in [".xls", ".xlsx"]:
        doc_type = "Spreadsheet"
    else:
        doc_type = "Unknown"

    elapsed = round(time.time() - start, 2)

    # --- Final JSON object ---
    result = {
        "Title": title,
        "Type": doc_type,
        "Category": category,
        "Department": department,
        "Priority": priority,
        "Summary": summary,
        "Source": source,
        "Filename": filename,
        "FilePath": pdf_path,
        "Uploaded by": uploaded_by,
        "Pages Processed": page_count if page_count else 1,
        "Processing Time": f"{elapsed} sec"
    }
    return result

# --------------------------
# Main orchestrator
# --------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-folder", "-i", default="input_docs", help="Folder containing files to deep-process")
    p.add_argument("--layer1-json", default="", help="Layer1 JSON file (optional). If provided, only files listed are processed")
    p.add_argument("--output-json", "-o", default="layer2_results.json", help="Output JSON file")
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1), help="Number of parallel processes for OCR")
    p.add_argument("--dpi", type=int, default=150, help="DPI for converting PDF pages to images")
    p.add_argument("--tesseract-path", default="", help="Full path to tesseract executable (optional)")
    p.add_argument("--tesseract-lang", default="eng", help="Tesseract languages, e.g. 'eng' or 'mal+eng'")
    p.add_argument("--use-easyocr", action="store_true", help="Use EasyOCR in workers if available")
    p.add_argument("--summary-sentences", type=int, default=3, help="Number of sentences for Sumy summarizer")
    p.add_argument("--summary-preview-words", type=int, default=200, help="Preview length in words")
    p.add_argument("--min-words-for-skip", type=int, dest="min_words_for_skip", default=300, help="If extracted text >= this, skip OCR")
    p.add_argument("--no-sumy", dest="use_sumy", action="store_false", help="Disable Sumy summarizer and use preview only")
    args = p.parse_args()

    # tesseract path
    if args.tesseract_path:
        setup_tesseract(args.tesseract_path)

    args.use_easyocr = args.use_easyocr and EASYOCR_AVAILABLE
    if args.use_easyocr and not EASYOCR_AVAILABLE:
        print("Note: easyocr not installed; falling back to pytesseract.")

    # Build list of files to process
    files_to_process = []
    if args.layer1_json and os.path.exists(args.layer1_json):
        try:
            with open(args.layer1_json, "r", encoding="utf-8") as fh:
                layer1 = json.load(fh)
                # layer1 expected to be list of dicts with "file" and optionally folder path
                for item in layer1:
                    fn = item.get("file") or item.get("filename")
                    if not fn:
                        continue
                    # look for file under input-folder
                    candidate = Path(args.input_folder) / fn
                    if candidate.exists():
                        files_to_process.append(str(candidate))
                    else:
                        # try searching input folder recursively
                        found = list(Path(args.input_folder).rglob(fn))
                        if found:
                            files_to_process.append(str(found[0]))
        except Exception as e:
            print("Error reading layer1 json:", e)

    # If no layer1 list, process all files in input folder
    if not files_to_process:
        for ext in ["*.pdf", "*.png", "*.jpg", "*.jpeg", "*.docx", "*.txt", "*.xlsx", "*.xls", "*.tiff", "*.bmp"]:
            files_to_process.extend([str(p) for p in Path(args.input_folder).glob(ext)])
    files_to_process = sorted(set(files_to_process))

    if not files_to_process:
        print("No files found to process. Put files into", args.input_folder)
        return

    results = []
    print(f"Processing {len(files_to_process)} files with {args.workers} workers (per-page OCR parallelism).")

    # Process files sequentially, but internally document OCR pages run in parallel (per-document).
    # This gives good balance of resource usage and progress reporting.
    for file_path in tqdm(files_to_process, desc="Documents", unit="doc"):
        try:
            res = process_document(file_path, args)
            results.append(res)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            results.append({"file": os.path.basename(file_path), "error": str(e)})

    # Write output JSON
    with open(args.output_json, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    print("Layer 2 finished. Results ->", args.output_json)

if __name__ == "__main__":
    main()
