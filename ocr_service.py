from fastapi import FastAPI, UploadFile, File
import uvicorn
import os
from pathlib import Path
from layer2_deep_processing import process_document  # We'll reuse your Layer 2 logic

app = FastAPI()

# Folder where uploaded files will be stored temporarily
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def home():
    return {"message": "OCR Service is running!"}
class Config:
    min_words_for_skip = 300
    summary_sentences = 3
    summary_preview_words = 200
    use_sumy = True
    dpi = 150
    tesseract_lang = "eng"
    use_easyocr = False
    workers = 2

args = Config()

@app.post("/process-doc")
async def process_doc(file: UploadFile = File(...)):
    # 1. Save uploaded file
    file_path = Path(UPLOAD_DIR) / file.filename
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # 2. Run your existing processing logic
    result = process_document(
        str(file_path),
        args=args,
        department="Auto",
        category="General",
        source="Upload",
        uploaded_by="Frontend"
    )

    # 3. Return result as JSON
    return result

if __name__ == "__main__":
    uvicorn.run("ocr_service:app", host="0.0.0.0", port=8000, reload=True)
