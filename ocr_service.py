from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import os, time
from layer2_deep_processing import process_document  # Your existing OCR & analysis function

# API Key for security
API_KEY = os.getenv("OCR_API_KEY", "demo_key")  # default for local testing
app = FastAPI()

# CORS if frontend calls OCR service directly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend domain for security
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key check
def require_api_key(x_api_key: str = Header(...)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API Key")

# Main endpoint
@app.post("/process-doc")
async def process_doc(
    file: UploadFile = File(...),
    x_api_key: str = Header(...),
    recipient: str = Form(...),   # who the doc is for
    sender: str = Form(...),      # who sent it
    source: str = Form(...),      # source: Gmail, WhatsApp, SharePoint, etc.
):
    require_api_key(x_api_key)
    start_time = time.time()

    # Save file temporarily
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        # Run your OCR + classification + summarization pipeline
        result = process_document(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    # Add new metadata fields
    total_time = round(time.time() - start_time, 2)
    result.update({
        "recipient": recipient,
        "sender": sender,
        "source": source,
        "filename": file.filename,
        "filepath": file_path,
        "total_time_sec": total_time
    })

    return result

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}
