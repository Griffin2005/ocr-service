# Use Python 3.11 slim base image for small size
FROM python:3.11-slim

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies: poppler for pdf2image, tesseract for OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    tesseract-ocr-eng \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set work directory inside the container
WORKDIR /app

# Copy only requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files into container
COPY . .

# Create uploads folder (optional, for temporary files)
RUN mkdir -p /app/uploads

# Expose port 8000 for FastAPI
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "ocr_service:app", "--host", "0.0.0.0", "--port", "8000"]
