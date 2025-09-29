FROM python:3.11.8-slim-bullseye

# Upgrade system packages and install tesseract with Thai & English language packs
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-tha \
    libtesseract-dev build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Render calls this by default if you set it as the start command, or via render.yaml
CMD ["python", "main.py"]
