FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-tha libtesseract-dev libleptonica-dev curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# optional: healthcheck that hits your own app using $PORT
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD sh -c 'curl -fsS "http://localhost:${PORT}/healthz?ts=$(date +%s)" || exit 1'

CMD ["python", "main.py"]
