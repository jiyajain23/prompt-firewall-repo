FROM python:3.11-slim

# Non-root user
RUN useradd -m -u 1000 firewall
WORKDIR /app

# System deps for faiss + torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/      src/
COPY configs/  configs/
COPY inference.py .

# Ownership
RUN chown -R firewall:firewall /app
USER firewall

# Model cache dir (HF downloads here at first startup)
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p $HF_HOME

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
