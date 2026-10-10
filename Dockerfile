FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRIAGE_DATA_DIR=/data \
    SENTENCE_TRANSFORMERS_HOME=/models
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.14.1+cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml ./
COPY app.py ./
COPY wireless_log_triage ./wireless_log_triage
COPY data/models.joblib /data/models.joblib
RUN pip install --no-cache-dir --no-deps . \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
