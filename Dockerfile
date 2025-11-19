
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128" \
    TOKENIZERS_PARALLELISM="false" \
    CUDA_VISIBLE_DEVICES="" \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
    FORCE_CPU=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers \
    HF_DATASETS_CACHE=/app/.cache/huggingface/datasets


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean


WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/.cache/huggingface && \
    python -c "from pysentimiento import create_analyzer; analyzer = create_analyzer(task='sentiment', lang='es'); print('✅ Modelo pre-descargado exitosamente en /app/.cache/huggingface')"


COPY . .

EXPOSE 8000

CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --loop asyncio"

