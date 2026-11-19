# TaxLens API image. Build from the project folder:
#   docker compose build api
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY taxlens ./taxlens
COPY db ./db
COPY eval/benchmark ./eval/benchmark
COPY pyproject.toml ./

RUN useradd --create-home app && mkdir -p /app/data/raw /app/models && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "taxlens.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
