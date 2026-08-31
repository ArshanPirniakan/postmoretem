# PostMortem -- single-file app, CPU-only, zero paid infrastructure required.
#
# Build:  docker build -t postmortem .
# Run:    docker run -p 8000:8000 -v postmortem_data:/app/data -v postmortem_artifacts:/app/artifacts postmortem
# Then:   open http://localhost:8000

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    "uvicorn[standard]==0.30.1" \
    pydantic==2.7.4 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    scikit-learn==1.4.2 \
    torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu \
    python-multipart==0.0.9

COPY postmortem.py schema.sql ./

# Non-root user, matching the app's own security posture.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data /app/artifacts \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "postmortem.py"]
