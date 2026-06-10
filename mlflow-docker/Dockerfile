FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev gcc curl \
    && rm -rf /var/lib/apt/lists/*
# Keep this mlflow version EQUAL to the client version in the recsys env (see requirements.txt)
RUN pip install --no-cache-dir mlflow==2.16.2 psycopg2-binary==2.9.9
EXPOSE 5000
HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
    CMD curl -f http://localhost:5000/health || exit 1
