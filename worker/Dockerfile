# Build from the repo root: docker build -f worker/Dockerfile .
# (docker-compose.yml does this for you)

# Resolve the commit hash in a throwaway stage so .git never lands in the
# final image. Tolerates a build context without .git (writes an empty file).
FROM alpine/git AS gitsha
COPY . /src
RUN git -C /src rev-parse HEAD > /GIT_SHA 2>/dev/null || : > /GIT_SHA

FROM python:3.12-slim

# g++ so PyTensor compiles its C backend instead of falling back to the slow
# Python-only mode with a warning on every import (worker/README.md
# "Deployment notes: Pin dependencies hard; PyMC on a fresh container will
# bite otherwise").
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY worker/requirements.txt worker/requirements.txt
RUN pip install --no-cache-dir -r worker/requirements.txt

COPY model model
COPY ingest ingest
COPY scripts scripts
COPY worker worker
COPY --from=gitsha /GIT_SHA GIT_SHA

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# docker-compose overrides this command for the worker-loop service.
CMD ["uvicorn", "worker.app:app", "--host", "0.0.0.0", "--port", "8000"]
