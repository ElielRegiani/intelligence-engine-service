# syntax=docker/dockerfile:1
#
# Imagens: linux/amd64 — wheels PyPI costumam bastar (sem compilador na imagem final).
# linux/arm64 (Apple Silicon / Graviton): o stage builder instala gcc/g++/gfortran só quando
# a arquitetura do stage é arm64, para permitir wheel local quando não houver wheel pré-compilado.
# Build explícito: docker buildx build --platform linux/amd64 -t ... .

FROM python:3.11-slim AS builder
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1

# Compiladores apenas em arm64 (sdist ou build de extensões C/C++/Fortran).
RUN set -eux; \
    apt-get update; \
    arch="$(dpkg --print-architecture)"; \
    if [ "$arch" = "arm64" ]; then \
      apt-get install -y --no-install-recommends \
        gcc g++ gfortran \
        libopenblas-dev; \
    fi; \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip wheel setuptools

COPY requirements-runtime.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements-runtime.txt

FROM python:3.11-slim AS runtime
WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# OpenMP runtime usado por numpy/sklearn em muitas wheels Linux (amd64 e arm64).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

COPY --from=builder /wheels /wheels
COPY requirements-runtime.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements-runtime.txt \
    && rm -rf /wheels

COPY . .

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
