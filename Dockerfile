# Environment for every experiment in the paper (CPU only).
#   docker build -t beta .
#   docker run --rm -v "$PWD":/beta beta python -m pytest tests -q
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 git \
 && rm -rf /var/lib/apt/lists/* \
 && git config --global --add safe.directory /beta

WORKDIR /beta
COPY pyproject.toml README.md ./
COPY beta ./beta
RUN pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir "autogluon.tabular[all]==1.5.0" "numpy>=1.24,<2.0" openml pytest \
 && pip install --no-cache-dir -e .
COPY . .
