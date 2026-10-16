# DIET classifier NLU server (CPU). Serves every language found in models/<lang>/
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN pip install --no-cache-dir uv

# Dependencies first (cached layer). The locked torch from PyPI pulls ~10 GB of CUDA libraries on
# x86; the server runs on the CPU, so install the CPU-only build of the same version instead.
COPY pyproject.toml uv.lock README.md ./
RUN uv venv /app/.venv \
 && uv export --frozen --no-dev --no-hashes --no-emit-project \
      | grep -Ev '^(torch|nvidia-|triton)' > /tmp/requirements.txt \
 && uv pip install --python /app/.venv/bin/python -r /tmp/requirements.txt \
 && uv pip install --python /app/.venv/bin/python "torch==2.5.1" \
      --extra-index-url https://download.pytorch.org/whl/cpu

COPY src ./src
ENV PYTHONPATH=/app/src

COPY scripts ./scripts
COPY data ./data

# Trained models are not part of the image: mount them at /app/models
# (python scripts/train.py --lang all  ->  models/<lang>/)
VOLUME ["/app/models"]

EXPOSE 8000

# Runs on CPU by default. Examples: --langs es en   |   --default-lang en
CMD ["/app/.venv/bin/python", "scripts/inference_fastapi.py", "-p", "8000"]
