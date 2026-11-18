# HoneyAI data-collection + baseline image (CPU only).
# GPU training of the QLoRA model is NOT done here: use Colab or your CUDA machine.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# baselines/audits need numpy + scikit-learn (+ xgboost for Model 1);
# collection scripts use only the standard library.
RUN pip install --no-cache-dir "numpy>=1.26" "scikit-learn>=1.4" "xgboost>=2.0"

# the repo is bind-mounted at /app (see docker-compose.yml), so files you
# collect stay on your machine even if the container is deleted
WORKDIR /app

CMD ["python", "scripts/run_agent_matrix.py", "--help"]
