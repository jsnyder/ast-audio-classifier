# AST Audio Classifier Dockerfile
# Debian slim — PyTorch CPU wheels need glibc (not Alpine musl)
# Works as both standalone Docker and HA addon (via build.yaml BUILD_FROM)

# PyTorch CPU wheels need glibc — cannot use Alpine-based HA base images
FROM python:3.11-slim-bookworm

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# System dependencies (jq needed for HA addon options parsing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for fast package management
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml ./
COPY src ./src

# Install Python dependencies (CPU-only PyTorch + torchaudio from CPU index)
RUN uv pip install --system --no-cache \
    torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && uv pip install --system --no-cache -e .

# Pre-download AST model at build time (~350MB)
# This avoids runtime downloads and makes the image air-gapped compatible
RUN python -c "from transformers import pipeline; pipeline('audio-classification', model='MIT/ast-finetuned-audioset-10-10-0.4593', device=-1)"

# Pre-download CLAP model at build time (~150MB)
# Used for zero-shot verification of AST classifications
RUN python -c "from transformers import pipeline; pipeline('zero-shot-audio-classification', model='laion/clap-htsat-fused', device=-1)"

# Copy startup script
COPY run.sh /
RUN chmod a+x /run.sh

# Note: HA addons run in sandboxed containers, non-root conflicts with supervisor
# For standalone use, uncomment: RUN useradd -m -r appuser && USER appuser

# Health check — long start period for model loading (AST + CLAP)
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["/run.sh"]

LABEL \
    org.opencontainers.image.title="AST Audio Classifier" \
    org.opencontainers.image.description="Audio Spectrogram Transformer sidecar for Home Assistant" \
    org.opencontainers.image.vendor="James Snyder" \
    org.opencontainers.image.authors="James Snyder" \
    org.opencontainers.image.licenses="MIT" \
    org.opencontainers.image.source="https://github.com/jsnyder/home_assistant"
