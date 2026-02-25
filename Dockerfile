FROM runpod/pytorch:2.8.0-py3.12-cuda12.8.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies — MOSS TTS requires transformers 5.0.0+
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 \
    runpod \
    "transformers>=5.0.0" \
    "torchaudio>=2.8.0" \
    accelerate \
    soundfile \
    librosa \
    huggingface_hub

# Copy handler
COPY handler.py /app/handler.py

# ALL caches and temp dirs must point to network volume to avoid filling root disk
# Network volume mounts at /runpod-volume
ENV HF_HOME=/runpod-volume/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface/hub
ENV HF_HUB_CACHE=/runpod-volume/huggingface/hub
ENV TMPDIR=/runpod-volume/tmp
ENV TEMP=/runpod-volume/tmp
ENV TMP=/runpod-volume/tmp
ENV MODEL_NAME=OpenMOSS-Team/MOSS-TTS

CMD ["python3", "-u", "/app/handler.py"]
