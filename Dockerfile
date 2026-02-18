FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    runpod \
    "transformers>=4.45.0" \
    "torchaudio>=2.4.0" \
    accelerate \
    soundfile \
    librosa \
    huggingface_hub

# Copy handler
COPY handler.py /app/handler.py

# Model will be downloaded on first cold start and cached to network volume
# Set HF_HOME to network volume mount point for persistence
ENV HF_HOME=/runpod-volume/huggingface
ENV MODEL_NAME=OpenMOSS-Team/MOSS-TTS

CMD ["python3", "-u", "/app/handler.py"]
