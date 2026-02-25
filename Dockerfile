FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade PyTorch + torchvision first (base has 2.4.0 which lacks pad_sequence padding_side
# and its torchvision conflicts with newer transformers). Handler has monkey-patch fallback.
RUN pip install --no-cache-dir \
    "torch>=2.5.0" \
    "torchaudio>=2.5.0" \
    "torchvision>=0.20.0" \
    --index-url https://download.pytorch.org/whl/cu124

# Install remaining Python dependencies
RUN pip install --no-cache-dir \
    runpod \
    "transformers>=4.45.0" \
    accelerate \
    soundfile \
    librosa \
    huggingface_hub

# Copy handler
COPY handler.py /app/handler.py

# ALL caches and temp dirs must point to network volume to avoid filling root disk
ENV HF_HOME=/runpod-volume/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface/hub
ENV HF_HUB_CACHE=/runpod-volume/huggingface/hub
ENV TMPDIR=/runpod-volume/tmp
ENV TEMP=/runpod-volume/tmp
ENV TMP=/runpod-volume/tmp
ENV MODEL_NAME=OpenMOSS-Team/MOSS-TTS

CMD ["python3", "-u", "/app/handler.py"]
