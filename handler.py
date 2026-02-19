"""
MOSS TTS RunPod Serverless Handler
Handles text-to-speech and voice cloning requests.
"""

import runpod
import torch
import torchaudio
import base64
import os
import tempfile
import traceback

# Global model references (loaded once on cold start)
model = None
processor = None
device = None
dtype = None


def load_model():
    """Load MOSS TTS model (runs once on cold start)."""
    global model, processor, device, dtype

    if model is not None:
        return  # Already loaded

    print("Loading MOSS TTS model...")
    from transformers import AutoModel, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Required: disable broken cuDNN SDPA backend
    if device == "cuda":
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

    pretrained = os.environ.get("MODEL_NAME", "OpenMOSS-Team/MOSS-TTS")

    print(f"Loading processor from {pretrained}...")
    processor = AutoProcessor.from_pretrained(
        pretrained,
        trust_remote_code=True,
    )
    processor.audio_tokenizer = processor.audio_tokenizer.to(device)

    print(f"Loading model from {pretrained}...")
    model = AutoModel.from_pretrained(
        pretrained,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    print(f"MOSS TTS loaded on {device} ({dtype})")


def generate_speech(text, voice_ref_audio=None, language="en", temperature=1.7, top_p=0.8, top_k=25, max_tokens=4096):
    """
    Generate speech from text, optionally cloning a reference voice.

    Args:
        text: Text to synthesize
        voice_ref_audio: Base64-encoded audio for voice cloning (optional)
        language: Language code (en, zh, es, fr, de, ja, ko, etc.)
        temperature: Audio sampling temperature (default 1.7 for Delay-8B)
        top_p: Top-p sampling (default 0.8)
        top_k: Top-k sampling (default 25)
        max_tokens: Max generation tokens (default 4096)

    Returns:
        dict with base64 audio and metadata
    """
    load_model()

    # Build the conversation/prompt
    if voice_ref_audio:
        # Voice cloning mode: decode reference audio to temp file
        ref_audio_bytes = base64.b64decode(voice_ref_audio)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(ref_audio_bytes)
            ref_path = f.name

        try:
            # Official API: reference=[path] for voice cloning
            conversations = [
                [processor.build_user_message(
                    text=text,
                    reference=[ref_path]
                )]
            ]
        finally:
            os.unlink(ref_path)
    else:
        # Standard generation mode
        conversations = [
            [processor.build_user_message(text=text)]
        ]

    # Tokenize
    batch = processor(conversations, mode="generation")
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            audio_temperature=temperature,
            audio_top_p=top_p,
            audio_top_k=top_k,
        )

    # Decode audio
    decoded = processor.decode(outputs)
    message = decoded[0]
    audio = message.audio_codes_list[0]
    sample_rate = processor.model_config.sampling_rate

    # Save to temp file and encode as base64
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        torchaudio.save(f.name, audio.unsqueeze(0).cpu(), sample_rate)
        with open(f.name, "rb") as audio_file:
            audio_b64 = base64.b64encode(audio_file.read()).decode("utf-8")
        os.unlink(f.name)

    return {
        "audio_base64": audio_b64,
        "sample_rate": sample_rate,
        "format": "wav",
        "text": text,
        "voice_cloned": voice_ref_audio is not None,
    }


def handler(job):
    """RunPod serverless handler."""
    try:
        job_input = job.get("input", {})

        text = job_input.get("text")
        if not text:
            return {"error": "Missing required parameter: text"}

        voice_ref_audio = job_input.get("voice_ref_audio")  # base64 audio
        language = job_input.get("language", "en")
        temperature = job_input.get("temperature", 1.7)
        top_p = job_input.get("top_p", 0.8)
        top_k = job_input.get("top_k", 25)
        max_tokens = job_input.get("max_tokens", 4096)

        result = generate_speech(
            text=text,
            voice_ref_audio=voice_ref_audio,
            language=language,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
        )

        return result

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}


# Pre-load model on container start for faster first request
print("Starting MOSS TTS handler...")
try:
    load_model()
    print("Model pre-loaded successfully!")
except Exception as e:
    print(f"Warning: Model pre-load failed ({e}), will retry on first request")
    model = None
    processor = None

runpod.serverless.start({"handler": handler})
