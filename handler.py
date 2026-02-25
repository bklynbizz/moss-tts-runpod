"""
MOSS TTS RunPod Serverless Handler
Handles text-to-speech and voice cloning requests.
"""

import runpod
import torch
import torchaudio
import base64
import os
import shutil
import tempfile
import traceback

# --- Monkey-patch pad_sequence for PyTorch < 2.5 ---
# MOSS TTS processor calls pad_sequence(padding_side=...) which was added in PyTorch 2.5.
# If we're on an older version, wrap the original to handle the kwarg.
_orig_pad_sequence = torch.nn.utils.rnn.pad_sequence

def _patched_pad_sequence(sequences, batch_first=False, padding_value=0.0, padding_side="right"):
    if padding_side == "left":
        # Reverse each sequence, pad (right), then reverse back
        reversed_seqs = [seq.flip(0) for seq in sequences]
        padded = _orig_pad_sequence(reversed_seqs, batch_first=batch_first, padding_value=padding_value)
        return padded.flip(1 if batch_first else 0)
    return _orig_pad_sequence(sequences, batch_first=batch_first, padding_value=padding_value)

if not hasattr(torch.nn.utils.rnn.pad_sequence, '_has_padding_side'):
    try:
        # Test if native pad_sequence already supports padding_side
        _test = torch.nn.utils.rnn.pad_sequence([torch.zeros(1)], padding_side="right")
    except TypeError:
        torch.nn.utils.rnn.pad_sequence = _patched_pad_sequence
        print(f"Patched pad_sequence for PyTorch {torch.__version__} (< 2.5 compat)")

# --- Storage setup ---
# RunPod network volume mounts at /runpod-volume
# We MUST point all caches AND temp dirs there to avoid filling root disk
VOLUME_PATH = "/runpod-volume"
HF_CACHE = os.path.join(VOLUME_PATH, "huggingface")
TEMP_DIR = os.path.join(VOLUME_PATH, "tmp")
VOICES_DIR = os.path.join(VOLUME_PATH, "voices")

# Set env vars BEFORE any imports that use them
os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE, "hub")
os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE, "hub")
os.environ["TMPDIR"] = TEMP_DIR
os.environ["TEMP"] = TEMP_DIR
os.environ["TMP"] = TEMP_DIR

# Create dirs
for d in [HF_CACHE, TEMP_DIR, os.path.join(HF_CACHE, "hub"), VOICES_DIR]:
    os.makedirs(d, exist_ok=True)

# Override tempfile default
tempfile.tempdir = TEMP_DIR

# Global model references (loaded once on cold start)
model = None
processor = None
device = None
dtype = None


def get_disk_info():
    """Return disk usage for debugging."""
    info = {}
    for path in ["/", VOLUME_PATH, "/tmp"]:
        try:
            usage = shutil.disk_usage(path)
            info[path] = {
                "total_gb": round(usage.total / (1024**3), 2),
                "used_gb": round(usage.used / (1024**3), 2),
                "free_gb": round(usage.free / (1024**3), 2),
            }
        except Exception as e:
            info[path] = {"error": str(e)}
    info["hf_home"] = os.environ.get("HF_HOME", "not set")
    info["tmpdir"] = os.environ.get("TMPDIR", "not set")
    info["volume_exists"] = os.path.isdir(VOLUME_PATH)
    info["volume_writable"] = os.access(VOLUME_PATH, os.W_OK)
    info["torch_version"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    return info


def load_model():
    """Load MOSS TTS model (runs once on cold start)."""
    global model, processor, device, dtype

    if model is not None:
        return  # Already loaded

    print("Loading MOSS TTS model...")
    print(f"Disk info: {get_disk_info()}")

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
    print(f"Disk info after load: {get_disk_info()}")


def save_voice(voice_id, audio_b64):
    """Save a voice reference to the network volume for reuse."""
    audio_bytes = base64.b64decode(audio_b64)
    voice_path = os.path.join(VOICES_DIR, f"{voice_id}.wav")
    with open(voice_path, "wb") as f:
        f.write(audio_bytes)
    return voice_path


def get_voice_path(voice_id):
    """Get path to a saved voice reference, or None if not found."""
    voice_path = os.path.join(VOICES_DIR, f"{voice_id}.wav")
    return voice_path if os.path.exists(voice_path) else None


def list_voices():
    """List all saved voice IDs."""
    voices = []
    for f in os.listdir(VOICES_DIR):
        if f.endswith(".wav"):
            voices.append(f[:-4])  # strip .wav
    return voices


def delete_voice(voice_id):
    """Delete a saved voice reference."""
    voice_path = os.path.join(VOICES_DIR, f"{voice_id}.wav")
    if os.path.exists(voice_path):
        os.remove(voice_path)
        return True
    return False


def generate_speech(text, voice_ref_audio=None, voice_id=None, language="en", temperature=1.7, top_p=0.8, top_k=25, max_tokens=4096):
    """
    Generate speech from text, optionally cloning a reference voice.

    Args:
        text: Text to synthesize
        voice_ref_audio: Base64-encoded audio for voice cloning (optional)
        voice_id: Name of a previously saved voice to use (optional)
        language: Language code (en, zh, es, fr, de, ja, ko, etc.)
        temperature: Audio sampling temperature (default 1.7 for Delay-8B)
        top_p: Top-p sampling (default 0.8)
        top_k: Top-k sampling (default 25)
        max_tokens: Max generation tokens (default 4096)

    Returns:
        dict with base64 audio and metadata
    """
    load_model()

    ref_path = None
    cleanup_ref = False

    # Resolve voice reference
    if voice_ref_audio:
        # New audio provided — decode to temp file
        ref_audio_bytes = base64.b64decode(voice_ref_audio)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(ref_audio_bytes)
            ref_path = f.name
            cleanup_ref = True
    elif voice_id:
        # Use a previously saved voice
        ref_path = get_voice_path(voice_id)
        if not ref_path:
            return {"error": f"Voice '{voice_id}' not found. Available: {list_voices()}"}

    # Build the conversation/prompt
    try:
        if ref_path:
            conversations = [
                [processor.build_user_message(
                    text=text,
                    reference=[ref_path]
                )]
            ]
        else:
            conversations = [
                [processor.build_user_message(text=text)]
            ]
    finally:
        if cleanup_ref and ref_path:
            os.unlink(ref_path)

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
        "voice_cloned": ref_path is not None,
        "voice_id": voice_id if voice_id else None,
    }


def handler(job):
    """RunPod serverless handler."""
    try:
        job_input = job.get("input", {})

        # Diagnostic mode
        if job_input.get("disk_info"):
            return get_disk_info()

        # Voice management actions
        action = job_input.get("action")
        if action == "save_voice":
            vid = job_input.get("voice_id")
            audio = job_input.get("voice_ref_audio")
            if not vid or not audio:
                return {"error": "save_voice requires voice_id and voice_ref_audio"}
            path = save_voice(vid, audio)
            return {"saved": vid, "path": path, "all_voices": list_voices()}

        if action == "list_voices":
            return {"voices": list_voices()}

        if action == "delete_voice":
            vid = job_input.get("voice_id")
            if not vid:
                return {"error": "delete_voice requires voice_id"}
            deleted = delete_voice(vid)
            return {"deleted": vid, "success": deleted, "all_voices": list_voices()}

        # TTS generation
        text = job_input.get("text")
        if not text:
            return {"error": "Missing required parameter: text"}

        voice_ref_audio = job_input.get("voice_ref_audio")  # base64 audio
        voice_id = job_input.get("voice_id")  # saved voice name
        save_as = job_input.get("save_voice_as")  # save this ref for future use
        language = job_input.get("language", "en")
        temperature = job_input.get("temperature", 1.7)
        top_p = job_input.get("top_p", 0.8)
        top_k = job_input.get("top_k", 25)
        max_tokens = job_input.get("max_tokens", 4096)

        # If new audio provided and save_voice_as set, save it for future use
        if voice_ref_audio and save_as:
            save_voice(save_as, voice_ref_audio)

        result = generate_speech(
            text=text,
            voice_ref_audio=voice_ref_audio,
            voice_id=voice_id,
            language=language,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
        )

        if voice_ref_audio and save_as:
            result["voice_saved_as"] = save_as

        return result

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}


# Pre-load model on container start for faster first request
print("Starting MOSS TTS handler...")
print(f"Initial disk info: {get_disk_info()}")
try:
    load_model()
    print("Model pre-loaded successfully!")
except Exception as e:
    print(f"Warning: Model pre-load failed ({e}), will retry on first request")
    model = None
    processor = None

runpod.serverless.start({"handler": handler})
