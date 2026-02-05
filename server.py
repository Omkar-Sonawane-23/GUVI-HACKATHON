import base64
import io
import numpy as np
import librosa
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

app = FastAPI(title="Audio AI Detection Server")

# ✅ API KEY (defined correctly at module level)
API_KEY = "jhvbbchjbvchbhjvbdsbhjbvhjbv"


# ---------- Request Schema ----------
class AudioRequest(BaseModel):
    audio_base64: str


# ---------- Response Schema ----------
class AudioResponse(BaseModel):
    confidence_score: float
    isAI: bool
    explanation: str


# ---------- Utility: Decode Base64 MP3 ----------
def decode_mp3_base64(audio_base64: str):
    try:
        audio_bytes = base64.b64decode(audio_base64)
        audio_buffer = io.BytesIO(audio_bytes)

        waveform, sr = librosa.load(audio_buffer, sr=None)
        return waveform, sr

    except Exception as e:
        raise ValueError(f"Invalid audio data: {e}")


# ---------- Dummy AI Detection Logic ----------
def detect_ai_voice(waveform: np.ndarray, sr: int):
    spectral_centroid = np.mean(
        librosa.feature.spectral_centroid(y=waveform, sr=sr)
    )
    zero_crossing_rate = np.mean(
        librosa.feature.zero_crossing_rate(waveform)
    )

    confidence_score = min(
        1.0,
        (spectral_centroid / 5000) + (zero_crossing_rate * 2)
    )

    is_ai = confidence_score > 0.6

    explanation = (
        "Audio shows spectral and temporal patterns commonly found in synthetic speech generation models."
        if is_ai
        else
        "Audio patterns are consistent with natural human speech."
    )

    return confidence_score, is_ai, explanation


# ---------- API Endpoint ----------
@app.post("/analyze-audio", response_model=AudioResponse)
def analyze_audio(
    request: AudioRequest,
    x_api_key: str = Header(...)
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        waveform, sr = decode_mp3_base64(request.audio_base64)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    confidence_score, is_ai, explanation = detect_ai_voice(waveform, sr)

    return AudioResponse(
        confidence_score=round(confidence_score, 3),
        isAI=is_ai,
        explanation=explanation
    )


# ---------- Health Check ----------
@app.get("/health")
def health():
    return {"status": "ok"}
