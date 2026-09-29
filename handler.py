import base64
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import runpod
import soundfile as sf
import torch
from pyannote.audio import Pipeline

MODEL_ID = os.environ.get("PYANNOTE_MODEL", "pyannote/speaker-diarization-community-1")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("Set HF_TOKEN to a Hugging Face token with access to the pyannote model")

os.environ.setdefault("HF_TOKEN", HF_TOKEN)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline = Pipeline.from_pretrained(MODEL_ID, token=HF_TOKEN)
pipeline.to(device)


def _download(url):
    suffix = Path(url.split("?", 1)[0]).suffix or ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    request = urllib.request.Request(url, headers={"User-Agent": "audio-diarization"})
    with urllib.request.urlopen(request, timeout=120) as response, open(path, "wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    return path


def _write_base64(payload, name="audio.wav"):
    if payload.strip().startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    suffix = Path(name).suffix or ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as handle:
        handle.write(base64.b64decode(payload))
    return path


def _load_waveform(path):
    wav_path = None
    try:
        try:
            data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        except Exception:
            wav_path = path + ".wav"
            subprocess.run(
                ["ffmpeg", "-y", "-i", path, "-ac", "1", wav_path],
                check=True,
                capture_output=True,
            )
            data, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
        return {"waveform": torch.from_numpy(data.T.copy()), "sample_rate": int(sample_rate)}
    finally:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)


def _audio_input(job_input):
    cleanup = []
    if job_input.get("audio_base64"):
        path = _write_base64(job_input["audio_base64"], job_input.get("filename", "audio.wav"))
        cleanup.append(path)
    else:
        audio = job_input.get("audio")
        if not audio:
            raise ValueError("Missing 'audio' (local path or URL) or 'audio_base64'")
        if str(audio).startswith(("http://", "https://")):
            path = _download(audio)
            cleanup.append(path)
        else:
            path = audio
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Audio file not found: {path}")
    return _load_waveform(path), cleanup


def _speaker_kwargs(job_input):
    kwargs = {}
    for key in ("num_speakers", "min_speakers", "max_speakers"):
        if job_input.get(key) is not None:
            kwargs[key] = int(job_input[key])
    return kwargs


def handler(job):
    job_input = job.get("input") or {}
    cleanup = []
    try:
        audio, cleanup = _audio_input(job_input)
        output = pipeline(audio, **_speaker_kwargs(job_input))
        segments = [
            {
                "speaker": str(speaker),
                "start": round(float(turn.start), 3),
                "end": round(float(turn.end), 3),
            }
            for turn, speaker in output.speaker_diarization
        ]
        return {
            "segments": segments,
            "speakers": sorted({segment["speaker"] for segment in segments}),
        }
    except (ValueError, FileNotFoundError) as exc:
        return {"error": str(exc)}
    finally:
        for path in cleanup:
            if os.path.exists(path):
                os.remove(path)


runpod.serverless.start({"handler": handler})
