"""whisper.cpp-backed ASR (asr.engine = whisper).

Real-time path: POST 16 kHz mono audio to a local whisper-server (the [models.files.asr_whisper] model, kept
loaded by the whisper service on ports.whisper). Falls back to spawning whisper-cli if the server is down.
On-demand path: WhisperASR.reprocess(wav, model="large-v3") runs whisper-cli with a bigger model, downloaded
next to the others on first use. `mise run whisper` builds both binaries into data/whisper.cpp/.
"""

import io
import json
import os
import pathlib
import subprocess
import urllib.request
from collections.abc import Mapping

import numpy as np

from liberational import config as cfg

WHISPER_CLI = cfg.ROOT / "data/whisper.cpp/build/bin/whisper-cli"  # where `mise run whisper` builds it
SERVER_URL = os.environ.get("WHISPER_SERVER", cfg.local_url("whisper", "/inference"))
GGML_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{}.bin"
SR = 16000


def _wav_bytes(audio16k: np.ndarray) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio16k.astype(np.float32), SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _multipart(fields: Mapping[str, object], file_bytes: bytes) -> tuple[bytes, str]:
    b = b"----packwhisper"
    out = []
    for k, v in fields.items():
        out.append(
            b"--"
            + b
            + b'\r\nContent-Disposition: form-data; name="'
            + k.encode()
            + b'"\r\n\r\n'
            + str(v).encode()
            + b"\r\n"
        )
    out.append(
        b"--"
        + b
        + b'\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\nContent-Type: audio/wav\r\n\r\n'
        + file_bytes
        + b"\r\n"
    )
    out.append(b"--" + b + b"--\r\n")
    return b"".join(out), "multipart/form-data; boundary=" + b.decode()


class WhisperASR:
    def __init__(self, model_path: str | os.PathLike[str] | None = None) -> None:
        self.model_path: pathlib.Path = pathlib.Path(model_path) if model_path else cfg.model_path("asr_whisper")
        self.server_ok: bool | None = None

    def __call__(self, audio16k: np.ndarray) -> str:
        if len(audio16k) < SR // 10:
            return ""
        wav = _wav_bytes(audio16k)
        try:
            body, ctype = _multipart({"response_format": "json", "temperature": 0.0, "language": "en"}, wav)
            req = urllib.request.Request(SERVER_URL, data=body, headers={"Content-Type": ctype})
            with urllib.request.urlopen(req, timeout=60) as r:
                self.server_ok = True
                return json.loads(r.read().decode())["text"].strip()
        except Exception:
            self.server_ok = False
            return self._cli(wav, self.model_path)

    @staticmethod
    def _cli(wav: bytes, model_path: pathlib.Path) -> str:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav)
            p = f.name
        try:
            out = subprocess.run(
                [str(WHISPER_CLI), "-m", str(model_path), "-f", p, "-nt", "-l", "en"],
                capture_output=True,
                text=True,
                timeout=600,
            )
            return " ".join(l.strip() for l in out.stdout.splitlines() if l.strip())
        finally:
            os.unlink(p)

    @classmethod
    def reprocess(cls, wav_path: str, model: str = "large-v3") -> str:
        """On-demand higher-accuracy pass with a bigger model (downloads it on first use)."""
        mp = cfg.model_path("asr_whisper").parent / f"ggml-{model}.bin"
        if not mp.exists():
            tmp = mp.with_name(mp.name + ".part")
            urllib.request.urlretrieve(GGML_URL.format(model), tmp)
            tmp.replace(mp)
        return cls._cli(pathlib.Path(wav_path).read_bytes(), mp)


if __name__ == "__main__":
    import argparse

    import soundfile as sf

    ap = argparse.ArgumentParser(description="Transcribe a wav; --large reprocesses with large-v3")
    ap.add_argument("wav")
    ap.add_argument("--large", action="store_true")
    a = ap.parse_args()
    if a.large:
        print(WhisperASR.reprocess(a.wav))
    else:
        audio, sr = sf.read(a.wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            from scipy.signal import resample_poly

            audio = resample_poly(audio, SR, sr)
        asr = WhisperASR()
        print(asr(audio))
        print("via", "server" if asr.server_ok else "cli", file=__import__("sys").stderr)
