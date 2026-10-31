"""
Speech-to-Text Conversion Using Deep Learning
---------------------------------------------
Converts speech to text using pretrained deep-learning speech models:
  * Whisper   (Transformer encoder-decoder, OpenAI)      -> multilingual, robust
  * Wav2Vec2  (self-supervised Transformer + CTC, Meta)  -> English, lightweight

Usage examples
  python speech_to_text.py audio.wav
  python speech_to_text.py audio.mp3 --backend whisper --model openai/whisper-small --timestamps
  python speech_to_text.py audio.wav --backend wav2vec2
  python speech_to_text.py --mic 8                       # record 8 seconds from microphone
  python speech_to_text.py audio.wav --reference ref.txt # compute Word Error Rate
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000  # both models expect 16 kHz mono audio

DEFAULT_MODELS = {
    "whisper": "openai/whisper-base",
    "wav2vec2": "facebook/wav2vec2-base-960h",
}


# --------------------------------------------------------------------------- #
# Audio utilities
# --------------------------------------------------------------------------- #
def load_audio(path: str) -> np.ndarray:
    """Load any audio file as a mono float32 array at 16 kHz."""
    import librosa

    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


def record_microphone(seconds: float) -> np.ndarray:
    """Record from the default microphone."""
    import sounddevice as sd

    print(f"Recording for {seconds:g}s... speak now.")
    data = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                  channels=1, dtype="float32")
    sd.wait()
    print("Recording finished.")
    return data.squeeze()


def normalize(audio: np.ndarray) -> np.ndarray:
    """Peak-normalize so quiet recordings are still recognised."""
    peak = np.max(np.abs(audio)) if audio.size else 0.0
    return audio / peak * 0.95 if peak > 0 else audio


# --------------------------------------------------------------------------- #
# Transcriber
# --------------------------------------------------------------------------- #
class Transcriber:
    """Thin wrapper around a Hugging Face ASR pipeline."""

    def __init__(self, backend: str = "whisper", model_name: str | None = None,
                 device: str | None = None):
        import torch
        from transformers import pipeline

        if backend not in DEFAULT_MODELS:
            raise ValueError(f"backend must be one of {list(DEFAULT_MODELS)}")
        self.backend = backend
        self.model_name = model_name or DEFAULT_MODELS[backend]

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"Loading {self.model_name} on {self.device} ...")
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_name,
            chunk_length_s=30,      # long audio is split into 30 s windows
            device=self.device,
        )

    def transcribe(self, audio: np.ndarray, language: str | None = None,
                   timestamps: bool = False) -> dict:
        """Return {'text': str, 'chunks': list|None, 'seconds': float}."""
        kwargs = {}
        if self.backend == "whisper":
            kwargs["generate_kwargs"] = {"task": "transcribe"}
            if language:
                kwargs["generate_kwargs"]["language"] = language
            if timestamps:
                kwargs["return_timestamps"] = True
        elif timestamps:
            kwargs["return_timestamps"] = "word"

        start = time.time()
        out = self.pipe({"raw": normalize(audio), "sampling_rate": SAMPLE_RATE}, **kwargs)
        return {
            "text": out["text"].strip(),
            "chunks": out.get("chunks"),
            "seconds": time.time() - start,
        }


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER = (substitutions + deletions + insertions) / reference words."""
    import jiwer
    import re

    clean = lambda s: re.sub(r"[^\w\s']", "", s.lower()).strip()
    return jiwer.wer(clean(reference), clean(hypothesis))


def format_timestamps(chunks) -> str:
    lines = []
    for c in chunks or []:
        start, end = c["timestamp"]
        end = start if end is None else end
        lines.append(f"[{start:7.2f} -> {end:7.2f}] {c['text'].strip()}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Speech-to-Text with pretrained deep-learning models")
    p.add_argument("audio", nargs="*", help="audio file(s): wav, mp3, flac, ogg, m4a ...")
    p.add_argument("--backend", choices=list(DEFAULT_MODELS), default="whisper")
    p.add_argument("--model", help="any compatible Hugging Face model id "
                                   "(e.g. openai/whisper-small, openai/whisper-large-v3)")
    p.add_argument("--language", help="Whisper only, e.g. english, hindi, french (default: auto-detect)")
    p.add_argument("--timestamps", action="store_true", help="print segment timestamps")
    p.add_argument("--mic", type=float, metavar="SECONDS", help="record from microphone instead of a file")
    p.add_argument("--reference", help="text file with the true transcript; prints the Word Error Rate")
    p.add_argument("--output", help="save the transcript to this .txt file")
    p.add_argument("--device", help="cpu / cuda:0 (default: auto)")
    args = p.parse_args()

    if not args.audio and not args.mic:
        p.error("provide an audio file or use --mic SECONDS")

    asr = Transcriber(args.backend, args.model, args.device)

    inputs = []
    if args.mic:
        inputs.append(("microphone", record_microphone(args.mic)))
    for path in args.audio:
        if not Path(path).exists():
            sys.exit(f"File not found: {path}")
        inputs.append((path, load_audio(path)))

    results = []
    for name, audio in inputs:
        res = asr.transcribe(audio, args.language, args.timestamps)
        duration = len(audio) / SAMPLE_RATE
        print(f"\n=== {name}  ({duration:.1f}s audio, {res['seconds']:.1f}s to transcribe) ===")
        print(res["text"])
        if args.timestamps and res["chunks"]:
            print("\n" + format_timestamps(res["chunks"]))
        if args.reference:
            ref = Path(args.reference).read_text(encoding="utf-8")
            print(f"\nWord Error Rate: {word_error_rate(ref, res['text']) * 100:.2f}%")
        results.append(f"# {name}\n{res['text']}\n")

    if args.output:
        Path(args.output).write_text("\n".join(results), encoding="utf-8")
        print(f"\nSaved transcript to {args.output}")


if __name__ == "__main__":
    main()
