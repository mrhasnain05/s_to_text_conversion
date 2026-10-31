"""Web demo:  python app.py   ->  open http://127.0.0.1:7860"""
import gradio as gr
from speech_to_text import Transcriber, load_audio, format_timestamps, DEFAULT_MODELS

_cache = {}

def get_model(backend, model_name):
    key = (backend, model_name)
    if key not in _cache:
        _cache[key] = Transcriber(backend, model_name or None)
    return _cache[key]

def run(audio_path, backend, model_name, language, timestamps):
    if audio_path is None:
        return "Please upload or record some audio.", ""
    asr = get_model(backend, model_name.strip())
    res = asr.transcribe(load_audio(audio_path), language.strip() or None, timestamps)
    return res["text"], format_timestamps(res["chunks"]) if timestamps else ""

demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Audio(sources=["upload", "microphone"], type="filepath", label="Speech"),
        gr.Radio(list(DEFAULT_MODELS), value="whisper", label="Backend"),
        gr.Textbox(value="", label="Model id (blank = default)", placeholder="openai/whisper-small"),
        gr.Textbox(value="", label="Language (Whisper only, blank = auto)", placeholder="english"),
        gr.Checkbox(label="Show timestamps (Whisper)"),
    ],
    outputs=[gr.Textbox(label="Transcript", lines=8), gr.Textbox(label="Timestamps", lines=6)],
    title="Speech-to-Text Using Deep Learning",
    description="Pretrained Whisper / Wav2Vec2 models convert speech to text.",
)

if __name__ == "__main__":
    demo.launch()
