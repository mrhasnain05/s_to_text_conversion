# Speech-to-Text Conversion Using Deep Learning

Converts speech to text using pretrained deep-learning speech models.

## Models
| Backend  | Default model                  | Architecture                         | Notes |
|----------|--------------------------------|--------------------------------------|-------|
| whisper  | `openai/whisper-base`          | Transformer encoder-decoder (seq2seq) | 99 languages, punctuation, timestamps |
| wav2vec2 | `facebook/wav2vec2-base-960h`  | Self-supervised Transformer + CTC     | English, fast, no punctuation |

Larger models are more accurate but slower: `whisper-tiny/base/small/medium/large-v3`.

## How it works
1. **Audio loading** – resampled to 16 kHz mono, peak-normalised.
2. **Feature extraction** – Whisper: log-Mel spectrogram; Wav2Vec2: raw waveform -> CNN encoder.
3. **Neural network** – Transformer layers turn audio features into text tokens (Whisper decodes
   autoregressively; Wav2Vec2 predicts characters with CTC).
4. **Long audio** – split into 30-second chunks and stitched back together.
5. **Evaluation** – Word Error Rate (WER) against a reference transcript.

## Setup
```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
# ffmpeg is needed for mp3/m4a:  sudo apt install ffmpeg   |   brew install ffmpeg
```

## Run
```bash
python speech_to_text.py sample.wav
python speech_to_text.py sample.mp3 --backend whisper --model openai/whisper-small --timestamps
python speech_to_text.py sample.wav --backend wav2vec2
python speech_to_text.py hindi.wav --language hindi
python speech_to_text.py --mic 10                       # microphone
python speech_to_text.py sample.wav --reference ref.txt # prints WER
python speech_to_text.py sample.wav --output out.txt
python app.py                                           # web interface
```
The first run downloads the model (~150 MB for whisper-base) and caches it.

## Project structure
```
speech_to_text/
├── speech_to_text.py   # core library + CLI
├── app.py              # Gradio web demo
├── requirements.txt
└── README.md
```

## Possible extensions
Fine-tune Wav2Vec2 on your own dataset, add real-time streaming, speaker diarization, or noise reduction.
