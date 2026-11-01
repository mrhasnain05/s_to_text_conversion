# 🎙️ HCL Live Speech-to-Text Converter

A native, interactive web application built with **Streamlit** and driven by **AssemblyAI**'s deep learning speech processing models. This project was developed as part of the **HCL Training Program** to demonstrate practical integration of cloud-based AI engines and custom web interfaces.

Designed specifically to run flawlessly on modern computer hardware architectures—including **Apple Silicon (M1/M2/M3/M4) and Intel processors**—without causing heavy local package or compilation errors.

---

## ✨ Features

- **🔴 Live Microphone Recording:** Record speech directly through your web browser using integrated custom microphone UI elements.
- **📁 File Upload Support:** Upload existing recorded audio files (`.mp3` or `.wav`) for immediate parsing.
- **⚡ Cloud AI Transcription:** Utilizes AssemblyAI's pipeline architecture for high-accuracy speech processing.
- **🔒 Secure Architecture:** Protected configuration setup using environment variables to keep developer credentials completely hidden.

---

## 🛠️ Tech Stack & Dependencies

- **Frontend Interface:** Streamlit (Python Web App Framework)
- **Audio Processing Backend:** AssemblyAI API
- **Live Recording Library:** audio-recorder-streamlit
- **Environment Management:** python-dotenv

---

## 🚀 Step-by-Step Local Setup

Follow these commands in your terminal to set up and run this application locally:

### 1. Clone the Repository
```bash
git clone https://github.com
cd YOUR_REPOSITORY_NAME
```

### 2. Set Up a Clean Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install the Core Requirements
```bash
pip install streamlit assemblyai audio-recorder-streamlit python-dotenv
```

### 4. Configure Your Secure Environment Variables
Create a file named `.env` in the root folder of the project and insert your private token details:
```text
ASSEMBLYAI_API_KEY=your_actual_assemblyai_api_key_here
```

### 5. Launch the Web UI Application
```bash
python -m streamlit run app.py
```
*Your browser will automatically pop open a window at `http://localhost:8501` showcasing the running system!*

---

## 📂 Project Directory Structure

```text
├── app.py          # Core application web layout & processing pipelines
├── .env            # Private local credentials (IGNORED from Git)
├── .gitignore      # Tells git to prevent uploading caches/keys/venv
└── README.md       # Project overview documentation
```
