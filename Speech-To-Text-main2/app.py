import streamlit as st
import assemblyai as aai
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("aai_api_key")
if not api_key:
    try:
        api_key = st.secrets.get("aai_api_key")
    except Exception:
        api_key = None

aai.settings.api_key = api_key
transcriber = aai.Transcriber()

st.set_page_config(page_title="HCL Live Speech-to-Text", page_icon="🎙️", layout="centered")
st.title("🎙️ HCL Live Speech-to-Text Converter")
st.write("Record your voice live or upload a file to convert speech to text instantly.")

tab1, tab2 = st.tabs(["🔴 Live Voice Recorder", "📁 Upload Audio File"])

with tab1:
    st.subheader("Speak into your Microphone")
    st.write("Click the mic icon to START recording. Click again to STOP.")
    
    audio_bytes = audio_recorder(
        text="Click to record",
        recording_color="#e85a4f",
        neutral_color="#6aa84f",
        icon_size="2x"
    )
    
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        
        temp_live_file = "live_recording.wav"
        with open(temp_live_file, "wb") as f:
            f.write(audio_bytes)
            
        if st.button("Convert Live Voice to Text ✨"):
            with st.spinner("Transcribing your voice..."):
                try:
                    transcriber = aai.Transcriber()
                    transcript = transcriber.transcribe(temp_live_file)
                    
                    st.subheader("📝 Transcribed Text:")
                    st.success(transcript.text)
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    if os.path.exists(temp_live_file):
                        os.remove(temp_live_file)

with tab2:
    st.subheader("Upload an existing Audio File")
    uploaded_file = st.file_uploader("Choose a file", type=["wav", "mp3"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        
        temp_upload_file = "uploaded_" + uploaded_file.name
        with open(temp_upload_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        if st.button("Convert Uploaded File to Text ✨"):
            with st.spinner("Transcribing file..."):
                try:
                    transcriber = aai.Transcriber()
                    transcript = transcriber.transcribe(temp_upload_file)
                    
                    st.subheader("📝 Transcribed Text:")
                    st.success(transcript.text)
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    if os.path.exists(temp_upload_file):
                        os.remove(temp_upload_file)
