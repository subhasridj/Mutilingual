import streamlit as st
import pyttsx3
import speech_recognition as sr
import cv2
import tempfile
import numpy as np
from pdf_processor import extract_pdf_to_chunks
from qa_system import QASystem

# -----------------------------
# TTS
# -----------------------------
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# -----------------------------
# Voice Input
# -----------------------------
def recognize_voice():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        st.write("🎤 Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except:
        return ""

# -----------------------------
# Video Input (demo: capture one frame)
# -----------------------------
def capture_video_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        st.image(frame, channels="BGR")
        return "Video captured! (demo only, not used for Q&A yet)"
    else:
        return "Failed to capture video."

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="AI Assistant", layout="wide")
st.title("🤖 AI Assistant with PDF + Text/Voice/Video")

# Session state
if "qa_system" not in st.session_state:
    st.session_state.qa_system = None

# Upload PDF
uploaded_pdf = st.file_uploader("📂 Upload a PDF", type="pdf")

if uploaded_pdf is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        tmpfile.write(uploaded_pdf.read())
        pdf_path = tmpfile.name

    chunks = extract_pdf_to_chunks(pdf_path)
    qa = QASystem()
    qa.build_faiss_index(chunks)
    st.session_state.qa_system = qa
    st.success("✅ PDF processed and indexed!")

# Tabs
tab1, tab2, tab3 = st.tabs(["💬 Text Mode", "🎤 Voice Mode", "📹 Video Mode"])

with tab1:
    st.subheader("Text Mode")
    if st.session_state.qa_system:
        query = st.text_input("Ask a question:")
        if st.button("Get Answer (Text)"):
            answer = st.session_state.qa_system.answer_query(query)
            st.write("💡 Answer:", answer)
            speak(answer)
    else:
        st.info("Upload a PDF first!")

with tab2:
    st.subheader("Voice Mode")
    if st.session_state.qa_system:
        if st.button("🎤 Speak Question"):
            text = recognize_voice()
            if text:
                st.write("🗣 You said:", text)
                answer = st.session_state.qa_system.answer_query(text)
                st.write("💡 Answer:", answer)
                speak(answer)
    else:
        st.info("Upload a PDF first!")

with tab3:
    st.subheader("Video Mode")
    if st.session_state.qa_system:
        if st.button("📹 Start Video"):
            msg = capture_video_frame()
            st.write(msg)
            # You could extend this with lip-reading / video Q&A
    else:
        st.info("Upload a PDF first!")

# Exit button
if st.button("🚪 Exit Assistant"):
    st.write("👋 Exiting...")
    st.stop()
