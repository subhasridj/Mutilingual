import os
import sys
import pyttsx3
import speech_recognition as sr
from pdf_processor import extract_pdf_to_chunks
from qa_system import QASystem

# -----------------------------
# Text-to-Speech Engine
# -----------------------------
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# -----------------------------
# Speech-to-Text Engine
# -----------------------------
def recognize_voice():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("🎤 Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print(f"🗣 Recognized: {text}")
        return text
    except sr.UnknownValueError:
        print("⚠️ Sorry, I could not understand the audio.")
        speak("Sorry, I could not understand.")
        return ""
    except sr.RequestError:
        print("⚠️ API unavailable.")
        speak("Voice service unavailable.")
        return ""

# -----------------------------
# Voice Command / Question Handler
# -----------------------------
def handle_voice_input(recognized_text, qa_system):
    recognized_text = recognized_text.strip().lower()

    # Exit Command
    if "exit" in recognized_text or "quit" in recognized_text:
        print("🚪 Exiting...")
        speak("Exiting the assistant. Goodbye.")
        return "command_exit", qa_system

    # Upload Command
    if "upload" in recognized_text or "new pdf" in recognized_text:
        print("📂 Voice command detected: Upload new PDF")
        speak("Please type the path of the PDF to upload.")
        pdf_path = input("Enter PDF path: ").strip()

        if not os.path.exists(pdf_path):
            print("❌ File not found.")
            speak("File not found. Please try again.")
            return "command_invalid", qa_system

        chunks = extract_pdf_to_chunks(pdf_path)
        qa_system = QASystem()
        qa_system.build_faiss_index(chunks)
        print("✅ PDF loaded successfully.")
        speak("PDF loaded successfully.")
        return "command_upload", qa_system

    # Otherwise, treat as a question
    if qa_system is None:
        print("⚠️ No PDF loaded yet.")
        speak("Please upload a PDF first.")
        return "command_invalid", qa_system

    print("❓ Question detected:", recognized_text)
    answer = qa_system.answer_query(recognized_text)
    print("💡 Answer:", answer)
    speak(answer)
    return "command_question", qa_system

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    qa_system = None

    print("=== AI Assistant with Voice & PDF Support ===")
    speak("Welcome to your AI Assistant. Please upload a PDF to begin.")

    while True:
        print("\n🎤 Say something (upload pdf / ask question / exit)...")
        voice_text = recognize_voice()

        if not voice_text:
            continue

        command_type, qa_system = handle_voice_input(voice_text, qa_system)

        if command_type == "command_exit":
            break
