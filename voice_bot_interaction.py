import threading
import speech_recognition as sr
import pyttsx3

def voice_bot_interaction(prompt_callback):
    recognizer = sr.Recognizer()
    speaker = pyttsx3.init()

    print("🎤 Voice Bot Mode. Say 'exit' to stop.")

    def listen_loop():
        while True:
            with sr.Microphone() as source:
                try:
                    audio = recognizer.listen(source)
                    user_input = recognizer.recognize_google(audio)
                    print(f"You (voice): {user_input}")

                    if user_input.lower() in ["exit", "quit"]:
                        print("🔊 Exiting Voice Bot...")
                        break

                    response = prompt_callback(user_input)
                    print(f"Bot (voice): {response}")
                    speaker.say(response)
                    speaker.runAndWait()

                except sr.UnknownValueError:
                    print("❗ Could not understand audio.")
                except Exception as e:
                    print(f"❗ Error: {str(e)}")

    # Run voice loop in a separate thread for async integration
    voice_thread = threading.Thread(target=listen_loop, daemon=True)
    voice_thread.start()
    voice_thread.join()
