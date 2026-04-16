import cv2
import threading
import pyttsx3
import speech_recognition as sr

def video_call_bot_interaction(prompt_callback):
    recognizer = sr.Recognizer()
    speaker = pyttsx3.init()
    cap = cv2.VideoCapture(0)

    print("📹 Starting Video Call Mode... Press 'q' to exit.")

    def listen_and_respond():
        while True:
            with sr.Microphone() as source:
                print("\n🎤 Speak your question:")
                audio = recognizer.listen(source)
                try:
                    user_input = recognizer.recognize_google(audio)
                    print(f"You (video): {user_input}")
                    if user_input.lower() in ["exit", "quit"]:
                        break
                    response = prompt_callback(user_input)
                    print(f"Bot (video): {response}")
                    speaker.say(response)
                    speaker.runAndWait()
                except Exception as e:
                    print(f"❗ Error: {str(e)}")

    # Run audio in separate thread for non-blocking video
    audio_thread = threading.Thread(target=listen_and_respond, daemon=True)
    audio_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Video Call (Press q to exit)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("📹 Exiting Video Call Mode...")
            break

    cap.release()
    cv2.destroyAllWindows()
