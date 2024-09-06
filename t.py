import cv2
import pytesseract  # Assuming audio is a custom module for audio output
import time
import threading
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
import time

def say(words):
    tts = gTTS(text=words, lang='en-au', slow=False)
    tts.save("output.mp3")
    playsound("output.mp3")


def listen():
    r = sr.Recognizer()
    mic_list = sr.Microphone.list_microphone_names()
    print(mic_list)
    mic_index = 0
    mic = sr.Microphone(device_index=mic_index)
    with mic as source:
        r.adjust_for_ambient_noise(source)
        say('LISTENING')
        print('listening....')
        time.sleep(1)
        audio = r.listen(source, timeout=7.0)

    text = r.recognize_google(audio)
    return text

def read_text(image):
    """Recognize text from an image using Tesseract OCR."""
    string = pytesseract.image_to_string(image)
    return string

def read_label():
    """Capture real-time video from webcam and perform text recognition."""
    start_time = time.time()
    cap = cv2.VideoCapture(1)

    while time.time() - start_time < 20:
        ret, frame = cap.read()
        
        if not ret:
            break

        cv2.imshow('Real-time', frame)

        try:    
            text = read_text(frame)
            
            if text:
                say(text)  # Assuming audio.say() reads out text
                break
        
        except Exception as e:
            print("Error:", e)
            # Handle specific exceptions if necessary

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def start_realtime_text_recognition():
    """Start real-time text recognition in a separate thread."""
    thread = threading.Thread(target=read_label)
    thread.start()

# Call the function to start real-time text recognition
start_realtime_text_recognition()
