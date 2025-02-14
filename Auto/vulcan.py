import multiprocessing.shared_memory
import cv2
from PIL import Image
import concurrent.futures
import struct
import time
import subprocess
import multiprocessing
import m_face as face_de
import m_expression as exp_re
import speech_recognition as sr
import pyttsx3
import openai
from threading import Event

# Set OpenAI API key
openai.api_key = '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'

cap = cv2.VideoCapture(0)
conversation_started = False  # Track if conversation has started
terminate_event = Event()  # Event to signal threads to terminate
glob_expression = ""  # Variable to store detected facial expression

# Shared memory setup
shm = multiprocessing.shared_memory.SharedMemory(create=True, size=250)

if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

def detect_phrase_in_shared_memory(target_phrase="hello"):
    existing_shm = multiprocessing.shared_memory.SharedMemory(name=shm.name)
    try:
        while not terminate_event.is_set():
            length_bytes = existing_shm.buf[:4]
            text_length = struct.unpack('I', length_bytes)[0]

            if text_length > 0:
                text_bytes = existing_shm.buf[4:4 + text_length]
                text = text_bytes.decode('utf-8')
                if target_phrase in text.lower():
                    print(f"Detected phrase: '{target_phrase}'")
                    existing_shm.buf[:4] = struct.pack('I', 0)  # Reset shared memory
                    return True
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping phrase detection.")
    finally:
        existing_shm.close()

def get_expression(gray):
    return exp_re.get_expression(gray)

# Text-to-speech function using Microsoft Zira (female voice)
def text_to_speech(text):
    engine = pyttsx3.init()
    # Set the voice to Microsoft Zira
    for voice in engine.getProperty('voices'):
        if voice.id == "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_ZIRA_11.0":
            engine.setProperty('voice', voice.id)
            break
    engine.say(text)
    engine.runAndWait()

# Speech-to-text and GPT conversation function
def conversation_code():
    global conversation_started, glob_expression
    recognizer = sr.Recognizer()
    
    while not terminate_event.is_set():
        with sr.Microphone() as source:
            print("Please speak:")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio)
            print("You said:", text)

            if "hello" in text.lower() and not conversation_started:
                conversation_started = True
                text_to_speech("Hello! How can I assist you?")
                continue

            if "bye" in text.lower():
                print("Conversation ended by user.")
                text_to_speech("Goodbye! See You Soon!!")
                cap.release()
                cv2.destroyAllWindows()
                shm.unlink()
                break

            if conversation_started:
                try:
                    # Build the GPT prompt by including detected emotion
                    gpt_prompt = f"User is feeling {glob_expression}. Respond to the user's query accordingly."
                    gpt_prompt += f" The user says: {text}"

                    # Send the prompt to GPT
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": gpt_prompt}],
                        max_tokens=50
                    )
                    output_text = response.choices[0].message['content'].strip()
                    print("GPT output:", output_text)
                    text_to_speech(output_text)
                except openai.error.OpenAIError as e:
                    print(f"OpenAI API error: {e}")
                except Exception as e:
                    print(f"Unexpected error: {e}")
   
        except sr.UnknownValueError:
            text_to_speech("Sorry, I could not understand what you said.")
        except sr.RequestError as e:
            text_to_speech(f"Could not request results from the speech recognition service: {e}")

# Vision function for facial expression detection
def vision():
    global cap, conversation_started, glob_expression
    last_expression = ""
    last_update_time = time.time()

    while cap.isOpened() and not terminate_event.is_set():
        _, frame = cap.read()
        bbox, gray = face_de.get_face_harr(frame=frame)

        # Check if 1 second has passed since the last expression update
        if bbox and conversation_started:
            current_time = time.time()
            for b in bbox:
                # Crop the face and detect expression
                face = gray[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
                expression_result = get_expression(face)

                # Update expression if it's different from the last one or 1 second has passed
                if expression_result != last_expression or (current_time - last_update_time) >= 1:
                    last_expression = expression_result
                    glob_expression = last_expression  # Update the global emotion
                    last_update_time = current_time

                # Draw bounding box and overlay expression text
                cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
                cv2.putText(
                    frame, glob_expression, (int(b[0]), int(b[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

        cv2.imshow('Video', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            terminate_event.set()  # Signal both threads to stop
            shm.unlink()
            break

    cap.release()
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(conversation_code)
        executor.submit(vision)
