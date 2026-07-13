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
import Check_folder as check
from RAG_LLM.app_lang2 import response_rag
import serial  # For Arduino communication
import random
import threading
import time

# Arduino serial port setup (update the port and baud rate as per your setup)
arduino = serial.Serial(port='COM8', baudrate=115200, timeout=1)


# Set OpenAI API key
# openai.api_key = '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'
# Replace with the URL of your IP camera's stream
ip_camera_url = "http://192.168.154.230:81/stream"

# Initialize VideoCapture with IP camera URL
cap = cv2.VideoCapture(ip_camera_url)


# cap = cv2.VideoCapture(0)
conversation_started = False  # Track if conversation has started
terminate_event = Event()  # Event to signal threads to terminate
glob_expression = ""
glob_name = ""# Variable to store detected facial expression

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
# Text-to-speech function using Microsoft Ravi (male voice)
def text_to_speech(text):
    engine = pyttsx3.init()
    # Set the voice to Microsoft Ravi
    for voice in engine.getProperty('voices'):
        if voice.id == "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_RAVI_11.0":
            engine.setProperty('voice', voice.id)
            break
    engine.say(text)
    engine.runAndWait()


# Speech-to-text and GPT conversation function
def conversation_code():
    global conversation_started, glob_expression
    recognizer = sr.Recognizer()
    
    while not terminate_event.is_set():
        with sr.Microphone(device_index=18) as source:
            print("Please speak:")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio)
            print("You said:", text)

            if "hello" in text.lower() and not conversation_started:
                conversation_started = True
                send_command_to_arduino(8) #mouth cycle start
                text_to_speech("Hello! How can I assist you?")
                send_command_to_arduino(9) #mouth cycle stop
                continue

            if "bye" in text.lower():
                print("Conversation ended by user.")
                send_command_to_arduino(8) #mouth cycle start
                text_to_speech("Goodbye! See You Soon!!")
                send_command_to_arduino(9) #mouth cycle stop
                conversation_started=False
                
            if "shut down" in text.lower():
                send_command_to_arduino(8) #mouth cycle start
                text_to_speech("Abort sequence initiated. Shutting down!!")
                send_command_to_arduino(9) #mouth cycle stop
                cap.release()
                cv2.destroyAllWindows()
                shm.unlink()
                break    
                

            if conversation_started:
                try:
                #     # Build the GPT prompt by including detected emotion
                #     gpt_prompt = f"You are talking to {glob_name}"
                #     gpt_prompt += f"User is feeling {glob_expression}. Respond to the user's query accordingly."
                #     gpt_prompt += f" The user says: {text}"

                #     # Send the prompt to GPT
                #     response = openai.ChatCompletion.create(
                #         model="gpt-3.5-turbo",
                #         messages=[{"role": "user", "content": gpt_prompt}],
                #         max_tokens=50
                #     )
                #     output_text = response.choices[0].message['content'].strip()
                #     print("GPT output:", output_text)
                #     text_to_speech(output_text)
                # except openai.error.OpenAIError as e:
                #     print(f"OpenAI API error: {e}")
                # except Exception as e:
                #     print(f"Unexpected error: {e}")
                    
                    response = response_rag(question=text, user_expression=glob_expression)
                    send_command_to_arduino(8) #mouth cycle stop 
                    text_to_speech(response)
                    send_command_to_arduino(9)
                    # output_text = response.choices[0].message['content'].strip()
                # except openai.error.OpenAIError as e:
                #   print(f"API error: {e}")
                except Exception as e:
                     print(f"Unexpected error: {e}")
                     
            else:
                send_command_to_arduino(8) #mouth cycle start
                text_to_speech("Greet me with HELLO VULCAN to start chatting!!")   
                send_command_to_arduino(9) #mouth cycle stop      
            
                         
             
   
        except sr.UnknownValueError:
            send_command_to_arduino(8) #mouth cycle start
            text_to_speech("Sorry, I could not understand what you said.")
            send_command_to_arduino(9) #mouth cycle stop
         
        except sr.RequestError as e:
            send_command_to_arduino(8) #mouth cycle start
            text_to_speech(f"Could not request results from the speech recognition service: {e}")
            send_command_to_arduino(9) #mouth cycle stop 

# Vision function for facial expression detection
# def vision():
#     global cap, conversation_started, glob_expression
#     last_expression = ""
#     last_name=""
#     last_update_time = time.time()

#     while cap.isOpened() and not terminate_event.is_set():
#         _, frame = cap.read()
#         bbox, gray = face_de.get_face_harr(frame=frame)

#         # Check if 1 second has passed since the last expression update
#         if bbox and conversation_started:
#             current_time = time.time()
#             for b in bbox:
#                 # Crop the face and detect expression
#                 face = gray[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
#                 cv2.imshow('frame',frame)
#                 # # name_result= check.check_folder(frame)
#                 # last_name=name_result
#                 # glob_name=last_name
#                 expression_result = get_expression(face)

#                 # Update expression if it's different from the last one or 1 second has passed
#                 if expression_result != last_expression or (current_time - last_update_time) >= 1:
#                     last_expression = expression_result
#                     glob_expression = last_expression  # Update the global emotion
#                     last_update_time = current_time

#                 # Draw bounding box and overlay expression text
#                 cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
#                 cv2.putText(
#                     frame, glob_expression, (int(b[0]), int(b[1]) - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
#                 )
#         # cv2.flip(frame,1)
#         # cv2.putText(frame, name, bbox[0:1])
#         # print(name)
#         # cv2.imshow('Video', frame)
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             terminate_event.set()  # Signal both threads to stop
#             shm.unlink()
#             break

#     cap.release()
#     cv2.destroyAllWindows()
    
    
  # Replace 'COM3' with your Arduino's port

# import serial  # For Arduino communication

# # Arduino serial port setup (update the port and baud rate as per your setup)
# arduino = serial.Serial(port='COM3', baudrate=9600, timeout=1)  # Replace 'COM3' with your Arduino's port

# import serial  # For Arduino communication

# # Arduino serial port setup (update the port and baud rate as per your setup)
# arduino = serial.Serial(port='COM3', baudrate=9600, timeout=1)  # Replace 'COM3' with your Arduino's port

def send_command_to_arduino(command):
    """Send a command to Arduino."""
    try:
        arduino.write(f"{command}\n".encode('utf-8'))
        print(f"Command sent to Arduino: {command}")
    except Exception as e:
        print(f"Failed to send command to Arduino: {e}")
        
   
    # Schedule the next command to be sent after 10 seconds   

def vision():
    global cap, conversation_started, glob_expression

    # Get frame dimensions for center calculation
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2

    # Tolerance for ignoring small deviations (10% of frame size)
    tolerance_x = frame_width * 0.08
    tolerance_y = frame_height * 0.08

    last_expression = ""  # Store last detected expression for updates
    last_update_time = time.time()
    last_command_time = time.time()
    last_command_time1 = time.time()


    while cap.isOpened() and not terminate_event.is_set():
        _, frame = cap.read()
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        bbox, gray = face_de.get_face_harr(frame=frame)
        
        current_time = time.time()
        current_time1 = time.time()
        
        # Check if 10 seconds have passed to send command 10
        if current_time - last_command_time >= 50:
            send_command_to_arduino(6)
            print("sent 6")# Send command 10
            last_command_time = current_time 
        if current_time1 - last_command_time1 >= 30:
            send_command_to_arduino(5)
            print("sent 5")# Send command 10
            last_command_time1 = current_time1     # Update the last command time

        if bbox and conversation_started:
                b=bbox[0]
                # Calculate the center of the face bounding box
                face_center_x = int((b[0] + b[2]) / 2)
                face_center_y = int((b[1] + b[3]) / 2)

                # Movement commands based on face alignment
                if abs(face_center_x - frame_center_x) > tolerance_x:
                    if face_center_x < frame_center_x:
                        send_command_to_arduino(2)  # Eye left
                    else:
                        send_command_to_arduino(1)  # Eye right

                if abs(face_center_y - frame_center_y) > tolerance_y:
                    if face_center_y < frame_center_y:
                        send_command_to_arduino(3)  # Eye up
                    else:
                        send_command_to_arduino(4)  # Eye down

                # Detect facial expression
                face = gray[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
                current_time = time.time()
                expression_result = get_expression(face)

                # Update expression if it's different or 1 second has passed
                if expression_result != last_expression or (current_time - last_update_time) >= 1:
                    last_expression = expression_result
                    glob_expression = last_expression  # Update the global expression
                    # print(f"Detected Expression: {glob_expression}")
                    last_update_time = current_time

                # Draw bounding box and markers
                cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
                cv2.circle(frame, (face_center_x, face_center_y), 5, (255, 0, 0), -1)  # Face center
                cv2.circle(frame, (frame_center_x, frame_center_y), 5, (0, 0, 255), -1)  # Frame center
                cv2.putText(frame, glob_expression, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Video', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            terminate_event.set()  # Signal both threads to stop
            shm.unlink()
            break

    cap.release()
    cv2.destroyAllWindows()
    #arduino.close()

    

# Main execution
if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(conversation_code)
        executor.submit(vision)
