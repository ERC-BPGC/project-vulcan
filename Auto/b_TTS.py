import multiprocessing.shared_memory
import pyttsx3
import sys
import multiprocessing
import struct

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed percent (can go over 100)
    engine.setProperty('volume', 0.2)  # Volume 0-1
    engine.say(text)
    engine.runAndWait()

def main_loop(shm2):
    existing_shm = multiprocessing.shared_memory.SharedMemory(name=shm2)
    text = ""
    last_text = ""
    while 1:
        while text == last_text:
            data_length = struct.unpack('I', existing_shm.buf[:4])[0]
            text = bytes(existing_shm.buf[4:4+data_length]).decode('utf-8')

        last_text = text
        text_to_speech(text)

if __name__ == "__main__":
    shm2 = sys.argv[1]
    main_loop(shm2) 
