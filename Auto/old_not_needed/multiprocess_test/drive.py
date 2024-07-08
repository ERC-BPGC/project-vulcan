import multiprocessing
import concurrent.futures
import multiprocessing.shared_memory
import subprocess
import time
import sys
from ctypes import c_wchar_p  # c_wchar_p


def speech_to_text(shared_string):
    # Convert shared_string from string representation to shared memory object
    shared_string = multiprocessing.shared_memory.SharedMemory(shared_string)
    # Update shared string in a loop
    for i in range(5):
        shared_string.buf[:3] = b'one'
        time.sleep(1)
        #print(shared_string.value.decode('utf-8'))

def start_bg_stt(shared_string):
    #speech_to_text(shared_string)
    bg_one = subprocess.Popen(['python3', 'one.py', str(shared_string)])

def start_bg_gpt(shared_string):
    bg_two = subprocess.Popen(['python3', 'two.py', str(shared_string)])

if __name__ == "__main__":
    # Expecting shared_string as argument
    shared_string = multiprocessing.shared_memory.SharedMemory(name = 'Parth',create=True, size=50)
    print("driver ", str(shared_string.name))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        
        bg_temp1 = executor.submit(start_bg_stt, shared_string.name)  # Start speech to text in background
        #bg_temp2 = executor.submit(start_bg_gpt, shared_string.name)  # Start GPT in background

        # Wait for both subprocesses to finish
        #bg_temp1.result()
        #bg_temp2.result()
        print()
