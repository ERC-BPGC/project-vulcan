import multiprocessing.shared_memory
import time
import sys
import multiprocessing
from ctypes import c_wchar_p  # c_wchar_p

def speech_to_text(shared_string):
    # Convert shared_string from string representation to shared memory object
    shared_string = multiprocessing.shared_memory.SharedMemory(shared_string)
    # Update shared string in a loop
    for i in range(5):
        shared_string.buf[:3] = b'one'
        time.sleep(1)
        #print(shared_string.value.decode('utf-8'))
    

if __name__ == "__main__":
    # Expecting shared_string as argument
    shared_string = sys.argv[1]
    # Call speech_to_text function with shared_string
    speech_to_text(shared_string)

