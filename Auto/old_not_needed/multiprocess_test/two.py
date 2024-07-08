import multiprocessing.shared_memory
import time
import sys
import multiprocessing

def gpt_process(shared_string):
    shared_string = multiprocessing.shared_memory.SharedMemory(shared_string)
    # Update shared string in a loop
    for i in range(5):
        print(bytes(shared_string.buf[:3].value))
        time.sleep(1)
        #print(shared_string.value.decode('utf-8'))

if __name__ == "__main__":
    # Expecting shared_string as argument
    shared_string = sys.argv[1]

    
    # Call gpt_process function with shared_string
    #gpt_process(shared_string)
