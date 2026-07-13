from multiprocessing import shared_memory
import numpy as np
import time
a = np.array([1, 1, 2, 3, 5, 8])

shm = shared_memory.SharedMemory(create=True, size=100)
shm.buf[:5] = b'howdy'

print(shm.name)
time.sleep(100)