from multiprocessing import shared_memory


existing_shm = shared_memory.SharedMemory(name=r'psm_44334a87')

print(bytes(existing_shm.buf[:10]))
