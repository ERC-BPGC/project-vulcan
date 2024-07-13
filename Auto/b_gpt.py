import multiprocessing.shared_memory
import openai
import multiprocessing
import sys
import struct
import time
SYSTEM_MESSAGE = """
Provide short, concise answers to the user's questions.
Your name is Vulcan.
You are created by ERC at BITS Pilani college.
The full form of ERC is Electronics and Robotics Club.
The full form of BITS is BIrla Institute of Technology.
You are designed to reply to queries and assist with various tasks.
You are supposed to answer in short to most queries asked. Not more than 3-5 lines in general.
"""

chat_history = []
#openai.api_key = ""


def test(shm,shm2):
    existing_shm = multiprocessing.shared_memory.SharedMemory(name=shm)
    existing_shm2 = multiprocessing.shared_memory.SharedMemory(name=shm2)
    
    prompt = ""
    last_prompt = ""
    while 1:
        while prompt == last_prompt:
            data_length = struct.unpack('I', existing_shm.buf[:4])[0]
            prompt = bytes(existing_shm.buf[4:4+data_length]).decode('utf-8')
            time.sleep(0.001)

        last_prompt = prompt
        if prompt:
            to_say = str(ask_gpt(prompt))
            existing_shm2.buf[:4] = struct.pack('I', len(to_say))
            existing_shm2.buf[4:4+len(to_say)] = to_say.encode()

def ask_gpt(prompt: str):
    global SYSTEM_MESSAGE, chat_history 
    
    user_prompt = {"role": "user", "content": prompt}
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            *chat_history,
            user_prompt,
        ],
    )

    content = response.choices[0].message.content
    chat_history.append(user_prompt)
    chat_history.append({"role": "assistant", "content": content})

    #print("\033[92m" + content + "\033[0m")
    return content


if __name__ == "__main__":
    shm = sys.argv[1]
    shm2 = sys.argv[2]
    test(shm, shm2)
