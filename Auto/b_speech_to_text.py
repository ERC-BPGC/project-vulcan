import multiprocessing.shared_memory
import speech_recognition as sr
from datetime import datetime, timedelta
from queue import Queue
import torch
import multiprocessing
import sys
import struct

torch.cuda.empty_cache()

 # The last time a recording was retreived from the queue.
phrase_time = None
# Current raw audio bytes.
last_sample = bytes()
# Thread safe Queue for passing data from the threaded recording callback.
data_queue = Queue()
# We use SpeechRecognizer to record our audio because it has a nice feauture where it can detect when speech ends.
recorder = sr.Recognizer()
# recorder.energy_threshold = 200
# Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
recorder.dynamic_energy_threshold = True

source = sr.Microphone(sample_rate=16000)

record_timeout = 0.75
phrase_timeout = 3

#temp_file = NamedTemporaryFile().name
transcription = ['']

phrase_complete = True
noise_tag = False
last_text = None
counter = 0


with source:
    recorder.adjust_for_ambient_noise(source)

def record_callback(_, audio:sr.AudioData) -> None:
    data = audio.get_raw_data()
    data_queue.put(data)

recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

def speech_to_text(shm):
    global noise_tag, last_sample, last_text, counter
    existing_shm = multiprocessing.shared_memory.SharedMemory(name=shm)
        
    while True:
        try:
            now = datetime.now()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty() and not noise_tag:
                phrase_complete = False
                phrase_time = now       

                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                try:
                    result = recorder.recognize_google(audio_data)
                    text = result.strip()

                    if text == last_text:
                        counter += 1
                    if counter == 2:
                        noise_tag = True

                    last_text = text
                except:
                    text = ""

                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text

                #for line in transcription:
                #    print(line)
                #    sys.stdout.flush()
                
            else:
                if noise_tag:
                    last_sample = bytes()
                    phrase_complete = True
                    
                    existing_shm.buf[:4] = struct.pack('I', len(text))
                    existing_shm.buf[4:4+len(text)] = text.encode()

                    noise_tag = False
                    counter = 0
                    
                else:
                    try :
                        if (not phrase_complete) and now - phrase_time > timedelta(seconds=phrase_timeout):
                            last_sample = bytes()
                            phrase_complete = True
                        existing_shm.buf[:4] = struct.pack('I', len(text))
                        existing_shm.buf[4:4+len(text)] = text.encode()

                    except:
                        pass


        except KeyboardInterrupt:
            exit()

if __name__ == "__main__":
    shm = sys.argv[1]
    speech_to_text(shm)
