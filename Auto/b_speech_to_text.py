import speech_recognition as sr

from datetime import datetime, timedelta
from queue import Queue
import torch

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

with open("rec_text.txt", 'w') as f:
    f.write("")

with source:
    recorder.adjust_for_ambient_noise(source)

def record_callback(_, audio:sr.AudioData) -> None:
    """
    Threaded callback function to recieve audio data when recordings finish.
    audio: An AudioData containing the recorded bytes.
    """
    # Grab the raw bytes and push it into the thread safe queue.
    data = audio.get_raw_data()
    data_queue.put(data)

# Create a background thread that will pass us raw audio bytes.
# We could do this manually but SpeechRecognizer provides a nice helper.
recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

a = 0
def speech_to_text():
    global noise_tag, last_sample, last_text, counter
    while True:
        try:
            now = datetime.utcnow()
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
                        #print(counter)
                    if counter == 2:
                        noise_tag = True

                    last_text = text
                except:
                    text = ""

                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text

                #os.system('cls' if os.name=='nt' else 'clear')
                for line in transcription:
                    print(line)

                # Flush stdout.
                #print('', end='', flush=True)

            else:
                if noise_tag:
                    last_sample = bytes()
                    phrase_complete = True
                    with open("rec_text.txt", 'a') as f:
                        f.write(text+"\n")

                    noise_tag = False
                    counter = 0
                    
                else:
                    try :
                        if (not phrase_complete) and now - phrase_time > timedelta(seconds=phrase_timeout):
                            last_sample = bytes()
                            phrase_complete = True
                            with open("rec_text.txt", 'a') as f:
                                f.write(text+"\n")
                    except:
                        pass


        except KeyboardInterrupt:
            exit()

while 1:
    speech_to_text()