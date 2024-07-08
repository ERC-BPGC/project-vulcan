import multiprocessing.shared_memory
import cv2
from PIL import Image
import concurrent.futures
import subprocess
import multiprocessing


import t_Hand_waving_Detection as hand_de
import t_gaze as gaze_es

import m_face as face_de
import m_expression as exp_re

cap = cv2.VideoCapture(0)
flag_trig = 0
first = 1

bg_speech_to_text = ''
bg_gpt = ''
shm = multiprocessing.shared_memory.SharedMemory(create=True, size=250)


if (cap.isOpened()== False):
    print("Error opening video stream or file")
    #exit()

def get_gaze_estimate(face):
    return gaze_es.get_gaze_estimate(face=face)

def detect_hand(frame):
    return hand_de.detect_hand(frame=frame)

def get_expression(gray):
    return exp_re.get_expression(gray)

def start_bg_stt(shm):
    global bg_one
    bg_one = subprocess.Popen(['python3', 'b_speech_to_text.py', shm])
    return bg_one

def start_bg_gpt(shm):
    global bg_one
    bg_two = subprocess.Popen(['python3', 'b_gpt.py', shm])
    return bg_two

def check_triggers(frame, face):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        trig1 = executor.submit(detect_hand, frame).result()                #tirgger 1: hand waving
        trig2 = executor.submit(get_gaze_estimate, face).result()           #trigger 2: gaze estimate
                
    if trig1:
        return 1
    elif trig2 >= 0.85:                                                     #threshold of gaze
        return 1

def main_loop():
    global cap,flag_trig, first, bg_speech_to_text, bg_gpt,shm

    while(cap.isOpened()):
        _, frame = cap.read()
        bbox, gray = face_de.get_face_harr(frame=frame)                      #do face detection
        if bbox:
            for b in bbox:
                frame_arr = Image.fromarray(frame)
                face = frame_arr.crop((b))
                if(flag_trig or first):
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future_expression = executor.submit(get_expression, gray[int(b[1]):int(b[3]), int(b[0]):int(b[2])])
                        expression_result = future_expression.result()

                    print(expression_result)
                else:
                    flag_trig = check_triggers(frame, face)

                first = 0
        cv2.imshow('',frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            bg_speech_to_text.terminate()
            bg_gpt.terminate()
            shm.unlink()
            break

    cap.release()
    hand_de.close_hands()
    cv2.destroyAllWindows()


with concurrent.futures.ThreadPoolExecutor() as executor:
    bg_temp1 = executor.submit(start_bg_stt, shm.name)                                                 # Start speech to text in background
    bg_speech_to_text = bg_temp1.result()       
    bg_temp2 = executor.submit(start_bg_gpt, shm.name)                                                 # Start GPT in background
    bg_gpt = bg_temp2.result()       
    
    executor.submit(main_loop)                                                                          #run main loop
