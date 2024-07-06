import cv2
from PIL import Image
import concurrent.futures
import subprocess

import t_Hand_waving_Detection as hand_de
import t_gaze as gaze_es

import m_face as face_de
import m_expression as exp_re

cap = cv2.VideoCapture(0)
flag_trig = 0
first = 1


if (cap.isOpened()== False):
    print("Error opening video stream or file")
    exit()

def get_gaze_estimate(face):
    return gaze_es.get_gaze_estimate(face=face)

def detect_hand(frame):
    return hand_de.detect_hand(frame=frame)

def get_expression(gray):
    return exp_re.get_expression(gray)


def check_triggers(frame, face):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        trig1 = executor.submit(detect_hand, frame).result()                #tirgger 1: hand waving
        trig2 = executor.submit(get_gaze_estimate, face).result()           #trigger 2: gaze estimate
                
    if trig1:
        return 1
    elif trig2 >= 0.85:                                                     #threshold of gaze
        return 1

def main_loop():
    global cap,flag_trig, first

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
            break

    cap.release()
    hand_de.close_hands()
    cv2.destroyAllWindows()



with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.submit(subprocess.Popen(['python3','b_speech_to_text.py']))                                     #run speech to text in background
    executor.submit(main_loop)                                                                               #run main loop
