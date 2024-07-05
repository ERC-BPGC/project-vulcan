import cv2
from PIL import Image
import concurrent.futures

import m_gaze as gaze_es
import m_face as face_re
import m_expression as exp_re


cap = cv2.VideoCapture(0)

if (cap.isOpened()== False):
    print("Error opening video stream or file")
    exit()

def get_gaze_estimate(face):
    return gaze_es.get_gaze_estimate(face=face)

def get_expression(gray):
    return exp_re.get_expression(gray)

while(cap.isOpened()):
    ret, frame = cap.read()
    bbox, gray = face_re.get_face_harr(frame=frame)
    if bbox:
        for b in bbox:
            frame_arr = Image.fromarray(frame)
            face = frame_arr.crop((b))
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_gaze = executor.submit(get_gaze_estimate, face)
                future_expression = executor.submit(get_expression, gray[int(b[1]):int(b[3]), int(b[0]):int(b[2])])
                gaze_result = future_gaze.result()
                expression_result = future_expression.result()

            print(gaze_result)
            print(expression_result)

    cv2.imshow('',frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()