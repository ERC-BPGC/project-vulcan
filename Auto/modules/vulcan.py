import cv2
from PIL import Image

import m_gaze as gaze_es
import m_face as face_re
import m_expression as exp_re


cap = cv2.VideoCapture(0)

if (cap.isOpened()== False):
    print("Error opening video stream or file")
    exit()

while(cap.isOpened()):
    ret, frame = cap.read()
    bbox, gray = face_re.get_face_harr(frame=frame)
    if bbox:
        for b in bbox:
            frame_arr = Image.fromarray(frame)
            face = frame_arr.crop((b))
            print(gaze_es.get_gaze_estimate(face=face))
            print(exp_re.get_expression(gray[int(b[1]):int(b[3]), int(b[0]):int(b[2])]))

    cv2.imshow('',frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()