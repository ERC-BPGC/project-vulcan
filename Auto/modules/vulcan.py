import m_gaze as gaze_es
import m_face as face_re
import cv2
from PIL import Image

cap = cv2.VideoCapture(0)

if (cap.isOpened()== False):
    print("Error opening video stream or file")
    exit()

while(cap.isOpened()):
    ret, frame = cap.read()
    bbox = face_re.get_face_coord(frame=frame)
    if bbox:
        for b in bbox:
            frame_arr = Image.fromarray(frame)
            face = frame_arr.crop((b))
            print(gaze_es.get_gaze_estimate(face=face))


    cv2.imshow('',frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()