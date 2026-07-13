import dlib
import cv2
from PIL import Image
import numpy as np

har_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')


CNN_FACE_MODEL = 'models/mmod_human_face_detector.dat' # from http://dlib.net/files/mmod_human_face_detector.dat.bz2
cnn_face_detector = dlib.cnn_face_detection_model_v1(CNN_FACE_MODEL)


def get_face_rgb(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bbox = []
    dets = cnn_face_detector(frame_rgb, 1)
    for d in dets:
        l = d.rect.left()
        r = d.rect.right()
        t = d.rect.top()
        b = d.rect.bottom()
        # expand a bit
        l -= int((r-l)*0.2)
        r += int((r-l)*0.2)
        t -= int((b-t)*0.2)
        b += int((b-t)*0.2)
        bbox.append([l,t,r,b])
            
    if bbox:
        b = bbox[0]
        frame_pil = Image.fromarray(frame_rgb)
        cropped_frame = frame_pil.crop((b[0], b[1], b[2], b[3]))
        return cv2.cvtColor(np.array(cropped_frame), cv2.COLOR_RGB2BGR)
    else:
        return frame  # Return the original frame if no face is detected



def get_face_coord(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bbox = []
    dets = cnn_face_detector(frame_rgb, 1)
    for d in dets:
        l = d.rect.left()
        r = d.rect.right()
        t = d.rect.top()
        b = d.rect.bottom()
        # expand a bit
        l -= int((r-l)*0.2)
        r += int((r-l)*0.2)
        t -= int((b-t)*0.2)
        b += int((b-t)*0.2)
        bbox.append([l,t,r,b])
            
    return bbox
   
    
def get_face_harr(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = har_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    bbox = []
    for (x, y, w, h) in faces:
        bbox.append([x,y,w+x,h+y])
    return bbox, gray