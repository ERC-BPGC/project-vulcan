import cv2
import numpy as np
from keras.models import load_model
import serial  

arduino = serial.Serial(port='COM5', baudrate=115200, timeout=.1)  

model_path = "fer_model.hdf5"
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
model = load_model(model_path, compile=False)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    return faces, gray


def predict_expression(roi_gray):
    roi_gray = cv2.resize(roi_gray, (64, 64))
    roi_gray = roi_gray.astype('float')
    roi_gray /= 255.0

    roi_gray = np.expand_dims(roi_gray, axis=0)
    roi_gray = np.expand_dims(roi_gray, axis=-1)

    prediction = model.predict(roi_gray)
    max_index = np.argmax(prediction)
    emotion_label = emotion_labels[max_index]
    return emotion_label


WIDTH = 1280
HEIGHT = 720

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces, gray = detect_face(frame)

   
    (x, y, w, h) = faces[0]
    roi_gray = gray[y:y + h, x:x + w]
    emotion_label = predict_expression(roi_gray)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, emotion_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if (x + w / 2) < (WIDTH / 4):
       
        arduino.write(str(0).encode())

    elif (x + w / 2) > (3 * WIDTH / 4):
       
        arduino.write(str(2).encode())

    elif (y + h / 2) < (HEIGHT / 4):
       
        arduino.write(str(1).encode())

    elif (y + h / 2) > (3 * HEIGHT / 4):
        
        arduino.write(str(3).encode())

    else:
        
        arduino.write(str(5).encode())

    cv2.imshow('Facial Expression Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
