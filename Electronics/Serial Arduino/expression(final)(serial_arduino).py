import cv2
import numpy as np
from keras.models import load_model
import serial  # pip install pyserial
import time

try:
    arduino = serial.Serial(port='COM37', baudrate=115200, timeout=5)  # replace COM4 by appropriate arduino port
except:
    print("arduino not connected issue")

model_path = "Code/fer_model.hdf5"
# model_path = "../fer_model.hdf5"
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


WIDTH = 1280/2
HEIGHT = 720/2

#make it square for easy things
# WIDTH = 512*2
# HEIGHT = 512*2

HEIGHT = 1280
WIDTH = 720

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

x= WIDTH/2
y = HEIGHT /2

# tmp=x
# x=y
# y=tmp

w =10
h=10

def crop_center_frame(frame):
    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2
    frame_size = min(width, height)
    x1 = center_x - frame_size // 2
    y1 = center_y - frame_size // 2
    x2 = center_x + frame_size // 2
    y2 = center_y + frame_size // 2
    cropped_frame = frame[y1:y2, x1:x2]
    return cropped_frame


start_time = time.time()  


while True:
    x, y, w, h = 0, 0, 0, 0  # Default values
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    
    # frame = frame[y:y + WIDTH, x:x + HEIGHT]

    faces, gray = detect_face(frame)

    # for (x, y, w, h) in faces:
    #     roi_gray = gray[y:y + h, x:x + w]
    #     emotion_label = predict_expression(roi_gray)
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     cv2.putText(frame, emotion_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    #print(faces)
    # success_face = 1
    # try:
    #     (x, y, w, h) = faces[0]
    #     # print(faces)

    # except:
    #     success_face = 0
    #     print("no face detected")


    success_face = 0  # Initialize to indicate no detected faces
    max_area = 0

    for (x, y, w, h) in faces:
        face_area = w * h
        if face_area > max_area:
            max_area = face_area
            selected_face = (x, y, w, h)
            success_face = 1

    if success_face:
        (x, y, w, h) = selected_face


    
    roi_gray = gray[y:y + h, x:x + w]
    try:
        emotion_label = predict_expression(roi_gray)
    except Exception:
        emotion_label = ""
    
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, emotion_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    try:
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Wait for the remaining time to achieve the desired frame rate
        if elapsed_time < 1:
            success_face = 0

        if(success_face==1):
            '''
            if (x + w / 2) < (WIDTH / 3):
                # move left
                print("move left")
                # arduino.write(str(7).encode())
                arduino.write(b'7')

            elif (x + w / 2) > (2 * WIDTH / 3):
                # move right            
                print("move right")

                # arduino.write(str(2).encode())
                arduino.write(b'2')

            elif (y + h / 2) < (HEIGHT / 3):
                # move up
                print("move up")

                # arduino.write(str(1).encode())
                arduino.write(b'1')

            elif (y + h / 2) > (2 * HEIGHT / 3):
                # move down
                print("move down")

                # arduino.write(str(3).encode())
                arduino.write(b'3')

            else:
                # move center
                print("center")
                # arduino.write(str(5).encode())
                arduino.write(b'5')
            '''
            if (x + w / 2) < (WIDTH / 2):
                # move left
                print("move left")
                # arduino.write(str(7).encode())
                arduino.write(b'7')

            elif (x + w / 2) > (WIDTH / 2):
                # move right            
                print("move right")

                # arduino.write(str(2).encode())
                arduino.write(b'2')

            if (y + h / 2) < (HEIGHT / 2):
                # move up
                print("move up")

                # arduino.write(str(1).encode())
                arduino.write(b'1')

            elif (y + h / 2) > (HEIGHT / 2):
                # move down
                print("move down")

                # arduino.write(str(3).encode())
                arduino.write(b'3')

            else:
                # move center
                print("center")
                # arduino.write(str(5).encode())

                arduino.write(b'5')
            start_time = time.time()  

    except:
        # print("arduino skill issue")
        print()

    time.sleep(0.1)

    cv2.imshow('Facial Expression Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
