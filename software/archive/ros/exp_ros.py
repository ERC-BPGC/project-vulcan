import cv2
import numpy as np
from keras.models import load_model
import rospy
from std_msgs.msg import String

model_path = "fer_model.hdf5"
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
model = load_model(model_path, compile=False)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
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

def facial_expression_recognition():
    rospy.init_node('facial_expression_recognition_node', anonymous=True)
    emotion_publisher = rospy.Publisher('emotion_label', String, queue_size=10)
    cap = cv2.VideoCapture(0)

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            break

        faces, gray = detect_face(frame)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            emotion_label = predict_expression(roi_gray)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

           
            emotion_publisher.publish(emotion_label)

        cv2.imshow('Facial Expression Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        facial_expression_recognition()
    except rospy.ROSInterruptException:
        pass
