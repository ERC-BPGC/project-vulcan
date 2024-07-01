import cv2
import numpy as np
from tensorflow.keras.models import load_model


model_path = "models/fer_model.hdf5"
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
model = load_model(model_path, compile=False)

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


def get_expression(roi_gray):
    return  predict_expression(roi_gray)